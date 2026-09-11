#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include <cstdint>
#include <optional>
#include <string>
#include <utility>

#include "gsddmm/gsddmm_launch.cuh"

// =============================================================================
// Host-side entry point for the GSDDMM forward kernel: argument parsing,
// validation and output allocation only.
//
// This translation unit instantiates no kernels -- it routes to one of the
// per-op launchers in gsddmm_launch_<op>.cu, each of which compiles its own
// slice of the (op x members x dtype x index x D x warps x stages) grid. See
// gsddmm_launch.cuh for the reasoning; deliberately do not include
// gsddmm_dispatch.cuh here.
// =============================================================================

namespace gsddmm {

namespace {

GSDDMM_OP parse_op(const std::string& op) {
    if (op == "add") return GSDDMM_OP::Add;
    if (op == "sub") return GSDDMM_OP::Sub;
    if (op == "mul") return GSDDMM_OP::Mul;
    if (op == "div") return GSDDMM_OP::Div;
    if (op == "dot") return GSDDMM_OP::Dot;
    if (op == "copy") return GSDDMM_OP::Copy;
    TORCH_CHECK(false, "GSDDMM: unsupported op '", op, "'; supported: add, sub, mul, div, dot, copy");
}

GSDDMM_MEMBER parse_member(const std::string& target, const char *which) {
    if (target == "src" || target == "u") return GSDDMM_MEMBER::Src_V;
    if (target == "dst" || target == "v") return GSDDMM_MEMBER::Dst_V;
    if (target == "edge" || target == "e") return GSDDMM_MEMBER::Edge;
    TORCH_CHECK(false, "GSDDMM: unsupported ", which, " target '", target, "'; supported: src, dst, edge");
}

}  // namespace

// =============================================================================
// Host launcher (defined in gsddmm_binding.cu)
// =============================================================================
//
// op:          "add" | "sub" | "mul" | "div" | "dot" | "copy"
// lhs_target / rhs_target: "src" | "dst" | "edge"
// L, R:        CUDA fp32/fp16/bf16, [N, D] for src/dst targets, [E, D] for edge
//              targets; D in {32, 64, 128, 256}, feature dim contiguous.
//              For op == "copy" R is never read (pass Edge) and L can not be Edge, it must be either Src_V or Dst_V.
// pipeline_stages: cp.async prefetch depth for the per-edge rows, in {0, 1, 2, 3}
//              (0 disables the pipeline).
// heavy_block_parts / heavy_edges_per_block: heavy-node chunking. When given,
//              heavy_nodes lists the node of every heavy BLOCK (a node repeated
//              once per chunk) and heavy_block_parts[b] the chunk index of block
//              b, which then owns edges [row_ptr[n] + part * K, +K) of node n,
//              K = heavy_edges_per_block. Absent: one block per heavy node.
// overlap_buckets: run the light bucket on a side stream concurrently with the
//              heavy bucket on the current stream (joined before returning control
//              of the current stream's order to the caller).
// Returns:     [E, D] for elementwise ops, [E] for "dot" (input dtype).
torch::Tensor gsddmm_forward_cuda(
    torch::Tensor L,
    torch::Tensor R,
    torch::Tensor row_ptr,
    torch::Tensor col_idx,
    std::string op,
    std::string lhs_target,
    std::string rhs_target,
    torch::Tensor light_nodes,
    torch::Tensor heavy_nodes,
    uint32_t light_warps_per_block,
    uint32_t heavy_warps_per_block,
    uint32_t pipeline_stages,
    std::optional<torch::Tensor>
        heavy_block_parts,
    uint32_t heavy_edges_per_block,
    bool overlap_buckets
) {
    const GSDDMM_OP op_enum        = parse_op(op);
    const GSDDMM_MEMBER lhs_member = parse_member(lhs_target, "lhs");
    const GSDDMM_MEMBER rhs_member = parse_member(rhs_target, "rhs");
    const bool is_dot              = op_enum == GSDDMM_OP::Dot;

    at::cuda::CUDAGuard device_guard(L.device());
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream(L.device().index());

    TORCH_CHECK(L.is_cuda() && R.is_cuda(), "L and R must be CUDA");
    TORCH_CHECK(row_ptr.is_cuda() && col_idx.is_cuda(), "CSR indices must be CUDA");
    TORCH_CHECK(light_nodes.is_cuda() && heavy_nodes.is_cuda(), "node buckets must be CUDA");
    TORCH_CHECK(L.dim() == 2 && R.dim() == 2, "L and R must be [*, D]");
    TORCH_CHECK(
        L.dtype() == torch::kFloat32 || L.dtype() == torch::kFloat16 || L.dtype() == torch::kBFloat16, "L must be float32, float16, or bfloat16"
    );
    TORCH_CHECK(R.dtype() == L.dtype(), "L and R must have the same dtype");
    TORCH_CHECK(L.stride(1) == 1 && R.stride(1) == 1, "Feature dim (D) must be contiguous for L and R");

    const auto idx_dtype = row_ptr.scalar_type();
    TORCH_CHECK(is_supported_index_type(idx_dtype), "row_ptr must be int32, int64, uint32, or uint64");
    // Only the index types listed in the dispatch (gsddmm_dispatch.cuh) are
    // instantiated; extend MakeIndexVariant there (at the cost of compile time)
    // to support more.
    TORCH_CHECK(
        idx_dtype == at::kInt || idx_dtype == at::kLong, "GSDDMM forward: row_ptr must be int32 or int64 (the instantiated index types)"
    );
    TORCH_CHECK(col_idx.scalar_type() == idx_dtype, "col_idx must have same dtype as row_ptr");
    TORCH_CHECK(
        light_nodes.scalar_type() == idx_dtype && heavy_nodes.scalar_type() == idx_dtype, "node buckets must have same dtype as row_ptr"
    );

    const int64_t N = row_ptr.size(0) - 1;
    const int64_t E = col_idx.size(0);
    const int64_t D = L.size(1);
    TORCH_CHECK(R.size(1) == D, "L and R must have the same feature dim D");

    auto check_rows = [N, E](const torch::Tensor& t, GSDDMM_MEMBER member, const char *name) {
        if (member == GSDDMM_MEMBER::Edge) {
            TORCH_CHECK(t.size(0) == E, name, " is edge-indexed and must have E=", E, " rows, got ", t.size(0));
        } else {
            TORCH_CHECK(t.size(0) == N, name, " is node-indexed and must have N=", N, " rows, got ", t.size(0));
        }
    };
    check_rows(L, lhs_member, "L");
    check_rows(R, rhs_member, "R");

    torch::Tensor O = is_dot ? torch::empty({E}, L.options()) : torch::empty({E, D}, L.options());

    TORCH_CHECK(D == 32 || D == 64 || D == 128 || D == 256, "GSDDMM forward: unsupported feature dim D=", D, "; supported: 32, 64, 128, 256");

    const bool chunked = heavy_block_parts.has_value() && heavy_block_parts->numel() > 0;
    if (chunked) {
        TORCH_CHECK(heavy_edges_per_block > 0, "GSDDMM forward: heavy_edges_per_block must be > 0 when heavy_block_parts is given");
        TORCH_CHECK(
            heavy_block_parts->is_cuda() && heavy_block_parts->scalar_type() == idx_dtype,
            "heavy_block_parts must be CUDA with the CSR index dtype"
        );
        TORCH_CHECK(
            heavy_block_parts->numel() == heavy_nodes.numel(), "GSDDMM forward: heavy_block_parts (", heavy_block_parts->numel(),
            ") and heavy_nodes (", heavy_nodes.numel(), ") must have one entry per heavy block"
        );
    }

    GsddmmLaunchArgs args{
        .L                     = L,
        .R                     = R,
        .O                     = O,
        .row_ptr               = row_ptr,
        .col_idx               = col_idx,
        .light_nodes           = light_nodes,
        .heavy_nodes           = heavy_nodes,
        .stream                = stream,
        .N                     = static_cast<uint64_t>(N),
        .D                     = static_cast<uint64_t>(D),
        .key                   = LRO{lhs_member, rhs_member, op_enum},
        .light_warps_per_block = static_cast<uint16_t>(light_warps_per_block),
        .heavy_warps_per_block = static_cast<uint16_t>(heavy_warps_per_block),
        .pipeline_stages       = static_cast<uint8_t>(pipeline_stages),
        .heavy_block_parts     = chunked ? &*heavy_block_parts : nullptr,
        .heavy_edges_per_block = chunked ? heavy_edges_per_block : 0u,
        .overlap_buckets       = overlap_buckets,
    };

    // One call per op, each resolved in its own translation unit.
    switch (op_enum) {
        case GSDDMM_OP::Add:
            gsddmm_forward_launch_add(args);
            break;
        case GSDDMM_OP::Sub:
            gsddmm_forward_launch_sub(args);
            break;
        case GSDDMM_OP::Mul:
            gsddmm_forward_launch_mul(args);
            break;
        case GSDDMM_OP::Div:
            gsddmm_forward_launch_div(args);
            break;
        case GSDDMM_OP::Dot:
            gsddmm_forward_launch_dot(args);
            break;
        case GSDDMM_OP::Copy:
            gsddmm_forward_launch_copy(args);
            break;
        default:
            TORCH_CHECK(false, "GSDDMM forward: op '", op, "' has no launcher");
    }

    CUDA_KERNEL_CHECK();

    return O;
}

// op / targets / L / R / D: as for gsddmm_forward_cuda. edge_list: [E, 2] uint64
//              (src, dst) node-id pairs, contiguous, reinterpreted as ulonglong2.
// pipeline_stages: cp.async prefetch depth of the per-edge operand rows, in
//              {0, 1, 2, 3} (0 = direct loads). Only meaningful with edges_per_warp > 1.
// edges_per_warp: contiguous edges each warp walks, in [1, kGsddmmEdgeMaxEdgesPerWarp].
// warps_per_block: independent warps packed per thread block, in [1, kGsddmmEdgeMaxWarpsPerBlock].
// canonical_edge_idx: optional [E] uint64 canonical edge id of every entry of
//              edge_list. Applied to the Edge operand rows and to the output row,
//              so a source-grouped edge_list (L2-friendly: consecutive warps share
//              the Src_V row) can still produce output numbered by forward-CSR
//              position. Omit when edge_list is already in the canonical order.
torch::Tensor gsddmm_forward_edge_blocks(
    torch::Tensor L,
    torch::Tensor R,
    torch::Tensor edge_list,
    std::string op,
    std::string lhs_target,
    std::string rhs_target,
    uint64_t N,
    uint32_t pipeline_stages                        = 0,
    uint32_t edges_per_warp                         = 4,
    uint32_t warps_per_block                        = 4,
    std::optional<torch::Tensor> canonical_edge_idx = std::nullopt
) {
    const GSDDMM_OP op_enum        = parse_op(op);
    const GSDDMM_MEMBER lhs_member = parse_member(lhs_target, "lhs");
    const GSDDMM_MEMBER rhs_member = parse_member(rhs_target, "rhs");
    const bool is_dot              = op_enum == GSDDMM_OP::Dot;

    at::cuda::CUDAGuard device_guard(L.device());
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream(L.device().index());

    TORCH_CHECK(L.is_cuda() && R.is_cuda(), "L and R must be CUDA");
    TORCH_CHECK(edge_list.is_cuda() && edge_list.stride(1) == 1, "Edge list should lay on CUDA device and be contigous by the second dim.");
    TORCH_CHECK(edge_list.dim() == 2 && edge_list.size(1) == 2, "Edge list must be a list of pairs of node indices.")
    TORCH_CHECK(L.dim() == 2 && R.dim() == 2, "L and R must be [*, D]");
    TORCH_CHECK(
        L.dtype() == torch::kFloat32 || L.dtype() == torch::kFloat16 || L.dtype() == torch::kBFloat16, "L must be float32, float16, or bfloat16"
    );
    TORCH_CHECK(R.dtype() == L.dtype(), "L and R must have the same dtype");
    TORCH_CHECK(L.stride(1) == 1 && R.stride(1) == 1, "Feature dim (D) must be contiguous for L and R");

    const auto idx_dtype = edge_list.scalar_type();
    TORCH_CHECK(is_supported_index_type(idx_dtype), "row_ptr must be int32, int64, uint32, or uint64");
    // Only the index types listed in the dispatch (gsddmm_dispatch.cuh) are
    // instantiated; extend MakeIndexVariant there (at the cost of compile time)
    // to support more.
    TORCH_CHECK(
        idx_dtype == at::kInt || idx_dtype == at::kUInt32 || idx_dtype == at::kLong || idx_dtype == at::kUInt64,
        "GSDDMM forward: row_ptr must be int32 or int64 (the instantiated index types)"
    );

    const uint64_t E = edge_list.size(0);
    TORCH_CHECK(E <= (1ull << 63ull) - 1ull, "Too many edges. This kernel supports only up to 2^63 - 1 edges.");
    const uint64_t D = L.size(1);
    TORCH_CHECK(R.size(1) == D, "L and R must have the same feature dim D");

    auto check_rows = [N, E](const torch::Tensor& t, GSDDMM_MEMBER member, const char *name) {
        if (member == GSDDMM_MEMBER::Edge) {
            TORCH_CHECK(t.size(0) == E, name, " is edge-indexed and must have E=", E, " rows, got ", t.size(0));
        } else {
            TORCH_CHECK(t.size(0) == N, name, " is node-indexed and must have N=", N, " rows, got ", t.size(0));
        }
    };
    check_rows(L, lhs_member, "L");
    check_rows(R, rhs_member, "R");

    torch::Tensor O = is_dot ? torch::empty({static_cast<int64_t>(E)}, L.options())
                             : torch::empty({static_cast<int64_t>(E), static_cast<int64_t>(D)}, L.options());

    TORCH_CHECK(D == 32 || D == 64 || D == 128 || D == 256, "GSDDMM forward: unsupported feature dim D=", D, "; supported: 32, 64, 128, 256");

    TORCH_CHECK(pipeline_stages <= 3, "GSDDMM forward (edge blocks): pipeline_stages must be in [0, 3], got ", pipeline_stages);
    TORCH_CHECK(
        edges_per_warp >= 1 && edges_per_warp <= kGsddmmEdgeMaxEdgesPerWarp, "GSDDMM forward (edge blocks): edges_per_warp must be in [1, ",
        kGsddmmEdgeMaxEdgesPerWarp, "], got ", edges_per_warp
    );
    TORCH_CHECK(
        warps_per_block >= 1 && warps_per_block <= kGsddmmEdgeMaxWarpsPerBlock, "GSDDMM forward (edge blocks): warps_per_block must be in [1, ",
        kGsddmmEdgeMaxWarpsPerBlock, "], got ", warps_per_block
    );

    unsigned long long const *canonical_ptr = nullptr;
    if (canonical_edge_idx.has_value()) {
        const torch::Tensor& canonical = *canonical_edge_idx;
        TORCH_CHECK(canonical.is_cuda(), "GSDDMM forward (edge blocks): canonical_edge_idx must be CUDA");
        TORCH_CHECK(
            canonical.scalar_type() == at::kUInt64, "GSDDMM forward (edge blocks): canonical_edge_idx must be uint64, got ",
            canonical.scalar_type()
        );
        TORCH_CHECK(canonical.is_contiguous(), "GSDDMM forward (edge blocks): canonical_edge_idx must be contiguous");
        TORCH_CHECK(
            static_cast<uint64_t>(canonical.numel()) == E, "GSDDMM forward (edge blocks): canonical_edge_idx must have E=", E, " entries, got ",
            canonical.numel()
        );
        canonical_ptr = reinterpret_cast<unsigned long long const *>(canonical.data_ptr<uint64_t>());
    }

    GsddmmLaunchArgsEdge args{
        .L                  = L,
        .R                  = R,
        .O                  = O,
        .edge_nodes_idx     = reinterpret_cast<ulonglong2 const *>(edge_list.data_ptr<uint64_t>()),
        .canonical_edge_idx = canonical_ptr,
        .stream             = stream,
        .E                  = E,
        .D                  = D,
        .key                = LRO{lhs_member, rhs_member, op_enum},
        .pipeline_stages    = static_cast<uint8_t>(pipeline_stages),
        .edges_per_warp     = static_cast<uint8_t>(edges_per_warp),
        .warps_per_block    = static_cast<uint8_t>(warps_per_block),
    };

    // One call per op, each resolved in its own translation unit.
    switch (op_enum) {
        case GSDDMM_OP::Add:
            gsddmm_forward_edge_launch_add(args);
            break;
        case GSDDMM_OP::Sub:
            gsddmm_forward_edge_launch_sub(args);
            break;
        case GSDDMM_OP::Mul:
            gsddmm_forward_edge_launch_mul(args);
            break;
        case GSDDMM_OP::Div:
            gsddmm_forward_edge_launch_div(args);
            break;
        case GSDDMM_OP::Dot:
            gsddmm_forward_edge_launch_dot(args);
            break;
        case GSDDMM_OP::Copy:
            gsddmm_forward_edge_launch_copy(args);
            break;
        default:
            TORCH_CHECK(false, "GSDDMM forward edge: op '", op, "' has no launcher");
    }

    CUDA_KERNEL_CHECK();

    return O;
}

};  // namespace gsddmm

// Global-scope entry point matching the declaration in kernels.cuh, which the
// pybind module binds; the implementation lives in namespace gsddmm. Default
// arguments are attached to the kernels.cuh declaration, not repeated here.
torch::Tensor gsddmm_forward_cuda(
    torch::Tensor L,
    torch::Tensor R,
    torch::Tensor row_ptr,
    torch::Tensor col_idx,
    std::string op,
    std::string lhs_target,
    std::string rhs_target,
    torch::Tensor light_nodes,
    torch::Tensor heavy_nodes,
    uint32_t light_warps_per_block,
    uint32_t heavy_warps_per_block,
    uint32_t pipeline_stages,
    std::optional<torch::Tensor>
        heavy_block_parts,
    uint32_t heavy_edges_per_block,
    bool overlap_buckets
) {
    return gsddmm::gsddmm_forward_cuda(
        std::move(L), std::move(R), std::move(row_ptr), std::move(col_idx), std::move(op), std::move(lhs_target), std::move(rhs_target),
        std::move(light_nodes), std::move(heavy_nodes), light_warps_per_block, heavy_warps_per_block, pipeline_stages,
        std::move(heavy_block_parts), heavy_edges_per_block, overlap_buckets
    );
}

torch::Tensor gsddmm_forward_edge_blocks(
    torch::Tensor L,
    torch::Tensor R,
    torch::Tensor edge_list,
    std::string op,
    std::string lhs_target,
    std::string rhs_target,
    uint64_t N,
    uint32_t pipeline_stages                        = 0,
    uint32_t edges_per_warp                         = 4,
    uint32_t warps_per_block                        = 4,
    std::optional<torch::Tensor> canonical_edge_idx = std::nullopt
) {
    return gsddmm::gsddmm_forward_edge_blocks(
        std::move(L), std::move(R), std::move(edge_list), std::move(op), std::move(lhs_target), std::move(rhs_target), N, pipeline_stages,
        edges_per_warp, warps_per_block, std::move(canonical_edge_idx)
    );
}
