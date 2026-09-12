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
    uint32_t pipeline_stages,
    uint32_t edges_per_warp,
    uint32_t warps_per_block,
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

namespace {

// Which passes a member pair needs, and hence which CSRs/buckets must be valid.
struct BackwardPasses {
    bool dst;   // an operand reads the destination vertex
    bool src;   // an operand reads the source vertex
    bool edge;  // an operand is edge-indexed (its gradient is a plain store)
};

BackwardPasses backward_passes(GSDDMM_OP op, GSDDMM_MEMBER lhs, GSDDMM_MEMBER rhs) {
    const bool uses_rhs = op != GSDDMM_OP::Copy;
    auto reads          = [&](GSDDMM_MEMBER m) { return lhs == m || (uses_rhs && rhs == m); };
    return {reads(GSDDMM_MEMBER::Dst_V), reads(GSDDMM_MEMBER::Src_V), uses_rhs && reads(GSDDMM_MEMBER::Edge)};
}

// Shared validation for both backward entry points. Returns the feature dim.
int64_t check_backward_operands(
    const torch::Tensor& L, const torch::Tensor& R, const torch::Tensor& dO, GSDDMM_OP op_enum, GSDDMM_MEMBER lhs_member,
    GSDDMM_MEMBER rhs_member, int64_t N, int64_t E
) {
    TORCH_CHECK(L.is_cuda() && R.is_cuda() && dO.is_cuda(), "L, R and dO must be CUDA");
    TORCH_CHECK(L.dim() == 2 && R.dim() == 2, "L and R must be [*, D]");
    TORCH_CHECK(
        L.dtype() == torch::kFloat32 || L.dtype() == torch::kFloat16 || L.dtype() == torch::kBFloat16, "L must be float32, float16, or bfloat16"
    );
    TORCH_CHECK(R.dtype() == L.dtype() && dO.dtype() == L.dtype(), "L, R and dO must share a dtype");
    TORCH_CHECK(L.stride(1) == 1 && R.stride(1) == 1, "Feature dim (D) must be contiguous for L and R");

    const int64_t D = L.size(1);
    TORCH_CHECK(R.size(1) == D, "L and R must have the same feature dim D");
    TORCH_CHECK(D == 32 || D == 64 || D == 128 || D == 256, "GSDDMM backward: unsupported feature dim D=", D, "; supported: 32, 64, 128, 256");

    auto check_rows = [N, E](const torch::Tensor& t, GSDDMM_MEMBER member, const char *name) {
        if (member == GSDDMM_MEMBER::Edge) {
            TORCH_CHECK(t.size(0) == E, name, " is edge-indexed and must have E=", E, " rows, got ", t.size(0));
        } else {
            TORCH_CHECK(t.size(0) == N, name, " is node-indexed and must have N=", N, " rows, got ", t.size(0));
        }
    };
    check_rows(L, lhs_member, "L");
    check_rows(R, rhs_member, "R");

    // dot reduces the feature axis, so its output -- and its gradient -- is [E].
    if (op_enum == GSDDMM_OP::Dot) {
        TORCH_CHECK(dO.dim() == 1 && dO.size(0) == E, "GSDDMM backward: dO for 'dot' must be [E=", E, "], got ", dO.sizes());
    } else {
        TORCH_CHECK(
            dO.dim() == 2 && dO.size(0) == E && dO.size(1) == D, "GSDDMM backward: dO must be [E=", E, ", D=", D, "], got ", dO.sizes()
        );
    }
    TORCH_CHECK(dO.is_contiguous(), "GSDDMM backward: dO must be contiguous");
    return D;
}

}  // namespace

// Node-parallel backward. Reduces each node operand's gradient in fp32 registers
// with one block per node, so no atomics are used and no buffer needs zeroing.
//
// L, R, dO / op / targets: as for gsddmm_forward_cuda; dO is [E, D], or [E] for
//              "dot".
// row_ptr / col_idx + light_nodes / heavy_nodes: the forward CSR and its buckets,
//              used by the destination-side pass.
// row_ptr_T / col_idx_T + light_nodes_T / heavy_nodes_T: the backward
//              (source-grouped) CSR and its buckets, used by the source-side pass.
// canonical_edge_idx: [E] uint64 mapping a backward-CSR slot to its forward-CSR
//              edge id, so the source pass can read dO and the edge operand in
//              their own numbering. Required whenever a source-side pass runs and
//              the two CSRs are distinct buffers.
// Returns:     {dL, dR}; dR is an empty tensor for "copy", which never reads R.
std::vector<torch::Tensor> gsddmm_backward_cuda(
    torch::Tensor L,
    torch::Tensor R,
    torch::Tensor dO,
    torch::Tensor row_ptr,
    torch::Tensor col_idx,
    torch::Tensor row_ptr_T,
    torch::Tensor col_idx_T,
    std::string op,
    std::string lhs_target,
    std::string rhs_target,
    torch::Tensor light_nodes,
    torch::Tensor heavy_nodes,
    torch::Tensor light_nodes_T,
    torch::Tensor heavy_nodes_T,
    std::optional<torch::Tensor>
        canonical_edge_idx,
    uint32_t light_warps_per_block,
    uint32_t heavy_warps_per_block
) {
    const GSDDMM_OP op_enum        = parse_op(op);
    const GSDDMM_MEMBER lhs_member = parse_member(lhs_target, "lhs");
    const GSDDMM_MEMBER rhs_member = parse_member(rhs_target, "rhs");
    const BackwardPasses passes    = backward_passes(op_enum, lhs_member, rhs_member);

    at::cuda::CUDAGuard device_guard(L.device());
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream(L.device().index());

    TORCH_CHECK(row_ptr.is_cuda() && col_idx.is_cuda(), "CSR indices must be CUDA");
    const auto idx_dtype = row_ptr.scalar_type();
    TORCH_CHECK(
        idx_dtype == at::kInt || idx_dtype == at::kLong, "GSDDMM backward: row_ptr must be int32 or int64 (the instantiated index types)"
    );
    TORCH_CHECK(col_idx.scalar_type() == idx_dtype, "col_idx must have same dtype as row_ptr");

    const int64_t N = row_ptr.size(0) - 1;
    const int64_t E = col_idx.size(0);
    const int64_t D = check_backward_operands(L, R, dO, op_enum, lhs_member, rhs_member, N, E);

    if (passes.src) {
        TORCH_CHECK(row_ptr_T.is_cuda() && col_idx_T.is_cuda(), "GSDDMM backward: the backward CSR must be CUDA for a source-side gradient");
        TORCH_CHECK(row_ptr_T.size(0) - 1 == N && col_idx_T.size(0) == E, "GSDDMM backward: the two CSRs must describe the same graph");
        TORCH_CHECK(
            row_ptr_T.scalar_type() == idx_dtype && col_idx_T.scalar_type() == idx_dtype, "GSDDMM backward: both CSRs must share an index dtype"
        );
    }

    unsigned long long const *canonical_ptr = nullptr;
    if (canonical_edge_idx.has_value() && canonical_edge_idx->numel() > 0) {
        const torch::Tensor& canonical = *canonical_edge_idx;
        TORCH_CHECK(canonical.is_cuda() && canonical.is_contiguous(), "GSDDMM backward: canonical_edge_idx must be CUDA and contiguous");
        TORCH_CHECK(canonical.scalar_type() == at::kUInt64, "GSDDMM backward: canonical_edge_idx must be uint64");
        TORCH_CHECK(static_cast<int64_t>(canonical.numel()) == E, "GSDDMM backward: canonical_edge_idx must have E=", E, " entries");
        canonical_ptr = reinterpret_cast<unsigned long long const *>(canonical.data_ptr<uint64_t>());
    } else if (passes.src && E > 0) {
        // The source pass walks the backward CSR, whose slots are CSC positions,
        // so it cannot address dO without the permutation. This holds even for an
        // undirected graph that aliases its two CSRs: a symmetric sparsity
        // pattern does not make the CSC slot order equal the CSR edge order --
        // row u then lists u's incoming edges, not the outgoing ones this pass
        // sums over, and those are different rows of dO.
        TORCH_CHECK(false, "GSDDMM backward: canonical_edge_idx is required for a source-side gradient");
    }

    auto rows_of     = [&](GSDDMM_MEMBER member) { return member == GSDDMM_MEMBER::Edge ? E : N; };
    torch::Tensor dL = torch::empty({rows_of(lhs_member), D}, L.options());
    // Copy never reads R, so it has no gradient to return.
    torch::Tensor dR = (op_enum == GSDDMM_OP::Copy) ? torch::empty({0}, L.options()) : torch::empty({rows_of(rhs_member), D}, L.options());

    GsddmmBackwardLaunchArgs args{
        .L       = L,
        .R       = R,
        .dO      = dO,
        .dL      = dL,
        .dR      = dR,
        .row_ptr = row_ptr,
        .col_idx = col_idx,
        // Both directions are always passed: the dispatch drops the pass a member
        // pair does not need at compile time, so the unused side is never read
        // (and these are references, which must not bind to temporaries).
        .light_nodes           = light_nodes,
        .heavy_nodes           = heavy_nodes,
        .row_ptr_T             = row_ptr_T,
        .col_idx_T             = col_idx_T,
        .light_nodes_T         = light_nodes_T,
        .heavy_nodes_T         = heavy_nodes_T,
        .canonical_edge_idx    = canonical_ptr,
        .stream                = stream,
        .N                     = static_cast<uint64_t>(N),
        .D                     = static_cast<uint64_t>(D),
        .key                   = LRO{lhs_member, rhs_member, op_enum},
        .light_warps_per_block = static_cast<uint16_t>(light_warps_per_block),
        .heavy_warps_per_block = static_cast<uint16_t>(heavy_warps_per_block),
    };

    switch (op_enum) {
        case GSDDMM_OP::Add:
            gsddmm_backward_launch_add(args);
            break;
        case GSDDMM_OP::Sub:
            gsddmm_backward_launch_sub(args);
            break;
        case GSDDMM_OP::Mul:
            gsddmm_backward_launch_mul(args);
            break;
        case GSDDMM_OP::Div:
            gsddmm_backward_launch_div(args);
            break;
        case GSDDMM_OP::Dot:
            gsddmm_backward_launch_dot(args);
            break;
        case GSDDMM_OP::Copy:
            gsddmm_backward_launch_copy(args);
            break;
        default:
            TORCH_CHECK(false, "GSDDMM backward: op '", op, "' has no launcher");
    }

    CUDA_KERNEL_CHECK();

    return {dL, dR};
}

// Edge-parallel backward. Perfectly load balanced, at the cost of accumulating
// each node gradient with atomics into an fp32 buffer that is cast back here.
//
// edge_list_dst: [E, 2] uint64 (src, dst) pairs in forward-CSR (destination
//              grouped) order -- the destination-side pass.
// edge_list_src: the same edges grouped by source, plus canonical_edge_idx
//              mapping its slots to forward-CSR ids -- the source-side pass.
//              Required only when a source-side gradient is needed; pass the
//              dst-grouped list (and no permutation) for a graph whose orders
//              coincide.
// Returns:     {dL, dR}, as for gsddmm_backward_cuda.
std::vector<torch::Tensor> gsddmm_backward_edge_blocks(
    torch::Tensor L,
    torch::Tensor R,
    torch::Tensor dO,
    torch::Tensor edge_list_dst,
    std::optional<torch::Tensor>
        edge_list_src,
    std::optional<torch::Tensor>
        canonical_edge_idx,
    std::string op,
    std::string lhs_target,
    std::string rhs_target,
    uint64_t N,
    uint32_t edges_per_warp,
    uint32_t warps_per_block
) {
    const GSDDMM_OP op_enum        = parse_op(op);
    const GSDDMM_MEMBER lhs_member = parse_member(lhs_target, "lhs");
    const GSDDMM_MEMBER rhs_member = parse_member(rhs_target, "rhs");
    const BackwardPasses passes    = backward_passes(op_enum, lhs_member, rhs_member);

    at::cuda::CUDAGuard device_guard(L.device());
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream(L.device().index());

    auto check_edge_list = [](const torch::Tensor& t, const char *name) {
        TORCH_CHECK(t.is_cuda() && t.dim() == 2 && t.size(1) == 2, name, " must be a CUDA [E, 2] list of node-id pairs");
        TORCH_CHECK(t.scalar_type() == at::kUInt64, name, " must be uint64");
        TORCH_CHECK(t.stride(1) == 1, name, " must be contiguous in its second dim");
    };
    check_edge_list(edge_list_dst, "edge_list_dst");

    const int64_t E = edge_list_dst.size(0);
    const int64_t D = check_backward_operands(L, R, dO, op_enum, lhs_member, rhs_member, static_cast<int64_t>(N), E);

    TORCH_CHECK(
        edges_per_warp >= 1 && edges_per_warp <= kGsddmmEdgeMaxEdgesPerWarp, "GSDDMM backward (edge blocks): edges_per_warp must be in [1, ",
        kGsddmmEdgeMaxEdgesPerWarp, "], got ", edges_per_warp
    );
    TORCH_CHECK(
        warps_per_block >= 1 && warps_per_block <= kGsddmmEdgeMaxWarpsPerBlock,
        "GSDDMM backward (edge blocks): warps_per_block must be in [1, ", kGsddmmEdgeMaxWarpsPerBlock, "], got ", warps_per_block
    );

    // The source pass needs its own grouping; without one it would lose the
    // shared-row ballot that keeps atomic traffic down, and without the
    // permutation it could not address dO at all.
    torch::Tensor src_list                  = edge_list_dst;
    unsigned long long const *canonical_ptr = nullptr;
    if (passes.src) {
        if (edge_list_src.has_value() && edge_list_src->numel() > 0) {
            check_edge_list(*edge_list_src, "edge_list_src");
            TORCH_CHECK(edge_list_src->size(0) == E, "GSDDMM backward (edge blocks): both edge lists must cover the same E");
            src_list = *edge_list_src;
        }
        if (canonical_edge_idx.has_value() && canonical_edge_idx->numel() > 0) {
            const torch::Tensor& canonical = *canonical_edge_idx;
            TORCH_CHECK(
                canonical.is_cuda() && canonical.is_contiguous() && canonical.scalar_type() == at::kUInt64,
                "GSDDMM backward (edge blocks): canonical_edge_idx must be a contiguous CUDA uint64 tensor"
            );
            TORCH_CHECK(canonical.numel() == E, "GSDDMM backward (edge blocks): canonical_edge_idx must have E=", E, " entries");
            canonical_ptr = reinterpret_cast<unsigned long long const *>(canonical.data_ptr<uint64_t>());
        } else {
            TORCH_CHECK(
                src_list.data_ptr() == edge_list_dst.data_ptr(),
                "GSDDMM backward (edge blocks): a source-grouped edge list also needs canonical_edge_idx"
            );
        }
    }

    auto rows_of           = [&](GSDDMM_MEMBER member) { return member == GSDDMM_MEMBER::Edge ? E : static_cast<int64_t>(N); };
    const auto f32_options = L.options().dtype(torch::kFloat32);

    torch::Tensor dL = torch::empty({rows_of(lhs_member), D}, L.options());
    torch::Tensor dR = (op_enum == GSDDMM_OP::Copy) ? torch::empty({0}, L.options()) : torch::empty({rows_of(rhs_member), D}, L.options());
    // Node gradients are accumulated atomically, so they start at zero and live
    // in fp32; an edge operand's gradient is a plain store into dL / dR directly.
    const bool lhs_is_node = lhs_member != GSDDMM_MEMBER::Edge;
    const bool rhs_is_node = op_enum != GSDDMM_OP::Copy && rhs_member != GSDDMM_MEMBER::Edge;
    torch::Tensor dL_f32   = lhs_is_node ? torch::zeros({rows_of(lhs_member), D}, f32_options) : torch::empty({0}, f32_options);
    torch::Tensor dR_f32   = rhs_is_node ? torch::zeros({rows_of(rhs_member), D}, f32_options) : torch::empty({0}, f32_options);

    GsddmmBackwardLaunchArgsEdge args{
        .L                  = L,
        .R                  = R,
        .dO                 = dO,
        .dL                 = dL,
        .dR                 = dR,
        .dL_f32             = dL_f32,
        .dR_f32             = dR_f32,
        .edge_nodes_idx_dst = reinterpret_cast<ulonglong2 const *>(edge_list_dst.data_ptr<uint64_t>()),
        .edge_nodes_idx_src = reinterpret_cast<ulonglong2 const *>(src_list.data_ptr<uint64_t>()),
        .canonical_edge_idx = canonical_ptr,
        .stream             = stream,
        .E                  = static_cast<uint64_t>(E),
        .D                  = static_cast<uint64_t>(D),
        .key                = LRO{lhs_member, rhs_member, op_enum},
        .edges_per_warp     = static_cast<uint8_t>(edges_per_warp),
        .warps_per_block    = static_cast<uint8_t>(warps_per_block),
    };

    switch (op_enum) {
        case GSDDMM_OP::Add:
            gsddmm_backward_edge_launch_add(args);
            break;
        case GSDDMM_OP::Sub:
            gsddmm_backward_edge_launch_sub(args);
            break;
        case GSDDMM_OP::Mul:
            gsddmm_backward_edge_launch_mul(args);
            break;
        case GSDDMM_OP::Div:
            gsddmm_backward_edge_launch_div(args);
            break;
        case GSDDMM_OP::Dot:
            gsddmm_backward_edge_launch_dot(args);
            break;
        case GSDDMM_OP::Copy:
            gsddmm_backward_edge_launch_copy(args);
            break;
        default:
            TORCH_CHECK(false, "GSDDMM backward edge: op '", op, "' has no launcher");
    }

    CUDA_KERNEL_CHECK();

    // Cast the atomically accumulated node gradients back to the input dtype.
    if (lhs_is_node) {
        dL = dL_f32.to(L.scalar_type());
    }
    if (rhs_is_node) {
        dR = dR_f32.to(L.scalar_type());
    }

    return {dL, dR};
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
    uint32_t pipeline_stages,
    uint32_t edges_per_warp,
    uint32_t warps_per_block,
    std::optional<torch::Tensor> canonical_edge_idx = std::nullopt
) {
    return gsddmm::gsddmm_forward_edge_blocks(
        std::move(L), std::move(R), std::move(edge_list), std::move(op), std::move(lhs_target), std::move(rhs_target), N, pipeline_stages,
        edges_per_warp, warps_per_block, std::move(canonical_edge_idx)
    );
}

std::vector<torch::Tensor> gsddmm_backward_cuda(
    torch::Tensor L,
    torch::Tensor R,
    torch::Tensor dO,
    torch::Tensor row_ptr,
    torch::Tensor col_idx,
    torch::Tensor row_ptr_T,
    torch::Tensor col_idx_T,
    std::string op,
    std::string lhs_target,
    std::string rhs_target,
    torch::Tensor light_nodes,
    torch::Tensor heavy_nodes,
    torch::Tensor light_nodes_T,
    torch::Tensor heavy_nodes_T,
    std::optional<torch::Tensor> canonical_edge_idx = std::nullopt,
    uint32_t light_warps_per_block                  = 4,
    uint32_t heavy_warps_per_block                  = 32
) {
    return gsddmm::gsddmm_backward_cuda(
        std::move(L), std::move(R), std::move(dO), std::move(row_ptr), std::move(col_idx), std::move(row_ptr_T), std::move(col_idx_T),
        std::move(op), std::move(lhs_target), std::move(rhs_target), std::move(light_nodes), std::move(heavy_nodes), std::move(light_nodes_T),
        std::move(heavy_nodes_T), std::move(canonical_edge_idx), light_warps_per_block, heavy_warps_per_block
    );
}

std::vector<torch::Tensor> gsddmm_backward_edge_blocks(
    torch::Tensor L,
    torch::Tensor R,
    torch::Tensor dO,
    torch::Tensor edge_list_dst,
    std::optional<torch::Tensor> edge_list_src      = std::nullopt,
    std::optional<torch::Tensor> canonical_edge_idx = std::nullopt,
    std::string op                                  = "mul",
    std::string lhs_target                          = "src",
    std::string rhs_target                          = "dst",
    uint64_t N                                      = 0,
    uint32_t edges_per_warp                         = 4,
    uint32_t warps_per_block                        = 4
) {
    return gsddmm::gsddmm_backward_edge_blocks(
        std::move(L), std::move(R), std::move(dO), std::move(edge_list_dst), std::move(edge_list_src), std::move(canonical_edge_idx),
        std::move(op), std::move(lhs_target), std::move(rhs_target), N, edges_per_warp, warps_per_block
    );
}
