#include <torch/extension.h>
#include <torch/torch.h>

#include <cstdint>
#include <string>
#include <variant>

#include "gsddmm/gsddmm.cu"

// =============================================================================
// Host-side launcher for the GSDDMM forward kernel.
//
// The kernel is fully templated on (op, lhs member, rhs member, warps/block, D,
// dtype, index type, pipeline stages) — every combination is a separate
// instantiation, so the variant sets below directly control this file's
// compile time. They cover the full op x member matrix and the usual
// dtype/D/warps/stages grid of the other kernels in this repo; trim the
// MakeIntVariant / MakeTypeVariant / MakeIndexVariant lists (or split this
// file per op) if the build gets too slow.
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

template <typename EnumT, EnumT... Values>
std::variant<std::integral_constant<EnumT, Values>...> MakeEnumVariant(EnumT value) {
    std::variant<std::integral_constant<EnumT, Values>...> result;
    bool found = false;
    (
        [&] {
            if (value == Values) {
                result.template emplace<std::integral_constant<EnumT, Values>>();
                found = true;
            }
        }(),
        ...);
    TORCH_CHECK(found, "GSDDMM: enum value not in the dispatch set");
    return result;
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
//              For op == "copy" R is never read (pass L).
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
    uint32_t light_warps_per_block = 1,
    uint32_t heavy_warps_per_block = 8,
    uint32_t pipeline_stages       = 0
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
    // Only the index types listed in the dispatch below are instantiated; extend
    // MakeIndexVariant there (at the cost of compile time) to support more.
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

    auto check_rows = [&](const torch::Tensor& t, GSDDMM_MEMBER member, const char *name) {
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

    auto op_variant =
        MakeEnumVariant<GSDDMM_OP, GSDDMM_OP::Add, GSDDMM_OP::Sub, GSDDMM_OP::Mul, GSDDMM_OP::Div, GSDDMM_OP::Dot, GSDDMM_OP::Copy>(op_enum);
    auto ll_variant = MakeEnumVariant<GSDDMM_MEMBER, GSDDMM_MEMBER::Src_V, GSDDMM_MEMBER::Dst_V, GSDDMM_MEMBER::Edge>(lhs_member);
    auto rr_variant = MakeEnumVariant<GSDDMM_MEMBER, GSDDMM_MEMBER::Src_V, GSDDMM_MEMBER::Dst_V, GSDDMM_MEMBER::Edge>(rhs_member);

    // Lambda to launch the kernel for a bucket of nodes with a given warp count
    auto launch_bucket = [&](torch::Tensor& node_indices, int num_nodes_bucket, auto warp_variant) {
        if (num_nodes_bucket == 0) return;

        std::visit(
            [&](auto op_c, auto ll_c, auto rr_c, auto idxInfo, auto typeInfo, auto d_c, auto warp_c, auto stages_c) {
                constexpr GSDDMM_OP OP     = decltype(op_c)::value;
                constexpr GSDDMM_MEMBER LL = decltype(ll_c)::value;
                constexpr GSDDMM_MEMBER RR = decltype(rr_c)::value;
                using index_t              = typename decltype(idxInfo)::Type;
                using torch_t              = typename decltype(typeInfo)::TorchType;
                using cuda_t               = typename decltype(typeInfo)::CudaType;
                constexpr size_t DC        = decltype(d_c)::value;
                constexpr size_t W         = decltype(warp_c)::value;
                constexpr int STAGES       = decltype(stages_c)::value;

                cuda_t const *L_ptr = reinterpret_cast<const cuda_t *>(L.data_ptr<torch_t>());
                cuda_t const *R_ptr = reinterpret_cast<const cuda_t *>(R.data_ptr<torch_t>());
                cuda_t *O_ptr       = reinterpret_cast<cuda_t *>(O.data_ptr<torch_t>());

                auto kernel = GSDDMM_forward_edge_block<OP, LL, RR, W, DC, cuda_t, index_t, float, STAGES>;

                constexpr size_t shmem = gsddmm_forward_shmem_bytes<OP, LL, RR, W, DC, cuda_t, STAGES>();
                ensure_dynamic_shmem(kernel, shmem, "GSDDMM forward");

                dim3 blocks(num_nodes_bucket);
                dim3 threads(kWarpSize, W);

                kernel<<<blocks, threads, shmem, stream>>>(
                    static_cast<size_t>(N), L_ptr, R_ptr, O_ptr, index_ptr<index_t>(row_ptr), index_ptr<index_t>(col_idx),
                    index_ptr<index_t>(node_indices)
                );
            },
            op_variant, ll_variant, rr_variant, MakeIndexVariant<int32_t, int64_t>(idx_dtype),
            MakeTypeVariant<float, at::Half, at::BFloat16>(L.scalar_type()), MakeIntVariant<32, 64, 128, 256>(static_cast<int>(D)),
            warp_variant, MakeIntVariant<0, 1>(pipeline_stages)
        );
    };

    // Light nodes
    launch_bucket(light_nodes, light_nodes.numel(), MakeIntVariant<1, 2, 4>(light_warps_per_block));

    // Heavy nodes
    launch_bucket(heavy_nodes, heavy_nodes.numel(), MakeIntVariant<8, 16, 32>(heavy_warps_per_block));

    CUDA_KERNEL_CHECK();

    return O;
}

};  // namespace gsddmm
