#include <algorithm>

#include "spmm/gspmm.h"
#include "spmm/gspmm_kernels.cuh"

namespace {

ReductionOp reduce_op_from_string(const std::string& reduce) {
    if (reduce == "sum") return ReductionOp::SUM;
    if (reduce == "min") return ReductionOp::MIN;
    if (reduce == "max") return ReductionOp::MAX;
    return ReductionOp::SUM;
}

bool op_uses_lhs(BinaryOp op) { return op != BinaryOp::COPY_E; }
bool op_uses_rhs(BinaryOp op) { return op != BinaryOp::COPY_U; }

struct FeatureLayout {
    int64_t d;
    bool rhs_broadcast;
};

bool deduce_rhs_broadcast(BinaryOp op, const torch::Tensor& rhs, int64_t d) {
    if (!op_uses_rhs(op) || d <= 1) {
        return false;
    }
    return (rhs.dim() == 1) || (rhs.size(1) == 1);
}

FeatureLayout deduce_feature_layout(BinaryOp op, const torch::Tensor& lhs, const torch::Tensor& rhs) {
    int64_t d = 1;
    if (op_uses_lhs(op)) {
        TORCH_CHECK(lhs.dim() == 2, "lhs must be 2-D [N, d] for op '", static_cast<int>(op), "', got ", lhs.dim(), "-D");
        d = lhs.size(1);
    } else {
        TORCH_CHECK(rhs.dim() == 1 || rhs.dim() == 2, "rhs must be [E], [E, 1] or [E, d], got ", rhs.dim(), "-D");
        d = (rhs.dim() > 1) ? rhs.size(1) : 1;
    }

    return {d, deduce_rhs_broadcast(op, rhs, d)};
}

void check_common_inputs(
    const torch::Tensor& edge_ptr, const torch::Tensor& edge_idx, const torch::Tensor& lhs, const torch::Tensor& rhs, BinaryOp bop,
    const FeatureLayout& layout
) {
    TORCH_CHECK(edge_ptr.is_cuda() && edge_idx.is_cuda(), "CSR tensors must be CUDA");
    TORCH_CHECK(edge_ptr.is_contiguous() && edge_idx.is_contiguous(), "CSR tensors must be contiguous");

    const auto idx_dtype = edge_ptr.scalar_type();
    TORCH_CHECK(
        idx_dtype == torch::kInt || idx_dtype == torch::kLong, "g-SpMM index tensors must be int32 or int64 (got ", idx_dtype,
        "); unsigned index types are supported by reduction_aggr only"
    );
    TORCH_CHECK(edge_idx.scalar_type() == idx_dtype, "edge_idx must have the same dtype as edge_ptr");

    const int64_t num_edges = edge_idx.numel();

    if (op_uses_lhs(bop)) {
        TORCH_CHECK(lhs.is_cuda() && lhs.is_contiguous(), "lhs must be a contiguous CUDA tensor");
        TORCH_CHECK(lhs.dim() == 2, "lhs must be 2-D [N, d], got ", lhs.dim(), "-D");
        TORCH_CHECK(lhs.size(0) == edge_ptr.numel() - 1, "lhs.size(0) (", lhs.size(0), ") must equal N = edge_ptr.numel() - 1");
        TORCH_CHECK(
            lhs.scalar_type() == torch::kFloat || lhs.scalar_type() == torch::kHalf || lhs.scalar_type() == torch::kBFloat16,
            "lhs must be float32/float16/bfloat16"
        );
    }

    if (op_uses_rhs(bop)) {
        TORCH_CHECK(rhs.is_cuda() && rhs.is_contiguous(), "rhs must be a contiguous CUDA tensor");
        TORCH_CHECK(rhs.dim() == 1 || rhs.dim() == 2, "rhs must be [E], [E, 1] or [E, d], got ", rhs.dim(), "-D");
        TORCH_CHECK(rhs.size(0) == num_edges, "rhs.size(0) (", rhs.size(0), ") must equal the edge count E = ", num_edges);
        if (!layout.rhs_broadcast && rhs.dim() == 2) {
            TORCH_CHECK(rhs.size(1) == layout.d, "rhs.size(1) (", rhs.size(1), ") must equal d = ", layout.d, " or 1 to broadcast");
        }
        if (op_uses_lhs(bop)) {
            TORCH_CHECK(
                rhs.scalar_type() == lhs.scalar_type(), "rhs dtype (", rhs.scalar_type(), ") must match lhs dtype (", lhs.scalar_type(), ")"
            );
        }
        TORCH_CHECK(
            rhs.scalar_type() == torch::kFloat || rhs.scalar_type() == torch::kHalf || rhs.scalar_type() == torch::kBFloat16,
            "rhs must be float32/float16/bfloat16"
        );
    }
}

// The dtype the kernels are templated on: whichever operand the op reads.
at::ScalarType value_dtype(BinaryOp bop, const torch::Tensor& lhs, const torch::Tensor& rhs) {
    return op_uses_lhs(bop) ? lhs.scalar_type() : rhs.scalar_type();
}

// g-SpMM picks its block shape at runtime, so it pins the __launch_bounds__
// template parameter of the shared kernels to the CUDA maximum rather than
// instantiating one kernel per warp count -- the cross product is already six
// ops by three reducers wide.  The kernels stride by blockDim.x, so the only
// consequence is the register budget nvcc plans for, and with VECTORIZE ==
// false these bodies are nowhere near the 64 registers a 1024-thread block
// leaves them.
constexpr size_t kGSpMMWarpsPerBlock = 1024 / kWarpSize;

// The vectorized TileOps path casts to a 16-byte aligned type, which traps
// unless every row start is 16-byte aligned.  g-SpMM accepts any feature width
// (d = 65 among them), so it always takes the scalar path.
constexpr bool kGSpMMVectorize = false;

}  // namespace

std::vector<torch::Tensor> gspmm_forward(
    const torch::Tensor& edge_ptr,
    const torch::Tensor& edge_idx,
    const torch::Tensor& lhs,
    const torch::Tensor& rhs,
    const torch::Tensor& light_nodes,
    const torch::Tensor& heavy_nodes,
    const std::string& op,
    const std::string& reduce,
    int warps_per_block,
    int features_per_block,
    int tiles_y
) {
    const BinaryOp bop    = binary_op_from_string(op);
    const ReductionOp rop = reduce_op_from_string(reduce);
    const auto layout     = deduce_feature_layout(bop, lhs, rhs);

    check_common_inputs(edge_ptr, edge_idx, lhs, rhs, bop, layout);

    TORCH_CHECK(light_nodes.is_cuda() && heavy_nodes.is_cuda(), "node buckets must be CUDA");
    TORCH_CHECK(
        light_nodes.scalar_type() == edge_ptr.scalar_type() && heavy_nodes.scalar_type() == edge_ptr.scalar_type(),
        "node buckets must have the same dtype as edge_ptr"
    );

    const int64_t num_nodes = edge_ptr.numel() - 1;
    const int64_t num_light = light_nodes.numel();
    const int64_t num_heavy = heavy_nodes.numel();
    TORCH_CHECK(
        num_light + num_heavy == num_nodes, "light_nodes (", num_light, ") + heavy_nodes (", num_heavy, ") must cover all ", num_nodes,
        " nodes -- output rows outside both buckets would be left uninitialized"
    );

    TORCH_CHECK(warps_per_block > 0 && warps_per_block <= 32, "warps_per_block must be in [1, 32]");
    if (num_heavy > 0) {
        TORCH_CHECK(tiles_y > 0 && tiles_y <= 32, "tiles_y must be in [1, 32]");
        TORCH_CHECK((tiles_y & (tiles_y - 1)) == 0, "tiles_y must be a power of 2 (shared-memory tree reduction)");
        TORCH_CHECK(features_per_block > 0 && features_per_block <= 1024, "features_per_block must be in [1, 1024]");
        TORCH_CHECK(features_per_block * tiles_y <= 1024, "features_per_block * tiles_y must be <= 1024");
    }

    const torch::Tensor& val_ref = op_uses_lhs(bop) ? lhs : rhs;

    auto out = torch::empty({num_nodes, layout.d}, val_ref.options());
    // min/max need the winning edge for their backward; sum does not, and at
    // [N, d] of index dtype that allocation would cost as much as the output.
    const bool tracks_arg = (rop != ReductionOp::SUM);
    auto arg_eid          = tracks_arg ? torch::empty({num_nodes, layout.d}, edge_ptr.options()) : torch::empty({0}, edge_ptr.options());

    std::visit(
        [&](auto idxInfo, auto typeInfo, auto op_c, auto rop_c, auto bcast_c) {
            using index_t = typename decltype(idxInfo)::Type;
            using torch_t = typename decltype(typeInfo)::TorchType;
            using cuda_t  = typename decltype(typeInfo)::CudaType;

            constexpr BinaryOp BOP    = static_cast<BinaryOp>(decltype(op_c)::value);
            constexpr ReductionOp ROP = static_cast<ReductionOp>(decltype(rop_c)::value);
            constexpr bool BCAST      = decltype(bcast_c)::value;
            using BOps                = BinaryOps<BOP>;

            // deduce_rhs_broadcast never raises the flag for an operation that
            // ignores edge data, so this half of the cross product is
            // unreachable -- pruned here rather than instantiated and skipped.
            if constexpr (BCAST && !BOps::USE_RHS) {
                return;
            } else {
                cuda_t const *lhs_ptr = nullptr;
                if constexpr (BOps::USE_LHS) {
                    lhs_ptr = reinterpret_cast<cuda_t const *>(lhs.data_ptr<torch_t>());
                }
                cuda_t const *rhs_ptr = nullptr;
                if constexpr (BOps::USE_RHS) {
                    rhs_ptr = reinterpret_cast<cuda_t const *>(rhs.data_ptr<torch_t>());
                }
                cuda_t *out_ptr = reinterpret_cast<cuda_t *>(out.data_ptr<torch_t>());
                const size_t d  = static_cast<size_t>(layout.d);

                if (num_light > 0) {
                    // Features along x, capped at the block; nodes fill y.
                    const size_t threads = static_cast<size_t>(warps_per_block) * kWarpSize;
                    const size_t tile_x  = std::min<size_t>(std::max<size_t>(d, 1), threads);
                    const size_t node_y  = std::max<size_t>(threads / tile_x, 1);

                    const dim3 block_l(static_cast<unsigned>(tile_x), static_cast<unsigned>(node_y));
                    const unsigned blocks_l = static_cast<unsigned>((static_cast<size_t>(num_light) + node_y - 1) / node_y);

                    reduction_aggr_forward_light_kernel_1d<
                        kGSpMMWarpsPerBlock, cuda_t, ROP, index_t, float, /*PIPELINE_STAGES=*/0, BOP, BCAST, /*ARG_IS_EDGE=*/true,
                        kGSpMMVectorize
                    ><<<blocks_l, block_l>>>(
                        index_ptr<index_t>(light_nodes),
                        index_ptr<index_t>(edge_ptr),
                        index_ptr<index_t>(edge_idx),
                        lhs_ptr,
                        out_ptr,
                        index_ptr_mut<index_t>(arg_eid),
                        d,
                        static_cast<size_t>(num_light),
                        rhs_ptr
                    );
                }

                if (num_heavy > 0) {
                    const dim3 grid_h(static_cast<unsigned>(num_heavy));
                    const dim3 block_h(static_cast<unsigned>(features_per_block), static_cast<unsigned>(tiles_y));
                    // Same expression the kernel places its index array with.
                    const size_t shmem =
                        aggr_heavy_shmem_bytes<index_t>(static_cast<size_t>(tiles_y) * static_cast<size_t>(features_per_block));

                    ensure_dynamic_shmem(
                        reduction_aggr_forward_heavy_kernel_2d<cuda_t, ROP, index_t, float, BOP, BCAST, /*ARG_IS_EDGE=*/true, kGSpMMVectorize>,
                        shmem, "gspmm heavy"
                    );

                    reduction_aggr_forward_heavy_kernel_2d<cuda_t, ROP, index_t, float, BOP, BCAST, /*ARG_IS_EDGE=*/true, kGSpMMVectorize>
                        <<<grid_h, block_h, shmem>>>(
                            index_ptr<index_t>(heavy_nodes),
                            index_ptr<index_t>(edge_ptr),
                            index_ptr<index_t>(edge_idx),
                            lhs_ptr,
                            out_ptr,
                            index_ptr_mut<index_t>(arg_eid),
                            d,
                            rhs_ptr
                        );
                }
            }
        },
        MakeIndexVariant<int32_t, int64_t>(edge_ptr.scalar_type()),
        MakeTypeVariant<float, at::Half, at::BFloat16>(value_dtype(bop, lhs, rhs)),
        MakeIntVariant<0, 1, 2, 3, 4, 5>(static_cast<int>(bop)),
        MakeIntVariant<0, 1, 2>(static_cast<int>(rop)),
        MakeBoolVariant<false, true>(layout.rhs_broadcast)
    );

    CUDA_KERNEL_CHECK();

    return {out, arg_eid};
}

std::vector<torch::Tensor> gspmm_backward_arg(
    const torch::Tensor& grad_out,
    const torch::Tensor& arg_eid,
    const torch::Tensor& edge_idx,
    const torch::Tensor& lhs,
    const torch::Tensor& rhs,
    const std::string& op,
    int warps_per_block
) {
    const BinaryOp bop = binary_op_from_string(op);

    const int64_t num_nodes = grad_out.size(0);
    const int64_t d         = grad_out.size(1);

    const bool uses_lhs = op_uses_lhs(bop);
    const bool uses_rhs = op_uses_rhs(bop);

    const auto grad_opts = grad_out.options().dtype(torch::kFloat);
    auto grad_lhs        = uses_lhs ? torch::zeros({num_nodes, d}, grad_opts) : torch::empty({0}, grad_opts);
    auto grad_rhs        = uses_rhs ? torch::zeros(rhs.sizes(), grad_opts) : torch::empty({0}, grad_opts);

    // No reducer axis here: min and max share one scatter.
    std::visit(
        [&](auto idxInfo, auto typeInfo, auto op_c, auto bcast_c) {
            using index_t = typename decltype(idxInfo)::Type;
            using torch_t = typename decltype(typeInfo)::TorchType;
            using cuda_t  = typename decltype(typeInfo)::CudaType;

            constexpr BinaryOp BOP = static_cast<BinaryOp>(decltype(op_c)::value);
            constexpr bool BCAST   = decltype(bcast_c)::value;
            using BOps             = BinaryOps<BOP>;

            if constexpr (BCAST && !BOps::USE_RHS) {
                return;
            } else {
                cuda_t const *lhs_ptr = nullptr;
                cuda_t const *rhs_ptr = nullptr;
                // Only mul and div differentiate to something that reads the
                // operands; the others would be loading from a null pointer.
                if constexpr (BOps::GRAD_USES_OPERANDS) {
                    lhs_ptr = reinterpret_cast<cuda_t const *>(lhs.data_ptr<torch_t>());
                    rhs_ptr = reinterpret_cast<cuda_t const *>(rhs.data_ptr<torch_t>());
                }

                const unsigned threads = static_cast<unsigned>(static_cast<size_t>(warps_per_block) * kWarpSize);

                reduction_aggr_backward_typed<kGSpMMWarpsPerBlock, cuda_t, index_t, BOP, BCAST, /*ARG_IS_EDGE=*/true, /*grad_t=*/float>
                    <<<static_cast<unsigned>(num_nodes), threads>>>(
                        reinterpret_cast<cuda_t const *>(grad_out.data_ptr<torch_t>()),
                        index_ptr<index_t>(arg_eid),
                        grad_lhs.data_ptr<float>(),
                        static_cast<size_t>(num_nodes),
                        static_cast<size_t>(d),
                        index_ptr<index_t>(edge_idx),
                        lhs_ptr,
                        rhs_ptr,
                        grad_rhs.data_ptr<float>()
                    );
            }
        },
        MakeIndexVariant<int32_t, int64_t>(edge_idx.scalar_type()),
        MakeTypeVariant<float, at::Half, at::BFloat16>(grad_out.scalar_type()),
        MakeIntVariant<0, 1, 2, 3, 4, 5>(static_cast<int>(bop)),
        MakeBoolVariant<false, true>(deduce_rhs_broadcast(bop, rhs, d))
    );

    CUDA_KERNEL_CHECK();

    return {grad_lhs, grad_rhs};
}

torch::Tensor gspmm_backward_edge(
    const torch::Tensor& edge_ptr,
    const torch::Tensor& edge_idx,
    const torch::Tensor& grad_out,
    const torch::Tensor& lhs,
    const torch::Tensor& rhs,
    const std::string& op,
    int warps_per_block
) {
    const BinaryOp bop = binary_op_from_string(op);

    const auto grad_opts = grad_out.options().dtype(torch::kFloat);
    if (!op_uses_rhs(bop)) {
        return torch::empty({0}, grad_opts);  // copy_u has no edge operand
    }

    TORCH_CHECK(grad_out.is_cuda() && edge_ptr.is_cuda() && edge_idx.is_cuda(), "inputs must be CUDA");
    TORCH_CHECK(grad_out.dim() == 2, "grad_out must be 2-D [N, d]");
    TORCH_CHECK(grad_out.size(0) == edge_ptr.numel() - 1, "grad_out.size(0) must equal N = edge_ptr.numel() - 1");
    TORCH_CHECK(warps_per_block > 0 && warps_per_block <= 32, "warps_per_block must be in [1, 32]");

    // Zeroed because the broadcast path accumulates with atomicAdd.  The
    // non-broadcast path writes every (eid, f) slot exactly once, so the memset
    // is redundant there but too cheap to branch on.
    auto grad_rhs = torch::zeros(rhs.sizes(), grad_opts);

    const int64_t num_nodes = grad_out.size(0);
    const int64_t d         = grad_out.size(1);

    std::visit(
        [&](auto idxInfo, auto typeInfo, auto op_c, auto bcast_c) {
            using index_t = typename decltype(idxInfo)::Type;
            using torch_t = typename decltype(typeInfo)::TorchType;
            using cuda_t  = typename decltype(typeInfo)::CudaType;

            constexpr BinaryOp BOP = static_cast<BinaryOp>(decltype(op_c)::value);
            constexpr bool BCAST   = decltype(bcast_c)::value;
            using BOps             = BinaryOps<BOP>;

            // copy_u returned above, so it never reaches a launch; the
            // broadcast flag is likewise impossible without an edge operand.
            if constexpr (!BOps::USE_RHS || (BCAST && !BOps::USE_RHS)) {
                return;
            } else {
                cuda_t const *lhs_ptr = nullptr;
                cuda_t const *rhs_ptr = nullptr;
                if constexpr (BOps::GRAD_USES_OPERANDS) {
                    lhs_ptr = reinterpret_cast<cuda_t const *>(lhs.data_ptr<torch_t>());
                    rhs_ptr = reinterpret_cast<cuda_t const *>(rhs.data_ptr<torch_t>());
                }

                const unsigned threads = static_cast<unsigned>(static_cast<size_t>(warps_per_block) * kWarpSize);
                // One block per destination node, capped: the kernel
                // grid-strides over whatever does not fit.
                const unsigned blocks = static_cast<unsigned>(std::min<int64_t>(num_nodes, 65535));

                gspmm_backward_edge_kernel<BOP, BCAST, cuda_t, index_t><<<blocks, threads>>>(
                    index_ptr<index_t>(edge_ptr),
                    index_ptr<index_t>(edge_idx),
                    reinterpret_cast<cuda_t const *>(grad_out.data_ptr<torch_t>()),
                    lhs_ptr,
                    rhs_ptr,
                    grad_rhs.data_ptr<float>(),
                    static_cast<size_t>(num_nodes),
                    static_cast<size_t>(d)
                );
            }
        },
        MakeIndexVariant<int32_t, int64_t>(edge_ptr.scalar_type()),
        MakeTypeVariant<float, at::Half, at::BFloat16>(grad_out.scalar_type()),
        MakeIntVariant<0, 1, 2, 3, 4, 5>(static_cast<int>(bop)),
        MakeBoolVariant<false, true>(deduce_rhs_broadcast(bop, rhs, d))
    );

    CUDA_KERNEL_CHECK();

    return grad_rhs;
}
