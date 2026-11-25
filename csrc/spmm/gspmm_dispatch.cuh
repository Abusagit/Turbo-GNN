#pragma once

#include <ATen/cuda/CUDAContext.h>

#include <limits>

#include "spmm/gspmm_edge_grad.cuh"
#include "spmm/gspmm_launch.h"

// =============================================================================
// The g-SpMM dispatch grid of one op, instantiated by the shards (gspmm_shard.inc) only.
// =============================================================================

namespace turbo_gnn::gspmm {

namespace red = turbo_gnn::reduction;

// The forward of op BOP on torch_t values: one reduction walk with message
// Message<BOP, BCAST, EMAP, ARG_IS_EDGE>. Only the combinations the binding admits are instantiated:
// broadcast for ops that combine node and edge data, an edge map for the sum backward of mul and div
// -- the only transposed walk that reads edge data -- and edge-position args for min/max, sums
// having none (which makes copy_u's sum exactly reduction_aggr's kernel).
template <BinaryOp BOP, typename torch_t>
void forward_shard(ForwardArgs const& a) {
    using Ops = BinaryOps<BOP>;
    std::visit(
        [&](auto idx_c, auto type_c, auto op_c, auto bcast_c, auto map_c) {
            using index_t            = typename decltype(idx_c)::Type;
            using cuda_t             = typename decltype(type_c)::CudaType;
            constexpr ReductionOp OP = decltype(op_c)::value;
            constexpr bool BCAST     = decltype(bcast_c)::value;
            constexpr bool EMAP      = decltype(map_c)::value;
            constexpr bool kAdmitted = (!BCAST || (Ops::USE_LHS && Ops::USE_RHS)) &&
                                       (!EMAP || (OP == ReductionOp::SUM && (BOP == BinaryOp::MUL || BOP == BinaryOp::DIV)));
            if constexpr (!kAdmitted) {
                TORCH_CHECK(false, "gspmm: unsupported combination of op, reducer, broadcast and edge map");
            } else {
                using M = red::Message<BOP, BCAST, EMAP, OP != ReductionOp::SUM>;
                red::ForwardParams<cuda_t, index_t> p{};
                if constexpr (M::kNode) {
                    p.X        = reinterpret_cast<cuda_t const *>(a.X.data_ptr());
                    p.stride_x = a.X.stride(0);
                }
                if constexpr (M::kEdge) {
                    p.E = reinterpret_cast<cuda_t const *>(a.E.data_ptr());
                }
                if constexpr (M::kMapped) {
                    p.edge_map = index_ptr<index_t>(a.edge_map);
                }
                a.plan->fill(p, a.row_ptr, a.col_idx, a.light, a.heavy, a.slice_size, OP != ReductionOp::SUM);
                red::run_forward<OP, M>(*a.plan, p, a.elem_size, "gspmm");
            }
        },
        MakeIndexWidthVariant(a.row_ptr.scalar_type()), MakeTypeVariant<torch_t>(c10::CppTypeToScalarType<torch_t>::value),
        MakeEnumVariant<ReductionOp, ReductionOp::MIN, ReductionOp::MAX, ReductionOp::SUM>(a.op), MakeBoolVariant<false, true>(a.bcast),
        MakeBoolVariant<false, true>(a.edge_map.defined() && a.edge_map.numel() > 0)
    );
}

// The sum backward's edge gradient of op BOP on torch_t values.
template <BinaryOp BOP, typename torch_t>
void edge_grad_shard(EdgeGradArgs const& a) {
    if constexpr (BOP == BinaryOp::COPY_U) {
        TORCH_CHECK(false, "gspmm: copy_u has no edge operand");
    } else {
        using Ops                 = BinaryOps<BOP>;
        const int64_t num_light   = a.light.numel();
        const int64_t num_slices  = a.slices->num;
        const size_t elem_size    = a.dE.element_size();
        const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        const uint32_t vec        = a.wide ? static_cast<uint32_t>(16 / elem_size) : 1u;
        const uint32_t nvec       = static_cast<uint32_t>(a.D) / vec;
        const int code            = rows::plan_sweep_shape(a.D, elem_size, a.wide);
        const uint32_t passes     = a.bcast ? 1u : rows::sweep_passes(code, nvec);
        const int64_t items       = (num_slices + num_light) * static_cast<int64_t>(passes);
        if (items == 0) {
            return;
        }
        std::visit(
            [&](auto idx_c, auto type_c, auto bcast_c, auto shape_c) {
                using index_t        = typename decltype(idx_c)::Type;
                using cuda_t         = typename decltype(type_c)::CudaType;
                constexpr bool BCAST = decltype(bcast_c)::value;
                constexpr int SHAPE  = decltype(shape_c)::value;
                if constexpr (BCAST && !(Ops::USE_LHS && Ops::USE_RHS)) {
                    TORCH_CHECK(false, "gspmm: copy_e never broadcasts");
                } else {
                    EdgeGradParams<cuda_t, index_t> p{};
                    p.G        = reinterpret_cast<cuda_t const *>(a.G.data_ptr());
                    p.stride_g = a.G.stride(0);
                    if constexpr (BOP == BinaryOp::MUL || BOP == BinaryOp::DIV) {
                        p.X        = reinterpret_cast<cuda_t const *>(a.X.data_ptr());
                        p.stride_x = a.X.stride(0);
                    }
                    if constexpr (BOP == BinaryOp::DIV) {
                        p.E = reinterpret_cast<cuda_t const *>(a.E.data_ptr());
                    }
                    p.dE          = reinterpret_cast<cuda_t *>(a.dE.data_ptr());
                    p.D           = static_cast<uint32_t>(a.D);
                    p.nvec        = nvec;
                    p.passes      = passes;
                    p.row_ptr     = index_ptr<index_t>(a.row_ptr);
                    p.col_idx     = index_ptr<index_t>(a.col_idx);
                    p.num_slices  = num_slices;
                    p.num_light   = num_light;
                    p.light_nodes = num_light > 0 ? index_ptr<index_t>(a.light) : nullptr;
                    if (num_slices > 0) {
                        p.heavy_nodes = index_ptr<index_t>(a.heavy);
                        p.chunk_node  = a.slices->chunk_node.data_ptr<int>();
                        p.chunk_start = a.slices->chunk_start.data_ptr<int>();
                        p.slice_size  = static_cast<uint32_t>(a.slice_size);
                    }
                    const int warps         = std::min(a.warps, static_cast<int>(kEdgeGradWarps));
                    const int64_t per_block = static_cast<int64_t>(warps) * static_cast<int64_t>(kWarpSize / Shape<SHAPE>::G);
                    const int64_t blocks    = ceil_div(items, per_block);
                    TORCH_CHECK(blocks <= std::numeric_limits<int32_t>::max(), "gspmm: too much work for one launch (", items, " items)");
                    edge_grad_kernel<BOP, BCAST, SHAPE, cuda_t, index_t>
                        <<<static_cast<unsigned>(blocks), static_cast<unsigned>(warps) * kWarpSize, 0, stream>>>(p);
                }
            },
            MakeIndexWidthVariant(a.row_ptr.scalar_type()), MakeTypeVariant<torch_t>(c10::CppTypeToScalarType<torch_t>::value),
            MakeBoolVariant<false, true>(a.bcast), rows::make_sweep_shape_variant(code)
        );
    }
}

}  // namespace turbo_gnn::gspmm
