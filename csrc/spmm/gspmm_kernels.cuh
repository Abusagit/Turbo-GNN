#pragma once

#include "common.cuh"
#include "common/gspmm_ops.cuh"
#include "reduction/reduction_aggr_kernels.cuh"

template <BinaryOp BOp, bool RHS_BROADCAST, FloatingNum cuda_t, typename index_t, FloatingNum accum_t = float>
__global__ void gspmm_backward_edge_kernel(
    index_t const *const __restrict__ edge_ptr,
    index_t const *const __restrict__ edge_idx,
    cuda_t const *const __restrict__ grad_out,
    cuda_t const *const __restrict__ lhs,
    cuda_t const *const __restrict__ rhs,
    accum_t *const __restrict__ grad_rhs,
    size_t num_nodes,
    size_t d
) {
    using BOps = BinaryOps<BOp>;

    for (size_t v = blockIdx.x; v < num_nodes; v += gridDim.x) {
        const index_t row_start = edge_ptr[v];
        const index_t row_end   = edge_ptr[v + 1];
        const size_t base       = v * d;

        for (index_t eid = row_start; eid < row_end; ++eid) {
            index_t u = index_t{};
            if constexpr (BOps::GRAD_USES_OPERANDS) {
                u = edge_idx[eid];
            }

            if constexpr (RHS_BROADCAST) {
                accum_t partial{};
                for (size_t f = threadIdx.x; f < d; f += blockDim.x) {
                    const accum_t g = static_cast<accum_t>(grad_out[base + f]);
                    accum_t u_val{};
                    accum_t e_val{};
                    if constexpr (BOps::GRAD_USES_OPERANDS) {
                        u_val = static_cast<accum_t>(lhs[static_cast<size_t>(u) * d + f]);
                        e_val = static_cast<accum_t>(rhs[static_cast<size_t>(eid)]);
                    }
                    partial += BOps::grad_rhs(u_val, e_val, g);
                }
                partial = warp_reduce_sum(partial);
                if ((threadIdx.x % kWarpSize) == 0) {
                    atomicAdd(&grad_rhs[static_cast<size_t>(eid)], partial);
                }
            } else {
                for (size_t f = threadIdx.x; f < d; f += blockDim.x) {
                    const accum_t g = static_cast<accum_t>(grad_out[base + f]);
                    accum_t u_val{};
                    accum_t e_val{};
                    if constexpr (BOps::GRAD_USES_OPERANDS) {
                        u_val = static_cast<accum_t>(lhs[static_cast<size_t>(u) * d + f]);
                        e_val = static_cast<accum_t>(rhs[static_cast<size_t>(eid) * d + f]);
                    }
                    grad_rhs[static_cast<size_t>(eid) * d + f] = BOps::grad_rhs(u_val, e_val, g);
                }
            }
        }
    }
}
