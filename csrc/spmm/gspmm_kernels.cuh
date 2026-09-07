#pragma once

#include "common.cuh"
#include "common/gspmm_ops.cuh"
#include "reduction/reduction_aggr_kernels.cuh"

// Gradient w.r.t. the edge operand of a sum reduction: every edge contributed
// exactly the destination's grad_out, so this walks the forward CSR and writes
// each edge's gradient once.  No atomics and no float staging buffer -- the
// message is still formed in accum_t, but the single store lands in grad_t
// (the operand dtype), which halves the traffic in fp16 and removes the cast
// pass the Python layer used to run over an [E, d] float32 buffer.
//
// A broadcast edge operand ([E] or [E, 1]) collapses all d features of an edge
// into one slot; the whole block cooperates on that sum so the write stays
// exactly-once, where a per-warp atomicAdd would have forced float again.
template <
    BinaryOp BOp, bool RHS_BROADCAST, FloatingNum cuda_t, typename index_t, FloatingNum grad_t = cuda_t,
    FloatingNum accum_t = float
>
__global__ void gspmm_backward_edge_kernel(
    index_t const *const __restrict__ edge_ptr,
    index_t const *const __restrict__ edge_idx,
    cuda_t const *const __restrict__ grad_out,
    cuda_t const *const __restrict__ lhs,
    cuda_t const *const __restrict__ rhs,
    grad_t *const __restrict__ grad_rhs,
    size_t num_nodes,
    size_t d
) {
    using BOps = BinaryOps<BOp>;

    __shared__ accum_t red_scratch[kMaxWarpsInBlock];

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
                // Every thread of the block must join: the row bounds are
                // block-uniform, so this is reached the same number of times by
                // all of them even when d is smaller than the block.
                const accum_t total = block_reduce_sum(partial, red_scratch);
                if (threadIdx.x == 0) {
                    grad_rhs[static_cast<size_t>(eid)] = static_cast<grad_t>(total);
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
                    grad_rhs[static_cast<size_t>(eid) * d + f] = static_cast<grad_t>(BOps::grad_rhs(u_val, e_val, g));
                }
            }
        }
    }
}
