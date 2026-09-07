#pragma once

#include "common.cuh"
#include "common/gspmm_ops.cuh"
#include "reduction/reduction_aggr_kernels.cuh"

// Gradient w.r.t. the edge operand of a sum reduction: every edge contributed
// exactly the destination's grad_out, so each edge's gradient is independent of
// every other's.  Nothing is reduced across edges and nothing accumulates, so
// this parallelizes over *edges* -- one warp each -- rather than over
// destination nodes.
//
// Walking rows instead would hand one block a heavy node's entire neighbor
// list: on a graph whose top in-degree is in the tens of thousands that single
// block sets the runtime while the rest of the GPU idles.
//
// A warp takes a contiguous run of EDGES_PER_WARP edges and locates the
// destination of the first one by binary search, then advances the row pointer
// as it walks the run -- both the run and the rows it spans are monotone.
// Searching per edge instead costs log(N) *dependent* loads for every edge,
// which on a uniform graph outweighs the whole per-edge computation; amortized
// over a run it disappears, and a row-per-edge array (another E indices to
// build and keep) is not needed either.
//
// The write lands in grad_t (the operand dtype) directly: it is exactly-once,
// so there is no atomic to protect and no float staging buffer for the caller
// to cast back.  A broadcast operand ([E] or [E, 1]) folds all d features of an
// edge into one slot, which is exactly the warp's own span, so its reduction
// stays inside the warp and still ends in a single store.
template <
    BinaryOp BOp, bool RHS_BROADCAST, FloatingNum cuda_t, typename index_t, FloatingNum grad_t = cuda_t,
    FloatingNum accum_t = float, size_t EDGES_PER_WARP = 32
>
__global__ void gspmm_backward_edge_kernel(
    index_t const *const __restrict__ edge_ptr,
    index_t const *const __restrict__ edge_idx,
    cuda_t const *const __restrict__ grad_out,
    cuda_t const *const __restrict__ lhs,
    cuda_t const *const __restrict__ rhs,
    grad_t *const __restrict__ grad_rhs,
    size_t num_nodes,
    size_t num_edges,
    size_t d
) {
    using BOps = BinaryOps<BOp>;

    const size_t lane        = threadIdx.x % kWarpSize;
    const size_t global_warp = (static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x) / kWarpSize;
    const size_t warp_stride = (static_cast<size_t>(gridDim.x) * blockDim.x) / kWarpSize;

    const size_t num_runs = (num_edges + EDGES_PER_WARP - 1) / EDGES_PER_WARP;

    for (size_t run = global_warp; run < num_runs; run += warp_stride) {
        const size_t e_begin = run * EDGES_PER_WARP;
        const size_t e_end   = min(e_begin + EDGES_PER_WARP, num_edges);

        // Uniform across the warp, so the search's loads coalesce into one
        // transaction per level, and it happens once per run.
        size_t v = csr_row_of<index_t>(edge_ptr, num_nodes, e_begin);

        for (size_t eid = e_begin; eid < e_end; ++eid) {
            // Monotone: the run walks edges in order, so the row only moves
            // forward.  The loop also steps over rows with no edges at all.
            while (static_cast<size_t>(edge_ptr[v + 1]) <= eid) {
                ++v;
            }
            const size_t base = v * d;

            size_t u_base = 0;
            if constexpr (BOps::GRAD_USES_OPERANDS) {
                u_base = static_cast<size_t>(edge_idx[eid]) * d;
            }

            if constexpr (RHS_BROADCAST) {
                accum_t partial{};
                for (size_t f = lane; f < d; f += kWarpSize) {
                    const accum_t g = static_cast<accum_t>(grad_out[base + f]);
                    accum_t u_val{};
                    accum_t e_val{};
                    if constexpr (BOps::GRAD_USES_OPERANDS) {
                        u_val = static_cast<accum_t>(lhs[u_base + f]);
                        e_val = static_cast<accum_t>(rhs[eid]);
                    }
                    partial += BOps::grad_rhs(u_val, e_val, g);
                }
                partial = warp_reduce_sum(partial);
                if (lane == 0) {
                    grad_rhs[eid] = static_cast<grad_t>(partial);
                }
            } else {
                for (size_t f = lane; f < d; f += kWarpSize) {
                    const accum_t g = static_cast<accum_t>(grad_out[base + f]);
                    accum_t u_val{};
                    accum_t e_val{};
                    if constexpr (BOps::GRAD_USES_OPERANDS) {
                        u_val = static_cast<accum_t>(lhs[u_base + f]);
                        e_val = static_cast<accum_t>(rhs[eid * d + f]);
                    }
                    grad_rhs[eid * d + f] = static_cast<grad_t>(BOps::grad_rhs(u_val, e_val, g));
                }
            }
        }
    }
}
