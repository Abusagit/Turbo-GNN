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

// Heavy-node forward for an accumulating reducer, sliced by edges.
//
// The unsliced heavy kernel gives one block to a node and splits its neighbors
// across blockDim.y.  That is fine until one node holds tens of thousands of
// edges: the block that draws it then runs long after every other block in the
// launch has retired, and the partition meant to balance the work is what
// creates the imbalance.  Here a node's neighbors are cut into slices of
// SLICE_EDGES and every slice gets its own block.
//
// Each block writes its own slot in `partials` rather than folding into a
// shared accumulator, so no atomics appear and the result does not depend on
// the order the slices happen to finish -- gspmm_heavy_reduce_slices_kernel
// then sums each node's slices in index order.  A comparison reducer cannot
// use this: it would have to carry the winning edge index alongside each
// partial value, which is what the packed uint64 path does for min/max.
//
// The slice a block owns comes from a binary search over `slice_offsets` (an
// exclusive prefix sum of each heavy node's slice count, shaped exactly like a
// CSR row pointer), whose last entry is the total slice count.  Reading the
// bound on the device is what lets the grid be sized for occupancy instead of
// for a count only the device knows.
//
// TW is 1: g-SpMM accepts any feature width, so it never takes the vectorized
// TileOps path (see kGSpMMVectorize).
template <
    FloatingNum cuda_t, typename index_t, BinaryOp BOp, bool RHS_BROADCAST, bool EDGE_MAP, size_t SLICE_EDGES,
    FloatingNum accum_t = float
>
__global__ void gspmm_heavy_sum_sliced_kernel(
    index_t const *const __restrict__ heavy_nodes,
    index_t const *const __restrict__ slice_offsets,
    index_t const *const __restrict__ edge_ptr,
    index_t const *const __restrict__ edge_idx,
    cuda_t const *const __restrict__ lhs,
    cuda_t const *const __restrict__ rhs,
    index_t const *const __restrict__ edge_map,
    accum_t *const __restrict__ partials,
    size_t num_heavy,
    size_t d
) {
    using BOps = BinaryOps<BOp>;

    extern __shared__ __align__(16) uint8_t shared_raw[];
    accum_t *const shmem = reinterpret_cast<accum_t *>(shared_raw);

    const size_t fid          = threadIdx.x;  // feature
    const size_t tid          = threadIdx.y;  // edge tile within the slice
    const size_t f_block      = blockDim.x;
    const size_t tiles        = blockDim.y;
    const size_t total_slices = static_cast<size_t>(slice_offsets[num_heavy]);

    for (size_t slice = blockIdx.x; slice < total_slices; slice += gridDim.x) {
        const size_t i       = csr_row_of<index_t>(slice_offsets, num_heavy, slice);
        const size_t within  = slice - static_cast<size_t>(slice_offsets[i]);
        const index_t v      = heavy_nodes[i];
        const index_t r_end  = edge_ptr[v + 1];
        const index_t s_from = edge_ptr[v] + static_cast<index_t>(within * SLICE_EDGES);
        const index_t s_to   = (s_from + static_cast<index_t>(SLICE_EDGES) < r_end) ? s_from + static_cast<index_t>(SLICE_EDGES) : r_end;

        // Split the slice across blockDim.y the same way the unsliced kernel
        // splits a whole row.
        const size_t span     = static_cast<size_t>(s_to - s_from);
        const size_t per_tile = (span + tiles - 1) / tiles;
        const index_t begin   = s_from + static_cast<index_t>(tid * per_tile);
        const index_t end_c   = begin + static_cast<index_t>(per_tile);
        const index_t end     = (end_c < s_to) ? end_c : s_to;

        // Trip count over the feature tiles is deliberately block-uniform, and
        // threads past d idle instead of exiting the loop early: the tree
        // reduction below is a barrier, and a barrier reached by only some of
        // the block is undefined behaviour.  Striding `f < d` directly would
        // do exactly that whenever d is not a multiple of blockDim.x.
        const size_t f_iters = (d + f_block - 1) / f_block;

        for (size_t it = 0; it < f_iters; ++it) {
            const size_t f     = it * f_block + fid;
            const bool in_range = f < d;

            accum_t acc{};
            if (in_range) {
                for (index_t eid = begin; eid < end; ++eid) {
                    cuda_t u_val{};
                    if constexpr (BOps::USE_LHS) {
                        u_val = lhs[static_cast<size_t>(edge_idx[eid]) * d + f];
                    }
                    accum_t msg[1];
                    const index_t e_pos = aggr_edge_data_pos<EDGE_MAP, index_t>(eid, edge_map);
                    aggr_edge_message<BOp, RHS_BROADCAST, 1, cuda_t, index_t, accum_t>(&u_val, rhs, e_pos, f, d, msg);
                    acc += msg[0];
                }
            }

            shmem[tid * f_block + fid] = acc;
            __syncthreads();

            for (size_t offset = tiles / 2; offset > 0; offset /= 2) {
                if (tid < offset) {
                    shmem[tid * f_block + fid] += shmem[(tid + offset) * f_block + fid];
                }
                __syncthreads();
            }

            if (tid == 0 && in_range) {
                partials[slice * d + f] = shmem[fid];
            }
            __syncthreads();
        }
    }
}

// Sums each heavy node's slices in index order and writes the node's output
// row.  Deterministic by construction, which an atomic fold would not be.
template <FloatingNum cuda_t, typename index_t, FloatingNum accum_t = float>
__global__ void gspmm_heavy_reduce_slices_kernel(
    index_t const *const __restrict__ heavy_nodes,
    index_t const *const __restrict__ slice_offsets,
    accum_t const *const __restrict__ partials,
    cuda_t *const __restrict__ out,
    size_t num_heavy,
    size_t d
) {
    const size_t total  = num_heavy * d;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;

    for (size_t k = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; k < total; k += stride) {
        const size_t i = k / d;
        const size_t f = k - i * d;

        const size_t from = static_cast<size_t>(slice_offsets[i]);
        const size_t to   = static_cast<size_t>(slice_offsets[i + 1]);

        accum_t acc{};
        for (size_t slice = from; slice < to; ++slice) {
            acc += partials[slice * d + f];
        }
        out[static_cast<size_t>(heavy_nodes[i]) * d + f] = static_cast<cuda_t>(acc);
    }
}
