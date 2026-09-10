#pragma once

#include "common.cuh"
#include "common/gspmm_ops.cuh"
#include "reduction/reduction_aggr_kernels.cuh"

// ---------------------------------------------------------------------------
// Backward kernels.
//
// Both gradients walk a CSR row per node and split the nodes into the same
// light/heavy buckets the forward uses: a light node is one warp-row of a
// 2-D block (features along x, nodes along y), a heavy node owns a whole block
// whose y-tiles split its edge list.  The two buckets are launched on separate
// streams by the host, exactly as in gspmm_forward.
//
// The edge gradient of a sum walks the *forward* CSR (rows are destinations,
// every edge of a row carries the row's grad_out), the gradient of an arg
// reducer walks the *transposed* CSR (rows are sources, so a source's node
// gradient is a plain sum over its out-edges and needs no atomics).
//
// A broadcast edge operand ([E] or [E, 1]) folds the d features of an edge
// into one slot.  Both kernels then require blockDim.x == kWarpSize so that a
// node's (or tile's) feature threads are exactly one warp and the fold is a
// shuffle reduction: the launcher forces that shape.
// ---------------------------------------------------------------------------

// Edge gradient of a sum reduction over edges [e_begin, e_end) of destination
// v, for the feature threads fid, fid + f_stride, ...  Every (e, f) slot is
// written exactly once, in grad_t directly.
//
// Features outer, edges inner: grad_out[v, f] is loaded once per feature and
// the stores of one warp land on d consecutive elements of one edge row.
template <
    BinaryOp BOp, bool RHS_BROADCAST, FloatingNum cuda_t, typename index_t, FloatingNum grad_t, FloatingNum accum_t,
    int PIPELINE_STAGES = 0
>
__device__ __forceinline__ void gspmm_edge_grad_span(
    size_t v,
    index_t e_begin,
    index_t e_end,
    size_t fid,
    size_t f_stride,
    index_t const *const __restrict__ edge_idx,
    cuda_t const *const __restrict__ grad_out,
    cuda_t const *const __restrict__ lhs,
    cuda_t const *const __restrict__ rhs,
    grad_t *const __restrict__ grad_rhs,
    size_t d,
    cuda_t *const dbuf = nullptr
) {
    using BOps        = BinaryOps<BOp>;
    const size_t base = v * d;

    static_assert(PIPELINE_STAGES >= 0, "pipeline_stages must be >= 0 (0 disables the pipeline)");
    constexpr bool USE_PIPELINE = PIPELINE_STAGES > 0 && BOps::GRAD_USES_OPERANDS;
    constexpr size_t NUM_STAGES = PIPELINE_STAGES > 0 ? PIPELINE_STAGES : 1;

    if constexpr (RHS_BROADCAST) {
        // f_stride == kWarpSize and the callers are one full warp.
        for (index_t eid = e_begin; eid < e_end; ++eid) {
            size_t u_base = 0;
            if constexpr (BOps::GRAD_USES_OPERANDS) {
                u_base = static_cast<size_t>(edge_idx[eid]) * d;
            }
            accum_t partial{};
            for (size_t f = fid; f < d; f += kWarpSize) {
                const accum_t g = static_cast<accum_t>(grad_out[base + f]);
                accum_t u_val{};
                accum_t e_val{};
                if constexpr (BOps::GRAD_USES_OPERANDS) {
                    u_val = static_cast<accum_t>(lhs[u_base + f]);
                    e_val = static_cast<accum_t>(rhs[static_cast<size_t>(eid)]);
                }
                partial += BOps::grad_rhs(u_val, e_val, g);
            }
            partial = warp_reduce_sum(partial);
            if (fid == 0) {
                grad_rhs[static_cast<size_t>(eid)] = static_cast<grad_t>(partial);
            }
        }
    } else {
        for (size_t f = fid; f < d; f += f_stride) {
            const accum_t g = static_cast<accum_t>(grad_out[base + f]);

            auto emit = [&](index_t eid, accum_t u_val) {
                const size_t e_off = static_cast<size_t>(eid) * d + f;
                accum_t e_val{};
                if constexpr (BOps::GRAD_USES_OPERANDS) {
                    e_val = static_cast<accum_t>(rhs[e_off]);
                }
                grad_rhs[e_off] = static_cast<grad_t>(BOps::grad_rhs(u_val, e_val, g));
            };

            if constexpr (USE_PIPELINE) {
                pipelined_thread_edge_scan<1, NUM_STAGES, cuda_t, index_t>(
                    e_begin, e_end, edge_idx, lhs, d, f, dbuf,
                    [&](index_t /*src*/, index_t eid, cuda_t const *uslice) { emit(eid, static_cast<accum_t>(uslice[0])); }
                );
            } else {
                for (index_t eid = e_begin; eid < e_end; ++eid) {
                    accum_t u_val{};
                    if constexpr (BOps::GRAD_USES_OPERANDS) {
                        u_val = static_cast<accum_t>(lhs[static_cast<size_t>(edge_idx[eid]) * d + f]);
                    }
                    emit(eid, u_val);
                }
            }
        }
    }
}

template <FloatingNum cuda_t, int STAGES>
__host__ __device__ inline size_t gspmm_edge_grad_shmem_bytes(size_t threads) {
    return (STAGES > 0) ? ((threads * static_cast<size_t>(STAGES) * sizeof(cuda_t) + 15) / 16) * 16 : 0;
}

template <
    BinaryOp BOp, bool RHS_BROADCAST, FloatingNum cuda_t, typename index_t, FloatingNum grad_t = cuda_t, FloatingNum accum_t = float,
    int PIPELINE_STAGES = 0
>
__global__ void gspmm_backward_edge_light_kernel(
    index_t const *const __restrict__ light_nodes,
    index_t const *const __restrict__ edge_ptr,
    index_t const *const __restrict__ edge_idx,
    cuda_t const *const __restrict__ grad_out,
    cuda_t const *const __restrict__ lhs,
    cuda_t const *const __restrict__ rhs,
    grad_t *const __restrict__ grad_rhs,
    size_t num_light,
    size_t d
) {
    extern __shared__ __align__(16) uint8_t edge_light_raw[];
    constexpr size_t NUM_STAGES = PIPELINE_STAGES > 0 ? PIPELINE_STAGES : 1;
    cuda_t *const dbuf = reinterpret_cast<cuda_t *>(edge_light_raw) + (threadIdx.y * blockDim.x + threadIdx.x) * NUM_STAGES;

    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.y + threadIdx.y;
    if (i >= num_light) {
        return;  // a whole y-row leaves together, so the broadcast shuffles below stay full-warp
    }
    const index_t v = light_nodes[i];
    gspmm_edge_grad_span<BOp, RHS_BROADCAST, cuda_t, index_t, grad_t, accum_t, PIPELINE_STAGES>(
        static_cast<size_t>(v), edge_ptr[v], edge_ptr[v + 1], threadIdx.x, blockDim.x, edge_idx, grad_out, lhs, rhs, grad_rhs, d, dbuf
    );
}

template <
    BinaryOp BOp, bool RHS_BROADCAST, FloatingNum cuda_t, typename index_t, size_t CHUNK_EDGES, FloatingNum grad_t = cuda_t,
    FloatingNum accum_t = float, int PIPELINE_STAGES = 0
>
__global__ void gspmm_backward_edge_heavy_kernel(
    index_t const *const __restrict__ heavy_nodes,
    index_t const *const __restrict__ slice_offsets,
    index_t const *const __restrict__ edge_ptr,
    index_t const *const __restrict__ edge_idx,
    cuda_t const *const __restrict__ grad_out,
    cuda_t const *const __restrict__ lhs,
    cuda_t const *const __restrict__ rhs,
    grad_t *const __restrict__ grad_rhs,
    size_t num_heavy,
    size_t d
) {
    extern __shared__ __align__(16) uint8_t edge_heavy_raw[];
    constexpr size_t NUM_STAGES = PIPELINE_STAGES > 0 ? PIPELINE_STAGES : 1;
    cuda_t *const dbuf = reinterpret_cast<cuda_t *>(edge_heavy_raw) + (threadIdx.y * blockDim.x + threadIdx.x) * NUM_STAGES;

    const size_t tiles = blockDim.y;
    const size_t tid   = threadIdx.y;

    // Edges [c_begin, c_begin + c_len) of heavy node i, split across the tiles.
    auto process = [&](size_t i, size_t c_begin, size_t c_len) {
        const index_t v         = heavy_nodes[i];
        const index_t row_start = edge_ptr[v];
        const size_t per_tile   = (c_len + tiles - 1) / tiles;
        const size_t t_begin    = min(tid * per_tile, c_len);
        const size_t t_end      = min(t_begin + per_tile, c_len);
        gspmm_edge_grad_span<BOp, RHS_BROADCAST, cuda_t, index_t, grad_t, accum_t, PIPELINE_STAGES>(
            static_cast<size_t>(v), row_start + static_cast<index_t>(c_begin + t_begin), row_start + static_cast<index_t>(c_begin + t_end),
            threadIdx.x, blockDim.x, edge_idx, grad_out, lhs, rhs, grad_rhs, d, dbuf
        );
    };

    if (slice_offsets == nullptr) {
        const index_t v = heavy_nodes[blockIdx.x];
        process(blockIdx.x, 0, static_cast<size_t>(edge_ptr[v + 1] - edge_ptr[v]));
        return;
    }

    const size_t total_slices = static_cast<size_t>(slice_offsets[num_heavy]);
    for (size_t slice = blockIdx.x; slice < total_slices; slice += gridDim.x) {
        const size_t i       = csr_row_of<index_t>(slice_offsets, num_heavy, slice);
        const size_t chunk   = slice - static_cast<size_t>(slice_offsets[i]);
        const index_t v      = heavy_nodes[i];
        const size_t degree  = static_cast<size_t>(edge_ptr[v + 1] - edge_ptr[v]);
        const size_t c_begin = chunk * CHUNK_EDGES;
        process(i, c_begin, min(CHUNK_EDGES, degree - c_begin));
    }
}

// Gradients of an arg-tracking reducer (min/max) over the backward-CSR edges
// [b_begin, b_end) of source u, for one feature f.  Returns this thread's
// share of grad_lhs[u, f]; the edge gradient is written on the way.
//
// Backward position b is the forward edge e = bwd_edge_map[b] into destination
// v = bwd_edge_idx[b]; that edge carried gradient iff it is the one the forward
// recorded in arg_eid[v, f].  Walking the transpose makes every (u, f) the
// exclusive property of one thread, so the node gradient needs no atomics and
// can leave in the operand dtype, and every forward edge appears exactly once,
// so a full-width edge gradient is written exactly once (in the operand dtype)
// and needs no zero fill.
//
// A broadcast edge gradient folds the features of an edge into one slot.  The
// fold over the 32 features of this call is a warp shuffle -- the callers are
// one full warp, f may be >= d for some lanes, which then only take part in
// the shuffles -- and the fold over the feature chunks is a plain accumulate
// into a float slot that this warp alone owns, so it needs no atomic either,
// only the zero fill the launcher provides.
template <BinaryOp BOp, bool RHS_BROADCAST, FloatingNum cuda_t, typename index_t, FloatingNum grad_rhs_t, FloatingNum accum_t>
__device__ __forceinline__ accum_t gspmm_arg_grad_span(
    size_t u,
    index_t b_begin,
    index_t b_end,
    size_t f,
    index_t const *const __restrict__ bwd_edge_idx,
    index_t const *const __restrict__ bwd_edge_map,
    index_t const *const __restrict__ arg_eid,
    cuda_t const *const __restrict__ grad_out,
    cuda_t const *const __restrict__ lhs,
    cuda_t const *const __restrict__ rhs,
    grad_rhs_t *const __restrict__ grad_rhs,
    size_t d
) {
    using BOps        = BinaryOps<BOp>;
    const bool active = f < d;

    accum_t u_val{};
    if constexpr (BOps::GRAD_USES_OPERANDS) {
        if (active) {
            u_val = static_cast<accum_t>(lhs[u * d + f]);
        }
    }

    accum_t acc{};
    for (index_t b = b_begin; b < b_end; ++b) {
        const index_t v = bwd_edge_idx[b];
        const index_t e = bwd_edge_map[b];

        bool hit = false;
        accum_t g{};
        accum_t e_val{};
        if (active) {
            const size_t at = static_cast<size_t>(v) * d + f;
            hit             = (arg_eid[at] == e);
            if (hit) {
                g = static_cast<accum_t>(grad_out[at]);
                if constexpr (BOps::GRAD_USES_OPERANDS) {
                    e_val = static_cast<accum_t>(RHS_BROADCAST ? rhs[static_cast<size_t>(e)] : rhs[static_cast<size_t>(e) * d + f]);
                }
            }
        }

        if constexpr (BOps::USE_LHS) {
            if (hit) {
                acc += BOps::grad_lhs(u_val, e_val, g);
            }
        }
        if constexpr (BOps::USE_RHS) {
            const accum_t c = hit ? BOps::grad_rhs(u_val, e_val, g) : accum_t{};
            if constexpr (RHS_BROADCAST) {
                const accum_t total = warp_reduce_sum(c);
                if (threadIdx.x == 0) {
                    grad_rhs[static_cast<size_t>(e)] += static_cast<grad_rhs_t>(total);
                }
            } else if (active) {
                grad_rhs[static_cast<size_t>(e) * d + f] = static_cast<grad_rhs_t>(c);
            }
        }
    }
    return acc;
}

// Light bucket of the min/max backward: block (tile_x, node_y), one source
// per y-row, its whole out-edge list.  Feature chunks are the outer loop, so
// the trip count is uniform across the row's threads.
template <
    BinaryOp BOp, bool RHS_BROADCAST, FloatingNum cuda_t, typename index_t, FloatingNum grad_rhs_t = cuda_t, FloatingNum accum_t = float
>
__global__ void gspmm_backward_arg_light_kernel(
    index_t const *const __restrict__ light_nodes,
    index_t const *const __restrict__ bwd_edge_ptr,
    index_t const *const __restrict__ bwd_edge_idx,
    index_t const *const __restrict__ bwd_edge_map,
    index_t const *const __restrict__ arg_eid,
    cuda_t const *const __restrict__ grad_out,
    cuda_t const *const __restrict__ lhs,
    cuda_t const *const __restrict__ rhs,
    cuda_t *const __restrict__ grad_lhs,
    grad_rhs_t *const __restrict__ grad_rhs,
    size_t num_light,
    size_t d
) {
    using BOps = BinaryOps<BOp>;

    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.y + threadIdx.y;
    if (i >= num_light) {
        return;  // a whole y-row leaves together, so the broadcast shuffles stay full-warp
    }
    const index_t u       = light_nodes[i];
    const index_t b_begin = bwd_edge_ptr[u];
    const index_t b_end   = bwd_edge_ptr[u + 1];

    for (size_t f0 = 0; f0 < d; f0 += blockDim.x) {
        const size_t f    = f0 + threadIdx.x;
        const accum_t acc = gspmm_arg_grad_span<BOp, RHS_BROADCAST, cuda_t, index_t, grad_rhs_t, accum_t>(
            static_cast<size_t>(u), b_begin, b_end, f, bwd_edge_idx, bwd_edge_map, arg_eid, grad_out, lhs, rhs, grad_rhs, d
        );
        if constexpr (BOps::USE_LHS) {
            if (f < d) {
                grad_lhs[static_cast<size_t>(u) * d + f] = static_cast<cuda_t>(acc);
            }
        }
    }
}

// Heavy bucket of the min/max backward: one block per source, block
// (features, tiles); the tiles split the out-edge list and their partial node
// gradients meet in a shared-memory tree, as in the forward's heavy kernel.
// Shared memory holds one float per thread; the launcher keeps the block
// within kGSpMMArgHeavyMaxThreads and tiles a power of two.
inline constexpr size_t kGSpMMArgHeavyMaxThreads = 1024;

template <
    BinaryOp BOp, bool RHS_BROADCAST, FloatingNum cuda_t, typename index_t, FloatingNum grad_rhs_t = cuda_t, FloatingNum accum_t = float
>
__global__ void gspmm_backward_arg_heavy_kernel(
    index_t const *const __restrict__ heavy_nodes,
    index_t const *const __restrict__ bwd_edge_ptr,
    index_t const *const __restrict__ bwd_edge_idx,
    index_t const *const __restrict__ bwd_edge_map,
    index_t const *const __restrict__ arg_eid,
    cuda_t const *const __restrict__ grad_out,
    cuda_t const *const __restrict__ lhs,
    cuda_t const *const __restrict__ rhs,
    cuda_t *const __restrict__ grad_lhs,
    grad_rhs_t *const __restrict__ grad_rhs,
    size_t d
) {
    using BOps = BinaryOps<BOp>;
    __shared__ accum_t partials[kGSpMMArgHeavyMaxThreads];

    const index_t u         = heavy_nodes[blockIdx.x];
    const index_t row_start = bwd_edge_ptr[u];
    const index_t row_end   = bwd_edge_ptr[u + 1];
    const size_t degree     = static_cast<size_t>(row_end - row_start);

    const size_t fid     = threadIdx.x;
    const size_t tid     = threadIdx.y;
    const size_t F_BLOCK = blockDim.x;
    const size_t TILES_Y = blockDim.y;

    const size_t per_tile  = (degree + TILES_Y - 1) / TILES_Y;
    const size_t t_begin   = min(tid * per_tile, degree);
    const size_t t_end     = min(t_begin + per_tile, degree);
    const index_t b_begin  = row_start + static_cast<index_t>(t_begin);
    const index_t b_end    = row_start + static_cast<index_t>(t_end);

    for (size_t f0 = 0; f0 < d; f0 += F_BLOCK) {
        const size_t f    = f0 + fid;
        const accum_t acc = gspmm_arg_grad_span<BOp, RHS_BROADCAST, cuda_t, index_t, grad_rhs_t, accum_t>(
            static_cast<size_t>(u), b_begin, b_end, f, bwd_edge_idx, bwd_edge_map, arg_eid, grad_out, lhs, rhs, grad_rhs, d
        );

        if constexpr (BOps::USE_LHS) {
            partials[tid * F_BLOCK + fid] = acc;
            __syncthreads();
            for (size_t offset = TILES_Y / 2; offset > 0; offset /= 2) {
                if (tid < offset) {
                    partials[tid * F_BLOCK + fid] += partials[(tid + offset) * F_BLOCK + fid];
                }
                __syncthreads();
            }
            if (tid == 0 && f < d) {
                grad_lhs[static_cast<size_t>(u) * d + f] = static_cast<cuda_t>(partials[fid]);
            }
            __syncthreads();
        }
    }
}

// Heavy-node forward, sliced by edges.
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
// then folds each node's slices in index order.  A comparison reducer carries
// the winning edge position alongside every partial value, in `partial_args`,
// which is what lets min/max take this path too.
//
// The slice a block owns comes from a binary search over `slice_offsets` (an
// exclusive prefix sum of each heavy node's slice count, shaped exactly like a
// CSR row pointer), whose last entry is the total slice count.  Reading the
// bound on the device is what lets the grid be sized for occupancy instead of
// for a count only the device knows.
//
// TW is 1: g-SpMM accepts any feature width, so it never takes the vectorized
// TileOps path (see kGSpMMVectorize).

// Dynamic shared-memory layout: `slots` accumulator floats, then -- only for a
// reducer that reports an argument -- `slots` index_t, then -- only with the
// pipeline on -- STAGES prefetch slots per thread.  Each region is padded to 16
// bytes so the next one stays aligned for any (tiles_y, features_per_block).
// The launcher sizes its allocation with this same function, so the two cannot
// disagree.
template <FloatingNum cuda_t, typename index_t, bool TRACKS_ARG, int STAGES>
__host__ __device__ inline size_t gspmm_sliced_shmem_bytes(size_t slots) {
    size_t bytes = ((slots * sizeof(float) + 15) / 16) * 16;
    if constexpr (TRACKS_ARG) {
        bytes += ((slots * sizeof(index_t) + 15) / 16) * 16;
    }
    if constexpr (STAGES > 0) {
        bytes += ((slots * static_cast<size_t>(STAGES) * sizeof(cuda_t) + 15) / 16) * 16;
    }
    return bytes;
}

template <
    FloatingNum cuda_t, typename index_t, ReductionOp ROp, BinaryOp BOp, bool RHS_BROADCAST, bool EDGE_MAP, size_t SLICE_EDGES,
    int PIPELINE_STAGES = 0, FloatingNum accum_t = float
>
__global__ void gspmm_heavy_sliced_kernel(
    index_t const *const __restrict__ heavy_nodes,
    index_t const *const __restrict__ slice_offsets,
    index_t const *const __restrict__ edge_ptr,
    index_t const *const __restrict__ edge_idx,
    cuda_t const *const __restrict__ lhs,
    cuda_t const *const __restrict__ rhs,
    index_t const *const __restrict__ edge_map,
    accum_t *const __restrict__ partials,
    index_t *const __restrict__ partial_args,
    size_t num_heavy,
    size_t d
) {
    using BOps     = BinaryOps<BOp>;
    using ROps     = ReductionOps<ROp>;
    using Sentinel = IndexSentinel<index_t>;
    // The reducer's own accumulate type, not accum_t: a comparison reducer
    // compares operand-dtype values, and comparing the unrounded float message
    // instead would pick a different winner among fp16 ties than the light and
    // unsliced heavy kernels do.  The partial that leaves the block is float
    // either way, which is order-preserving for both.
    using acc_t = typename ROps::template AccumType<cuda_t, accum_t>;

    constexpr bool TRACKS_ARG   = ROps::TRACKS_ARG;
    constexpr bool USE_PIPELINE = PIPELINE_STAGES > 0 && BOps::USE_LHS;
    constexpr int NUM_STAGES    = PIPELINE_STAGES > 0 ? PIPELINE_STAGES : 1;

    extern __shared__ __align__(16) uint8_t shared_raw[];

    const size_t fid          = threadIdx.x;  // feature
    const size_t tid          = threadIdx.y;  // edge tile within the slice
    const size_t f_block      = blockDim.x;
    const size_t tiles        = blockDim.y;
    const size_t slots        = f_block * tiles;
    const size_t slot         = tid * f_block + fid;
    const size_t total_slices = static_cast<size_t>(slice_offsets[num_heavy]);

    accum_t *const shmem_val = reinterpret_cast<accum_t *>(shared_raw);
    index_t *const shmem_arg =
        reinterpret_cast<index_t *>(shared_raw + gspmm_sliced_shmem_bytes<cuda_t, index_t, false, 0>(slots));
    cuda_t *const my_pipe =
        reinterpret_cast<cuda_t *>(shared_raw + gspmm_sliced_shmem_bytes<cuda_t, index_t, TRACKS_ARG, 0>(slots)) + slot * NUM_STAGES;

    const acc_t identity_val = static_cast<acc_t>(ROps::IDENTITY);

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
            const size_t f      = it * f_block + fid;
            const bool in_range = f < d;

            acc_t acc        = identity_val;
            index_t best_arg = Sentinel::INVALID;

            auto visit = [&](index_t /*src*/, index_t eid, cuda_t const *uslice) {
                accum_t msg[1];
                const index_t e_pos = aggr_edge_data_pos<EDGE_MAP, index_t>(eid, edge_map);
                aggr_edge_message<BOp, RHS_BROADCAST, 1, cuda_t, index_t, accum_t>(uslice, rhs, e_pos, f, d, msg);
                bool upgrade_index = false;
                acc                = ROps::reduce(static_cast<acc_t>(msg[0]), acc, upgrade_index);
                if constexpr (TRACKS_ARG) {
                    if (upgrade_index) {
                        best_arg = eid;
                    }
                }
            };

            if (in_range) {
                if constexpr (USE_PIPELINE) {
                    pipelined_thread_edge_scan<1, static_cast<size_t>(NUM_STAGES), cuda_t, index_t>(
                        begin, end, edge_idx, lhs, d, f, my_pipe, visit
                    );
                } else {
                    for (index_t eid = begin; eid < end; ++eid) {
                        cuda_t u_val{};
                        cuda_t const *uslice = nullptr;
                        if constexpr (BOps::USE_LHS) {
                            u_val  = lhs[static_cast<size_t>(edge_idx[eid]) * d + f];
                            uslice = &u_val;
                        }
                        visit(index_t{}, eid, uslice);
                    }
                }
            }

            shmem_val[slot] = static_cast<accum_t>(acc);
            if constexpr (TRACKS_ARG) {
                shmem_arg[slot] = best_arg;
            }
            __syncthreads();

            for (size_t offset = tiles / 2; offset > 0; offset /= 2) {
                if (tid < offset) {
                    const size_t a = slot;
                    const size_t b = slot + offset * f_block;
                    if constexpr (TRACKS_ARG) {
                        // Tie-break on the smaller edge position, so the arg a
                        // slice reports does not depend on how its edges split
                        // across tiles.
                        const accum_t val_b = shmem_val[b];
                        const index_t arg_b = shmem_arg[b];
                        bool take_b         = false;
                        ROps::reduce(val_b, shmem_val[a], take_b);
                        if (take_b ||
                            (val_b == shmem_val[a] && Sentinel::is_valid(arg_b) &&
                             (!Sentinel::is_valid(shmem_arg[a]) || arg_b < shmem_arg[a]))) {
                            shmem_val[a] = val_b;
                            shmem_arg[a] = arg_b;
                        }
                    } else {
                        shmem_val[a] += shmem_val[b];
                    }
                }
                __syncthreads();
            }

            if (tid == 0 && in_range) {
                partials[slice * d + f] = shmem_val[fid];
                if constexpr (TRACKS_ARG) {
                    partial_args[slice * d + f] = shmem_arg[fid];
                }
            }
            __syncthreads();
        }
    }
}

// Folds each heavy node's slices in index order and writes the node's output
// row (and, for a comparison reducer, the winning edge position).
// Deterministic by construction, which an atomic fold would not be.
template <FloatingNum cuda_t, typename index_t, ReductionOp ROp, FloatingNum accum_t = float>
__global__ void gspmm_heavy_reduce_slices_kernel(
    index_t const *const __restrict__ heavy_nodes,
    index_t const *const __restrict__ slice_offsets,
    accum_t const *const __restrict__ partials,
    index_t const *const __restrict__ partial_args,
    cuda_t *const __restrict__ out,
    index_t *const __restrict__ arg_idx,
    size_t num_heavy,
    size_t d
) {
    using ROps     = ReductionOps<ROp>;
    using Sentinel = IndexSentinel<index_t>;

    const size_t total  = num_heavy * d;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;

    for (size_t k = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; k < total; k += stride) {
        const size_t i = k / d;
        const size_t f = k - i * d;

        const size_t from = static_cast<size_t>(slice_offsets[i]);
        const size_t to   = static_cast<size_t>(slice_offsets[i + 1]);

        accum_t acc = static_cast<accum_t>(ROps::IDENTITY);
        index_t arg = Sentinel::INVALID;
        for (size_t slice = from; slice < to; ++slice) {
            const accum_t val = partials[slice * d + f];
            if constexpr (ROps::TRACKS_ARG) {
                const index_t cand  = partial_args[slice * d + f];
                bool upgrade_index  = false;
                const accum_t taken = ROps::reduce(val, acc, upgrade_index);
                // Slices are visited in ascending edge order, so a tie keeps
                // the arg already held -- the smaller edge position.
                if (upgrade_index && Sentinel::is_valid(cand)) {
                    arg = cand;
                } else if (!Sentinel::is_valid(arg) && Sentinel::is_valid(cand) && val == taken) {
                    arg = cand;
                }
                acc = taken;
            } else {
                bool upgrade_index = false;
                acc                = ROps::reduce(val, acc, upgrade_index);
            }
        }

        const size_t at = static_cast<size_t>(heavy_nodes[i]) * d + f;
        if constexpr (ROps::TRACKS_ARG) {
            // A node with no in-edges keeps the identity, which is not a
            // meaningful feature value -- report zero and let the invalid arg
            // suppress its gradient, exactly as the unsliced kernels do.
            out[at]     = Sentinel::is_valid(arg) ? static_cast<cuda_t>(acc) : cuda_t{};
            arg_idx[at] = arg;
        } else {
            out[at] = static_cast<cuda_t>(acc);
        }
    }
}
