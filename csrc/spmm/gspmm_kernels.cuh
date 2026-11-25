#pragma once

#include "common.cuh"
#include "common/gspmm_ops.cuh"
#include "reduction/reduction_aggr_kernels.cuh"

// ---------------------------------------------------------------------------
// The min/max backward of g-SpMM.
//
// Only the winning edge of each output element carried gradient. On a sparse graph the gradient
// walks the *transposed* CSR (rows are sources), so a source's node gradient is a plain sum over its
// out-edges and needs no atomics; nodes are split into light and heavy buckets launched on separate
// streams, a light node being one y-row of a 2-D block (features along x), a heavy node a whole block
// whose y-tiles split its edge list. On a dense graph the launcher scatters over the winning edges
// instead (reduction_aggr_backward_typed).
//
// A broadcast edge operand ([E] or [E, 1]) folds the d features of an edge into one slot. The kernels
// then require blockDim.x == kWarpSize so that a node's (or tile's) feature threads are exactly one
// warp and the fold is a shuffle reduction: the launcher forces that shape.
//
// The forward and the sum backward are the row-shape kernels of reduction/reduction_forward.cuh and
// gspmm_edge_grad.cuh.
// ---------------------------------------------------------------------------

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
//
// With PIPELINE_STAGES > 0 the one dependent gather of the loop, the winner
// arg_eid[v, f] behind v = bwd_edge_idx[b], is prefetched PIPELINE_STAGES edges
// ahead through a per-thread cp.async pipeline into dbuf (this thread's own
// PIPELINE_STAGES index_t of shared memory).  grad_out and the operands are
// only read on a hit, about one edge in avg-degree, and stay synchronous.  A
// lane past d prefetches a clamped, in-bounds address and discards it, so the
// trip count stays warp-uniform for the broadcast shuffles.
template <
    BinaryOp BOp, bool RHS_BROADCAST, FloatingNum cuda_t, typename index_t, FloatingNum grad_rhs_t, FloatingNum accum_t, int PIPELINE_STAGES = 0
>
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
    size_t d,
    index_t *const dbuf = nullptr
) {
    using BOps        = BinaryOps<BOp>;
    const bool active = f < d;

    static_assert(PIPELINE_STAGES >= 0, "pipeline_stages must be >= 0 (0 disables the pipeline)");

    accum_t u_val{};
    if constexpr (BOps::GRAD_USES_OPERANDS) {
        if (active) {
            u_val = static_cast<accum_t>(lhs[u * d + f]);
        }
    }

    accum_t acc{};

    // One backward position: destination v, forward edge e, and whether e is
    // the winner the forward recorded at (v, f).
    auto consume = [&](index_t v, index_t e, bool won) {
        const bool hit = active && won;
        accum_t g{};
        accum_t e_val{};
        if (hit) {
            g = static_cast<accum_t>(grad_out[static_cast<size_t>(v) * d + f]);
            if constexpr (BOps::GRAD_USES_OPERANDS) {
                e_val = static_cast<accum_t>(RHS_BROADCAST ? rhs[static_cast<size_t>(e)] : rhs[static_cast<size_t>(e) * d + f]);
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
    };

    if constexpr (PIPELINE_STAGES > 0) {
        constexpr int NUM_STAGES = PIPELINE_STAGES;
        const size_t f_c         = active ? f : d - 1;  // in-bounds address for an idle lane
        const index_t num_b      = b_end - b_begin;

        index_t v_buf[NUM_STAGES];
        cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

        auto prefetch = [&](index_t it) {
            pipe.producer_acquire();
            if (it < num_b) {
                const index_t v          = bwd_edge_idx[b_begin + it];
                v_buf[it % NUM_STAGES]   = v;
                index_t const *const src = arg_eid + static_cast<size_t>(v) * d + f_c;
                cuda::memcpy_async(dbuf + it % NUM_STAGES, src, cuda::aligned_size_t<sizeof(index_t)>(sizeof(index_t)), pipe);
            }
            pipe.producer_commit();
        };

#pragma unroll
        for (int s = 0; s < NUM_STAGES; ++s) {
            prefetch(static_cast<index_t>(s));
        }

        for (index_t it = 0; it < num_b; ++it) {
            cuda::pipeline_consumer_wait_prior<NUM_STAGES - 1>(pipe);
            const index_t e = bwd_edge_map[b_begin + it];
            consume(v_buf[it % NUM_STAGES], e, dbuf[it % NUM_STAGES] == e);
            pipe.consumer_release();
            prefetch(it + NUM_STAGES);
        }
    } else {
        for (index_t b = b_begin; b < b_end; ++b) {
            const index_t v = bwd_edge_idx[b];
            const index_t e = bwd_edge_map[b];
            const bool won  = active && (arg_eid[static_cast<size_t>(v) * d + f] == e);
            consume(v, e, won);
        }
    }
    return acc;
}

// Dynamic shared memory of the two min/max backward kernels: with the pipeline
// on, STAGES prefetch slots of index_t per thread, else nothing.  The launcher
// sizes its allocation with this same function.
template <typename index_t, int STAGES>
__host__ __device__ inline size_t gspmm_arg_grad_shmem_bytes(size_t threads) {
    return (STAGES > 0) ? ((threads * static_cast<size_t>(STAGES) * sizeof(index_t) + 15) / 16) * 16 : 0;
}

// Light bucket of the min/max backward: block (tile_x, node_y), one source
// per y-row, its whole out-edge list.  Feature chunks are the outer loop, so
// the trip count is uniform across the row's threads.
template <
    BinaryOp BOp, bool RHS_BROADCAST, FloatingNum cuda_t, typename index_t, FloatingNum grad_rhs_t = cuda_t, FloatingNum accum_t = float,
    int PIPELINE_STAGES = 0
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

    extern __shared__ __align__(16) uint8_t arg_light_raw[];
    constexpr size_t NUM_STAGES = PIPELINE_STAGES > 0 ? PIPELINE_STAGES : 1;
    index_t *const dbuf         = reinterpret_cast<index_t *>(arg_light_raw) + (threadIdx.y * blockDim.x + threadIdx.x) * NUM_STAGES;

    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.y + threadIdx.y;
    if (i >= num_light) {
        return;  // a whole y-row leaves together, so the broadcast shuffles stay full-warp
    }
    const index_t u       = light_nodes[i];
    const index_t b_begin = bwd_edge_ptr[u];
    const index_t b_end   = bwd_edge_ptr[u + 1];

    for (size_t f0 = 0; f0 < d; f0 += blockDim.x) {
        const size_t f    = f0 + threadIdx.x;
        const accum_t acc = gspmm_arg_grad_span<BOp, RHS_BROADCAST, cuda_t, index_t, grad_rhs_t, accum_t, PIPELINE_STAGES>(
            static_cast<size_t>(u), b_begin, b_end, f, bwd_edge_idx, bwd_edge_map, arg_eid, grad_out, lhs, rhs, grad_rhs, d, dbuf
        );
        if constexpr (BOps::USE_LHS) {
            if (f < d) {
                grad_lhs[static_cast<size_t>(u) * d + f] = static_cast<cuda_t>(acc);
            }
        }
    }
}

// Heavy bucket of the min/max backward: block (features, tiles); the tiles
// split an out-edge span and their partial node gradients meet in a
// shared-memory tree, as in the forward's heavy kernel.  Shared memory holds
// one float per thread for that tree; the launcher keeps the block within
// kGSpMMArgHeavyMaxThreads and tiles a power of two.
//
// Two launch shapes, told apart by slice_offsets:
//
// - nullptr: block i owns heavy source i whole and writes grad_lhs[u] itself.
// - otherwise a hub's out-edges are cut into chunks of CHUNK_EDGES with a block
//   each, found through the exclusive prefix sum of chunk counts in
//   slice_offsets (shaped like a CSR row pointer, last entry the total), and
//   the blocks grid-stride over the chunks.  A chunk's node gradient goes to
//   its own row of `partials`, which gspmm_heavy_reduce_slices_kernel<SUM>
//   then folds per node in chunk order -- no atomics, so the result does not
//   depend on which chunk finishes first.  The edge gradient needs no fold:
//   every forward edge sits in exactly one chunk.
inline constexpr size_t kGSpMMArgHeavyMaxThreads = 1024;

template <
    BinaryOp BOp, bool RHS_BROADCAST, FloatingNum cuda_t, typename index_t, size_t CHUNK_EDGES, FloatingNum grad_rhs_t = cuda_t,
    FloatingNum accum_t = float, int PIPELINE_STAGES = 0
>
__global__ void gspmm_backward_arg_heavy_kernel(
    index_t const *const __restrict__ heavy_nodes,
    index_t const *const __restrict__ slice_offsets,
    index_t const *const __restrict__ bwd_edge_ptr,
    index_t const *const __restrict__ bwd_edge_idx,
    index_t const *const __restrict__ bwd_edge_map,
    index_t const *const __restrict__ arg_eid,
    cuda_t const *const __restrict__ grad_out,
    cuda_t const *const __restrict__ lhs,
    cuda_t const *const __restrict__ rhs,
    cuda_t *const __restrict__ grad_lhs,
    grad_rhs_t *const __restrict__ grad_rhs,
    accum_t *const __restrict__ partials,
    size_t num_heavy,
    size_t d
) {
    using BOps = BinaryOps<BOp>;
    __shared__ accum_t tree[kGSpMMArgHeavyMaxThreads];

    extern __shared__ __align__(16) uint8_t arg_heavy_raw[];
    constexpr size_t NUM_STAGES = PIPELINE_STAGES > 0 ? PIPELINE_STAGES : 1;
    index_t *const dbuf         = reinterpret_cast<index_t *>(arg_heavy_raw) + (threadIdx.y * blockDim.x + threadIdx.x) * NUM_STAGES;

    const size_t fid     = threadIdx.x;
    const size_t tid     = threadIdx.y;
    const size_t F_BLOCK = blockDim.x;
    const size_t TILES_Y = blockDim.y;

    // Out-edges [c_begin, c_begin + c_len) of heavy source i, split across the
    // tiles; the span's node gradient for feature f leaves through store(f, val).
    auto process = [&](size_t i, size_t c_begin, size_t c_len, auto&& store) {
        const index_t u         = heavy_nodes[i];
        const index_t row_start = bwd_edge_ptr[u];
        const size_t per_tile   = (c_len + TILES_Y - 1) / TILES_Y;
        const size_t t_begin    = min(tid * per_tile, c_len);
        const size_t t_end      = min(t_begin + per_tile, c_len);
        const index_t b_begin   = row_start + static_cast<index_t>(c_begin + t_begin);
        const index_t b_end     = row_start + static_cast<index_t>(c_begin + t_end);

        for (size_t f0 = 0; f0 < d; f0 += F_BLOCK) {
            const size_t f    = f0 + fid;
            const accum_t acc = gspmm_arg_grad_span<BOp, RHS_BROADCAST, cuda_t, index_t, grad_rhs_t, accum_t, PIPELINE_STAGES>(
                static_cast<size_t>(u), b_begin, b_end, f, bwd_edge_idx, bwd_edge_map, arg_eid, grad_out, lhs, rhs, grad_rhs, d, dbuf
            );

            if constexpr (BOps::USE_LHS) {
                tree[tid * F_BLOCK + fid] = acc;
                __syncthreads();
                for (size_t offset = TILES_Y / 2; offset > 0; offset /= 2) {
                    if (tid < offset) {
                        tree[tid * F_BLOCK + fid] += tree[(tid + offset) * F_BLOCK + fid];
                    }
                    __syncthreads();
                }
                if (tid == 0 && f < d) {
                    store(f, tree[fid]);
                }
                __syncthreads();
            }
        }
    };

    if (slice_offsets == nullptr) {
        const index_t u = heavy_nodes[blockIdx.x];
        process(blockIdx.x, 0, static_cast<size_t>(bwd_edge_ptr[u + 1] - bwd_edge_ptr[u]), [&](size_t f, accum_t val) {
            grad_lhs[static_cast<size_t>(u) * d + f] = static_cast<cuda_t>(val);
        });
        return;
    }

    const size_t total_slices = static_cast<size_t>(slice_offsets[num_heavy]);
    for (size_t slice = blockIdx.x; slice < total_slices; slice += gridDim.x) {
        const size_t i       = csr_row_of<index_t>(slice_offsets, num_heavy, slice);
        const size_t chunk   = slice - static_cast<size_t>(slice_offsets[i]);
        const index_t u      = heavy_nodes[i];
        const size_t degree  = static_cast<size_t>(bwd_edge_ptr[u + 1] - bwd_edge_ptr[u]);
        const size_t c_begin = chunk * CHUNK_EDGES;
        process(i, c_begin, min(CHUNK_EDGES, degree - c_begin), [&](size_t f, accum_t val) { partials[slice * d + f] = val; });
    }
}

// Folds each heavy node's slices in index order and writes the node's output row (and, for a
// comparison reducer, the winning edge position): the min/max backward's sliced heavy kernel folds its
// node-gradient partials with the sum reducer.
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
