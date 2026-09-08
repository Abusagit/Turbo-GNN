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

// Backward of a comparison reducer: only the winning edge of each output
// element contributed anything, so both gradients are one scatter over the
// saved arg indices.
//
// reduction_aggr_backward_typed gives one block to a node, which leaves that
// block d elements of work whatever the node's degree is: at d = 32 with a
// 256-thread block seven eighths of the threads exit immediately, and the grid
// is as long as the node count.  Degree does not enter into it -- the work is
// uniform per output *element*, so bucketing by degree has nothing to balance
// here.  What does help is dropping the node-to-block mapping: this kernel
// grid-strides over the flat [N, d] output, so every block is full and the grid
// is sized for the device rather than for the graph.
template <
    BinaryOp BOp, bool RHS_BROADCAST, FloatingNum cuda_t, typename index_t, FloatingNum grad_t = float,
    FloatingNum grad_rhs_t = cuda_t, FloatingNum accum_t = float
>
__global__ void gspmm_backward_arg_kernel(
    cuda_t const *const __restrict__ grad_out,
    index_t const *const __restrict__ arg_idx,
    index_t const *const __restrict__ edge_idx,
    cuda_t const *const __restrict__ lhs,
    cuda_t const *const __restrict__ rhs,
    grad_t *const __restrict__ grad_lhs,
    grad_rhs_t *const __restrict__ grad_rhs,
    size_t num_nodes,
    size_t d
) {
    using BOps     = BinaryOps<BOp>;
    using Sentinel = IndexSentinel<index_t>;

    const size_t total  = num_nodes * d;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;

    for (size_t k = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; k < total; k += stride) {
        const index_t arg = arg_idx[k];
        if (!Sentinel::is_valid(arg)) {
            continue;  // node with no in-edges: nothing reached it, nothing flows back
        }

        const size_t f     = k - (k / d) * d;
        const accum_t g    = static_cast<accum_t>(grad_out[k]);
        const size_t e_off = RHS_BROADCAST ? static_cast<size_t>(arg) : static_cast<size_t>(arg) * d + f;

        // copy_e never reads the source node, and only mul and div
        // differentiate to something that reads the operands at all.
        size_t u_off = 0;
        if constexpr (BOps::USE_LHS) {
            u_off = static_cast<size_t>(edge_idx[static_cast<size_t>(arg)]) * d + f;
        }
        accum_t u_val{};
        accum_t e_val{};
        if constexpr (BOps::GRAD_USES_OPERANDS) {
            u_val = static_cast<accum_t>(lhs[u_off]);
            e_val = static_cast<accum_t>(rhs[e_off]);
        }

        if constexpr (BOps::USE_LHS) {
            atomicAdd(&grad_lhs[u_off], static_cast<grad_t>(BOps::grad_lhs(u_val, e_val, g)));
        }
        if constexpr (BOps::USE_RHS) {
            const grad_rhs_t contribution = static_cast<grad_rhs_t>(BOps::grad_rhs(u_val, e_val, g));
            if constexpr (RHS_BROADCAST) {
                // One slot per edge, shared by all d features of it.
                atomicAdd(&grad_rhs[e_off], contribution);
            } else {
                // (arg, f) is unique across the whole output, so nothing else
                // ever touches this slot.
                grad_rhs[e_off] = contribution;
            }
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
