#pragma once

#include <cstdint>

#include "common.cuh"
#include "common/gspmm_ops.cuh"

// =============================================================================
// Neighbourhood-reduction kernels, shared by reduction_aggr and g-SpMM.
//
// The kernels moved here out of reduction_aggr.cu unchanged, then grew one axis:
// the *edge operand*.  What they compute is
//
//     out[v, f] = reduce_{eid in row(v)} op( X[edge_idx[eid], f], rhs[eid, f] )
//
// and the four trailing template parameters select the degenerate cases:
//
//   BOp            which binary op combines node and edge data.  The default
//                  BinaryOp::COPY_U ignores `rhs` entirely -- that is plain
//                  reduction_aggr, and `rhs` is then passed as nullptr.
//   RHS_BROADCAST  edge data is one scalar per edge (`rhs[eid]`) rather than one
//                  value per (edge, feature) (`rhs[eid * d + f]`).
//   ARG_IS_EDGE    record the winning *edge position* in arg_idx instead of the
//                  winning source node.  g-SpMM needs the edge, because both of
//                  its gradients have to reach rhs[eid]; the source node is
//                  recoverable as edge_idx[eid] while the reverse is not.
//   VECTORIZE      false forces TW = 1.  TileOps::read casts to a 16-byte
//                  aligned vector type, so the vectorized path requires
//                  d * sizeof(cuda_t) % 16 == 0 -- g-SpMM accepts any d and
//                  therefore opts out.
//
// Every branch the axis adds is `if constexpr`, so the COPY_U instantiations
// generate the code they did before it existed.
// =============================================================================

// Per-thread pipelined scan over edges [start, end): each edge contributes the
// TW-wide slice X[edge_idx[eid]*d + base_f : +TW]. Per-thread (not per-warp)
// parallelism, so each thread prefetches its own <=16B slice.
//
// NOTE: benchmarks show PIPELINE_STAGES>0 regresses min/max_aggr -- visit() is
// a few compares, too little compute to hide the cp.async latency, while the
// pipeline serializes edges the compiler could otherwise overlap. Keep at 0.
//
// visit(src, val): val is the prefetched slice, valid only inside the call.
// dbuf: this thread's scratch, NUM_STAGES * TW elements.
template <size_t TW, size_t NUM_STAGES, FloatingNum cuda_t, typename index_t, typename VisitFn>
__device__ __forceinline__ void pipelined_thread_edge_scan(
    index_t start, index_t end, index_t const *__restrict__ edge_idx, cuda_t const *__restrict__ X, size_t d, size_t base_f, cuda_t *dbuf,
    VisitFn&& visit
) {
    if (end <= start) {
        return;
    }
    const index_t num_edges = end - start;

    cuda_t *slots[NUM_STAGES];
#pragma unroll
    for (size_t s = 0; s < NUM_STAGES; ++s) {
        slots[s] = dbuf + s * TW;
    }
    index_t src_buf[NUM_STAGES];

    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

    auto prefetch = [&pipe, num_edges, start, edge_idx, &src_buf, X, d, base_f, &slots](index_t it) {
        pipe.producer_acquire();
        if (it < num_edges) {
            const index_t eid        = start + it;
            const index_t src        = edge_idx[eid];
            src_buf[it % NUM_STAGES] = src;
            const cuda_t *src_ptr    = X + static_cast<size_t>(src) * d + base_f;
            async_copy_slice_thread<TW, cuda_t>(slots[it % NUM_STAGES], src_ptr, pipe);
        }
        pipe.producer_commit();
    };

#pragma unroll
    for (size_t s = 0; s < NUM_STAGES; ++s) {
        prefetch(s);
    }

    for (index_t it = 0; it < num_edges; ++it) {
        cuda::pipeline_consumer_wait_prior<NUM_STAGES - 1>(pipe);
        visit(src_buf[it % NUM_STAGES], slots[it % NUM_STAGES]);
        pipe.consumer_release();
        prefetch(it + NUM_STAGES);
    }
}

template <bool VECTORIZE, FloatingNum cuda_t>
inline constexpr size_t aggr_tile_width = VECTORIZE ? VecFloat<1, cuda_t>::max_vec_size_bytes / sizeof(cuda_t) : 1;

template <BinaryOp BOp, bool RHS_BROADCAST, size_t TW, FloatingNum cuda_t, typename index_t, FloatingNum accum_t>
__device__ __forceinline__ void aggr_edge_message(
    cuda_t const *const __restrict__ uslice, cuda_t const *const __restrict__ rhs, index_t eid, size_t fv, size_t d, accum_t (&msg)[TW]
) {
    using BOps = BinaryOps<BOp>;
    using Tile = TileOps<TW, cuda_t>;

    accum_t e_scalar{};
    typename Tile::vec_t e_slice{};
    if constexpr (BOps::USE_RHS) {
        if constexpr (RHS_BROADCAST) {
            e_scalar = static_cast<accum_t>(rhs[static_cast<size_t>(eid)]);
        } else {
            e_slice = Tile::read(&rhs[static_cast<size_t>(eid) * d], fv);
        }
    }

    for (size_t e = 0; e < TW; ++e) {
        accum_t u_val{};
        accum_t e_val{};
        if constexpr (BOps::USE_LHS) {
            u_val = static_cast<accum_t>(uslice[e]);
        }
        if constexpr (BOps::USE_RHS) {
            e_val = RHS_BROADCAST ? e_scalar : static_cast<accum_t>(e_slice[e]);
        }
        msg[e] = BOps::call(u_val, e_val);
    }
}

// PIPELINE_STAGES>0 regresses this kernel, see pipelined_thread_edge_scan.
template <
    size_t WARPS_PER_BLOCK, FloatingNum cuda_t, ReductionOp Op, typename index_t, FloatingNum accum_t = float, int PIPELINE_STAGES = 0,
    BinaryOp BOp = BinaryOp::COPY_U, bool RHS_BROADCAST = false, bool ARG_IS_EDGE = false, bool VECTORIZE = true
>
__global__ void __launch_bounds__(WARPS_PER_BLOCK *kWarpSize) reduction_aggr_forward_light_kernel_1d(
    index_t const *const __restrict__ light_nodes_indices,
    index_t const *const __restrict__ edge_ptr,
    index_t const *const __restrict__ edge_idx,
    cuda_t const *const __restrict__ X,
    cuda_t *const __restrict__ out,
    index_t *const __restrict__ arg_idx,
    size_t d,
    size_t num_light,
    cuda_t const *const __restrict__ rhs = nullptr
) {
    using ROps     = ReductionOps<Op>;
    using BOps     = BinaryOps<BOp>;
    using Sentinel = IndexSentinel<index_t>;
    using acc_t    = typename ROps::template AccumType<cuda_t, accum_t>;
    // constexpr size_t TW = (sizeof(cuda_t) <= 2) ? 2 : 1;
    constexpr size_t TW = aggr_tile_width<VECTORIZE, cuda_t>;
    using Tile          = TileOps<TW, cuda_t>;

    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.y + threadIdx.y;
    if (i >= num_light) {
        return;
    }
    const index_t v = light_nodes_indices[i];

    const index_t row_start = edge_ptr[v];
    const index_t row_end   = edge_ptr[v + 1];

    const size_t tid      = threadIdx.x;
    const size_t tile_dim = blockDim.x;

    const size_t node_stride = static_cast<size_t>(v) * d;

    const acc_t identity_val = static_cast<acc_t>(ROps::IDENTITY);
    constexpr cuda_t zero_val{};

    const size_t d_vec = d / TW;

    static_assert(PIPELINE_STAGES >= 0, "pipeline_stages must be >= 0 (0 disables the pipeline)");
    constexpr bool USE_PIPELINE = PIPELINE_STAGES > 0;
    constexpr int NUM_STAGES    = PIPELINE_STAGES;

    extern __shared__ __align__(16) uint8_t sh_raw[];
    cuda_t *val_dbuf = reinterpret_cast<cuda_t *>(sh_raw);  // only meaningful when USE_PIPELINE
    // Flat thread id, not threadIdx.x: this kernel is 2-D blocked (x = feature
    // tile, y = node), so indexing the scratch by threadIdx.x alone would hand
    // the same slots to every node in the block.
    const size_t flat_tid = threadIdx.y * blockDim.x + threadIdx.x;
    cuda_t *my_dbuf       = val_dbuf + flat_tid * NUM_STAGES * TW;

    for (size_t fv = tid; fv < d_vec; fv += tile_dim) {
        const size_t base_f = fv * TW;

        acc_t best_vals[TW];
        index_t best_args[TW];
#pragma unroll
        for (size_t e = 0; e < TW; ++e) {
            best_vals[e] = identity_val;
            best_args[e] = Sentinel::INVALID;
        }

        auto visit = [&](index_t src, index_t eid, cuda_t const *uslice) {
            accum_t msg[TW];
            aggr_edge_message<BOp, RHS_BROADCAST, TW, cuda_t, index_t, accum_t>(uslice, rhs, eid, fv, d, msg);
#pragma unroll
            for (size_t e = 0; e < TW; ++e) {
                const acc_t v_e    = static_cast<acc_t>(msg[e]);
                bool upgrade_index = false;
                best_vals[e]       = ROps::reduce(v_e, best_vals[e], upgrade_index);
                if constexpr (ROps::TRACKS_ARG) {
                    if (upgrade_index) {
                        best_args[e] = ARG_IS_EDGE ? eid : src;
                    }
                }
            }
        };

        // The pipeline prefetches X only, so it stays available exactly where
        // reduction_aggr uses it: the node-only operation reporting node args.
        if constexpr (USE_PIPELINE && !BOps::USE_RHS && !ARG_IS_EDGE) {
            auto visit_node = [&visit](index_t src, cuda_t const *val) { visit(src, index_t{}, val); };
            pipelined_thread_edge_scan<TW, NUM_STAGES, cuda_t, index_t>(row_start, row_end, edge_idx, X, d, base_f, my_dbuf, visit_node);
        } else {
            for (index_t eid = row_start; eid < row_end; ++eid) {
                if constexpr (BOps::USE_LHS) {
                    const index_t src              = edge_idx[eid];
                    const typename Tile::vec_t val = Tile::read(&X[static_cast<size_t>(src) * d], fv);
                    visit(src, eid, val.data);
                } else if constexpr (ARG_IS_EDGE) {
                    // copy_e reporting edge args touches neither X nor edge_idx
                    visit(index_t{}, eid, nullptr);
                } else {
                    visit(edge_idx[eid], eid, nullptr);
                }
            }
        }

        cuda_t result[TW];
#pragma unroll
        for (size_t e = 0; e < TW; ++e) {
            if constexpr (ROps::TRACKS_ARG) {
                // A node with no in-edges keeps the identity (+-inf), which is
                // not a meaningful feature value -- report zero instead, and
                // let the invalid arg index suppress its gradient.
                result[e]                         = Sentinel::is_valid(best_args[e]) ? static_cast<cuda_t>(best_vals[e]) : zero_val;
                arg_idx[node_stride + base_f + e] = best_args[e];
            } else {
                // Accumulating reducers need no such fixup: the identity is
                // already the right answer for an empty neighborhood.
                result[e] = static_cast<cuda_t>(best_vals[e]);
            }
        }
        Tile::write(&out[node_stride], fv, *reinterpret_cast<Tile::vec_t const *>(&result));
    }

    // Scalar tail for d % TW != 0 (compiles away for TW=1)
    if constexpr (TW > 1) {
        if (d % TW != 0 && tid == 0) {
            for (size_t f = d_vec * TW; f < d; ++f) {
                acc_t best_val   = identity_val;
                index_t best_arg = Sentinel::INVALID;
                for (index_t eid = row_start; eid < row_end; ++eid) {
                    index_t src = index_t{};
                    cuda_t u_val{};
                    if constexpr (BOps::USE_LHS) {
                        src   = edge_idx[eid];
                        u_val = X[static_cast<size_t>(src) * d + f];
                    } else if constexpr (!ARG_IS_EDGE) {
                        src = edge_idx[eid];
                    }
                    accum_t msg[1];
                    aggr_edge_message<BOp, RHS_BROADCAST, 1, cuda_t, index_t, accum_t>(&u_val, rhs, eid, f, d, msg);
                    bool upgrade_index = false;
                    best_val           = ROps::reduce(static_cast<acc_t>(msg[0]), best_val, upgrade_index);
                    if constexpr (ROps::TRACKS_ARG) {
                        if (upgrade_index) {
                            best_arg = ARG_IS_EDGE ? eid : src;
                        }
                    }
                }
                if constexpr (ROps::TRACKS_ARG) {
                    out[node_stride + f]     = Sentinel::is_valid(best_arg) ? static_cast<cuda_t>(best_val) : zero_val;
                    arg_idx[node_stride + f] = best_arg;
                } else {
                    out[node_stride + f] = static_cast<cuda_t>(best_val);
                }
            }
        }
    }
}

__device__ __forceinline__ uint32_t float_to_ordered_uint(float x) {
    uint32_t bits = __float_as_uint(x);
    if (bits & 0x80000000u) {
        // negative: invert bits so ordering is preserved
        return ~bits;
    } else {
        // non-negative: set sign bit so they come after all negatives
        return bits | 0x80000000u;
    }
}

__device__ __forceinline__ float ordered_uint_to_float(uint32_t key) {
    uint32_t bits;
    if (key & 0x80000000u) {
        // non-negative branch
        bits = key & 0x7fffffffu;
    } else {
        // negative branch
        bits = ~key;
    }
    return __uint_as_float(bits);
}

// pack float and int into uint64 for atomic updates (32-bit indices only)
__device__ __forceinline__ uint64_t pack_val_idx(float val, int idx) {
    uint32_t key = float_to_ordered_uint(val);
    return (static_cast<uint64_t>(key) << 32) | static_cast<uint32_t>(idx);
}

// unpack float and int from uint64
__device__ __forceinline__ void unpack_val_idx(uint64_t packed, float& val, int& idx) {
    uint32_t key  = static_cast<uint32_t>(packed >> 32);
    uint32_t idxu = static_cast<uint32_t>(packed & 0xFFFFFFFFu);

    val = ordered_uint_to_float(key);
    idx = static_cast<int>(idxu);
}

// Packed heavy kernel: blockIdx.x = node, blockIdx.y = edge chunk
// Only for 32-bit index types (packs float32 + int32 into uint64)
// PIPELINE_STAGES>0 regresses this kernel, see pipelined_thread_edge_scan.
template <
    size_t EDGES_PER_BLOCK, size_t WARPS_PER_BLOCK, FloatingNum cuda_t, ReductionOp Op, typename index_t, FloatingNum accum_t = float,
    int PIPELINE_STAGES = 0
>
__global__ void __launch_bounds__(WARPS_PER_BLOCK *kWarpSize) reduction_aggr_forward_heavy_kernel(
    index_t const *const __restrict__ heavy_nodes_indices,
    index_t const *const __restrict__ edge_ptr,
    index_t const *const __restrict__ edge_idx,
    cuda_t const *const __restrict__ X,
    uint64_t *const __restrict__ packed,
    size_t d
) {
    static_assert(sizeof(index_t) <= 4, "Packed heavy kernel only supports 32-bit index types");
    using ROps     = ReductionOps<Op>;
    using Sentinel = IndexSentinel<index_t>;
    // constexpr size_t TW  = (sizeof(cuda_t) <= 2) ? 2 : 1;
    constexpr size_t TW = VecFloat<1, cuda_t>::max_vec_size_bytes / sizeof(cuda_t);
    using Tile          = TileOps<TW, cuda_t>;

    const size_t node_idx  = blockIdx.x;
    const size_t chunk_idx = blockIdx.y;
    const index_t v        = heavy_nodes_indices[node_idx];

    const index_t row_start = edge_ptr[v];
    const index_t row_end   = edge_ptr[v + 1];

    const index_t chunk_start         = row_start + static_cast<index_t>(chunk_idx * EDGES_PER_BLOCK);
    const index_t chunk_end_candidate = chunk_start + static_cast<index_t>(EDGES_PER_BLOCK);
    const index_t chunk_end           = (chunk_end_candidate < row_end) ? chunk_end_candidate : row_end;

    // exit for chunks beyond this node's edges
    if (chunk_start >= row_end) [[unlikely]] {
        return;
    }

    const size_t tid           = threadIdx.x;
    constexpr size_t BLOCK_DIM = WARPS_PER_BLOCK * kWarpSize;
    const cuda_t identity_val  = static_cast<cuda_t>(ROps::IDENTITY);

    const size_t d_vec = d / TW;

    static_assert(PIPELINE_STAGES >= 0, "pipeline_stages must be >= 0 (0 disables the pipeline)");
    constexpr bool USE_PIPELINE = PIPELINE_STAGES > 0;
    constexpr int NUM_STAGES    = PIPELINE_STAGES;

    extern __shared__ __align__(16) uint8_t sh_raw[];
    cuda_t *val_dbuf = reinterpret_cast<cuda_t *>(sh_raw);  // only meaningful when USE_PIPELINE
    cuda_t *my_dbuf  = val_dbuf + tid * NUM_STAGES * TW;

    for (size_t fv = tid; fv < d_vec; fv += BLOCK_DIM) {
        const size_t base_f = fv * TW;

        cuda_t best_vals[TW];
        index_t best_srcs[TW];
#pragma unroll
        for (size_t e = 0; e < TW; ++e) {
            best_vals[e] = identity_val;
            best_srcs[e] = Sentinel::INVALID;
        }

        auto visit = [&best_vals, &best_srcs](index_t src, cuda_t const *val) {
#pragma unroll
            for (size_t e = 0; e < TW; ++e) {
                cuda_t v_e = val[e];
                bool upgrade_index = false;
                best_vals[e] = ROps::reduce(v_e, best_vals[e], upgrade_index);
                if (upgrade_index) {
                    best_srcs[e] = src;
                }
            }
        };

        if constexpr (USE_PIPELINE) {
            pipelined_thread_edge_scan<TW, NUM_STAGES, cuda_t, index_t>(chunk_start, chunk_end, edge_idx, X, d, base_f, my_dbuf, visit);
        } else {
            for (index_t eid = chunk_start; eid < chunk_end; ++eid) {
                index_t src                    = edge_idx[eid];
                const typename Tile::vec_t val = Tile::read(&X[static_cast<size_t>(src) * d], fv);
                visit(src, val.data);
            }
        }

#pragma unroll
        for (size_t e = 0; e < TW; ++e) {
            if (Sentinel::is_valid(best_srcs[e])) {
                uint64_t new_val = pack_val_idx(static_cast<accum_t>(best_vals[e]), static_cast<size_t>(best_srcs[e]));
                ROps::atomic_reduce(&packed[node_idx * d + base_f + e], new_val);
            }
        }
    }

    // scalar tail for d % TW != 0
    if (d % TW != 0 && tid == 0) {
        for (size_t f = d_vec * TW; f < d; ++f) {
            cuda_t local_best = identity_val;
            index_t local_arg = Sentinel::INVALID;

            for (index_t eid = chunk_start; eid < chunk_end; ++eid) {
                const index_t src = edge_idx[eid];
                const cuda_t val  = X[static_cast<size_t>(src) * d + f];
                bool upgrade_index = false;
                local_best = ROps::reduce(val, local_best, upgrade_index);
                if (upgrade_index) {
                    local_arg  = src;
                }
            }

            if (Sentinel::is_valid(local_arg)) {
                const uint64_t new_val = pack_val_idx(static_cast<accum_t>(local_best), static_cast<size_t>(local_arg));
                ROps::atomic_reduce(&packed[node_idx * d + f], new_val);
            }
        }
    }
}

// unpack results back to separate arrays (32-bit indices only, pairs with heavy kernel)
template <size_t WARPS_PER_BLOCK, FloatingNum cuda_t, typename index_t>
__global__ void __launch_bounds__(WARPS_PER_BLOCK *kWarpSize) unpack_results_kernel(
    uint64_t const *const __restrict__ packed,
    index_t const *const __restrict__ nodes,
    cuda_t *const __restrict__ out,
    index_t *const __restrict__ arg_idx,
    size_t num_nodes,
    size_t d
) {
    static_assert(sizeof(index_t) <= 4, "Unpack kernel only supports 32-bit index types");
    constexpr size_t BLOCK_DIM = WARPS_PER_BLOCK * kWarpSize;
    const size_t tid           = blockIdx.x * BLOCK_DIM + threadIdx.x;

    for (size_t i = tid; i < num_nodes * d; i += gridDim.x * BLOCK_DIM) {
        size_t node_idx = i / d;
        size_t f        = i % d;
        index_t v       = nodes[node_idx];

        float val;
        int idx;
        unpack_val_idx(packed[node_idx * d + f], val, idx);

        out[static_cast<size_t>(v) * d + f]     = (idx > -1) ? static_cast<cuda_t>(val) : cuda_t{};
        arg_idx[static_cast<size_t>(v) * d + f] = static_cast<index_t>(idx);
    }
}

// Dynamic shared-memory layout of the tiled heavy kernel: `slots` floats,
// then `slots` index_t.  The float region is rounded up to 16 bytes so that an
// 8-byte index type stays aligned for any (tiles_y, features_per_block) -- with
// the regions merely adjacent, an odd slot count misaligns the index array.
// The launcher must size its allocation with aggr_heavy_shmem_bytes().
template <typename index_t>
__host__ __device__ inline size_t aggr_heavy_shmem_val_bytes(size_t slots) {
    return ((slots * sizeof(float) + 15) / 16) * 16;
}

template <typename index_t>
__host__ __device__ inline size_t aggr_heavy_shmem_bytes(size_t slots) {
    return aggr_heavy_shmem_val_bytes<index_t>(slots) + ((slots * sizeof(index_t) + 15) / 16) * 16;
}

// 2D kernel: blockIdx.x = node, threadIdx.x = feature, threadIdx.y = edge tile
// uses shared memory tree reduction across tiles instead of packed atomicMin/Max
// Works with all index sizes (no packing constraint)
template <
    FloatingNum cuda_t, ReductionOp Op, typename index_t, FloatingNum accum_t = float, BinaryOp BOp = BinaryOp::COPY_U,
    bool RHS_BROADCAST = false, bool ARG_IS_EDGE = false, bool VECTORIZE = true
>
__global__ void reduction_aggr_forward_heavy_kernel_2d(
    const index_t *__restrict__ nodes,
    const index_t *__restrict__ edge_ptr,
    const index_t *__restrict__ edge_idx,
    const cuda_t *__restrict__ X,
    cuda_t *__restrict__ out,
    index_t *__restrict__ arg_idx,
    size_t d,
    const cuda_t *__restrict__ rhs = nullptr
) {
    using ROps     = ReductionOps<Op>;
    using BOps     = BinaryOps<BOp>;
    using Sentinel = IndexSentinel<index_t>;
    using acc_t    = typename ROps::template AccumType<cuda_t, accum_t>;
    // constexpr int TW  = (sizeof(cuda_t) <= 2) ? 2 : 1;
    constexpr size_t TW = aggr_tile_width<VECTORIZE, cuda_t>;
    using Tile          = TileOps<TW, cuda_t>;

    size_t i  = blockIdx.x;
    index_t v = nodes[i];

    index_t row_start   = edge_ptr[v];
    index_t row_end     = edge_ptr[v + 1];
    const size_t degree = static_cast<size_t>(row_end - row_start);

    size_t fid = threadIdx.x;  // feature dimension
    size_t tid = threadIdx.y;  // tile index

    const size_t F_BLOCK      = blockDim.x;
    const size_t TILES_Y      = blockDim.y;
    const size_t SHMEM_STRIDE = F_BLOCK * TW;

    extern __shared__ __align__(16) uint8_t shared_mem[];
    float *shmem_val = reinterpret_cast<float *>(shared_mem);
    // index_t shared memory for arg indices
    index_t *shmem_idx = reinterpret_cast<index_t *>(shared_mem + aggr_heavy_shmem_val_bytes<index_t>(TILES_Y * SHMEM_STRIDE));

    const acc_t identity_val = static_cast<acc_t>(ROps::IDENTITY);
    constexpr cuda_t zero_val{};

    size_t tile_size_ceil = (degree + TILES_Y - 1) / TILES_Y;
    index_t start         = row_start + static_cast<index_t>(tid * tile_size_ceil);
    index_t end_candidate = start + static_cast<index_t>(tile_size_ceil);
    index_t end           = (end_candidate < row_end) ? end_candidate : row_end;

    size_t node_stride = static_cast<size_t>(v) * d;
    const size_t d_vec = d / TW;

    for (size_t fv = fid; fv < d_vec; fv += F_BLOCK) {
        const size_t base_f = fv * TW;

        acc_t best_vals[TW];
        index_t best_args[TW];
#pragma unroll
        for (size_t e = 0; e < TW; ++e) {
            best_vals[e] = identity_val;
            best_args[e] = Sentinel::INVALID;
        }

        auto visit = [&](index_t src, index_t eid, cuda_t const *uslice) {
            accum_t msg[TW];
            aggr_edge_message<BOp, RHS_BROADCAST, TW, cuda_t, index_t, accum_t>(uslice, rhs, eid, fv, d, msg);
#pragma unroll
            for (size_t e = 0; e < TW; ++e) {
                const acc_t v_e    = static_cast<acc_t>(msg[e]);
                bool upgrade_index = false;
                best_vals[e]       = ROps::reduce(v_e, best_vals[e], upgrade_index);
                if constexpr (ROps::TRACKS_ARG) {
                    if (upgrade_index) {
                        best_args[e] = ARG_IS_EDGE ? eid : src;
                    }
                }
            }
        };

        for (index_t eid = start; eid < end; ++eid) {
            if constexpr (BOps::USE_LHS) {
                const index_t src              = edge_idx[eid];
                const typename Tile::vec_t val = Tile::read(&X[static_cast<size_t>(src) * d], fv);
                visit(src, eid, val.data);
            } else if constexpr (ARG_IS_EDGE) {
                // copy_e reporting edge args touches neither X nor edge_idx
                visit(index_t{}, eid, nullptr);
            } else {
                visit(edge_idx[eid], eid, nullptr);
            }
        }

// Write to shmem (convert to float for cross-tile reduction)
#pragma unroll
        for (size_t e = 0; e < TW; ++e) {
            shmem_val[tid * SHMEM_STRIDE + fid * TW + e] = static_cast<accum_t>(best_vals[e]);
            shmem_idx[tid * SHMEM_STRIDE + fid * TW + e] = best_args[e];
        }

        __syncthreads();

        // Tree reduction across tiles
        for (size_t offset = TILES_Y / 2; offset > 0; offset /= 2) {
            if (tid < offset) {
#pragma unroll
                for (size_t e = 0; e < TW; ++e) {
                    const size_t a = tid * SHMEM_STRIDE + fid * TW + e;
                    const size_t b = (tid + offset) * SHMEM_STRIDE + fid * TW + e;

                    const float val_a   = shmem_val[a];
                    const index_t idx_a = shmem_idx[a];
                    const float val_b   = shmem_val[b];
                    const index_t idx_b = shmem_idx[b];

                    if constexpr (ROps::TRACKS_ARG) {
                        bool take_b = false;
                        ROps::reduce(val_b, val_a, take_b);
                        // Tie-break on the smaller arg index (source node, or
                        // edge position under ARG_IS_EDGE) so the result does
                        // not depend on how edges split across tiles.
                        if (take_b || (val_b == val_a && Sentinel::is_valid(idx_b) && (!Sentinel::is_valid(idx_a) || idx_b < idx_a))) {
                            shmem_val[a] = val_b;
                            shmem_idx[a] = idx_b;
                        }
                    } else {
                        bool unused  = false;
                        shmem_val[a] = ROps::reduce(val_b, val_a, unused);
                    }
                }
            }
            __syncthreads();
        }

        // Vectorized final write
        if (tid == 0) {
            cuda_t result[TW];
#pragma unroll
            for (size_t e = 0; e < TW; ++e) {
                if constexpr (ROps::TRACKS_ARG) {
                    index_t best_idx                  = shmem_idx[fid * TW + e];
                    result[e]                         = Sentinel::is_valid(best_idx) ? static_cast<cuda_t>(shmem_val[fid * TW + e]) : zero_val;
                    arg_idx[node_stride + base_f + e] = best_idx;
                } else {
                    result[e] = static_cast<cuda_t>(shmem_val[fid * TW + e]);
                }
            }
            Tile::write(&out[node_stride], fv, *reinterpret_cast<Tile::vec_t const *>(&result));
        }

        __syncthreads();
    }

    // scalar tail for d % TW != 0 (compiles away for TW=1)
    if constexpr (TW > 1) {
        if (d % TW != 0) {
            const size_t tail_f = d_vec * TW;

            // only fid==0 does actual edge scanning; others contribute identity/INVALID.
            float local_best  = ROps::IDENTITY;
            index_t local_arg = Sentinel::INVALID;

            if (fid == 0) {
                for (index_t eid = start; eid < end; ++eid) {
                    index_t src = index_t{};
                    cuda_t u_val{};
                    if constexpr (BOps::USE_LHS) {
                        src   = edge_idx[eid];
                        u_val = X[static_cast<size_t>(src) * d + tail_f];
                    } else if constexpr (!ARG_IS_EDGE) {
                        src = edge_idx[eid];
                    }
                    accum_t msg[1];
                    aggr_edge_message<BOp, RHS_BROADCAST, 1, cuda_t, index_t, accum_t>(&u_val, rhs, eid, tail_f, d, msg);
                    bool upgrade_index = false;
                    local_best         = ROps::reduce(static_cast<float>(msg[0]), local_best, upgrade_index);
                    if constexpr (ROps::TRACKS_ARG) {
                        if (upgrade_index) {
                            local_arg = ARG_IS_EDGE ? eid : src;
                        }
                    }
                }
            }

            shmem_val[tid * SHMEM_STRIDE + fid] = local_best;
            shmem_idx[tid * SHMEM_STRIDE + fid] = local_arg;

            __syncthreads();

            for (size_t offset = TILES_Y / 2; offset > 0; offset /= 2) {
                if (tid < offset && fid == 0) {
                    const size_t a = tid * SHMEM_STRIDE;
                    const size_t b = (tid + offset) * SHMEM_STRIDE;

                    const float val_a   = shmem_val[a];
                    const index_t idx_a = shmem_idx[a];
                    const float val_b   = shmem_val[b];
                    const index_t idx_b = shmem_idx[b];

                    if constexpr (ROps::TRACKS_ARG) {
                        bool take_b = false;
                        ROps::reduce(val_b, val_a, take_b);
                        if (take_b || (val_b == val_a && Sentinel::is_valid(idx_b) && (!Sentinel::is_valid(idx_a) || idx_b < idx_a))) {
                            shmem_val[a] = val_b;
                            shmem_idx[a] = idx_b;
                        }
                    } else {
                        bool unused  = false;
                        shmem_val[a] = ROps::reduce(val_b, val_a, unused);
                    }
                }
                __syncthreads();
            }

            if (tid == 0 && fid == 0) {
                float best_val = shmem_val[0];
                if constexpr (ROps::TRACKS_ARG) {
                    index_t best_idx              = shmem_idx[0];
                    out[node_stride + tail_f]     = Sentinel::is_valid(best_idx) ? static_cast<cuda_t>(best_val) : zero_val;
                    arg_idx[node_stride + tail_f] = best_idx;
                } else {
                    out[node_stride + tail_f] = static_cast<cuda_t>(best_val);
                }
            }
        }
    }
}

// Backward of an arg-tracking reducer: only the winning edge of each output
// element contributed, so one scatter over arg_idx yields every gradient.
//
// With the defaults this is plain reduction_aggr -- arg_idx holds the source
// node, the message was the node value, and grad_x is the only output.  Under
// ARG_IS_EDGE the index is a CSR edge position instead, `edge_idx` recovers the
// source node from it, and the edge half of the gradient lands in grad_rhs
// (shaped [E, d], or [E] when the edge operand broadcast over the features).
//
// grad_t is the accumulate dtype: cuda_t for reduction_aggr, float for g-SpMM,
// whose Python layer casts back -- fp16 atomicAdd is arch-gated and lossy.
template <
    size_t WARPS_PER_BLOCK, FloatingNum cuda_t, typename index_t, BinaryOp BOp = BinaryOp::COPY_U, bool RHS_BROADCAST = false,
    bool ARG_IS_EDGE = false, FloatingNum grad_t = cuda_t, FloatingNum accum_t = float
>
__global__ void __launch_bounds__(WARPS_PER_BLOCK *kWarpSize) reduction_aggr_backward_typed(
    const cuda_t *__restrict__ grad_out,
    const index_t *__restrict__ arg_idx,
    grad_t *__restrict__ grad_x,
    size_t num_nodes,
    size_t d,
    const index_t *__restrict__ edge_idx = nullptr,
    const cuda_t *__restrict__ lhs       = nullptr,
    const cuda_t *__restrict__ rhs       = nullptr,
    grad_t *__restrict__ grad_rhs        = nullptr
) {
    using Sentinel = IndexSentinel<index_t>;
    using BOps     = BinaryOps<BOp>;

    size_t block_idx = blockIdx.x;
    if (block_idx >= num_nodes) {
        return;
    }

    size_t tid = threadIdx.x;
    // blockDim.x, not WARPS_PER_BLOCK * kWarpSize: reduction_aggr launches with
    // exactly that many threads, but g-SpMM pins WARPS_PER_BLOCK to the CUDA
    // maximum (so it needs one instantiation, not one per warp count) and picks
    // its block size at runtime.  WARPS_PER_BLOCK still sets __launch_bounds__.
    const size_t block_dim   = blockDim.x;
    const size_t base_offset = block_idx * d;

    for (size_t f = tid; f < d; f += block_dim) {
        const index_t arg = arg_idx[base_offset + f];
        if (!Sentinel::is_valid(arg)) {
            continue;  // node with no in-edges: nothing reached it, nothing flows back
        }

        const index_t u    = ARG_IS_EDGE ? edge_idx[static_cast<size_t>(arg)] : arg;
        const accum_t g    = static_cast<accum_t>(grad_out[base_offset + f]);
        const size_t e_off = RHS_BROADCAST ? static_cast<size_t>(arg) : static_cast<size_t>(arg) * d + f;

        // Only mul and div differentiate to something that reads the operands;
        // for the rest the loads would be dead, and for copy_u/copy_e the
        // pointer they would read is null.
        accum_t u_val{};
        accum_t e_val{};
        if constexpr (BOps::GRAD_USES_OPERANDS) {
            u_val = static_cast<accum_t>(lhs[static_cast<size_t>(u) * d + f]);
            e_val = static_cast<accum_t>(rhs[e_off]);
        }

        if constexpr (BOps::USE_LHS) {
            atomicAdd(&grad_x[static_cast<size_t>(u) * d + f], static_cast<grad_t>(BOps::grad_lhs(u_val, e_val, g)));
        }
        if constexpr (BOps::USE_RHS) {
            // Both writes need atomics: several destinations can pick the same
            // source node, and a broadcast edge operand collapses all d
            // features of one edge into a single slot.
            atomicAdd(&grad_rhs[e_off], static_cast<grad_t>(BOps::grad_rhs(u_val, e_val, g)));
        }
    }
}
