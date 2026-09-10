#include <bit>
#include <cstdint>
#include <type_traits>

#include "common/misc.cuh"
#include "common/tile.cuh"
#include "common/traits.cuh"
#include "gsddmm/gsddmm.cuh"

namespace gsddmm {

// =============================================================================
// Elementwise GSDDMM op on vector tiles. Dot is reduced separately (it is not
// elementwise); Copy never touches the right operand.
// =============================================================================
template <GSDDMM_OP op, size_t TW, FloatingNum cuda_t>
struct GsddmmVecOp {
    using vec_t = VecFloat<TW, cuda_t>;

    static constexpr __device__ __forceinline__ vec_t apply(vec_t l, vec_t r) {
        if constexpr (op == GSDDMM_OP::Add) {
            l.add_(r);
        } else if constexpr (op == GSDDMM_OP::Sub) {
            l.sub_(r);
        } else if constexpr (op == GSDDMM_OP::Mul) {
            l.mul_(r);
        } else if constexpr (op == GSDDMM_OP::Div) {
            l.div_(r);
        } else if constexpr (op == GSDDMM_OP::Copy) {
            // left operand passes through unchanged
        } else if constexpr (op == GSDDMM_OP::Dot) {
            // operators are unchanged there
        } else {
            // Dependent-false: only fires if this branch is ever instantiated
            // (it is not -- every op is covered above).
            static_assert(!sizeof(cuda_t), "GsddmmVecOp::apply reached an unhandled op");
            __builtin_unreachable();
        }
        return l;
    }
};

// Lane-parallel cp.async of NUM_ROWS operand rows into one ring-buffer slot.
// The NUM_ROWS * (row bytes / 16) 16-byte chunks are spread over the warp as
// ONE flat chunk index, so e.g. an L+R pair of 256-byte rows costs a single
// cp.async per lane instead of two half-warp-predicated ones. Operand r of the
// slot lives at slot_rows + r * ROW_STRIDE (see gsddmm_pipelined_row_loop).
// The per-lane source is picked with an unrolled select chain: indexing srcs[]
// by a lane-dependent value would demote the array to local memory.
template <size_t D_CONST, size_t NUM_ROWS, size_t ROW_STRIDE, FloatingNum cuda_t>
__device__ __forceinline__ void gsddmm_async_copy_rows_warp(
    cuda_t *slot_rows, cuda_t const *const (&srcs)[NUM_ROWS], cuda::pipeline<cuda::thread_scope_thread>& pipe, size_t lane
) {
    constexpr size_t ROW_BYTES = D_CONST * sizeof(cuda_t);
    static_assert(ROW_BYTES % 16 == 0, "Row width must be a multiple of 16 bytes for aligned async copies");
    constexpr size_t CHUNK_ELEMS    = 16 / sizeof(cuda_t);
    constexpr size_t CHUNKS_PER_ROW = ROW_BYTES / 16;
    static_assert(std::popcount(CHUNKS_PER_ROW) == 1, "Chunks per row must be a power of two so the row of a chunk is a shift");
    constexpr size_t TOTAL_CHUNKS = NUM_ROWS * CHUNKS_PER_ROW;
    using chunk_t                 = Vec<CHUNK_ELEMS, cuda_t>;

#pragma unroll
    for (size_t base = 0; base < TOTAL_CHUNKS; base += kWarpSize) {
        const size_t c = base + lane;
        if (c < TOTAL_CHUNKS) {
            const size_t r    = c / CHUNKS_PER_ROW;
            const size_t i    = c % CHUNKS_PER_ROW;
            cuda_t const *src = srcs[0];
            cuda_t *dst       = slot_rows;
#pragma unroll
            for (size_t rr = 1; rr < NUM_ROWS; ++rr) {
                if (r == rr) {
                    src = srcs[rr];
                    dst = slot_rows + rr * ROW_STRIDE;
                }
            }
            cuda::memcpy_async(
                reinterpret_cast<chunk_t *>(dst) + i, reinterpret_cast<chunk_t const *>(src) + i, cuda::aligned_size_t<16>(sizeof(chunk_t)), pipe
            );
        }
    }
}

// =============================================================================
// cp.async pipeline over a warp's sequence of loop_iters row gathers:
// NUM_STAGES-deep prefetch of NUM_ROWS rows per iteration into a per-warp
// shared ring buffer. Modeled on pipelined_neighbor_row_loop (pipeline.cuh);
// the addressing is delegated to addr() because GSDDMM rows are indexed by
// neighbor id, by edge position or through an explicit edge list, none of
// which col_idx alone expresses -- which is why this variant exists.
//
// Unlike pipelined_neighbor_row_loop, the prefetch of stage iter+NUM_STAGES is
// issued BEFORE consume(iter), so the global->shared copy overlaps with the
// compute. The ring buffer therefore has NUM_STAGES + 1 slots: one slot is
// being consumed while up to NUM_STAGES more are in flight.
//
// Ring layout: operand r of slot s is at dbuf + s * D_CONST + r * ROW_STRIDE,
// ROW_STRIDE = NUM_SLOTS * D_CONST. Slot addresses are COMPUTED from the
// running slot counters, never read from a pointer array: a register array
// indexed by a runtime slot is demoted to local memory (measured: 19M local
// loads and a 4x instruction count for NUM_SLOTS = 3), and for the same reason
// consume() receives the iteration index and recomputes its edge id itself.
//
// addr(it, srcs): fills srcs[r] with the global address of operand r's row for
// iteration it (must be warp-uniform).
// consume(it, rows): rows[r] is operand r's prefetched row in shared memory;
// valid only inside the call (the slot is recycled on return).
//
// dbuf: this warp's private shared scratch, NUM_ROWS * (NUM_STAGES + 1) * D_CONST elements.
// =============================================================================
template <size_t D_CONST, size_t NUM_STAGES, size_t NUM_ROWS, FloatingNum cuda_t, typename AddrFn, typename ConsumeFn>
__device__ __forceinline__ void gsddmm_pipelined_row_loop(size_t lane, size_t loop_iters, cuda_t *__restrict__ dbuf, AddrFn&& addr, ConsumeFn&& consume) {
    static_assert(NUM_STAGES >= 1, "The pipeline needs at least one stage in flight; PIPELINE_STAGES == 0 takes the direct-load loop");
    if (loop_iters == 0) [[unlikely]] {
        return;
    }

    // NUM_STAGES prefetch slots + the slot currently being consumed.
    constexpr size_t NUM_SLOTS  = NUM_STAGES + 1;
    constexpr size_t ROW_STRIDE = NUM_SLOTS * D_CONST;

    auto advance = [](size_t slot) { return (slot + 1 == NUM_SLOTS) ? size_t{0} : slot + 1; };

    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

    auto prefetch = [&pipe, dbuf, loop_iters, lane, addr_ = std::move(addr)](size_t it, size_t slot) {
        pipe.producer_acquire();
        if (it < loop_iters) {
            cuda_t const *srcs[NUM_ROWS];
            addr_(it, srcs);
            // All NUM_ROWS rows of the stage go into one commit group, so the
            // operands (L and R) are in flight concurrently.
            gsddmm_async_copy_rows_warp<D_CONST, NUM_ROWS, ROW_STRIDE, cuda_t>(dbuf + slot * D_CONST, srcs, pipe, lane);
        }
        pipe.producer_commit();
    };

    // Fill the pipeline: NUM_STAGES stages in flight, one slot still free.
#pragma unroll
    for (size_t s = 0; s < NUM_STAGES; ++s) {
        prefetch(s, s);
    }

    size_t consume_slot  = 0;
    size_t prefetch_slot = NUM_STAGES;  // the one free slot
    for (size_t iter = 0; iter < loop_iters; ++iter) {
        cuda::pipeline_consumer_wait_prior<NUM_STAGES - 1>(pipe);

        // Issue the next stage's copies before consuming the current one: the
        // global->shared transfer overlaps with the compute below. The target
        // slot is the one the previous iteration's consume freed -- never the
        // slot about to be read.
        prefetch(iter + NUM_STAGES, prefetch_slot);

        // The thread-scope wait covers only this lane's own cp.async groups; a
        // lane may read chunks copied by other lanes (tile < 16B, or a row <
        // 512B leaves lanes idle), so completion must be observed warp-wide.
        __syncwarp();

        cuda_t const *cur_rows[NUM_ROWS];
#pragma unroll
        for (size_t r = 0; r < NUM_ROWS; ++r) {
            cur_rows[r] = dbuf + consume_slot * D_CONST + r * ROW_STRIDE;
        }

        consume(iter, cur_rows);

        // All lanes must finish reading the slot before prefetch() reuses it.
        __syncwarp();
        pipe.consumer_release();

        consume_slot  = advance(consume_slot);
        prefetch_slot = advance(prefetch_slot);
    }
}

// CSR-row flavour of the loop above for GSDDMM_forward_normal: warp-strided
// walk over the edge list of one CSR row, Src_V rows indexed by the neighbor
// id j = col_idx[e], Edge rows indexed by the edge position e. consume(e, rows)
// receives the edge position.
template <size_t N_PER_BLOCK, size_t D_CONST, size_t NUM_STAGES, size_t NUM_ROWS, FloatingNum cuda_t, IntegralNum index_t, typename ConsumeFn>
__device__ __forceinline__ void gsddmm_pipelined_edge_loop(
    size_t warp_id,
    size_t lane,
    size_t num_edges,
    index_t edge_start,
    index_t const *__restrict__ col_idx,
    cuda_t const *__restrict__ const (&row_bases)[NUM_ROWS],
    bool const (&row_edge_indexed)[NUM_ROWS],
    cuda_t *__restrict__ dbuf,
    ConsumeFn&& consume
) {
    const size_t loop_iters = (num_edges > warp_id) ? ceil_div(num_edges - warp_id, N_PER_BLOCK) : 0;

    auto edge_of = [edge_start, warp_id](size_t it) { return edge_start + static_cast<index_t>(warp_id + it * N_PER_BLOCK); };

    auto addr = [edge_of, row_bases, col_idx, row_edge_indexed](size_t it, cuda_t const *(&srcs)[NUM_ROWS]) {
        const index_t e = edge_of(it);
        const index_t j = col_idx[e];
#pragma unroll
        for (size_t r = 0; r < NUM_ROWS; ++r) {
            const size_t row_id = row_edge_indexed[r] ? static_cast<size_t>(e) : static_cast<size_t>(j);
            srcs[r]             = row_bases[r] + row_id * D_CONST;
        }
    };
    auto consume_it = [edge_of, consume_ = std::move(consume)](size_t it, cuda_t const *const (&rows)[NUM_ROWS]) {
        consume_(static_cast<size_t>(edge_of(it)), rows);
    };

    gsddmm_pipelined_row_loop<D_CONST, NUM_STAGES, NUM_ROWS, cuda_t>(lane, loop_iters, dbuf, addr, consume_it);
}

// =============================================================================
// GSDDMM forward kernel. See gsddmm.cuh for semantics and conventions.
// =============================================================================
template <GSDDMM_OP op, GSDDMM_MEMBER ll, GSDDMM_MEMBER rr, size_t N_PER_BLOCK, size_t D_CONST, FloatingNum cuda_t, IntegralNum index_t, FloatingNum accum_t, int PIPELINE_STAGES>
__global__ void __launch_bounds__(N_PER_BLOCK *kWarpSize) GSDDMM_forward_normal( // no-format
    size_t N,
    cuda_t const *__restrict__ L, cuda_t const *__restrict__ R, cuda_t *__restrict__ O,
    index_t const *__restrict__ row_ptr, index_t const *__restrict__ col_idx,
    index_t const *__restrict__ node_indices,
    index_t const *__restrict__ block_part, uint32_t edges_per_block
) {
    static_assert(D_CONST % 32 == 0, "D_CONST must be a multiple of 32 so a warp covers the row an integral number of times");
    static_assert(std::popcount(D_CONST / 32) == 1, "D_CONST / 32 must be a power of two for the tile decomposition");
    static_assert((D_CONST * sizeof(cuda_t)) % 16 == 0, "Row width in bytes must be a multiple of 16 for wide copies");

    using Plan = GsddmmPlan<op, ll, rr>;

    using TW_SELECTOR = SelectTW<D_CONST, cuda_t>;

    constexpr size_t TW = TW_SELECTOR::value;  // Tile width
    static_assert(D_CONST % TW == 0, "Feature dim should be divisible by Tile width");
    constexpr size_t TILES            = D_CONST / TW;
    constexpr size_t TILES_PER_THREAD = ceil_div(TILES, kWarpSize);

    using Tile  = TileOps<TW, cuda_t, accum_t>;
    using vec_t = typename Tile::vec_t;
    using VecOp = GsddmmVecOp<op, TW, cuda_t>;

    static_assert(PIPELINE_STAGES >= 0, "pipeline_stages must be >= 0 (0 disables the pipeline)");
    constexpr int NUM_STAGES = PIPELINE_STAGES;
    // With no per-edge rows (both operands are Dst_V, or Copy from Dst_V) there
    // is nothing to prefetch — the pipeline degenerates to the plain loop.
    constexpr bool USE_PIPELINE = (PIPELINE_STAGES > 0) && (Plan::NUM_EDGE_ROWS > 0);

    constexpr size_t ROWS_CAP = Plan::NUM_EDGE_ROWS > 0 ? Plan::NUM_EDGE_ROWS : 1;  // arrays need a nonzero bound

    const size_t node_i = static_cast<size_t>(node_indices[blockIdx.x]);

    __builtin_assume(threadIdx.y < static_cast<unsigned>(N_PER_BLOCK));
    const size_t lane_id = threadIdx.x;
    __builtin_assume(lane_id < static_cast<size_t>(kWarpSize));
    const size_t warp_id = threadIdx.y;

    if (node_i >= N) [[unlikely]] {
        return;
    }

    index_t edge_start = row_ptr[node_i];
    index_t edge_end   = row_ptr[node_i + 1];
    // Heavy-node chunking: node_indices then repeats a node once per chunk and
    // block_part says which edges_per_block-wide slice of its edge list this
    // block owns. Without it (nullptr) the block owns the whole row -- the
    // one-block-per-node layout, whose runtime is set by the largest degree.
    if (block_part != nullptr) {
        edge_start += static_cast<index_t>(block_part[blockIdx.x]) * static_cast<index_t>(edges_per_block);
        const index_t chunk_end = edge_start + static_cast<index_t>(edges_per_block);
        edge_end                = chunk_end < edge_end ? chunk_end : edge_end;
    }
    const size_t num_edges = static_cast<size_t>(edge_end - edge_start);

    // Isolated node: no edges, hence no output rows — nothing to do.
    if (num_edges == 0) [[unlikely]] {
        return;
    }

    // Shared memory layout (everything written through 16-byte vectors comes first):
    //   dst_sh[NUM_DST_ROWS * D_CONST] as cuda_t
    //       -- Dst_V operand rows, identical for every edge of this CSR row
    //   edge_dbuf[N_PER_BLOCK * NUM_EDGE_ROWS * (NUM_STAGES + 1) * D_CONST] as cuda_t
    //       -- per-warp ping-pong for gathered Src_V/Edge rows, only when USE_PIPELINE
    extern __shared__ __align__(16) uint8_t sh_raw[];
    cuda_t *const dst_sh    = reinterpret_cast<cuda_t *>(sh_raw);
    cuda_t *const edge_dbuf = dst_sh + Plan::NUM_DST_ROWS * D_CONST;

    // Cooperative staging of the Dst_V operand rows via 16-byte copies (all threads).
    if constexpr (Plan::NUM_DST_ROWS > 0) {
        constexpr size_t ELEMS_PER_F4 = sizeof(float4) / sizeof(cuda_t);  // remainder is guaranteed to be zero
        constexpr size_t NUM_LOADS    = D_CONST / ELEMS_PER_F4;
        const size_t tid              = warp_id * kWarpSize + lane_id;
        constexpr size_t NUM_THREADS  = N_PER_BLOCK * kWarpSize;

        if constexpr (Plan::L_DST) {
            float4 const *const src = reinterpret_cast<float4 const *>(L + node_i * D_CONST);
            float4 *const dst       = reinterpret_cast<float4 *>(dst_sh);
            for (size_t i = tid; i < NUM_LOADS; i += NUM_THREADS) {
                dst[i] = src[i];
            }
        }
        if constexpr (Plan::R_DST) {
            float4 const *const src = reinterpret_cast<float4 const *>(R + node_i * D_CONST);
            float4 *const dst       = reinterpret_cast<float4 *>(dst_sh + (Plan::L_DST ? D_CONST : 0));
            for (size_t i = tid; i < NUM_LOADS; i += NUM_THREADS) {
                dst[i] = src[i];
            }
        }
        __syncthreads();
    }

    cuda_t const *const sh_l = dst_sh;
    cuda_t const *const sh_r = dst_sh + (Plan::L_DST ? D_CONST : 0);

    cuda_t *const my_dbuf = edge_dbuf + warp_id * Plan::NUM_EDGE_ROWS * (NUM_STAGES + 1) * D_CONST;

    // consume(e, rows): apply the op for edge e. rows[] holds the per-edge
    // operand rows (shared-memory slots when pipelined, global rows otherwise);
    // Dst_V operands are taken from the staged shared rows instead.
    auto consume = [lane_id, sh_l, sh_r, &O](size_t e, cuda_t const *const(&rows)[ROWS_CAP]) {
        if constexpr (!Plan::IS_DOT) {
            cuda_t *const o_row = O + e * D_CONST;
#pragma unroll
            for (size_t t = 0; t < TILES_PER_THREAD; ++t) {
                const size_t v = lane_id + kWarpSize * t;
                if (v < TILES) [[likely]] {
                    const vec_t lv = Plan::L_DST ? Tile::read(sh_l, v) : Tile::read(rows[Plan::L_SLOT], v);
                    if constexpr (op == GSDDMM_OP::Copy) {
                        Tile::write(o_row, v, lv);
                    } else {
                        const vec_t rv = Plan::R_DST ? Tile::read(sh_r, v) : Tile::read(rows[Plan::R_SLOT], v);
                        Tile::write(o_row, v, VecOp::apply(lv, rv));
                    }
                }
            }
        } else {
            accum_t partial{};
#pragma unroll
            for (size_t t = 0; t < TILES_PER_THREAD; ++t) {
                const size_t v = lane_id + kWarpSize * t;
                if (v < TILES) [[likely]] {
                    const vec_t lv = Plan::L_DST ? Tile::read(sh_l, v) : Tile::read(rows[Plan::L_SLOT], v);
                    const vec_t rv = Plan::R_DST ? Tile::read(sh_r, v) : Tile::read(rows[Plan::R_SLOT], v);
                    lv.template dot_product_<accum_t>(&partial, rv);
                }
            }
            partial = warp_reduce_sum(partial);
            if (lane_id == 0) {
                O[e] = static_cast<cuda_t>(partial);
            }
        }
    };

    if constexpr (USE_PIPELINE) {
        cuda_t const *row_bases[ROWS_CAP];
        bool row_edge_indexed[ROWS_CAP];
        if constexpr (!Plan::L_DST) {
            row_bases[Plan::L_SLOT]        = L;
            row_edge_indexed[Plan::L_SLOT] = Plan::L_EDGE_INDEXED;
        }
        if constexpr (!Plan::R_DST) {
            row_bases[Plan::R_SLOT]        = R;
            row_edge_indexed[Plan::R_SLOT] = Plan::R_EDGE_INDEXED;
        }
        gsddmm_pipelined_edge_loop<N_PER_BLOCK, D_CONST, NUM_STAGES, Plan::NUM_EDGE_ROWS, cuda_t, index_t>(
            warp_id, lane_id, num_edges, edge_start, col_idx, row_bases, row_edge_indexed, my_dbuf, consume
        );
    } else {
        // Exact per-warp trip count (no early break) so the loop can be unrolled:
        // with L/R/O restrict-qualified the compiler then hoists the next edge's
        // loads above the current edge's stores, keeping two edges' worth of
        // requests in flight per warp instead of one.
        const size_t loop_iters = (num_edges > warp_id) ? ceil_div(num_edges - warp_id, N_PER_BLOCK) : 0;
#pragma unroll 2
        for (size_t it = 0; it < loop_iters; ++it) {
            const index_t e = edge_start + static_cast<index_t>(warp_id + it * N_PER_BLOCK);
            const index_t j = col_idx[e];

            cuda_t const *rows[ROWS_CAP];
            if constexpr (!Plan::L_DST) {
                rows[Plan::L_SLOT] = L + (Plan::L_EDGE_INDEXED ? static_cast<size_t>(e) : static_cast<size_t>(j)) * D_CONST;
            }
            if constexpr (!Plan::R_DST) {
                rows[Plan::R_SLOT] = R + (Plan::R_EDGE_INDEXED ? static_cast<size_t>(e) : static_cast<size_t>(j)) * D_CONST;
            }
            consume(static_cast<size_t>(e), rows);
        }
    }
}

// Row of operand `m` for edge k of this warp's chunk. Node ids come from the
// (src, dst) pair lane k cached in my_pair (warp-wide shuffle broadcast, so all
// lanes must reach this together); Edge rows are indexed by the edge position.
template <GSDDMM_MEMBER m, size_t D_CONST, FloatingNum cuda_t>
__device__ __forceinline__ cuda_t const *gsddmm_edge_member_row(cuda_t const *__restrict__ base, const ulonglong2& my_pair, size_t k, uint64_t e) {
    if constexpr (m == GSDDMM_MEMBER::Src_V) {
        return base + D_CONST * __shfl_sync(FULL_WARP_MASK, my_pair.x, static_cast<int>(k));
    } else if constexpr (m == GSDDMM_MEMBER::Dst_V) {
        return base + D_CONST * __shfl_sync(FULL_WARP_MASK, my_pair.y, static_cast<int>(k));
    } else if constexpr (m == GSDDMM_MEMBER::Edge) {
        return base + D_CONST * e;
    } else {
        static_assert(!sizeof(cuda_t), "Unreachable branch");
        __builtin_unreachable();
    }
}

// =============================================================================
// Edge-parallel GSDDMM forward. See gsddmm.cuh for semantics and conventions.
//
// Work distribution: warp w (global warp index over the whole grid) owns the
// contiguous edge chunk [w * edges_per_warp, (w + 1) * edges_per_warp) of the
// explicit edge list; the blockDim.y warps of a block are independent and only
// share a block to lift the blocks-per-SM occupancy cap of 32-thread blocks.
// edges_per_warp <= kWarpSize, so lane k caches edge k's (src, dst) pair from
// ONE coalesced 16B-per-lane load and broadcasts it later with a shuffle. This
// removes the dependent scalar index load in front of every edge, which is
// what lets the cp.async pipeline issue row copies PIPELINE_STAGES edges ahead
// without a second exposed latency per stage.
//
// Both operand rows of a stage are issued back-to-back into the same cp.async
// group, so L and R are read concurrently. Output rows are stored directly from
// registers: a global store never blocks the warp (nothing here reads O back),
// so the store of edge k already overlaps the copies of edges k+1..k+STAGES and
// gains nothing from staging (sm_80 also has no async shared->global bulk copy;
// that is sm_90+).
// =============================================================================
template <GSDDMM_OP op, GSDDMM_MEMBER ll, GSDDMM_MEMBER rr, size_t D_CONST, FloatingNum cuda_t, FloatingNum accum_t, size_t PIPELINE_STAGES>
__global__ void __launch_bounds__(kWarpSize *kGsddmmEdgeMaxWarpsPerBlock) GSDDMM_forward_edge_block( // no-format
    uint64_t E, // total edge count
    cuda_t const *__restrict__ L, cuda_t const *__restrict__ R, cuda_t *__restrict__ O,
    ulonglong2 const * __restrict__ edge_nodes_idx,
    uint32_t edges_per_warp
) {
    static_assert(D_CONST % 32 == 0, "D_CONST must be a multiple of 32 so a warp covers the row an integral number of times");
    static_assert(std::popcount(D_CONST / 32) == 1, "D_CONST / 32 must be a power of two for the tile decomposition");
    static_assert((D_CONST * sizeof(cuda_t)) % 16 == 0, "Row width in bytes must be a multiple of 16 for wide copies");

    using Plan = GsddmmPlan<op, ll, rr>;

    using TW_SELECTOR = SelectTW<D_CONST, cuda_t>;

    constexpr size_t TW = TW_SELECTOR::value;  // Tile width
    static_assert(D_CONST % TW == 0, "Feature dim should be divisible by Tile width");
    constexpr size_t TILES            = D_CONST / TW;
    constexpr size_t TILES_PER_THREAD = ceil_div(TILES, kWarpSize);

    using Tile  = TileOps<TW, cuda_t, accum_t>;
    using vec_t = typename Tile::vec_t;
    using VecOp = GsddmmVecOp<op, TW, cuda_t>;

    // Every operand is gathered per edge here: slot 0 is the L row, slot 1 the
    // R row (absent for Copy, which never reads R).
    constexpr size_t NUM_ROWS  = Plan::USE_R ? 2 : 1;
    constexpr size_t L_SLOT    = 0;
    constexpr size_t R_SLOT    = 1;
    constexpr bool USE_PIPELINE = PIPELINE_STAGES > 0;
    constexpr size_t NUM_STAGES = USE_PIPELINE ? static_cast<size_t>(PIPELINE_STAGES) : 1;  // only read when USE_PIPELINE

    __builtin_assume(threadIdx.x < static_cast<unsigned>(kWarpSize));
    __builtin_assume(threadIdx.y < static_cast<unsigned>(kGsddmmEdgeMaxWarpsPerBlock));
    __builtin_assume(edges_per_warp >= 1 && edges_per_warp <= kGsddmmEdgeMaxEdgesPerWarp);
    const size_t lane_id = threadIdx.x;
    const size_t warp_id = threadIdx.y;

    const uint64_t block_linear = (static_cast<uint64_t>(blockIdx.z) * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
    const uint64_t global_warp  = block_linear * blockDim.y + warp_id;
    const uint64_t edge_base    = global_warp * edges_per_warp;
    // Warp-uniform exit: there is no block-level synchronization in this kernel.
    if (edge_base >= E) [[unlikely]] {
        return;
    }
    const size_t num_edges = static_cast<size_t>((E - edge_base < edges_per_warp) ? (E - edge_base) : edges_per_warp);

    // Lane k caches the (src, dst) pair of edge edge_base + k. The chunk is unique
    // to this warp, so the load is streamed past L1.
    ulonglong2 my_pair{};
    if (lane_id < num_edges) {
        my_pair = __ldcs(&edge_nodes_idx[edge_base + lane_id]);
    }

    // Global row addresses of the operands for edge k of the chunk (warp-uniform).
    auto edge_rows = [edge_base, my_pair, L, R](size_t k, cuda_t const *(&srcs)[NUM_ROWS]) {
        const uint64_t e = edge_base + k;
        srcs[L_SLOT]     = gsddmm_edge_member_row<ll, D_CONST, cuda_t>(L, my_pair, k, e);
        if constexpr (Plan::USE_R) {
            srcs[R_SLOT] = gsddmm_edge_member_row<rr, D_CONST, cuda_t>(R, my_pair, k, e);
        }
    };

    // consume(k, rows): apply the op for edge k of the chunk from the prefetched
    // shared-memory rows (pipelined path only; the direct path below keeps its
    // own loop so it can cache the shared node row in registers).
    // O is written once and never read back here, so the stores are streaming
    // (evict-first): the [E, D] output must not push the gathered node rows,
    // which ARE re-read by other edges, out of L2.
    auto consume = [lane_id, edge_base, O](size_t k, cuda_t const *const(&rows)[NUM_ROWS]) {
        const uint64_t e = edge_base + k;
        if constexpr (!Plan::IS_DOT) {
            cuda_t *const O_row = O + e * D_CONST;  // elementwise ops write the edge's own row
#pragma unroll
            for (size_t t = 0; t < TILES_PER_THREAD; ++t) {
                const size_t tile_idx = lane_id + kWarpSize * t;
                if (tile_idx < TILES) [[likely]] {
                    const vec_t L_feats = Tile::read<Tile::MemoryHint::NoHint>(rows[L_SLOT], tile_idx);
                    if constexpr (op == GSDDMM_OP::Copy) {
                        Tile::write<Tile::MemoryHint::Streaming>(O_row, tile_idx, L_feats);
                    } else {
                        const vec_t R_feats = Tile::read<Tile::MemoryHint::NoHint>(rows[R_SLOT], tile_idx);
                        Tile::write<Tile::MemoryHint::Streaming>(O_row, tile_idx, VecOp::apply(L_feats, R_feats));
                    }
                }
            }
        } else {
            accum_t partial{};
#pragma unroll
            for (size_t t = 0; t < TILES_PER_THREAD; ++t) {
                const size_t tile_idx = lane_id + kWarpSize * t;
                if (tile_idx < TILES) [[likely]] {
                    const vec_t L_feats = Tile::read<Tile::MemoryHint::NoHint>(rows[L_SLOT], tile_idx);
                    const vec_t R_feats = Tile::read<Tile::MemoryHint::NoHint>(rows[R_SLOT], tile_idx);
                    L_feats.dot_product_(&partial, R_feats);
                }
            }
            partial = warp_reduce_sum(partial);
            if (lane_id == 0) {
                // Dot output is [E] (one scalar per edge), not an [E, D] row.
                O[e] = static_cast<cuda_t>(partial);
            }
        }
    };

    if constexpr (USE_PIPELINE) {
        // Per-warp ring buffer of NUM_ROWS * (NUM_STAGES + 1) rows (see
        // gsddmm_forward_edge_shmem_bytes_per_warp); warps are laid out back to back.
        extern __shared__ __align__(16) uint8_t sh_raw[];
        cuda_t *const my_dbuf = reinterpret_cast<cuda_t *>(sh_raw) + warp_id * NUM_ROWS * (NUM_STAGES + 1) * D_CONST;
        gsddmm_pipelined_row_loop<D_CONST, NUM_STAGES, NUM_ROWS, cuda_t>(lane_id, num_edges, my_dbuf, edge_rows, consume);
    } else {
        // Direct path. The edge list is grouped by dst when an operand reads Dst_V
        // and by src otherwise (see _graph_edge_list), so a warp's chunk usually
        // shares that operand's row. As in the CSR kernel -- where every edge of a
        // block shares the Dst_V row by construction and it is staged once -- the
        // shared row is loaded once per chunk and kept in registers; here the
        // sharing is established with one warp ballot on the lanes' cached node
        // ids, and the specialized loop then has no per-edge check at all (a
        // per-edge id compare was measured to cost more issue slots than the
        // load it saves on the issue-bound dot variant). Mixed chunks take the
        // generic loop, which is the original per-edge gather.
        constexpr bool GROUPED_BY_DST  = (ll == GSDDMM_MEMBER::Dst_V) || (Plan::USE_R && rr == GSDDMM_MEMBER::Dst_V);
        constexpr GSDDMM_MEMBER SHARED = GROUPED_BY_DST ? GSDDMM_MEMBER::Dst_V : GSDDMM_MEMBER::Src_V;
        constexpr bool L_SHARED        = (ll == SHARED);
        constexpr bool R_SHARED        = Plan::USE_R && (rr == SHARED);
        static_assert(L_SHARED || R_SHARED, "every instantiated member pair has a node operand");

        vec_t shared_tiles[TILES_PER_THREAD];

        // edge_body(cache_c, k): op for edge k of the chunk; with cache_c true the
        // SHARED operand comes from shared_tiles instead of its own gathered row.
        auto edge_body = [L, R, O, edge_base, shared_tiles, my_pair, lane_id](auto cache_c, size_t k) {
            constexpr bool USE_CACHE = decltype(cache_c)::value;
            const uint64_t e         = edge_base + k;
            cuda_t const *l_row      = nullptr;
            cuda_t const *r_row      = nullptr;
            if constexpr (!(USE_CACHE && L_SHARED)) {
                l_row = gsddmm_edge_member_row<ll, D_CONST, cuda_t>(L, my_pair, k, e);
            }
            if constexpr (Plan::USE_R && !(USE_CACHE && R_SHARED)) {
                r_row = gsddmm_edge_member_row<rr, D_CONST, cuda_t>(R, my_pair, k, e);
            }

            cuda_t *const O_row = O + e * D_CONST;  // elementwise ops write the edge's own row
            accum_t partial{};
#pragma unroll
            for (size_t t = 0; t < TILES_PER_THREAD; ++t) {
                const size_t tile_idx = lane_id + kWarpSize * t;
                if (tile_idx < TILES) [[likely]] {
                    vec_t L_feats;
                    if constexpr (USE_CACHE && L_SHARED) {
                        L_feats = shared_tiles[t];
                    } else {
                        L_feats = Tile::read<Tile::MemoryHint::NoHint>(l_row, tile_idx);
                    }
                    if constexpr (op == GSDDMM_OP::Copy) {
                        Tile::write<Tile::MemoryHint::Streaming>(O_row, tile_idx, L_feats);
                    } else {
                        vec_t R_feats;
                        if constexpr (USE_CACHE && R_SHARED) {
                            R_feats = shared_tiles[t];
                        } else {
                            R_feats = Tile::read<Tile::MemoryHint::NoHint>(r_row, tile_idx);
                        }
                        if constexpr (Plan::IS_DOT) {
                            L_feats.dot_product_(&partial, R_feats);
                        } else {
                            Tile::write<Tile::MemoryHint::Streaming>(O_row, tile_idx, VecOp::apply(L_feats, R_feats));
                        }
                    }
                }
            }
            if constexpr (Plan::IS_DOT) {
                partial = warp_reduce_sum(partial);
                if (lane_id == 0) {
                    // Dot output is [E] (one scalar per edge), not an [E, D] row.
                    O[e] = static_cast<cuda_t>(partial);
                }
            }
        };

        // One ballot decides the chunk: lanes beyond the chunk vote "same".
        const uint64_t my_key   = GROUPED_BY_DST ? my_pair.y : my_pair.x;
        const uint64_t key0     = __shfl_sync(FULL_WARP_MASK, my_key, 0);
        const bool chunk_shared = __all_sync(FULL_WARP_MASK, (lane_id >= num_edges) || (my_key == key0)) != 0;
        if (chunk_shared) {
            cuda_t const *const row = (L_SHARED ? L : R) + D_CONST * key0;
#pragma unroll
            for (size_t t = 0; t < TILES_PER_THREAD; ++t) {
                const size_t tile_idx = lane_id + kWarpSize * t;
                if (tile_idx < TILES) [[likely]] {
                    shared_tiles[t] = Tile::read<Tile::MemoryHint::NoHint>(row, tile_idx);
                }
            }
            for (size_t k = 0; k < num_edges; ++k) {
                edge_body(std::true_type{}, k);
            }
        } else {
            for (size_t k = 0; k < num_edges; ++k) {
                edge_body(std::false_type{}, k);
            }
        }
    }
}

};  // namespace gsddmm
