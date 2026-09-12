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
                reinterpret_cast<chunk_t *>(dst) + i, reinterpret_cast<chunk_t const *>(src) + i, cuda::aligned_size_t<16>(sizeof(chunk_t)),
                pipe
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
__device__ __forceinline__ void gsddmm_pipelined_row_loop(
    size_t lane, size_t loop_iters, cuda_t *__restrict__ dbuf, AddrFn&& addr, ConsumeFn&& consume
) {
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
    auto consume_it = [edge_of, consume_ = std::move(consume)](size_t it, cuda_t const *const(&rows)[NUM_ROWS]) {
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
__device__ __forceinline__ cuda_t const *gsddmm_edge_member_row(
    cuda_t const *__restrict__ base, const ulonglong2& my_pair, size_t k, uint64_t e
) {
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
//
// Edge numbering: the traversal list may be grouped by SOURCE for locality (so a
// warp's chunk shares the Src_V row) while the caller numbers edges by their
// forward-CSR position. canonical_edge_idx bridges the two: slot k of the chunk
// carries canonical edge id canonical_edge_idx[edge_base + k], which indexes BOTH
// the Edge operand rows and the output row -- remapping only the store would pair
// an edge-indexed operand with the wrong edge. Lane k caches its slot's id with
// the same coalesced load pattern as the (src, dst) pair and broadcasts it with a
// shuffle. nullptr means the traversal order already IS the canonical order, and
// the id degenerates to the slot index (no load, no shuffle).
// =============================================================================
template <GSDDMM_OP op, GSDDMM_MEMBER ll, GSDDMM_MEMBER rr, size_t D_CONST, FloatingNum cuda_t, FloatingNum accum_t, size_t PIPELINE_STAGES>
__global__ void __launch_bounds__(kWarpSize *kGsddmmEdgeMaxWarpsPerBlock) GSDDMM_forward_edge_block( // no-format
    uint64_t E, // total edge count
    cuda_t const *__restrict__ L, cuda_t const *__restrict__ R, cuda_t *__restrict__ O,
    ulonglong2 const * __restrict__ edge_nodes_idx,
    unsigned long long const *__restrict__ canonical_edge_idx,
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
    constexpr size_t NUM_ROWS   = Plan::USE_R ? 2 : 1;
    constexpr size_t L_SLOT     = 0;
    constexpr size_t R_SLOT     = 1;
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

    // Lane k also caches the canonical edge id of slot k, same access pattern. The
    // pointer is grid-uniform, so the branch below never diverges and the shuffle
    // inside canonical_of stays convergent.
    const bool remap_edge_ids       = canonical_edge_idx != nullptr;
    unsigned long long my_canonical = 0;
    if (remap_edge_ids && lane_id < num_edges) {
        my_canonical = __ldcs(&canonical_edge_idx[edge_base + lane_id]);
    }

    // Canonical edge id of slot k of this chunk (warp-uniform result).
    auto canonical_of = [remap_edge_ids, my_canonical, edge_base](size_t k) -> uint64_t {
        return remap_edge_ids ? __shfl_sync(FULL_WARP_MASK, my_canonical, static_cast<int>(k)) : edge_base + k;
    };

    // Global row addresses of the operands for edge k of the chunk (warp-uniform).
    auto edge_rows = [canonical_of, my_pair, L, R](size_t k, cuda_t const *(&srcs)[NUM_ROWS]) {
        const uint64_t e = canonical_of(k);
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
    auto consume = [lane_id, canonical_of, O](size_t k, cuda_t const *const(&rows)[NUM_ROWS]) {
        const uint64_t e = canonical_of(k);
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
        // shared_tiles is captured BY REFERENCE: it is filled below, after this
        // closure is created, so a by-value capture would compute on the copy
        // taken while the array was still uninitialized.
        auto edge_body = [L, R, O, canonical_of, &shared_tiles, my_pair, lane_id](auto cache_c, size_t k) {
            constexpr bool USE_CACHE = decltype(cache_c)::value;
            const uint64_t e         = canonical_of(k);
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

// =============================================================================
// Backward: per-edge gradient contribution, on vector tiles.
//
// Returns the part of the partial that does NOT depend on the reduced operand's
// own row, so a whole node's edges can be summed before the row-dependent factor
// is applied once (see GsddmmBackwardPlan::NEEDS_SELF_ROW):
//
//     add / copy:            d_out
//     sub:                   d_out                (negated by the caller for rhs)
//     mul:                   d_out * other
//     div, lhs (numerator):  d_out / other
//     div, rhs (denominator): d_out * other       (* -1/self^2 applied per node)
//     dot:                   d_out_scalar * other (broadcast by the caller)
// =============================================================================
template <GSDDMM_OP op, bool SELF_IS_LHS, size_t TW, FloatingNum cuda_t>
struct GsddmmGradOp {
    using vec_t = VecFloat<TW, cuda_t>;

    // Add/Sub/Copy have constant partials; the rest scale d_out by the other row.
    static constexpr bool READS_OTHER = (op == GSDDMM_OP::Mul || op == GSDDMM_OP::Div || op == GSDDMM_OP::Dot);

    static constexpr __device__ __forceinline__ vec_t apply(vec_t d_out, vec_t other) {
        if constexpr (op == GSDDMM_OP::Add || op == GSDDMM_OP::Sub || op == GSDDMM_OP::Copy) {
            // Constant partial: the other operand is never read.
        } else if constexpr (op == GSDDMM_OP::Mul || op == GSDDMM_OP::Dot) {
            d_out.mul_(other);
        } else if constexpr (op == GSDDMM_OP::Div) {
            if constexpr (SELF_IS_LHS) {
                d_out.div_(other);  // dL = dO / R
            } else {
                d_out.mul_(other);  // dR = -dO * L / R^2; the -1/R^2 comes later
            }
        } else {
            static_assert(!sizeof(cuda_t), "GsddmmGradOp::apply reached an unhandled op");
            __builtin_unreachable();
        }
        return d_out;
    }
};

// acc[0, TW) += v, widened to accum_t. Mirrors tile.cuh's accumulate helpers: the
// array is stepped in chunks of min(TW, sizeof(accum_t)) elements so each chunk
// is one vector register rather than spilling a wide fp32 vector to local memory.
template <size_t TW, FloatingNum cuda_t, FloatingNum accum_t>
__device__ __forceinline__ void gsddmm_accum_add(accum_t *const __restrict__ acc, VecFloat<TW, cuda_t> v) {
    constexpr size_t compact_N  = std::min(TW, sizeof(accum_t));
    constexpr size_t repeat_cnt = TW / compact_N;
#pragma unroll
    for (size_t i = 0; i < repeat_cnt; ++i) {
        reinterpret_cast<VecFloat<compact_N, accum_t> *>(acc)[i] +=
            reinterpret_cast<VecFloat<compact_N, cuda_t> const *>(&v)[i].template convert_vec<accum_t>();
    }
}

// The reduced tile, converted back to the storage type.
template <size_t TW, FloatingNum cuda_t, FloatingNum accum_t>
__device__ __forceinline__ VecFloat<TW, cuda_t> gsddmm_accum_to_vec(accum_t const *const __restrict__ acc) {
    constexpr size_t compact_N  = std::min(TW, sizeof(accum_t));
    constexpr size_t repeat_cnt = TW / compact_N;
    VecFloat<TW, cuda_t> out;
#pragma unroll
    for (size_t i = 0; i < repeat_cnt; ++i) {
        reinterpret_cast<VecFloat<compact_N, cuda_t> *>(&out)[i] =
            reinterpret_cast<VecFloat<compact_N, accum_t> const *>(acc)[i].template convert_vec<cuda_t>();
    }
    return out;
}

// d_out tile for edge e: the edge's own row for elementwise ops, or its broadcast
// scalar for dot, whose output -- and so whose gradient -- is [E] rather than
// [E, D]. Every op's partial is linear in d_out, so broadcasting the scalar here
// lets one tile-shaped code path serve both.
template <bool IS_DOT, size_t D_CONST, size_t TW, FloatingNum cuda_t>
__device__ __forceinline__ VecFloat<TW, cuda_t> gsddmm_grad_out_tile(cuda_t const *__restrict__ dO, size_t edge_id, size_t tile_idx) {
    using Tile = TileOps<TW, cuda_t, float>;
    if constexpr (IS_DOT) {
        typename Tile::vec_t v;
        v.store_zero_();
        v.scalar_add_(dO[edge_id]);
        return v;
    } else {
        return Tile::template read<Tile::MemoryHint::Streaming>(dO + edge_id * D_CONST, tile_idx);
    }
}

// =============================================================================
// Node-parallel GSDDMM backward. See gsddmm.cuh for the partials and for which
// passes a given member pair needs.
//
// One block per (bucketed) node of the pass's CSR; the block's warps stride over
// that node's edges exactly as the forward does, and each lane keeps its feature
// tile's running sum in fp32 REGISTERS. The node's row is therefore written once,
// by one warp, with no atomics anywhere and no need for a zeroed output -- the
// reduction is a warp-level tree over the block's warps, not a memory-visible
// accumulation. That is what makes this the accurate variant; the price is the
// forward's load imbalance (a block per node, runtime set by the largest degree),
// which heavy-node chunking bounds via block_part.
// =============================================================================
template <GSDDMM_OP op, GSDDMM_MEMBER ll, GSDDMM_MEMBER rr, GSDDMM_REDUCE reduce, size_t N_PER_BLOCK, size_t D_CONST, FloatingNum cuda_t, IntegralNum index_t, FloatingNum accum_t>
__global__ void __launch_bounds__(N_PER_BLOCK *kWarpSize) GSDDMM_backward_normal( // no-format
    size_t N,
    cuda_t const *__restrict__ L, cuda_t const *__restrict__ R, cuda_t const *__restrict__ dO,
    cuda_t *__restrict__ d_self, cuda_t *__restrict__ d_edge,
    index_t const *__restrict__ row_ptr, index_t const *__restrict__ col_idx,
    index_t const *__restrict__ node_indices,
    unsigned long long const *__restrict__ canonical_edge_idx
) {
    static_assert(D_CONST % 32 == 0, "D_CONST must be a multiple of 32 so a warp covers the row an integral number of times");
    static_assert(std::popcount(D_CONST / 32) == 1, "D_CONST / 32 must be a power of two for the tile decomposition");

    using Plan = GsddmmBackwardPlan<op, ll, rr, reduce>;
    static_assert(Plan::HAS_SELF, "this pass has no operand to reduce; the host must not launch it");

    constexpr size_t TW = SelectTW<D_CONST, cuda_t>::value;
    static_assert(D_CONST % TW == 0, "Feature dim should be divisible by Tile width");
    constexpr size_t TILES            = D_CONST / TW;
    constexpr size_t TILES_PER_THREAD = ceil_div(TILES, kWarpSize);

    using Tile       = TileOps<TW, cuda_t, accum_t>;
    using vec_t      = typename Tile::vec_t;
    using GradOp     = GsddmmGradOp<op, Plan::SELF_IS_LHS, TW, cuda_t>;
    using EdgeGradOp = GsddmmGradOp<op, !Plan::SELF_IS_LHS, TW, cuda_t>;

    const size_t node_i = static_cast<size_t>(node_indices[blockIdx.x]);
    __builtin_assume(threadIdx.y < static_cast<unsigned>(N_PER_BLOCK));
    const size_t lane_id = threadIdx.x;
    __builtin_assume(lane_id < static_cast<size_t>(kWarpSize));
    const size_t warp_id = threadIdx.y;

    if (node_i >= N) [[unlikely]] {
        return;
    }

    const index_t edge_start = row_ptr[node_i];
    const index_t edge_end   = row_ptr[node_i + 1];
    const size_t num_edges   = static_cast<size_t>(edge_end - edge_start);

    cuda_t const *const self_base  = Plan::SELF_IS_LHS ? L : R;
    cuda_t const *const other_base = Plan::SELF_IS_LHS ? R : L;

    // The node's own row is constant over its edges, so each lane keeps its tiles
    // of it in registers: Div's -1/self^2 factor and the edge operand's gradient
    // both need it, and neither should re-read it per edge.
    constexpr bool KEEP_SELF_TILES = Plan::NEEDS_SELF_ROW || (Plan::WRITE_EDGE_GRAD && EdgeGradOp::READS_OTHER);
    vec_t self_tiles[TILES_PER_THREAD];
    if constexpr (KEEP_SELF_TILES) {
#pragma unroll
        for (size_t t = 0; t < TILES_PER_THREAD; ++t) {
            const size_t v = lane_id + kWarpSize * t;
            if (v < TILES) [[likely]] {
                self_tiles[t] = Tile::read(self_base + node_i * D_CONST, v);
            }
        }
    }

    // Per-lane fp32 running sums, one per feature tile this lane owns.
    accum_t acc[TILES_PER_THREAD][TW];
#pragma unroll
    for (size_t t = 0; t < TILES_PER_THREAD; ++t) {
#pragma unroll
        for (size_t j = 0; j < TW; ++j) {
            acc[t][j] = accum_t{0};
        }
    }

    // Warp-strided walk of the node's edges, mirroring the forward's loop so the
    // trip count is exact and the body can be unrolled.
    const size_t loop_iters = (num_edges > warp_id) ? ceil_div(num_edges - warp_id, N_PER_BLOCK) : 0;
#pragma unroll 2
    for (size_t it = 0; it < loop_iters; ++it) {
        const index_t slot = edge_start + static_cast<index_t>(warp_id + it * N_PER_BLOCK);
        // Walking the backward CSR visits edges in CSC order, while dO and the
        // Edge operand are numbered by forward-CSR position.
        const size_t edge_id   = (canonical_edge_idx != nullptr) ? static_cast<size_t>(canonical_edge_idx[slot]) : static_cast<size_t>(slot);
        const size_t other_row = Plan::OTHER_IS_EDGE ? edge_id : static_cast<size_t>(col_idx[slot]);

#pragma unroll
        for (size_t t = 0; t < TILES_PER_THREAD; ++t) {
            const size_t v = lane_id + kWarpSize * t;
            if (v < TILES) [[likely]] {
                const vec_t d_out = gsddmm_grad_out_tile<Plan::IS_DOT, D_CONST, TW, cuda_t>(dO, edge_id, v);
                // Add/Sub/Copy have constant partials and never read this; it is
                // still zeroed rather than left undefined, both to keep the
                // value passed to apply() defined and to keep the build quiet.
                vec_t other;
                if constexpr (Plan::READS_OTHER) {
                    other = Tile::read(other_base + other_row * D_CONST, v);
                } else {
                    other.store_zero_();
                }
                gsddmm_accum_add<TW, cuda_t, accum_t>(acc[t], GradOp::apply(d_out, other));

                // The edge operand owns one row per edge, so its gradient is a
                // plain store: the node row (in registers) is its "other" operand.
                if constexpr (Plan::WRITE_EDGE_GRAD) {
                    vec_t node_row;
                    if constexpr (EdgeGradOp::READS_OTHER) {
                        node_row = self_tiles[t];
                    } else {
                        // Add/Sub/Copy ignore it; keep it defined rather than
                        // feeding an uninitialized register to apply().
                        node_row.store_zero_();
                    }
                    vec_t g_edge = EdgeGradOp::apply(d_out, node_row);
                    if constexpr (op == GSDDMM_OP::Div && Plan::OTHER == rr) {
                        // The edge operand is the denominator: -dO * self / edge^2.
                        g_edge.div_(other);
                        g_edge.div_(other);
                        g_edge.neg_();
                    } else if constexpr (op == GSDDMM_OP::Sub && Plan::OTHER == rr) {
                        g_edge.neg_();
                    }
                    Tile::template write<Tile::MemoryHint::Streaming>(d_edge + edge_id * D_CONST, v, g_edge);
                }
            }
        }
    }

    // Reduce the block's warps: every warp parks its partial row in shared memory
    // (one fp32 row per warp, the GT backward's layout) and warp 0 sums them.
    // Lanes own disjoint features throughout, so no lane-wise exchange is needed.
    if constexpr (N_PER_BLOCK > 1) {
        extern __shared__ __align__(16) uint8_t sh_raw[];
        accum_t *const warp_acc = reinterpret_cast<accum_t *>(sh_raw);
#pragma unroll
        for (size_t t = 0; t < TILES_PER_THREAD; ++t) {
            const size_t v = lane_id + kWarpSize * t;
            if (v < TILES) [[likely]] {
#pragma unroll
                for (size_t j = 0; j < TW; ++j) {
                    warp_acc[warp_id * D_CONST + v * TW + j] = acc[t][j];
                }
            }
        }
        __syncthreads();
        if (warp_id != 0) {
            return;
        }
#pragma unroll
        for (size_t t = 0; t < TILES_PER_THREAD; ++t) {
            const size_t v = lane_id + kWarpSize * t;
            if (v < TILES) [[likely]] {
                for (size_t w = 1; w < N_PER_BLOCK; ++w) {
#pragma unroll
                    for (size_t j = 0; j < TW; ++j) {
                        acc[t][j] += warp_acc[w * D_CONST + v * TW + j];
                    }
                }
            }
        }
    }

#pragma unroll
    for (size_t t = 0; t < TILES_PER_THREAD; ++t) {
        const size_t v = lane_id + kWarpSize * t;
        if (v < TILES) [[likely]] {
            vec_t out = gsddmm_accum_to_vec<TW, cuda_t, accum_t>(acc[t]);
            if constexpr (Plan::NEEDS_SELF_ROW) {
                // -1/self^2: constant over the sum, so applied once, here.
                out.div_(self_tiles[t]);
                out.div_(self_tiles[t]);
            }
            if constexpr (Plan::NEGATE_SELF) {
                out.neg_();
            }
            Tile::template write<Tile::MemoryHint::Streaming>(d_self + node_i * D_CONST, v, out);
        }
    }
}

// =============================================================================
// Edge-parallel GSDDMM backward. Work distribution is the forward's: warp w owns
// the contiguous chunk [w * edges_per_warp, (w + 1) * edges_per_warp) of the
// explicit edge list, lane k caching edge k's (src, dst) pair and canonical id
// from one coalesced load.
//
// The node gradient is a reduction that crosses warps here, so it is accumulated
// with atomicAdd into an fp32 buffer -- fp16/bf16 atomics would both serialize on
// the same address and lose the sum's precision, and the repo already takes the
// "fp32 buffer, cast on the host" route for the GT backward's dK.
//
// The list is grouped by the node being reduced, so a warp's chunk usually shares
// its target row. One ballot (the forward's shared-row trick, applied to the
// accumulation side instead of the load side) detects that, and the warp then
// sums its whole chunk in registers and issues ONE atomicAdd per feature tile
// instead of one per edge -- for a 32-edge chunk that is a 32x cut in atomic
// traffic on exactly the high-degree nodes where contention would otherwise bite.
// Mixed chunks fall back to a per-edge atomic.
// =============================================================================
template <GSDDMM_OP op, GSDDMM_MEMBER ll, GSDDMM_MEMBER rr, GSDDMM_REDUCE reduce, size_t D_CONST, FloatingNum cuda_t, FloatingNum accum_t>
__global__ void __launch_bounds__(kWarpSize *kGsddmmEdgeMaxWarpsPerBlock) GSDDMM_backward_edge_block( // no-format
    uint64_t E,
    cuda_t const *__restrict__ L, cuda_t const *__restrict__ R, cuda_t const *__restrict__ dO,
    accum_t *__restrict__ d_self_f32, cuda_t *__restrict__ d_edge,
    ulonglong2 const *__restrict__ edge_nodes_idx,
    unsigned long long const *__restrict__ canonical_edge_idx,
    uint32_t edges_per_warp
) {
    static_assert(D_CONST % 32 == 0, "D_CONST must be a multiple of 32 so a warp covers the row an integral number of times");
    static_assert(std::popcount(D_CONST / 32) == 1, "D_CONST / 32 must be a power of two for the tile decomposition");

    using Plan = GsddmmBackwardPlan<op, ll, rr, reduce>;
    static_assert(Plan::HAS_SELF, "this pass has no operand to reduce; the host must not launch it");

    constexpr size_t TW               = SelectTW<D_CONST, cuda_t>::value;
    constexpr size_t TILES            = D_CONST / TW;
    constexpr size_t TILES_PER_THREAD = ceil_div(TILES, kWarpSize);

    using Tile       = TileOps<TW, cuda_t, accum_t>;
    using vec_t      = typename Tile::vec_t;
    using GradOp     = GsddmmGradOp<op, Plan::SELF_IS_LHS, TW, cuda_t>;
    using EdgeGradOp = GsddmmGradOp<op, !Plan::SELF_IS_LHS, TW, cuda_t>;

    __builtin_assume(threadIdx.x < static_cast<unsigned>(kWarpSize));
    __builtin_assume(threadIdx.y < static_cast<unsigned>(kGsddmmEdgeMaxWarpsPerBlock));
    __builtin_assume(edges_per_warp >= 1 && edges_per_warp <= kGsddmmEdgeMaxEdgesPerWarp);
    const size_t lane_id = threadIdx.x;
    const size_t warp_id = threadIdx.y;

    const uint64_t block_linear = (static_cast<uint64_t>(blockIdx.z) * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
    const uint64_t global_warp  = block_linear * blockDim.y + warp_id;
    const uint64_t edge_base    = global_warp * edges_per_warp;
    if (edge_base >= E) [[unlikely]] {
        return;
    }
    const size_t num_edges = static_cast<size_t>((E - edge_base < edges_per_warp) ? (E - edge_base) : edges_per_warp);

    // Lane k caches edge k's endpoints and canonical id (see the forward kernel).
    ulonglong2 my_pair{};
    if (lane_id < num_edges) {
        my_pair = __ldcs(&edge_nodes_idx[edge_base + lane_id]);
    }
    const bool remap_edge_ids       = canonical_edge_idx != nullptr;
    unsigned long long my_canonical = 0;
    if (remap_edge_ids && lane_id < num_edges) {
        my_canonical = __ldcs(&canonical_edge_idx[edge_base + lane_id]);
    }
    auto canonical_of = [remap_edge_ids, my_canonical, edge_base](size_t k) -> uint64_t {
        return remap_edge_ids ? __shfl_sync(FULL_WARP_MASK, my_canonical, static_cast<int>(k)) : edge_base + k;
    };
    // Node this pass reduces into, and the node the other operand is read from.
    auto self_node_of = [my_pair](size_t k) -> uint64_t {
        return __shfl_sync(FULL_WARP_MASK, reduce == GSDDMM_REDUCE::Dst ? my_pair.y : my_pair.x, static_cast<int>(k));
    };
    auto other_node_of = [my_pair](size_t k) -> uint64_t {
        return __shfl_sync(FULL_WARP_MASK, reduce == GSDDMM_REDUCE::Dst ? my_pair.x : my_pair.y, static_cast<int>(k));
    };

    // One ballot for the chunk: lanes past the end vote "same". The list is
    // grouped by the reduced node, so this is the common case.
    const uint64_t my_key   = (reduce == GSDDMM_REDUCE::Dst) ? my_pair.y : my_pair.x;
    const uint64_t key0     = __shfl_sync(FULL_WARP_MASK, my_key, 0);
    const bool chunk_shared = __all_sync(FULL_WARP_MASK, (lane_id >= num_edges) || (my_key == key0)) != 0;

    // edge_body(k, acc): the per-edge partial for edge k of the chunk, added into
    // acc (fp32). Also stores the edge operand's gradient, which needs no sum.
    auto edge_body = [&](size_t k, accum_t acc[TILES_PER_THREAD][TW]) {
        const uint64_t edge_id   = canonical_of(k);
        const uint64_t self_node = self_node_of(k);
        const uint64_t other_row = Plan::OTHER_IS_EDGE ? edge_id : other_node_of(k);

#pragma unroll
        for (size_t t = 0; t < TILES_PER_THREAD; ++t) {
            const size_t v = lane_id + kWarpSize * t;
            if (v < TILES) [[likely]] {
                const vec_t d_out = gsddmm_grad_out_tile<Plan::IS_DOT, D_CONST, TW, cuda_t>(dO, edge_id, v);
                vec_t other;
                if constexpr (Plan::READS_OTHER) {
                    other = Tile::read((Plan::SELF_IS_LHS ? R : L) + other_row * D_CONST, v);
                } else {
                    other.store_zero_();
                }

                vec_t contrib = GradOp::apply(d_out, other);
                if constexpr (Plan::NEEDS_SELF_ROW) {
                    // Div's denominator: -1/self^2. Unlike the node-parallel
                    // variant this warp does not own the whole node, so the
                    // factor cannot be deferred past the atomic and is applied
                    // per edge from the (broadcast) self row.
                    const vec_t self_row = Tile::read((Plan::SELF_IS_LHS ? L : R) + self_node * D_CONST, v);
                    contrib.div_(self_row);
                    contrib.div_(self_row);
                }
                if constexpr (Plan::NEGATE_SELF) {
                    contrib.neg_();
                }
                gsddmm_accum_add<TW, cuda_t, accum_t>(acc[t], contrib);

                if constexpr (Plan::WRITE_EDGE_GRAD) {
                    // Only read the node row when the op's partial uses it: an
                    // operand the backward never touches may be a 1-row stand-in.
                    vec_t node_row;
                    if constexpr (EdgeGradOp::READS_OTHER) {
                        node_row = Tile::read((Plan::SELF_IS_LHS ? L : R) + self_node * D_CONST, v);
                    } else {
                        node_row.store_zero_();
                    }
                    vec_t g_edge = EdgeGradOp::apply(d_out, node_row);
                    if constexpr (op == GSDDMM_OP::Div && Plan::OTHER == rr) {
                        g_edge.div_(other);
                        g_edge.div_(other);
                        g_edge.neg_();
                    } else if constexpr (op == GSDDMM_OP::Sub && Plan::OTHER == rr) {
                        g_edge.neg_();
                    }
                    Tile::template write<Tile::MemoryHint::Streaming>(d_edge + edge_id * D_CONST, v, g_edge);
                }
            }
        }
    };

    accum_t acc[TILES_PER_THREAD][TW];
    auto clear_acc = [&acc]() {
#pragma unroll
        for (size_t t = 0; t < TILES_PER_THREAD; ++t) {
#pragma unroll
            for (size_t j = 0; j < TW; ++j) {
                acc[t][j] = accum_t{0};
            }
        }
    };
    // flush(node): atomically add this warp's accumulated tiles into node's row.
    auto flush_acc = [&](uint64_t node) {
#pragma unroll
        for (size_t t = 0; t < TILES_PER_THREAD; ++t) {
            const size_t v = lane_id + kWarpSize * t;
            if (v < TILES) [[likely]] {
#pragma unroll
                for (size_t j = 0; j < TW; ++j) {
                    atomicAdd(&d_self_f32[node * D_CONST + v * TW + j], acc[t][j]);
                }
            }
        }
    };

    if (chunk_shared) {
        // The whole chunk targets key0: sum it in registers, one atomic per tile.
        clear_acc();
        for (size_t k = 0; k < num_edges; ++k) {
            edge_body(k, acc);
        }
        flush_acc(key0);
    } else {
        for (size_t k = 0; k < num_edges; ++k) {
            clear_acc();
            edge_body(k, acc);
            flush_acc(self_node_of(k));
        }
    }
}

}  // namespace gsddmm
