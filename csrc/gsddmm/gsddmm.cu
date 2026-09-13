#include <bit>
#include <cstdint>
#include <type_traits>

#include "common/misc.cuh"
#include "common/pipeline.cuh"
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

// Tile decomposition shared by all four GSDDMM kernels: the warp-per-row
// validity checks, the tile width (SelectTW), and the derived trip counts.
template <size_t D_CONST, FloatingNum cuda_t, FloatingNum accum_t>
struct GsddmmRowShape {
    static_assert(D_CONST % 32 == 0, "D_CONST must be a multiple of 32 so a warp covers the row an integral number of times");
    static_assert(std::popcount(D_CONST / 32) == 1, "D_CONST / 32 must be a power of two for the tile decomposition");
    static_assert((D_CONST * sizeof(cuda_t)) % 16 == 0, "Row width in bytes must be a multiple of 16 for wide copies");
    static constexpr size_t TW = SelectTW<D_CONST, cuda_t>::value;
    static_assert(D_CONST % TW == 0, "Feature dim should be divisible by Tile width");
    static constexpr size_t TILES            = D_CONST / TW;
    static constexpr size_t TILES_PER_THREAD = ceil_div(TILES, kWarpSize);
    using Tile  = TileOps<TW, cuda_t, accum_t>;
    using vec_t = typename Tile::vec_t;
};

// CSR-row adapter over pipelined_row_loop (common/pipeline.cuh) for
// GSDDMM_forward_normal: warp-strided walk over the edge list of one CSR row,
// Src_V rows indexed by the neighbor id j = col_idx[e], Edge rows indexed by
// the edge position e. That per-operand choice is what the generic loop's
// addr() callback exists for -- the common CSR adapter
// (pipelined_neighbor_row_loop) can only index rows by neighbor id.
// consume(e, rows) receives the edge position.
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
    const uint32_t loop_iters = static_cast<uint32_t>((num_edges > warp_id) ? ceil_div(num_edges - warp_id, N_PER_BLOCK) : size_t{0});

    auto edge_of = [edge_start, warp_id](uint32_t it) {
        // 32-bit slot arithmetic (see pipelined_row_loop); only the final add is
        // done in index_t.
        const uint32_t k = static_cast<uint32_t>(warp_id) + it * static_cast<uint32_t>(N_PER_BLOCK);
        return edge_start + static_cast<index_t>(k);
    };

    auto addr = [edge_of, row_bases, col_idx, row_edge_indexed](uint32_t it, cuda_t const *(&srcs)[NUM_ROWS]) {
        const index_t e = edge_of(it);
        const index_t j = col_idx[e];
#pragma unroll
        for (size_t r = 0; r < NUM_ROWS; ++r) {
            const size_t row_id = row_edge_indexed[r] ? static_cast<size_t>(e) : static_cast<size_t>(j);
            srcs[r]             = row_bases[r] + row_id * D_CONST;
        }
    };
    auto consume_it = [edge_of, &consume](uint32_t it, cuda_t const *const (&rows)[NUM_ROWS]) {
        consume(static_cast<size_t>(edge_of(it)), rows);
    };

    pipelined_row_loop<D_CONST, NUM_STAGES, NUM_ROWS, cuda_t>(lane, loop_iters, dbuf, addr, consume_it);
}

// =============================================================================
// GSDDMM forward kernel. See gsddmm.cuh for semantics and conventions.
// =============================================================================
template <GSDDMM_OP op, GSDDMM_MEMBER ll, GSDDMM_MEMBER rr, size_t N_PER_BLOCK, size_t D_CONST, FloatingNum cuda_t, IntegralNum index_t, FloatingNum accum_t, uint8_t PIPELINE_STAGES>
__global__ void __launch_bounds__(N_PER_BLOCK *kWarpSize) GSDDMM_forward_normal( // no-format
    size_t N,
    cuda_t const *__restrict__ L, cuda_t const *__restrict__ R, cuda_t *__restrict__ O,
    index_t const *__restrict__ row_ptr, index_t const *__restrict__ col_idx,
    index_t const *__restrict__ node_indices,
    index_t const *__restrict__ block_part, uint32_t edges_per_block
) {
    using Plan  = GsddmmPlan<op, ll, rr>;
    using Shape = GsddmmRowShape<D_CONST, cuda_t, accum_t>;

    constexpr size_t TW               = Shape::TW;
    constexpr size_t TILES            = Shape::TILES;
    constexpr size_t TILES_PER_THREAD = Shape::TILES_PER_THREAD;

    using Tile  = Shape::Tile;
    using vec_t = Shape::vec_t;
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
    //   edge_dbuf[N_PER_BLOCK * pipelined_ring_elems(NUM_STAGES, NUM_EDGE_ROWS, D_CONST)] as cuda_t
    //       -- per-warp cp.async ring for gathered Src_V/Edge rows, only when USE_PIPELINE
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

    cuda_t *const my_dbuf = edge_dbuf + warp_id * pipelined_ring_elems(NUM_STAGES, Plan::NUM_EDGE_ROWS, D_CONST);

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
        if constexpr (Plan::R_EDGE_ROW) {
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
        // Counter left as size_t, unlike pipelined_row_loop's 32-bit trip count
        // and unlike the edge kernels' chunk counters. Narrowing it here was
        // measured to make things WORSE: it frees just enough register pressure
        // for ptxas to drop from 34-40 registers into the 32-register bucket
        // (100% occupancy for a 1024-thread block), which it then pays a 4-16
        // byte spill to stay in -- 9 of the 10 D=256 configs that spill at all
        // stop spilling when this is a size_t. The occupancy trade may well be
        // worth taking, but it belongs in __launch_bounds__'s second argument
        // with a timing run behind it, not as a side effect of an index width.
        const size_t loop_iters = (num_edges > warp_id) ? ceil_div(num_edges - warp_id, N_PER_BLOCK) : 0;
#pragma unroll 2
        for (size_t it = 0; it < loop_iters; ++it) {
            const index_t e = edge_start + static_cast<index_t>(warp_id + it * N_PER_BLOCK);
            const index_t j = col_idx[e];

            cuda_t const *rows[ROWS_CAP];
            if constexpr (!Plan::L_DST) {
                rows[Plan::L_SLOT] = L + (Plan::L_EDGE_INDEXED ? static_cast<size_t>(e) : static_cast<size_t>(j)) * D_CONST;
            }
            if constexpr (Plan::R_EDGE_ROW) {
                rows[Plan::R_SLOT] = R + (Plan::R_EDGE_INDEXED ? static_cast<size_t>(e) : static_cast<size_t>(j)) * D_CONST;
            }
            consume(static_cast<size_t>(e), rows);
        }
    }
}

// Row of operand `m` for edge k of this warp's chunk. Node ids come from the
// (src, dst) pair lane k cached in my_pair (warp-wide shuffle broadcast, so all
// lanes must reach this together); Edge rows are indexed by the edge position.
//
// The shuffle moves an index_t, so a 32-bit graph pays one SHFL where a 64-bit
// one pays two. The row OFFSET is always computed in size_t: D_CONST is a
// size_t, so the multiply widens before it can overflow a 32-bit id.
template <GSDDMM_MEMBER m, size_t D_CONST, FloatingNum cuda_t, IntegralNum index_t>
__device__ __forceinline__ cuda_t const *gsddmm_edge_member_row(
    cuda_t const *__restrict__ base, const index_pair_t<index_t>& my_pair, uint32_t k, std::make_unsigned_t<index_t> e
) {
    if constexpr (m == GSDDMM_MEMBER::Src_V) {
        return base + D_CONST * static_cast<size_t>(__shfl_sync(FULL_WARP_MASK, my_pair.x, static_cast<int>(k)));
    } else if constexpr (m == GSDDMM_MEMBER::Dst_V) {
        return base + D_CONST * static_cast<size_t>(__shfl_sync(FULL_WARP_MASK, my_pair.y, static_cast<int>(k)));
    } else if constexpr (m == GSDDMM_MEMBER::Edge) {
        return base + D_CONST * static_cast<size_t>(e);
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
template <GSDDMM_OP op, GSDDMM_MEMBER ll, GSDDMM_MEMBER rr, size_t D_CONST, FloatingNum cuda_t, IntegralNum index_t, FloatingNum accum_t, uint8_t PIPELINE_STAGES>
__global__ void __launch_bounds__(kWarpSize *kGsddmmEdgeMaxWarpsPerBlock) GSDDMM_forward_edge_block( // no-format
    index_t E, // total edge count
    cuda_t const *__restrict__ L, cuda_t const *__restrict__ R, cuda_t *__restrict__ O,
    index_pair_t<index_t> const * __restrict__ edge_nodes_idx,
    index_t const *__restrict__ canonical_edge_idx,
    uint32_t edges_per_warp
) {
    // Node and edge ids are non-negative, so they ride in index_t's unsigned form:
    // that is what index_pair_t already holds, and it keeps the id arithmetic one
    // width instead of promoting a signed index_t to 64 bits at every step.
    using eidx_t = std::make_unsigned_t<index_t>;
    using Plan   = GsddmmPlan<op, ll, rr>;
    using Shape = GsddmmRowShape<D_CONST, cuda_t, accum_t>;

    constexpr size_t TW               = Shape::TW;
    constexpr size_t TILES            = Shape::TILES;
    constexpr size_t TILES_PER_THREAD = Shape::TILES_PER_THREAD;

    using Tile  = Shape::Tile;
    using vec_t = Shape::vec_t;
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

    // The grid is sized to ceil(E / edges_per_warp) warps, so a warp index fits
    // index_t whenever E does -- the whole chunk walk is therefore index_t-wide,
    // not unconditionally 64-bit.
    const eidx_t block_linear = static_cast<eidx_t>((static_cast<uint64_t>(blockIdx.z) * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x);
    const eidx_t global_warp  = block_linear * static_cast<eidx_t>(blockDim.y) + static_cast<eidx_t>(warp_id);
    const eidx_t edge_base    = global_warp * static_cast<eidx_t>(edges_per_warp);
    // Warp-uniform exit: there is no block-level synchronization in this kernel.
    const eidx_t num_total = static_cast<eidx_t>(E);
    if (edge_base >= num_total) [[unlikely]] {
        return;
    }
    // At most kGsddmmEdgeMaxEdgesPerWarp (32), so the chunk-local counter and
    // every loop over it are 32-bit no matter how wide index_t is -- the rule
    // pipelined_row_loop states for its own trip count, applied here.
    const uint32_t num_edges = static_cast<uint32_t>((num_total - edge_base < edges_per_warp) ? (num_total - edge_base) : edges_per_warp);

    // Lane k caches the (src, dst) pair of edge edge_base + k. The chunk is unique
    // to this warp, so the load is streamed past L1.
    index_pair_t<index_t> my_pair{};
    if (lane_id < num_edges) {
        my_pair = __ldcs(&edge_nodes_idx[edge_base + lane_id]);
    }

    // Lane k also caches the canonical edge id of slot k, same access pattern. The
    // pointer is grid-uniform, so the branch below never diverges and the shuffle
    // inside canonical_of stays convergent.
    const bool remap_edge_ids = canonical_edge_idx != nullptr;
    eidx_t my_canonical       = 0;
    if (remap_edge_ids && lane_id < num_edges) {
        my_canonical = static_cast<eidx_t>(__ldcs(&canonical_edge_idx[edge_base + lane_id]));
    }

    // Canonical edge id of slot k of this chunk (warp-uniform result).
    auto canonical_of = [remap_edge_ids, my_canonical, edge_base](uint32_t k) -> eidx_t {
        return remap_edge_ids ? __shfl_sync(FULL_WARP_MASK, my_canonical, static_cast<int>(k)) : edge_base + static_cast<eidx_t>(k);
    };

    // Global row addresses of the operands for edge k of the chunk (warp-uniform).
    auto edge_rows = [canonical_of, my_pair, L, R](uint32_t k, cuda_t const *(&srcs)[NUM_ROWS]) {
        const eidx_t e = canonical_of(k);
        srcs[L_SLOT]   = gsddmm_edge_member_row<ll, D_CONST, cuda_t, index_t>(L, my_pair, k, e);
        if constexpr (Plan::USE_R) {
            srcs[R_SLOT] = gsddmm_edge_member_row<rr, D_CONST, cuda_t, index_t>(R, my_pair, k, e);
        }
    };

    // consume(k, rows): apply the op for edge k of the chunk from the prefetched
    // shared-memory rows (pipelined path only; the direct path below keeps its
    // own loop so it can cache the shared node row in registers).
    // O is written once and never read back here, so the stores are streaming
    // (evict-first): the [E, D] output must not push the gathered node rows,
    // which ARE re-read by other edges, out of L2.
    auto consume = [lane_id, canonical_of, O](uint32_t k, cuda_t const *const(&rows)[NUM_ROWS]) {
        const eidx_t e = canonical_of(k);
        if constexpr (!Plan::IS_DOT) {
            cuda_t *const O_row = O + static_cast<size_t>(e) * D_CONST;  // elementwise ops write the edge's own row
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
        // Per-warp cp.async ring of NUM_ROWS rows per slot (see
        // gsddmm_forward_edge_shmem_bytes_per_warp); warps are laid out back to back.
        extern __shared__ __align__(16) uint8_t sh_raw[];
        cuda_t *const my_dbuf = reinterpret_cast<cuda_t *>(sh_raw) + warp_id * pipelined_ring_elems(NUM_STAGES, NUM_ROWS, D_CONST);
        pipelined_row_loop<D_CONST, NUM_STAGES, NUM_ROWS, cuda_t>(lane_id, num_edges, my_dbuf, edge_rows, consume);
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
        auto edge_body = [L, R, O, canonical_of, &shared_tiles, my_pair, lane_id](auto cache_c, uint32_t k) {
            constexpr bool USE_CACHE = decltype(cache_c)::value;
            const eidx_t e           = canonical_of(k);
            cuda_t const *l_row      = nullptr;
            cuda_t const *r_row      = nullptr;
            if constexpr (!(USE_CACHE && L_SHARED)) {
                l_row = gsddmm_edge_member_row<ll, D_CONST, cuda_t, index_t>(L, my_pair, k, e);
            }
            if constexpr (Plan::USE_R && !(USE_CACHE && R_SHARED)) {
                r_row = gsddmm_edge_member_row<rr, D_CONST, cuda_t, index_t>(R, my_pair, k, e);
            }

            cuda_t *const O_row = O + static_cast<size_t>(e) * D_CONST;  // elementwise ops write the edge's own row
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
        const eidx_t my_key     = GROUPED_BY_DST ? my_pair.y : my_pair.x;
        const eidx_t key0       = __shfl_sync(FULL_WARP_MASK, my_key, 0);
        const bool chunk_shared = __all_sync(FULL_WARP_MASK, (lane_id >= num_edges) || (my_key == key0)) != 0;
        if (chunk_shared) {
            cuda_t const *const row = (L_SHARED ? L : R) + D_CONST * static_cast<size_t>(key0);
#pragma unroll
            for (size_t t = 0; t < TILES_PER_THREAD; ++t) {
                const size_t tile_idx = lane_id + kWarpSize * t;
                if (tile_idx < TILES) [[likely]] {
                    shared_tiles[t] = Tile::read<Tile::MemoryHint::NoHint>(row, tile_idx);
                }
            }
            for (uint32_t k = 0; k < num_edges; ++k) {
                edge_body(std::true_type{}, k);
            }
        } else {
            for (uint32_t k = 0; k < num_edges; ++k) {
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

// The tile-INDEPENDENT part of d_out for edge e: dot's output -- and so its
// gradient -- is [E] rather than [E, D], and every op's partial is linear in
// d_out, so broadcasting that one scalar lets a single tile-shaped code path
// serve both shapes.
//
// Split out of the per-tile read on purpose: the splat is the same for every
// tile of the edge, so building it inside the tile loop re-loaded dO[edge_id]
// and rebuilt the vector TILES_PER_THREAD times per edge. Callers hoist this
// above the loop and read gsddmm_grad_out_row_tile per tile instead. Returns a
// defined (zero) vector for the elementwise ops, which never read it -- the
// store is dead once the op is known and is eliminated.
//
// The broadcast is Vec's scalar constructor (reached through C++20 parenthesized
// aggregate initialization, as gsddmm_inv_sq_chunk and the GATv2 helpers already
// do). Measured identical, instruction for instruction and register for
// register, to a hand-written packed splat -- and 32 instructions cheaper across
// six Dot kernels than store_zero_() followed by scalar_add_(), which pays an
// add per element to reach a value already known.
template <bool IS_DOT, size_t TW, FloatingNum cuda_t>
__device__ __forceinline__ VecFloat<TW, cuda_t> gsddmm_grad_out_splat(cuda_t const *__restrict__ dO, size_t edge_id) {
    if constexpr (IS_DOT) {
        return VecFloat<TW, cuda_t>(dO[edge_id]);
    } else {
        VecFloat<TW, cuda_t> v;
        return v;
    }
}

// One tile of an elementwise op's d_out row. `hint` is Streaming when the row is
// read straight from global (each edge's dO row is read exactly once, so it must
// not evict the node rows that ARE reused) and NoHint when a cp.async pipeline
// has already staged it into shared memory, where a global cache policy would be
// meaningless.
template <size_t TW, FloatingNum cuda_t, TileOps<TW, cuda_t, float>::MemoryHint hint>
__device__ __forceinline__ VecFloat<TW, cuda_t> gsddmm_grad_out_row_tile(cuda_t const *__restrict__ dO_row, size_t tile_idx) {
    using Tile = TileOps<TW, cuda_t, float>;
    return Tile::template read<hint>(dO_row, tile_idx);
}

// 1/v^2 for one chunk, evaluated in accum_t. The reciprocal is taken once and
// squared rather than dividing twice: the approximate reciprocal is a
// quarter-rate unit on sm_80 (16 results/clk/SM against 64 for an fp32
// multiply), so halving the special-function work is the win, and the two fp32
// multiplies replacing the second division run at full rate. accum_t is what
// keeps the multiplier finite -- 1/v^2 leaves fp16's range already for
// |v| < 2^-8 -- and it costs nothing to stay wide here, since fp16 has no
// divide unit: __h2div widens to fp32 and narrows back anyway.
template <size_t N, FloatingNum cuda_t, FloatingNum accum_t>
__device__ __forceinline__ VecFloat<N, accum_t> gsddmm_inv_sq_chunk(VecFloat<N, cuda_t> v) {
    VecFloat<N, accum_t> r(accum_t(1));
    r.div_(v.template convert_vec<accum_t>());
    r.mul_(r);
    return r;
}

// x / v^2 for the edge gradient, whose destination is a cuda_t tensor: one
// narrowing is unavoidable, so it is done last, on the result, never on the
// 1/v^2 multiplier. Chunked like the accumulate helpers below so that no wide
// accum_t vector is materialized into local memory.
template <size_t TW, FloatingNum cuda_t, FloatingNum accum_t>
__device__ __forceinline__ VecFloat<TW, cuda_t> gsddmm_div_sq(VecFloat<TW, cuda_t> x, VecFloat<TW, cuda_t> v) {
    constexpr size_t compact_N  = std::min(TW, sizeof(accum_t));
    constexpr size_t repeat_cnt = TW / compact_N;
    VecFloat<TW, cuda_t> out;
#pragma unroll
    for (size_t i = 0; i < repeat_cnt; ++i) {
        VecFloat<compact_N, accum_t> q =
            gsddmm_inv_sq_chunk<compact_N, cuda_t, accum_t>(reinterpret_cast<VecFloat<compact_N, cuda_t> const *>(&v)[i]);
        q.mul_(reinterpret_cast<VecFloat<compact_N, cuda_t> const *>(&x)[i].template convert_vec<accum_t>());
        reinterpret_cast<VecFloat<compact_N, cuda_t> *>(&out)[i] = q.template convert_vec<cuda_t>();
    }
    return out;
}

// acc = -acc / v^2, in place. The node-parallel backward owns the whole node, so
// the factor is applied once to the finished fp32 sum -- and the free place for
// it is before gsddmm_accum_to_vec: that narrowing already exists, and keeping
// the scaling on its accum_t side means the multiplier never has to fit cuda_t.
template <size_t TW, FloatingNum cuda_t, FloatingNum accum_t>
__device__ __forceinline__ void gsddmm_accum_neg_div_sq(accum_t *const __restrict__ acc, VecFloat<TW, cuda_t> v) {
    constexpr size_t compact_N  = std::min(TW, sizeof(accum_t));
    constexpr size_t repeat_cnt = TW / compact_N;
#pragma unroll
    for (size_t i = 0; i < repeat_cnt; ++i) {
        VecFloat<compact_N, accum_t> q =
            gsddmm_inv_sq_chunk<compact_N, cuda_t, accum_t>(reinterpret_cast<VecFloat<compact_N, cuda_t> const *>(&v)[i]);
        q.neg_();
        reinterpret_cast<VecFloat<compact_N, accum_t> *>(acc)[i].mul_(q);
    }
}

// acc += a * b with both operands widened first, so the product never rounds to
// the storage type. Only Div's reduced partial needs this: the -1/self^2 applied
// afterwards amplifies whatever the product lost, and once the sum cancels --
// which it does on a random graph -- that loss stops being a relative epsilon
// and becomes the answer. The other ops multiply in the storage type, where the
// packed path is faster and nothing downstream amplifies the rounding.
template <size_t TW, FloatingNum cuda_t, FloatingNum accum_t>
__device__ __forceinline__ void gsddmm_accum_fma(accum_t *const __restrict__ acc, VecFloat<TW, cuda_t> a, VecFloat<TW, cuda_t> b) {
    constexpr size_t compact_N  = std::min(TW, sizeof(accum_t));
    constexpr size_t repeat_cnt = TW / compact_N;
#pragma unroll
    for (size_t i = 0; i < repeat_cnt; ++i) {
        reinterpret_cast<VecFloat<compact_N, accum_t> *>(acc)[i].fmaa_(
            reinterpret_cast<VecFloat<compact_N, cuda_t> const *>(&a)[i].template convert_vec<accum_t>(),
            reinterpret_cast<VecFloat<compact_N, cuda_t> const *>(&b)[i].template convert_vec<accum_t>()
        );
    }
}

// acc -= a * b / v^2: the whole per-edge Div term in accum_t. The edge-parallel
// backward cannot defer the 1/self^2 past its atomic, so both reasons to stay
// wide apply at once -- the product for the accuracy above, and the scaled
// result because it does not fit the storage type where 1/v^2 is largest.
template <size_t TW, FloatingNum cuda_t, FloatingNum accum_t>
__device__ __forceinline__ void gsddmm_accum_sub_prod_div_sq(
    accum_t *const __restrict__ acc, VecFloat<TW, cuda_t> a, VecFloat<TW, cuda_t> b, VecFloat<TW, cuda_t> v
) {
    constexpr size_t compact_N  = std::min(TW, sizeof(accum_t));
    constexpr size_t repeat_cnt = TW / compact_N;
#pragma unroll
    for (size_t i = 0; i < repeat_cnt; ++i) {
        VecFloat<compact_N, accum_t> p = reinterpret_cast<VecFloat<compact_N, cuda_t> const *>(&a)[i].template convert_vec<accum_t>();
        p.mul_(reinterpret_cast<VecFloat<compact_N, cuda_t> const *>(&b)[i].template convert_vec<accum_t>());
        p.mul_(gsddmm_inv_sq_chunk<compact_N, cuda_t, accum_t>(reinterpret_cast<VecFloat<compact_N, cuda_t> const *>(&v)[i]));
        reinterpret_cast<VecFloat<compact_N, accum_t> *>(acc)[i] -= p;
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
    index_t const *__restrict__ canonical_edge_idx
) {
    using Plan = GsddmmBackwardPlan<op, ll, rr, reduce>;
    static_assert(Plan::HAS_SELF, "this pass has no operand to reduce; the host must not launch it");
    using Shape = GsddmmRowShape<D_CONST, cuda_t, accum_t>;

    constexpr size_t TW               = Shape::TW;
    constexpr size_t TILES            = Shape::TILES;
    constexpr size_t TILES_PER_THREAD = Shape::TILES_PER_THREAD;

    using Tile       = Shape::Tile;
    using vec_t      = Shape::vec_t;
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
    constexpr bool KEEP_SELF_TILES = Plan::KEEP_SELF_ROW;
    static_assert(KEEP_SELF_TILES == (Plan::NEEDS_SELF_ROW || (Plan::WRITE_EDGE_GRAD && EdgeGradOp::READS_OTHER)), "Plan::KEEP_SELF_ROW must match what this kernel actually needs");
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
    // trip count is exact and the body can be unrolled -- and size_t for the same
    // measured reason the forward's is (see there).
    const size_t loop_iters = (num_edges > warp_id) ? ceil_div(num_edges - warp_id, N_PER_BLOCK) : 0;
#pragma unroll 2
    for (size_t it = 0; it < loop_iters; ++it) {
        const index_t slot = edge_start + static_cast<index_t>(warp_id + it * N_PER_BLOCK);
        // Walking the backward CSR visits edges in CSC order, while dO and the
        // Edge operand are numbered by forward-CSR position.
        // Widen through index_t's UNSIGNED form: ids are non-negative, and a
        // zero-extend from a 32-bit unsigned is free, where the sign-extend of a
        // signed index_t costs an instruction plus a second live value per use.
        // Measured at D=256: the signed form spills 4-16 bytes at the same
        // register count, and this spelling also clears a spill the uint64
        // canonical index had before it was narrowed. Same reason the edge
        // kernels carry every id in eidx_t.
        using eidx_t           = std::make_unsigned_t<index_t>;
        const size_t edge_id   = (canonical_edge_idx != nullptr) ? static_cast<size_t>(static_cast<eidx_t>(canonical_edge_idx[slot]))
                                                                 : static_cast<size_t>(static_cast<eidx_t>(slot));
        const size_t other_row = Plan::OTHER_IS_EDGE ? edge_id : static_cast<size_t>(static_cast<eidx_t>(col_idx[slot]));

        // Dot's d_out is one broadcast scalar for the whole edge: build it once
        // here rather than per tile. The elementwise ops read a row per tile.
        const vec_t d_out_splat   = gsddmm_grad_out_splat<Plan::IS_DOT, TW, cuda_t>(dO, edge_id);
        cuda_t const *const dO_row = Plan::IS_DOT ? nullptr : dO + edge_id * D_CONST;

#pragma unroll
        for (size_t t = 0; t < TILES_PER_THREAD; ++t) {
            const size_t v = lane_id + kWarpSize * t;
            if (v < TILES) [[likely]] {
                vec_t d_out;
                if constexpr (Plan::IS_DOT) {
                    d_out = d_out_splat;
                } else {
                    d_out = gsddmm_grad_out_row_tile<TW, cuda_t, Tile::MemoryHint::Streaming>(dO_row, v);
                }
                // Add/Sub/Copy have constant partials and never read this; it is
                // still zeroed rather than left undefined, both to keep the
                // value passed to apply() defined and to keep the build quiet.
                vec_t other;
                if constexpr (Plan::READS_OTHER) {
                    other = Tile::read(other_base + other_row * D_CONST, v);
                } else {
                    other.store_zero_();
                }
                if constexpr (Plan::NEEDS_SELF_ROW) {
                    // Div's denominator partial is dO * other (GsddmmGradOp's
                    // Div/rhs branch), formed in accum_t because the -1/self^2
                    // applied once after the sum amplifies any rounding taken
                    // here. Every other op accumulates its cuda_t partial.
                    static_assert(op == GSDDMM_OP::Div, "NEEDS_SELF_ROW is the Div/rhs case");
                    gsddmm_accum_fma<TW, cuda_t, accum_t>(acc[t], d_out, other);
                } else {
                    gsddmm_accum_add<TW, cuda_t, accum_t>(acc[t], GradOp::apply(d_out, other));
                }

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
                        g_edge = gsddmm_div_sq<TW, cuda_t, accum_t>(g_edge, other);
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
            if constexpr (Plan::NEEDS_SELF_ROW) {
                // -1/self^2: constant over the sum, so applied once, here -- on
                // the fp32 accumulator, before the narrowing below. Dividing by
                // the self row is exactly the negating case, so the sign rides
                // along with the scaling instead of costing a second pass.
                static_assert(Plan::NEGATE_SELF, "Div by the self row must negate the self gradient");
                gsddmm_accum_neg_div_sq<TW, cuda_t, accum_t>(acc[t], self_tiles[t]);
            }
            vec_t out = gsddmm_accum_to_vec<TW, cuda_t, accum_t>(acc[t]);
            if constexpr (Plan::NEGATE_SELF && !Plan::NEEDS_SELF_ROW) {
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
// RUNS. The list is grouped by the node being reduced, so a chunk is a short
// sequence of runs of edges sharing a target node -- usually exactly one. The
// warp walks the chunk in runs: sum a run in registers, flush it, start the next.
// This subsumes the old shared-chunk ballot (a shared chunk is one run) without
// its fallback, so the per-edge body exists once instead of twice, and it earns
// two things per run rather than per chunk:
//   * ONE set of atomics per run instead of one per edge;
//   * ONE read of the reduced node's own row -- which Div's -1/self^2 factor and
//     the edge operand's gradient both want -- held in registers for the run,
//     where it used to be re-read once or twice per EDGE.
//
// ATOMIC SHAPE. acc is blocked by lane: lane l owns features [(l + 32t)*TW, +TW).
// That is what makes every operand load one 16-byte vector, and it is
// simultaneously the worst possible shape for a scalar atomic -- at a fixed j the
// warp's 32 lanes are TW * sizeof(accum_t) bytes apart, so one atomic touches 32
// distinct sectors and uses 4 bytes of each (measured on A100: exactly 32.00
// sectors per request, against the 4 a 128-byte request needs; the kernel sat at
// 60% lg_throttle / 50% mio_throttle and 4% SM throughput because of it). The
// flush therefore stages the row through per-warp shared memory and re-reads it
// at s * 32 + lane, so each atomic is one contiguous 128-byte, 4-sector request.
// Instruction count is unchanged -- D_CONST/32 atomics either way -- for
// D_CONST/32 shared stores and loads and two __syncwarp().
//
// PIPELINE_STAGES prefetches the per-edge rows (the dO row, and the other
// operand's when the partial reads it) that many edges ahead with cp.async, as
// the forward edge kernel does. The reduced node's row is deliberately not in the
// ring: it is a per-RUN constant, so it belongs in registers.
// =============================================================================
template <GSDDMM_OP op, GSDDMM_MEMBER ll, GSDDMM_MEMBER rr, GSDDMM_REDUCE reduce, size_t D_CONST, FloatingNum cuda_t, IntegralNum index_t, FloatingNum accum_t, uint8_t PIPELINE_STAGES>
__global__ void __launch_bounds__(kWarpSize *kGsddmmEdgeMaxWarpsPerBlock) GSDDMM_backward_edge_block( // no-format
    index_t E,
    cuda_t const *__restrict__ L, cuda_t const *__restrict__ R, cuda_t const *__restrict__ dO,
    accum_t *__restrict__ d_self_f32, cuda_t *__restrict__ d_edge,
    index_pair_t<index_t> const *__restrict__ edge_nodes_idx,
    index_t const *__restrict__ canonical_edge_idx,
    uint32_t edges_per_warp
) {
    // See GSDDMM_forward_edge_block: ids ride in index_t's unsigned form.
    using eidx_t = std::make_unsigned_t<index_t>;
    using Plan   = GsddmmBackwardPlan<op, ll, rr, reduce>;
    static_assert(Plan::HAS_SELF, "this pass has no operand to reduce; the host must not launch it");
    static_assert(Plan::NUM_EDGE_ROWS >= 1, "a pass with a SELF operand always gathers at least its dO or its other row per edge");
    using Shape = GsddmmRowShape<D_CONST, cuda_t, accum_t>;

    constexpr size_t TW               = Shape::TW;
    constexpr size_t TILES            = Shape::TILES;
    constexpr size_t TILES_PER_THREAD = Shape::TILES_PER_THREAD;
    static_assert(TILES * TW == D_CONST, "the flush transpose below relies on the lanes' tiles covering the row exactly");
    static_assert(D_CONST % kWarpSize == 0, "the flush issues one atomic per 32 consecutive features");
    constexpr size_t FLUSH_STEPS = D_CONST / kWarpSize;

    using Tile       = Shape::Tile;
    using vec_t      = Shape::vec_t;
    using GradOp     = GsddmmGradOp<op, Plan::SELF_IS_LHS, TW, cuda_t>;
    using EdgeGradOp = GsddmmGradOp<op, !Plan::SELF_IS_LHS, TW, cuda_t>;

    constexpr bool USE_PIPELINE = PIPELINE_STAGES > 0;
    constexpr size_t NUM_STAGES = USE_PIPELINE ? static_cast<size_t>(PIPELINE_STAGES) : 1;  // only read when USE_PIPELINE
    constexpr size_t NUM_ROWS   = Plan::NUM_EDGE_ROWS;
    constexpr int DO_SLOT       = Plan::DO_SLOT;
    constexpr int OTHER_SLOT    = Plan::OTHER_SLOT;
    // Prefetched rows live in shared memory; directly gathered ones in global,
    // where each dO row is read exactly once and so should evict first.
    constexpr auto DO_HINT = USE_PIPELINE ? Tile::MemoryHint::NoHint : Tile::MemoryHint::Streaming;

    __builtin_assume(threadIdx.x < static_cast<unsigned>(kWarpSize));
    __builtin_assume(threadIdx.y < static_cast<unsigned>(kGsddmmEdgeMaxWarpsPerBlock));
    __builtin_assume(edges_per_warp >= 1 && edges_per_warp <= kGsddmmEdgeMaxEdgesPerWarp);
    const size_t lane_id = threadIdx.x;
    const size_t warp_id = threadIdx.y;

    // Chunk arithmetic in index_t: the grid covers ceil(E / edges_per_warp) warps,
    // so a warp index fits index_t whenever E does.
    const eidx_t block_linear = static_cast<eidx_t>((static_cast<uint64_t>(blockIdx.z) * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x);
    const eidx_t global_warp  = block_linear * static_cast<eidx_t>(blockDim.y) + static_cast<eidx_t>(warp_id);
    const eidx_t edge_base    = global_warp * static_cast<eidx_t>(edges_per_warp);
    const eidx_t num_total = static_cast<eidx_t>(E);
    if (edge_base >= num_total) [[unlikely]] {
        return;
    }
    // <= kGsddmmEdgeMaxEdgesPerWarp (32): chunk-local counters stay 32-bit.
    const uint32_t num_edges = static_cast<uint32_t>((num_total - edge_base < edges_per_warp) ? (num_total - edge_base) : edges_per_warp);

    // Lane k caches edge k's endpoints and canonical id (see the forward kernel).
    index_pair_t<index_t> my_pair{};
    if (lane_id < num_edges) {
        my_pair = __ldcs(&edge_nodes_idx[edge_base + lane_id]);
    }
    const bool remap_edge_ids = canonical_edge_idx != nullptr;
    eidx_t my_canonical       = 0;
    if (remap_edge_ids && lane_id < num_edges) {
        my_canonical = static_cast<eidx_t>(__ldcs(&canonical_edge_idx[edge_base + lane_id]));
    }
    auto canonical_of = [remap_edge_ids, my_canonical, edge_base](uint32_t k) -> eidx_t {
        return remap_edge_ids ? __shfl_sync(FULL_WARP_MASK, my_canonical, static_cast<int>(k)) : edge_base + static_cast<eidx_t>(k);
    };
    // Node this pass reduces into, and the node the other operand is read from.
    // Warp-uniform: every lane must reach these together.
    auto self_node_of = [my_pair](uint32_t k) -> eidx_t {
        return __shfl_sync(FULL_WARP_MASK, reduce == GSDDMM_REDUCE::Dst ? my_pair.y : my_pair.x, static_cast<int>(k));
    };
    auto other_node_of = [my_pair](uint32_t k) -> eidx_t {
        return __shfl_sync(FULL_WARP_MASK, reduce == GSDDMM_REDUCE::Dst ? my_pair.x : my_pair.y, static_cast<int>(k));
    };

    // Per-warp shared slice: the flush staging row first (a multiple of 16 bytes,
    // so the ring behind it stays 16-byte aligned for cp.async), then the ring.
    // Addressed from warp_id alone, so no warp needs to know blockDim.y.
    constexpr size_t FLUSH_BYTES    = D_CONST * sizeof(accum_t);
    constexpr size_t RING_ELEMS     = USE_PIPELINE ? pipelined_ring_elems(NUM_STAGES, NUM_ROWS, D_CONST) : size_t{0};
    constexpr size_t PER_WARP_BYTES = FLUSH_BYTES + RING_ELEMS * sizeof(cuda_t);
    static_assert(FLUSH_BYTES % 16 == 0, "flush row must be a multiple of 16 bytes to keep the ring aligned");
    extern __shared__ __align__(16) uint8_t sh_raw[];
    uint8_t *const my_sh      = sh_raw + warp_id * PER_WARP_BYTES;
    accum_t *const flush_sh   = reinterpret_cast<accum_t *>(my_sh);
    cuda_t *const my_ring     = reinterpret_cast<cuda_t *>(my_sh + FLUSH_BYTES);

    // Per-lane fp32 running sums for the current run, one per feature tile.
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

    // The run's reduced node row, constant for the whole run (see KEEP_SELF_ROW).
    vec_t self_tiles[TILES_PER_THREAD];
    auto load_self_tiles = [&self_tiles, lane_id, L, R](eidx_t node) {
        if constexpr (Plan::KEEP_SELF_ROW) {
            cuda_t const *const row = (Plan::SELF_IS_LHS ? L : R) + static_cast<size_t>(node) * D_CONST;
#pragma unroll
            for (size_t t = 0; t < TILES_PER_THREAD; ++t) {
                const size_t v = lane_id + kWarpSize * t;
                if (v < TILES) [[likely]] {
                    self_tiles[t] = Tile::read(row, v);
                }
            }
        }
    };

    // flush(node): add the finished run's tiles into node's row, transposed
    // through shared so each atomic is one 128-byte request (see the header).
    auto flush_acc = [&](eidx_t node) {
#pragma unroll
        for (size_t t = 0; t < TILES_PER_THREAD; ++t) {
            const size_t v = lane_id + kWarpSize * t;
            if (v < TILES) [[likely]] {
#pragma unroll
                for (size_t j = 0; j < TW; ++j) {
                    flush_sh[v * TW + j] = acc[t][j];
                }
            }
        }
        // Order this run's stores before the reads below, and -- on the way out --
        // the reads before the next run's stores into the same buffer.
        __syncwarp();
        accum_t *const dst = d_self_f32 + static_cast<size_t>(node) * D_CONST;
#pragma unroll
        for (size_t s = 0; s < FLUSH_STEPS; ++s) {
            const size_t f = s * kWarpSize + lane_id;
            atomicAdd(&dst[f], flush_sh[f]);
        }
        __syncwarp();
    };

    // body(k, rows): edge k's partial into acc, plus the edge operand's gradient
    // (a plain per-edge store, needing no sum). rows[] holds the prefetched
    // shared-memory rows when pipelined and the gathered global rows otherwise;
    // the reduced node's row always comes from self_tiles.
    auto body = [&](uint32_t k, cuda_t const *const (&rows)[NUM_ROWS]) {
        const eidx_t edge_id = canonical_of(k);

        // Dot's d_out is one broadcast scalar per edge: built once, not per tile.
        const vec_t d_out_splat = gsddmm_grad_out_splat<Plan::IS_DOT, TW, cuda_t>(dO, static_cast<size_t>(edge_id));

#pragma unroll
        for (size_t t = 0; t < TILES_PER_THREAD; ++t) {
            const size_t v = lane_id + kWarpSize * t;
            if (v < TILES) [[likely]] {
                vec_t d_out;
                if constexpr (Plan::IS_DOT) {
                    d_out = d_out_splat;
                } else {
                    d_out = gsddmm_grad_out_row_tile<TW, cuda_t, DO_HINT>(rows[DO_SLOT], v);
                }
                // Add/Sub/Copy have constant partials and never read this; it is
                // still zeroed rather than left undefined, both to keep the value
                // passed to apply() defined and to keep the build quiet.
                vec_t other;
                if constexpr (Plan::READS_OTHER) {
                    other = Tile::read(rows[OTHER_SLOT], v);
                } else {
                    other.store_zero_();
                }

                if constexpr (Plan::NEEDS_SELF_ROW) {
                    // Div's denominator: -(dO * other) / self^2. Unlike the
                    // node-parallel variant this warp does not own the whole
                    // node, so the factor cannot be deferred past the atomic and
                    // is applied per edge -- from the run's cached self row, in
                    // accum_t, product included. Div/rhs negates.
                    static_assert(op == GSDDMM_OP::Div && Plan::NEGATE_SELF, "NEEDS_SELF_ROW is the negating Div/rhs case");
                    gsddmm_accum_sub_prod_div_sq<TW, cuda_t, accum_t>(acc[t], d_out, other, self_tiles[t]);
                } else {
                    vec_t contrib = GradOp::apply(d_out, other);
                    if constexpr (Plan::NEGATE_SELF) {
                        contrib.neg_();
                    }
                    gsddmm_accum_add<TW, cuda_t, accum_t>(acc[t], contrib);
                }

                if constexpr (Plan::WRITE_EDGE_GRAD) {
                    // Only read the node row when the op's partial uses it: an
                    // operand the backward never touches may be a 1-row stand-in.
                    vec_t node_row;
                    if constexpr (EdgeGradOp::READS_OTHER) {
                        node_row = self_tiles[t];
                    } else {
                        node_row.store_zero_();
                    }
                    vec_t g_edge = EdgeGradOp::apply(d_out, node_row);
                    if constexpr (op == GSDDMM_OP::Div && Plan::OTHER == rr) {
                        g_edge = gsddmm_div_sq<TW, cuda_t, accum_t>(g_edge, other);
                        g_edge.neg_();
                    } else if constexpr (op == GSDDMM_OP::Sub && Plan::OTHER == rr) {
                        g_edge.neg_();
                    }
                    Tile::template write<Tile::MemoryHint::Streaming>(d_edge + static_cast<size_t>(edge_id) * D_CONST, v, g_edge);
                }
            }
        }
    };

    // Run bookkeeping, shared by the pipelined and direct paths: both visit the
    // chunk's edges in order, so a run ends exactly when the target node changes.
    // num_edges >= 1 here (the chunk would have exited otherwise), so the first
    // run can be opened before the loop and closed after it.
    eidx_t run_node = self_node_of(0);
    clear_acc();
    load_self_tiles(run_node);
    auto consume = [&](uint32_t k, cuda_t const *const (&rows)[NUM_ROWS]) {
        const eidx_t node = self_node_of(k);
        if (node != run_node) {
            flush_acc(run_node);
            run_node = node;
            clear_acc();
            load_self_tiles(node);
        }
        body(k, rows);
    };

    if constexpr (USE_PIPELINE) {
        // Global row addresses of edge k's prefetched rows (warp-uniform).
        auto addr = [&](uint32_t k, cuda_t const *(&srcs)[NUM_ROWS]) {
            const eidx_t e = canonical_of(k);
            if constexpr (!Plan::IS_DOT) {
                srcs[DO_SLOT] = dO + static_cast<size_t>(e) * D_CONST;
            }
            if constexpr (Plan::READS_OTHER) {
                const eidx_t other_row = Plan::OTHER_IS_EDGE ? e : other_node_of(k);
                srcs[OTHER_SLOT]       = (Plan::SELF_IS_LHS ? R : L) + static_cast<size_t>(other_row) * D_CONST;
            }
        };
        pipelined_row_loop<D_CONST, NUM_STAGES, NUM_ROWS, cuda_t>(lane_id, num_edges, my_ring, addr, consume);
    } else {
        for (uint32_t k = 0; k < num_edges; ++k) {
            const eidx_t e = canonical_of(k);
            cuda_t const *rows[NUM_ROWS];
            if constexpr (!Plan::IS_DOT) {
                rows[DO_SLOT] = dO + static_cast<size_t>(e) * D_CONST;
            }
            if constexpr (Plan::READS_OTHER) {
                const eidx_t other_row = Plan::OTHER_IS_EDGE ? e : other_node_of(k);
                rows[OTHER_SLOT]       = (Plan::SELF_IS_LHS ? R : L) + static_cast<size_t>(other_row) * D_CONST;
            }
            consume(k, rows);
        }
    }
    flush_acc(run_node);
}

}  // namespace gsddmm
