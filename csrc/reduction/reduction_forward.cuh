#pragma once

#include <cstdint>

#include "common.cuh"
#include "common/row_group.cuh"
#include "common/row_kernels.cuh"
#include "reduction/reduction_common.cuh"

// ================================================================================================
// reduction_aggr forward (see reduction_common.cuh for what it computes).
//
// The work is a list of items: a light row, or a slice of a heavy row's edges, times one pass over
// the features. A sweep shape (common/row_shape.cuh) lays a row across a group of G lanes, and the
// worker of an item is
//
//   G < 32   a lane group: kWarpSize / G items per warp, each group walking its own item's edges
//            with UNROLL rows in flight and the next ids loading meanwhile. Rows of up to 16
//            vectors (64 fp32, 128 fp16 elements) have too little work per edge to share a row's
//            edges across a warp: the per-row merge and the idle groups at the end of a short row
//            cost more than they save -- measured 1.6-2.5x slower at degree 10.
//   G == 32  the warp, which takes ids 32 at a time and broadcasts them. A row longer than one pass
//            of the warp (32 vectors, or 64 for rows of single elements and past 256 vectors -- see
//            plan_sweep_shape) is swept in passes, every pass an item of its own: the reduction
//            being elementwise, passes need nothing from each other and no chunked path.
//
// Light rows are written in place. A slice of a heavy row writes an fp32 partial sum, which
// sum_slices_kernel adds into the row in a fixed order, or, for min/max, folds its
// candidates into the row with 64-bit atomics on keys that order them as the light rows do (see
// pack_candidate), which finalize_heavy_kernel decodes. Either way heavy rows come out
// deterministically and with the light rows' tie-break.
//
// Edges are walked last to first: an update on <= then leaves the earliest extremum without a
// position compare per element, and an incumbent still equal to the +-inf identity is replaced by
// an input of +-inf.
// ================================================================================================

namespace turbo_gnn::reduction {

// Warps of a forward block. They share nothing -- every warp, or lane group, has items of its own --
// so the block size only sets the granularity of scheduling; from two warps up a block reaches the
// full 64 warps per SM within the 32-blocks-per-SM limit of sm_90.
inline constexpr uint32_t kForwardMaxWarps = 8;

// Edges of a walk unrolled together, so that a lane has that many row loads in flight. Measured on
// H100 at degree 10: 2 and 8 lose to 4 by 1-25% for lane groups. For whole-warp rows 4 still wins
// by 3-4% where elements are 16-bit or the reducer is a min/max, but fp32 sums run 3-7% faster
// with 2.
template <ReductionOp OP, FloatingNum cuda_t, bool WARP_ROWS>
inline constexpr size_t kForwardUnroll = WARP_ROWS && OP == ReductionOp::SUM && sizeof(cuda_t) == 4 ? 2 : 4;

// A lane's running result for its E elements of the row.
template <ReductionOp OP, size_t E, bool = kTracksArg<OP>>
struct Partial;

template <ReductionOp OP, size_t E>
struct Partial<OP, E, /*TRACKS_ARG=*/false> {
    alignas(16) float val[E];

    __device__ __forceinline__ void init() {
#pragma unroll
        for (size_t e = 0; e < E; ++e) {
            val[e] = 0.0f;
        }
    }
    __device__ __forceinline__ void update(float const (&x)[E], uint32_t) {
#pragma unroll
        for (size_t e = 0; e < E; ++e) {
            val[e] += x[e];
        }
    }
};

template <ReductionOp OP, size_t E>
struct Partial<OP, E, /*TRACKS_ARG=*/true> {
    using Ext = Extremum<OP>;
    alignas(16) float val[E];
    alignas(16) uint32_t pos[E];

    __device__ __forceinline__ void init() {
#pragma unroll
        for (size_t e = 0; e < E; ++e) {
            val[e] = Ext::kIdentity;
            pos[e] = kNoPos;
        }
    }
    __device__ __forceinline__ void update(float const (&x)[E], uint32_t edge) {
#pragma unroll
        for (size_t e = 0; e < E; ++e) {
            if (Ext::takes(x[e], val[e])) {
                val[e] = x[e];
                pos[e] = edge;
            }
        }
    }
};

// Resident 8-warp blocks requested per SM; 8, 6 and 5 leave 32, 40 and 48 registers a thread. On a
// launch with many waves of work items the kernel lives on occupancy, and ptxas, unprompted, took
// 38-64 registers. Each entry is the tightest bound that does not spill, where it measured faster
// on H100 (uniform and power-law graphs, degree 10, and the transposed power-law graph the sum
// backward walks):
//
//   whole-warp rows, one vector a lane, sum                    8   +5-8% sum fp16 D = 256
//   whole-warp rows, one vector a lane, fp32 min/max           6   +3-10% over 8, which spills since
//                                                                  the gather keeps its addresses live
//   lane groups, fp32 sum                                      6   +5-11%
//   lane groups, 16-bit sum and fp32 min/max                   5   +3-13%, but -2% for min fp32 D = 8
//
// 16-bit min/max and two vectors a lane spill under any bound, and 64-bit indices take more
// registers throughout; those keep the compiler's choice. Two single elements a lane -- whole-warp
// rows not 16-byte aligned (row_shape.cuh) -- hold no more data than one vector but a second gather
// address, which spills at 32 registers: their sums and fp32 min/max take 6 blocks, 40 registers,
// measured 1.12x faster for fp32 sums at D = 33 than the compiler's 42 and neutral elsewhere. Whole-warp kernels for other
// architectures than sm_90 spill inside the walk at these bounds (sm_80/86/100, 8-16 bytes), so
// they ask for 5 blocks, 48 registers -- spill-checked, not timed: no other GPU to measure on.
//
// OCCUPANCY = false drops the bound. When the longest rows set the finish -- a launch whose longest
// light row is long against each resident warp's share of the edges -- the unbounded build keeps
// more rows in flight per warp: measured on real graphs, the whole-warp fp32 min kernel ran 13-28%
// faster unbounded on tolokers-2 and hm-categories and 4-9% slower on flickr, ogbn-arxiv,
// twitch-views and reddit. Lane groups showed under 6% either way and have no unbounded twin. The
// host picks (plan_forward, reduction/reduction_launch.cuh).
//
// g-SpMM messages that combine a node row with edge data (reduction_common.cuh) hold a second row,
// or a scalar, and its addresses; at the bounds above most of them spill inside the walk. Where a
// bound is set they take 64 registers (4 blocks) for a full edge row and 48 (5) for a broadcast one
// on whole-warp rows -- spill-free on sm_90 -- which measured, against the bounds above, on five
// real graphs at D = 16-256: mul 1.03x (fp32 sum) to 1.21x (fp16 sum) faster, broadcast min/max
// 1.08x and 16-bit broadcast sums 1.10x (up to 1.9x). Two cases stay at the bounds above, where the
// occupancy outweighed the spills: broadcast fp32 sums on whole-warp rows (0.81-0.90x at 5 blocks)
// and lane-group fp32 sums of full rows, which do not spill at 6 -- unless the walk reads an edge map
// (the sum backward of mul and div), whose one-lane groups spill 14-16 values in the walk there.
// copy_e, a single row like copy_u, keeps them too (0.92-1.01x at 5) but for 16-bit whole-warp sums,
// which spill 14 values at 32 registers and ran 1.01x (up to 1.08x) at 5. Other architectures than
// sm_90 spill message walks at these bounds (up to 14 values on sm_100), so there any message takes
// at most 4 blocks, 64 registers -- spill-checked on sm_80/86/89/100, not timed.

template <ReductionOp OP, int SHAPE, FloatingNum cuda_t, IntegralNum index_t, bool OCCUPANCY = true, class M = NodeRows>
inline constexpr int kForwardMinBlocks = [] {
    constexpr bool kSum  = OP == ReductionOp::SUM;
    constexpr bool kFp32 = sizeof(cuda_t) == 4;
    constexpr bool kWarp = Shape<SHAPE>::G == kWarpSize;
    // Two single elements a lane, the narrow rows' whole-warp shape: no more data than one vector,
    // but a second gather address, which spills at 32 registers.
    constexpr bool kNarrow2 = Shape<SHAPE>::S == 2 && !Shape<SHAPE>::kWide;
    constexpr int kBlocks   = !OCCUPANCY || sizeof(index_t) != 4 ? 1
                              : kNarrow2 && kWarp                ? (kSum || kFp32 ? (kCudaArch != 900 ? 5 : 6) : 1)
                              : kWarp                            ? (Shape<SHAPE>::S == 1 && (kSum || kFp32) ? (kCudaArch != 900 ? 5
                                                                                                                  : kSum        ? 8
                                                                                                                                : 6)
                                                                                                            : 1)
                              : kSum                             ? (kFp32 ? 6 : 5)
                                                                 : (kFp32 ? 5 : 1);
    int blocks              = kBlocks;
    if constexpr (M::kCombines) {
        if (M::kBcast) {
            blocks = kWarp && !(kSum && kFp32) ? 5 : kBlocks;
        } else {
            blocks = kWarp ? (kSum ? 5 : 4) : (kSum && kFp32 && !M::kMapped ? kBlocks : 4);
        }
        if (kNarrow2) {
            blocks = std::min(blocks, 5);
        }
    } else if constexpr (M::kEdge) {
        blocks = kWarp && kSum && !kFp32 ? 5 : kBlocks;  // copy_e
    }
    if constexpr (M::kEdge && kCudaArch != 900) {
        blocks = std::min(blocks, 4);  // see above: spill-checked, not timed
    }
    blocks = std::min(blocks, kBlocks);  // never above the bound of the plain walk, nor where it has none
    return std::min(blocks, kMaxThreadsPerSM / static_cast<int>(kForwardMaxWarps * kWarpSize));
}();
static_assert(kForwardMaxWarps * kWarpSize == 256, "the bounds above are for 256-thread blocks");

// Whether a configuration has an unbounded twin.
template <ReductionOp OP, int SHAPE, FloatingNum cuda_t, IntegralNum index_t, class M = NodeRows>
inline constexpr bool kHasIlpTwin = Shape<SHAPE>::G == kWarpSize && kForwardMinBlocks<OP, SHAPE, cuda_t, index_t, true, M> > 1;

// The walks of a message that reads edge data: for_each_neighbor_reverse and
// for_each_neighbor_reverse_unbatched (common/row_group.cuh) with each slot's edge position alongside
// its neighbour id -- first + slot, or edge_map[first + slot] on a transposed walk, fetched with the
// ids so that no edge waits on a dependent load of its row index.
template <bool MAPPED, IntegralNum index_t>
__device__ __forceinline__ index_t edge_position(index_t const *const __restrict__ map, index_t first, uint32_t slot) {
    const index_t e = first + static_cast<index_t>(slot);
    if constexpr (MAPPED) {
        return map[e];
    } else {
        return e;
    }
}

// fn(slot, active, nb, e), called warp-uniformly as for for_each_neighbor_reverse.
template <size_t G, size_t UNROLL, bool MAPPED, IntegralNum index_t, typename Fn>
__device__ __forceinline__ void for_each_edge_reverse(
    index_t const *const __restrict__ ids, index_t const *const __restrict__ map, index_t first, uint32_t begin, uint32_t end, uint32_t lane,
    Fn&& fn
) {
    constexpr uint32_t kStep = kWarpSize / G;
    const uint32_t g         = lane / G;

    uint32_t top   = end;
    uint32_t cnt   = min(static_cast<uint32_t>(kWarpSize), top - begin);
    uint32_t lo    = top - cnt;
    const bool ok  = top > begin && lane < cnt;
    index_t held   = ok ? ids[lo + lane] : index_t{};
    index_t held_e = ok ? edge_position<MAPPED>(map, first, lo + lane) : index_t{};
    while (top > begin) {
        const uint32_t next_cnt = min(static_cast<uint32_t>(kWarpSize), lo - begin);
        const uint32_t next_lo  = lo - next_cnt;
        const bool next_ok      = lo > begin && lane < next_cnt;
        const index_t next      = next_ok ? ids[next_lo + lane] : index_t{};
        const index_t next_e    = next_ok ? edge_position<MAPPED>(map, first, next_lo + lane) : index_t{};
        const uint32_t steps    = ceil_div(cnt, kStep);

#pragma unroll UNROLL
        for (uint32_t q = 0; q < steps; ++q) {
            const uint32_t src = (steps - 1 - q) * kStep + g;
            const index_t nb   = __shfl_sync(FULL_WARP_MASK, held, src);
            const index_t e    = __shfl_sync(FULL_WARP_MASK, held_e, src);
            fn(lo + src, G == kWarpSize || src < cnt, nb, e);
        }
        top    = lo;
        cnt    = next_cnt;
        lo     = next_lo;
        held   = next;
        held_e = next_e;
    }
}

// load(j, e, active), use(slot, row), as for for_each_neighbor_reverse_unbatched.
template <size_t UNROLL, bool MAPPED, IntegralNum index_t, typename LoadFn, typename UseFn>
__device__ __forceinline__ void for_each_edge_reverse_unbatched(
    index_t const *const __restrict__ ids, index_t const *const __restrict__ map, index_t first, uint32_t begin, uint32_t end, LoadFn&& load,
    UseFn&& use
) {
    using row_t = decltype(load(index_t{}, index_t{}, true));
    auto fetch  = [&](uint32_t top, index_t(&j)[UNROLL], index_t(&e)[UNROLL]) {
#pragma unroll
        for (size_t u = 0; u < UNROLL; ++u) {
            const bool ok = top > begin + static_cast<uint32_t>(u);
            j[u]          = ok ? ids[top - 1 - u] : index_t{};
            e[u]          = ok ? edge_position<MAPPED>(map, first, top - 1 - static_cast<uint32_t>(u)) : index_t{};
        }
    };

    index_t j[UNROLL], e[UNROLL];
    fetch(end, j, e);
    for (uint32_t top = end; top > begin;) {
        const uint32_t n = min(static_cast<uint32_t>(UNROLL), top - begin);
        row_t rows[UNROLL];
#pragma unroll
        for (size_t u = 0; u < UNROLL; ++u) {
            rows[u] = load(j[u], e[u], u < n);
        }
        const uint32_t next = top - n;
        fetch(next, j, e);
#pragma unroll
        for (size_t u = 0; u < UNROLL; ++u) {
            if (u < n) {
                use(top - 1 - static_cast<uint32_t>(u), rows[u]);
            }
        }
        top = next;
    }
}

// One edge's operands as a lane holds them: the node row and the edge row (or scalar), whichever the
// message reads. Unread members are dead and cost nothing.
template <class RG, FloatingNum cuda_t>
struct EdgeRows {
    typename RG::Frag node;
    typename RG::Frag edge;
    cuda_t scalar;
};

// The lane's message values of one edge, in fp32, from the operands it loaded.
template <class M, ReductionOp OP, class RG, FloatingNum cuda_t>
__device__ __forceinline__ void form_message(EdgeRows<RG, cuda_t> const& rows, float (&x)[RG::kAccs]) {
    if constexpr (!M::kEdge) {
        RG::to_accum(rows.node, x);
    } else if constexpr (!M::kNode) {
        static_assert(!M::kBcast, "copy_e takes its width from the edge operand, so it never broadcasts");
        RG::to_accum(rows.edge, x);
    } else {
        alignas(16) float u[RG::kAccs];
        RG::to_accum(rows.node, u);
        if constexpr (M::kBcast) {
            const float ev = static_cast<float>(rows.scalar);
#pragma unroll
            for (size_t k = 0; k < RG::kAccs; ++k) {
                x[k] = M::Ops::call(u[k], ev);
            }
        } else {
            alignas(16) float ev[RG::kAccs];
            RG::to_accum(rows.edge, ev);
#pragma unroll
            for (size_t k = 0; k < RG::kAccs; ++k) {
                x[k] = M::Ops::call(u[k], ev[k]);
            }
        }
        if constexpr (M::template kRounds<OP, cuda_t>) {
#pragma unroll
            for (size_t k = 0; k < RG::kAccs; ++k) {
                x[k] = static_cast<float>(static_cast<cuda_t>(x[k]));
            }
        }
    }
}

template <ReductionOp OP, int SHAPE, FloatingNum cuda_t, IntegralNum index_t, bool OCCUPANCY = true, class M = NodeRows>
__global__ void __launch_bounds__(kForwardMaxWarps *kWarpSize, kForwardMinBlocks<OP, SHAPE, cuda_t, index_t, OCCUPANCY, M>)
reduction_forward_kernel(ForwardParams<cuda_t, index_t> const p) {
    using Sh                        = Shape<SHAPE>;
    static constexpr size_t G       = Sh::G;
    static constexpr size_t VEC     = Sh::template vec<cuda_t>;
    using RG                        = RowGroup<G, Sh::S, VEC, cuda_t, float>;
    using Frag                      = typename RG::Frag;
    using Rows                      = EdgeRows<RG, cuda_t>;
    static constexpr size_t kAccs   = RG::kAccs;
    static constexpr uint32_t kPass = rows::kPassVecs<SHAPE>;
    // Whole-warp rows; otherwise every lane group is a worker of its own.
    static constexpr bool kWarpRows = G == kWarpSize;
    // Only whole-warp shapes sweep a row in more than one pass; for lane groups passes == 1.
    static constexpr bool kMultiPass = kWarpRows;
    using Sentinel                   = IndexSentinel<index_t>;
    static_assert(!(M::kMapped && kTracksArg<OP>), "an edge-mapped walk is a sum backward; min/max never take one");

    const uint32_t lane    = threadIdx.x % kWarpSize;
    const uint32_t r       = lane % G;
    const int64_t warp     = static_cast<int64_t>(blockIdx.x) * (blockDim.x / kWarpSize) + threadIdx.x / kWarpSize;
    const int64_t worker   = warp * static_cast<int64_t>(RG::kGroupsPerWarp) + lane / G;
    const int64_t row_item = kMultiPass ? worker / p.passes : worker;
    const uint32_t pass    = kMultiPass ? static_cast<uint32_t>(worker - row_item * p.passes) : 0;
    if (row_item >= p.num_slices + p.num_light) {
        return;  // whole groups: uniform across the warp for whole-warp rows
    }

    const bool sliced       = row_item < p.num_slices;
    const int heavy_slot    = sliced ? p.chunk_node[row_item] : 0;
    const int64_t node      = static_cast<int64_t>(sliced ? p.heavy_nodes[heavy_slot] : p.light_nodes[row_item - p.num_slices]);
    const index_t row_begin = p.row_ptr[node];
    const uint32_t degree   = static_cast<uint32_t>(p.row_ptr[node + 1] - row_begin);
    const uint32_t lo       = sliced ? min(static_cast<uint32_t>(p.chunk_start[row_item]), degree) : 0u;
    const uint32_t hi       = sliced ? lo + min(p.slice_size, degree - lo) : degree;

    // Where the result goes, settled before the walk so that only this stays live across it: a slice
    // of a heavy row folds into the row's merge (its packed candidates, or its partial-sum row), while
    // a light row -- and a heavy row short enough for one slice, which is the whole row -- is written
    // in place. Kept until the end instead, the row's degree, node and item raised fp32 sums from 32
    // and 40 registers to 38 and 48, and their time by 8-10%.
    const bool partial = sliced && (lo != 0 || hi != degree);
    const int64_t dst  = !partial ? node : kTracksArg<OP> ? static_cast<int64_t>(heavy_slot) : row_item;

    const uint32_t v0                     = pass * kPass;
    const uint32_t cnvec                  = min(p.nvec - v0, kPass);
    cuda_t const *const x_pass            = p.X + static_cast<int64_t>(v0) * VEC;
    index_t const *const __restrict__ ids = p.col_idx + row_begin;

    // The gather's addressing, hoisted out of the walk: each lane's vector address within a row is
    // fixed, so a row costs one multiply-add of the neighbour id. A lane past the end of a short pass
    // reads the pass's last vector instead of nothing -- a line the group reads anyway -- and drops
    // it at the store, and an inactive step of a whole-warp walk reads row 0, never used: no per-load
    // predicate. Measured 95 instructions per 4 edges against 179 for the fp32 one-vector shape, and
    // 1.1-1.6x faster wherever the walk is issue-bound (dense graphs, 16-bit rows). The host keeps
    // the row stride within 32 bits; the id times the stride is formed in 64. An edge row of a message
    // is addressed the same way, at the edge operand's stride D.
    //
    // Except for whole-warp sums of 16-bit rows: there the held addresses spill at the 32 registers
    // those kernels run best at, and at 40 they lose 4-14% to the predicated gather at 32.
    static constexpr bool kHoisted = !(kWarpRows && OP == ReductionOp::SUM && sizeof(cuda_t) == 2);
    cuda_t const *lane_x[Sh::S];
    cuda_t const *lane_e[Sh::S];
#pragma unroll
    for (size_t s = 0; s < Sh::S; ++s) {
        const int64_t at = static_cast<int64_t>(min(RG::vec_index(r, s), cnvec - 1)) * VEC;
        if constexpr (M::kNode) {
            lane_x[s] = x_pass + at;
        }
        if constexpr (M::kEdge && !M::kBcast) {
            lane_e[s] = p.E + static_cast<int64_t>(v0) * VEC + at;
        }
    }
    const uint32_t stride_x = static_cast<uint32_t>(p.stride_x);

    Partial<OP, kAccs> acc;
    acc.init();
    // stride: the hoisted gather's 32-bit row stride; stride64: the predicated gather's.
    auto load_row = [&](cuda_t const *const(&lanes)[Sh::S], cuda_t const *const base, uint32_t stride, int64_t stride64, index_t k,
                        bool active) {
        Frag f;
        if constexpr (kHoisted) {
            const uint64_t off = static_cast<uint64_t>(static_cast<std::make_unsigned_t<index_t>>(k)) * stride;
#pragma unroll
            for (size_t s = 0; s < Sh::S; ++s) {
                f.v[s] = RG::Tile::read(lanes[s] + off, 0);
            }
        } else {
            RG::load(f, base + static_cast<int64_t>(k) * stride64, r, cnvec, active);
        }
        return f;
    };
    // j: the neighbour; e: the edge's row of the edge operand.
    auto load = [&](index_t j, index_t e, bool active) {
        Rows rows;
        if constexpr (M::kNode) {
            rows.node = load_row(lane_x, x_pass, stride_x, p.stride_x, j, active);
        }
        if constexpr (M::kBcast) {
            rows.scalar = p.E[static_cast<std::make_unsigned_t<index_t>>(e)];
        } else if constexpr (M::kEdge) {
            rows.edge = load_row(lane_e, p.E + static_cast<int64_t>(v0) * VEC, p.D, static_cast<int64_t>(p.D), e, active);
        }
        return rows;
    };
    // pos: the edge's position in the row.
    auto use = [&](uint32_t pos, Rows const& rows) {
        alignas(16) float x[kAccs];
        form_message<M, OP, RG, cuda_t>(rows, x);
        acc.update(x, pos);
    };
    if constexpr (!M::kEdge) {
        // The neighbour's row is the whole message: the walks of reduction_aggr, on bare fragments.
        auto load_node = [&](index_t j, bool active) { return load_row(lane_x, x_pass, stride_x, p.stride_x, j, active); };
        auto use_node  = [&](uint32_t pos, Frag const& f) {
            alignas(16) float x[kAccs];
            RG::to_accum(f, x);
            acc.update(x, pos);
        };
        if constexpr (kWarpRows) {
            for_each_neighbor_reverse<G, kForwardUnroll<OP, cuda_t, true>>(ids, lo, hi, lane, [&](uint32_t pos, bool active, index_t j) {
                const Frag f = load_node(j, active);
                if (active) {
                    use_node(pos, f);
                }
            });
        } else {
            for_each_neighbor_reverse_unbatched<kForwardUnroll<OP, cuda_t, false>>(ids, lo, hi, load_node, use_node);
        }
    } else if constexpr (kWarpRows) {
        for_each_edge_reverse<G, kForwardUnroll<OP, cuda_t, true>, M::kMapped>(
            ids, p.edge_map, row_begin, lo, hi, lane, [&](uint32_t pos, bool active, index_t j, index_t e) {
                const Rows rows = load(j, e, active);
                if (active) {
                    use(pos, rows);
                }
            }
        );
    } else {
        for_each_edge_reverse_unbatched<kForwardUnroll<OP, cuda_t, false>, M::kMapped>(ids, p.edge_map, row_begin, lo, hi, load, use);
    }

    const int64_t elem0 = static_cast<int64_t>(v0) * VEC;
    if (partial) {
#pragma unroll
        for (size_t s = 0; s < Sh::S; ++s) {
            const uint32_t v = RG::vec_index(r, s);
            if (v < cnvec) {
                const int64_t elem = elem0 + static_cast<int64_t>(v) * VEC;
                if constexpr (kTracksArg<OP>) {
                    unsigned long long *const keys = p.packed + dst * p.D + elem;
#pragma unroll
                    for (size_t e = 0; e < VEC; ++e) {
                        if (acc.pos[s * VEC + e] != kNoPos) {
                            fold_candidate<OP>(keys + e, pack_candidate<OP>(acc.val[s * VEC + e], acc.pos[s * VEC + e]));
                        }
                    }
                } else {
                    store_words<VEC>(p.part + dst * p.D + elem, *reinterpret_cast<float const(*)[VEC]>(&acc.val[s * VEC]));
                }
            }
        }
        return;
    }

    cuda_t *const out_row = p.out + dst * p.D + elem0;
    if constexpr (kTracksArg<OP>) {
        index_t *const arg_row = p.arg + dst * p.D + elem0;
#pragma unroll
        for (size_t s = 0; s < Sh::S; ++s) {
            const uint32_t v = RG::vec_index(r, s);
            if (v < cnvec) {
                alignas(16) index_t args[VEC];
#pragma unroll
                for (size_t e = 0; e < VEC; ++e) {
                    const uint32_t q = acc.pos[s * VEC + e];
                    if constexpr (M::kArgIsEdge) {
                        args[e] = q == kNoPos ? Sentinel::INVALID : row_begin + static_cast<index_t>(q);
                    } else {
                        args[e] = q == kNoPos ? Sentinel::INVALID : ids[q];
                    }
                    if (q == kNoPos) {
                        acc.val[s * VEC + e] = 0.0f;
                    }
                }
                store_words<VEC>(arg_row + static_cast<int64_t>(v) * VEC, args);
            }
        }
    }
    RG::store(out_row, acc.val, 1.0f, r, cnvec);
}

// The message of edge eid at feature f, formed as the forward forms it: for a zero result, whose
// key cannot tell -0 and subnormals from +0.
template <class M, ReductionOp OP, FloatingNum cuda_t, IntegralNum index_t>
__device__ __forceinline__ float message_at(
    cuda_t const *const __restrict__ X, int64_t stride_x, cuda_t const *const __restrict__ E, uint32_t D,
    index_t const *const __restrict__ col_idx, index_t eid, int64_t f
) {
    float u  = 0.0f;
    float ev = 0.0f;
    if constexpr (M::kNode) {
        u = static_cast<float>(X[static_cast<int64_t>(col_idx[eid]) * stride_x + f]);
    }
    if constexpr (M::kBcast) {
        ev = static_cast<float>(E[eid]);
    } else if constexpr (M::kEdge) {
        ev = static_cast<float>(E[static_cast<int64_t>(eid) * D + f]);
    }
    float x = M::Ops::call(u, ev);
    if constexpr (M::template kRounds<OP, cuda_t>) {
        x = static_cast<float>(static_cast<cuda_t>(x));
    }
    return x;
}

// ------------------------------------------------------------------------------------------------
// Heavy rows of min/max: the slices have folded their candidates into one packed key per element
// (see pack_candidate); this decodes the keys into the rows' values and args. One thread per row
// vector, grid (num_heavy, 1, vector blocks) as rows::merge_launch lays it out.
// ------------------------------------------------------------------------------------------------
template <ReductionOp OP, bool WIDE, FloatingNum cuda_t, IntegralNum index_t, class M = NodeRows>
__global__ void __launch_bounds__(rows::kMergeThreads) finalize_heavy_kernel(
    index_t const *const __restrict__ heavy_nodes, int num_heavy, int const *const __restrict__ node_chunk_offset,
    index_t const *const __restrict__ row_ptr, index_t const *const __restrict__ col_idx, cuda_t const *const __restrict__ X, int64_t stride_x,
    uint32_t D, uint32_t nvec, unsigned long long const *const __restrict__ packed, cuda_t *const __restrict__ out,
    index_t *const __restrict__ arg, cuda_t const *const __restrict__ E
) {
    static constexpr size_t VEC = WIDE ? 16 / sizeof(cuda_t) : 1;
    using Tile                  = TileOps<VEC, cuda_t, float>;
    using Sentinel              = IndexSentinel<index_t>;

    const int slot   = blockIdx.x;
    const uint32_t v = blockIdx.z * blockDim.x + threadIdx.x;
    if (slot >= num_heavy || v >= nvec) [[unlikely]] {
        return;
    }
    if (node_chunk_offset[slot + 1] - node_chunk_offset[slot] <= 1) {
        return;  // a single slice: the forward wrote the row in place
    }
    const int64_t node       = static_cast<int64_t>(heavy_nodes[slot]);
    const int64_t elem       = static_cast<int64_t>(v) * VEC;
    const index_t row_begin  = row_ptr[node];
    index_t const *const ids = col_idx + row_begin;

    alignas(16) unsigned long long keys[VEC];
    load_words<VEC>(keys, packed + static_cast<int64_t>(slot) * D + elem);
    alignas(16) float vals[VEC];
    alignas(16) index_t args[VEC];
#pragma unroll
    for (size_t e = 0; e < VEC; ++e) {
        if (keys[e] == kPackedEmpty<OP>) {
            vals[e] = 0.0f;
            args[e] = Sentinel::INVALID;
        } else if constexpr (M::kArgIsEdge) {
            args[e] = row_begin + static_cast<index_t>(unpack_pos<OP>(keys[e]));
            vals[e] = unpack_value<OP>(keys[e]);
            if (vals[e] == 0.0f) {
                vals[e] = message_at<M, OP>(X, stride_x, E, D, col_idx, args[e], elem + static_cast<int64_t>(e));
            }
        } else {
            args[e] = ids[unpack_pos<OP>(keys[e])];
            vals[e] = unpack_value<OP>(keys[e]);
            if (vals[e] == 0.0f) {
                // Every zero and subnormal shares +0's key: the winner's own bits are in X.
                vals[e] = static_cast<float>(X[static_cast<int64_t>(args[e]) * stride_x + elem + static_cast<int64_t>(e)]);
            }
        }
    }
    Tile::write(out + node * D, v, Tile::convert_from_accum(vals));
    store_words<VEC>(arg + node * D + elem, args);
}

// ------------------------------------------------------------------------------------------------
// Heavy rows of sums: the row's per-slice partials added in a fixed order. A block per (heavy row,
// 32 row vectors); its warps split the slices -- warp w takes slices w, w + W, ... -- each lane
// summing one vector, kSumMergeBatch slices' loads in flight at a time, and the warps' sums are
// added in warp order. With one thread per vector folding all of a row's slices, as in
// rows::sum_merge_slices_kernel, a hub's hundreds of slices make one chain of dependent loads:
// measured 0.20 ms against the forward's 0.08 at fp16, D = 64 on a power-law graph.
// ------------------------------------------------------------------------------------------------
inline constexpr uint32_t kSumMergeWarps = 8;
inline constexpr uint32_t kSumMergeBatch = 4;

template <bool WIDE, FloatingNum cuda_t, IntegralNum index_t>
__global__ void __launch_bounds__(kSumMergeWarps *kWarpSize) sum_slices_kernel(
    index_t const *const __restrict__ heavy_nodes, int num_heavy, int const *const __restrict__ node_chunk_offset, uint32_t D, uint32_t nvec,
    float const *const __restrict__ part, cuda_t *const __restrict__ out
) {
    static constexpr size_t VEC = WIDE ? 16 / sizeof(cuda_t) : 1;
    using Tile                  = TileOps<VEC, cuda_t, float>;
    using vals_t                = float[VEC];
    __shared__ __align__(16) float sums[kSumMergeWarps - 1][kWarpSize][VEC];

    const int slot = blockIdx.x;
    if (slot >= num_heavy) [[unlikely]] {
        return;
    }
    const uint32_t lane = threadIdx.x % kWarpSize;
    const uint32_t warp = threadIdx.x / kWarpSize;
    const int64_t node  = static_cast<int64_t>(heavy_nodes[slot]);
    const int lo        = node_chunk_offset[slot];
    const int hi        = node_chunk_offset[slot + 1];
    if (hi - lo <= 1) {
        return;  // a single slice: the forward wrote the row in place
    }

    const uint32_t chunks = ceil_div(nvec, static_cast<uint32_t>(kWarpSize));
    for (uint32_t c = blockIdx.z; c < chunks; c += gridDim.z) {
        const uint32_t v   = c * kWarpSize + lane;
        const bool active  = v < nvec;
        const int64_t elem = static_cast<int64_t>(v) * VEC;

        alignas(16) float acc[VEC] = {};
        for (int s0 = lo + static_cast<int>(warp); s0 < hi; s0 += kSumMergeBatch * kSumMergeWarps) {
            alignas(16) float batch[kSumMergeBatch][VEC];
#pragma unroll
            for (uint32_t b = 0; b < kSumMergeBatch; ++b) {
                const int s = s0 + static_cast<int>(b * kSumMergeWarps);
                if (active && s < hi) {
                    load_words<VEC>(batch[b], part + static_cast<int64_t>(s) * D + elem);
                } else {
#pragma unroll
                    for (size_t e = 0; e < VEC; ++e) {
                        batch[b][e] = 0.0f;
                    }
                }
            }
#pragma unroll
            for (uint32_t b = 0; b < kSumMergeBatch; ++b) {
#pragma unroll
                for (size_t e = 0; e < VEC; ++e) {
                    acc[e] += batch[b][e];
                }
            }
        }

        if (warp > 0) {
            store_words<VEC>(sums[warp - 1][lane], *reinterpret_cast<vals_t const *>(acc));
        }
        __syncthreads();
        if (warp == 0 && active) {
#pragma unroll
            for (uint32_t w = 0; w + 1 < kSumMergeWarps; ++w) {
#pragma unroll
                for (size_t e = 0; e < VEC; ++e) {
                    acc[e] += sums[w][lane][e];
                }
            }
            Tile::write(out + node * D, v, Tile::convert_from_accum(acc));
        }
        __syncthreads();
    }
}

template <FloatingNum cuda_t, IntegralNum index_t>
inline void launch_sum_slices(
    ForwardParams<cuda_t, index_t> const& p, int num_heavy, int const *node_chunk_offset, bool wide, cudaStream_t stream
) {
    if (num_heavy == 0) {
        return;
    }
    const uint32_t chunks = ceil_div(p.nvec, static_cast<uint32_t>(kWarpSize));
    const dim3 grid(static_cast<unsigned>(num_heavy), 1, std::min(chunks, 65535u));
    std::visit(
        [&](auto wide_c) {
            sum_slices_kernel<decltype(wide_c)::value, cuda_t, index_t>
                <<<grid, kSumMergeWarps * kWarpSize, 0, stream>>>(p.heavy_nodes, num_heavy, node_chunk_offset, p.D, p.nvec, p.part, p.out);
        },
        MakeBoolVariant<false, true>(wide)
    );
}

template <ReductionOp OP, FloatingNum cuda_t, IntegralNum index_t, class M = NodeRows>
inline void launch_finalize_heavy(
    ForwardParams<cuda_t, index_t> const& p, int num_heavy, int const *node_chunk_offset, bool wide, cudaStream_t stream
) {
    if (num_heavy == 0) {
        return;
    }
    const auto [grid, threads] = rows::merge_launch(num_heavy, 1, p.nvec);
    std::visit(
        [&](auto wide_c) {
            finalize_heavy_kernel<OP, decltype(wide_c)::value, cuda_t, index_t, M><<<grid, threads, 0, stream>>>(
                p.heavy_nodes, num_heavy, node_chunk_offset, p.row_ptr, p.col_idx, p.X, p.stride_x, p.D, p.nvec, p.packed, p.out, p.arg, p.E
            );
        },
        MakeBoolVariant<false, true>(wide)
    );
}

}  // namespace turbo_gnn::reduction
