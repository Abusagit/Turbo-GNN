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
//            of the warp (32 or, past 256 vectors, 64 vectors) is swept in passes, every pass an item
//            of its own: the reduction being elementwise, passes need nothing from each other and no
//            chunked path.
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
// registers throughout; those keep the compiler's choice. Whole-warp kernels for other
// architectures than sm_90 spill inside the walk at these bounds (sm_80/86/100, 8-16 bytes), so
// they ask for 5 blocks, 48 registers -- spill-checked, not timed: no other GPU to measure on.
//
// OCCUPANCY = false drops the bound. When the longest rows set the finish -- a launch whose longest
// light row is long against each resident warp's share of the edges -- the unbounded build keeps
// more rows in flight per warp: measured on real graphs, the whole-warp fp32 min kernel ran 13-28%
// faster unbounded on tolokers-2 and hm-categories and 4-9% slower on flickr, ogbn-arxiv,
// twitch-views and reddit. Lane groups showed under 6% either way and have no unbounded twin. The
// host picks (reduction_aggr_forward_cuda).
template <ReductionOp OP, int SHAPE, FloatingNum cuda_t, IntegralNum index_t, bool OCCUPANCY = true>
inline constexpr int kForwardMinBlocks = [] {
    constexpr bool kSum   = OP == ReductionOp::SUM;
    constexpr bool kFp32  = sizeof(cuda_t) == 4;
    constexpr int kBlocks = !OCCUPANCY || sizeof(index_t) != 4 ? 1
                            : Shape<SHAPE>::G == kWarpSize     ? (Shape<SHAPE>::S == 1 && (kSum || kFp32) ? (kCudaArch != 900 ? 5
                                                                                                                : kSum        ? 8
                                                                                                                              : 6)
                                                                                                          : 1)
                            : kSum                             ? (kFp32 ? 6 : 5)
                                                               : (kFp32 ? 5 : 1);
    return std::min(kBlocks, kMaxThreadsPerSM / static_cast<int>(kForwardMaxWarps * kWarpSize));
}();
static_assert(kForwardMaxWarps * kWarpSize == 256, "the bounds above are for 256-thread blocks");

// Whether a configuration has an unbounded twin.
template <ReductionOp OP, int SHAPE, FloatingNum cuda_t, IntegralNum index_t>
inline constexpr bool kHasIlpTwin = Shape<SHAPE>::G == kWarpSize && kForwardMinBlocks<OP, SHAPE, cuda_t, index_t, true> > 1;

template <ReductionOp OP, int SHAPE, FloatingNum cuda_t, IntegralNum index_t, bool OCCUPANCY = true>
__global__ void __launch_bounds__(kForwardMaxWarps *kWarpSize, kForwardMinBlocks<OP, SHAPE, cuda_t, index_t, OCCUPANCY>)
reduction_forward_kernel(ForwardParams<cuda_t, index_t> const p) {
    using Sh                        = Shape<SHAPE>;
    static constexpr size_t G       = Sh::G;
    static constexpr size_t VEC     = Sh::template vec<cuda_t>;
    using RG                        = RowGroup<G, Sh::S, VEC, cuda_t, float>;
    using Frag                      = typename RG::Frag;
    static constexpr size_t kAccs   = RG::kAccs;
    static constexpr uint32_t kPass = rows::kPassVecs<SHAPE>;
    // Whole-warp rows; otherwise every lane group is a worker of its own.
    static constexpr bool kWarpRows = G == kWarpSize;
    // Only whole-warp shapes sweep a row in more than one pass; for lane groups passes == 1.
    static constexpr bool kMultiPass = kWarpRows;
    using Sentinel                   = IndexSentinel<index_t>;

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
    // the row stride within 32 bits; the id times the stride is formed in 64.
    //
    // Except for whole-warp sums of 16-bit rows: there the held addresses spill at the 32 registers
    // those kernels run best at, and at 40 they lose 4-14% to the predicated gather at 32.
    static constexpr bool kHoisted = !(kWarpRows && OP == ReductionOp::SUM && sizeof(cuda_t) == 2);
    cuda_t const *lane_x[Sh::S];
#pragma unroll
    for (size_t s = 0; s < Sh::S; ++s) {
        lane_x[s] = x_pass + static_cast<int64_t>(min(RG::vec_index(r, s), cnvec - 1)) * VEC;
    }
    const uint32_t stride_x = static_cast<uint32_t>(p.stride_x);

    Partial<OP, kAccs> acc;
    acc.init();
    auto load = [&](index_t j, bool active) {
        Frag f;
        if constexpr (kHoisted) {
            const uint64_t off = static_cast<uint64_t>(static_cast<std::make_unsigned_t<index_t>>(j)) * stride_x;
#pragma unroll
            for (size_t s = 0; s < Sh::S; ++s) {
                f.v[s] = RG::Tile::read(lane_x[s] + off, 0);
            }
        } else {
            RG::load(f, x_pass + static_cast<int64_t>(j) * p.stride_x, r, cnvec, active);
        }
        return f;
    };
    // pos: the edge's position in the row.
    auto use = [&](uint32_t pos, Frag const& f) {
        alignas(16) float x[kAccs];
        RG::to_accum(f, x);
        acc.update(x, pos);
    };
    if constexpr (kWarpRows) {
        for_each_neighbor_reverse<G, kForwardUnroll<OP, cuda_t, true>>(ids, lo, hi, lane, [&](uint32_t pos, bool active, index_t j) {
            const Frag f = load(j, active);
            if (active) {
                use(pos, f);
            }
        });
    } else {
        for_each_neighbor_reverse_unbatched<kForwardUnroll<OP, cuda_t, false>>(ids, lo, hi, load, use);
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
                    args[e]          = q == kNoPos ? Sentinel::INVALID : ids[q];
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

// ------------------------------------------------------------------------------------------------
// Heavy rows of min/max: the slices have folded their candidates into one packed key per element
// (see pack_candidate); this decodes the keys into the rows' values and args. One thread per row
// vector, grid (num_heavy, 1, vector blocks) as rows::merge_launch lays it out.
// ------------------------------------------------------------------------------------------------
template <ReductionOp OP, bool WIDE, FloatingNum cuda_t, IntegralNum index_t>
__global__ void __launch_bounds__(rows::kMergeThreads) finalize_heavy_kernel(
    index_t const *const __restrict__ heavy_nodes, int num_heavy, int const *const __restrict__ node_chunk_offset,
    index_t const *const __restrict__ row_ptr, index_t const *const __restrict__ col_idx, cuda_t const *const __restrict__ X, int64_t stride_x,
    uint32_t D, uint32_t nvec, unsigned long long const *const __restrict__ packed, cuda_t *const __restrict__ out,
    index_t *const __restrict__ arg
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
    index_t const *const ids = col_idx + row_ptr[node];

    alignas(16) unsigned long long keys[VEC];
    load_words<VEC>(keys, packed + static_cast<int64_t>(slot) * D + elem);
    alignas(16) float vals[VEC];
    alignas(16) index_t args[VEC];
#pragma unroll
    for (size_t e = 0; e < VEC; ++e) {
        if (keys[e] == kPackedEmpty<OP>) {
            vals[e] = 0.0f;
            args[e] = Sentinel::INVALID;
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

template <ReductionOp OP, FloatingNum cuda_t, IntegralNum index_t>
inline void launch_finalize_heavy(
    ForwardParams<cuda_t, index_t> const& p, int num_heavy, int const *node_chunk_offset, bool wide, cudaStream_t stream
) {
    if (num_heavy == 0) {
        return;
    }
    const auto [grid, threads] = rows::merge_launch(num_heavy, 1, p.nvec);
    std::visit(
        [&](auto wide_c) {
            finalize_heavy_kernel<OP, decltype(wide_c)::value, cuda_t, index_t><<<grid, threads, 0, stream>>>(
                p.heavy_nodes, num_heavy, node_chunk_offset, p.row_ptr, p.col_idx, p.X, p.stride_x, p.D, p.nvec, p.packed, p.out, p.arg
            );
        },
        MakeBoolVariant<false, true>(wide)
    );
}

}  // namespace turbo_gnn::reduction
