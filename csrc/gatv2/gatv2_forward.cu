#pragma once

#include <cstdint>

#include "common.cuh"
#include "common/pipeline.cuh"
#include "common/row_group.cuh"
#include "gatv2/gatv2_common.cuh"

// ================================================================================================
// GATv2 forward, for any head dim D in [1, rows::kMaxHeadDim].
//
//   e(i, j) = <a, LeakyReLU(l[i] + r[j])>    over the incoming edges j -> i of node i
//   out[i]  = sum_j softmax_j(e(i, j)) * r[j]
//   lse[i]  = log sum_j exp(e(i, j))          saved for the backward
//
// One block of W warps per (node, head) -- or per slice of a heavy node's edge list -- with the
// edges split into contiguous per-warp shares. The fused path keeps l[i], a and the accumulator in
// registers and gathers one row, r[j], per edge; the chunked path handles rows too long for that
// in two passes. The structure is the graph transformer forward's (gt/gt_forward.cu) with the
// score swapped, and one row gathered per edge instead of two.
// ================================================================================================

namespace turbo_gnn::gatv2 {

// ------------------------------------------------------------------------------------------------
// Fused path. Each lane group runs an online softmax over its edges; the partial states are merged
// group -> warp (shuffles) -> block (shared memory) at the end.
// ------------------------------------------------------------------------------------------------
template <size_t W, int SHAPE, int STAGES, FloatingNum cuda_t, IntegralNum index_t, FloatingNum accum_t>
struct ForwardFused {
    using Sh = Shape<SHAPE>;
    static_assert(!Sh::kChunked, "ForwardFused takes a fused shape");
    static_assert(STAGES == 0 || Sh::kPipelinable, "The cp.async ring needs whole-warp rows of 16-byte vectors");

    static constexpr size_t G     = Sh::G;
    static constexpr size_t VEC   = Sh::template vec<cuda_t>;
    using RG                      = RowGroup<G, Sh::S, VEC, cuda_t, accum_t>;
    using Frag                    = typename RG::Frag;
    using Params                  = ForwardParams<cuda_t, index_t, accum_t>;
    static constexpr size_t kAccs = RG::kAccs;
    static constexpr size_t kRows = 1;  // staged per edge: r[j]

    // [ cp.async ring | cross-warp merge buffer ] (aliased: the merge starts only once every warp
    // has left its edge loop) followed by 2 * W floats of softmax stats.
    static __host__ __device__ size_t stats_offset(uint32_t D) {
        const size_t ring  = W * pipelined_rows_ring_elems(STAGES, kRows, D) * sizeof(cuda_t);
        const size_t merge = kTeamSumBufferElems<W, G, kAccs, accum_t> * sizeof(accum_t);
        return ceil_div(ring > merge ? ring : merge, size_t{16}) * 16;
    }
    static __host__ __device__ size_t smem_bytes(uint32_t D) { return stats_offset(D) + 2 * W * sizeof(accum_t); }

    // Runs the block over the n edges of one row (ids at `edges`). On return warp 0 holds the
    // block's softmax state in every lane and, in lanes [0, G), the un-normalised accumulator for
    // vectors lane, lane + G, ...
    static __device__ __forceinline__ void run(
        Params const& p, int64_t node, int64_t head, index_t const *const __restrict__ edges, uint32_t n, uint8_t *smem, OnlineSoftmaxState& st,
        accum_t (&acc)[kAccs]
    ) {
        const uint32_t lane = threadIdx.x % kWarpSize;
        const uint32_t warp = threadIdx.x / kWarpSize;
        const uint32_t r    = lane % G;
        const accum_t ns    = p.negative_slope;

        // Every group owns the same vectors of every row, so each needs the same part of l[i] and a;
        // widened once here rather than on every edge.
        alignas(16) accum_t lf[kAccs], af[kAccs];
        {
            Frag f;
            RG::load(f, p.L + node * p.stride_l_n + head * p.stride_l_h, r, p.nvec);
            RG::to_accum(f, lf);
            RG::load(f, p.A + head * static_cast<int64_t>(p.D), r, p.nvec);
            RG::to_accum(f, af);
        }
        st = OnlineSoftmaxState{};
        RG::zero(acc);

        // The head's offset is loop-invariant; folded into the per-edge expression it was not hoisted.
        cuda_t const *const r_head = p.R + head * p.stride_r_h;
        auto r_row                 = [&](index_t j) { return r_head + static_cast<int64_t>(j) * p.stride_r_n; };

        // r[j] is widened once, for the score and the accumulation both.
        auto edge = [&](bool active, Frag const& rf) {
            alignas(16) accum_t rv[kAccs];
            RG::to_accum(rf, rv);
            const accum_t score = RG::reduce(score_part(lf, rv, af, ns));
            if (active) {
                accum_t w;
                const accum_t rescale = st.update(score, w);
#pragma unroll
                for (size_t e = 0; e < kAccs; ++e) {
                    acc[e] = AdOps<accum_t>::fma(w, rv[e], acc[e] * rescale);
                }
            }
        };

        const uint2 share = warp_share(n, warp, W, RG::kGroupsPerWarp);
        if constexpr (STAGES > 0) {
            cuda_t *const ring = reinterpret_cast<cuda_t *>(smem) + warp * pipelined_rows_ring_elems(STAGES, kRows, p.D);
            NeighborCache<index_t, NoPayloadLoad> ids(edges, share.y, NoPayloadLoad{});
            pipelined_rows_loop<STAGES, kRows>(
                lane, share.y - share.x, p.D, ring,
                [&](uint32_t it, cuda_t const *(&srcs)[kRows]) {
                    NoPayload unused;
                    srcs[0] = r_row(ids.get(share.x + it, lane, unused));
                },
                [&](uint32_t, cuda_t const *const(&staged)[kRows]) {
                    Frag rf;
                    RG::load(rf, staged[0], r, p.nvec);
                    edge(true, rf);
                }
            );
        } else {
            // A whole-warp group with one vector per lane takes one edge per step and has a single row
            // load in flight; unrolled by two, the next row's load issues before this one's softmax
            // update. Measured 8-12% faster for that shape (fp16 D = 256, fp32 D = 128), and slower
            // for the others, which already have 32 / G edges or two vectors per lane in flight.
            constexpr size_t kUnroll = G == kWarpSize && Sh::S == 1 ? 2 : 1;
            for_each_neighbor<G, kUnroll>(edges, share.x, share.y, lane, NoPayloadLoad{}, [&](uint32_t, bool active, index_t j, NoPayload) {
                Frag rf;
                RG::load(rf, r_row(j), r, p.nvec, active);
                edge(active, rf);
            });
        }

        // Groups -> warp, then warps -> warp 0. The barrier in merge_block_softmax also retires
        // every warp's use of the ring, which the merge buffer aliases.
        merge_groups_softmax<G>(st, acc);
        if constexpr (W > 1) {
            RG::scale(acc, merge_block_softmax<W>(st, reinterpret_cast<accum_t *>(smem + stats_offset(p.D)), warp, lane));
            team_sum<W, G, kAccs>(acc, reinterpret_cast<accum_t *>(smem), warp, lane, W);
        }
    }
};

// ------------------------------------------------------------------------------------------------
// Chunked path, for rows too long to keep in registers.
//
//   pass 1  warps split the edges; each warp computes full-row scores one edge at a time (lanes
//           stride over the row), writes them to the score scratch and keeps an online
//           (max, sum). The block's states are merged; the barrier also publishes the scores.
//   pass 2  the row is cut into chunks of 32 * VEC features, one vector per lane, swept by teams of
//           warps (see ChunkTeams); a team's warps split the edges and accumulate
//           exp(score - max) * r[j] for their chunk in registers.
//
// Traffic: r[j] twice per edge, against once on the fused path, plus a 4-byte score written and
// read per chunk round. l[i] and a are re-read per edge in pass 1, from L1: the r gathers bypass L1
// so they do not evict them.
// ------------------------------------------------------------------------------------------------
template <size_t W, bool WIDE, FloatingNum cuda_t, IntegralNum index_t, FloatingNum accum_t>
struct ForwardChunked {
    static constexpr size_t VEC   = WIDE ? 16 / sizeof(cuda_t) : 1;
    using RG                      = RowGroup<kWarpSize, 1, VEC, cuda_t, accum_t>;
    using Tile                    = typename RG::Tile;
    using Frag                    = typename RG::Frag;
    using Params                  = ForwardParams<cuda_t, index_t, accum_t>;
    using AccumOps                = AdOps<accum_t>;
    static constexpr size_t kAccs = RG::kAccs;

    // [ team-sum buffer | 2 * W softmax stats ]
    static constexpr size_t kStatsOffset = ceil_div(kTeamSumBufferElems<W, kWarpSize, kAccs, accum_t> * sizeof(accum_t), size_t{16}) * 16;
    static __host__ __device__ size_t smem_bytes(uint32_t) { return kStatsOffset + 2 * W * sizeof(accum_t); }

    // One vector of a row, widened.
    template <typename Tile::MemoryHint hint = Tile::MemoryHint::NoHint>
    static __device__ __forceinline__ void load_accum(accum_t (&out)[kAccs], cuda_t const *const __restrict__ row, uint32_t v) {
        Frag f;
        f.v[0] = Tile::template read<hint>(row, v);
        RG::to_accum(f, out);
    }

    // `scores` is this row's slice of the scratch. flush(coff, cnvec, acc) is called by one warp per
    // chunk with that chunk's un-normalised accumulator (lane = vector index within the chunk,
    // cnvec vectors valid, coff the chunk's first element); on return `st` holds the row's softmax
    // state in every thread.
    template <typename Flush>
    static __device__ __forceinline__ void run(
        Params const& p, int64_t node, int64_t head, index_t const *const __restrict__ edges, uint32_t n, accum_t *const __restrict__ scores,
        uint8_t *smem, OnlineSoftmaxState& st, Flush&& flush
    ) {
        const uint32_t lane = threadIdx.x % kWarpSize;
        const uint32_t warp = threadIdx.x / kWarpSize;
        const accum_t ns    = p.negative_slope;

        // ---- pass 1: scores
        cuda_t const *const l_row  = p.L + node * p.stride_l_n + head * p.stride_l_h;
        cuda_t const *const a_row  = p.A + head * static_cast<int64_t>(p.D);
        cuda_t const *const r_head = p.R + head * p.stride_r_h;
        st                         = OnlineSoftmaxState{};
        const uint2 share          = warp_share(n, warp, W, 1);
        for_each_neighbor<kWarpSize>(edges, share.x, share.y, lane, NoPayloadLoad{}, [&](uint32_t slot, bool, index_t j, NoPayload) {
            cuda_t const *const r_row = r_head + static_cast<int64_t>(j) * p.stride_r_n;
            accum_t part{};
#pragma unroll 2
            for (uint32_t v = lane; v < p.nvec; v += kWarpSize) {
                alignas(16) accum_t lv[kAccs], rv[kAccs], av[kAccs];
                load_accum(lv, l_row, v);
                load_accum<Tile::MemoryHint::L2Only>(rv, r_row, v);
                load_accum(av, a_row, v);
                part += score_part(lv, rv, av, ns);
            }
            const accum_t score = warp_reduce_sum(part);
            if (lane == 0) {
                scores[slot] = score;
            }
            accum_t w;
            st.update(score, w);
        });

        merge_block_softmax<W>(st, reinterpret_cast<accum_t *>(smem + kStatsOffset), warp, lane);
        if constexpr (W == 1) {
            __syncwarp();  // publishes lane 0's scores to the warp
        }

        // ---- pass 2: chunked accumulation
        const uint32_t nchunks = ceil_div(p.nvec, static_cast<uint32_t>(kWarpSize));
        const ChunkTeams ct    = chunk_teams<W>(nchunks, warp);
        const uint2 my_edges   = warp_share(n, ct.member, ct.T, kWarpSize);

        // Weight of a slot, computed once per slot by the lane that fetches its id.
        auto weight = [&](index_t, uint32_t slot) -> accum_t { return AccumOps::exp(scores[slot] - st.max_val); };

        for (uint32_t c0 = 0; c0 < nchunks; c0 += ct.teams) {
            const uint32_t c     = c0 + ct.team;
            const bool has_chunk = c < nchunks;
            const int64_t coff   = static_cast<int64_t>(c) * kWarpSize * VEC;
            const uint32_t cnvec = has_chunk ? min(p.nvec - c * static_cast<uint32_t>(kWarpSize), static_cast<uint32_t>(kWarpSize)) : 0;

            alignas(16) accum_t acc[kAccs];
            RG::zero(acc);
            if (has_chunk) {
                cuda_t const *const r_chunk = r_head + coff;
                for_each_neighbor<kWarpSize, 4>(edges, my_edges.x, my_edges.y, lane, weight, [&](uint32_t, bool, index_t j, accum_t w) {
                    Frag rf;
                    RG::load(rf, r_chunk + static_cast<int64_t>(j) * p.stride_r_n, lane, cnvec);
                    RG::fma(acc, rf, w);
                });
            }
            team_sum<W, kWarpSize, kAccs, /*TRAILING_SYNC=*/true>(acc, reinterpret_cast<accum_t *>(smem), warp, lane, ct.T);
            if (has_chunk && ct.member == 0) {
                flush(coff, cnvec, acc);
            }
        }
    }
};

// ------------------------------------------------------------------------------------------------
// Kernel. SLICED selects the split-K heavy path: one block per slice of a heavy node's edge list,
// writing an un-normalised partial (accumulator, max, sum) that rows::softmax_merge_slices_kernel
// folds into the node's row. Otherwise one block per node, through the node scheduler.
// ------------------------------------------------------------------------------------------------
template <
    sched::ScheduleKind SK, size_t W, int SHAPE, int STAGES, bool SLICED, FloatingNum cuda_t, IntegralNum index_t, FloatingNum accum_t = float
>
__global__ void __launch_bounds__(W *kWarpSize, kForwardMinBlocksPerSM<W, Shape<SHAPE>::S * Shape<SHAPE>::template vec<cuda_t>>)
gatv2_forward_kernel(ForwardParams<cuda_t, index_t, accum_t> const p, sched::SchedulerParams<index_t> sched_params) {
    using Sh                    = Shape<SHAPE>;
    static constexpr size_t VEC = Sh::template vec<cuda_t>;
    using AccumOps              = AdOps<accum_t>;

    extern __shared__ __align__(16) uint8_t sh_raw[];

    const int64_t head = blockIdx.y;
    if (head >= p.H) [[unlikely]] {
        return;
    }
    const uint32_t lane = threadIdx.x % kWarpSize;
    const uint32_t warp = threadIdx.x / kWarpSize;

    // One row: node `node`, edges [edge_begin, edge_begin + n); `out` is the node (in place) or the
    // slice (SLICED) the result is written for.
    auto process_row = [&](int64_t node, index_t edge_begin, uint32_t n, int64_t out) {
        cuda_t *const o_row     = SLICED ? nullptr : p.O + dense_row(node, head, p.H, p.D);
        accum_t *const part_row = SLICED ? p.part_o + dense_row(out, head, p.H, p.D) : nullptr;
        accum_t *const lse      = SLICED ? p.part_ml + 2 * (out * p.H + head) : p.logsumexp + node * p.H + head;

        auto write_stats = [&](OnlineSoftmaxState const& st) {
            if constexpr (SLICED) {
                lse[0] = st.max_val;
                lse[1] = st.sum_exp;
            } else {
                lse[0] = st.sum_exp > accum_t{} ? st.max_val + AccumOps::log(st.sum_exp) : -INFINITY;
            }
        };

        if (n == 0) [[unlikely]] {
            // No edges: zero output and -inf logsumexp; for a slice, the identity partial.
            if constexpr (SLICED) {
                zero_row<1>(part_row, p.D, threadIdx.x, blockDim.x);
            } else {
                zero_row<VEC>(o_row, p.nvec, threadIdx.x, blockDim.x);
            }
            if (threadIdx.x == 0) {
                write_stats(OnlineSoftmaxState{});
            }
            return;
        }

        index_t const *const edges = p.col_idx + edge_begin;
        if constexpr (Sh::kChunked) {
            using C = ForwardChunked<W, Sh::kWide, cuda_t, index_t, accum_t>;
            OnlineSoftmaxState st;
            accum_t *const scores = p.scores + head * p.num_edges + static_cast<int64_t>(edge_begin);
            C::run(p, node, head, edges, n, scores, sh_raw, st, [&](int64_t coff, uint32_t cnvec, accum_t(&acc)[C::kAccs]) {
                if constexpr (SLICED) {
                    C::RG::store_accum(part_row + coff, acc, lane, cnvec);
                } else {
                    C::RG::store(o_row + coff, acc, accum_t{1} / st.sum_exp, lane, cnvec);
                }
            });
            if (threadIdx.x == 0) {
                write_stats(st);
            }
        } else {
            using F = ForwardFused<W, SHAPE, STAGES, cuda_t, index_t, accum_t>;
            OnlineSoftmaxState st;
            alignas(16) accum_t acc[F::kAccs];
            F::run(p, node, head, edges, n, sh_raw, st, acc);
            if (warp == 0) {
                if (lane < F::G) {
                    if constexpr (SLICED) {
                        F::RG::store_accum(part_row, acc, lane, p.nvec);
                    } else {
                        F::RG::store(o_row, acc, accum_t{1} / st.sum_exp, lane, p.nvec);
                    }
                }
                if (lane == 0) {
                    write_stats(st);
                }
            }
        }
    };

    if constexpr (SLICED) {
        const int slice = blockIdx.x;
        if (slice >= p.num_slices) [[unlikely]] {
            return;
        }
        const int64_t node      = static_cast<int64_t>(p.heavy_nodes[p.chunk_node[slice]]);
        const index_t row_begin = p.row_ptr[node];
        const uint32_t degree   = static_cast<uint32_t>(p.row_ptr[node + 1] - row_begin);
        const uint32_t lo       = static_cast<uint32_t>(p.chunk_start[slice]);
        const uint32_t hi       = min(lo + static_cast<uint32_t>(p.slice_size), degree);
        process_row(node, row_begin + static_cast<index_t>(lo), hi > lo ? hi - lo : 0, slice);
    } else {
        using Sched = sched::NodeScheduler<SK, index_t, /*SyncBlock=*/true>;
        __shared__ typename Sched::SharedStorage sched_smem;
        Sched sched(sched_params, sched_smem);
        for (auto work = sched.first(); sched.valid(work); work = sched.next(work)) {
            const int64_t node = static_cast<int64_t>(sched.node(work));
            if (node >= p.N) [[unlikely]] {
                continue;
            }
            const index_t row_begin = p.row_ptr[node];
            process_row(node, row_begin, static_cast<uint32_t>(p.row_ptr[node + 1] - row_begin), node);
        }
    }
}

}  // namespace turbo_gnn::gatv2
