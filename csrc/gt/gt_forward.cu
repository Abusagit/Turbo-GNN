#include <cstdint>

#include "common.cuh"
#include "common/pipeline.cuh"
#include "common/row_group.cuh"
#include "gt/gt_common.cuh"

// ================================================================================================
// Graph transformer forward, for any head dim D in [1, kMaxHeadDim].
//
//   score(i, j) = scale * <K[i], Q[j]>        over the incoming edges j -> i of node i
//   O[i]        = sum_j softmax_j(score(i, j)) * V[j]
//   lse[i]      = log sum_j exp(score(i, j))   saved for the backward
//
// One block of W warps per (node, head) -- or per slice of a heavy node's edge list -- with the
// edges split into contiguous per-warp shares. Rows are laid out across lanes as described in
// gt/gt_common.cuh: the fused path keeps everything in registers and visits each edge once, the
// chunked path handles rows too long for that in two passes.
// ================================================================================================

namespace turbo_gnn::gt {

// Merges the softmax states of a block's warps -- lane 0 of each holding its warp's -- into `st`
// of every thread, and returns exp(m_warp - M): the factor that rescales this warp's accumulators
// to the block-wide max M. stats: 2 * W accum_t of shared memory. Must be called by every thread.
template <size_t W>
__device__ __forceinline__ float merge_block_softmax(OnlineSoftmaxState& st, float *const __restrict__ stats, uint32_t warp, uint32_t lane) {
    if constexpr (W == 1) {
        return 1.0f;
    } else {
        if (lane == 0) {
            stats[warp]     = st.max_val;
            stats[W + warp] = st.sum_exp;
        }
        __syncthreads();
        float block_max = -FLT_MAX;
#pragma unroll
        for (size_t w = 0; w < W; ++w) {
            block_max = fmaxf(block_max, stats[w]);
        }
        float block_sum = 0.0f;
#pragma unroll
        for (size_t w = 0; w < W; ++w) {
            block_sum = fmaf(stats[W + w], __expf(stats[w] - block_max), block_sum);
        }
        const float rescale = __expf(st.max_val - block_max);
        st.max_val          = block_max;
        st.sum_exp          = block_sum;
        return rescale;
    }
}

// ------------------------------------------------------------------------------------------------
// Fused path. Each lane group runs an online softmax over its edges with K[i], Q[j], V[j] and the
// accumulator in registers; the partial states are merged group -> warp (shuffles) -> block
// (shared memory) at the end.
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
    static constexpr size_t kRows = 2;  // staged per edge: Q[j], V[j]

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

        // Every group owns the same vectors of every row, so each needs the same part of K[i].
        Frag kf;
        RG::load(kf, p.K + node * p.stride_k_n + head * p.stride_k_h, r, p.nvec);
        st = OnlineSoftmaxState{};
        RG::zero(acc);

        // The head's offset is loop-invariant; folded into the per-edge expression it was not hoisted.
        cuda_t const *const q_head = p.Q + head * p.stride_q_h;
        cuda_t const *const v_head = p.V + head * p.stride_v_h;
        auto q_row                 = [&](index_t j) { return q_head + static_cast<int64_t>(j) * p.stride_q_n; };
        auto v_row                 = [&](index_t j) { return v_head + static_cast<int64_t>(j) * p.stride_v_n; };

        // Both rows are loaded before the score's shuffle chain starts, so the V gather is already
        // in flight while the reduction runs.
        auto edge = [&](bool active, Frag const& qf, Frag const& vf) {
            const accum_t score = RG::reduce(RG::dot(kf, qf)) * p.scale;
            if (active) {
                accum_t w;
                const accum_t rescale = st.update(score, w);
                RG::rescale_fma(acc, rescale, vf, w);
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
                    const index_t j = ids.get(share.x + it, lane, unused);
                    srcs[0]         = q_row(j);
                    srcs[1]         = v_row(j);
                },
                [&](uint32_t, cuda_t const *const(&rows)[kRows]) {
                    Frag qf, vf;
                    RG::load(qf, rows[0], r, p.nvec);
                    RG::load(vf, rows[1], r, p.nvec);
                    edge(true, qf, vf);
                }
            );
        } else {
            for_each_neighbor<G>(edges, share.x, share.y, lane, NoPayloadLoad{}, [&](uint32_t, bool active, index_t j, NoPayload) {
                Frag qf, vf;
                RG::load(qf, q_row(j), r, p.nvec, active);
                RG::load(vf, v_row(j), r, p.nvec, active);
                edge(active, qf, vf);
            });
        }

        // Groups -> warp: an online-softmax merge with each xor partner.
        if constexpr (G < kWarpSize) {
#pragma unroll
            for (uint32_t off = G; off < kWarpSize; off <<= 1) {
                const float2 c = st.merge(__shfl_xor_sync(FULL_WARP_MASK, st.max_val, off), __shfl_xor_sync(FULL_WARP_MASK, st.sum_exp, off));
#pragma unroll
                for (size_t e = 0; e < kAccs; ++e) {
                    const accum_t other = __shfl_xor_sync(FULL_WARP_MASK, acc[e], off);
                    acc[e]              = AdOps<accum_t>::fma(acc[e], c.x, other * c.y);
                }
            }
        }

        // Warps -> warp 0. The barrier in merge_block_softmax also retires every warp's use of the
        // ring, which the merge buffer aliases.
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
//   pass 2  the row is cut into chunks of 32 * VEC features, one vector per lane. The block's
//           warps form teams of T, one chunk per team at a time; a team's warps split the edges,
//           accumulate exp(score - max) * V[j] for their chunk in registers and sum within the
//           team. With at least as many chunks as warps T == 1 and no shared memory is touched;
//           with fewer, the edges are still spread over all warps.
//
// Traffic matches the fused path -- Q and V once per edge -- plus a 4-byte score written and read
// per chunk round, against rows of more than 1 KiB in the wide layout. K[i] is read once per edge
// in pass 1, from L1: the Q gathers bypass L1 so they do not evict it.
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

        // ---- pass 1: scores
        cuda_t const *const k_row  = p.K + node * p.stride_k_n + head * p.stride_k_h;
        st                         = OnlineSoftmaxState{};
        const uint2 share          = warp_share(n, warp, W, 1);
        cuda_t const *const q_head = p.Q + head * p.stride_q_h;
        for_each_neighbor<kWarpSize>(edges, share.x, share.y, lane, NoPayloadLoad{}, [&](uint32_t slot, bool, index_t j, NoPayload) {
            cuda_t const *const q_row = q_head + static_cast<int64_t>(j) * p.stride_q_n;
            accum_t part{};
#pragma unroll 4
            for (uint32_t v = lane; v < p.nvec; v += kWarpSize) {
                Tile::read(k_row, v).dot_product_(&part, Tile::template read<Tile::MemoryHint::L2Only>(q_row, v));
            }
            const accum_t score = warp_reduce_sum(part) * p.scale;
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
        // Teams: a power of two no larger than the chunk count (rounded up) or the warp count.
        uint32_t teams = 1;
        while (teams < W && teams < nchunks) {
            teams <<= 1;
        }
        const uint32_t T      = static_cast<uint32_t>(W) / teams;
        const uint32_t team   = warp / T;
        const uint32_t member = warp % T;
        const uint2 my_edges  = warp_share(n, member, T, kWarpSize);

        // Weight of a slot, computed once per slot by the lane that fetches its id.
        auto weight = [&](index_t, uint32_t slot) -> accum_t { return AccumOps::exp(scores[slot] - st.max_val); };

        for (uint32_t c0 = 0; c0 < nchunks; c0 += teams) {
            const uint32_t c     = c0 + team;
            const bool has_chunk = c < nchunks;
            const int64_t coff   = static_cast<int64_t>(c) * kWarpSize * VEC;
            const uint32_t cnvec = has_chunk ? min(p.nvec - c * static_cast<uint32_t>(kWarpSize), static_cast<uint32_t>(kWarpSize)) : 0;

            alignas(16) accum_t acc[kAccs];
            RG::zero(acc);
            if (has_chunk) {
                cuda_t const *const v_chunk = p.V + head * p.stride_v_h + coff;
                for_each_neighbor<kWarpSize, 4>(edges, my_edges.x, my_edges.y, lane, weight, [&](uint32_t, bool, index_t j, accum_t w) {
                    Frag vf;
                    RG::load(vf, v_chunk + static_cast<int64_t>(j) * p.stride_v_n, lane, cnvec);
                    RG::fma(acc, vf, w);
                });
            }
            team_sum<W, kWarpSize, kAccs, /*TRAILING_SYNC=*/true>(acc, reinterpret_cast<accum_t *>(smem), warp, lane, T);
            if (has_chunk && member == 0) {
                flush(coff, cnvec, acc);
            }
        }
    }
};

// ------------------------------------------------------------------------------------------------
// Kernel. SLICED selects the split-K heavy path: one block per slice of a heavy node's edge list,
// writing an un-normalised partial (accumulator, max, sum) that gt_forward_merge_slices_kernel
// folds into the node's row. Otherwise one block per node, through the node scheduler.
// ------------------------------------------------------------------------------------------------
template <
    sched::ScheduleKind SK, size_t W, int SHAPE, int STAGES, bool SLICED, FloatingNum cuda_t, IntegralNum index_t, FloatingNum accum_t = float
>
__global__ void __launch_bounds__(W *kWarpSize, kForwardMinBlocksPerSM<W>) gt_forward_kernel(
    ForwardParams<cuda_t, index_t, accum_t> const p, sched::SchedulerParams<index_t> sched_params
) {
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
        cuda_t *const o_row     = SLICED ? nullptr : p.O + node * p.stride_o_n + head * p.stride_o_h;
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

// Folds every slice partial of one heavy node into its output row and logsumexp: the same n-way
// online-softmax merge the slice kernel ran across warps, across slices, followed by the
// normalisation the slices deliberately skipped. Grid (num_heavy, H, vector blocks), one vector of
// the row per thread. A node whose slices are all empty comes out as a zero row with -inf
// logsumexp, as in the in-place kernel.

template <bool WIDE, FloatingNum cuda_t, IntegralNum index_t, FloatingNum accum_t = float>
__global__ void __launch_bounds__(kMergeThreads) gt_forward_merge_slices_kernel(
    ForwardParams<cuda_t, index_t, accum_t> const p, int const *__restrict__ node_chunk_offset, int num_heavy
) {
    static constexpr size_t VEC = WIDE ? 16 / sizeof(cuda_t) : 1;
    using Tile                  = TileOps<VEC, cuda_t, accum_t>;
    using AccumOps              = AdOps<accum_t>;

    const int slot     = blockIdx.x;
    const int64_t head = blockIdx.y;
    if (slot >= num_heavy || head >= p.H) [[unlikely]] {
        return;
    }
    const int64_t node = static_cast<int64_t>(p.heavy_nodes[slot]);
    const int lo       = node_chunk_offset[slot];
    const int hi       = node_chunk_offset[slot + 1];

    accum_t m = -FLT_MAX;
    for (int s = lo; s < hi; ++s) {
        m = AccumOps::max(m, p.part_ml[2 * (s * p.H + head)]);
    }
    accum_t l{};
    for (int s = lo; s < hi; ++s) {
        l = AccumOps::fma(p.part_ml[2 * (s * p.H + head) + 1], AccumOps::exp(p.part_ml[2 * (s * p.H + head)] - m), l);
    }
    if (blockIdx.z == 0 && threadIdx.x == 0) {
        p.logsumexp[node * p.H + head] = l > accum_t{} ? m + AccumOps::log(l) : -INFINITY;
    }

    const uint32_t v = blockIdx.z * blockDim.x + threadIdx.x;
    if (v >= p.nvec) {
        return;
    }
    alignas(16) accum_t acc[VEC] = {};
    for (int s = lo; s < hi; ++s) {
        const accum_t c          = AccumOps::exp(p.part_ml[2 * (s * p.H + head)] - m);
        accum_t const *const row = p.part_o + dense_row(s, head, p.H, p.D) + static_cast<int64_t>(v) * VEC;
#pragma unroll
        for (size_t e = 0; e < VEC; ++e) {
            acc[e] = AccumOps::fma(c, row[e], acc[e]);
        }
    }
    const accum_t inv = l > accum_t{} ? accum_t{1} / l : accum_t{};
#pragma unroll
    for (size_t e = 0; e < VEC; ++e) {
        acc[e] *= inv;
    }
    Tile::write(p.O + node * p.stride_o_n + head * p.stride_o_h, v, Tile::convert_from_accum(acc));
}

}  // namespace turbo_gnn::gt
