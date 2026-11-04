#include <cstdint>

#include "common.cuh"
#include "common/pipeline.cuh"
#include "common/row_group.cuh"
#include "gt/gt_common.cuh"

// ================================================================================================
// Graph transformer backward, for any head dim D in [1, kMaxHeadDim].
//
// With alpha(i, j) = exp(score(i, j) - lse[i]) recomputed from the saved logsumexp and
// Delta[i] = <dO[i], O[i]>, every edge j -> i contributes
//
//   dS(i, j) = alpha(i, j) * (<dO[i], V[j]> - Delta[i]) * scale
//   dV[j] += alpha(i, j) * dO[i]       dQ[j] += dS(i, j) * K[i]       dK[i] += dS(i, j) * Q[j]
//
// Both modes (gt/gt_common.cuh) run one block of W warps per row node d, its neighbours s split
// into per-warp shares, rows laid out across lanes as in the forward. In the notation of a row,
// the "reverse" half of an edge (d is the j above, s the i) gives dQ[d] and dV[d]; the "forward"
// half (d is the i, s the j) gives dK[d]. A directed launch walks the CSR^T, runs the reverse half
// and scatters dK[s] with fp32 atomics; an undirected one walks the forward CSR, which is its own
// transpose, and runs both halves with no atomics at all.
// ================================================================================================

namespace turbo_gnn::gt {

// Gradient accumulators of a row, by index.
enum : size_t { kGradQ = 0, kGradV = 1, kGradK = 2 };

template <BackwardMode MODE>
inline constexpr size_t kBackwardGrads = MODE == BackwardMode::Undirected ? 3 : 2;  // dQ, dV (, dK)

// ------------------------------------------------------------------------------------------------
// Fused path: the node's own rows and the gradient accumulators in registers, each edge visited
// once. Partials are summed group -> warp (shuffles) -> block (shared memory, warp order) at the end.
// ------------------------------------------------------------------------------------------------
template <BackwardMode MODE, size_t W, int SHAPE, int STAGES, FloatingNum cuda_t, IntegralNum index_t, FloatingNum accum_t>
struct BackwardFused {
    using Sh = Shape<SHAPE>;
    static_assert(!Sh::kChunked, "BackwardFused takes a fused shape");
    static_assert(STAGES == 0 || Sh::kPipelinable, "The cp.async ring needs whole-warp rows of 16-byte vectors");

    static constexpr bool kUndirected = MODE == BackwardMode::Undirected;
    static constexpr size_t G         = Sh::G;
    static constexpr size_t VEC       = Sh::template vec<cuda_t>;
    using RG                          = RowGroup<G, Sh::S, VEC, cuda_t, accum_t>;
    using Frag                        = typename RG::Frag;
    using Params                      = BackwardParams<cuda_t, index_t, accum_t>;
    using AccumOps                    = AdOps<accum_t>;
    static constexpr size_t kAccs     = RG::kAccs;
    static constexpr size_t kGrads    = kBackwardGrads<MODE>;

    // Rows gathered per edge s: K[s] and dO[s] for the reverse half, Q[s] and V[s] for the forward one.
    enum : size_t { kRowK = 0, kRowO = 1, kRowQ = 2, kRowV = 3 };
    static constexpr size_t kRows = kUndirected ? 4 : 2;

    // [ cp.async ring | cross-warp merge buffer ], aliased: the merge starts after the barrier
    // that follows every warp's edge loop.
    static __host__ __device__ size_t smem_bytes(uint32_t D) {
        const size_t ring  = W * pipelined_rows_ring_elems(STAGES, kRows, D) * sizeof(cuda_t);
        const size_t merge = kTeamSumBufferElems<W, G, kGrads * kAccs, accum_t> * sizeof(accum_t);
        return ring > merge ? ring : merge;
    }

    // On return lanes [0, G) of warp 0 hold the row's gradients, grads[k] for vectors lane,
    // lane + G, ...
    static __device__ __forceinline__ void run(
        Params const& p, int64_t node, int64_t head, index_t const *const __restrict__ edges, uint32_t n, uint8_t *smem,
        accum_t (&grads)[kGrads][kAccs]
    ) {
        const uint32_t lane = threadIdx.x % kWarpSize;
        const uint32_t warp = threadIdx.x / kWarpSize;
        const uint32_t r    = lane % G;
        // A short row goes to warp 0 alone (see solo_row): the other warps exit before any barrier.
        const bool solo = solo_row<W, RG::kGroupsPerWarp>(n);
        if (solo && warp != 0) {
            return;
        }

        // The node's own rows: Q[d], V[d] for the reverse half; K[d], dO[d] for the forward one.
        Frag qd, vd, kd, od;
        RG::load(qd, p.Q + node * p.stride_q_n + head * p.stride_q_h, r, p.nvec);
        RG::load(vd, p.V + node * p.stride_v_n + head * p.stride_v_h, r, p.nvec);
        accum_t lse_d{}, delta_d{};
        if constexpr (kUndirected) {
            RG::load(kd, p.K + node * p.stride_k_n + head * p.stride_k_h, r, p.nvec);
            RG::load(od, p.dO + dense_row(node, head, p.H, p.D), r, p.nvec);
            lse_d   = p.logsumexp[node * p.H + head];
            delta_d = p.delta[node * p.H + head];
        }
#pragma unroll
        for (size_t k = 0; k < kGrads; ++k) {
            RG::zero(grads[k]);
        }

        // Head offsets are loop-invariant; folded into the per-edge expression they were not hoisted.
        cuda_t const *const k_head = p.K + head * p.stride_k_h;
        cuda_t const *const o_head = p.dO + head * static_cast<int64_t>(p.D);
        cuda_t const *const q_head = p.Q + head * p.stride_q_h;
        cuda_t const *const v_head = p.V + head * p.stride_v_h;
        const int64_t stride_o_n   = p.H * static_cast<int64_t>(p.D);
        auto row_of                = [&](size_t which, index_t s) {
            const int64_t si = static_cast<int64_t>(s);
            switch (which) {
                case kRowK:
                    return k_head + si * p.stride_k_n;
                case kRowO:
                    return o_head + si * stride_o_n;
                case kRowQ:
                    return q_head + si * p.stride_q_n;
                default:
                    return v_head + si * p.stride_v_n;
            }
        };
        // The neighbour's (lse, Delta), fetched with its id.
        auto stats_of = [&](index_t s, uint32_t) -> float2 {
            const int64_t idx = static_cast<int64_t>(s) * p.H + head;
            return make_float2(p.logsumexp[idx], p.delta[idx]);
        };

        auto edge = [&](bool active, index_t s, Frag const(&rows)[kRows], float2 stats_s) {
            // Every shuffle chain is started before any of them is consumed, so they overlap.
            const accum_t qk = RG::reduce(RG::dot(qd, rows[kRowK]));
            const accum_t vo = RG::reduce(RG::dot(vd, rows[kRowO]));
            accum_t kq{}, ov{};
            if constexpr (kUndirected) {
                kq = RG::reduce(RG::dot(kd, rows[kRowQ]));
                ov = RG::reduce(RG::dot(od, rows[kRowV]));
            }
            // Reverse half: d attends from s.
            const accum_t alpha_r = AccumOps::exp(qk * p.scale - stats_s.x);
            const accum_t ds_r    = alpha_r * (vo - stats_s.y) * p.scale;
            if (active) {
                RG::fma(grads[kGradQ], rows[kRowK], ds_r);
                RG::fma(grads[kGradV], rows[kRowO], alpha_r);
                if constexpr (kUndirected) {
                    // Forward half: d attends to s.
                    const accum_t alpha_f = AccumOps::exp(kq * p.scale - lse_d);
                    const accum_t ds_f    = alpha_f * (ov - delta_d) * p.scale;
                    RG::fma(grads[kGradK], rows[kRowQ], ds_f);
                }
            }
            if constexpr (!kUndirected) {
                // Outside the branch: the paired atomics shuffle across the warp.
                RG::atomic_add(p.dK_atomic + dense_row(static_cast<int64_t>(s), head, p.H, p.D), qd, ds_r, r, p.nvec, active);
            }
        };

        const uint2 share = solo ? make_uint2(0, n) : warp_share(n, warp, W, RG::kGroupsPerWarp);
        if constexpr (STAGES > 0) {
            cuda_t *const ring = reinterpret_cast<cuda_t *>(smem) + warp * pipelined_rows_ring_elems(STAGES, kRows, p.D);
            // Producer and consumer walk the share at different points, so each has its own cache.
            NeighborCache<index_t, NoPayloadLoad> prefetch_ids(edges, share.y, NoPayloadLoad{});
            NeighborCache<index_t, decltype(stats_of)> consume_ids(edges, share.y, stats_of);
            pipelined_rows_loop<STAGES, kRows>(
                lane, share.y - share.x, p.D, ring,
                [&](uint32_t it, cuda_t const *(&srcs)[kRows]) {
                    NoPayload unused;
                    const index_t s = prefetch_ids.get(share.x + it, lane, unused);
#pragma unroll
                    for (size_t k = 0; k < kRows; ++k) {
                        srcs[k] = row_of(k, s);
                    }
                },
                [&](uint32_t it, cuda_t const *const(&staged)[kRows]) {
                    float2 stats_s;
                    const index_t s = consume_ids.get(share.x + it, lane, stats_s);
                    Frag rows[kRows];
#pragma unroll
                    for (size_t k = 0; k < kRows; ++k) {
                        RG::load(rows[k], staged[k], r, p.nvec);
                    }
                    edge(true, s, rows, stats_s);
                }
            );
        } else {
            for_each_neighbor<G>(edges, share.x, share.y, lane, stats_of, [&](uint32_t, bool active, index_t s, float2 stats_s) {
                Frag rows[kRows];
#pragma unroll
                for (size_t k = 0; k < kRows; ++k) {
                    RG::load(rows[k], row_of(k, s), r, p.nvec, active);
                }
                edge(active, s, rows, stats_s);
            });
        }

#pragma unroll
        for (size_t k = 0; k < kGrads; ++k) {
            RG::sum_groups(grads[k]);
        }
        if (W > 1 && !solo) {  // block-uniform: a solo row has no other warp to merge with
            if constexpr (STAGES > 0) {
                __syncthreads();  // every warp is out of its loop: the merge buffer may overwrite the ring
            }
            team_sum<W, G, kGrads * kAccs>(&grads[0][0], reinterpret_cast<accum_t *>(smem), warp, lane, W);
        }
    }
};

// ------------------------------------------------------------------------------------------------
// Chunked path, for rows too long to keep in registers (see ForwardChunked for the structure).
//
//   pass 1  one warp per edge: the full-row dot products give the edge's scalars -- alpha and dS
//           of the reverse half, and dS of the forward half when undirected -- written to the
//           scratch.
//   pass 2  teams of warps sweep 32 * VEC-wide feature chunks, accumulating the gradients of their
//           chunk from the scalars; a directed launch scatters its dK[s] chunk there.
// ------------------------------------------------------------------------------------------------
template <BackwardMode MODE, size_t W, bool WIDE, FloatingNum cuda_t, IntegralNum index_t, FloatingNum accum_t>
struct BackwardChunked {
    static constexpr bool kUndirected = MODE == BackwardMode::Undirected;
    static constexpr size_t VEC       = WIDE ? 16 / sizeof(cuda_t) : 1;
    using RG                          = RowGroup<kWarpSize, 1, VEC, cuda_t, accum_t>;
    using Tile                        = typename RG::Tile;
    using Frag                        = typename RG::Frag;
    using Params                      = BackwardParams<cuda_t, index_t, accum_t>;
    using AccumOps                    = AdOps<accum_t>;
    using Hint                        = typename Tile::MemoryHint;
    static constexpr size_t kAccs     = RG::kAccs;
    static constexpr size_t kGrads    = kBackwardGrads<MODE>;

    // Per-edge scalars: alpha and dS of the reverse half, dS of the forward half.
    enum : size_t { kAlphaR = 0, kDsR = 1, kDsF = 2 };
    static constexpr size_t kScalars = kUndirected ? 3 : 2;
    using scalars_t                  = std::conditional_t<kUndirected, float3, float2>;

    static __host__ __device__ size_t smem_bytes(uint32_t) {
        return kTeamSumBufferElems<W, kWarpSize, kGrads * kAccs, accum_t> * sizeof(accum_t);
    }

    // scalars: this row's slice of the scratch, scalar k at scalars + k * stride. flush(coff, cnvec,
    // grads) is called by one warp per chunk (lane = vector within the chunk, cnvec of them valid).
    template <typename Flush>
    static __device__ __forceinline__ void run(
        Params const& p, int64_t node, int64_t head, index_t const *const __restrict__ edges, uint32_t n, accum_t *const __restrict__ scalars,
        int64_t stride, uint8_t *smem, Flush&& flush
    ) {
        const uint32_t lane = threadIdx.x % kWarpSize;
        const uint32_t warp = threadIdx.x / kWarpSize;

        cuda_t const *const q_d = p.Q + node * p.stride_q_n + head * p.stride_q_h;
        cuda_t const *const v_d = p.V + node * p.stride_v_n + head * p.stride_v_h;
        cuda_t const *const k_d = p.K + node * p.stride_k_n + head * p.stride_k_h;
        cuda_t const *const o_d = p.dO + dense_row(node, head, p.H, p.D);

        // ---- pass 1: per-edge scalars
        accum_t lse_d{}, delta_d{};
        if constexpr (kUndirected) {
            lse_d   = p.logsumexp[node * p.H + head];
            delta_d = p.delta[node * p.H + head];
        }
        auto stats_of = [&](index_t s, uint32_t) -> float2 {
            const int64_t idx = static_cast<int64_t>(s) * p.H + head;
            return make_float2(p.logsumexp[idx], p.delta[idx]);
        };
        const uint2 share = warp_share(n, warp, W, 1);
        for_each_neighbor<kWarpSize>(edges, share.x, share.y, lane, stats_of, [&](uint32_t slot, bool, index_t s, float2 stats_s) {
            const int64_t si        = static_cast<int64_t>(s);
            cuda_t const *const k_s = p.K + si * p.stride_k_n + head * p.stride_k_h;
            cuda_t const *const o_s = p.dO + dense_row(si, head, p.H, p.D);
            cuda_t const *const q_s = p.Q + si * p.stride_q_n + head * p.stride_q_h;
            cuda_t const *const v_s = p.V + si * p.stride_v_n + head * p.stride_v_h;
            accum_t qk{}, vo{}, kq{}, ov{};
#pragma unroll 2
            for (uint32_t v = lane; v < p.nvec; v += kWarpSize) {
                // The neighbour's rows stream through L2 only; the node's are re-read per edge and
                // should stay in L1.
                Tile::read(q_d, v).dot_product_(&qk, Tile::template read<Hint::L2Only>(k_s, v));
                Tile::read(v_d, v).dot_product_(&vo, Tile::template read<Hint::L2Only>(o_s, v));
                if constexpr (kUndirected) {
                    Tile::read(k_d, v).dot_product_(&kq, Tile::template read<Hint::L2Only>(q_s, v));
                    Tile::read(o_d, v).dot_product_(&ov, Tile::template read<Hint::L2Only>(v_s, v));
                }
            }
            qk                    = warp_reduce_sum(qk);
            vo                    = warp_reduce_sum(vo);
            const accum_t alpha_r = AccumOps::exp(qk * p.scale - stats_s.x);
            if (lane == 0) {
                scalars[kAlphaR * stride + slot] = alpha_r;
                scalars[kDsR * stride + slot]    = alpha_r * (vo - stats_s.y) * p.scale;
            }
            if constexpr (kUndirected) {
                kq                    = warp_reduce_sum(kq);
                ov                    = warp_reduce_sum(ov);
                const accum_t alpha_f = AccumOps::exp(kq * p.scale - lse_d);
                if (lane == 0) {
                    scalars[kDsF * stride + slot] = alpha_f * (ov - delta_d) * p.scale;
                }
            }
        });
        // Publish the scalars to the block.
        if constexpr (W > 1) {
            __syncthreads();
        } else {
            __syncwarp();
        }

        // ---- pass 2: chunked accumulation
        const uint32_t nchunks = ceil_div(p.nvec, static_cast<uint32_t>(kWarpSize));
        const ChunkTeams ct    = chunk_teams<W>(nchunks, warp);
        const uint2 my_edges   = warp_share(n, ct.member, ct.T, kWarpSize);

        auto scalars_of = [&](index_t, uint32_t slot) -> scalars_t {
            if constexpr (kUndirected) {
                return make_float3(scalars[kAlphaR * stride + slot], scalars[kDsR * stride + slot], scalars[kDsF * stride + slot]);
            } else {
                return make_float2(scalars[kAlphaR * stride + slot], scalars[kDsR * stride + slot]);
            }
        };

        for (uint32_t c0 = 0; c0 < nchunks; c0 += ct.teams) {
            const uint32_t c     = c0 + ct.team;
            const bool has_chunk = c < nchunks;
            const int64_t coff   = static_cast<int64_t>(c) * kWarpSize * VEC;
            const uint32_t cnvec = has_chunk ? min(p.nvec - c * static_cast<uint32_t>(kWarpSize), static_cast<uint32_t>(kWarpSize)) : 0;

            alignas(16) accum_t grads[kGrads][kAccs];
#pragma unroll
            for (size_t k = 0; k < kGrads; ++k) {
                RG::zero(grads[k]);
            }
            if (has_chunk) {
                Frag qd;  // the node's Q chunk, scattered into dK[s] by a directed launch
                if constexpr (!kUndirected) {
                    RG::load(qd, q_d + coff, lane, cnvec);
                }
                cuda_t const *const k_chunk = p.K + head * p.stride_k_h + coff;
                cuda_t const *const o_chunk = p.dO + head * static_cast<int64_t>(p.D) + coff;
                cuda_t const *const q_chunk = p.Q + head * p.stride_q_h + coff;
                const int64_t stride_o_n    = p.H * static_cast<int64_t>(p.D);
                for_each_neighbor<kWarpSize, 2>(edges, my_edges.x, my_edges.y, lane, scalars_of, [&](uint32_t, bool, index_t s, scalars_t e) {
                    const int64_t si = static_cast<int64_t>(s);
                    Frag ks, os;
                    RG::load(ks, k_chunk + si * p.stride_k_n, lane, cnvec);
                    RG::load(os, o_chunk + si * stride_o_n, lane, cnvec);
                    RG::fma(grads[kGradQ], ks, e.y);
                    RG::fma(grads[kGradV], os, e.x);
                    if constexpr (kUndirected) {
                        Frag qs;
                        RG::load(qs, q_chunk + si * p.stride_q_n, lane, cnvec);
                        RG::fma(grads[kGradK], qs, e.z);
                    } else {
                        RG::atomic_add(p.dK_atomic + dense_row(si, head, p.H, p.D) + coff, qd, e.y, lane, cnvec);
                    }
                });
            }
            team_sum<W, kWarpSize, kGrads * kAccs, /*TRAILING_SYNC=*/true>(&grads[0][0], reinterpret_cast<accum_t *>(smem), warp, lane, ct.T);
            if (has_chunk && ct.member == 0) {
                flush(coff, cnvec, grads);
            }
        }
    }
};

// ------------------------------------------------------------------------------------------------
// Kernel. SLICED selects the split-K heavy path: one block per slice of a heavy node's edge list,
// writing fp32 partial gradients that rows::sum_merge_slices_kernel sums. The partials are plain
// sums -- alpha comes from the saved logsumexp, not from a running max -- so slicing changes only
// the order of additions. A directed launch's dK scatter is unaffected by slicing.
// ------------------------------------------------------------------------------------------------
template <
    BackwardMode MODE, sched::ScheduleKind SK, size_t W, int SHAPE, int STAGES, bool SLICED, FloatingNum cuda_t, IntegralNum index_t,
    FloatingNum accum_t = float
>
__global__ void __launch_bounds__(W *kWarpSize) gt_backward_kernel(
    BackwardParams<cuda_t, index_t, accum_t> const p, sched::SchedulerParams<index_t> sched_params
) {
    using Sh                       = Shape<SHAPE>;
    static constexpr size_t VEC    = Sh::template vec<cuda_t>;
    static constexpr size_t kGrads = kBackwardGrads<MODE>;

    extern __shared__ __align__(16) uint8_t sh_raw[];

    const int64_t head = blockIdx.y;
    if (head >= p.H) [[unlikely]] {
        return;
    }
    const uint32_t lane = threadIdx.x % kWarpSize;
    const uint32_t warp = threadIdx.x / kWarpSize;

    auto process_row = [&](int64_t node, index_t edge_begin, uint32_t n, int64_t out) {
        // Where gradient k of this row goes: the typed output (in place) or the fp32 partial (SLICED).
        cuda_t *const outs[3]   = {p.dQ, p.dV, p.dK};
        accum_t *const parts[3] = {p.part_q, p.part_v, p.part_k};
        const int64_t row_off   = dense_row(out, head, p.H, p.D);  // of `out` in outs (in place) or parts (SLICED)

        if (n == 0) [[unlikely]] {
#pragma unroll
            for (size_t k = 0; k < kGrads; ++k) {
                if constexpr (SLICED) {
                    zero_row<1>(parts[k] + row_off, p.D, threadIdx.x, blockDim.x);
                } else {
                    zero_row<VEC>(outs[k] + row_off, p.nvec, threadIdx.x, blockDim.x);
                }
            }
            return;
        }

        index_t const *const edges = p.col_idx + edge_begin;
        if constexpr (Sh::kChunked) {
            using C                = BackwardChunked<MODE, W, Sh::kWide, cuda_t, index_t, accum_t>;
            const int64_t stride   = p.H * p.num_edges;
            accum_t *const scalars = p.edge_scalars + head * p.num_edges + static_cast<int64_t>(edge_begin);
            C::run(p, node, head, edges, n, scalars, stride, sh_raw, [&](int64_t coff, uint32_t cnvec, accum_t(&grads)[kGrads][C::kAccs]) {
#pragma unroll
                for (size_t k = 0; k < kGrads; ++k) {
                    if constexpr (SLICED) {
                        C::RG::store_accum(parts[k] + row_off + coff, grads[k], lane, cnvec);
                    } else {
                        C::RG::store(outs[k] + row_off + coff, grads[k], accum_t{1}, lane, cnvec);
                    }
                }
            });
        } else {
            using F = BackwardFused<MODE, W, SHAPE, STAGES, cuda_t, index_t, accum_t>;
            alignas(16) accum_t grads[kGrads][F::kAccs];
            F::run(p, node, head, edges, n, sh_raw, grads);
            if (warp == 0 && lane < F::G) {
#pragma unroll
                for (size_t k = 0; k < kGrads; ++k) {
                    if constexpr (SLICED) {
                        F::RG::store_accum(parts[k] + row_off, grads[k], lane, p.nvec);
                    } else {
                        F::RG::store(outs[k] + row_off, grads[k], accum_t{1}, lane, p.nvec);
                    }
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

}  // namespace turbo_gnn::gt
