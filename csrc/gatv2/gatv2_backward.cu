#pragma once

#include <cstdint>

#include "common.cuh"
#include "common/pipeline.cuh"
#include "common/row_group.cuh"
#include "gatv2/gatv2_common.cuh"

// ================================================================================================
// GATv2 backward, for any head dim D in [1, rows::kMaxHeadDim].
//
// With g = dL/dout, alpha(i, j) = exp(e(i, j) - lse[i]) recomputed from the saved logsumexp and
// Delta[i] = <g[i], out[i]>, every edge j -> i contributes
//
//   de(i, j) = alpha(i, j) * (<g[i], r[j]> - Delta[i])            (dL/de)
//   dl[i] += de * a * LeakyReLU'(s)      da += de * LeakyReLU(s)
//   dr[j] += alpha(i, j) * g[i] + de * a * LeakyReLU'(s)          with s = l[i] + r[j]
//
// Delta is <g, out> because sum_j alpha(i, j) <g[i], r[j]> = <g[i], out[i]>: one dense pass over
// the rows (rows::row_dot_kernel) instead of a pass over every node's edges, which is what the
// previous kernels spent the first half of their directed launch on.
//
// The two modes (gatv2/gatv2_common.cuh) run one block of W warps per row node d of the CSR they
// walk, d's neighbours s split into per-warp shares, rows laid out across lanes as in the forward.
// AL is d attending to s and gives dl[d] and da; R is s attending to d and gives dr[d]. Row
// gradients are written once, without atomics; the gradient of `a` is summed per block and added
// into one of kGradASlots fp32 rows.
// ================================================================================================

namespace turbo_gnn::gatv2 {

// ------------------------------------------------------------------------------------------------
// Fused path: the node's own rows (widened to fp32) and the gradient accumulators in registers,
// each edge visited once. Partials are summed group -> warp (shuffles) -> block (shared memory,
// warp order) at the end.
// ------------------------------------------------------------------------------------------------
template <BackwardMode MODE, size_t W, int SHAPE, int STAGES, FloatingNum cuda_t, IntegralNum index_t, FloatingNum accum_t>
struct BackwardFused {
    using Sh = Shape<SHAPE>;
    static_assert(!Sh::kChunked, "BackwardFused takes a fused shape");
    static_assert(STAGES == 0 || Sh::kPipelinable, "The cp.async ring needs whole-warp rows of 16-byte vectors");

    static constexpr bool kAL     = MODE == BackwardMode::AL;
    static constexpr size_t G     = Sh::G;
    static constexpr size_t VEC   = Sh::template vec<cuda_t>;
    using RG                      = RowGroup<G, Sh::S, VEC, cuda_t, accum_t>;
    using Frag                    = typename RG::Frag;
    using Params                  = BackwardParams<cuda_t, index_t, accum_t>;
    using AccumOps                = AdOps<accum_t>;
    static constexpr size_t kAccs = RG::kAccs;
    static constexpr size_t kGrad = kBackwardGrads<MODE>;
    // Rows gathered per edge s: r[s] (AL); l[s] and g[s] (R).
    static constexpr size_t kRows = kAL ? 1 : 2;

    // [ cp.async ring | cross-warp merge buffer ], aliased: the merge starts after the barrier
    // that follows every warp's edge loop.
    static __host__ __device__ size_t smem_bytes(uint32_t D) {
        const size_t ring  = W * pipelined_rows_ring_elems(STAGES, kRows, D) * sizeof(cuda_t);
        const size_t merge = kTeamSumBufferElems<W, G, kGrad * kAccs, accum_t> * sizeof(accum_t);
        return ring > merge ? ring : merge;
    }

    // On return lanes [0, G) of warp 0 hold the row's gradients -- dl, da (AL) or dr (R) -- for
    // vectors lane, lane + G, ...
    static __device__ __forceinline__ void run(
        Params const& p, int64_t node, int64_t head, index_t const *const __restrict__ edges, uint32_t n, uint8_t *smem,
        accum_t (&grads)[kGrad][kAccs]
    ) {
        const uint32_t lane = threadIdx.x % kWarpSize;
        const uint32_t warp = threadIdx.x / kWarpSize;
        const uint32_t r    = lane % G;
        // A short row goes to warp 0 alone (see solo_row): the other warps exit before any barrier.
        const bool solo = solo_row<W, RG::kGroupsPerWarp>(n);
        if (solo && warp != 0) {
            return;
        }
        const accum_t ns = p.negative_slope;

        // The node's own rows, widened once: a; and l[d], g[d] (AL) or r[d] (R).
        [[maybe_unused]] alignas(16) accum_t af[kAccs], ld[kAccs], gd[kAccs], rd[kAccs];
        {
            Frag f;
            RG::load(f, p.A + head * static_cast<int64_t>(p.D), r, p.nvec);
            RG::to_accum(f, af);
            if constexpr (kAL) {
                RG::load(f, p.L + node * p.stride_l_n + head * p.stride_l_h, r, p.nvec);
                RG::to_accum(f, ld);
                RG::load(f, p.G + node * p.stride_g_n + head * p.stride_g_h, r, p.nvec);
                RG::to_accum(f, gd);
            } else {
                RG::load(f, p.R + node * p.stride_r_n + head * p.stride_r_h, r, p.nvec);
                RG::to_accum(f, rd);
            }
        }
        [[maybe_unused]] accum_t lse_d{}, delta_d{};
        if constexpr (kAL) {
            lse_d   = p.logsumexp[node * p.H + head];
            delta_d = p.delta[node * p.H + head];
        }
#pragma unroll
        for (size_t k = 0; k < kGrad; ++k) {
            RG::zero(grads[k]);
        }

        // Head offsets are loop-invariant; folded into the per-edge expression they were not hoisted.
        cuda_t const *const r_head = p.R + head * p.stride_r_h;
        cuda_t const *const l_head = p.L + head * p.stride_l_h;
        cuda_t const *const g_head = p.G + head * p.stride_g_h;
        auto row_of                = [&](size_t which, index_t s) {
            const int64_t si = static_cast<int64_t>(s);
            if constexpr (kAL) {
                return r_head + si * p.stride_r_n;
            } else {
                return which == 0 ? l_head + si * p.stride_l_n : g_head + si * p.stride_g_n;
            }
        };
        // R needs the neighbour's (lse, Delta), fetched with its id.
        auto payload = [&] {
            if constexpr (kAL) {
                return NoPayloadLoad{};
            } else {
                return [&](index_t s, uint32_t) -> float2 {
                    const int64_t idx = static_cast<int64_t>(s) * p.H + head;
                    return make_float2(p.logsumexp[idx], p.delta[idx]);
                };
            }
        }();

        // Both shuffle chains are started before either is consumed, so they overlap.
        auto edge = [&](bool active, Frag const(&rows)[kRows], auto stats_s) {
            if constexpr (kAL) {
                alignas(16) accum_t rs[kAccs];
                RG::to_accum(rows[0], rs);
                const accum_t e  = RG::reduce(score_part(ld, rs, af, ns));
                const accum_t pg = RG::reduce(dot_part(gd, rs));
                if (active) {
                    const accum_t alpha = AccumOps::exp(e - lse_d);
                    accum_left(grads[0], grads[1], alpha * (pg - delta_d), ld, rs, af, ns);
                }
            } else {
                alignas(16) accum_t ls[kAccs], gs[kAccs];
                RG::to_accum(rows[0], ls);
                RG::to_accum(rows[1], gs);
                const accum_t e  = RG::reduce(score_part(ls, rd, af, ns));
                const accum_t pg = RG::reduce(dot_part(gs, rd));
                if (active) {
                    const accum_t alpha = AccumOps::exp(e - stats_s.x);
                    accum_right(grads[0], alpha, gs, alpha * (pg - stats_s.y), ls, rd, af, ns);
                }
            }
        };

        const uint2 share = solo ? make_uint2(0, n) : warp_share(n, warp, W, RG::kGroupsPerWarp);
        if constexpr (STAGES > 0) {
            cuda_t *const ring = reinterpret_cast<cuda_t *>(smem) + warp * pipelined_rows_ring_elems(STAGES, kRows, p.D);
            // Producer and consumer walk the share at different points, so each has its own cache.
            NeighborCache<index_t, NoPayloadLoad> prefetch_ids(edges, share.y, NoPayloadLoad{});
            NeighborCache<index_t, decltype(payload)> consume_ids(edges, share.y, payload);
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
                    typename NeighborCache<index_t, decltype(payload)>::payload_t stats_s;
                    consume_ids.get(share.x + it, lane, stats_s);
                    Frag rows[kRows];
#pragma unroll
                    for (size_t k = 0; k < kRows; ++k) {
                        RG::load(rows[k], staged[k], r, p.nvec);
                    }
                    edge(true, rows, stats_s);
                }
            );
        } else {
            for_each_neighbor<G>(edges, share.x, share.y, lane, payload, [&](uint32_t, bool active, index_t s, auto stats_s) {
                Frag rows[kRows];
#pragma unroll
                for (size_t k = 0; k < kRows; ++k) {
                    RG::load(rows[k], row_of(k, s), r, p.nvec, active);
                }
                edge(active, rows, stats_s);
            });
        }

#pragma unroll
        for (size_t k = 0; k < kGrad; ++k) {
            RG::sum_groups(grads[k]);
        }
        if (W > 1 && !solo) {  // block-uniform: a solo row has no other warp to merge with
            if constexpr (STAGES > 0) {
                __syncthreads();  // every warp is out of its loop: the merge buffer may overwrite the ring
            }
            team_sum<W, G, kGrad * kAccs>(&grads[0][0], reinterpret_cast<accum_t *>(smem), warp, lane, W);
        }
    }
};

// ------------------------------------------------------------------------------------------------
// Chunked path, for rows too long to keep in registers (see ForwardChunked for the structure).
//
//   pass 1  one warp per edge: the full-row dot products give the edge's scalars -- de (AL), or
//           alpha and de (R) -- written to the scratch.
//   pass 2  teams of warps sweep 32 * VEC-wide feature chunks, accumulating the gradients of their
//           chunk from the scalars and the chunk's rows.
// ------------------------------------------------------------------------------------------------
template <BackwardMode MODE, size_t W, bool WIDE, FloatingNum cuda_t, IntegralNum index_t, FloatingNum accum_t>
struct BackwardChunked {
    static constexpr bool kAL     = MODE == BackwardMode::AL;
    static constexpr size_t VEC   = WIDE ? 16 / sizeof(cuda_t) : 1;
    using RG                      = RowGroup<kWarpSize, 1, VEC, cuda_t, accum_t>;
    using Tile                    = typename RG::Tile;
    using Frag                    = typename RG::Frag;
    using Params                  = BackwardParams<cuda_t, index_t, accum_t>;
    using AccumOps                = AdOps<accum_t>;
    using Hint                    = typename Tile::MemoryHint;
    static constexpr size_t kAccs = RG::kAccs;
    static constexpr size_t kGrad = kBackwardGrads<MODE>;
    // de (AL); alpha, de (R).
    using scalars_t = std::conditional_t<kAL, float, float2>;

    static __host__ __device__ size_t smem_bytes(uint32_t) {
        return kTeamSumBufferElems<W, kWarpSize, kGrad * kAccs, accum_t> * sizeof(accum_t);
    }

    // One vector (pass 1) or one chunk (pass 2: v = lane, nvec of them valid) of a row, widened.
    template <Hint hint = Hint::NoHint>
    static __device__ __forceinline__ void load_accum(accum_t (&out)[kAccs], cuda_t const *const __restrict__ row, uint32_t v, uint32_t nvec) {
        Frag f;
        RG::template load<hint>(f, row, v, nvec);
        RG::to_accum(f, out);
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
        const accum_t ns    = p.negative_slope;

        cuda_t const *const a_row  = p.A + head * static_cast<int64_t>(p.D);
        cuda_t const *const l_d    = p.L + node * p.stride_l_n + head * p.stride_l_h;
        cuda_t const *const g_d    = p.G + node * p.stride_g_n + head * p.stride_g_h;
        cuda_t const *const r_d    = p.R + node * p.stride_r_n + head * p.stride_r_h;
        cuda_t const *const r_head = p.R + head * p.stride_r_h;
        cuda_t const *const l_head = p.L + head * p.stride_l_h;
        cuda_t const *const g_head = p.G + head * p.stride_g_h;

        // ---- pass 1: per-edge scalars
        [[maybe_unused]] accum_t lse_d{}, delta_d{};
        if constexpr (kAL) {
            lse_d   = p.logsumexp[node * p.H + head];
            delta_d = p.delta[node * p.H + head];
        }
        auto payload = [&] {
            if constexpr (kAL) {
                return NoPayloadLoad{};
            } else {
                return [&](index_t s, uint32_t) -> float2 {
                    const int64_t idx = static_cast<int64_t>(s) * p.H + head;
                    return make_float2(p.logsumexp[idx], p.delta[idx]);
                };
            }
        }();
        const uint2 share = warp_share(n, warp, W, 1);
        for_each_neighbor<kWarpSize>(edges, share.x, share.y, lane, payload, [&](uint32_t slot, bool, index_t s, auto stats_s) {
            const int64_t si = static_cast<int64_t>(s);
            accum_t e{}, pg{};
#pragma unroll 2
            for (uint32_t v = lane; v < p.nvec; v += kWarpSize) {
                // The neighbour's rows stream through L2 only; the node's are re-read per edge and
                // should stay in L1.
                alignas(16) accum_t av[kAccs], x[kAccs], y[kAccs], z[kAccs];
                load_accum(av, a_row, v, p.nvec);
                if constexpr (kAL) {
                    load_accum(x, l_d, v, p.nvec);
                    load_accum(y, g_d, v, p.nvec);
                    load_accum<Hint::L2Only>(z, r_head + si * p.stride_r_n, v, p.nvec);
                    e += score_part(x, z, av, ns);
                    pg += dot_part(y, z);
                } else {
                    load_accum(x, r_d, v, p.nvec);
                    load_accum<Hint::L2Only>(y, l_head + si * p.stride_l_n, v, p.nvec);
                    load_accum<Hint::L2Only>(z, g_head + si * p.stride_g_n, v, p.nvec);
                    e += score_part(y, x, av, ns);
                    pg += dot_part(z, x);
                }
            }
            e  = warp_reduce_sum(e);
            pg = warp_reduce_sum(pg);
            if constexpr (kAL) {
                const accum_t alpha = AccumOps::exp(e - lse_d);
                if (lane == 0) {
                    scalars[slot] = alpha * (pg - delta_d);
                }
            } else {
                const accum_t alpha = AccumOps::exp(e - stats_s.x);
                if (lane == 0) {
                    scalars[slot]          = alpha;
                    scalars[stride + slot] = alpha * (pg - stats_s.y);
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
            if constexpr (kAL) {
                return scalars[slot];
            } else {
                return make_float2(scalars[slot], scalars[stride + slot]);
            }
        };

        for (uint32_t c0 = 0; c0 < nchunks; c0 += ct.teams) {
            const uint32_t c     = c0 + ct.team;
            const bool has_chunk = c < nchunks;
            const int64_t coff   = static_cast<int64_t>(c) * kWarpSize * VEC;
            const uint32_t cnvec = has_chunk ? min(p.nvec - c * static_cast<uint32_t>(kWarpSize), static_cast<uint32_t>(kWarpSize)) : 0;

            alignas(16) accum_t grads[kGrad][kAccs];
#pragma unroll
            for (size_t k = 0; k < kGrad; ++k) {
                RG::zero(grads[k]);
            }
            if (has_chunk) {
                // The node's chunk of its own rows, widened once per chunk: a, and l[d] (AL) or r[d] (R).
                alignas(16) accum_t ac[kAccs], oc[kAccs];
                load_accum(ac, a_row + coff, lane, cnvec);
                load_accum(oc, (kAL ? l_d : r_d) + coff, lane, cnvec);
                for_each_neighbor<kWarpSize, 2>(edges, my_edges.x, my_edges.y, lane, scalars_of, [&](uint32_t, bool, index_t s, scalars_t sc) {
                    const int64_t si = static_cast<int64_t>(s);
                    if constexpr (kAL) {
                        alignas(16) accum_t rs[kAccs];
                        load_accum(rs, r_head + coff + si * p.stride_r_n, lane, cnvec);
                        accum_left(grads[0], grads[1], sc, oc, rs, ac, ns);
                    } else {
                        alignas(16) accum_t ls[kAccs], gs[kAccs];
                        load_accum(ls, l_head + coff + si * p.stride_l_n, lane, cnvec);
                        load_accum(gs, g_head + coff + si * p.stride_g_n, lane, cnvec);
                        accum_right(grads[0], sc.x, gs, sc.y, ls, oc, ac, ns);
                    }
                });
            }
            team_sum<W, kWarpSize, kGrad * kAccs, /*TRAILING_SYNC=*/true>(&grads[0][0], reinterpret_cast<accum_t *>(smem), warp, lane, ct.T);
            if (has_chunk && ct.member == 0) {
                flush(coff, cnvec, grads);
            }
        }
    }
};

// ------------------------------------------------------------------------------------------------
// Kernel. SLICED selects the split-K heavy path: one block per slice of a heavy node's edge list,
// writing an fp32 partial row gradient that rows::sum_merge_slices_kernel sums. The partials are
// plain sums -- alpha comes from the saved logsumexp, not from a running max -- so slicing changes
// only the order of additions. The gradient of `a` goes to its slot row either way.
// ------------------------------------------------------------------------------------------------
template <
    BackwardMode MODE, sched::ScheduleKind SK, size_t W, int SHAPE, int STAGES, bool SLICED, FloatingNum cuda_t, IntegralNum index_t,
    FloatingNum accum_t = float
>
__global__ void __launch_bounds__(W *kWarpSize) gatv2_backward_kernel(
    BackwardParams<cuda_t, index_t, accum_t> const p, sched::SchedulerParams<index_t> sched_params
) {
    using Sh                      = Shape<SHAPE>;
    static constexpr bool kAL     = MODE == BackwardMode::AL;
    static constexpr size_t VEC   = Sh::template vec<cuda_t>;
    static constexpr size_t kGrad = kBackwardGrads<MODE>;

    extern __shared__ __align__(16) uint8_t sh_raw[];

    const int64_t head = blockIdx.y;
    if (head >= p.H) [[unlikely]] {
        return;
    }
    const uint32_t lane = threadIdx.x % kWarpSize;
    const uint32_t warp = threadIdx.x / kWarpSize;

    auto process_row = [&](int64_t node, index_t edge_begin, uint32_t n, int64_t out) {
        // The row's gradient goes to dl (AL) or dr (R) in place, or to the slice's fp32 partial.
        const int64_t row       = dense_row(out, head, p.H, p.D);
        cuda_t *const grad_row  = SLICED ? nullptr : (kAL ? p.dL : p.dR) + row;
        accum_t *const part_row = SLICED ? p.part + row : nullptr;
        accum_t *const da_row   = p.dA_slots + dense_row(blockIdx.x & (kGradASlots - 1), head, p.H, p.D);

        if (n == 0) [[unlikely]] {
            // No edges: a zero row gradient, nothing for `a`.
            if constexpr (SLICED) {
                zero_row<1>(part_row, p.D, threadIdx.x, blockDim.x);
            } else {
                zero_row<VEC>(grad_row, p.nvec, threadIdx.x, blockDim.x);
            }
            return;
        }

        // Writes the gradients of one stretch of the row: `off` its first element, lane r owning
        // vectors r, r + G, ... below nvec. (The array bounds of these lambdas' parameters spell out
        // kBackwardGrads<MODE>: the local kGrad there crashes cudafe++ of CUDA 13.4.)
        auto write = [&]<typename RG>(int64_t off, accum_t(&grads)[kBackwardGrads<MODE>][RG::kAccs], uint32_t r, uint32_t nvec) {
            if constexpr (SLICED) {
                RG::store_accum(part_row + off, grads[0], r, nvec);
            } else {
                RG::store(grad_row + off, grads[0], accum_t{1}, r, nvec);
            }
            if constexpr (kAL) {
                RG::atomic_add_accum(da_row + off, grads[1], r, nvec);
            }
        };

        index_t const *const edges = p.col_idx + edge_begin;
        if constexpr (Sh::kChunked) {
            using C                = BackwardChunked<MODE, W, Sh::kWide, cuda_t, index_t, accum_t>;
            const int64_t stride   = p.H * p.num_edges;
            accum_t *const scalars = p.edge_scalars + head * p.num_edges + static_cast<int64_t>(edge_begin);
            C::run(
                p, node, head, edges, n, scalars, stride, sh_raw,
                [&](int64_t coff, uint32_t cnvec, accum_t(&grads)[kBackwardGrads<MODE>][C::kAccs]) {
                    write.template operator()<typename C::RG>(coff, grads, lane, cnvec);
                }
            );
        } else {
            using F = BackwardFused<MODE, W, SHAPE, STAGES, cuda_t, index_t, accum_t>;
            alignas(16) accum_t grads[kGrad][F::kAccs];
            F::run(p, node, head, edges, n, sh_raw, grads);
            if (warp == 0 && lane < F::G) {
                write.template operator()<typename F::RG>(0, grads, lane, p.nvec);
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
