#pragma once

#include <algorithm>
#include <bit>
#include <cfloat>
#include <cstdint>
#include <type_traits>

#include "common/misc.cuh"
#include "common/tile.cuh"
#include "common/traits.cuh"

// =============================================================================
// Row groups: rows of a runtime length laid out across a group of lanes.
//
// The neighbour-gather kernels read one feature row per edge. With the row length a template
// parameter every head dim is its own instantiation, and only the few that were instantiated
// exist at all. Here the length D is a runtime value; what is compile-time is the *layout* of a
// row across lanes:
//
//   G    lanes that share one row, a power of two dividing the warp. A warp works on
//        kWarpSize / G rows at once, so a short row does not leave most of the warp idle.
//   VEC  elements per access: one 16-byte access when every row is 16-byte aligned, one element
//        otherwise.
//   S    vectors each lane holds in registers; one pass of a group covers G * S * VEC elements.
//
// Lane r of a group owns vectors r, r + G, ..., r + (S - 1) * G, so neighbouring lanes touch
// neighbouring vectors and every access is coalesced. Vectors at or past nvec = D / VEC are never
// loaded and read as zero, so a row shorter than G * S * VEC needs no special case anywhere: the
// zeros drop out of every dot product and every accumulation.
// =============================================================================

template <size_t G, size_t S, size_t VEC, FloatingNum cuda_t, FloatingNum accum_t = float>
struct RowGroup {
    static_assert(std::popcount(G) == 1 && G <= kWarpSize, "A row group is a power-of-two slice of one warp");
    static_assert(S >= 1, "A lane holds at least one vector of the row");

    static constexpr size_t kGroupsPerWarp = kWarpSize / G;
    static constexpr size_t kSpan          = G * S * VEC;  // row elements one pass of a group covers
    static constexpr size_t kAccs          = S * VEC;      // accumulator elements per lane

    using Tile  = TileOps<VEC, cuda_t, accum_t>;
    using vec_t = typename Tile::vec_t;
    using Hint  = typename Tile::MemoryHint;

    // A lane's share of one row.
    struct Frag {
        vec_t v[S];
    };

    static __device__ __forceinline__ uint32_t vec_index(uint32_t r, size_t s) { return r + static_cast<uint32_t>(s * G); }

    // Loads the lane's share of `row`; vectors past nvec, and all of them when !active, read as zero.
    //
    // The choice between data and zero is made on the raw bits of the whole vector, which compiles
    // to a predicated load into zeroed registers. Made on the vector itself it is one select per
    // element: sixteen byte permutes per 16-byte fp16 row, and through Vec::store_zero_() (an
    // `unsigned __int128 *` store) it also pins the fragment to a stack slot -- one dead STL.128
    // per row per lane, as much local-memory traffic as the gather itself.
    template <Hint hint = Hint::NoHint>
    static __device__ __forceinline__ void load(Frag& f, cuda_t const *const __restrict__ row, uint32_t r, uint32_t nvec, bool active = true) {
#pragma unroll
        for (size_t s = 0; s < S; ++s) {
            const uint32_t v = vec_index(r, s);
            typename Tile::access_t bits{};
            if (active && v < nvec) {
                bits = Tile::template read_raw<hint>(row, v);
            }
            f.v[s] = Tile::from_raw(bits);
        }
    }

    // The lane's part of <a, b>; reduce() completes it.
    static __device__ __forceinline__ accum_t dot(Frag const& a, Frag const& b) {
        accum_t acc{};
#pragma unroll
        for (size_t s = 0; s < S; ++s) {
            a.v[s].dot_product_(&acc, b.v[s]);
        }
        return acc;
    }

    // Sum over the G lanes of every group at once; each lane receives its group's total. xor
    // offsets below G never leave the group, so the groups of a warp reduce independently.
    static __device__ __forceinline__ accum_t reduce(accum_t x) {
#pragma unroll
        for (uint32_t off = G / 2; off > 0; off >>= 1) {
            x += __shfl_xor_sync(FULL_WARP_MASK, x, off);
        }
        return x;
    }

    static __device__ __forceinline__ void zero(accum_t (&acc)[kAccs]) {
#pragma unroll
        for (size_t e = 0; e < kAccs; ++e) {
            acc[e] = accum_t{};
        }
    }

    // The lane's share of a row, widened to accum_t: for rows a kernel uses on every edge, so the
    // conversion is paid once rather than per use.
    static __device__ __forceinline__ void to_accum(Frag const& f, accum_t (&out)[kAccs]) {
        constexpr size_t kChunk = std::min(VEC, VecFloat<1, float>::max_vec_size_bytes / std::max(sizeof(accum_t), sizeof(cuda_t)));
        using src_t             = VecFloat<kChunk, cuda_t>;
        using dst_t             = VecFloat<kChunk, accum_t>;
#pragma unroll
        for (size_t s = 0; s < S; ++s) {
#pragma unroll
            for (size_t c = 0; c < VEC / kChunk; ++c) {
                reinterpret_cast<dst_t *>(&out[s * VEC])[c] = reinterpret_cast<src_t const *>(&f.v[s])[c].template convert_vec<accum_t>();
            }
        }
    }

    static __device__ __forceinline__ void scale(accum_t (&acc)[kAccs], accum_t c) {
#pragma unroll
        for (size_t e = 0; e < kAccs; ++e) {
            acc[e] *= c;
        }
    }

    // acc += w * f
    static __device__ __forceinline__ void fma(accum_t (&acc)[kAccs], Frag const& f, accum_t w) {
#pragma unroll
        for (size_t s = 0; s < S; ++s) {
            f.v[s].template weighted_accum_<accum_t>(&acc[s * VEC], w);
        }
    }

    // acc = acc * c + w * f: the online-softmax update.
    static __device__ __forceinline__ void rescale_fma(accum_t (&acc)[kAccs], accum_t c, Frag const& f, accum_t w) {
        scale(acc, c);
        fma(acc, f, w);
    }

    // Sums acc over the kGroupsPerWarp groups of the warp; every group ends up with the total.
    static __device__ __forceinline__ void sum_groups(accum_t (&acc)[kAccs]) {
#pragma unroll
        for (uint32_t off = G; off < kWarpSize; off <<= 1) {
#pragma unroll
            for (size_t e = 0; e < kAccs; ++e) {
                acc[e] += __shfl_xor_sync(FULL_WARP_MASK, acc[e], off);
            }
        }
    }

    // row[v] = cuda_t(mul * acc) for the lane's vectors below nvec.
    static __device__ __forceinline__ void store(
        cuda_t *const __restrict__ row, accum_t (&acc)[kAccs], accum_t mul, uint32_t r, uint32_t nvec
    ) {
        scale(acc, mul);
#pragma unroll
        for (size_t s = 0; s < S; ++s) {
            const uint32_t v = vec_index(r, s);
            if (v < nvec) {
                Tile::write(row, v, Tile::convert_from_accum(&acc[s * VEC]));
            }
        }
    }

    // row[v] += acc in accum_t for the lane's vectors below nvec: one vector atomic per 16 bytes on
    // sm_90+, scalar atomics before. row + v * VEC must be aligned to min(16, VEC * 4) bytes, which
    // holds for any row of a contiguous fp32 [rows, D] tensor whose D is a multiple of VEC.
    static __device__ __forceinline__ void atomic_add_accum(
        accum_t *const __restrict__ row, accum_t const (&acc)[kAccs], uint32_t r, uint32_t nvec
    ) {
        static_assert(std::is_same_v<accum_t, float>, "fp32 atomics only");
        // Dependent on accum_t, so the sm_90-only vector overloads are looked up at instantiation,
        // i.e. only in the branch that is kept.
        using f4_t = std::conditional_t<sizeof(accum_t) == 4, float4, accum_t>;
        using f2_t = std::conditional_t<sizeof(accum_t) == 4, float2, accum_t>;
#pragma unroll
        for (size_t s = 0; s < S; ++s) {
            const uint32_t v = vec_index(r, s);
            if (v < nvec) {
                accum_t *const dst      = row + static_cast<size_t>(v) * VEC;
                accum_t const *const sv = &acc[s * VEC];
                if constexpr (kCudaArch >= 900 && VEC % 4 == 0) {
#pragma unroll
                    for (size_t i = 0; i < VEC; i += 4) {
                        atomicAdd(reinterpret_cast<f4_t *>(dst + i), *reinterpret_cast<f4_t const *>(sv + i));
                    }
                } else if constexpr (kCudaArch >= 900 && VEC == 2) {
                    atomicAdd(reinterpret_cast<f2_t *>(dst), *reinterpret_cast<f2_t const *>(sv));
                } else {
#pragma unroll
                    for (size_t i = 0; i < VEC; ++i) {
                        atomicAdd(dst + i, sv[i]);
                    }
                }
            }
        }
    }

    // row[v] = acc, unconverted: for the fp32 partials of the split-K kernels.
    static __device__ __forceinline__ void store_accum(
        accum_t *const __restrict__ row, accum_t const (&acc)[kAccs], uint32_t r, uint32_t nvec
    ) {
#pragma unroll
        for (size_t s = 0; s < S; ++s) {
            const uint32_t v = vec_index(r, s);
            if (v < nvec) {
#pragma unroll
                for (size_t e = 0; e < VEC; ++e) {
                    row[static_cast<size_t>(v) * VEC + e] = acc[s * VEC + e];
                }
            }
        }
    }

    // row[v] += w * f in fp32 for the lane's vectors below nvec; nothing at all when !active. Must
    // be called warp-uniformly: the paired form shuffles.
    //
    // sm_90 vector atomics make one reduction per 16 bytes. A 16-bit row in 16-byte vectors gives
    // every lane 32 bytes of fp32 -- one whole L2 sector -- which takes two such instructions, and
    // each of them covers only half of every sector it touches: two atomic transactions per sector,
    // where scalar atomics over 4 bytes per lane need one. Rows that take many updates (hubs) pay
    // for it in L2 serialisation: measured 15% slower than the scalar layout on a power-law graph.
    // Neighbouring lanes therefore swap halves first, so that each instruction covers whole sectors:
    // the even lane's sector in the first, the odd lane's in the second.
    static constexpr bool kPairedAtomics = kCudaArch >= 900 && VEC * sizeof(accum_t) == 32 && sizeof(accum_t) == 4 && G >= 2;

    static __device__ __forceinline__ void atomic_add(
        accum_t *const __restrict__ row, Frag const& f, accum_t w, uint32_t r, uint32_t nvec, bool active = true
    ) {
#pragma unroll
        for (size_t s = 0; s < S; ++s) {
            const uint32_t v = vec_index(r, s);
            const bool valid = active && v < nvec;
            if constexpr (kPairedAtomics) {
                using half_t = VecFloat<VEC / 2, cuda_t>;
                using quad_t = VecFloat<4, accum_t>;
                // Dependent on accum_t, so the sm_90-only float4 overload is looked up at
                // instantiation, i.e. only where this branch is kept. Spelled float4, the call is
                // resolved when the template is parsed and fails to compile for sm_80-sm_89.
                using f4_t = std::conditional_t<sizeof(accum_t) == 4, float4, accum_t>;
                quad_t lo  = reinterpret_cast<half_t const *>(&f.v[s])[0].template convert_vec<accum_t>();
                quad_t hi  = reinterpret_cast<half_t const *>(&f.v[s])[1].template convert_vec<accum_t>();
                lo *= w;
                hi *= w;
                const bool odd    = (r & 1u) != 0;
                const quad_t give = odd ? lo : hi;
                quad_t got;
#pragma unroll
                for (size_t e = 0; e < 4; ++e) {
                    got[e] = __shfl_xor_sync(FULL_WARP_MASK, give[e], 1);
                }
                const bool partner_valid = __shfl_xor_sync(FULL_WARP_MASK, static_cast<int>(valid), 1) != 0;
                // Vector v ^ 1 belongs to the partner lane (G is even, so v and r share parity).
                accum_t *const even_sector = row + static_cast<size_t>(v & ~1u) * VEC + (odd ? 4 : 0);
                accum_t *const odd_sector  = row + static_cast<size_t>(v | 1u) * VEC + (odd ? 4 : 0);
                if (odd ? partner_valid : valid) {
                    atomicAdd(reinterpret_cast<f4_t *>(even_sector), *reinterpret_cast<f4_t const *>(odd ? &got : &lo));
                }
                if (odd ? valid : partner_valid) {
                    atomicAdd(reinterpret_cast<f4_t *>(odd_sector), *reinterpret_cast<f4_t const *>(odd ? &hi : &got));
                }
            } else if (valid) {
                Tile::atomic_add_scaled_vec(row, v, w, f.v[s]);
            }
        }
    }
};

// Zeroes a whole row of nvec vectors with `threads` cooperating threads; for rows with no edges.
template <size_t VEC, typename T>
__device__ __forceinline__ void zero_row(T *const __restrict__ row, uint32_t nvec, uint32_t tid, uint32_t threads) {
    if constexpr (VEC * sizeof(T) <= 16 && std::popcount(VEC * sizeof(T)) == 1) {
        using vec_t = Vec<VEC, T>;
        for (uint32_t v = tid; v < nvec; v += threads) {
            reinterpret_cast<vec_t *>(row)[v].store_zero_();
        }
    } else {
        for (uint32_t e = tid; e < nvec * VEC; e += threads) {
            row[e] = T{};
        }
    }
}

// =============================================================================
// Walking a CSR row
// =============================================================================

// Per-slot data broadcast with the neighbour id (see for_each_neighbor). Shuffles are 32-bit, so
// the payload is one, two or three floats or nothing.
struct NoPayload {};

__device__ __forceinline__ NoPayload shfl_payload(NoPayload p, uint32_t) { return p; }
__device__ __forceinline__ float shfl_payload(float p, uint32_t src) { return __shfl_sync(FULL_WARP_MASK, p, src); }
__device__ __forceinline__ float2 shfl_payload(float2 p, uint32_t src) {
    return make_float2(__shfl_sync(FULL_WARP_MASK, p.x, src), __shfl_sync(FULL_WARP_MASK, p.y, src));
}
__device__ __forceinline__ float3 shfl_payload(float3 p, uint32_t src) {
    return make_float3(__shfl_sync(FULL_WARP_MASK, p.x, src), __shfl_sync(FULL_WARP_MASK, p.y, src), __shfl_sync(FULL_WARP_MASK, p.z, src));
}

// A load functor for walks that need no payload.
struct NoPayloadLoad {
    template <typename index_t>
    __device__ __forceinline__ NoPayload operator()(index_t, uint32_t) const {
        return {};
    }
};

// One warp's walk over slots [begin, end) of a CSR row whose neighbour ids start at `ids`, one slot
// per lane group: kWarpSize / G slots per step.
//
// fn(slot, active, nb, payload) is called warp-uniformly -- by every lane, the same number of
// times -- so it may use full-warp shuffles. A group whose slot is past `end` gets active == false
// and a meaningless nb/payload; it must still take part in the shuffles but load and write nothing.
//
// Neighbour ids are fetched kWarpSize at a time with one coalesced load and broadcast by shuffle,
// so the dependent chain id -> row address -> row data starts from a register instead of a global
// load per edge. load(nb, slot) fetches per-slot payload -- the neighbour's logsumexp, a score
// stored by an earlier pass -- the same way: one lane per slot, once per batch.
//
// UNROLL unrolls the walk over a batch, for bodies small enough to profit from having several
// rows in flight at once.
template <size_t G, size_t UNROLL = 1, IntegralNum index_t, typename LoadFn, typename Fn>
__device__ __forceinline__ void for_each_neighbor(
    index_t const *const __restrict__ ids, uint32_t begin, uint32_t end, uint32_t lane, LoadFn&& load, Fn&& fn
) {
    using payload_t          = decltype(load(index_t{}, uint32_t{}));
    constexpr uint32_t kStep = kWarpSize / G;
    const uint32_t g         = lane / G;

    for (uint32_t base = begin; base < end; base += kWarpSize) {
        const uint32_t pos           = base + lane;
        const bool ok                = pos < end;
        const index_t held           = ok ? ids[pos] : index_t{};
        const payload_t held_payload = ok ? load(held, pos) : payload_t{};
        const uint32_t cnt           = min(static_cast<uint32_t>(kWarpSize), end - base);

#pragma unroll UNROLL
        for (uint32_t t = 0; t < cnt; t += kStep) {
            const uint32_t src = t + g;
            const index_t nb   = __shfl_sync(FULL_WARP_MASK, held, src);
            fn(base + src, src < cnt, nb, shfl_payload(held_payload, src));
        }
    }
}

// Random access to the neighbour ids (and payload) of a CSR row, for a warp that walks it in order
// from more than one place at once -- the producer and the consumer of a copy pipeline. Ids are
// fetched kWarpSize at a time and broadcast by shuffle, as in for_each_neighbor. get() must be
// called warp-uniformly with slots that never decrease.
template <IntegralNum index_t, typename LoadFn>
struct NeighborCache {
    using payload_t = decltype(std::declval<LoadFn&>()(index_t{}, uint32_t{}));

    index_t const *ids;
    uint32_t end;
    LoadFn load;
    uint32_t base = ~0u;
    index_t held{};
    payload_t held_payload{};

    __device__ __forceinline__ NeighborCache(index_t const *ids_, uint32_t end_, LoadFn load_) : ids(ids_), end(end_), load(load_) {}

    __device__ __forceinline__ index_t get(uint32_t slot, uint32_t lane, payload_t& payload) {
        const uint32_t b = slot & ~static_cast<uint32_t>(kWarpSize - 1);
        if (b != base) {
            base               = b;
            const uint32_t pos = b + lane;
            const bool ok      = pos < end;
            held               = ok ? ids[pos] : index_t{};
            held_payload       = ok ? load(held, pos) : payload_t{};
        }
        const uint32_t src = slot & (kWarpSize - 1);
        payload            = shfl_payload(held_payload, src);
        return __shfl_sync(FULL_WARP_MASK, held, src);
    }
};

// The contiguous share of slots [0, n) of warp `warp` out of `num_warps`, in whole multiples of
// `step` slots so that only the last non-empty share ends in a partial step. Contiguous rather
// than strided: a warp's ids are then adjacent and arrive in one coalesced load per 32 slots.
__device__ __forceinline__ uint2 warp_share(uint32_t n, uint32_t warp, uint32_t num_warps, uint32_t step) {
    const uint32_t per   = ceil_div(ceil_div(n, num_warps), step) * step;
    const uint64_t begin = min(static_cast<uint64_t>(warp) * per, static_cast<uint64_t>(n));
    const uint64_t end   = min(begin + per, static_cast<uint64_t>(n));
    return make_uint2(static_cast<uint32_t>(begin), static_cast<uint32_t>(end));
}

// =============================================================================
// Combining per-warp partials
// =============================================================================

// Online-softmax states and their accumulators, groups -> warp: a merge with each xor partner.
// Afterwards every group of the warp holds the warp's state and accumulator. Warp-uniform.
template <size_t G, size_t E, FloatingNum accum_t>
__device__ __forceinline__ void merge_groups_softmax(OnlineSoftmaxState& st, accum_t (&acc)[E]) {
    if constexpr (G < kWarpSize) {
#pragma unroll
        for (uint32_t off = G; off < kWarpSize; off <<= 1) {
            const float2 c = st.merge(__shfl_xor_sync(FULL_WARP_MASK, st.max_val, off), __shfl_xor_sync(FULL_WARP_MASK, st.sum_exp, off));
#pragma unroll
            for (size_t e = 0; e < E; ++e) {
                const accum_t other = __shfl_xor_sync(FULL_WARP_MASK, acc[e], off);
                acc[e]              = AdOps<accum_t>::fma(acc[e], c.x, other * c.y);
            }
        }
    }
}

// Merges the softmax states of a block's warps -- lane 0 of each holding its warp's -- into `st`
// of every thread, and returns exp(m_warp - M): the factor that rescales this warp's accumulators
// to the block-wide max M. stats: 2 * W floats of shared memory. Must be called by every thread.
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

// The W warps of a block as teams for a chunked sweep over nchunks feature chunks: one chunk per
// team at a time, the T warps of a team splitting the edges. The team count is a power of two no
// larger than the chunk count (rounded up) or W, so with at least as many chunks as warps T == 1
// and the warps never need to combine anything; with fewer, the edges are still spread over all
// warps. Identical in every thread of the block.
struct ChunkTeams {
    uint32_t teams;   // chunks in flight at once
    uint32_t T;       // warps per team
    uint32_t team;    // this warp's team
    uint32_t member;  // this warp's place in it
};

template <size_t W>
__device__ __forceinline__ ChunkTeams chunk_teams(uint32_t nchunks, uint32_t warp) {
    uint32_t teams = 1;
    while (teams < W && teams < nchunks) {
        teams <<= 1;
    }
    const uint32_t T = static_cast<uint32_t>(W) / teams;
    return {teams, T, warp / T, warp % T};
}

// Shared scratch, in accum_t elements, per round of team_sum: the per-round width is chosen so one
// round never needs more than kTeamSumBudgetBytes, whatever W and E are. A block-wide merge of a
// full row (the old layout) needs W * D accumulators and would not fit for a 32-warp block at
// large D; rounds keep it bounded at the cost of two barriers per round.
inline constexpr size_t kTeamSumBudgetBytes = 16 * 1024;

template <size_t W, size_t LANES, size_t E, FloatingNum accum_t = float>
inline constexpr size_t kTeamSumRound = std::clamp<size_t>(kTeamSumBudgetBytes / (W * LANES * sizeof(accum_t)), 1, E);

template <size_t W, size_t LANES, size_t E, FloatingNum accum_t = float>
inline constexpr size_t kTeamSumBufferElems = W == 1 ? 0 : W * LANES * kTeamSumRound<W, LANES, E, accum_t>;

// Sums vals[0, E) over the warps of each team of T consecutive warps, leaving every team's total in
// its first warp. Only lanes [0, LANES) carry values (the lanes of one row group after
// sum_groups()). T is a power of two dividing W and must be the same in every thread; T == 1 is a
// no-op. Deterministic: partials are always added in warp order, so repeated runs agree bitwise.
//
// Must be called by every thread of the block. buf holds kTeamSumBufferElems<W, LANES, E> elements;
// it is free again once this returns if TRAILING_SYNC, otherwise only after the next barrier.
template <size_t W, size_t LANES, size_t E, bool TRAILING_SYNC = false, FloatingNum accum_t>
__device__ __forceinline__ void team_sum(accum_t *const vals, accum_t *const __restrict__ buf, uint32_t warp, uint32_t lane, uint32_t T) {
    if constexpr (W > 1) {
        if (T <= 1) {
            return;
        }
        constexpr size_t R    = kTeamSumRound<W, LANES, E, accum_t>;
        const uint32_t member = warp & (T - 1);
        const bool carries    = lane < LANES;
#pragma unroll
        for (size_t e0 = 0; e0 < E; e0 += R) {
            // [warp][element][lane]: consecutive lanes hit consecutive banks.
            if (carries && member != 0) {
#pragma unroll
                for (size_t e = 0; e < R && e0 + e < E; ++e) {
                    buf[(warp * R + e) * LANES + lane] = vals[e0 + e];
                }
            }
            __syncthreads();
            if (carries && member == 0) {
                for (uint32_t m = 1; m < T; ++m) {
#pragma unroll
                    for (size_t e = 0; e < R && e0 + e < E; ++e) {
                        vals[e0 + e] += buf[((warp + m) * R + e) * LANES + lane];
                    }
                }
            }
            if (TRAILING_SYNC || e0 + R < E) {
                __syncthreads();
            }
        }
    }
}
