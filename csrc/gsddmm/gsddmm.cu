#include <algorithm>
#include <cstdint>
#include <type_traits>

#include "common/misc.cuh"
#include "common/row_group.cuh"
#include "common/row_shape.cuh"
#include "common/tile.cuh"
#include "common/traits.cuh"
#include "gsddmm/gsddmm.cuh"

// g-SDDMM kernels; see gsddmm.cuh for the semantics, the row layout and the kernel overview.
// Included by gsddmm_dispatch.cuh only.

namespace gsddmm {

// Launch bounds. The kernels are memory bound with little work per edge, so they live on
// occupancy, and ptxas left to itself spends registers on them that cut it: 35-54 registers and
// 49-62% achieved occupancy measured on the node backward, against 89-92% for the former fixed-D
// kernels. Each kernel is given a register target:
//
//   sm_90 (timed): 32 for the forward, 40 for the backward, 64 for two vectors a lane.
//
// These were picked by timing, not by spill counts: on 25 cells of 7 graphs, against the former
// kernels, they measured 1.19x geomean, a table raised wherever ptxas spilled inside an edge loop
// (40-48 for wide or whole-warp backward rows, 64 for 16-bit Div) 1.16x, and 32 throughout 1.15x.
// A few spills are L1 hits; the occupancy they buy is worth more -- 16-bit Div's node forward, for
// one, ran 0.165 ms at 32 registers with spills and 0.211 ms at 64 without.
//
//   other architectures (not timed: no other GPU to measure on): 48 for the forward over groups of
//   up to 16 lanes, 56 over whole-warp rows; 56 for the backward, 64 for its 16-byte rows of 16-bit
//   elements or of a whole warp; and left to ptxas for two vectors a lane, for 16-bit Div and for
//   the edge backward over whole-warp rows. That is the sm_90 spill-free table plus 16, checked to
//   remove every spill inside an edge loop on sm_80, sm_86, sm_89, sm_100 and sm_103 but those of
//   1024-thread blocks, which are held at 64 registers by their size.
//
// For a block of THREADS threads the target becomes a resident-block count, as __launch_bounds__
// takes it, capped at what the SM can hold; 0 leaves the kernel to ptxas.
enum class Family : uint8_t { Forward, BackwardNode, BackwardEdge };

template <Family F, int SHAPE, GSDDMM_OP op, FloatingNum cuda_t>
inline constexpr int kRegTarget = [] {
    using Sh = Shape<SHAPE>;
    if constexpr (kCudaArch == 900) {
        return Sh::S == 2 ? 64 : (F == Family::Forward ? 32 : 40);
    } else {
        constexpr bool half  = sizeof(cuda_t) == 2;
        constexpr bool div16 = op == GSDDMM_OP::Div && half;
        if (Sh::S == 2 || div16 || (F == Family::BackwardEdge && Sh::G == kWarpSize)) {
            return 0;
        }
        if constexpr (F == Family::Forward) {
            return Sh::G == kWarpSize ? 56 : 48;
        } else if constexpr (F == Family::BackwardNode) {
            return Sh::kWide && (half || Sh::G == kWarpSize) ? 64 : 56;
        } else {
            return Sh::kWide ? 64 : 56;
        }
    }
}();

template <Family F, size_t THREADS, int SHAPE, GSDDMM_OP op, FloatingNum cuda_t>
inline constexpr int kMinBlocks = [] {
    // The 32-warp node backward at 32 registers, two blocks an SM: a 1024-thread block holding more
    // runs one per SM, and the heavy bucket of a low-degree graph is short rows whose blocks are all
    // latency -- city-roads-M's node backward went 0.58x -> 0.91x of the former kernels (D = 256,
    // fp16) and 0.63x -> 0.86x (D = 64, fp32), for 0.95x -> 0.87x on ogbn-arxiv, where heavy rows
    // are long enough to feel the spills. 3.7% geomean over 16 node-backward cells of 9 graphs.
    if constexpr (kCudaArch == 900 && F == Family::BackwardNode && THREADS == 1024 && Shape<SHAPE>::S == 1) {
        return 2;
    }
    constexpr int regs = kRegTarget<F, SHAPE, op, cuda_t>;
    constexpr int fit  = 65536 / ((regs == 0 ? 64 : regs) * static_cast<int>(THREADS));
    return regs == 0 ? 1 : std::max(1, std::min({fit, static_cast<int>(kMaxThreadsPerSM / THREADS), kMaxResidentBlocksPerSM}));
}();

// =============================================================================
// Shared pieces
// =============================================================================

// Elementwise op on vectors. Dot is reduced separately (it is not elementwise); Copy never touches
// the right operand.
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
        } else if constexpr (op == GSDDMM_OP::Copy || op == GSDDMM_OP::Dot) {
            // Copy passes the left operand through; Dot is reduced by the caller.
        } else {
            static_assert(!sizeof(cuda_t), "GsddmmVecOp::apply reached an unhandled op");
            __builtin_unreachable();
        }
        return l;
    }
};

// Row k of `base`, `stride` bytes apart. Ids are non-negative, so they are widened through their
// unsigned form: a zero-extend is free where a sign-extend costs an instruction and a second live
// value per use, and the 32-bit stride makes the product a single wide multiply-add.
template <typename id_t, FloatingNum cuda_t>
__device__ __forceinline__ cuda_t const *row_at(cuda_t const *const base, id_t k, uint32_t stride) {
    return reinterpret_cast<cuda_t const *>(
        reinterpret_cast<char const *>(base) + static_cast<uint64_t>(static_cast<std::make_unsigned_t<id_t>>(k)) * stride
    );
}

// The layout of one shape over cuda_t rows, and the per-lane row pointers a pass works with.
//
// A lane's vectors sit at the same offset in every row, so the walks hoist that offset out of the
// edge loop: per pass, each operand gets one pointer per vector a lane holds -- the operand's base
// plus the lane's offset -- and an edge costs one wide multiply-add of its row id per operand. Load
// pointers are clamped to the pass's last vector, so a lane past the end of a short row re-reads a
// vector its group reads anyway and needs no predicate; inactive slots are given a valid row id by
// the walks for the same reason, and their values are dropped. Only stores are predicated. Pointers
// are advanced in bytes (see Operands). Measured against the predicated loads of RowGroup::load and
// per-edge address math, in SASS instructions per edge: the node backward of dot over fp32 rows of
// 128 went from 28 to 14 (the former fixed-D kernel: 18.5), the edge forward of the same from 63 to
// 33.5 (former: 29).
template <int SHAPE, FloatingNum cuda_t, FloatingNum accum_t>
struct Layout {
    using Sh                          = Shape<SHAPE>;
    static constexpr size_t G         = Sh::G;
    static constexpr size_t S         = Sh::S;
    static constexpr size_t VEC       = Sh::template vec<cuda_t>;
    using RG                          = RowGroup<G, S, VEC, cuda_t, accum_t>;
    using Frag                        = typename RG::Frag;
    using Tile                        = typename RG::Tile;
    using vec_t                       = typename RG::vec_t;
    static constexpr uint32_t kPass   = kPassVecs<SHAPE>;
    static constexpr uint32_t kGroups = static_cast<uint32_t>(RG::kGroupsPerWarp);
    static constexpr size_t kAccs     = RG::kAccs;
    static constexpr bool kMulti      = kMultiPass<SHAPE>;

    // The hint for rows read exactly once (dO, edge operands): Streaming (evict-first), so that they
    // do not push the node rows, which other edges re-read, out of L2 -- with plain loads the node
    // backward read 23% more DRAM (L2 hit rate 36% -> 25%) and ran 18% slower on reddit. But only
    // for rows whose pass spans at least a 128-byte line: two 64-byte rows share one, and dropping it
    // early refetched it for the neighbouring edge -- 5% slower on ogbn-products (D = 32, fp16).
    static constexpr typename Tile::MemoryHint kOnce = G * VEC * sizeof(cuda_t) >= 128 ? Tile::MemoryHint::Streaming : Tile::MemoryHint::NoHint;

    // One operand's lane pointers for a pass, clamped to its last vector.
    struct In {
        cuda_t const *p[S];
    };
    // An output's lane pointers for a pass, and which of the lane's vectors exist.
    struct Out {
        cuda_t *p[S];
        bool ok[S];
    };

    static __device__ __forceinline__ In lanes_in(cuda_t const *const base, uint32_t v0, uint32_t rem, uint32_t r) {
        In in;
#pragma unroll
        for (size_t s = 0; s < S; ++s) {
            const uint32_t v = min(RG::vec_index(r, s), rem - 1);
            in.p[s]          = base + static_cast<size_t>(v0 + v) * VEC;
        }
        return in;
    }
    static __device__ __forceinline__ Out lanes_out(cuda_t *const base, uint32_t v0, uint32_t rem, uint32_t r) {
        Out out;
#pragma unroll
        for (size_t s = 0; s < S; ++s) {
            const uint32_t v = RG::vec_index(r, s);
            out.p[s]         = base + static_cast<size_t>(v0 + v) * VEC;
            out.ok[s]        = v < rem;
        }
        return out;
    }

    // Row `id` (stride in bytes) of an operand, the lane's vectors of the pass.
    template <typename Tile::MemoryHint HINT = Tile::MemoryHint::NoHint, typename id_t>
    static __device__ __forceinline__ Frag gather(In const& in, id_t id, uint32_t stride) {
        Frag f;
        const uint64_t off = static_cast<uint64_t>(id) * stride;
#pragma unroll
        for (size_t s = 0; s < S; ++s) {
            f.v[s] = Tile::template read<HINT>(reinterpret_cast<cuda_t const *>(reinterpret_cast<char const *>(in.p[s]) + off), 0);
        }
        return f;
    }

    // Row `id` of an output. Streaming: every output row is written once and never read back here,
    // so it must not evict the gathered operand rows, which are.
    template <typename id_t>
    static __device__ __forceinline__ void scatter(Out const& out, id_t id, uint32_t stride, Frag const& f, bool active) {
        const uint64_t off = static_cast<uint64_t>(id) * stride;
#pragma unroll
        for (size_t s = 0; s < S; ++s) {
            if (active && out.ok[s]) {
                Tile::template write<Tile::MemoryHint::Streaming>(
                    reinterpret_cast<cuda_t *>(reinterpret_cast<char *>(out.p[s]) + off), 0, f.v[s]
                );
            }
        }
    }
};

// =============================================================================
// Forward: the per-edge work of one pass.
//
// make() sets up a pass's lane pointers; step() runs one edge of it, the cached operand (if any)
// coming from `cache`. Dot leaves its partial in `partial`, for the caller to reduce over the group
// and store once all passes are done. Must be called warp-uniformly.
// =============================================================================
template <GSDDMM_OP op, GSDDMM_MEMBER ll, GSDDMM_MEMBER rr, int SHAPE, FloatingNum cuda_t, FloatingNum accum_t>
struct ForwardEdge {
    using Plan = GsddmmPlan<op, ll, rr>;
    using Lay  = Layout<SHAPE, cuda_t, accum_t>;
    using RG   = typename Lay::RG;
    using Frag = typename Lay::Frag;

    struct Pass {
        typename Lay::In l, r;
        typename Lay::Out o;  // Dot: only o.ok, which masks its partial sum
    };

    // Edge-indexed operand rows are read once; node rows are re-read by other edges.
    static constexpr typename Lay::Tile::MemoryHint hint(GSDDMM_MEMBER m) {
        return m == GSDDMM_MEMBER::Edge ? Lay::kOnce : Lay::Tile::MemoryHint::NoHint;
    }

    static __device__ __forceinline__ Pass make(Operands<cuda_t> const& x, cuda_t *const O, uint32_t v0, uint32_t r) {
        const uint32_t rem = x.nvec - v0;
        Pass P;
        P.l = Lay::lanes_in(x.L, v0, rem, r);
        if constexpr (Plan::USE_R) {
            P.r = Lay::lanes_in(x.R, v0, rem, r);
        }
        // Dot needs the validity flags too: a lane past the end of a short row reads the clamped last
        // vector, which an elementwise op drops at the store but a dot product would sum.
        P.o = Lay::lanes_out(O, v0, rem, r);
        return P;
    }

    template <bool CACHE_L, bool CACHE_R, typename id_t>
    static __device__ __forceinline__ void step(
        Pass const& P, Frag const& cache, uint32_t stride_l, uint32_t stride_r, uint32_t row_bytes, id_t l_id, id_t r_id, id_t e, bool active,
        accum_t& partial
    ) {
        static_assert(!(CACHE_L && CACHE_R), "at most one operand is shared by a block's or a chunk's edges");
        Frag lf = CACHE_L ? cache : Lay::template gather<hint(ll)>(P.l, l_id, stride_l);
        if constexpr (Plan::IS_DOT) {
            const Frag rf = CACHE_R ? cache : Lay::template gather<hint(rr)>(P.r, r_id, stride_r);
#pragma unroll
            for (size_t s = 0; s < Lay::S; ++s) {
                const accum_t d = lf.v[s].template dot_product_<accum_t>(rf.v[s]);
                partial += P.o.ok[s] ? d : accum_t{};
            }
        } else {
            if constexpr (Plan::USE_R) {
                const Frag rf = CACHE_R ? cache : Lay::template gather<hint(rr)>(P.r, r_id, stride_r);
#pragma unroll
                for (size_t s = 0; s < Lay::S; ++s) {
                    lf.v[s] = GsddmmVecOp<op, Lay::VEC, cuda_t>::apply(lf.v[s], rf.v[s]);
                }
            }
            Lay::scatter(P.o, e, row_bytes, lf, active);
        }
    }

    // One edge over all passes: single-pass rows reuse `single`, longer ones set each pass up anew
    // (amortized over the pass's 64 vectors). Dot's result is stored here.
    template <bool CACHE_L, bool CACHE_R, typename id_t>
    static __device__ __forceinline__ void edge(
        Operands<cuda_t> const& x, cuda_t *const __restrict__ O, Pass const& single, Frag const& cache, uint32_t r, id_t l_id, id_t r_id,
        id_t e, bool active
    ) {
        accum_t partial{};
        if (!Lay::kMulti || x.passes == 1) {
            step<CACHE_L, CACHE_R>(single, cache, x.stride_l, x.stride_r, x.row_bytes, l_id, r_id, e, active, partial);
        } else {
            for (uint32_t pass = 0; pass < x.passes; ++pass) {
                step<false, false>(
                    make(x, O, pass * Lay::kPass, r), cache, x.stride_l, x.stride_r, x.row_bytes, l_id, r_id, e, active, partial
                );
            }
        }
        if constexpr (Plan::IS_DOT) {
            partial = RG::reduce(partial);
            if (active && r == 0) {
                O[e] = static_cast<cuda_t>(partial);
            }
        }
    }
};

// =============================================================================
// Forward, node-parallel. One block of W warps per bucketed CSR row (or chunk of one); each warp
// takes a contiguous share of the row's edges, its groups an edge each, with the ids fetched 32 at
// a time and broadcast by shuffle (for_each_neighbor). The Dst_V operand row is the same for every
// edge of the block, so for single-pass rows each lane holds its share of it in registers: one load
// per warp instead of one per edge, and no shared memory or barrier at all.
// =============================================================================
template <GSDDMM_OP op, GSDDMM_MEMBER ll, GSDDMM_MEMBER rr, size_t W, int SHAPE, FloatingNum cuda_t, IntegralNum index_t, FloatingNum accum_t>
__global__ void __launch_bounds__(W *kWarpSize, kMinBlocks<Family::Forward, W * kWarpSize, SHAPE, op, cuda_t>) gsddmm_forward_node_kernel(
    NodeForwardParams<cuda_t, index_t> p
) {
    using uidx_t = std::make_unsigned_t<index_t>;
    using Plan   = GsddmmPlan<op, ll, rr>;
    using Lay    = Layout<SHAPE, cuda_t, accum_t>;
    using RG     = typename Lay::RG;
    using Frag   = typename Lay::Frag;
    using Edge   = ForwardEdge<op, ll, rr, SHAPE, cuda_t, accum_t>;

    const uint32_t lane = threadIdx.x % kWarpSize;
    const uint32_t warp = threadIdx.x / kWarpSize;
    const uint32_t r    = lane % Lay::G;

    const index_t node = p.node_indices[blockIdx.x];
    if (static_cast<int64_t>(node) >= p.N) [[unlikely]] {
        return;
    }
    index_t begin = p.row_ptr[node];
    index_t end   = p.row_ptr[node + 1];
    // Heavy-node chunking: node_indices then repeats a node once per chunk and block_part says which
    // edges_per_block-wide slice of its edge list this block owns.
    if (p.block_part != nullptr) {
        begin += p.block_part[blockIdx.x] * static_cast<index_t>(p.edges_per_block);
        const index_t chunk_end = begin + static_cast<index_t>(p.edges_per_block);
        end                     = chunk_end < end ? chunk_end : end;
    }
    const uint32_t n = static_cast<uint32_t>(end - begin);
    // No barrier anywhere below, so a warp without edges simply leaves.
    const uint2 share = warp_share(n, warp, W, Lay::kGroups);
    if (share.x >= share.y) {
        return;
    }

    Operands<cuda_t> const& x     = p.x;
    const uidx_t first            = static_cast<uidx_t>(begin);
    const uidx_t self             = static_cast<uidx_t>(node);
    const typename Edge::Pass one = Edge::make(x, p.O, 0, r);

    auto walk = [&](auto cache_c) {
        constexpr bool CACHE = decltype(cache_c)::value;
        Frag cache;
        if constexpr (CACHE) {
            RG::load(cache, Plan::L_DST ? row_at(x.L, node, x.stride_l) : row_at(x.R, node, x.stride_r), r, x.nvec);
        }
        for_each_neighbor<Lay::G>(
            p.col_idx + begin, share.x, share.y, lane, NoPayloadLoad{}, [&](uint32_t slot, bool active, index_t j, NoPayload) {
                // An inactive slot's neighbour id is 0 (a real row); its edge id is pointed at the row's
                // first edge, so no load needs a predicate.
                const uidx_t e    = active ? first + slot : first;
                const uidx_t src  = static_cast<uidx_t>(j);
                const uidx_t l_id = ll == GSDDMM_MEMBER::Edge ? e : (ll == GSDDMM_MEMBER::Dst_V ? self : src);
                const uidx_t r_id = rr == GSDDMM_MEMBER::Edge ? e : (rr == GSDDMM_MEMBER::Dst_V ? self : src);
                Edge::template edge<CACHE && Plan::L_DST, CACHE && Plan::R_DST>(x, p.O, one, cache, r, l_id, r_id, e, active);
            }
        );
    };
    if (Plan::READS_DST && (!Lay::kMulti || x.passes == 1)) {
        walk(std::bool_constant<Plan::READS_DST>{});
    } else {
        walk(std::false_type{});
    }
}

// =============================================================================
// Forward, edge-parallel. Warp w (global over the grid) owns the contiguous chunk
// [w * chunk, (w + 1) * chunk) of the edge list, chunk <= 32: lane k caches edge k's (src, dst)
// pair -- and canonical id -- from one coalesced load and broadcasts it by shuffle, so no edge
// waits on a dependent index load. The warp's groups take the chunk's edges 32 / G at a time.
//
// The list is grouped by the member of Plan::GROUPING, so a chunk's edges usually share that
// operand's row: one warp vote establishes it, and the row is then loaded once and held in
// registers -- for a single-pass row -- with no per-edge check left in the loop. A per-edge id
// compare was measured to cost more issue slots than the load it saves.
// =============================================================================
template <
    GSDDMM_OP op, GSDDMM_MEMBER ll, GSDDMM_MEMBER rr, int SHAPE, FloatingNum cuda_t, IntegralNum index_t, FloatingNum accum_t, bool REMAP_IDS
>
__global__ void
    __launch_bounds__(kWarpSize *kGsddmmEdgeMaxWarpsPerBlock, kMinBlocks<Family::Forward, kWarpSize * kGsddmmEdgeMaxWarpsPerBlock, SHAPE, op, cuda_t>)
    gsddmm_forward_edge_kernel(EdgeForwardParams<cuda_t, index_t> p) {
    using eidx_t = std::make_unsigned_t<index_t>;
    using Plan   = GsddmmPlan<op, ll, rr>;
    using Lay    = Layout<SHAPE, cuda_t, accum_t>;
    using RG     = typename Lay::RG;
    using Frag   = typename Lay::Frag;
    using Edge   = ForwardEdge<op, ll, rr, SHAPE, cuda_t, accum_t>;

    const uint32_t lane = threadIdx.x % kWarpSize;
    const uint32_t warp = threadIdx.x / kWarpSize;
    const uint32_t g    = lane / Lay::G;
    const uint32_t r    = lane % Lay::G;

    // The grid covers ceil(E / chunk) warps, so a warp index fits eidx_t whenever E does.
    const eidx_t block_linear = static_cast<eidx_t>((static_cast<uint64_t>(blockIdx.z) * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x);
    const eidx_t edge_base    = (block_linear * static_cast<eidx_t>(blockDim.x / kWarpSize) + warp) * static_cast<eidx_t>(p.chunk);
    if (edge_base >= p.E) [[unlikely]] {
        return;
    }
    const uint32_t cnt = static_cast<uint32_t>(p.E - edge_base < p.chunk ? p.E - edge_base : p.chunk);

    // The chunk is this warp's alone, so its ids are streamed past L1. Lanes past the chunk hold
    // (0, 0) and canonical id 0 -- real rows, read by the inactive groups of the last step.
    index_pair_t<index_t> my_pair{};
    eidx_t my_canon{};
    if (lane < cnt) {
        my_pair = __ldcs(&p.pairs[edge_base + lane]);
        if constexpr (REMAP_IDS) {
            my_canon = static_cast<eidx_t>(p.canonical[edge_base + lane]);
        }
    }

    Operands<cuda_t> const& x     = p.x;
    const typename Edge::Pass one = Edge::make(x, p.O, 0, r);
    const eidx_t my_key           = Plan::GROUPING == GSDDMM_MEMBER::Dst_V ? static_cast<eidx_t>(my_pair.y) : static_cast<eidx_t>(my_pair.x);
    const eidx_t key0             = __shfl_sync(FULL_WARP_MASK, my_key, 0);
    const bool shared             = __all_sync(FULL_WARP_MASK, lane >= cnt || my_key == key0) != 0;

    auto walk = [&](auto cache_c) {
        constexpr bool CACHE = decltype(cache_c)::value;
        Frag cache;
        if constexpr (CACHE) {
            RG::load(cache, Plan::L_GROUPED ? row_at(x.L, key0, x.stride_l) : row_at(x.R, key0, x.stride_r), r, x.nvec);
        }
        for (uint32_t t = 0; t < cnt; t += Lay::kGroups) {
            const uint32_t k  = t + g;
            const bool active = k < cnt;
            const eidx_t src  = static_cast<eidx_t>(__shfl_sync(FULL_WARP_MASK, my_pair.x, static_cast<int>(k)));
            const eidx_t dst  = static_cast<eidx_t>(__shfl_sync(FULL_WARP_MASK, my_pair.y, static_cast<int>(k)));
            eidx_t e;
            if constexpr (REMAP_IDS) {
                e = __shfl_sync(FULL_WARP_MASK, my_canon, static_cast<int>(k));
            } else {
                e = edge_base + (active ? k : 0u);
            }
            const eidx_t l_id = ll == GSDDMM_MEMBER::Src_V ? src : (ll == GSDDMM_MEMBER::Dst_V ? dst : e);
            const eidx_t r_id = rr == GSDDMM_MEMBER::Src_V ? src : (rr == GSDDMM_MEMBER::Dst_V ? dst : e);
            Edge::template edge<CACHE && Plan::L_GROUPED, CACHE && Plan::R_GROUPED>(x, p.O, one, cache, r, l_id, r_id, e, active);
        }
    };
    if (shared && (!Lay::kMulti || x.passes == 1)) {
        walk(std::true_type{});
    } else {
        walk(std::false_type{});
    }
}

// =============================================================================
// Backward: per-edge gradient contribution, on vectors.
//
// Returns the part of the partial that does NOT depend on the reduced operand's own row, so a whole
// node's edges can be summed before the row-dependent factor is applied once (see
// GsddmmBackwardPlan::NEEDS_SELF_ROW):
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

// acc[0, TW) += v, widened to accum_t, in chunks of min(TW, sizeof(accum_t)) elements so that each
// chunk is one vector register rather than a wide fp32 vector spilled to local memory.
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

// The reduced vector, converted back to the storage type.
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

// 1/v^2 for one chunk, evaluated in accum_t. The reciprocal is taken once and squared rather than
// dividing twice: the approximate reciprocal is a quarter-rate unit, so halving the special-function
// work is the win. accum_t is what keeps the multiplier finite -- 1/v^2 leaves fp16's range already
// for |v| < 2^-8.
template <size_t N, FloatingNum cuda_t, FloatingNum accum_t>
__device__ __forceinline__ VecFloat<N, accum_t> gsddmm_inv_sq_chunk(VecFloat<N, cuda_t> v) {
    VecFloat<N, accum_t> r(accum_t(1));
    r.div_(v.template convert_vec<accum_t>());
    r.mul_(r);
    return r;
}

// x / v^2 for the edge gradient, whose destination is a cuda_t tensor: one narrowing is
// unavoidable, so it is done last, on the result, never on the 1/v^2 multiplier.
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

// acc = -acc / v^2, in place: the node-parallel backward owns the whole node, so the factor is
// applied once to the finished fp32 sum, before the narrowing to the storage type.
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

// acc += a * b with both operands widened first, so the product never rounds to the storage type.
// Only Div's reduced partial needs this: the -1/self^2 applied afterwards amplifies whatever the
// product lost, and once the sum cancels that loss becomes the answer.
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

// acc -= a * b / v^2: the whole per-edge Div term in accum_t, for the edge-parallel backward, which
// cannot defer the 1/self^2 past its atomic.
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
// Backward: the per-edge work of one pass of the reduced node's gradient.
//
// make() sets up a pass's lane pointers for the edge's dO row, the other operand's row and the edge
// operand's gradient; step() adds one edge to acc and stores its edge gradient. The reduced node's
// own row, where KEEP_SELF_ROW, comes in `self`. PER_EDGE_DIV_SQ: Div's -1/self^2 is applied per edge
// (the edge-parallel kernel, whose partial sums cross an atomic) rather than once at the end (the
// node-parallel one).
// =============================================================================
template <GSDDMM_OP op, GSDDMM_MEMBER ll, GSDDMM_MEMBER rr, GSDDMM_REDUCE reduce, int SHAPE, FloatingNum cuda_t, FloatingNum accum_t>
struct BackwardEdge {
    using Plan       = GsddmmBackwardPlan<op, ll, rr, reduce>;
    using Lay        = Layout<SHAPE, cuda_t, accum_t>;
    using RG         = typename Lay::RG;
    using Frag       = typename Lay::Frag;
    using vec_t      = typename Lay::vec_t;
    using GradOp     = GsddmmGradOp<op, Plan::SELF_IS_LHS, Lay::VEC, cuda_t>;
    using EdgeGradOp = GsddmmGradOp<op, !Plan::SELF_IS_LHS, Lay::VEC, cuda_t>;
    static_assert(Plan::HAS_SELF, "this pass has no operand to reduce; the host must not launch it");
    static_assert(
        Plan::KEEP_SELF_ROW == (Plan::NEEDS_SELF_ROW || (Plan::WRITE_EDGE_GRAD && EdgeGradOp::READS_OTHER)),
        "Plan::KEEP_SELF_ROW must match what step() reads"
    );

    struct Pass {
        typename Lay::In d_out, other;
        typename Lay::Out d_edge;
        uint32_t other_stride;
    };

    static __device__ __forceinline__ Pass make(
        Operands<cuda_t> const& x, cuda_t const *const dO, cuda_t *const d_edge, uint32_t v0, uint32_t r
    ) {
        const uint32_t rem = x.nvec - v0;
        Pass P;
        if constexpr (!Plan::IS_DOT) {
            P.d_out = Lay::lanes_in(dO, v0, rem, r);
        }
        if constexpr (Plan::READS_OTHER) {
            P.other = Lay::lanes_in(Plan::SELF_IS_LHS ? x.R : x.L, v0, rem, r);
        }
        if constexpr (Plan::WRITE_EDGE_GRAD) {
            P.d_edge = Lay::lanes_out(d_edge, v0, rem, r);
        }
        P.other_stride = Plan::SELF_IS_LHS ? x.stride_r : x.stride_l;
        return P;
    }

    template <bool PER_EDGE_DIV_SQ, typename id_t>
    static __device__ __forceinline__ void step(
        Pass const& P, accum_t (&acc)[Lay::kAccs], Frag const& self, cuda_t const *const __restrict__ dO, uint32_t row_bytes, id_t e,
        id_t other_id, bool active
    ) {
        Frag d_out;
        if constexpr (Plan::IS_DOT) {
            // Dot's dO is one scalar per edge, broadcast to the row. Each is read once: streamed.
            const vec_t splat(__ldcs(dO + static_cast<uint64_t>(e)));
#pragma unroll
            for (size_t s = 0; s < Lay::S; ++s) {
                d_out.v[s] = splat;
            }
        } else {
            d_out = Lay::template gather<Lay::kOnce>(P.d_out, e, row_bytes);
        }
        Frag other;
        if constexpr (Plan::READS_OTHER) {
            other = Lay::template gather<Plan::OTHER_IS_EDGE ? Lay::kOnce : Lay::Tile::MemoryHint::NoHint>(P.other, other_id, P.other_stride);
        } else {
            other = d_out;  // never read: constant partials
        }
        // An inactive slot read real rows (see the walks), whose values a Div could still turn into
        // a NaN; it adds nothing and stores nothing.
        if (!active) {
            return;
        }
        [[maybe_unused]] Frag g_edge;
#pragma unroll
        for (size_t s = 0; s < Lay::S; ++s) {
            accum_t *const a = &acc[s * Lay::VEC];
            if constexpr (Plan::NEEDS_SELF_ROW) {
                static_assert(op == GSDDMM_OP::Div && Plan::NEGATE_SELF, "NEEDS_SELF_ROW is the negating Div/rhs case");
                if constexpr (PER_EDGE_DIV_SQ) {
                    gsddmm_accum_sub_prod_div_sq<Lay::VEC, cuda_t, accum_t>(a, d_out.v[s], other.v[s], self.v[s]);
                } else {
                    // Formed in accum_t: the -1/self^2 applied once after the sum amplifies any
                    // rounding taken here.
                    gsddmm_accum_fma<Lay::VEC, cuda_t, accum_t>(a, d_out.v[s], other.v[s]);
                }
            } else {
                vec_t contrib = GradOp::apply(d_out.v[s], other.v[s]);
                if constexpr (PER_EDGE_DIV_SQ && Plan::NEGATE_SELF) {
                    contrib.neg_();  // the node-parallel kernel negates once, at the end
                }
                gsddmm_accum_add<Lay::VEC, cuda_t, accum_t>(a, contrib);
            }

            if constexpr (Plan::WRITE_EDGE_GRAD) {
                // The node row is the edge operand's "other" operand. An operand the backward never
                // reads may be a 1-row stand-in, so it is only touched when the partial uses it.
                vec_t node_row;
                if constexpr (EdgeGradOp::READS_OTHER) {
                    node_row = self.v[s];
                } else {
                    node_row = d_out.v[s];  // unread: constant partials
                }
                g_edge.v[s] = EdgeGradOp::apply(d_out.v[s], node_row);
                if constexpr (op == GSDDMM_OP::Div && Plan::OTHER == rr) {
                    // The edge operand is the denominator: -dO * self / edge^2.
                    g_edge.v[s] = gsddmm_div_sq<Lay::VEC, cuda_t, accum_t>(g_edge.v[s], other.v[s]);
                    g_edge.v[s].neg_();
                } else if constexpr (op == GSDDMM_OP::Sub && Plan::OTHER == rr) {
                    g_edge.v[s].neg_();
                }
            }
        }
        if constexpr (Plan::WRITE_EDGE_GRAD) {
            Lay::scatter(P.d_edge, e, row_bytes, g_edge, true);
        }
    }
};

// Sums acc over warps [0, T) of the block into warp 0 (lanes [0, LANES)), in warp order --
// deterministic. Rounds of R elements bound the shared scratch, as in team_sum (row_group.cuh),
// but the warp loop is kept rolled: unrolled over 31 warps it held dozens of loads in flight and set
// the register allocation of the whole kernel -- 54 registers for the 32-warp block against 47 at 4
// warps, which halves the resident 1024-thread blocks, for a merge that runs once per node. Every
// thread of the block calls it, warps past T included (they only pass the barriers).
// Edges a warp of the node backward walks before another joins its row (see the kernel). 8 and 4 tied
// on geomean; 8 merges fewer partial sums.
inline constexpr uint32_t kTeamSteps = 8;

template <size_t W, size_t LANES, size_t E, FloatingNum accum_t>
__device__ __forceinline__ void block_merge(accum_t (&acc)[E], accum_t *const __restrict__ buf, uint32_t warp, uint32_t lane, uint32_t T) {
    constexpr size_t R = kTeamSumRound<W, LANES, E, accum_t>;
#pragma unroll
    for (size_t e0 = 0; e0 < E; e0 += R) {
        if (lane < LANES && warp != 0 && warp < T) {
#pragma unroll
            for (size_t e = 0; e < R && e0 + e < E; ++e) {
                buf[(warp * R + e) * LANES + lane] = acc[e0 + e];
            }
        }
        __syncthreads();
        if (lane < LANES && warp == 0) {
#pragma unroll 1
            for (uint32_t w = 1; w < T; ++w) {
#pragma unroll
                for (size_t e = 0; e < R && e0 + e < E; ++e) {
                    acc[e0 + e] += buf[(w * R + e) * LANES + lane];
                }
            }
        }
        __syncthreads();
    }
}

// One CSR walk with the forward-CSR edge id of every slot: canon[pos] where the walk is over the
// backward CSR (REMAP), the slot's own position otherwise. As for_each_neighbor, ids are fetched 32
// at a time and broadcast by shuffle, the canonical ids alongside; fn(active, j, e) is called
// warp-uniformly. An inactive slot gets neighbour 0 and the walk's first edge id -- real rows.
template <size_t G, bool REMAP, IntegralNum index_t, typename Fn>
__device__ __forceinline__ void for_each_edge(
    index_t const *const __restrict__ ids, index_t const *const __restrict__ canon, std::make_unsigned_t<index_t> first, uint32_t begin,
    uint32_t end, uint32_t lane, Fn&& fn
) {
    using uidx_t             = std::make_unsigned_t<index_t>;
    constexpr uint32_t kStep = kWarpSize / G;
    const uint32_t g         = lane / G;
    for (uint32_t base = begin; base < end; base += kWarpSize) {
        const uint32_t pos = base + lane;
        const bool ok      = pos < end;
        const uidx_t held  = ok ? static_cast<uidx_t>(ids[pos]) : uidx_t{};
        uidx_t held_e{};
        if constexpr (REMAP) {
            held_e = ok ? static_cast<uidx_t>(canon[pos]) : uidx_t{};
        }
        const uint32_t cnt = min(static_cast<uint32_t>(kWarpSize), end - base);
        for (uint32_t t = 0; t < cnt; t += kStep) {
            const uint32_t src = t + g;
            const bool active  = src < cnt;
            const uidx_t j     = __shfl_sync(FULL_WARP_MASK, held, src);
            const uidx_t e     = REMAP ? __shfl_sync(FULL_WARP_MASK, held_e, src) : first + (active ? base + src : begin);
            fn(active, j, e);
        }
    }
}

// =============================================================================
// Backward, node-parallel. One block per node of the pass's CSR; its warps take contiguous shares
// of the node's edges and its groups an edge each, every lane summing its vectors in fp32
// registers. Groups are merged by shuffles and warps by block_merge -- in a fixed order, so the
// result is deterministic -- and warp 0 stores the node's row once: no atomics and no zeroed
// output. Rows of several passes are reduced pass by pass.
//
// A row takes only as many warps as give each at least kTeamSteps steps of edges: the merge is paid
// per warp that joins, and a 32-warp block spreading a row of a few edges over all its warps (the
// heavy bucket of a low-degree graph -- city-roads-M sends 4705 rows of about 7 edges there) spent
// 2.7x the former kernel's time merging mostly empty partial sums. A row for one warp skips the
// merge, the others leaving before any barrier; a larger team merges its T warps only. row_ptr / col_idx are the forward CSR for
// GSDDMM_REDUCE::Dst and the backward one for ::Src, whose slots are CSC positions mapped to dO's numbering by canonical.
// =============================================================================
template <
    GSDDMM_OP op, GSDDMM_MEMBER ll, GSDDMM_MEMBER rr, GSDDMM_REDUCE reduce, size_t W, int SHAPE, FloatingNum cuda_t, IntegralNum index_t,
    FloatingNum accum_t
>
__global__ void __launch_bounds__(W *kWarpSize, kMinBlocks<Family::BackwardNode, W * kWarpSize, SHAPE, op, cuda_t>) gsddmm_backward_node_kernel(
    NodeBackwardParams<cuda_t, index_t> p
) {
    using uidx_t           = std::make_unsigned_t<index_t>;
    using Plan             = GsddmmBackwardPlan<op, ll, rr, reduce>;
    using Lay              = Layout<SHAPE, cuda_t, accum_t>;
    using RG               = typename Lay::RG;
    using Frag             = typename Lay::Frag;
    using vec_t            = typename Lay::vec_t;
    using Tile             = typename Lay::Tile;
    using Edge             = BackwardEdge<op, ll, rr, reduce, SHAPE, cuda_t, accum_t>;
    constexpr bool REMAP   = reduce == GSDDMM_REDUCE::Src;
    constexpr size_t kAccs = Lay::kAccs;

    const uint32_t lane = threadIdx.x % kWarpSize;
    const uint32_t warp = threadIdx.x / kWarpSize;
    const uint32_t r    = lane % Lay::G;

    const index_t node = p.node_indices[blockIdx.x];
    if (static_cast<int64_t>(node) >= p.N) [[unlikely]] {
        return;
    }
    const index_t begin = p.row_ptr[node];
    const uint32_t n    = static_cast<uint32_t>(p.row_ptr[node + 1] - begin);
    const uint32_t T    = min(static_cast<uint32_t>(W), max(1u, ceil_div(n, kTeamSteps * Lay::kGroups)));
    const bool solo     = T == 1;
    if (solo && warp != 0) {
        return;
    }
    const uint2 share = warp < T ? warp_share(n, warp, T, Lay::kGroups) : make_uint2(0, 0);

    Operands<cuda_t> const& x    = p.x;
    cuda_t const *const self_row = Plan::SELF_IS_LHS ? row_at(x.L, node, x.stride_l) : row_at(x.R, node, x.stride_r);
    cuda_t *const d_self_row     = p.d_self + static_cast<uint64_t>(static_cast<uidx_t>(node)) * x.D;
    index_t const *const ids     = p.col_idx + begin;
    index_t const *const canon   = REMAP ? p.canonical + begin : nullptr;
    extern __shared__ __align__(16) uint8_t sh_raw[];
    accum_t *const merge_buf = reinterpret_cast<accum_t *>(sh_raw);

    const uint32_t passes = Lay::kMulti ? x.passes : 1u;
    for (uint32_t pass = 0; pass < passes; ++pass) {
        const uint32_t v0  = pass * Lay::kPass;
        const uint32_t rem = x.nvec - v0;
        Frag self;
        if constexpr (Plan::KEEP_SELF_ROW) {
            RG::load(self, self_row + static_cast<size_t>(v0) * Lay::VEC, r, rem);
        }
        accum_t acc[kAccs];
        RG::zero(acc);
        if (share.x < share.y) {
            const typename Edge::Pass P = Edge::make(x, p.dO, p.d_edge, v0, r);
            for_each_edge<Lay::G, REMAP>(ids, canon, static_cast<uidx_t>(begin), share.x, share.y, lane, [&](bool active, uidx_t j, uidx_t e) {
                Edge::template step<false>(P, acc, self, p.dO, x.row_bytes, e, Plan::OTHER_IS_EDGE ? e : j, active);
            });
        }

        // Groups -> warp, then warps -> warp 0 (every warp takes part in the barriers).
        RG::sum_groups(acc);
        if (W > 1 && !solo) {  // block-uniform
            block_merge<W, Lay::G, kAccs>(acc, merge_buf, warp, lane, T);
        }
        if (warp == 0 && lane < Lay::G) {
#pragma unroll
            for (size_t s = 0; s < Lay::S; ++s) {
                const uint32_t v = RG::vec_index(r, s);
                if (v < rem) {
                    accum_t *const a = &acc[s * Lay::VEC];
                    if constexpr (Plan::NEEDS_SELF_ROW) {
                        // -1/self^2, constant over the sum: applied once, on the fp32 sum. Dividing by
                        // the own row is exactly the negating case, so the sign rides along.
                        gsddmm_accum_neg_div_sq<Lay::VEC, cuda_t, accum_t>(a, self.v[s]);
                    }
                    vec_t out = gsddmm_accum_to_vec<Lay::VEC, cuda_t, accum_t>(a);
                    if constexpr (Plan::NEGATE_SELF && !Plan::NEEDS_SELF_ROW) {
                        out.neg_();
                    }
                    Tile::template write<Tile::MemoryHint::Streaming>(d_self_row + static_cast<size_t>(v0) * Lay::VEC, v, out);
                }
            }
        }
    }
}

// =============================================================================
// Backward, edge-parallel. Work distribution is the forward's: warp w owns a contiguous chunk of the
// edge list -- grouped by the node this pass reduces -- lane k caching edge k's ids.
//
// RUNS. A chunk is a short sequence of runs of edges sharing a target node, usually exactly one. One
// warp vote finds where the runs start; the warp then takes the runs in turn, its groups splitting
// each run's edges, sums the run in fp32 registers and adds it to the node's row once. The reduced
// node's own row, which Div's -1/self^2 and the edge operand's gradient need, is read once per run.
//
// FLUSH. After the groups are merged, group 0 holds the run's sum blocked by lane -- lane r owns
// vectors r, r + G, ... -- which is the worst shape for an atomic: scalar atomics at a fixed element
// touch one sector per lane (measured 32 sectors per request on an A100, against the 4 a 128-byte
// request needs). The sum is therefore staged through a per-warp shared row and added in contiguous
// spans: float4 atomics on sm_90+ for 16-byte rows (512 bytes per instruction), scalar ones of 128
// bytes otherwise.
// =============================================================================
template <
    GSDDMM_OP op, GSDDMM_MEMBER ll, GSDDMM_MEMBER rr, GSDDMM_REDUCE reduce, int SHAPE, FloatingNum cuda_t, IntegralNum index_t,
    FloatingNum accum_t, bool REMAP_IDS
>
__global__ void
    __launch_bounds__(kWarpSize *kGsddmmEdgeMaxWarpsPerBlock, kMinBlocks<Family::BackwardEdge, kWarpSize * kGsddmmEdgeMaxWarpsPerBlock, SHAPE, op, cuda_t>)
    gsddmm_backward_edge_kernel(EdgeBackwardParams<cuda_t, index_t, accum_t> p) {
    using eidx_t           = std::make_unsigned_t<index_t>;
    using Plan             = GsddmmBackwardPlan<op, ll, rr, reduce>;
    using Lay              = Layout<SHAPE, cuda_t, accum_t>;
    using RG               = typename Lay::RG;
    using Frag             = typename Lay::Frag;
    using Edge             = BackwardEdge<op, ll, rr, reduce, SHAPE, cuda_t, accum_t>;
    constexpr size_t kAccs = Lay::kAccs;
    static_assert(std::is_same_v<accum_t, float>, "fp32 atomics only");

    const uint32_t lane = threadIdx.x % kWarpSize;
    const uint32_t warp = threadIdx.x / kWarpSize;
    const uint32_t g    = lane / Lay::G;
    const uint32_t r    = lane % Lay::G;

    const eidx_t block_linear = static_cast<eidx_t>((static_cast<uint64_t>(blockIdx.z) * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x);
    const eidx_t edge_base    = (block_linear * static_cast<eidx_t>(blockDim.x / kWarpSize) + warp) * static_cast<eidx_t>(p.chunk);
    if (edge_base >= p.E) [[unlikely]] {
        return;
    }
    const uint32_t cnt = static_cast<uint32_t>(p.E - edge_base < p.chunk ? p.E - edge_base : p.chunk);

    // Lanes past the chunk hold node 0 and canonical id 0: real rows for inactive groups to read.
    index_pair_t<index_t> my_pair{};
    eidx_t my_canon{};
    if (lane < cnt) {
        my_pair = __ldcs(&p.pairs[edge_base + lane]);
        if constexpr (REMAP_IDS) {
            my_canon = static_cast<eidx_t>(p.canonical[edge_base + lane]);
        }
    }
    const eidx_t my_self  = static_cast<eidx_t>(reduce == GSDDMM_REDUCE::Dst ? my_pair.y : my_pair.x);
    const eidx_t my_other = static_cast<eidx_t>(reduce == GSDDMM_REDUCE::Dst ? my_pair.x : my_pair.y);
    const eidx_t prev     = __shfl_up_sync(FULL_WARP_MASK, my_self, 1);
    const uint32_t starts = __ballot_sync(FULL_WARP_MASK, lane < cnt && (lane == 0 || my_self != prev));

    Operands<cuda_t> const& x  = p.x;
    const uint32_t D           = x.D;
    cuda_t const *const self_b = Plan::SELF_IS_LHS ? x.L : x.R;
    const uint32_t self_stride = Plan::SELF_IS_LHS ? x.stride_l : x.stride_r;
    extern __shared__ __align__(16) uint8_t sh_raw[];
    accum_t *const stage = reinterpret_cast<accum_t *>(sh_raw) + static_cast<size_t>(warp) * Lay::kPass * Lay::VEC;

    // Adds the merged sum (group 0) of one pass of a run into node's row.
    auto flush = [&](eidx_t node, accum_t const(&acc)[kAccs], uint32_t v0, uint32_t rem) {
        if (lane < Lay::G) {
#pragma unroll
            for (size_t s = 0; s < Lay::S; ++s) {
#pragma unroll
                for (size_t e = 0; e < Lay::VEC; ++e) {
                    stage[RG::vec_index(r, s) * Lay::VEC + e] = acc[s * Lay::VEC + e];
                }
            }
        }
        __syncwarp();
        const uint32_t count = (rem < Lay::kPass ? rem : Lay::kPass) * static_cast<uint32_t>(Lay::VEC);
        accum_t *const dst   = p.d_self + static_cast<uint64_t>(node) * D + static_cast<size_t>(v0) * Lay::VEC;
        // 16-byte rows keep every row offset a multiple of 4 floats. The vector type is dependent on
        // accum_t, so the sm_90-only overload is looked up only where this branch is kept.
        using f4_t = std::conditional_t<sizeof(accum_t) == 4, float4, accum_t>;
        if constexpr (kCudaArch >= 900 && Shape<SHAPE>::kWide && Lay::VEC % 4 == 0) {
            for (uint32_t i = lane; i < count / 4; i += kWarpSize) {
                atomicAdd(reinterpret_cast<f4_t *>(dst) + i, reinterpret_cast<f4_t const *>(stage)[i]);
            }
        } else {
            for (uint32_t i = lane; i < count; i += kWarpSize) {
                atomicAdd(dst + i, stage[i]);
            }
        }
        // The next run's staging must not overwrite values still being read.
        __syncwarp();
    };

    const uint32_t passes = Lay::kMulti ? x.passes : 1u;
    for (uint32_t pass = 0; pass < passes; ++pass) {
        const uint32_t v0           = pass * Lay::kPass;
        const uint32_t rem          = x.nvec - v0;
        const typename Edge::Pass P = Edge::make(x, p.dO, p.d_edge, v0, r);
        for (uint32_t runs = starts; runs != 0;) {
            const uint32_t a = static_cast<uint32_t>(__ffs(runs)) - 1;
            runs &= runs - 1;
            const uint32_t b  = runs != 0 ? static_cast<uint32_t>(__ffs(runs)) - 1 : cnt;
            const eidx_t node = __shfl_sync(FULL_WARP_MASK, my_self, static_cast<int>(a));
            Frag self;
            if constexpr (Plan::KEEP_SELF_ROW) {
                RG::load(self, row_at(self_b, node, self_stride) + static_cast<size_t>(v0) * Lay::VEC, r, rem);
            }
            accum_t acc[kAccs];
            RG::zero(acc);
            for (uint32_t t = a; t < b; t += Lay::kGroups) {
                const uint32_t k  = t + g;
                const bool active = k < b;
                eidx_t e;
                if constexpr (REMAP_IDS) {
                    e = __shfl_sync(FULL_WARP_MASK, my_canon, static_cast<int>(k));
                } else {
                    e = edge_base + (active ? k : a);
                }
                const eidx_t j = __shfl_sync(FULL_WARP_MASK, my_other, static_cast<int>(k));
                Edge::template step<true>(P, acc, self, p.dO, x.row_bytes, e, Plan::OTHER_IS_EDGE ? e : j, active);
            }
            RG::sum_groups(acc);
            flush(node, acc, v0, rem);
        }
    }
}

}  // namespace gsddmm
