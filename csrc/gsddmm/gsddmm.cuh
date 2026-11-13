#pragma once

#include <cstdint>
#include <type_traits>

#include "common.cuh"
#include "common/row_group.cuh"
#include "common/row_shape.cuh"

// =============================================================================
// g-SDDMM (generalized sampled dense-dense matrix multiplication), forward and backward, for any
// feature dim D in [1, rows::kMaxHeadDim].
//
// The graph is given in CSR (row_ptr/col_idx). Row node i is the *destination* of the edges stored
// in its row, col node j = col_idx[e] is the *source* -- the same message-passing orientation as
// the GT/GATv2 kernels (edge e carries the message j -> i). For every edge position e in
// [row_ptr[i], row_ptr[i+1]):
//
//     lhs = L[sel(ll), :],  rhs = R[sel(rr), :]
//     sel(Src_V) = j   (neighbor node row, L/R are [N, D])
//     sel(Dst_V) = i   (row node row,       L/R are [N, D])
//     sel(Edge)  = e   (edge row,           L/R are [E, D])
//
//     op in {Add, Sub, Mul, Div}:  O[e, :] = op(lhs, rhs)   (O is [E, D])
//     op == Copy:                  O[e, :] = lhs            (R is never read)
//     op == Dot:                   O[e]     = sum_d lhs[d] * rhs[d]
//                                (products formed in cuda_t, reduced in accum_t, result written
//                                 as cuda_t;  O is [E])
//
// Row layout. A row of D features is laid across a lane group of G lanes as in
// common/row_group.cuh: 16-byte vectors when every row read is 16-byte aligned and D a whole
// number of them, single elements otherwise. Short rows take one group per edge -- G = 4, 8 or 16
// lanes, so a warp works on 32 / G edges at once -- and rows of more than 32 vectors take whole
// warps with two vectors per lane, swept in passes of 64 vectors when longer than that. Every op is
// elementwise over the row except Dot's reduction, which a group accumulates over its passes before
// reducing across its lanes, so no path needs scratch memory or a second kernel.
//
// Four kernels implement it:
//
//   forward_node   one block per (bucketed) CSR row -- or per chunk of a heavy row -- its warps
//                  splitting the row's edges in contiguous shares. The Dst_V operand row is the same
//                  for every edge of the block and is held in registers.
//   forward_edge   one warp per contiguous chunk of an explicit [E, 2] edge list. A chunk whose
//                  edges share the grouping node's row (the list is grouped by it) holds that row
//                  in registers too.
//   backward_node  one block per node of the pass's CSR; the node's gradient is reduced in fp32
//                  registers across the block and stored once. No atomics, deterministic.
//   backward_edge  one warp per edge chunk; the chunk's runs of edges sharing a target node are
//                  summed in registers and added once per run with fp32 atomics. Load balanced.
// =============================================================================

namespace gsddmm {

using turbo_gnn::rows::Shape;

// Operation to do over
enum class GSDDMM_OP : uint8_t {
    Add,   // Add the right operand to the left(elementwise)
    Sub,   // Subtract the right operand from the left(elementwise)
    Mul,   // Multiply the right operand by the left(elementwise)
    Div,   // Divide the left operand by the right(elementwise)
    Dot,   // Get a dot-product of left and right operands(multiply elementsie and then reduce by summation). The feature axis is reduced
    Copy,  // Copy left operand feature to the corresponding edges
};

// Member to take feature values from. Maybe Source vertex, destination vertex or edge
enum class GSDDMM_MEMBER : uint8_t {
    Src_V,  // Source vertex
    Dst_V,  // Destination vertex
    Edge,   // Edge
};

// Compile-time operand routing for one forward instantiation.
template <GSDDMM_OP op, GSDDMM_MEMBER ll, GSDDMM_MEMBER rr>
struct GsddmmPlan {
    // The right operand is read for every op except Copy (Copy propagates lhs only).
    static constexpr bool USE_R = (op != GSDDMM_OP::Copy);
    static_assert((ll != rr) || !USE_R, "You should use more efficient methods to do ops between the same dense data.");
    static_assert(op != GSDDMM_OP::Copy || ll != GSDDMM_MEMBER::Edge, "We can not copy edge features to the edge.");

    static constexpr bool L_DST     = (ll == GSDDMM_MEMBER::Dst_V);
    static constexpr bool R_DST     = USE_R && (rr == GSDDMM_MEMBER::Dst_V);
    static constexpr bool READS_DST = L_DST || R_DST;
    static constexpr bool IS_DOT    = (op == GSDDMM_OP::Dot);

    // The edge kernel's list is grouped by destination when an operand reads it, by source
    // otherwise, so that the operand of the grouping member is shared by a chunk's edges.
    static constexpr GSDDMM_MEMBER GROUPING = READS_DST ? GSDDMM_MEMBER::Dst_V : GSDDMM_MEMBER::Src_V;
    static constexpr bool L_GROUPED         = (ll == GROUPING);
    static constexpr bool R_GROUPED         = USE_R && (rr == GROUPING);
    static_assert(L_GROUPED || R_GROUPED, "every instantiated member pair has a node operand");
};

// =============================================================================
// Backward
// =============================================================================
//
// With O[e] = op(L[sel(ll)], R[sel(rr)]) and dO given, the per-edge partials are
//
//     add:   dL = dO,            dR = dO
//     sub:   dL = dO,            dR = -dO
//     mul:   dL = dO * R,        dR = dO * L
//     div:   dL = dO / R,        dR = -dO * L / R^2
//     copy:  dL = dO             (R is never read)
//     dot:   dL = dO[e] * R,     dR = dO[e] * L        (dO is one scalar per edge)
//
// Each partial then lands on the row its operand was read from, which is what makes the backward a
// *reduction* rather than a map:
//
//   - an Edge operand owns one row per edge, so its gradient is a plain store;
//   - a Src_V operand's gradient sums over the node's OUTGOING edges, i.e. the rows of the backward
//     (source-grouped) CSR;
//   - a Dst_V operand's gradient sums over its INCOMING edges, i.e. the rows of the forward CSR.
//
// GSDDMM_REDUCE names which of those two node reductions a launch performs. Every instantiated
// member pair holds at most one Src_V and one Dst_V operand, so the passes a call needs follow from
// (ll, rr) alone and never write the same buffer: (Src_V, Dst_V) runs both passes, a pair with an
// Edge operand runs the single pass for its node operand and writes the edge gradient on the way
// through, and Copy runs the one pass for its lhs.
//
// Every partial is elementwise over the feature dim -- Dot's included, its dO being a scalar per
// edge -- so a row of any length is reduced pass by pass with nothing to carry between passes.
enum class GSDDMM_REDUCE : uint8_t {
    Dst,  // walk the forward CSR, reduce per destination node
    Src,  // walk the backward CSR, reduce per source node
};

// Compile-time operand routing for one backward pass.
template <GSDDMM_OP op, GSDDMM_MEMBER ll, GSDDMM_MEMBER rr, GSDDMM_REDUCE reduce>
struct GsddmmBackwardPlan {
    using Fwd = GsddmmPlan<op, ll, rr>;

    static constexpr GSDDMM_MEMBER SELF_MEMBER = (reduce == GSDDMM_REDUCE::Dst) ? GSDDMM_MEMBER::Dst_V : GSDDMM_MEMBER::Src_V;

    // Which operand this pass reduces. Exactly one of the two can match, because the instantiated
    // pairs never repeat a member.
    static constexpr bool SELF_IS_LHS = (ll == SELF_MEMBER);
    static constexpr bool SELF_IS_RHS = Fwd::USE_R && (rr == SELF_MEMBER);
    static constexpr bool HAS_SELF    = SELF_IS_LHS || SELF_IS_RHS;
    static_assert(!(SELF_IS_LHS && SELF_IS_RHS), "a member appears at most once per instantiated pair");

    // The other operand, gathered per edge. For Copy there is none.
    static constexpr bool HAS_OTHER      = Fwd::USE_R && HAS_SELF;
    static constexpr GSDDMM_MEMBER OTHER = SELF_IS_LHS ? rr : ll;
    static constexpr bool OTHER_IS_EDGE  = HAS_OTHER && (OTHER == GSDDMM_MEMBER::Edge);

    // Mul/Div/Dot multiply by the other operand's row; Add/Sub/Copy have constant partials, so those
    // passes are pure reductions of dO and read no operands.
    static constexpr bool NEEDS_OTHER_ROW = HAS_OTHER && (op == GSDDMM_OP::Mul || op == GSDDMM_OP::Div || op == GSDDMM_OP::Dot);

    // Div's denominator gradient is -dO * L / R^2. R is this pass's own row, so the -1/R^2 factor is
    // constant across the sum: the node-parallel kernel applies it once at the end, the edge-parallel
    // one once per edge from the run's cached row -- the own row is read once per node or run, never
    // per edge.
    static constexpr bool NEEDS_SELF_ROW = (op == GSDDMM_OP::Div) && SELF_IS_RHS;
    // Negated partials: sub's rhs, and div's rhs (through the factor above).
    static constexpr bool NEGATE_SELF = SELF_IS_RHS && (op == GSDDMM_OP::Sub || op == GSDDMM_OP::Div);

    // The edge operand's gradient is a plain per-edge store, written by this pass (the only one that
    // runs when a pair has an Edge operand).
    static constexpr bool WRITE_EDGE_GRAD = OTHER_IS_EDGE;
    // ... and for Div it needs the edge row itself (dR = -dO * L / R^2 with R edge-indexed), so that
    // row is read per edge even when SELF does not need it.
    static constexpr bool NEEDS_OTHER_ROW_FOR_EDGE_GRAD = WRITE_EDGE_GRAD && (op == GSDDMM_OP::Div) && (OTHER == rr);

    static constexpr bool READS_OTHER = NEEDS_OTHER_ROW || NEEDS_OTHER_ROW_FOR_EDGE_GRAD;
    static constexpr bool IS_DOT      = Fwd::IS_DOT;

    // The reduced operand's own row is needed by Div's -1/self^2 factor and by the edge operand's
    // gradient (where the node row is its "other" operand).
    static constexpr bool KEEP_SELF_ROW =
        NEEDS_SELF_ROW || (WRITE_EDGE_GRAD && (op == GSDDMM_OP::Mul || op == GSDDMM_OP::Div || op == GSDDMM_OP::Dot));
};

// =============================================================================
// Row shapes
//
// A shape code as in common/row_shape.cuh: 1000 * WIDE + 10 * G + S. The shapes compiled are
//
//   G = 4, 8, 16, 32 with one vector a lane  rows of up to G vectors, one group per edge;
//   G = 32 with two vectors a lane           rows of more than 32 vectors, in passes of 64.
//
// G = 4 is the smallest group, as for the attention kernels: below it the per-edge address math and
// id shuffles, not idle lanes, bound a row of a few elements. Whole warps sweep long rows rather
// than take a group per edge, since a row of more than 32 vectors fills a warp on its own.
// =============================================================================

inline int plan_shape(int64_t D, size_t elem_size, bool wide) {
    const int64_t nvec = D / (wide ? static_cast<int64_t>(16 / elem_size) : 1);
    for (size_t g : {size_t{4}, size_t{8}, size_t{16}, kWarpSize}) {
        if (nvec <= static_cast<int64_t>(g)) {
            return turbo_gnn::rows::shape_code(wide, g, 1);
        }
    }
    return turbo_gnn::rows::shape_code(wide, kWarpSize, 2);
}

inline auto make_shape_variant(int code) { return MakeIntVariant<1041, 1081, 1161, 1321, 1322, 41, 81, 161, 321, 322>(code); }

// Vectors one pass of a shape covers; the S = 2 whole-warp shape is the only one that may take
// several passes.
template <int SHAPE>
inline constexpr uint32_t kPassVecs = static_cast<uint32_t>(Shape<SHAPE>::G * Shape<SHAPE>::S);

template <int SHAPE>
inline constexpr bool kMultiPass = Shape<SHAPE>::G == kWarpSize && Shape<SHAPE>::S == 2;

template <int SHAPE>
inline uint32_t row_passes(uint32_t nvec) {
    return ceil_div(nvec, kPassVecs<SHAPE>);
}

// Edge-block kernel launch limits (validated by the binding, baked into __launch_bounds__).
inline constexpr size_t kGsddmmEdgeMaxWarpsPerBlock = 8;
inline constexpr size_t kGsddmmEdgeMaxEdgesPerWarp  = kWarpSize;  // lane k caches edge k's (src, dst) pair

// Edges a warp of the edge kernels walks for a requested edges_per_warp. The parameter counts the
// edges each lane GROUP walks: a warp of 32 / G groups takes edges_per_warp * 32 / G of them, at
// most kGsddmmEdgeMaxEdgesPerWarp, so a short row that puts several groups in a warp does not leave
// all but edges_per_warp of them idle. For whole-warp rows (G = 32) it is the warp's count.
template <int SHAPE>
inline uint32_t edge_chunk(uint32_t edges_per_warp) {
    const uint32_t groups = static_cast<uint32_t>(kWarpSize / Shape<SHAPE>::G);
    const uint32_t chunk  = edges_per_warp * groups;
    return chunk < kGsddmmEdgeMaxEdgesPerWarp ? chunk : static_cast<uint32_t>(kGsddmmEdgeMaxEdgesPerWarp);
}

// =============================================================================
// Kernel parameters
// =============================================================================

// The two operands' rows: row k of L starts stride_l BYTES after row 0 (any value, including 0 for an
// expanded operand), features contiguous. Strides are in bytes so that a row address is one 32x32-bit
// wide multiply-add onto a byte pointer; in elements, ptxas re-associates the scaling into four
// instructions per row. row_bytes = D * sizeof(cuda_t), the stride of the [E, D] outputs and dO.
// nvec = D / VEC vectors per row; passes = row_passes<SHAPE>(nvec).
template <FloatingNum cuda_t>
struct Operands {
    cuda_t const *L;
    cuda_t const *R;
    uint32_t stride_l;
    uint32_t stride_r;
    uint32_t D;
    uint32_t row_bytes;
    uint32_t nvec;
    uint32_t passes;
};

template <FloatingNum cuda_t, IntegralNum index_t>
struct NodeForwardParams {
    Operands<cuda_t> x;
    cuda_t *O;  // [E, D] contiguous, or [E] for Dot
    index_t const *row_ptr;
    index_t const *col_idx;
    index_t const *node_indices;  // node_i = node_indices[blockIdx.x]
    index_t const *block_part;    // nullptr, or the chunk of block blockIdx.x within its node's edge list
    uint32_t edges_per_block;     // chunk width when block_part != nullptr
    int64_t N;
};

// The edge kernels read an explicit [E, 2] list of (src, dst) node-id pairs, reinterpreted as
// index_pair_t<index_t>, and carry every id in index_t's unsigned form. canonical (nullable, [E])
// is the canonical edge id of each traversal slot, applied to the Edge operand rows and the output
// row, so the list may be grouped by source for locality while the output stays numbered by
// forward-CSR position. REMAP_IDS is the compile-time shadow of its nullness.
template <FloatingNum cuda_t, IntegralNum index_t>
struct EdgeForwardParams {
    Operands<cuda_t> x;
    cuda_t *O;
    index_pair_t<index_t> const *pairs;
    index_t const *canonical;
    std::make_unsigned_t<index_t> E;
    uint32_t chunk;  // edges per warp, edge_chunk<SHAPE>(edges_per_warp)
};

// dO is [E, D] contiguous ([E] for Dot); d_self the reduced node gradient, [N, D] contiguous;
// d_edge the edge operand's gradient, [E, D] contiguous, written only when the pair has an Edge
// operand (nullptr otherwise).
template <FloatingNum cuda_t, IntegralNum index_t>
struct NodeBackwardParams {
    Operands<cuda_t> x;
    cuda_t const *dO;
    cuda_t *d_self;
    cuda_t *d_edge;
    index_t const *row_ptr;  // the forward CSR for GSDDMM_REDUCE::Dst, the backward one for ::Src
    index_t const *col_idx;
    index_t const *node_indices;
    index_t const *canonical;  // ::Src: maps a backward-CSR slot to its forward-CSR edge id
    int64_t N;
};

template <FloatingNum cuda_t, IntegralNum index_t, FloatingNum accum_t>
struct EdgeBackwardParams {
    Operands<cuda_t> x;
    cuda_t const *dO;
    accum_t *d_self;  // fp32 accumulation buffer, zeroed by the caller
    cuda_t *d_edge;
    index_pair_t<index_t> const *pairs;  // grouped by the node this pass reduces
    index_t const *canonical;
    std::make_unsigned_t<index_t> E;
    uint32_t chunk;
};

// Shared memory of one backward_node block: the deterministic cross-warp merge (block_merge in
// gsddmm.cu, sized as team_sum's rounds).
template <size_t W, int SHAPE, FloatingNum cuda_t, FloatingNum accum_t = float>
inline constexpr size_t backward_node_smem_bytes() {
    constexpr size_t G    = Shape<SHAPE>::G;
    constexpr size_t accs = Shape<SHAPE>::S * Shape<SHAPE>::template vec<cuda_t>;
    return kTeamSumBufferElems<W, G, accs, accum_t> * sizeof(accum_t);
}

// Shared memory of one backward_edge warp: the staging row through which a run's sum is flushed,
// one pass of accum_t elements.
template <int SHAPE, FloatingNum cuda_t, FloatingNum accum_t = float>
inline constexpr size_t backward_edge_smem_bytes_per_warp() {
    return kPassVecs<SHAPE> * Shape<SHAPE>::template vec<cuda_t> * sizeof(accum_t);
}

};  // namespace gsddmm
