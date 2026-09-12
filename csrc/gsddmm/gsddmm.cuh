#pragma once

#include "common.cuh"

namespace gsddmm {

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

// =============================================================================
// GSDDMM (Generalized Sampled-Dense-Dense Matrix Multiplication) forward
// =============================================================================
//
// The graph is given in CSR (row_ptr/col_idx). Row node i is the *destination*
// of the edges stored in its row, col node j = col_idx[e] is the *source* —
// the same message-passing orientation as the GT/GATv2 kernels (edge e carries
// the message j -> i). For every edge position e in [row_ptr[i], row_ptr[i+1]):
//
//     lhs = L[sel(ll), :],  rhs = R[sel(rr), :]
//     sel(Src_V) = j   (neighbor node row, L/R are [N, D])
//     sel(Dst_V) = i   (row node row,       L/R are [N, D])
//     sel(Edge)  = e   (edge row,           L/R are [E, D])
//
//     op in {Add, Sub, Mul, Div}:  O[e, :] = op(lhs, rhs)   (O is [E, D])
//     op == Copy:                  O[e, :] = lhs            (R is never read)
//     op == Dot:                   O[e]     = sum_d lhs[d] * rhs[d]
//                                (products formed in cuda_t, reduced in accum_t,
//                                 result written as cuda_t;  O is [E])
//
// Parallelization (mirrors the GT forward kernel): the grid covers a bucket of
// nodes through the node_indices indirection (node_i = node_indices[blockIdx.x]),
// the block is (kWarpSize, N_PER_BLOCK); each warp owns one edge at a time and
// walks the node's edge list with stride N_PER_BLOCK, lanes splitting the
// D_CONST features into TW-wide vector tiles.
//
// Compile-time routing (GsddmmPlan): Dst_V operand rows are identical for every
// edge of the CSR row, so they are staged once per block in shared memory with
// 16-byte copies; Src_V / Edge rows are gathered per edge — either by direct
// wide global loads (PIPELINE_STAGES == 0) or through a cp.async multi-stage
// pipeline into shared memory (PIPELINE_STAGES >= 1). All op/member branches
// are resolved with if constexpr, so unused operands are never loaded.

// Compile-time operand routing for one GSDDMM instantiation.
template <GSDDMM_OP op, GSDDMM_MEMBER ll, GSDDMM_MEMBER rr>
struct GsddmmPlan {
    // The right operand is read for every op except Copy (Copy propagates lhs only).
    static constexpr bool USE_R = (op != GSDDMM_OP::Copy);
    static_assert((ll != rr) || !USE_R, "You should use more efficient methods to do ops between the same dense data.");
    static_assert(op != GSDDMM_OP::Copy || ll != GSDDMM_MEMBER::Edge, "We can not copy edge features to the edge.");

    static constexpr bool L_DST = (ll == GSDDMM_MEMBER::Dst_V);
    static constexpr bool R_DST = USE_R && (rr == GSDDMM_MEMBER::Dst_V);

    // Dst_V rows are staged once per block in shared memory (they are the same
    // for every edge of the CSR row); Src_V / Edge rows are gathered per edge.
    static constexpr size_t NUM_DST_ROWS  = (L_DST ? 1 : 0) + (R_DST ? 1 : 0);
    static constexpr size_t NUM_EDGE_ROWS = (L_DST ? 0 : 1) + (R_DST ? 0 : 1);

    // Slot of each operand inside the per-edge rows[] array (-1: staged in shared).
    static constexpr int L_SLOT = L_DST ? -1 : 0;
    static constexpr int R_SLOT = R_DST ? -1 : (L_DST ? 0 : 1);

    // true: the per-edge row is indexed by edge position e; false: by neighbor node j.
    static constexpr bool L_EDGE_INDEXED = (ll == GSDDMM_MEMBER::Edge);
    static constexpr bool R_EDGE_INDEXED = USE_R && (rr == GSDDMM_MEMBER::Edge);

    static constexpr bool R_FIRST = R_DST || (rr == GSDDMM_MEMBER::Src_V && ll == GSDDMM_MEMBER::Edge);

    static constexpr bool IS_DOT = (op == GSDDMM_OP::Dot);
};

// Dynamic shared memory requirement of GSDDMM_forward_normal for one instantiation.
template <GSDDMM_OP op, GSDDMM_MEMBER ll, GSDDMM_MEMBER rr, size_t N_PER_BLOCK, size_t D_CONST, FloatingNum cuda_t, uint8_t PIPELINE_STAGES>
inline consteval size_t gsddmm_forward_shmem_bytes() {
    using Plan = GsddmmPlan<op, ll, rr>;
    return Plan::NUM_DST_ROWS * D_CONST * sizeof(cuda_t) +
           (PIPELINE_STAGES > 0 ? N_PER_BLOCK * Plan::NUM_EDGE_ROWS * (PIPELINE_STAGES + 1) * D_CONST * sizeof(cuda_t) : 0);
}

// Edge-block kernel launch limits (validated by the binding, baked into __launch_bounds__).
inline constexpr size_t kGsddmmEdgeMaxWarpsPerBlock = 8;
inline constexpr size_t kGsddmmEdgeMaxEdgesPerWarp  = kWarpSize;  // lane k caches edge k's (src, dst) pair

// Dynamic shared memory requirement of GSDDMM_forward_edge_block per WARP (the
// block needs warps_per_block times this). Every operand of the edge kernel is
// gathered per edge (there is no shared Dst_V row across a warp's edge chunk), so
// the ring buffer holds one L row and, unless the op is Copy, one R row per slot.
template <GSDDMM_OP op, GSDDMM_MEMBER ll, GSDDMM_MEMBER rr, size_t D_CONST, FloatingNum cuda_t, uint8_t PIPELINE_STAGES>
inline consteval size_t gsddmm_forward_edge_shmem_bytes_per_warp() {
    constexpr size_t rows = GsddmmPlan<op, ll, rr>::USE_R ? 2 : 1;
    return PIPELINE_STAGES > 0 ? rows * (PIPELINE_STAGES + 1) * D_CONST * sizeof(cuda_t) : 0;
}

// forward kernel: one thread block per (bucketed) CSR row node -- or, when
// block_part != nullptr, per edges_per_block-wide chunk of a node's edge list
// (node_indices then lists the node of every block, block_part its chunk index;
// this bounds a block's work by the chunk size instead of the node's degree).
// Each warp of the block owns one edge at a time, lanes split the D_CONST
// features into vector tiles.
template <GSDDMM_OP op, GSDDMM_MEMBER ll, GSDDMM_MEMBER rr, size_t N_PER_BLOCK, size_t D_CONST, FloatingNum cuda_t, IntegralNum index_t, FloatingNum accum_t = float, int PIPELINE_STAGES = 0>
__global__ void __launch_bounds__(N_PER_BLOCK * kWarpSize) GSDDMM_forward_normal ( // no-format
    size_t N,
    cuda_t const *__restrict__ L, cuda_t const *__restrict__ R, cuda_t *__restrict__ O,
    index_t const *__restrict__ row_ptr, index_t const *__restrict__ col_idx,
    index_t const *__restrict__ node_indices,  // node indirection: node_i = node_indices[blockIdx.x]
    index_t const *__restrict__ block_part,    // nullptr, or chunk index of block blockIdx.x within its node's edge list
    uint32_t edges_per_block                   // chunk width when block_part != nullptr
);

// Edge-block variant: reads an explicit [E, 2] edge list of (src, dst) node-id
// pairs (reinterpreted as ulonglong2) instead of the CSR. Warp w (global warp
// index over the grid, blockDim.y warps per block) owns the contiguous edge
// chunk [w * edges_per_warp, (w + 1) * edges_per_warp), 1 <= edges_per_warp <=
// kGsddmmEdgeMaxEdgesPerWarp. PIPELINE_STAGES == 0 gathers the operand rows with
// direct wide loads; PIPELINE_STAGES >= 1 prefetches them that many edges ahead
// with cp.async into a per-warp shared ring buffer (L and R rows in the same
// stage). Defaults (stages 0, one edge per warp, one warp per block) reproduce
// the original one-edge-per-block layout exactly.
//
// canonical_edge_idx (nullable, [E]): canonical edge id of each traversal slot,
// applied to the Edge operand rows AND the output row, so the list may be grouped
// by source for locality while the output stays numbered by forward-CSR position.
// nullptr: the traversal order is already the canonical one (slot index == id).
template <GSDDMM_OP op, GSDDMM_MEMBER ll, GSDDMM_MEMBER rr, size_t D_CONST, FloatingNum cuda_t, FloatingNum accum_t = float, size_t PIPELINE_STAGES = 0>
__global__ void __launch_bounds__(kWarpSize * kGsddmmEdgeMaxWarpsPerBlock) GSDDMM_forward_edge_block ( // no-format
    uint64_t E,
    cuda_t const *__restrict__ L, cuda_t const *__restrict__ R, cuda_t *__restrict__ O,
    ulonglong2 const *__restrict__ edge_nodes_idx,
    unsigned long long const *__restrict__ canonical_edge_idx,
    uint32_t edges_per_warp
);

// =============================================================================
// GSDDMM backward
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
// Each partial then lands on the row its operand was read from, which is what
// makes the backward a *reduction* rather than a map:
//
//   - an Edge operand owns one row per edge, so its gradient is a plain store;
//   - a Src_V operand's gradient sums over the node's OUTGOING edges, i.e. the
//     rows of the backward (source-grouped) CSR;
//   - a Dst_V operand's gradient sums over its INCOMING edges, i.e. the rows of
//     the forward CSR.
//
// GSDDMM_REDUCE names which of those two node reductions a launch performs. Every
// instantiated member pair holds at most one Src_V and one Dst_V operand, so the
// passes a call needs follow from (ll, rr) alone and never write the same buffer:
// (Src_V, Dst_V) runs both passes, a pair with an Edge operand runs the single
// pass for its node operand and writes the edge gradient on the way through, and
// Copy runs the one pass for its lhs.
enum class GSDDMM_REDUCE : uint8_t {
    Dst,  // walk the forward CSR, reduce per destination node
    Src,  // walk the backward CSR, reduce per source node
};

// Compile-time operand routing for one backward pass.
template <GSDDMM_OP op, GSDDMM_MEMBER ll, GSDDMM_MEMBER rr, GSDDMM_REDUCE reduce>
struct GsddmmBackwardPlan {
    using Fwd = GsddmmPlan<op, ll, rr>;

    static constexpr GSDDMM_MEMBER SELF_MEMBER = (reduce == GSDDMM_REDUCE::Dst) ? GSDDMM_MEMBER::Dst_V : GSDDMM_MEMBER::Src_V;

    // Which operand this pass reduces. Exactly one of the two can match, because
    // the instantiated pairs never repeat a member.
    static constexpr bool SELF_IS_LHS = (ll == SELF_MEMBER);
    static constexpr bool SELF_IS_RHS = Fwd::USE_R && (rr == SELF_MEMBER);
    static constexpr bool HAS_SELF    = SELF_IS_LHS || SELF_IS_RHS;
    static_assert(!(SELF_IS_LHS && SELF_IS_RHS), "a member appears at most once per instantiated pair");

    // The other operand, gathered per edge. For Copy there is none.
    static constexpr bool HAS_OTHER      = Fwd::USE_R && HAS_SELF;
    static constexpr GSDDMM_MEMBER OTHER = SELF_IS_LHS ? rr : ll;
    static constexpr bool OTHER_IS_EDGE  = HAS_OTHER && (OTHER == GSDDMM_MEMBER::Edge);

    // Mul/Div/Dot multiply by the other operand's row; Add/Sub/Copy have constant
    // partials, so those passes are pure reductions of dO and read no operands.
    static constexpr bool NEEDS_OTHER_ROW = HAS_OTHER && (op == GSDDMM_OP::Mul || op == GSDDMM_OP::Div || op == GSDDMM_OP::Dot);

    // Div's denominator gradient is -dO * L / R^2. R is this pass's own row, so
    // the -1/R^2 factor is constant across the sum and is applied once at the
    // end instead of per edge -- the own row is read once per node, never per edge.
    static constexpr bool NEEDS_SELF_ROW = (op == GSDDMM_OP::Div) && SELF_IS_RHS;
    // Negated partials: sub's rhs, and div's rhs (through the factor above).
    static constexpr bool NEGATE_SELF = SELF_IS_RHS && (op == GSDDMM_OP::Sub || op == GSDDMM_OP::Div);

    // The edge operand's gradient is a plain per-edge store, written by this pass
    // (the only one that runs when a pair has an Edge operand).
    static constexpr bool WRITE_EDGE_GRAD = OTHER_IS_EDGE;
    // ... and for Div it needs the edge row itself (dR = -dO * L / R^2 with R
    // edge-indexed), so that row is read per edge even when SELF does not need it.
    static constexpr bool NEEDS_OTHER_ROW_FOR_EDGE_GRAD = WRITE_EDGE_GRAD && (op == GSDDMM_OP::Div) && (OTHER == rr);

    static constexpr bool READS_OTHER = NEEDS_OTHER_ROW || NEEDS_OTHER_ROW_FOR_EDGE_GRAD;
    static constexpr bool IS_DOT      = Fwd::IS_DOT;
};

// Dynamic shared memory for one GSDDMM_backward_normal launch: one fp32 feature
// row per warp, holding that warp's partial sum so warp 0 can reduce them (the
// same layout the GT backward uses). A single warp per block needs none.
template <size_t N_PER_BLOCK, size_t D_CONST, FloatingNum accum_t = float>
inline consteval size_t gsddmm_backward_shmem_bytes() {
    return N_PER_BLOCK > 1 ? N_PER_BLOCK * D_CONST * sizeof(accum_t) : 0;
}

// Node-parallel backward: one thread block per (bucketed) node of the CSR that
// `reduce` selects, warps striding over that node's edges. The node's gradient is
// accumulated in fp32 registers and written once, so this variant needs NO
// atomics and no zeroed output -- it is the accurate, load-imbalanced choice.
//
// row_ptr/col_idx are the CSR for the pass direction: the forward CSR for
// GSDDMM_REDUCE::Dst, the backward (source-grouped) CSR for ::Src. In the latter
// case the slot walked is a CSC position, so canonical_edge_idx (required there,
// unless the graph aliases its two CSRs) maps it to the forward-CSR edge id that
// numbers dO and the Edge operand.
//
// d_self is the reduced node gradient, [N, D]. d_edge is the edge operand's
// gradient, [E, D], written only when the pair has an Edge operand; pass nullptr
// otherwise.
//
// Unlike the forward there is no heavy-node chunking here: splitting a node over
// several blocks would make them all write its row, which is exactly what this
// variant exists to avoid. Use the edge-parallel backward when the degree
// distribution makes one-block-per-node the bottleneck.
template <GSDDMM_OP op, GSDDMM_MEMBER ll, GSDDMM_MEMBER rr, GSDDMM_REDUCE reduce, size_t N_PER_BLOCK, size_t D_CONST, FloatingNum cuda_t, IntegralNum index_t, FloatingNum accum_t = float>
__global__ void __launch_bounds__(N_PER_BLOCK * kWarpSize) GSDDMM_backward_normal ( // no-format
    size_t N,
    cuda_t const *__restrict__ L, cuda_t const *__restrict__ R, cuda_t const *__restrict__ dO,
    cuda_t *__restrict__ d_self, cuda_t *__restrict__ d_edge,
    index_t const *__restrict__ row_ptr, index_t const *__restrict__ col_idx,
    index_t const *__restrict__ node_indices,
    unsigned long long const *__restrict__ canonical_edge_idx
);

// Edge-parallel backward: one warp per contiguous chunk of the explicit edge
// list, mirroring GSDDMM_forward_edge_block. Perfectly load balanced, at the cost
// of accumulating the node gradient with atomics -- into an fp32 buffer
// (d_self_f32, zero-initialized by the caller, cast back on the host), because
// fp16/bf16 atomics would both contend and lose the reduction's precision.
//
// The edge list is grouped by the node being reduced, so a warp's chunk usually
// shares its target row: one warp ballot (as in the forward's shared-row cache)
// decides that, the chunk is then summed in registers, and the warp issues ONE
// atomicAdd per feature tile instead of one per edge. Mixed chunks fall back to
// per-edge atomics.
template <GSDDMM_OP op, GSDDMM_MEMBER ll, GSDDMM_MEMBER rr, GSDDMM_REDUCE reduce, size_t D_CONST, FloatingNum cuda_t, FloatingNum accum_t = float>
__global__ void __launch_bounds__(kWarpSize * kGsddmmEdgeMaxWarpsPerBlock) GSDDMM_backward_edge_block ( // no-format
    uint64_t E,
    cuda_t const *__restrict__ L, cuda_t const *__restrict__ R, cuda_t const *__restrict__ dO,
    accum_t *__restrict__ d_self_f32, cuda_t *__restrict__ d_edge,
    ulonglong2 const *__restrict__ edge_nodes_idx,
    unsigned long long const *__restrict__ canonical_edge_idx,
    uint32_t edges_per_warp
);

};  // namespace gsddmm
