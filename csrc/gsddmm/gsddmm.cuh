#pragma once

#include "common.cuh"

namespace gsddmm {

// Operation to do over
enum class GSDDMM_OP: uint8_t {
    Add,   // Add the right operand to the left(elementwise)
    Sub,   // Subtract the right operand from the left(elementwise)
    Mul,   // Multiply the right operand by the left(elementwise)
    Div,   // Divide the left operand by the right(elementwise)
    Dot,   // Get a dot-product of left and right operands(multiply elementsie and then reduce by summation). The feature axis is reduced
    Copy,  // Copy left operand feature to the corresponding edges
};

// Member to take feature values from. Maybe Source vertex, destination vertex or edge
enum class GSDDMM_MEMBER: uint8_t {
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
template <GSDDMM_OP op, GSDDMM_MEMBER ll, GSDDMM_MEMBER rr, size_t N_PER_BLOCK, size_t D_CONST, FloatingNum cuda_t, int PIPELINE_STAGES>
inline consteval size_t gsddmm_forward_shmem_bytes() {
    using Plan = GsddmmPlan<op, ll, rr>;
    return Plan::NUM_DST_ROWS * D_CONST * sizeof(cuda_t) +
           (PIPELINE_STAGES > 0 ? N_PER_BLOCK * Plan::NUM_EDGE_ROWS * (PIPELINE_STAGES + 1) * D_CONST * sizeof(cuda_t) : 0);
}

// forward kernel: one thread block per (bucketed) CSR row node; each warp of the
// block owns one edge at a time, lanes split the D_CONST features into vector tiles
template <GSDDMM_OP op, GSDDMM_MEMBER ll, GSDDMM_MEMBER rr, size_t N_PER_BLOCK, size_t D_CONST, FloatingNum cuda_t, typename index_t, FloatingNum accum_t = float, int PIPELINE_STAGES = 0>
__global__ void __launch_bounds__(N_PER_BLOCK * kWarpSize) GSDDMM_forward_normal ( // no-format
    size_t N,
    cuda_t const *__restrict__ L, cuda_t const *__restrict__ R, cuda_t *__restrict__ O,
    index_t const *__restrict__ row_ptr, index_t const *__restrict__ col_idx,
    index_t const *__restrict__ node_indices  // node indirection: node_i = node_indices[blockIdx.x]
);

template <GSDDMM_OP op, GSDDMM_MEMBER ll, GSDDMM_MEMBER rr, size_t N_PER_BLOCK, size_t D_CONST, FloatingNum cuda_t, typename index_t, FloatingNum accum_t = float, int PIPELINE_STAGES = 0>
__global__ void __launch_bounds__(N_PER_BLOCK * kWarpSize) GSDDMM_forward_edge_block ( // no-format
    size_t N,
    cuda_t const *__restrict__ L, cuda_t const *__restrict__ R, cuda_t *__restrict__ O,
    index_t const *__restrict__ row_ptr, index_t const *__restrict__ col_idx,
    index_t const *__restrict__ node_indices  // node indirection: node_i = node_indices[blockIdx.x]
);

};  // namespace gsddmm
