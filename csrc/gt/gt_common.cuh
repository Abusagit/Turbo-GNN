#pragma once

#include <cstdint>

#include "common.cuh"
#include "common/row_group.cuh"
#include "common/row_shape.cuh"

// Shared pieces of the graph transformer kernels: kernel parameters and the backward's modes.
//
// The head dim D is a runtime value, anything in [1, kMaxHeadDim]. How a row of D features is laid
// out across lanes -- fused in registers, or chunked -- is picked per launch from the shapes of
// common/row_shape.cuh; the chunked path parks its per-edge scalars (score, or alpha and dS in the
// backward) in a [scalars, H, E] fp32 scratch between its two passes.
namespace turbo_gnn::gt {

using rows::dense_row;
using rows::kMaxHeadDim;
using rows::Shape;

// The forward's light buckets live on occupancy; see rows::kLightMinBlocksPerSM. The backward keeps
// more rows per edge in flight -- four plus three accumulators when undirected -- and is left
// uncapped: at 64 registers its two-vector shapes spill.
template <size_t W>
inline constexpr int kForwardMinBlocksPerSM = rows::kLightMinBlocksPerSM<W>;

// ------------------------------------------------------------------------------------------------
// Kernel parameters. Passed by value: one struct instead of thirty arguments, and the in-place,
// split-K and chunked variants of a kernel share it, each reading the fields it needs.
// ------------------------------------------------------------------------------------------------

template <FloatingNum cuda_t, IntegralNum index_t, FloatingNum accum_t>
struct ForwardParams {
    int64_t N;
    int64_t H;
    uint32_t D;     // head dim
    uint32_t nvec;  // head dim in vectors of the dispatched width

    cuda_t const *Q;
    cuda_t const *K;
    cuda_t const *V;
    int64_t stride_q_n, stride_q_h;
    int64_t stride_k_n, stride_k_h;
    int64_t stride_v_n, stride_v_h;

    index_t const *row_ptr;
    index_t const *col_idx;
    int64_t num_edges;  // col_idx length: the row stride of the chunked path's score scratch
    accum_t scale;

    // In-place output.
    cuda_t *O;
    int64_t stride_o_n, stride_o_h;
    accum_t *logsumexp;  // [N, H]

    // Split-K heavy path: one block per slice of a heavy node's edge list.
    index_t const *heavy_nodes;
    int const *chunk_node;
    int const *chunk_start;
    int slice_size;
    int num_slices;
    accum_t *part_o;   // [num_slices, H, D], un-normalised
    accum_t *part_ml;  // [num_slices, H, 2]: slice max and sum of exponentials

    // Chunked path: per-edge scores, [H, num_edges].
    accum_t *scores;
};

// Which backward a launch computes. Both walk one CSR, one block per row node d with neighbours s.
//
//   Directed    the CSR^T: s ranges over the nodes that d attends *into* (d is a source of s).
//               Accumulates dQ[d] and dV[d] and scatters dK[s] with fp32 atomics.
//   Undirected  the forward CSR, which is its own transpose: each edge is used in both directions,
//               so dK[d] comes from d's own row and no atomics are needed.
enum class BackwardMode : int { Directed = 0, Undirected = 1 };

template <FloatingNum cuda_t, IntegralNum index_t, FloatingNum accum_t>
struct BackwardParams {
    int64_t N;
    int64_t H;
    uint32_t D;
    uint32_t nvec;

    cuda_t const *Q;
    cuda_t const *K;
    cuda_t const *V;
    cuda_t const *dO;  // contiguous [N, H, D]
    int64_t stride_q_n, stride_q_h;
    int64_t stride_k_n, stride_k_h;
    int64_t stride_v_n, stride_v_h;

    accum_t const *logsumexp;  // [N, H]
    accum_t const *delta;      // [N, H]: <dO, O> per row

    index_t const *row_ptr;  // the CSR the launch walks (see BackwardMode)
    index_t const *col_idx;
    int64_t num_edges;
    accum_t scale;

    // In-place outputs, contiguous [N, H, D].
    cuda_t *dQ;
    cuda_t *dV;
    cuda_t *dK;          // Undirected
    accum_t *dK_atomic;  // Directed: fp32 scatter target

    // Split-K heavy path.
    index_t const *heavy_nodes;
    int const *chunk_node;
    int const *chunk_start;
    int slice_size;
    int num_slices;
    accum_t *part_q;  // [num_slices, H, D]
    accum_t *part_v;
    accum_t *part_k;  // Undirected

    // Chunked path: per-edge scalars, [scalars, H, num_edges].
    accum_t *edge_scalars;
};

}  // namespace turbo_gnn::gt
