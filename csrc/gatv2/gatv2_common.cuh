#pragma once

#include <cstdint>

#include "common.cuh"
#include "common/row_group.cuh"
#include "common/row_shape.cuh"

// Shared pieces of the GATv2 kernels: kernel parameters, the backward's modes and the per-element
// edge math.
//
// For an edge j -> i (i the destination, the "left" node; j the source, the "right" one) and head h:
//
//   s(i, j) = l[i] + r[j]              z = LeakyReLU(s)            e(i, j) = <a, z>
//   alpha(i, j) = softmax_j e(i, j)    out[i] = sum_j alpha(i, j) * r[j]
//
// The head dim D is a runtime value, anything in [1, rows::kMaxHeadDim]; the row layout -- fused in
// registers or chunked -- is picked per launch from the shapes of common/row_shape.cuh.
//
// The elementwise part of the score and of the gradients is computed in fp32 from rows widened on
// load: only the inputs and outputs carry the storage precision. Before, l + r, the LeakyReLU, the
// product with `a` and the edge gradient were rounded to fp16/bf16 element by element. Against a
// float64 reference (uniform graph, D = 128) the output's error went from 4.0e-3 to 2.0e-3 of its
// largest entry in bf16 and from 4.4e-4 to 2.5e-4 in fp16, with dr and da improving alike; dl, which
// now takes Delta from the stored output, stays at about two bf16 ulps (5.5e-3 -> 6.0e-3).
//
// LeakyReLU'(0) is the negative slope, as in PyTorch. The previous kernels' fp32 path -- also taken
// by fp16/bf16 at D = 32 -- used 1 there, which for fp16 inputs, where l + r == 0 exactly is common,
// put errors of up to 60% of the largest entry into dl and 15% into dr.
namespace turbo_gnn::gatv2 {

using rows::dense_row;
using rows::Shape;

// ------------------------------------------------------------------------------------------------
// Edge math on a lane's share of the rows, widened to accum_t (RowGroup::to_accum). Vectors past
// the row's end are zero in every operand and drop out of all of it.
// ------------------------------------------------------------------------------------------------

// The lane's part of e = <a, LeakyReLU(l + r)>.
template <size_t E, FloatingNum accum_t>
__device__ __forceinline__ accum_t score_part(accum_t const (&l)[E], accum_t const (&r)[E], accum_t const (&a)[E], accum_t ns) {
    accum_t acc{};
#pragma unroll
    for (size_t e = 0; e < E; ++e) {
        const accum_t s = l[e] + r[e];
        acc             = AdOps<accum_t>::fma(a[e], s > accum_t{} ? s : ns * s, acc);
    }
    return acc;
}

// The lane's part of <x, y>.
template <size_t E, FloatingNum accum_t>
__device__ __forceinline__ accum_t dot_part(accum_t const (&x)[E], accum_t const (&y)[E]) {
    accum_t acc{};
#pragma unroll
    for (size_t e = 0; e < E; ++e) {
        acc = AdOps<accum_t>::fma(x[e], y[e], acc);
    }
    return acc;
}

// With de = dL/de(i, j), the edge's share of the gradients of l[i] and of a:
//   gl += de * a * LeakyReLU'(s)          ga += de * LeakyReLU(s) = de * LeakyReLU'(s) * s
// LeakyReLU'(s) is 1 for s > 0 and ns otherwise, as in PyTorch.
template <size_t E, FloatingNum accum_t>
__device__ __forceinline__ void accum_left(
    accum_t (&gl)[E], accum_t (&ga)[E], accum_t de, accum_t const (&l)[E], accum_t const (&r)[E], accum_t const (&a)[E], accum_t ns
) {
    const accum_t de_ns = de * ns;
#pragma unroll
    for (size_t e = 0; e < E; ++e) {
        const accum_t s = l[e] + r[e];
        const accum_t t = s > accum_t{} ? de : de_ns;
        gl[e]           = AdOps<accum_t>::fma(t, a[e], gl[e]);
        ga[e]           = AdOps<accum_t>::fma(t, s, ga[e]);
    }
}

// The edge's share of the gradient of r[j], from the aggregation and from the score:
//   gr += alpha * g[i] + de * a * LeakyReLU'(s)
template <size_t E, FloatingNum accum_t>
__device__ __forceinline__ void accum_right(
    accum_t (&gr)[E], accum_t alpha, accum_t const (&g)[E], accum_t de, accum_t const (&l)[E], accum_t const (&r)[E], accum_t const (&a)[E],
    accum_t ns
) {
    const accum_t de_ns = de * ns;
#pragma unroll
    for (size_t e = 0; e < E; ++e) {
        const accum_t s = l[e] + r[e];
        const accum_t t = s > accum_t{} ? de : de_ns;
        gr[e]           = AdOps<accum_t>::fma(alpha, g[e], AdOps<accum_t>::fma(t, a[e], gr[e]));
    }
}

// ------------------------------------------------------------------------------------------------
// Launch bounds and register budget. The forward's light buckets live on occupancy and are capped
// at 64 registers (see rows::kLightMinBlocksPerSM) -- except where a lane holds ACCS = 16 elements
// of l, a and the accumulator (two 16-byte fp16 vectors): those three alone take 48 registers and
// spilled 88 bytes at 64, so they get two thirds of the blocks, ~96 registers. The backward keeps
// more rows per edge in flight and is left to the compiler.
// ------------------------------------------------------------------------------------------------

template <size_t W, size_t ACCS>
inline constexpr int kForwardMinBlocksPerSM = ACCS > 8 ? std::max(1, rows::kLightMinBlocksPerSM<W> * 2 / 3) : rows::kLightMinBlocksPerSM<W>;

// ------------------------------------------------------------------------------------------------
// Kernel parameters. Passed by value: one struct instead of thirty arguments, shared by the
// in-place, split-K and chunked variants of a kernel, each reading the fields it needs.
// ------------------------------------------------------------------------------------------------

template <FloatingNum cuda_t, IntegralNum index_t, FloatingNum accum_t>
struct ForwardParams {
    int64_t N;
    int64_t H;
    uint32_t D;     // head dim
    uint32_t nvec;  // head dim in vectors of the dispatched width

    cuda_t const *L;  // [N, H, D], unit stride over D: the destination side
    cuda_t const *R;  // [N, H, D], unit stride over D: the source side, also the aggregated rows
    int64_t stride_l_n, stride_l_h;
    int64_t stride_r_n, stride_r_h;
    cuda_t const *A;  // [H, D], contiguous
    accum_t negative_slope;

    index_t const *row_ptr;  // forward CSR: rows are destinations
    index_t const *col_idx;
    int64_t num_edges;  // col_idx length: the row stride of the chunked path's score scratch

    // In-place outputs.
    cuda_t *O;           // contiguous [N, H, D]
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

// What a backward launch computes. Every launch runs one block per row node d of the CSR it walks,
// d's neighbours s split into per-warp shares; per edge it recomputes alpha from the saved
// logsumexp and de = alpha * (<g, r> - Delta) with Delta[i] = <g[i], out[i]>, g the gradient of
// the output.
//
//   AL  the forward CSR (d = i, s = j): gradients of l[d] and of a. Gathers r[s].
//   R   the CSR^T (d = j, s = i): the gradient of r[d]. Gathers l[s] and g[s]. On an undirected
//       graph the forward CSR is its own transpose and serves both.
//
// The two are independent -- disjoint outputs, inputs only -- and run concurrently. They replace a
// fused kernel for undirected graphs that gathered all three rows per edge in one pass: the same
// traffic and arithmetic, but at 128 registers (fp16) against 72-79, and with one vector per lane
// where AL and R take two. Measured 1.05-1.45x slower than the overlapped pair on every undirected
// case, sliced or not, and 3.5x at fp32 D = 256, which its register budget sent to the chunked path.
enum class BackwardMode : int { AL = 0, R = 1 };

// Gradient accumulators of a row, by mode: dl and da for AL, dr for R.
template <BackwardMode MODE>
inline constexpr size_t kBackwardGrads = MODE == BackwardMode::AL ? 2 : 1;

// Per-edge scalars the chunked path stores between its passes: de for AL; alpha and de for R.
template <BackwardMode MODE>
inline constexpr size_t kBackwardScalars = MODE == BackwardMode::AL ? 1 : 2;

// Rows of a [kGradASlots, H, D] fp32 buffer the backward blocks add their gradient of `a` into,
// block b into slot b % kGradASlots: the sum over all edges is one row per head, and fanning it out
// keeps the atomics of concurrently finishing blocks off each other's addresses. A final reduction
// over the slots gives the gradient. Replaces an [N, H, D] fp32 buffer of per-node partials.
inline constexpr uint32_t kGradASlots = 128;

template <FloatingNum cuda_t, IntegralNum index_t, FloatingNum accum_t>
struct BackwardParams {
    int64_t N;
    int64_t H;
    uint32_t D;
    uint32_t nvec;

    cuda_t const *L;
    cuda_t const *R;
    cuda_t const *G;  // gradient of the output, [N, H, D], unit stride over D
    int64_t stride_l_n, stride_l_h;
    int64_t stride_r_n, stride_r_h;
    int64_t stride_g_n, stride_g_h;
    cuda_t const *A;
    accum_t negative_slope;

    accum_t const *logsumexp;  // [N, H]
    accum_t const *delta;      // [N, H]: <g, out> per row

    index_t const *row_ptr;  // the CSR the launch walks (see BackwardMode)
    index_t const *col_idx;
    int64_t num_edges;

    // In-place outputs.
    cuda_t *dL;         // contiguous [N, H, D]
    cuda_t *dR;         // contiguous [N, H, D]
    accum_t *dA_slots;  // [kGradASlots, H, D], zeroed

    // Split-K heavy path.
    index_t const *heavy_nodes;
    int const *chunk_node;
    int const *chunk_start;
    int slice_size;
    int num_slices;
    accum_t *part;  // [num_slices, H, D]: the partial row gradient, dl (AL) or dr (R)

    // Chunked path: per-edge scalars, [kBackwardScalars, H, num_edges].
    accum_t *edge_scalars;
};

}  // namespace turbo_gnn::gatv2
