#pragma once

#include <algorithm>
#include <cstdint>

#include "common.cuh"
#include "common/row_group.cuh"

// Shared pieces of the graph transformer kernels: row shapes, kernel parameters and the bits of
// host-side planning that have to agree with the device code.
//
// The head dim D is a runtime value, anything in [1, kMaxHeadDim]. What is compiled is how a row
// of D features is laid out across lanes (see common/row_group.cuh), picked per launch:
//
//   fused    D fits one pass of a lane group, G * S * VEC features (up to 256 fp32 / 512 fp16
//            elements). Every row a node needs lives in registers and each edge is visited once.
//   chunked  anything longer. A first pass over the edges computes the per-edge scalars (score,
//            or alpha and dS in the backward) with full-row dot products and parks them in a
//            [scalars, H, E] fp32 scratch; a second pass sweeps the row in 32 * VEC-wide feature
//            chunks with register accumulators. Neither registers nor shared memory grow with D.
namespace turbo_gnn::gt {

inline constexpr int64_t kMaxHeadDim = 65536;

// ------------------------------------------------------------------------------------------------
// Row shapes
//
// A shape code names one layout: code = 1000 * WIDE + 10 * G + S, where WIDE selects 16-byte
// vectors (every row 16-byte aligned and D a whole number of them) over single elements, G is the
// lanes per row and S the vectors per lane. G == 0 marks the chunked path, which always uses whole
// warps and one vector per lane per chunk.
// ------------------------------------------------------------------------------------------------

template <int CODE>
struct Shape {
    static constexpr bool kWide    = CODE >= 1000;
    static constexpr size_t G      = (CODE % 1000) / 10;
    static constexpr size_t S      = CODE % 10;
    static constexpr bool kChunked = G == 0;
    // The cp.async ring copies whole rows in 16-byte chunks, one warp per row.
    static constexpr bool kPipelinable = kWide && G == kWarpSize;

    template <FloatingNum cuda_t>
    static constexpr size_t vec = kWide ? 16 / sizeof(cuda_t) : 1;
};

inline constexpr int shape_code(bool wide, size_t g, size_t s) { return (wide ? 1000 : 0) + static_cast<int>(10 * g + s); }

// Every shape a launcher may dispatch to: the fused group sizes with one vector per lane, the
// whole-warp shape with two (when MAX_STEPS allows it, see kMaxSteps), and the chunked path --
// each wide and narrow. G = 4 is the smallest group: below it the per-edge overhead, not the idle
// lanes, dominates a row of a few elements.
template <int MAX_STEPS>
inline auto make_shape_variant(int code) {
    if constexpr (MAX_STEPS >= 2) {
        return MakeIntVariant<1041, 1081, 1161, 1321, 1322, 1000, 41, 81, 161, 321, 322, 0>(code);
    } else {
        return MakeIntVariant<1041, 1081, 1161, 1321, 1000, 41, 81, 161, 321, 0>(code);
    }
}

// The shape for a head dim. max_steps caps S: a kernel that keeps several rows per lane in
// registers sends long rows to the chunked path sooner rather than spill.
inline int plan_shape(int64_t D, size_t elem_size, bool wide, int max_steps) {
    const int64_t nvec = D / (wide ? static_cast<int64_t>(16 / elem_size) : 1);
    for (size_t g : {size_t{4}, size_t{8}, size_t{16}, kWarpSize}) {
        if (nvec <= static_cast<int64_t>(g)) {
            return shape_code(wide, g, 1);
        }
    }
    if (max_steps >= 2 && nvec <= static_cast<int64_t>(2 * kWarpSize)) {
        return shape_code(wide, kWarpSize, 2);
    }
    return shape_code(wide, 0, 0);
}

// Whether every row of t ([N, H, D], unit stride over D) starts on a 16-byte boundary and spans a
// whole number of 16-byte vectors -- the precondition of the wide shapes.
inline bool rows_16b_aligned(at::Tensor const& t) {
    const int64_t elem = static_cast<int64_t>(t.element_size());
    if ((t.size(2) * elem) % 16 != 0 || reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 != 0) {
        return false;
    }
    return (t.size(0) <= 1 || (t.stride(0) * elem) % 16 == 0) && (t.size(1) <= 1 || (t.stride(1) * elem) % 16 == 0);
}

// Largest of the supported pipeline depths, at most `requested`, whose shared memory fits the
// device. A depth that does not fit used to fail the launch; for a head dim chosen at run time a
// shallower pipeline is the better answer than an error.
template <typename SmemFn>
inline int fit_pipeline_stages(int requested, SmemFn&& smem_for, at::Device device) {
    static constexpr int kDepths[] = {12, 6, 2};
    int max_optin                  = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&max_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device.index()));
    for (int depth : kDepths) {
        if (depth <= requested && smem_for(depth) <= static_cast<size_t>(max_optin)) {
            return depth;
        }
    }
    return 0;
}

// Runs fn.template operator()<STAGES>() for a supported pipeline depth.
template <typename Fn>
inline void with_pipeline_stages(int stages, Fn&& fn) {
    std::visit([&](auto stages_c) { fn.template operator()<decltype(stages_c)::value>(); }, MakeIntVariant<0, 2, 6, 12>(stages));
}

// ------------------------------------------------------------------------------------------------
// Launch bounds and register budget
//
// The forward's light buckets run 1-4 warps per block and are latency bound: they live on
// occupancy. Asking for 32 / W resident blocks caps them at 64 registers, which every fused forward
// shape fits without spilling. (The old cap of 32 registers would force spills now that the node's
// own row lives in registers rather than shared memory.) Where the hardware holds fewer blocks
// (sm_86/89) the request is clamped to what it holds: those blocks are then the occupancy limit,
// and asking for more would only make ptxas drop the bound. The backward keeps more rows per edge in
// flight -- four plus three accumulators when undirected -- and is left uncapped, as it always was:
// at 64 registers its two-vector shapes spill. Heavy buckets (8-32 warps) are left to the
// compiler; a 32-warp block is capped at 64 registers by its size alone, so the launchers give it
// one-vector shapes only (kMaxSteps).
// ------------------------------------------------------------------------------------------------

template <size_t W>
inline constexpr int kForwardMinBlocksPerSM = W <= 4 ? std::min(static_cast<int>(32 / W), kMaxResidentBlocksPerSM) : 1;

// S, the vectors per lane of the fused shapes, for a block of W warps.
inline constexpr int kMaxSteps(int warps) { return warps >= 32 ? 1 : 2; }

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

// Threads per block of the slice-merge kernels: one row vector per thread.
inline constexpr uint32_t kMergeThreads = 128;

// Offset of row (node, head) in a contiguous [N, H, D] tensor.
__host__ __device__ __forceinline__ int64_t dense_row(int64_t node, int64_t head, int64_t H, uint32_t D) {
    return (node * H + head) * static_cast<int64_t>(D);
}

}  // namespace turbo_gnn::gt
