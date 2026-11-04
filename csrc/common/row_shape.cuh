#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>

#include "common.cuh"

// =============================================================================
// Row shapes: how a gather kernel lays a row of D features across lanes, for D known only at run
// time. Device-side building blocks are in common/row_group.cuh; this header holds the shapes and
// the host-side planning that has to agree with the device code. Used by the graph transformer and
// GATv2 kernels, and meant for any kernel that gathers whole feature rows per edge.
//
// What is compiled is a small set of layouts, one picked per launch:
//
//   fused    D fits one pass of a lane group, G * S * VEC features (up to 256 fp32 / 512 fp16
//            elements). Every row a node needs lives in registers and each edge is visited once.
//   chunked  anything longer. A kernel sweeps the row in 32 * VEC-wide feature chunks with register
//            accumulators, so neither registers nor shared memory grow with D. Kernels whose
//            per-edge work needs the whole row (a softmax score, a dot product) first compute those
//            scalars in a pass of their own and park them in an fp32 scratch.
//
// Adopting this for a kernel family (gt/ and gatv2/ are the worked examples):
//
//   host    check_head_dim; rows_16b_aligned over every tensor read in vectors, to pick the wide
//           or narrow layout; with_shape<kMaxSteps(W)> (or a tighter cap for kernels with more
//           rows per lane) to dispatch the shape; light_warps_for for light buckets;
//           with_fitted_stages for kernels with a cp.async ring; MakeIndexWidthVariant for the
//           index type; check_node_list for bucket lists.
//   device  RowGroup (common/row_group.cuh) for loads, dot products, reductions and stores of a
//           lane's share of a row; for_each_neighbor / NeighborCache to walk a CSR row with batched
//           id loads; team_sum, merge_groups_softmax and merge_block_softmax to combine partials;
//           chunk_teams for the chunked sweep; pipelined_rows_loop (common/pipeline.cuh) for the
//           ring.
//   kernels row_dot_kernel and the split-K slice merges (common/row_kernels.cuh).
//
// Elementwise per-edge ops need no scratch on the chunked path, every chunk being independent; a
// per-edge reduction over D needs one only when an edge's row is split across warps. Such kernels
// can skip the chunked path altogether and sweep a long row in passes of a fused shape instead (see
// "Sweep shapes" below; reduction/ is the worked example).
// =============================================================================
namespace turbo_gnn::rows {

// Largest supported row length. Bounded by the 32-bit vector indices of RowGroup, and kept at a
// round size the tests exercise.
inline constexpr int64_t kMaxHeadDim = 65536;

// ------------------------------------------------------------------------------------------------
// Shapes
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

inline constexpr bool is_chunked(int code) { return code % 1000 == 0; }

// Every shape a launcher may dispatch to: the fused group sizes with one vector per lane, the
// whole-warp shape with two (when MAX_STEPS allows it), and the chunked path -- each wide and
// narrow. G = 4 is the smallest group: below it the per-edge overhead, not the idle lanes,
// dominates a row of a few elements.
template <int MAX_STEPS>
inline auto make_shape_variant(int code) {
    static_assert(MAX_STEPS == 1 || MAX_STEPS == 2, "Shapes hold one or two vectors per lane");
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

// Default cap on S for a block of W warps. A 32-warp block is limited to 64 registers per thread
// by its size alone, which a two-vector fused shape does not fit; smaller blocks take two.
inline constexpr int kMaxSteps(int warps) { return warps >= 32 ? 1 : 2; }

// Runs fn(shape_c) for the shape of a head dim, shape_c an integral_constant; MAX_STEPS as in
// plan_shape and make_shape_variant.
template <int MAX_STEPS, typename Fn>
inline void with_shape(int64_t D, size_t elem_size, bool wide, Fn&& fn) {
    std::visit(std::forward<Fn>(fn), make_shape_variant<MAX_STEPS>(plan_shape(D, elem_size, wide, MAX_STEPS)));
}

// ------------------------------------------------------------------------------------------------
// Sweep shapes
//
// For kernels whose per-edge work is elementwise over the row -- a min, max or sum of neighbour
// rows -- every span of a row is independent of the others. A row longer than one pass of a group
// then needs no chunked path: it is swept in passes of a whole-warp shape, each pass a work item of
// its own. One vector a lane up to 256 vectors (8 passes), two beyond: measured on the reduction
// kernels, the narrower passes are 2-11% faster at D = 256-1024 (fp32) or 512-2048 (fp16) -- twice
// the items, at the occupancy of a 32-register kernel -- and 1-3% slower at 4096, where the ids
// re-read per pass begin to count.
//
// Short rows go the other way: with nothing to combine across a row's edges but the reduction
// itself, a lane group can be a worker of its own, one row per group and kWarpSize / G rows per
// warp. The group is then sized to the row alone -- down to one lane -- since it no longer has to
// share the warp's edges out.
// ------------------------------------------------------------------------------------------------

inline int plan_sweep_shape(int64_t D, size_t elem_size, bool wide) {
    const int64_t nvec = D / (wide ? static_cast<int64_t>(16 / elem_size) : 1);
    for (size_t g = 1; g <= kWarpSize; g <<= 1) {
        if (nvec <= static_cast<int64_t>(g)) {
            return shape_code(wide, g, 1);
        }
    }
    return shape_code(wide, kWarpSize, nvec <= 8 * static_cast<int64_t>(kWarpSize) ? 1 : 2);
}

inline auto make_sweep_shape_variant(int code) {
    return MakeIntVariant<1011, 1021, 1041, 1081, 1161, 1321, 1322, 11, 21, 41, 81, 161, 321, 322>(code);
}

// Runs fn(shape_c) for the sweep shape of a head dim, shape_c an integral_constant.
template <typename Fn>
inline void with_sweep_shape(int64_t D, size_t elem_size, bool wide, Fn&& fn) {
    std::visit(std::forward<Fn>(fn), make_sweep_shape_variant(plan_sweep_shape(D, elem_size, wide)));
}

// Vectors one pass of a shape covers, and the passes a row of nvec vectors takes.
template <int SHAPE>
inline constexpr uint32_t kPassVecs = static_cast<uint32_t>(Shape<SHAPE>::G * Shape<SHAPE>::S);

inline uint32_t sweep_passes(int code, uint32_t nvec) {
    const uint32_t span = static_cast<uint32_t>(((code % 1000) / 10) * (code % 10));
    return std::max(ceil_div(nvec, span), 1u);
}

// ------------------------------------------------------------------------------------------------
// Launch bounds
//
// Light buckets run 1-4 warps per block and are latency bound: they live on occupancy. Asking for
// 32 / W resident blocks caps them at 64 registers. Where the hardware holds fewer blocks
// (sm_86/89) the request is clamped to what it holds: those blocks are then the occupancy limit,
// and asking for more would only make ptxas drop the bound. Wider blocks are left to the compiler.
// ------------------------------------------------------------------------------------------------

template <size_t W>
inline constexpr int kLightMinBlocksPerSM = W <= 4 ? std::min(static_cast<int>(32 / W), kMaxResidentBlocksPerSM) : 1;

// ------------------------------------------------------------------------------------------------
// Host-side planning
// ------------------------------------------------------------------------------------------------

inline void check_head_dim(int64_t D, char const *what) {
    TORCH_CHECK(D >= 1 && D <= kMaxHeadDim, what, ": head dim D=", D, " is out of range; supported: 1..", kMaxHeadDim);
}

// Unit stride over the head dim of an [N, H, D] tensor; a size-1 dim may carry any stride.
inline bool features_contiguous(at::Tensor const& t) { return t.size(2) == 1 || t.stride(2) == 1; }

// Whether every row of t (rows along the last dim, unit stride over it: [N, H, D] or [N, D])
// starts on a 16-byte boundary and spans a whole number of 16-byte vectors -- the precondition of
// the wide shapes.
inline bool rows_16b_aligned(at::Tensor const& t) {
    const int64_t elem = static_cast<int64_t>(t.element_size());
    if ((t.size(-1) * elem) % 16 != 0 || reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 != 0) {
        return false;
    }
    for (int64_t d = 0; d + 1 < t.dim(); ++d) {
        if (t.size(d) > 1 && (t.stride(d) * elem) % 16 != 0) {
            return false;
        }
    }
    return true;
}

// Offset of row (node, head) in a contiguous [N, H, D] tensor.
__host__ __device__ __forceinline__ int64_t dense_row(int64_t node, int64_t head, int64_t H, uint32_t D) {
    return (node * H + head) * static_cast<int64_t>(D);
}

// Warps per block for a light bucket. Rows too long for one pass are swept chunk by chunk, and a
// chunked block's warps split the chunks as well as the edges. Huge head dims leave memory for few
// nodes, so one node per block of one or two warps then leaves most of the device idle -- measured
// on the graph transformer at D = 65536: 1.8x (fp16) to 1.9x (fp32) faster with 4 warps than with 1.
// For chunked rows the requested count is raised, to at most the light bucket's 4 and to no more
// warps than chunks, until the launch holds about 32 warps per SM. Fused rows keep the requested
// count. max_steps: the kernel's cap on S (see plan_shape).
inline int light_warps_for(int requested, int64_t D, size_t elem_size, bool wide, int64_t blocks, int max_steps) {
    if (!is_chunked(plan_shape(D, elem_size, wide, max_steps))) {
        return requested;
    }
    const int64_t nvec    = D / (wide ? static_cast<int64_t>(16 / elem_size) : 1);
    const int64_t nchunks = ceil_div(nvec, static_cast<int64_t>(kWarpSize));
    const int64_t target  = int64_t{32} * sched::sm_count();
    int warps             = requested;
    while (warps < 4 && warps < nchunks && blocks * warps < target) {
        warps <<= 1;
    }
    return warps;
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

// Runs launch.template operator()<STAGES>() with the deepest pipeline the shape supports and the
// device's shared memory fits, no deeper than `requested`; STAGES == 0 for shapes the ring does
// not support. smem_for.template operator()<STAGES>() is the launch's shared memory at a depth.
template <int SHAPE, typename SmemFn, typename LaunchFn>
inline void with_fitted_stages(int requested, at::Device device, SmemFn&& smem_for, LaunchFn&& launch) {
    if constexpr (Shape<SHAPE>::kPipelinable) {
        const int fitted = fit_pipeline_stages(
            requested,
            [&](int depth) {
                size_t bytes = 0;
                with_pipeline_stages(depth, [&]<int S>() { bytes = smem_for.template operator()<S>(); });
                return bytes;
            },
            device
        );
        with_pipeline_stages(fitted, launch);
    } else {
        launch.template operator()<0>();
    }
}

// A CSR with uint32 indices is read through int32 kernels (see MakeIndexWidthVariant), which is
// exact while it has fewer than 2^31 nodes and edges: column ids are below N and row offsets at
// most E.
inline void check_index_range(at::Tensor const& row_ptr, at::Tensor const& col_idx, char const *what) {
    TORCH_CHECK(
        row_ptr.scalar_type() != c10::ScalarType::UInt32 ||
            (row_ptr.numel() - 1 <= std::numeric_limits<int32_t>::max() && col_idx.numel() <= std::numeric_limits<int32_t>::max()),
        what, ": uint32 indices support graphs of fewer than 2^31 nodes and edges; use int64 indices"
    );
}

// Node lists are read through the CSR's index type (see MakeIndexWidthVariant), so they must have
// its width. A list of another width would be reinterpreted, not converted.
inline void check_node_list(at::Tensor const& nodes, at::Tensor const& row_ptr, char const *what) {
    TORCH_CHECK(nodes.numel() == 0 || nodes.is_cuda(), what, " must be a CUDA tensor");
    TORCH_CHECK(nodes.numel() == 0 || nodes.is_contiguous(), what, " must be contiguous");
    TORCH_CHECK(
        nodes.numel() == 0 || nodes.element_size() == row_ptr.element_size(), what, " must have the width of the CSR index type (",
        row_ptr.scalar_type(), "), got ", nodes.scalar_type()
    );
}

}  // namespace turbo_gnn::rows
