#pragma once

#include <cstdint>
#include <utility>

#include "common.cuh"
#include "common/row_group.cuh"
#include "common/row_shape.cuh"

// =============================================================================
// Small kernels over whole rows of a runtime length, shared by the attention kernels and meant for
// any kernel built on common/row_shape.cuh:
//
//   row_dot_kernel               out[n, h] = <A[n, h], B[n, h]>: the backward's Delta = <dO, O>
//   softmax_merge_slices_kernel  the merge of a split-K softmax aggregation's slice partials
//   sum_merge_slices_kernel      the merge of a split-K sum's slice partials
//
// Each has a host launcher that takes care of the row layout. All launch on the stream given.
// =============================================================================
namespace turbo_gnn::rows {

// ------------------------------------------------------------------------------------------------
// Row dot product. One lane group per row; the groups of a warp take consecutive rows.
// ------------------------------------------------------------------------------------------------
inline constexpr uint32_t kRowDotThreads = 256;

template <size_t G, bool WIDE, FloatingNum cuda_t, FloatingNum accum_t = float>
__global__ void __launch_bounds__(kRowDotThreads) row_dot_kernel(
    int64_t rows, int64_t H, uint32_t nvec, cuda_t const *const __restrict__ A, int64_t stride_a_n, int64_t stride_a_h,
    cuda_t const *const __restrict__ B, int64_t stride_b_n, int64_t stride_b_h, accum_t *const __restrict__ out
) {
    static constexpr size_t VEC = WIDE ? 16 / sizeof(cuda_t) : 1;
    using RG                    = RowGroup<G, 1, VEC, cuda_t, accum_t>;
    using Tile                  = typename RG::Tile;

    const uint32_t lane         = threadIdx.x % kWarpSize;
    const uint32_t r            = lane % G;
    const int64_t warp_global   = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / kWarpSize;
    const int64_t warps_total   = static_cast<int64_t>(gridDim.x) * blockDim.x / kWarpSize;
    constexpr int64_t kRowsWarp = RG::kGroupsPerWarp;

    // Warp-uniform trip count: the group reduction shuffles across the whole warp.
    for (int64_t base = warp_global * kRowsWarp; base < rows; base += warps_total * kRowsWarp) {
        const int64_t row = base + lane / G;
        accum_t part{};
        if (row < rows) {
            const int64_t n           = row / H;
            const int64_t h           = row - n * H;
            cuda_t const *const a_row = A + n * stride_a_n + h * stride_a_h;
            cuda_t const *const b_row = B + n * stride_b_n + h * stride_b_h;
            for (uint32_t v = r; v < nvec; v += G) {
                Tile::read(a_row, v).dot_product_(&part, Tile::read(b_row, v));
            }
        }
        part = RG::reduce(part);
        if (row < rows && r == 0) {
            out[row] = part;
        }
    }
}

// out[n, h] = <a[n, h, :], b[n, h, :]> for [N, H, D] tensors with unit stride over D; out is a
// contiguous fp32 [N, H] tensor. The vector width follows a's and b's own alignment.
template <FloatingNum cuda_t>
inline void launch_row_dot(at::Tensor const& a, at::Tensor const& b, at::Tensor& out, cudaStream_t stream) {
    const int64_t rows = a.size(0) * a.size(1);
    if (rows == 0) {
        return;
    }
    const bool wide       = rows_16b_aligned(a) && rows_16b_aligned(b);
    const uint32_t nvec   = static_cast<uint32_t>(a.size(2)) / (wide ? static_cast<uint32_t>(16 / sizeof(cuda_t)) : 1u);
    const int group       = nvec <= 4 ? 4 : nvec <= 8 ? 8 : nvec <= 16 ? 16 : 32;
    const int64_t per_blk = (kRowDotThreads / kWarpSize) * (kWarpSize / group);
    const int64_t blocks  = std::min<int64_t>(ceil_div(rows, per_blk), static_cast<int64_t>(sched::sm_count()) * 32);
    std::visit(
        [&](auto group_c, auto wide_c) {
            constexpr size_t G  = decltype(group_c)::value;
            constexpr bool WIDE = decltype(wide_c)::value;
            row_dot_kernel<G, WIDE, cuda_t><<<static_cast<unsigned>(blocks), kRowDotThreads, 0, stream>>>(
                rows, a.size(1), nvec, reinterpret_cast<cuda_t const *>(a.data_ptr()), a.stride(0), a.stride(1),
                reinterpret_cast<cuda_t const *>(b.data_ptr()), b.stride(0), b.stride(1), out.data_ptr<float>()
            );
        },
        MakeIntVariant<4, 8, 16, 32>(group), MakeBoolVariant<false, true>(wide)
    );
}

// ------------------------------------------------------------------------------------------------
// Split-K merges. A split-K launch runs one block per fixed-size slice of a heavy node's edge list
// and writes an fp32 partial per slice, [num_slices, H, D]; these fold a node's slices into its
// row. Grid (num_heavy, H, vector blocks), one vector of the row per thread.
// ------------------------------------------------------------------------------------------------
inline constexpr uint32_t kMergeThreads = 128;

// The heavy nodes of a split-K launch and where each one's slices are.
template <IntegralNum index_t>
struct SliceTable {
    index_t const *heavy_nodes;    // [num_heavy]
    int const *node_chunk_offset;  // [num_heavy + 1]: node k owns slices [offset[k], offset[k + 1])
    int num_heavy;
};

// The softmax version: every slice partial is an un-normalised accumulator plus its (max, sum of
// exponentials), part_ml [num_slices, H, 2]. Runs the n-way online-softmax merge the slice kernels
// ran across warps, across slices, followed by the normalisation the slices deliberately skipped.
// A node whose slices are all empty comes out as a zero row with -inf logsumexp.
template <bool WIDE, FloatingNum cuda_t, IntegralNum index_t, FloatingNum accum_t = float>
__global__ void __launch_bounds__(kMergeThreads) softmax_merge_slices_kernel(
    SliceTable<index_t> const table, int64_t H, uint32_t D, uint32_t nvec, accum_t const *const __restrict__ part_o,
    accum_t const *const __restrict__ part_ml, cuda_t *const __restrict__ out, int64_t stride_out_n, int64_t stride_out_h,
    accum_t *const __restrict__ lse
) {
    static constexpr size_t VEC = WIDE ? 16 / sizeof(cuda_t) : 1;
    using Tile                  = TileOps<VEC, cuda_t, accum_t>;
    using AccumOps              = AdOps<accum_t>;

    const int slot     = blockIdx.x;
    const int64_t head = blockIdx.y;
    if (slot >= table.num_heavy || head >= H) [[unlikely]] {
        return;
    }
    const int64_t node = static_cast<int64_t>(table.heavy_nodes[slot]);
    const int lo       = table.node_chunk_offset[slot];
    const int hi       = table.node_chunk_offset[slot + 1];

    accum_t m = -FLT_MAX;
    for (int s = lo; s < hi; ++s) {
        m = AccumOps::max(m, part_ml[2 * (s * H + head)]);
    }
    accum_t l{};
    for (int s = lo; s < hi; ++s) {
        l = AccumOps::fma(part_ml[2 * (s * H + head) + 1], AccumOps::exp(part_ml[2 * (s * H + head)] - m), l);
    }
    if (blockIdx.z == 0 && threadIdx.x == 0) {
        lse[node * H + head] = l > accum_t{} ? m + AccumOps::log(l) : -INFINITY;
    }

    const uint32_t v = blockIdx.z * blockDim.x + threadIdx.x;
    if (v >= nvec) {
        return;
    }
    alignas(16) accum_t acc[VEC] = {};
    for (int s = lo; s < hi; ++s) {
        const accum_t c          = AccumOps::exp(part_ml[2 * (s * H + head)] - m);
        accum_t const *const row = part_o + dense_row(s, head, H, D) + static_cast<int64_t>(v) * VEC;
#pragma unroll
        for (size_t e = 0; e < VEC; ++e) {
            acc[e] = AccumOps::fma(c, row[e], acc[e]);
        }
    }
    const accum_t inv = l > accum_t{} ? accum_t{1} / l : accum_t{};
#pragma unroll
    for (size_t e = 0; e < VEC; ++e) {
        acc[e] *= inv;
    }
    Tile::write(out + node * stride_out_n + head * stride_out_h, v, Tile::convert_from_accum(acc));
}

// The sum version, for NUM partial/output pairs at once: outs[k] (contiguous [N, H, D], input
// dtype) = the sum of parts[k] over the node's slices. A node without slices gets a zero row.
template <size_t NUM, FloatingNum cuda_t, FloatingNum accum_t = float>
struct SumSlices {
    accum_t const *parts[NUM];
    cuda_t *outs[NUM];
};

template <size_t NUM, bool WIDE, FloatingNum cuda_t, IntegralNum index_t, FloatingNum accum_t = float>
__global__ void __launch_bounds__(kMergeThreads) sum_merge_slices_kernel(
    SliceTable<index_t> const table, int64_t H, uint32_t D, uint32_t nvec, SumSlices<NUM, cuda_t, accum_t> const io
) {
    static constexpr size_t VEC = WIDE ? 16 / sizeof(cuda_t) : 1;
    using Tile                  = TileOps<VEC, cuda_t, accum_t>;

    const int slot     = blockIdx.x;
    const int64_t head = blockIdx.y;
    const uint32_t v   = blockIdx.z * blockDim.x + threadIdx.x;
    if (slot >= table.num_heavy || head >= H || v >= nvec) [[unlikely]] {
        return;
    }
    const int64_t node = static_cast<int64_t>(table.heavy_nodes[slot]);
    const int lo       = table.node_chunk_offset[slot];
    const int hi       = table.node_chunk_offset[slot + 1];
#pragma unroll
    for (size_t k = 0; k < NUM; ++k) {
        alignas(16) accum_t acc[VEC] = {};
        for (int s = lo; s < hi; ++s) {
            accum_t const *const row = io.parts[k] + dense_row(s, head, H, D) + static_cast<int64_t>(v) * VEC;
#pragma unroll
            for (size_t e = 0; e < VEC; ++e) {
                acc[e] += row[e];
            }
        }
        Tile::write(io.outs[k] + dense_row(node, head, H, D), v, Tile::convert_from_accum(acc));
    }
}

// Grid of the merge kernels: one thread per row vector.
inline std::pair<dim3, uint32_t> merge_launch(int num_heavy, int64_t H, uint32_t nvec) {
    const uint32_t threads = std::min(kMergeThreads, ceil_div(nvec, static_cast<uint32_t>(kWarpSize)) * static_cast<uint32_t>(kWarpSize));
    return {dim3(num_heavy, static_cast<unsigned>(H), ceil_div(nvec, threads)), threads};
}

template <FloatingNum cuda_t, IntegralNum index_t>
inline void launch_softmax_merge(
    SliceTable<index_t> const& table, int64_t H, uint32_t D, uint32_t nvec, bool wide, float const *part_o, float const *part_ml, cuda_t *out,
    int64_t stride_out_n, int64_t stride_out_h, float *lse, cudaStream_t stream
) {
    if (table.num_heavy == 0) {
        return;
    }
    const auto [grid, threads] = merge_launch(table.num_heavy, H, nvec);
    std::visit(
        [&](auto wide_c) {
            softmax_merge_slices_kernel<decltype(wide_c)::value, cuda_t, index_t>
                <<<grid, threads, 0, stream>>>(table, H, D, nvec, part_o, part_ml, out, stride_out_n, stride_out_h, lse);
        },
        MakeBoolVariant<false, true>(wide)
    );
}

template <size_t NUM, FloatingNum cuda_t, IntegralNum index_t>
inline void launch_sum_merge(
    SliceTable<index_t> const& table, int64_t H, uint32_t D, uint32_t nvec, bool wide, SumSlices<NUM, cuda_t, float> const& io,
    cudaStream_t stream
) {
    if (table.num_heavy == 0) {
        return;
    }
    const auto [grid, threads] = merge_launch(table.num_heavy, H, nvec);
    std::visit(
        [&](auto wide_c) {
            sum_merge_slices_kernel<NUM, decltype(wide_c)::value, cuda_t, index_t><<<grid, threads, 0, stream>>>(table, H, D, nvec, io);
        },
        MakeBoolVariant<false, true>(wide)
    );
}

}  // namespace turbo_gnn::rows
