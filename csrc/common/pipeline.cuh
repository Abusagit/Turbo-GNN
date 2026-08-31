#pragma once

#include <cuda/pipeline>

#include "common/misc.cuh"
#include "common/traits.cuh"

// =============================================================================
// cp.async prefetch primitives for neighbor loops (cuda::pipeline, thread scope).
//
// async_copy_row_warp   -- warp copies one D-wide row, 16B chunk per lane.
// async_copy_slice_thread -- one thread copies one <=16B slice.
// pipelined_neighbor_row_loop -- warp-strided loop with NUM_STAGES-deep
//   prefetch of NUM_ROWS rows per neighbor. min_aggr has per-thread, not
//   per-warp, parallelism and uses async_copy_slice_thread directly.
// =============================================================================

// Lane i copies 16B chunks i, i+32, ... of one ROW_ELEMS-wide row
template <int ROW_ELEMS, FloatingNum cuda_t>
__device__ __forceinline__ void async_copy_row_warp(
    cuda_t *dst, const cuda_t *src, cuda::pipeline<cuda::thread_scope_thread> &pipe, int lane
) {
    constexpr int ROW_BYTES = ROW_ELEMS * static_cast<int>(sizeof(cuda_t));
    static_assert(ROW_BYTES % 16 == 0, "Row width must be a multiple of 16 bytes for aligned async copies");
    constexpr int F4_PER_ROW = ROW_BYTES / 16;

#pragma unroll
    for (int i = lane; i < F4_PER_ROW; i += kWarpSize) {
        cuda::memcpy_async(
            reinterpret_cast<char *>(dst) + i * 16, reinterpret_cast<const char *>(src) + i * 16, cuda::aligned_size_t<16>(16), pipe
        );
    }
}

// Single-thread copy of one <=16B slice.
template <int ELEMS, FloatingNum cuda_t>
__device__ __forceinline__ void async_copy_slice_thread(cuda_t *dst, const cuda_t *src, cuda::pipeline<cuda::thread_scope_thread> &pipe) {
    constexpr int SLICE_BYTES = ELEMS * static_cast<int>(sizeof(cuda_t));
    static_assert(SLICE_BYTES <= 16, "async_copy_slice_thread is for small (<=16B) per-thread slices; use async_copy_row_warp for full rows");
    static_assert(16 % SLICE_BYTES == 0, "Slice width must evenly divide 16 bytes");

    cuda::memcpy_async(dst, src, cuda::aligned_size_t<SLICE_BYTES>(SLICE_BYTES), pipe);
}

// Visits neighbor slots k = warp_id + it*WARPS_PER_BLOCK for it in
// [0, ceil((num_neighbors - warp_id) / WARPS_PER_BLOCK)). WARPS_PER_BLOCK=1
// gives a plain sequential loop.
//
// consume(neighbor_idx, rows): rows[r] is source r's prefetched row; valid
// only inside the call (the slot is recycled on return).
//
// dbuf: this warp's private scratch, NUM_ROWS * NUM_STAGES * D_CONST elements.
template <int WARPS_PER_BLOCK, int D_CONST, int NUM_STAGES, int NUM_ROWS, FloatingNum cuda_t, typename index_t, typename ConsumeFn>
__device__ __forceinline__ void pipelined_neighbor_row_loop(
    int warp_id, int lane, int num_neighbors, index_t edge_start, index_t const *__restrict__ col_idx,
    cuda_t const *const (&row_bases)[NUM_ROWS], int64_t const (&stride_n)[NUM_ROWS], int64_t const (&stride_h)[NUM_ROWS], int head_h,
    cuda_t *dbuf, ConsumeFn &&consume
) {
    const int loop_iters = (num_neighbors > warp_id) ? (num_neighbors - warp_id + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK : 0;
    if (loop_iters == 0) return;

    cuda_t *rows[NUM_ROWS][NUM_STAGES];
#pragma unroll
    for (int r = 0; r < NUM_ROWS; ++r) {
#pragma unroll
        for (int s = 0; s < NUM_STAGES; ++s) {
            rows[r][s] = dbuf + (r * NUM_STAGES + s) * D_CONST;
        }
    }

    index_t neighbor_idx_buf[NUM_STAGES];

    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

    auto prefetch = [&](int it) {
        pipe.producer_acquire();
        if (it < loop_iters) {
            const int k                       = warp_id + it * WARPS_PER_BLOCK;
            const index_t nb                  = col_idx[edge_start + static_cast<index_t>(k)];
            neighbor_idx_buf[it % NUM_STAGES] = nb;
#pragma unroll
            for (int r = 0; r < NUM_ROWS; ++r) {
                const cuda_t *src = row_bases[r] + nb * stride_n[r] + head_h * stride_h[r];
                async_copy_row_warp<D_CONST, cuda_t>(rows[r][it % NUM_STAGES], src, pipe, lane);
            }
        }
        pipe.producer_commit();
    };

#pragma unroll
    for (int s = 0; s < NUM_STAGES; ++s) {
        prefetch(s);
    }

    for (int iter = 0; iter < loop_iters; ++iter) {
        cuda::pipeline_consumer_wait_prior<NUM_STAGES - 1>(pipe);
        // Thread-scope wait covers only this lane's own cp.async; a lane may read
        // chunks copied by other lanes (tile < 16B, or row < 512B leaves lanes idle).
        __syncwarp();

        const int slot = iter % NUM_STAGES;
        cuda_t const *cur_rows[NUM_ROWS];
#pragma unroll
        for (int r = 0; r < NUM_ROWS; ++r) {
            cur_rows[r] = rows[r][slot];
        }

        consume(neighbor_idx_buf[slot], cur_rows);

        // All lanes must finish reading the slot before prefetch() reuses it.
        __syncwarp();
        pipe.consumer_release();
        prefetch(iter + NUM_STAGES);
    }
}
