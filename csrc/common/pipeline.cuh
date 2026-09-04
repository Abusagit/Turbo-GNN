#pragma once

#include <cuda/pipeline>

#include "common/misc.cuh"
#include "common/tile.cuh"
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
template <size_t ROW_ELEMS, FloatingNum cuda_t>
__device__ __forceinline__ void async_copy_row_warp(
    cuda_t *dst, const cuda_t *src, cuda::pipeline<cuda::thread_scope_thread>& pipe, size_t lane
) {
    constexpr size_t ROW_BYTES = ROW_ELEMS * sizeof(cuda_t);
    static_assert(ROW_BYTES % 16 == 0, "Row width must be a multiple of 16 bytes for aligned async copies");
    constexpr size_t CHUNK_ELEMS = 16 / sizeof(cuda_t);
    using chunk_t                = Vec<CHUNK_ELEMS, cuda_t>;
    constexpr size_t F4_PER_ROW  = ROW_BYTES / 16;

    chunk_t *dst_v       = reinterpret_cast<chunk_t *>(dst);
    chunk_t const *src_v = reinterpret_cast<chunk_t const *>(src);

#pragma unroll
    for (size_t i = lane; i < F4_PER_ROW; i += kWarpSize) {
        cuda::memcpy_async(dst_v + i, src_v + i, cuda::aligned_size_t<16>(sizeof(chunk_t)), pipe);
    }
}

// Single-thread copy of one <=16B slice.
template <size_t ELEMS, FloatingNum cuda_t>
__device__ __forceinline__ void async_copy_slice_thread(cuda_t *dst, const cuda_t *src, cuda::pipeline<cuda::thread_scope_thread>& pipe) {
    constexpr size_t SLICE_BYTES = ELEMS * sizeof(cuda_t);
    static_assert(SLICE_BYTES <= 16, "async_copy_slice_thread is for small (<=16B) per-thread slices; use async_copy_row_warp for full rows");
    static_assert(16 % SLICE_BYTES == 0, "Slice width must evenly divide 16 bytes");

    using slice_t = Vec<ELEMS, cuda_t>;
    cuda::memcpy_async(
        reinterpret_cast<slice_t *>(dst), reinterpret_cast<slice_t const *>(src), cuda::aligned_size_t<SLICE_BYTES>(sizeof(slice_t)), pipe
    );
}

// Visits neighbor slots k = warp_id + it*WARPS_PER_BLOCK for it in
// [0, ceil((num_neighbors - warp_id) / WARPS_PER_BLOCK)). WARPS_PER_BLOCK=1
// gives a plain sequential loop.
//
// consume(neighbor_idx, rows): rows[r] is source r's prefetched row; valid
// only inside the call (the slot is recycled on return).
//
// dbuf: this warp's private scratch, NUM_ROWS * NUM_STAGES * D_CONST elements.
template <size_t WARPS_PER_BLOCK, size_t D_CONST, size_t NUM_STAGES, size_t NUM_ROWS, FloatingNum cuda_t, typename index_t, typename ConsumeFn>
__device__ __forceinline__ void pipelined_neighbor_row_loop(
    size_t warp_id, size_t lane, size_t num_neighbors, index_t edge_start, index_t const *__restrict__ col_idx,
    cuda_t const *__restrict__ const (&row_bases)[NUM_ROWS], int64_t const (&stride_n)[NUM_ROWS], int64_t const (&stride_h)[NUM_ROWS],
    size_t head_h, cuda_t *dbuf, ConsumeFn&& consume
) {
    const size_t loop_iters = (num_neighbors > warp_id) ? (num_neighbors - warp_id + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK : 0;
    if (loop_iters == 0) {
        return;
    }

    cuda_t *rows[NUM_ROWS][NUM_STAGES];
#pragma unroll
    for (size_t r = 0; r < NUM_ROWS; ++r) {
#pragma unroll
        for (size_t s = 0; s < NUM_STAGES; ++s) {
            rows[r][s] = dbuf + (r * NUM_STAGES + s) * D_CONST;
        }
    }

    index_t neighbor_idx_buf[NUM_STAGES];

    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

    auto prefetch = [&pipe, loop_iters, warp_id, col_idx, edge_start, &neighbor_idx_buf, &row_bases, &stride_n, &stride_h, head_h, &rows,
                        lane](size_t it) {
        pipe.producer_acquire();
        if (it < loop_iters) {
            const size_t k                    = warp_id + it * WARPS_PER_BLOCK;
            const index_t nb                  = col_idx[edge_start + static_cast<index_t>(k)];
            neighbor_idx_buf[it % NUM_STAGES] = nb;
#pragma unroll
            for (size_t r = 0; r < NUM_ROWS; ++r) {
                const cuda_t *src = row_bases[r] + nb * stride_n[r] + head_h * stride_h[r];
                async_copy_row_warp<D_CONST, cuda_t>(rows[r][it % NUM_STAGES], src, pipe, lane);
            }
        }
        pipe.producer_commit();
    };

#pragma unroll
    for (size_t s = 0; s < NUM_STAGES; ++s) {
        prefetch(s);
    }

    for (size_t iter = 0; iter < loop_iters; ++iter) {
        cuda::pipeline_consumer_wait_prior<NUM_STAGES - 1>(pipe);
        // Thread-scope wait covers only this lane's own cp.async; a lane may read
        // chunks copied by other lanes (tile < 16B, or row < 512B leaves lanes idle).
        __syncwarp();

        const size_t slot = iter % NUM_STAGES;
        cuda_t const *cur_rows[NUM_ROWS];
#pragma unroll
        for (size_t r = 0; r < NUM_ROWS; ++r) {
            cur_rows[r] = rows[r][slot];
        }

        consume(neighbor_idx_buf[slot], cur_rows);

        // All lanes must finish reading the slot before prefetch() reuses it.
        __syncwarp();
        pipe.consumer_release();
        prefetch(iter + NUM_STAGES);
    }
}
