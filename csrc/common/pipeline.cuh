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
template <int ROW_ELEMS, FloatingNum cuda_t>
__device__ __forceinline__ void async_copy_row_warp(cuda_t *dst, const cuda_t *src, cuda::pipeline<cuda::thread_scope_thread>& pipe, int lane) {
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
__device__ __forceinline__ void async_copy_slice_thread(cuda_t *dst, const cuda_t *src, cuda::pipeline<cuda::thread_scope_thread>& pipe) {
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
    cuda_t *dbuf, ConsumeFn&& consume
) {
    const int loop_iters = (num_neighbors > warp_id) ? (num_neighbors - warp_id + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK : 0;
    if (loop_iters == 0) return;

    // No local arrays here, deliberately. A `cuda_t *rows[NUM_ROWS][NUM_STAGES]` indexed by a
    // runtime slot, or a `neighbor_idx_buf[NUM_STAGES]` ring, cannot stay in registers -- nvcc
    // places them in local memory, and the spill grows with NUM_STAGES (measured: STACK:96 at
    // NUM_STAGES=3, NUM_ROWS=3). The slot address is affine in the slot, so compute it; the
    // neighbour index is re-read from col_idx, which the prefetch just touched and is an L1 hit.
    auto slot_row    = [dbuf](int r, int slot) -> cuda_t    *{ return dbuf + (r * NUM_STAGES + slot) * D_CONST; };
    auto neighbor_at = [col_idx, edge_start, warp_id](int it) -> index_t {
        return col_idx[edge_start + static_cast<index_t>(warp_id + it * WARPS_PER_BLOCK)];
    };

    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

    auto prefetch = [&](int it) {
        pipe.producer_acquire();
        if (it < loop_iters) {
            const index_t nb = neighbor_at(it);
#pragma unroll
            for (int r = 0; r < NUM_ROWS; ++r) {
                const cuda_t *src = row_bases[r] + nb * stride_n[r] + head_h * stride_h[r];
                async_copy_row_warp<D_CONST, cuda_t>(slot_row(r, it % NUM_STAGES), src, pipe, lane);
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
        // Built fresh each iteration from affine arithmetic; NUM_ROWS is a small compile-time
        // constant and the loop is unrolled, so these stay in registers.
        cuda_t const *cur_rows[NUM_ROWS];
#pragma unroll
        for (int r = 0; r < NUM_ROWS; ++r) {
            cur_rows[r] = slot_row(r, slot);
        }

        consume(neighbor_at(iter), cur_rows);

        // All lanes must finish reading the slot before prefetch() reuses it.
        __syncwarp();
        pipe.consumer_release();
        prefetch(iter + NUM_STAGES);
    }
}

// =============================================================================
// Runtime row length.
//
// The loop above takes the row width as a template parameter. The graph transformer kernels take
// any head dim, so their row width is known only at launch and these variants take it as an
// argument. They follow the hold convention of gsddmm_pipeline.cuh rather than the loop above:
// the copy for iteration it + NUM_STAGES is issued *before* iteration it is consumed, into a
// spare slot, so NUM_STAGES copies overlap every consume() at the cost of one extra ring slot.
// The loop above issues its next copy only after consuming, leaving NUM_STAGES - 1 in flight.
//
// Rows must be 16-byte aligned and row_elems * sizeof(cuda_t) a multiple of 16.
// =============================================================================

// Elements of one warp's ring: NUM_STAGES + 1 slots of num_rows rows each; 0 when the pipeline is
// off, so launchers can add the term unconditionally.
__host__ __device__ constexpr size_t pipelined_rows_ring_elems(size_t num_stages, size_t num_rows, size_t row_elems) {
    return num_stages == 0 ? 0 : (num_stages + 1) * num_rows * row_elems;
}

// Lane-parallel cp.async of NUM_ROWS rows, 16-byte chunks; row r of the stage lands at
// dst + r * row_elems.
template <size_t NUM_ROWS, FloatingNum cuda_t>
__device__ __forceinline__ void async_copy_rows_warp_rt(
    cuda_t *dst, uint32_t row_elems, cuda_t const *const (&srcs)[NUM_ROWS], cuda::pipeline<cuda::thread_scope_thread>& pipe, uint32_t lane
) {
    using chunk_t                 = Vec<16 / sizeof(cuda_t), cuda_t>;
    const uint32_t chunks_per_row = row_elems * static_cast<uint32_t>(sizeof(cuda_t)) / 16;
#pragma unroll
    for (size_t r = 0; r < NUM_ROWS; ++r) {
        chunk_t *const d       = reinterpret_cast<chunk_t *>(dst + r * row_elems);
        chunk_t const *const s = reinterpret_cast<chunk_t const *>(srcs[r]);
        for (uint32_t i = lane; i < chunks_per_row; i += kWarpSize) {
            cuda::memcpy_async(d + i, s + i, cuda::aligned_size_t<16>(sizeof(chunk_t)), pipe);
        }
    }
}

// One warp's loop over loop_iters row gathers, NUM_STAGES deep.
//
// addr(it, srcs): fills srcs[r] with the global address of row r of iteration it; warp-uniform.
// consume(it, rows): rows[r] is that row's copy in shared memory, valid for the whole call.
//
// dbuf: this warp's private ring, pipelined_rows_ring_elems(NUM_STAGES, NUM_ROWS, row_elems)
// elements. Slots are walked with pointer cursors rather than indexed: a register array indexed by
// a runtime slot is demoted to local memory (see gsddmm_pipeline.cuh).
template <size_t NUM_STAGES, size_t NUM_ROWS, FloatingNum cuda_t, typename AddrFn, typename ConsumeFn>
__device__ __forceinline__ void pipelined_rows_loop(
    uint32_t lane, uint32_t loop_iters, uint32_t row_elems, cuda_t *__restrict__ dbuf, AddrFn&& addr, ConsumeFn&& consume
) {
    static_assert(NUM_STAGES >= 1, "The pipeline needs at least one stage in flight; PIPELINE_STAGES == 0 takes the direct-load loop");
    if (loop_iters == 0) [[unlikely]] {
        return;
    }

    // cp.async.wait_group takes an immediate of at most 8 pending groups. A deeper pipeline waits
    // for a few groups more than iteration `it` needs -- still correct, only no deeper than 8.
    constexpr int kWaitPrior = NUM_STAGES - 1 < 8 ? static_cast<int>(NUM_STAGES - 1) : 8;

    const size_t slot_elems = NUM_ROWS * static_cast<size_t>(row_elems);
    cuda_t *const ring_end  = dbuf + (NUM_STAGES + 1) * slot_elems;
    auto advance            = [dbuf, ring_end, slot_elems](cuda_t *slot) {
        cuda_t *const next = slot + slot_elems;
        return next == ring_end ? dbuf : next;
    };

    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

    auto prefetch = [&](uint32_t it, cuda_t *slot) {
        pipe.producer_acquire();
        if (it < loop_iters) {
            cuda_t const *srcs[NUM_ROWS];
            addr(it, srcs);
            async_copy_rows_warp_rt<NUM_ROWS, cuda_t>(slot, row_elems, srcs, pipe, lane);
        }
        pipe.producer_commit();
    };

    cuda_t *prefetch_slot = dbuf;
#pragma unroll
    for (uint32_t s = 0; s < NUM_STAGES; ++s) {
        prefetch(s, prefetch_slot);
        prefetch_slot = advance(prefetch_slot);
    }

    cuda_t *consume_slot = dbuf;
    for (uint32_t iter = 0; iter < loop_iters; ++iter) {
        cuda::pipeline_consumer_wait_prior<kWaitPrior>(pipe);
        // The spare slot held iteration iter - 1, released at the end of the previous iteration.
        prefetch(iter + static_cast<uint32_t>(NUM_STAGES), prefetch_slot);
        // The thread-scope wait covers only this lane's own copies; lanes read each other's chunks.
        __syncwarp();

        cuda_t const *rows[NUM_ROWS];
#pragma unroll
        for (size_t r = 0; r < NUM_ROWS; ++r) {
            rows[r] = consume_slot + r * row_elems;
        }
        consume(iter, rows);

        // All lanes must finish reading the slot before a later prefetch() refills it.
        __syncwarp();
        pipe.consumer_release();
        prefetch_slot = advance(prefetch_slot);
        consume_slot  = advance(consume_slot);
    }
}
