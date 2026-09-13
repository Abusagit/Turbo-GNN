#pragma once

#include <cuda/pipeline>

#include "common/misc.cuh"
#include "common/tile.cuh"
#include "common/traits.cuh"

// =============================================================================
// cp.async prefetch primitives for neighbor loops (cuda::pipeline, thread scope).
//
// async_copy_rows_warp        -- warp copies the NUM_ROWS rows of one stage,
//   packing several narrow rows per cp.async (the copy step of pipelined_row_loop).
// async_copy_slice_thread     -- one thread copies one <=16B slice.
// pipelined_row_loop          -- the generic warp loop: NUM_STAGES-deep prefetch
//   of NUM_ROWS rows per iteration into a per-warp shared ring buffer, the row
//   addresses supplied by a callback.
// pipelined_neighbor_row_loop -- CSR adapter over it: rows indexed by
//   col_idx[edge_start + k] with per-row (node, head) strides.
// pipelined_ring_slots / pipelined_ring_elems -- ring-buffer sizing, shared by
//   the kernels' shared-memory layouts and their host launchers.
//
// Stage convention, for every loop in this file: NUM_STAGES is the number of
// iterations whose copies are IN FLIGHT while one iteration is consumed, so
// NUM_STAGES = 1 already overlaps the next gather with the current compute.
// Two consumer styles are supported, selected by EARLY_RELEASE:
//
//   hold  (EARLY_RELEASE = false): consume(it, rows) reads the shared rows for
//         its whole duration. The prefetch of iteration it + NUM_STAGES is
//         issued BEFORE consume(it) into a spare slot, so the ring has
//         NUM_STAGES + 1 slots: the one being read plus NUM_STAGES in flight.
//   early (EARLY_RELEASE = true): consume(it, rows, release) copies the rows
//         into registers, calls release() -- warp-uniformly, exactly once --
//         and computes from the registers. The prefetch of it + NUM_STAGES is
//         issued by release() into the slot just freed, so NUM_STAGES slots
//         suffice for the same overlap: half the shared memory for
//         NUM_STAGES = 1, which is what keeps the heavy 32-warp blocks at wide
//         D from losing occupancy. A consumer that forgets release() is still
//         correct (it is called on return), it just loses the overlap.
//
// Issuing the prefetch after consume() without an early release would keep
// only NUM_STAGES - 1 copies in flight during the compute -- none at all for
// NUM_STAGES = 1 -- for the same shared memory; no loop here does that.
//
// For thread-scope pipelines producer_acquire()/consumer_release() are no-ops,
// producer_commit() is cp.async.commit_group and pipeline_consumer_wait_prior<N>
// is cp.async.wait_group N ("at most N groups still pending"); the ordering
// above follows directly from that.
// =============================================================================

// Ring-buffer slots a NUM_STAGES-deep pipeline needs; 0 when the pipeline is
// off (NUM_STAGES == 0), so launchers can add the term unconditionally.
__host__ __device__ constexpr size_t pipelined_ring_slots(size_t num_stages, bool early_release = false) {
    return num_stages == 0 ? 0 : (early_release ? num_stages : num_stages + 1);
}

// Elements of ONE warp's ring buffer: NUM_ROWS rows of D_CONST per slot.
__host__ __device__ constexpr size_t pipelined_ring_elems(size_t num_stages, size_t num_rows, size_t d_const, bool early_release = false) {
    return pipelined_ring_slots(num_stages, early_release) * num_rows * d_const;
}

// Lane-parallel cp.async of the NUM_ROWS rows of one pipeline stage, 16-byte
// chunks. Row r of the stage lives at slot_rows + r * ROW_STRIDE (see
// pipelined_row_loop).
//
// A row narrower than the warp (fewer than 32 chunks) would leave lanes idle
// and cost one predicated cp.async per row, so ROWS_PER_GROUP = 32 / chunks
// such rows are packed into ONE warp-wide cp.async: e.g. two 256-byte rows
// (fp16 D=128) cost a single instruction per lane instead of two half-warp
// ones -- measured at 4-9% of the issue-bound GSDDMM dot kernel. The lane's
// row within a group is chosen with an unrolled select chain over the
// ROWS_PER_GROUP candidates (indexing srcs[] by a lane-dependent value would
// demote the array to local memory), so the chain is one compare for two rows
// per group and vanishes for one. A row of 32 chunks or more already fills
// the warp and is copied plainly, chunk lane, lane + 32, ...: packing could
// not save an instruction there and its select chain over NUM_ROWS rows was
// measured at +50% instructions on the four-row fp32 D=256 GT backward.
template <size_t ROW_ELEMS, size_t NUM_ROWS, size_t ROW_STRIDE, FloatingNum cuda_t>
__device__ __forceinline__ void async_copy_rows_warp(
    cuda_t *slot_rows, cuda_t const *const (&srcs)[NUM_ROWS], cuda::pipeline<cuda::thread_scope_thread>& pipe, size_t lane
) {
    constexpr size_t ROW_BYTES = ROW_ELEMS * sizeof(cuda_t);
    static_assert(ROW_BYTES % 16 == 0, "Row width must be a multiple of 16 bytes for aligned async copies");
    constexpr size_t CHUNK_ELEMS    = 16 / sizeof(cuda_t);
    constexpr size_t CHUNKS_PER_ROW = ROW_BYTES / 16;
    using chunk_t                   = Vec<CHUNK_ELEMS, cuda_t>;

    auto copy_chunk = [&pipe](cuda_t *dst, cuda_t const *src, size_t i) {
        cuda::memcpy_async(
            reinterpret_cast<chunk_t *>(dst) + i, reinterpret_cast<chunk_t const *>(src) + i, cuda::aligned_size_t<16>(sizeof(chunk_t)), pipe
        );
    };

    if constexpr (CHUNKS_PER_ROW >= kWarpSize || kWarpSize % CHUNKS_PER_ROW != 0) {
        // Wide rows (or a width that does not tile the warp): plain per-row
        // copy, chunk lane + 32k. The trip count is a compile-time constant on
        // purpose: written as `for (i = lane; i < CHUNKS; i += 32)` the start
        // depends on the lane and the compiler emits a real loop -- measured as
        // a 12-instruction loop per prefetch site, +10-35% instructions on the
        // 32-warp GSDDMM kernels -- instead of one straight-line cp.async per k.
        constexpr size_t COPIES_PER_LANE = ceil_div(CHUNKS_PER_ROW, kWarpSize);
#pragma unroll
        for (size_t r = 0; r < NUM_ROWS; ++r) {
#pragma unroll
            for (size_t k = 0; k < COPIES_PER_LANE; ++k) {
                const size_t i = lane + k * kWarpSize;
                if (CHUNKS_PER_ROW % kWarpSize == 0 || i < CHUNKS_PER_ROW) {
                    copy_chunk(slot_rows + r * ROW_STRIDE, srcs[r], i);
                }
            }
        }
    } else {
        // Narrow rows: ROWS_PER_GROUP rows per warp-wide cp.async. The lane's
        // row within the group and its chunk are fixed for every group (the
        // divisor is a power of two, so this is a shift and a mask).
        constexpr size_t ROWS_PER_GROUP = kWarpSize / CHUNKS_PER_ROW;
        const size_t sub                = lane / CHUNKS_PER_ROW;
        const size_t i                  = lane % CHUNKS_PER_ROW;
#pragma unroll
        for (size_t g = 0; g < NUM_ROWS; g += ROWS_PER_GROUP) {
            if (g + sub < NUM_ROWS) {  // the last group may be partial
                cuda_t const *src = srcs[g];
                cuda_t *dst       = slot_rows + g * ROW_STRIDE;
#pragma unroll
                for (size_t k = 1; k < ROWS_PER_GROUP; ++k) {
                    if (g + k < NUM_ROWS && sub == k) {
                        src = srcs[g + k];
                        dst = slot_rows + (g + k) * ROW_STRIDE;
                    }
                }
                copy_chunk(dst, src, i);
            }
        }
    }
}

// Single-thread copy of one <=16B slice.
template <size_t ELEMS, FloatingNum cuda_t>
__device__ __forceinline__ void async_copy_slice_thread(cuda_t *dst, const cuda_t *src, cuda::pipeline<cuda::thread_scope_thread>& pipe) {
    constexpr size_t SLICE_BYTES = ELEMS * sizeof(cuda_t);
    static_assert(SLICE_BYTES <= 16, "async_copy_slice_thread is for small (<=16B) per-thread slices; use async_copy_rows_warp for full rows");
    static_assert(16 % SLICE_BYTES == 0, "Slice width must evenly divide 16 bytes");

    using slice_t = Vec<ELEMS, cuda_t>;
    cuda::memcpy_async(
        reinterpret_cast<slice_t *>(dst), reinterpret_cast<slice_t const *>(src), cuda::aligned_size_t<SLICE_BYTES>(sizeof(slice_t)), pipe
    );
}

// =============================================================================
// Generic cp.async row loop over a warp's sequence of loop_iters row gathers.
//
// addr(it, srcs):    fills srcs[r] with the global address of row r for
//                    iteration it. Must be warp-uniform. It is what makes the
//                    loop generic: rows may be indexed by neighbor id, by edge
//                    position, through an explicit edge list -- anything the
//                    caller can compute from the iteration index.
// consume(it, rows)            [hold]:  rows[r] is row r's prefetched copy in
//                    shared memory, valid for the whole call.
// consume(it, rows, release)   [early]: rows[r] is valid until release() is
//                    called (see the header comment for the contract).
//
// Ring layout: row r of slot s is at dbuf + s * D_CONST + r * ROW_STRIDE with
// ROW_STRIDE = NUM_SLOTS * D_CONST. Slots are addressed through running
// POINTER cursors, never through a pointer array or a slot index: a register
// array indexed by a runtime slot is demoted to local memory (measured: 19M
// local loads and a 4x instruction count for three slots), and a slot index
// costs a 64-bit multiply per use, which on the light kernels -- ~50
// instructions per neighbor -- is a measurable share of the loop. For the same
// reason consume() receives the iteration index and recomputes whatever it
// needs from it instead of reading a per-slot side buffer.
//
// loop_iters is 32-bit on purpose: it counts one warp's share of one CSR row
// (or edge chunk), which never approaches 2^32, and a size_t counter costs a
// two-instruction compare and add per use -- measured at +15-25% instructions
// on the light kernels, whose whole per-neighbor body is ~50 instructions.
//
// dbuf: this warp's private shared scratch,
//       pipelined_ring_elems(NUM_STAGES, NUM_ROWS, D_CONST, EARLY_RELEASE) elements.
// =============================================================================
template <size_t D_CONST, size_t NUM_STAGES, size_t NUM_ROWS, FloatingNum cuda_t, bool EARLY_RELEASE = false, typename AddrFn, typename ConsumeFn>
__device__ __forceinline__ void pipelined_row_loop(
    size_t lane, uint32_t loop_iters, cuda_t *__restrict__ dbuf, AddrFn&& addr, ConsumeFn&& consume
) {
    static_assert(NUM_STAGES >= 1, "The pipeline needs at least one stage in flight; PIPELINE_STAGES == 0 takes the direct-load loop");
    static_assert(NUM_STAGES <= 8, "cp.async.wait_group takes an immediate of at most 8 pending groups");
    if (loop_iters == 0) [[unlikely]] {
        return;
    }

    constexpr size_t NUM_SLOTS  = pipelined_ring_slots(NUM_STAGES, EARLY_RELEASE);
    constexpr size_t ROW_STRIDE = NUM_SLOTS * D_CONST;

    // Slot cursor: row 0 of a slot; the next slot is D_CONST further, wrapping
    // at the end of row 0's stripe. A one-slot ring (early release, one stage)
    // is special-cased so the cursor is a compile-time constant: left as a
    // loop-carried compare-select the compiler cannot fold it, and was measured
    // to rematerialize the ring base and the lane's tile addresses from
    // threadIdx every iteration -- +28 instructions on a 121-instruction loop.
    cuda_t *const ring_end = dbuf + ROW_STRIDE;
    auto advance           = [dbuf, ring_end](cuda_t *slot) {
        if constexpr (NUM_SLOTS == 1) {
            return dbuf;
        } else {
            cuda_t *const next = slot + D_CONST;
            return next == ring_end ? dbuf : next;
        }
    };

    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

    auto prefetch = [&](uint32_t it, cuda_t *slot_rows) {
        pipe.producer_acquire();
        if (it < loop_iters) {
            cuda_t const *srcs[NUM_ROWS];
            addr(it, srcs);
            // All NUM_ROWS rows of the stage go into one commit group, so the
            // operands of one iteration are in flight concurrently.
            async_copy_rows_warp<D_CONST, NUM_ROWS, ROW_STRIDE, cuda_t>(slot_rows, srcs, pipe, lane);
        }
        pipe.producer_commit();
    };

    // Fill the pipeline: NUM_STAGES stages in flight.
    {
        cuda_t *slot = dbuf;
#pragma unroll
        for (uint32_t s = 0; s < NUM_STAGES; ++s) {
            prefetch(s, slot);
            slot = advance(slot);
        }
    }

    constexpr uint32_t STAGES_U32 = static_cast<uint32_t>(NUM_STAGES);
    cuda_t *consume_slot          = dbuf;
    cuda_t *prefetch_slot         = dbuf + NUM_STAGES * D_CONST;  // hold mode: the one free slot
    for (uint32_t iter = 0; iter < loop_iters; ++iter) {
        cuda::pipeline_consumer_wait_prior<NUM_STAGES - 1>(pipe);

        if constexpr (!EARLY_RELEASE) {
            // Issue the next stage's copies before consuming the current one:
            // the global->shared transfer overlaps with the compute below. The
            // target slot is the one the previous iteration's consume freed --
            // never the slot about to be read.
            prefetch(iter + STAGES_U32, prefetch_slot);
        }

        // The thread-scope wait covers only this lane's own cp.async groups; a
        // lane may read chunks copied by other lanes (tile < 16B, or a row <
        // 512B leaves lanes idle), so completion must be observed warp-wide.
        __syncwarp();

        cuda_t const *cur_rows[NUM_ROWS];
#pragma unroll
        for (size_t r = 0; r < NUM_ROWS; ++r) {
            cur_rows[r] = consume_slot + r * ROW_STRIDE;
        }

        if constexpr (EARLY_RELEASE) {
            // release(): all lanes are done reading the slot -> hand it back
            // and refill it with stage iter + NUM_STAGES while the consumer
            // computes from its registers.
            bool released = false;
            auto release  = [&]() {
                __syncwarp();
                pipe.consumer_release();
                prefetch(iter + STAGES_U32, consume_slot);
                released = true;
            };
            consume(iter, cur_rows, release);
            if (!released) {
                release();
            }
        } else {
            consume(iter, cur_rows);
            // All lanes must finish reading the slot before prefetch() reuses it.
            __syncwarp();
            pipe.consumer_release();
            prefetch_slot = advance(prefetch_slot);
        }
        consume_slot = advance(consume_slot);
    }
}

// =============================================================================
// CSR adapter: visits neighbor slots k = warp_id + it * WARPS_PER_BLOCK for it
// in [0, ceil((num_neighbors - warp_id) / WARPS_PER_BLOCK)), WARPS_PER_BLOCK = 1
// giving a plain sequential loop. Row r of neighbor j = col_idx[edge_start + k]
// is row_bases[r] + j * stride_n[r] + head_h * stride_h[r].
//
// consume(neighbor_j, rows) in hold mode, consume(neighbor_j, rows, release)
// with EARLY_RELEASE (see pipelined_row_loop). neighbor_j is re-read from
// col_idx (an L1 hit -- the prefetch loaded it NUM_STAGES iterations earlier)
// rather than buffered per slot, and the load is dead-code-eliminated for
// consumers that ignore it.
//
// dbuf: this warp's private scratch,
//       pipelined_ring_elems(NUM_STAGES, NUM_ROWS, D_CONST, EARLY_RELEASE) elements.
// =============================================================================
template <
    size_t WARPS_PER_BLOCK, size_t D_CONST, size_t NUM_STAGES, size_t NUM_ROWS, FloatingNum cuda_t, IntegralNum index_t,
    bool EARLY_RELEASE = false, typename ConsumeFn
>
__device__ __forceinline__ void pipelined_neighbor_row_loop(
    size_t warp_id, size_t lane, size_t num_neighbors, index_t edge_start, index_t const *__restrict__ col_idx,
    cuda_t const *__restrict__ const (&row_bases)[NUM_ROWS], int64_t const (&stride_n)[NUM_ROWS], int64_t const (&stride_h)[NUM_ROWS],
    size_t head_h, cuda_t *dbuf, ConsumeFn&& consume
) {
    const uint32_t loop_iters =
        static_cast<uint32_t>((num_neighbors > warp_id) ? ceil_div(num_neighbors - warp_id, WARPS_PER_BLOCK) : size_t{0});

    auto neighbor_of = [warp_id, edge_start, col_idx](uint32_t it) -> index_t {
        // 32-bit slot arithmetic (see pipelined_row_loop); only the final add is
        // done in index_t.
        const uint32_t k = static_cast<uint32_t>(warp_id) + it * static_cast<uint32_t>(WARPS_PER_BLOCK);
        return col_idx[edge_start + static_cast<index_t>(k)];
    };

    auto addr = [&](uint32_t it, cuda_t const *(&srcs)[NUM_ROWS]) {
        const index_t nb = neighbor_of(it);
#pragma unroll
        for (size_t r = 0; r < NUM_ROWS; ++r) {
            srcs[r] = row_bases[r] + nb * stride_n[r] + head_h * stride_h[r];
        }
    };

    if constexpr (EARLY_RELEASE) {
        auto consume_it = [&](uint32_t it, cuda_t const *const (&rows)[NUM_ROWS], auto&& release) { consume(neighbor_of(it), rows, release); };
        pipelined_row_loop<D_CONST, NUM_STAGES, NUM_ROWS, cuda_t, true>(lane, loop_iters, dbuf, addr, consume_it);
    } else {
        auto consume_it = [&](uint32_t it, cuda_t const *const (&rows)[NUM_ROWS]) { consume(neighbor_of(it), rows); };
        pipelined_row_loop<D_CONST, NUM_STAGES, NUM_ROWS, cuda_t, false>(lane, loop_iters, dbuf, addr, consume_it);
    }
}
