#pragma once

#include <cstdint>

#include "common.cuh"
#include "common/tile.cuh"
#include "reduction/reduction_common.cuh"

// ================================================================================================
// reduction_aggr backward for min/max: only the winning edge of every output element contributed,
// so its gradient goes to that edge's source,
//
//     grad_x[arg[v, f], f] += grad_out[v, f]       for every valid arg[v, f]
//
// The gradient accumulates in fp32 whatever the input dtype; the caller casts it once at the end.
// A 16-bit atomicAdd is a compare-and-swap loop on the 32-bit word holding the value -- sm_90 has no
// scalar 16-bit atomic, so ptxas lowers atom.add.f16 / .bf16 to ATOM.CAS -- and the two features of
// a word belong to one source row, so neighbouring lanes picking the same source collide with each
// other and retry. On dense graphs whose hubs win many outputs that cost 50-200x: max_aggr backward
// on twitch-views (13.8M edges) took 13-74 ms in fp16 against 0.15-1.3 ms in fp32, in both this
// kernel's 16-bit version and the one before it. fp32 atomics are fire-and-forget RED.ADD.F32.
//
// One thread per element, consecutive threads on consecutive elements, so that each atomic
// instruction of a warp covers 32 adjacent features. Measured against one thread per 16-byte
// vector of grad_out and arg -- each atomic instruction then spreads over 128-256 features -- that
// is 1.2x (fp32) to 1.65x (fp16) faster at D = 64-256, and matches the earlier one-block-per-row
// kernel without its idle threads on short rows. (Sums differentiate to a sum over the transposed
// graph, which is the forward kernel again.)
// ================================================================================================

namespace turbo_gnn::reduction {

inline constexpr uint32_t kScatterThreads = 256;

template <FloatingNum cuda_t, IntegralNum index_t>
__global__ void __launch_bounds__(kScatterThreads) arg_scatter_kernel(
    cuda_t const *const __restrict__ grad_out, int64_t stride_g, index_t const *const __restrict__ arg, float *const __restrict__ grad_x,
    int64_t rows, uint32_t D
) {
    using Sentinel = IndexSentinel<index_t>;

    auto scatter = [&](int64_t row, uint32_t f, int64_t i) {
        const index_t a = arg[i];
        if (Sentinel::is_valid(a)) {
            atomicAdd(grad_x + static_cast<int64_t>(a) * D + f, static_cast<float>(grad_out[row * stride_g + f]));
        }
    };

    // Index math in 32 bits whenever the grid allows: the 64-bit division costs several times the
    // 32-bit one. The bound keeps i + stride from wrapping.
    const uint64_t total  = static_cast<uint64_t>(rows) * D;
    const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
    const uint64_t first  = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (total + stride <= 0xffffffffull) {
        for (uint32_t i = static_cast<uint32_t>(first); i < static_cast<uint32_t>(total); i += static_cast<uint32_t>(stride)) {
            const uint32_t row = i / D;
            scatter(row, i - row * D, i);
        }
    } else {
        for (uint64_t i = first; i < total; i += stride) {
            const uint64_t row = i / D;
            scatter(static_cast<int64_t>(row), static_cast<uint32_t>(i - row * D), static_cast<int64_t>(i));
        }
    }
}

}  // namespace turbo_gnn::reduction
