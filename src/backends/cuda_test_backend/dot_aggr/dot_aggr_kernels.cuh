#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <stdint.h>
#include <type_traits>

// ---------------------------------------------
// Constants
// ---------------------------------------------
static constexpr int kWarpSize = 32;

// ---------------------------------------------
// bf16 bit helpers
// ---------------------------------------------
__device__ __forceinline__ float bf16_bits_to_float(uint16_t bits) {
    return __bfloat162float(__ushort_as_bfloat16(bits));
}

// ---------------------------------------------
// 16B pack load via uint4
// ---------------------------------------------
union Pack16 {
    uint4    u4;
    float    f32[4];
    uint16_t u16[8];
};

template <typename scalar_t>
struct ScalarIO {
    __device__ __forceinline__ static float load_scalar(const scalar_t* ptr) {
        if constexpr (std::is_same<scalar_t, float>::value) {
            return *ptr;
        } else { // at::BFloat16 bits
            const uint16_t* p = reinterpret_cast<const uint16_t*>(ptr);
            return bf16_bits_to_float(*p);
        }
    }

    template <int PACK_ELEMS>
    __device__ __forceinline__ static void load_pack16(const scalar_t* base, float (&vals)[PACK_ELEMS]) {
        static_assert(PACK_ELEMS == 4 || PACK_ELEMS == 8, "PACK_ELEMS must be 4(fp32) or 8(bf16)");
        // Requires 16B alignment of `base`
        Pack16 pk;
        pk.u4 = *reinterpret_cast<const uint4*>(base);

        if constexpr (std::is_same<scalar_t, float>::value) {
#pragma unroll
            for (int i = 0; i < 4; ++i) vals[i] = pk.f32[i];
        } else {
#pragma unroll
            for (int i = 0; i < 8; ++i) vals[i] = bf16_bits_to_float(pk.u16[i]);
        }
    }
};

// ---------------------------------------------
// Warp reduce sum
// ---------------------------------------------
__device__ __forceinline__ float warp_reduce_sum(float x) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        x += __shfl_down_sync(0xffffffff, x, offset);
    }
    return x;
}

// ============================================================================
// Kernel 0: feature-parallel / feature-coalesced
// 1 warp per node. Lanes handle feature packs.
// For each neighbor: accumulate partial dot. Final warp-reduce.
// ============================================================================
template <typename scalar_t, bool kUseSecondAccess, bool kUseVectorizedLoads, int kWarpsPerBlock>
__global__ void dot_aggr_kernel_feature_parallel(
    const int32_t* __restrict__ edge_ptr,  // [N+1]
    const int32_t* __restrict__ edge_idx,  // [E]
    const scalar_t* __restrict__ X,        // [N, D]
    float* __restrict__ out,               // [N]
    float* __restrict__ scratch,           // [N] or nullptr
    int32_t N,
    int32_t D
) {
    const int tid = threadIdx.x;
    const int warp_id = tid / kWarpSize;
    const int lane = tid % kWarpSize;

    const int32_t v = (int32_t)(blockIdx.x * kWarpsPerBlock + warp_id);
    if (v >= N) return;

    const int32_t row_start = edge_ptr[v];
    const int32_t row_end   = edge_ptr[v + 1];

    constexpr int kPackBytes = 16;
    constexpr int kPackElems = kUseVectorizedLoads ? (kPackBytes / (int)sizeof(scalar_t)) : 1;

    auto run_pass = [&](int pass_id) {
        float lane_sum = 0.0f;

        // Each lane owns a strided set of feature packs.
        for (int32_t f_base = lane * kPackElems; f_base < D; f_base += kWarpSize * kPackElems) {
            const int64_t xv_off = (int64_t)v * (int64_t)D + (int64_t)f_base;

            if constexpr (kUseVectorizedLoads) {
                float xv[kPackElems];
                ScalarIO<scalar_t>::template load_pack16<kPackElems>(X + xv_off, xv);

                for (int32_t e = row_start; e < row_end; ++e) {
                    const int32_t u = edge_idx[e];
                    const int64_t xu_off = (int64_t)u * (int64_t)D + (int64_t)f_base;

                    float xu[kPackElems];
                    ScalarIO<scalar_t>::template load_pack16<kPackElems>(X + xu_off, xu);

#pragma unroll
                    for (int i = 0; i < kPackElems; ++i) {
                        lane_sum += xv[i] * xu[i];
                    }
                }
            } else {
                const float xv = ScalarIO<scalar_t>::load_scalar(X + xv_off);
                for (int32_t e = row_start; e < row_end; ++e) {
                    const int32_t u = edge_idx[e];
                    const int64_t xu_off = (int64_t)u * (int64_t)D + (int64_t)f_base;
                    lane_sum += xv * ScalarIO<scalar_t>::load_scalar(X + xu_off);
                }
            }
        }

        float y = warp_reduce_sum(lane_sum);
        if (lane == 0) {
            if constexpr (kUseSecondAccess) {
                if (pass_id == 0 && scratch) scratch[v] = y; // anchor pass0
            }
            out[v] = y;
        }
    };

    run_pass(0);
    if constexpr (kUseSecondAccess) run_pass(1);
}

// ============================================================================
// Kernel 1: neighbor-parallel / layout-sensitive
// 1 warp per node. Lanes split neighbors.
// For each neighbor: lane computes full dot sequentially over features.
// Final warp-reduce.
// ============================================================================
template <typename scalar_t, bool kUseSecondAccess, bool kUseVectorizedLoads, int kWarpsPerBlock>
__global__ void dot_aggr_kernel_neighbor_parallel(
    const int32_t* __restrict__ edge_ptr,
    const int32_t* __restrict__ edge_idx,
    const scalar_t* __restrict__ X,
    float* __restrict__ out,
    float* __restrict__ scratch,
    int32_t N,
    int32_t D
) {
    const int tid = threadIdx.x;
    const int warp_id = tid / kWarpSize;
    const int lane = tid % kWarpSize;

    const int32_t v = (int32_t)(blockIdx.x * kWarpsPerBlock + warp_id);
    if (v >= N) return;

    const int32_t row_start = edge_ptr[v];
    const int32_t row_end   = edge_ptr[v + 1];

    constexpr int kPackBytes = 16;
    constexpr int kPackElems = kUseVectorizedLoads ? (kPackBytes / (int)sizeof(scalar_t)) : 1;

    auto run_pass = [&](int pass_id) {
        float lane_sum = 0.0f;

        // Each lane handles neighbors in a strided fashion.
        for (int32_t e = row_start + lane; e < row_end; e += kWarpSize) {
            const int32_t u = edge_idx[e];
            float dot = 0.0f;

            // Full dot over features (sequential in each lane).
            for (int32_t f_base = 0; f_base < D; f_base += kPackElems) {
                const int64_t xv_off = (int64_t)v * (int64_t)D + (int64_t)f_base;
                const int64_t xu_off = (int64_t)u * (int64_t)D + (int64_t)f_base;

                if constexpr (kUseVectorizedLoads) {
                    float xv[kPackElems];
                    float xu[kPackElems];
                    ScalarIO<scalar_t>::template load_pack16<kPackElems>(X + xv_off, xv);
                    ScalarIO<scalar_t>::template load_pack16<kPackElems>(X + xu_off, xu);
#pragma unroll
                    for (int i = 0; i < kPackElems; ++i) {
                        dot += xv[i] * xu[i];
                    }
                } else {
                    dot += ScalarIO<scalar_t>::load_scalar(X + xv_off) *
                           ScalarIO<scalar_t>::load_scalar(X + xu_off);
                }
            }

            lane_sum += dot;
        }

        float y = warp_reduce_sum(lane_sum);
        if (lane == 0) {
            if constexpr (kUseSecondAccess) {
                if (pass_id == 0 && scratch) scratch[v] = y; // anchor pass0
            }
            out[v] = y;
        }
    };

    run_pass(0);
    if constexpr (kUseSecondAccess) run_pass(1);
}

// ============================================================================
// Launchers (templated)
// ============================================================================
template <typename scalar_t, bool kUseSecondAccess, bool kUseVectorizedLoads>
inline void dot_aggr_launch_kernel_feature_parallel(
    const int32_t* edge_ptr,
    const int32_t* edge_idx,
    const scalar_t* X,
    float* out,
    float* scratch,
    int32_t N,
    int32_t D,
    cudaStream_t stream
) {
    constexpr int kWarpsPerBlock = 4;
    constexpr int kThreads = kWarpsPerBlock * kWarpSize;
    dim3 block(kThreads);
    dim3 grid((N + kWarpsPerBlock - 1) / kWarpsPerBlock);

    dot_aggr_kernel_feature_parallel<scalar_t, kUseSecondAccess, kUseVectorizedLoads, kWarpsPerBlock>
        <<<grid, block, 0, stream>>>(edge_ptr, edge_idx, X, out, scratch, N, D);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t, bool kUseSecondAccess, bool kUseVectorizedLoads>
inline void dot_aggr_launch_kernel_neighbor_parallel(
    const int32_t* edge_ptr,
    const int32_t* edge_idx,
    const scalar_t* X,
    float* out,
    float* scratch,
    int32_t N,
    int32_t D,
    cudaStream_t stream
) {
    constexpr int kWarpsPerBlock = 4;
    constexpr int kThreads = kWarpsPerBlock * kWarpSize;
    dim3 block(kThreads);
    dim3 grid((N + kWarpsPerBlock - 1) / kWarpsPerBlock);

    dot_aggr_kernel_neighbor_parallel<scalar_t, kUseSecondAccess, kUseVectorizedLoads, kWarpsPerBlock>
        <<<grid, block, 0, stream>>>(edge_ptr, edge_idx, X, out, scratch, N, D);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
