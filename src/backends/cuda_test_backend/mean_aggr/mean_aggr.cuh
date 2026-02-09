#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <stdint.h>
#include <type_traits>

constexpr int kWarpSize = 32;

__device__ __forceinline__ float bf16_bits_to_float(uint16_t bits) {
    return __bfloat162float(__ushort_as_bfloat16(bits));
}
__device__ __forceinline__ uint16_t float_to_bf16_bits_rn(float x) {
    return __bfloat16_as_ushort(__float2bfloat16_rn(x));
}

union Pack16 {
    uint4 u4;
    float f32[4];
    uint16_t u16[8];
};

template <typename scalar_t>
struct ScalarIO {
    // scalar load -> float
    __device__ __forceinline__ static float load_scalar(const scalar_t* ptr) {
        if constexpr (std::is_same<scalar_t, float>::value) {
            return *ptr;
        } else { // at::BFloat16 (2 bytes) – treat as bits
            const uint16_t* p = reinterpret_cast<const uint16_t*>(ptr);
            return bf16_bits_to_float(*p);
        }
    }

    // float -> scalar store
    __device__ __forceinline__ static void store_scalar(scalar_t* ptr, float x) {
        if constexpr (std::is_same<scalar_t, float>::value) {
            *ptr = x;
        } else {
            uint16_t* p = reinterpret_cast<uint16_t*>(ptr);
            *p = float_to_bf16_bits_rn(x);
        }
    }

    // vectorized 16B load: fills vals[PACK_ELEMS]
    template <int PACK_ELEMS>
    __device__ __forceinline__ static void load_pack16(const scalar_t* base, float (&vals)[PACK_ELEMS]) {
        static_assert(PACK_ELEMS == 4 || PACK_ELEMS == 8, "PACK_ELEMS must be 4 (fp32) or 8 (bf16)");
        const char* p = reinterpret_cast<const char*>(base);
        Pack16 pk;
        // 16B aligned load
        pk.u4 = *reinterpret_cast<const uint4*>(p);

        if constexpr (std::is_same<scalar_t, float>::value) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                vals[i] = pk.f32[i];
        }
        } else {
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                vals[i] = bf16_bits_to_float(pk.u16[i]);
            }
        }
    }

    // vectorized 16B store: writes PACK_ELEMS values
    template <int PACK_ELEMS>
    __device__ __forceinline__ static void store_pack16(scalar_t* base, const float (&vals)[PACK_ELEMS]) {
        static_assert(PACK_ELEMS == 4 || PACK_ELEMS == 8, "PACK_ELEMS must be 4 (fp32) or 8 (bf16)");
        Pack16 pk;

        if constexpr (std::is_same<scalar_t, float>::value) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                pk.f32[i] = vals[i];
            }
        } else {
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                pk.u16[i] = float_to_bf16_bits_rn(vals[i]);
            }
        }

        char* p = reinterpret_cast<char*>(base);
        *reinterpret_cast<uint4*>(p) = pk.u4;
    }
};

// -----------------------------
// Kernel 0: feature-coalesced
// 1 warp per node, lanes cover feature packs
// -----------------------------
template <typename scalar_t, bool kUseSecondAccess, bool kUseVectorizedLoads, int kWarpsPerBlock>
__global__ void mean_aggr_kernel_feature_coalesced(
    const int32_t* __restrict__ edge_ptr,
    const int32_t* __restrict__ edge_idx,
    const scalar_t* __restrict__ X,   // [N, D]
    scalar_t* __restrict__ out,       // [N, D]
    float* __restrict__ scratch,      // [N] or nullptr
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
    const int32_t deg = row_end - row_start;

    const float inv_deg = (deg > 0) ? (1.0f / (float)deg) : 0.0f;

    constexpr int kPackBytes = 16;
    constexpr int kPackElems = kUseVectorizedLoads ? (kPackBytes / (int)sizeof(scalar_t)) : 1;

    auto run_pass = [&](int pass_id) {
        // iterate feature packs
        for (int32_t f_base = lane * kPackElems; f_base < D; f_base += kWarpSize * kPackElems) {
            float acc[kPackElems];

            #pragma unroll
            for (int i = 0; i < kPackElems; ++i) {
                acc[i] = 0.0f;
            }

            // sum over neighbors
            for (int32_t e = row_start; e < row_end; ++e) {
                const int32_t u = edge_idx[e];
                const int64_t base_idx = (int64_t)u * (int64_t)D + (int64_t)f_base;

                if constexpr (kUseVectorizedLoads) {
                    // safe because wrapper enforces: X aligned 16B and D* sizeof(scalar_t) multiple of 16
                    float vals[kPackElems];
                    ScalarIO<scalar_t>::template load_pack16<kPackElems>(X + base_idx, vals);

                    #pragma unroll
                    for (int i = 0; i < kPackElems; ++i) {
                        acc[i] += vals[i];
                    }
                } else {
                    // scalar path
                    if (f_base < D) {
                        acc[0] += ScalarIO<scalar_t>::load_scalar(X + base_idx);
                    }
                }
            }

            // scale and store
            #pragma unroll
            for (int i = 0; i < kPackElems; ++i) {
                acc[i] *= inv_deg;
            }

            const int64_t out_base = (int64_t)v * (int64_t)D + (int64_t)f_base;

            if constexpr (kUseVectorizedLoads) {
                if (f_base + kPackElems <= D) {
                    ScalarIO<scalar_t>::template store_pack16<kPackElems>(out + out_base, acc);
                } else {
                    // tail (should not happen if D % kPackElems == 0, but keep safe)
                    #pragma unroll
                    for (int i = 0; i < kPackElems; ++i) {
                        const int32_t f = f_base + i;
                        if (f < D) {
                            ScalarIO<scalar_t>::store_scalar(out + out_base + i, acc[i]);
                        }
                    }
                }
            } else {
                ScalarIO<scalar_t>::store_scalar(out + out_base, acc[0]);
            }

            // write one float from pass 0 to keep it alive
            if constexpr (kUseSecondAccess) {
                if (pass_id == 0 && scratch != nullptr) {
                    // lane 0 owns f_base==0 in both scalar & vector paths
                    if (f_base == 0 && lane == 0) scratch[v] = acc[0];
                }
            }
        }
    };

    run_pass(0);
    if constexpr (kUseSecondAccess) run_pass(1);
}

// -----------------------------
// Kernel 1: neighbor-parallel (reordering-sensitive)
// 1 warp per node, split warp into 4 feature-groups × 8 neighbor-lanes
// Each group processes one feature pack; lanes in group load different neighbors.
// -----------------------------
template <typename scalar_t, bool kUseSecondAccess, bool kUseVectorizedLoads, int kWarpsPerBlock>
__global__ void mean_aggr_kernel_neighbor_parallel(
    const int32_t* __restrict__ edge_ptr,
    const int32_t* __restrict__ edge_idx,
    const scalar_t* __restrict__ X,
    scalar_t* __restrict__ out,
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
    const int32_t deg = row_end - row_start;

    const float inv_deg = (deg > 0) ? (1.0f / (float)deg) : 0.0f;

    constexpr int kPackBytes = 16;
    constexpr int kPackElems = kUseVectorizedLoads ? (kPackBytes / (int)sizeof(scalar_t)) : 1;

    // 4 groups × 8 lanes = 32 lanes
    static constexpr int kNeighborGroup = 32;
    static constexpr int kFeatGroups = 1;
    static_assert(kNeighborGroup * kFeatGroups == 32, "grouping must cover a warp");

    const int group_id = lane / kNeighborGroup;      // 0..3
    const int lane_in_group = lane % kNeighborGroup; // 0..7

    auto run_pass = [&](int pass_id) {
        // feature-pack index (in units of kPackElems)
        const int32_t num_packs = (D + kPackElems - 1) / kPackElems;

        for (int32_t pack0 = group_id; pack0 < num_packs; pack0 += kFeatGroups) {
            const int32_t f_base = pack0 * kPackElems;

            float acc[kPackElems];
            #pragma unroll
            for (int i = 0; i < kPackElems; ++i) {
                acc[i] = 0.0f;
            }

            // iterate neighbors in chunks of 8 (one per lane_in_group)
            for (int32_t nbr_base = 0; nbr_base < deg; nbr_base += kNeighborGroup) {
                const int32_t e = row_start + nbr_base + lane_in_group;

                float tmp[kPackElems];

                #pragma unroll
                for (int i = 0; i < kPackElems; ++i) {
                    tmp[i] = 0.0f;
                }

                if (e < row_end) {
                    const int32_t u = edge_idx[e];
                    const int64_t base_idx = (int64_t)u * (int64_t)D + (int64_t)f_base;

                    if constexpr (kUseVectorizedLoads) {
                        ScalarIO<scalar_t>::template load_pack16<kPackElems>(X + base_idx, tmp);
                    } else {
                        if (f_base < D) tmp[0] = ScalarIO<scalar_t>::load_scalar(X + base_idx);
                    }
                }

                // reduce within group (width=8) for each element
                #pragma unroll
                for (int offset = kNeighborGroup / 2; offset > 0; offset >>= 1) {
                    #pragma unroll
                    for (int i = 0; i < kPackElems; ++i) {
                        tmp[i] += __shfl_down_sync(0xffffffff, tmp[i], offset, kNeighborGroup);
                    }
                }

                // group leader accumulates
                if (lane_in_group == 0) {
                    #pragma unroll
                    for (int i = 0; i < kPackElems; ++i) {
                        acc[i] += tmp[i];
                    }
                }
            }

            if (lane_in_group == 0) {
                #pragma unroll
                for (int i = 0; i < kPackElems; ++i) {
                    acc[i] *= inv_deg;
                }

                const int64_t out_base = (int64_t)v * (int64_t)D + (int64_t)f_base;

                if constexpr (kUseVectorizedLoads) {
                    if (f_base + kPackElems <= D) {
                        ScalarIO<scalar_t>::template store_pack16<kPackElems>(out + out_base, acc);
                    } else {
                        #pragma unroll
                        for (int i = 0; i < kPackElems; ++i) {
                            const int32_t f = f_base + i;
                            if (f < D) {
                                ScalarIO<scalar_t>::store_scalar(out + out_base + i, acc[i]);
                            }
                        }
                    }
                } else {
                    ScalarIO<scalar_t>::store_scalar(out + out_base, acc[0]);
                }

                if constexpr (kUseSecondAccess) {
                    if (pass_id == 0 && scratch != nullptr) {
                        // group 0 leader (lane==0) owns f_base==0
                        if (group_id == 0 && f_base == 0) {
                            scratch[v] = acc[0];
                        }
                    }
                }
            }
        }
    };

    run_pass(0);
    if constexpr (kUseSecondAccess) run_pass(1);
}

// -----------------------------
// Launchers
// -----------------------------
template <typename scalar_t, bool kUseSecondAccess, bool kUseVectorizedLoads>
void mean_aggr_launch_kernel_feature_coalesced(
    const int32_t* edge_ptr,
    const int32_t* edge_idx,
    const scalar_t* X,
    scalar_t* out,
    float* scratch,
    int32_t N,
    int32_t D,
    cudaStream_t stream
) {
    constexpr int kWarpsPerBlock = 4;
    constexpr int kThreads = kWarpsPerBlock * kWarpSize;
    dim3 block(kThreads);
    dim3 grid((N + kWarpsPerBlock - 1) / kWarpsPerBlock);

    mean_aggr_kernel_feature_coalesced<scalar_t, kUseSecondAccess, kUseVectorizedLoads, kWarpsPerBlock>
        <<<grid, block, 0, stream>>>(edge_ptr, edge_idx, X, out, scratch, N, D);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t, bool kUseSecondAccess, bool kUseVectorizedLoads>
void mean_aggr_launch_kernel_neighbor_parallel(
    const int32_t* edge_ptr,
    const int32_t* edge_idx,
    const scalar_t* X,
    scalar_t* out,
    float* scratch,
    int32_t N,
    int32_t D,
    cudaStream_t stream
) {
    constexpr int kWarpsPerBlock = 4;
    constexpr int kThreads = kWarpsPerBlock * kWarpSize;
    dim3 block(kThreads);
    dim3 grid((N + kWarpsPerBlock - 1) / kWarpsPerBlock);

    mean_aggr_kernel_neighbor_parallel<scalar_t, kUseSecondAccess, kUseVectorizedLoads, kWarpsPerBlock>
        <<<grid, block, 0, stream>>>(edge_ptr, edge_idx, X, out, scratch, N, D);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
