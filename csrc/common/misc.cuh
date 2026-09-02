#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include <type_traits>

#ifndef FULL_WARP_MASK
#define FULL_WARP_MASK 0xffffffff
#endif

inline constexpr size_t kWarpSize          = 32;
inline constexpr size_t kMaxThreadsInBlock = 1024;

#if defined(__CUDA_ARCH__)
inline constexpr bool is_device_pass = true;
inline constexpr int kCudaArch       = __CUDA_ARCH__;
#else
inline constexpr bool is_device_pass = false;
inline constexpr int kCudaArch       = 0;
#endif

#ifdef CUDA_KERNEL_DEBUG
#define CUDA_KERNEL_CHECK()                                                    \
  do {                                                                         \
    cudaDeviceSynchronize();                                                   \
    C10_CUDA_KERNEL_LAUNCH_CHECK();                                            \
  } while (0)
#else
#define CUDA_KERNEL_CHECK() C10_CUDA_KERNEL_LAUNCH_CHECK()
#endif

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t error = call;                                                  \
    if (error != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(error));                                      \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// ============================================================================
// CUDA comparison operators -- pytorch disables them
// ============================================================================
#ifdef __CUDA_NO_HALF_OPERATORS__
__device__ __forceinline__ bool operator<(const __half a, const __half b) { return __hlt(a, b); }
__device__ __forceinline__ bool operator>(const __half a, const __half b) { return __hgt(a, b); }
__device__ __forceinline__ bool operator<=(const __half a, const __half b) { return __hle(a, b); }
__device__ __forceinline__ bool operator>=(const __half a, const __half b) { return __hge(a, b); }
__device__ __forceinline__ bool operator==(const __half a, const __half b) { return __heq(a, b); }
__device__ __forceinline__ bool operator!=(const __half a, const __half b) { return __hne(a, b); }
#endif

// Helper to extract typed pointer from tensor using untyped data_ptr()
// Uses void* cast to avoid PyTorch's scalar-type assertion that may
// not handle uint types correctly in all versions.
template <typename index_t>
index_t const *index_ptr(const at::Tensor& t) {
    return static_cast<index_t const *>(t.data_ptr());
}

template <typename index_t>
index_t *index_ptr_mut(at::Tensor& t) {
    return static_cast<index_t *>(t.data_ptr());
}

// Check whether a scalar type is a supported index type
inline bool is_supported_index_type(at::ScalarType type) {
    return type == at::kInt || type == at::kLong || type == c10::ScalarType::UInt32 || type == c10::ScalarType::UInt64;
}

// Raises the kernel's dynamic-shmem limit above the 48 KiB default when needed
// (large warps_per_block x pipeline_stages). Pass the kernel function itself, not a pointer.
template <typename KernelFn>
inline void ensure_dynamic_shmem(KernelFn kernel, size_t shmem, const char *context = "kernel launch") {
    constexpr size_t kStaticShmemLimit = 48 * 1024;
    if (shmem <= kStaticShmemLimit) return;

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    int max_shmem_optin = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&max_shmem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    TORCH_CHECK(
        shmem <= static_cast<size_t>(max_shmem_optin), context, ": requested shared memory (", shmem,
        " bytes) exceeds this GPU's max opt-in shared memory per block (", max_shmem_optin,
        " bytes). Reduce warps_per_block or pipeline_stages."
    );
    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shmem)));
}

// Warp reductions

template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T x) {
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(FULL_WARP_MASK, x, offset);
    }
    return x;
}

template <typename T>
__device__ __forceinline__ T warp_reduce_max(T x) {
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        x = max(x, __shfl_xor_sync(FULL_WARP_MASK, x, offset));
    }
    return x;
}

struct OnlineSoftmaxState {
    float max_val = -FLT_MAX;
    float sum_exp = 0.0f;

    __device__ float update(float logit) {
        float old_max = max_val;
        max_val       = fmaxf(max_val, logit);

        // correction factor for previous sum when max changes
        float correction = __expf(old_max - max_val);
        sum_exp          = sum_exp * correction + __expf(logit - max_val);

        return correction;
    }

    __device__ float get_alpha(float logit) const { return __expf(logit - max_val) / sum_exp; }

    // FlashAttention logsumexp trick
    static __device__ __forceinline__ float recompute_alpha(
        float e_ij,  // logit
        float L_i    // saved log-sum-exp
    ) {
        return __expf(e_ij - L_i);
    }
};

// =============================================================================
// ReductionOps<Op> — compile-time traits for min/max reduction kernels
// =============================================================================

enum class ReductionOp { MIN, MAX };

template <ReductionOp Op>
struct ReductionOps;

template <>
struct ReductionOps<ReductionOp::MIN> {
    static constexpr float IDENTITY           = INFINITY;  // +inf
    static constexpr uint64_t PACKED_IDENTITY = 0xff800000ffffffffULL;

    template <typename cuda_t>
    static __device__ __forceinline__ bool is_better(cuda_t a, cuda_t b) {
        return a < b;
    }

    static __device__ __forceinline__ bool is_better_f(float a, float b) { return a < b; }

    static __device__ __forceinline__ uint64_t atomic_reduce(uint64_t *addr, uint64_t val) {
        return atomicMin(reinterpret_cast<unsigned long long *>(addr), val);
    }
};

template <>
struct ReductionOps<ReductionOp::MAX> {
    static constexpr float IDENTITY           = -INFINITY;  // -inf
    static constexpr uint64_t PACKED_IDENTITY = 0x007fffffffffffffULL;

    template <typename cuda_t>
    static __device__ __forceinline__ bool is_better(cuda_t a, cuda_t b) {
        return a > b;
    }

    static __device__ __forceinline__ bool is_better_f(float a, float b) { return a > b; }

    static __device__ __forceinline__ uint64_t atomic_reduce(uint64_t *addr, uint64_t val) {
        return atomicMax(reinterpret_cast<unsigned long long *>(addr), val);
    }
};

template <size_t bytes>
inline consteval auto deduce_uint_type() {
    if constexpr (bytes == 1) {
        return uint8_t{};
    } else if constexpr (bytes == 2) {
        return uint16_t{};
    } else if constexpr (bytes == 4) {
        return uint32_t{};
    } else if constexpr (bytes == 8) {
        return uint64_t{};
    } else if constexpr (bytes == 16) {
        return static_cast<unsigned __int128>(0);
    } else {
        __builtin_unreachable();
    }
}

template <size_t bytes>
using deduce_uint_type_t = decltype(deduce_uint_type<bytes>());

template <typename T>
    requires(
        std::is_same_v<std::remove_cvref_t<T>, half> || std::is_same_v<std::remove_cvref_t<T>, nv_bfloat16> ||
        std::is_same_v<std::remove_cvref_t<T>, float>
    )
inline consteval auto deduce_packed_type() {
    if constexpr (std::is_same_v<std::remove_cvref_t<T>, half>) {
        return half2{};
    } else if constexpr (std::is_same_v<std::remove_cvref_t<T>, nv_bfloat16>) {
        return nv_bfloat162{};
    } else if constexpr (std::is_same_v<std::remove_cvref_t<T>, float>) {
        return float2{};
    } else {
        __builtin_unreachable();
    }
}

template <typename f16>
using deduce_packed_type_t = decltype(deduce_packed_type<f16>());
