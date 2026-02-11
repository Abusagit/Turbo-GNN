#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <torch/torch.h>
#include <cstddef>
#include <cfloat>
#include <variant>
#include <cmath>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <iostream>
#include <vector>
#include <iomanip>
#include <algorithm>
#include <random>

#ifdef CUDA_KERNEL_DEBUG
    #define CUDA_KERNEL_CHECK() do { \
        cudaDeviceSynchronize(); \
        C10_CUDA_KERNEL_LAUNCH_CHECK(); \
    } while(0)
#else
    #define CUDA_KERNEL_CHECK() C10_CUDA_KERNEL_LAUNCH_CHECK()
#endif


#ifndef FULL_WARP_MASK
#define FULL_WARP_MASK 0xffffffff
#endif

#ifndef kWarpSize
constexpr int kWarpSize = 32;
#endif

#ifndef kMaxThreadsInWarp
constexpr int kMaxThreadsInWarp = 32;
#endif


#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)


// ============================================================================
// CUDA comparison operators -- pytorch disables them
// ============================================================================
#ifdef __CUDA_NO_HALF_OPERATORS__
__device__ __forceinline__ bool operator<(const __half& a, const __half& b) {
    return __hlt(a, b);
}

__device__ __forceinline__ bool operator>(const __half& a, const __half& b) {
    return __hgt(a, b);
}

__device__ __forceinline__ bool operator<=(const __half& a, const __half& b) {
    return __hle(a, b);
}

__device__ __forceinline__ bool operator>=(const __half& a, const __half& b) {
    return __hge(a, b);
}

__device__ __forceinline__ bool operator==(const __half& a, const __half& b) {
    return __heq(a, b);
}

__device__ __forceinline__ bool operator!=(const __half& a, const __half& b) {
    return __hne(a, b);
}
#endif


// Dispatch and datatype Traits TODO move to the separate file in the final version

template <typename T>
struct TTypeTraits;

// Spec for float
template <>
struct TTypeTraits<float> {
    using TorchType = float;
    using CudaType = float;
    static constexpr c10::ScalarType ScalarType = c10::ScalarType::Float;
};

// Spec for double
template <>
struct TTypeTraits<double> {
    using TorchType = double;
    using CudaType = double;
    static constexpr c10::ScalarType ScalarType = c10::ScalarType::Double;
};

// Spec for at::Half
template <>
struct TTypeTraits<at::Half> {
    using TorchType = at::Half;
    using CudaType = __half;
    static constexpr c10::ScalarType ScalarType = c10::ScalarType::Half;
};

// Spec for at::BFloat16
template <>
struct TTypeTraits<at::BFloat16> {
    using TorchType = at::BFloat16;
    using CudaType = __nv_bfloat16;
    static constexpr c10::ScalarType ScalarType = c10::ScalarType::BFloat16;
};

// Helper for obtaining CUDA type from PyTorch  type
template <typename TorchT>
using ToCudaType = typename TTypeTraits<TorchT>::CudaType;


template <int... Values>
std::variant<std::integral_constant<int, Values>...> MakeIntVariant(int value) {
    std::variant<std::integral_constant<int, Values>...> result;
    bool found = false;
    ([&] {
        if (value == Values) {
            result.template emplace<std::integral_constant<int, Values>>();
            found = true;
        }
    }(), ...);
    if (!found) {
        throw std::runtime_error("Wrong int value: " + std::to_string(value));
    }
    return result;
}

template <typename T>
struct TTypeInfo {
    using Traits = TTypeTraits<T>;
    using TorchType = typename Traits::TorchType;
    using CudaType = typename Traits::CudaType;
    static constexpr c10::ScalarType ScalarType = Traits::ScalarType;
};

template <typename... T>
inline std::variant<TTypeInfo<T>...> MakeTypeVariant(at::ScalarType type) {
    std::variant<TTypeInfo<T>...> result;
    bool found = false;
    ([&] {
        if (TTypeInfo<T>::ScalarType == type) {
            result.template emplace<TTypeInfo<T>>();
            found = true;
        }
    }(), ...);
    if (!found) {
        throw std::runtime_error("Unsupported scalar type");
    }
    return result;
}

template <typename cuda_t>
__device__ __forceinline__ cuda_t make_cuda_value(float val);

template <>
__device__ __forceinline__ float make_cuda_value<float>(float val) {
    return val;
}

template <>
__device__ __forceinline__ double make_cuda_value<double>(float val) {
    return static_cast<double>(val);
}

template <>
__device__ __forceinline__ __half make_cuda_value<__half>(float val) {
    return __float2half(val);
}

template <>
__device__ __forceinline__ __nv_bfloat16 make_cuda_value<__nv_bfloat16>(float val) {
    return __float2bfloat16(val);
}


// CUDA type --> float
template <typename cuda_t>
__device__ __forceinline__ float cuda_to_float(cuda_t val);

template <>
__device__ __forceinline__ float cuda_to_float<float>(float val) {
    return val;
}

template <>
__device__ __forceinline__ float cuda_to_float<double>(double val) {
    return static_cast<float>(val);
}

template <>
__device__ __forceinline__ float cuda_to_float<__half>(__half val) {
    return __half2float(val);
}

template <>
__device__ __forceinline__ float cuda_to_float<__nv_bfloat16>(__nv_bfloat16 val) {
    return __bfloat162float(val);
}


// Vec2 instructions

template <typename cuda_t>
struct Vec2 {
    cuda_t x, y;
};

template <typename cuda_t>
__device__ __forceinline__ Vec2<cuda_t> load_vec2(const cuda_t* ptr) {
    static_assert(sizeof(cuda_t) == 2, "Vec2 only for 16-bit types");
    uint32_t data = *reinterpret_cast<const uint32_t*>(ptr);
    Vec2<cuda_t> result;
    result.x = reinterpret_cast<const cuda_t*>(&data)[0];
    result.y = reinterpret_cast<const cuda_t*>(&data)[1];
    return result;
}

template <typename cuda_t>
__device__ __forceinline__ void store_vec2(cuda_t* ptr, Vec2<cuda_t> val) {
    static_assert(sizeof(cuda_t) == 2, "Vec2 only for 16-bit types");
    uint32_t data;
    reinterpret_cast<cuda_t*>(&data)[0] = val.x;
    reinterpret_cast<cuda_t*>(&data)[1] = val.y;
    *reinterpret_cast<uint32_t*>(ptr) = data;
}



// Warp reductions


__device__ __forceinline__ float warp_reduce_sum(float x) {
    #pragma unroll
    for (int offset = kMaxThreadsInWarp / 2; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(FULL_WARP_MASK, x, offset);
    }
    return x;
}

__device__ __forceinline__ float warp_reduce_max(float x) {
    #pragma unroll
    for (int offset = kMaxThreadsInWarp / 2; offset > 0; offset >>= 1) {
        x = fmaxf(x, __shfl_xor_sync(FULL_WARP_MASK, x, offset));
    }
    return x;
}

__device__ __forceinline__ int warp_bcast_i32(int v0) {
    return __shfl_sync(FULL_WARP_MASK, v0, 0);
}

__device__ __forceinline__ float warp_bcast_f32(float v0) {
    return __shfl_sync(FULL_WARP_MASK, v0, 0);
}

__device__ __forceinline__ float dot_product_f4(float4 a, float4 b) {
    float acc = 0.f;
    acc = fmaf(a.x, b.x, acc);
    acc = fmaf(a.y, b.y, acc);
    acc = fmaf(a.z, b.z, acc);
    acc = fmaf(a.w, b.w, acc);
    return acc;
}

struct OnlineSoftmaxState {
    float max_val;
    float sum_exp;

    __device__ __forceinline__ OnlineSoftmaxState() : max_val(-FLT_MAX), sum_exp(0.0f) {}

    __device__ __forceinline__ float update(float logit) {
        float old_max = max_val;
        max_val = fmaxf(max_val, logit);

        // correction factor for previous sum when max changes
        float correction = __expf(old_max - max_val);
        sum_exp = sum_exp * correction + __expf(logit - max_val);
        return correction;
    }

    __device__ __forceinline__ float get_alpha(float logit) const {
        return __expf(logit - max_val) / sum_exp;
    }
};
