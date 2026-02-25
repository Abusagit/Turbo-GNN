#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
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
#include <cuda_runtime_api.h>
#include <ATen/cuda/CUDAContext.h>

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


// Vec2Ops: type-generic packed operations for 16-bit types

template <typename cuda_t>
struct Vec2Ops;

template <>
struct Vec2Ops<__half> {
    using vec2_t = __half2;
    static __device__ __forceinline__ vec2_t get_zero() { return __float2half2_rn(0.0f); }
    static __device__ __forceinline__ vec2_t add(vec2_t a, vec2_t b) { return __hadd2(a, b); }
    static __device__ __forceinline__ vec2_t mul(vec2_t a, vec2_t b) { return __hmul2(a, b); }
    static __device__ __forceinline__ vec2_t from_float(float v) { return __float2half2_rn(v); }
    static __device__ __forceinline__ vec2_t max2(vec2_t a, vec2_t b) { return __hmax2(a, b); }
    static __device__ __forceinline__ vec2_t min2(vec2_t a, vec2_t b) { return __hmin2(a, b); }
    static __device__ __forceinline__ float2 to_float2(vec2_t v) { return __half22float2(v); }
    static __device__ __forceinline__ vec2_t from_float2(float2 v) { return __float22half2_rn(v); }
    static __device__ __forceinline__ vec2_t fma(vec2_t a, vec2_t b, vec2_t c) {
        return __hfma2(a, b, c);
    }
    static __device__ __forceinline__ vec2_t leaky_relu(vec2_t x, vec2_t neg_slope) {
        vec2_t z = get_zero();
        return __hadd2(__hmax2(x, z), __hmul2(neg_slope, __hmin2(x, z)));
    }
};

template <>
struct Vec2Ops<__nv_bfloat16> {
    using vec2_t = __nv_bfloat162;
    static __device__ __forceinline__ vec2_t get_zero() { return __float2bfloat162_rn(0.0f); }
    static __device__ __forceinline__ vec2_t add(vec2_t a, vec2_t b) { return __hadd2(a, b); }
    static __device__ __forceinline__ vec2_t mul(vec2_t a, vec2_t b) { return __hmul2(a, b); }
    static __device__ __forceinline__ vec2_t from_float(float v) { return __float2bfloat162_rn(v); }
    static __device__ __forceinline__ vec2_t max2(vec2_t a, vec2_t b) { return __hmax2(a, b); }
    static __device__ __forceinline__ vec2_t min2(vec2_t a, vec2_t b) { return __hmin2(a, b); }
    static __device__ __forceinline__ float2 to_float2(vec2_t v) { return __bfloat1622float2(v); }
    static __device__ __forceinline__ vec2_t from_float2(float2 v) { return __float22bfloat162_rn(v); }
    static __device__ __forceinline__ vec2_t fma(vec2_t a, vec2_t b, vec2_t c) {
        return __hfma2(a, b, c);
    }
    static __device__ __forceinline__ vec2_t leaky_relu(vec2_t x, vec2_t neg_slope) {
        vec2_t z = get_zero();
        return __hadd2(__hmax2(x, z), __hmul2(neg_slope, __hmin2(x, z)));
    }
};


// Vec8: 128-bit load/store for 16-bit types = 8 scalars = 4 vec2 pairs

template <typename cuda_t>
struct Vec8 {
    static_assert(sizeof(cuda_t) == 2, "Vec8 only for 16-bit types");
    using vec2_t = typename Vec2Ops<cuda_t>::vec2_t;
    vec2_t v[4];
};

template <typename cuda_t>
__device__ __forceinline__ Vec8<cuda_t> load_vec8(const cuda_t* ptr) {
    static_assert(sizeof(cuda_t) == 2, "Vec8 only for 16-bit types");
    using vec2_t = typename Vec2Ops<cuda_t>::vec2_t;
    // Single 128-bit transaction
    float4 raw = *reinterpret_cast<const float4*>(ptr);
    Vec8<cuda_t> result;
    const vec2_t* v2 = reinterpret_cast<const vec2_t*>(&raw);
    result.v[0] = v2[0];
    result.v[1] = v2[1];
    result.v[2] = v2[2];
    result.v[3] = v2[3];
    return result;
}

template <typename cuda_t>
__device__ __forceinline__ void store_vec8(cuda_t* ptr, const Vec8<cuda_t>& val) {
    static_assert(sizeof(cuda_t) == 2, "Vec8 only for 16-bit types");
    float4 raw;
    using vec2_t = typename Vec2Ops<cuda_t>::vec2_t;
    vec2_t* v2 = reinterpret_cast<vec2_t*>(&raw);
    v2[0] = val.v[0];
    v2[1] = val.v[1];
    v2[2] = val.v[2];
    v2[3] = val.v[3];
    *reinterpret_cast<float4*>(ptr) = raw;
}


// Warp reductions


template<typename T>
__device__ __forceinline__ T warp_reduce_sum(T x) {
    #pragma unroll
    for (int offset = kMaxThreadsInWarp / 2; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(FULL_WARP_MASK, x, offset);
    }
    return x;
}

template<typename T>
__device__ __forceinline__ T warp_reduce_max(T x) {
    #pragma unroll
    for (int offset = kMaxThreadsInWarp / 2; offset > 0; offset >>= 1) {
        x = max(x, __shfl_xor_sync(FULL_WARP_MASK, x, offset));
    }
    return x;
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

// FlashAttention logsumexp trick
__device__ __forceinline__ float recompute_alpha(
    float e_ij,          // logit
    float L_i            // saved log-sum-exp
) {
    return __expf(e_ij - L_i);
}

__device__ __forceinline__ float dot_product_f4(float4 a, float4 b) {
    float acc = 0.f;
    acc = fmaf(a.x, b.x, acc);
    acc = fmaf(a.y, b.y, acc);
    acc = fmaf(a.z, b.z, acc);
    acc = fmaf(a.w, b.w, acc);
    return acc;
}

__device__ __forceinline__ float leaky_relu_elementwise(float x, float negative_slope) {
    return  (x > 0.0f) ? x : negative_slope * x;
}

__device__ __forceinline__ float leaky_relu_der_elementwise(float x, float negative_slope) {
    return (x > 0.0f) ? 1.0f : negative_slope;
}


__device__ __forceinline__ float4 f4_leaky_relu_der(float4 edge, float ns) {
    return make_float4(
        leaky_relu_der_elementwise(edge.x, ns),
        leaky_relu_der_elementwise(edge.y, ns),
        leaky_relu_der_elementwise(edge.z, ns),
        leaky_relu_der_elementwise(edge.w, ns)
    );
}

__device__ __forceinline__ float4 f4_add(float4 a, float4 b) {
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__device__ __forceinline__ void f4_fma(float4& acc, float s, float4 v) {
    acc.x = fmaf(s, v.x, acc.x);
    acc.y = fmaf(s, v.y, acc.y);
    acc.z = fmaf(s, v.z, acc.z);
    acc.w = fmaf(s, v.w, acc.w);
}

__device__ __forceinline__ void f4_fma_vec(float4& acc, float4 s, float4 v) {
    acc.x = fmaf(s.x, v.x, acc.x);
    acc.y = fmaf(s.y, v.y, acc.y);
    acc.z = fmaf(s.z, v.z, acc.z);
    acc.w = fmaf(s.w, v.w, acc.w);
}

__device__ __forceinline__ float4 f4_mul(float4 a, float4 b) {
    return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}

// =============================================================================
// TileOps<VW, cuda_t> — vectorized load/compute/store traits for forward kernel
// =============================================================================

// template (undefined)
template<int VW, typename cuda_t>
struct TileOps;

// --- VW=1, float: scalar loads ---
template<>
struct TileOps<1, float> {
    using vec_t = float;
    using ns_t  = float;
    static constexpr int ELEM_PER_VEC = 1;
    static constexpr vec_t zero = 0.0f;

    static __device__ __forceinline__ vec_t load(const float* ptr, int vec_idx) {
        return __ldg(&ptr[vec_idx]);
    }
    static __device__ __forceinline__ ns_t make_ns(float ns) { return ns; }

    static __device__ __forceinline__ float gatv2_dot_leaky_relu(vec_t l, vec_t r, vec_t a, ns_t ns) {
        float s = leaky_relu_elementwise(l + r, ns);
        return s * a;
    }
    static __device__ __forceinline__ float dot_product(vec_t a, vec_t b) {
        return a * b;
    }
    static __device__ __forceinline__ void weighted_accum(float* acc, float w, vec_t r) {
        acc[0] = fmaf(w, r, acc[0]);
    }
    static __device__ __forceinline__ void gatv2_accum_grad_al(float* ga, float* gl, float ge, vec_t l, vec_t r, vec_t a, float ns) {
        float edge = l + r;
        float tder = leaky_relu_der_elementwise(edge, ns);
        float t_ij = tder * edge;
        ga[0] = fmaf(ge, t_ij, ga[0]);
        gl[0] = fmaf(ge * tder, a, gl[0]);
    }
    static __device__ __forceinline__ void gatv2_accum_grad_r(float* gr, float alpha, vec_t gh, float ge, vec_t l, vec_t r, vec_t a, float ns) {
        float edge = l + r;
        float tder = leaky_relu_der_elementwise(edge, ns);
        gr[0] = fmaf(alpha, gh, gr[0]);
        gr[0] = fmaf(ge * tder, a, gr[0]);
    }
    static __device__ __forceinline__ void write(float* out, int vec_idx, const float* acc, float inv_sum) {
        out[vec_idx] = acc[0] * inv_sum;
    }
    static __device__ __forceinline__ void write_typed(float* out, int vec_idx, const float* acc) {
        out[vec_idx] = acc[0];
    }
    static __device__ __forceinline__ void write_float(float* out, int vec_idx, const float* acc) {
        out[vec_idx] = acc[0];
    }
    static __device__ __forceinline__ void write_zero(float* out, int vec_idx) {
        out[vec_idx] = zero;
    }

    // --- generic element access ---
    static __device__ __forceinline__ float extract(vec_t v, int /*i*/) { return v; }
    static __device__ __forceinline__ float extract_float(vec_t v, int /*i*/) { return v; }
    static __device__ __forceinline__ void store_vec(float* ptr, int vec_idx, vec_t v) {
        ptr[vec_idx] = v;
    }
    static __device__ __forceinline__ vec_t build(const float* arr) { return arr[0]; }
    static __device__ __forceinline__ vec_t build_from_float(const float* arr) { return arr[0]; }

    // --- GT backward: typed atomic add of scalar * vec ---
    static __device__ __forceinline__ void gt_atomic_add_scaled(
        float* ptr, int base_f, float scalar, vec_t v
    ) {
        atomicAdd(&ptr[base_f], scalar * v);
    }
};

// --- VW=4, float: float4 loads ---
template<>
struct TileOps<4, float> {
    using vec_t = float4;
    using ns_t  = float;
    static constexpr int ELEM_PER_VEC = 4;
    static constexpr vec_t zero = {0.f, 0.f, 0.f, 0.f};

    static __device__ __forceinline__ vec_t load(const float* ptr, int vec_idx) {
        return reinterpret_cast<const float4*>(ptr)[vec_idx];
    }
    static __device__ __forceinline__ ns_t make_ns(float ns) { return ns; }

    static __device__ __forceinline__ float gatv2_dot_leaky_relu(vec_t l, vec_t r, vec_t a, ns_t ns) {
        float4 sum = make_float4(
            leaky_relu_elementwise(l.x + r.x, ns),
            leaky_relu_elementwise(l.y + r.y, ns),
            leaky_relu_elementwise(l.z + r.z, ns),
            leaky_relu_elementwise(l.w + r.w, ns)
        );
            return dot_product_f4(sum, a);
    }
    static __device__ __forceinline__ float dot_product(vec_t a, vec_t b) {
        return dot_product_f4(a, b);
    }
    static __device__ __forceinline__ void weighted_accum(float* acc, float w, vec_t r) {
        acc[0] = fmaf(w, r.x, acc[0]);
        acc[1] = fmaf(w, r.y, acc[1]);
        acc[2] = fmaf(w, r.z, acc[2]);
        acc[3] = fmaf(w, r.w, acc[3]);
    }
    static __device__ __forceinline__ void gatv2_accum_grad_al(float* ga, float* gl, float ge, vec_t l, vec_t r, vec_t a, float ns) {
        float4 edge = f4_add(l, r);
        float4 tder = f4_leaky_relu_der(edge, ns);
        float4 t_ij = f4_mul(tder, edge);
        f4_fma(*(float4*)ga, ge, t_ij);
        f4_fma(*(float4*)gl, ge, f4_mul(tder, a));
    }
    static __device__ __forceinline__ void gatv2_accum_grad_r(float* gr, float alpha, vec_t gh, float ge, vec_t l, vec_t r, vec_t a, float ns) {
        float4 edge = f4_add(l, r);
        float4 tder = f4_leaky_relu_der(edge, ns);
        f4_fma(*(float4*)gr, alpha, gh);
        f4_fma(*(float4*)gr, ge, f4_mul(tder, a));
    }
    static __device__ __forceinline__ void write(float* out, int vec_idx, const float* acc, float inv_sum) {
        reinterpret_cast<float4*>(out)[vec_idx] = make_float4(
            acc[0] * inv_sum, acc[1] * inv_sum,
            acc[2] * inv_sum, acc[3] * inv_sum);
    }
    static __device__ __forceinline__ void write_typed(float* out, int vec_idx, const float* acc) {
        reinterpret_cast<float4*>(out)[vec_idx] = make_float4(acc[0], acc[1], acc[2], acc[3]);
    }
    static __device__ __forceinline__ void write_float(float* out, int vec_idx, const float* acc) {
        reinterpret_cast<float4*>(out)[vec_idx] = make_float4(acc[0], acc[1], acc[2], acc[3]);
    }
    static __device__ __forceinline__ void write_zero(float* out, int vec_idx) {
        reinterpret_cast<float4*>(out)[vec_idx] = zero;
    }

    // --- generic element access ---
    static __device__ __forceinline__ float extract(vec_t v, int i) {
        return (&v.x)[i];
    }
    static __device__ __forceinline__ float extract_float(vec_t v, int i) {
        return (&v.x)[i];
    }
    static __device__ __forceinline__ void store_vec(float* ptr, int vec_idx, vec_t v) {
        reinterpret_cast<float4*>(ptr)[vec_idx] = v;
    }
    static __device__ __forceinline__ vec_t build(const float* arr) {
        return {arr[0], arr[1], arr[2], arr[3]};
    }
    static __device__ __forceinline__ vec_t build_from_float(const float* arr) {
        return {arr[0], arr[1], arr[2], arr[3]};
    }

    // --- GT backward: typed atomic add of scalar * vec ---
    static __device__ __forceinline__ void gt_atomic_add_scaled(
        float* ptr, int base_f, float scalar, vec_t v
    ) {
        atomicAdd(&ptr[base_f + 0], scalar * v.x);
        atomicAdd(&ptr[base_f + 1], scalar * v.y);
        atomicAdd(&ptr[base_f + 2], scalar * v.z);
        atomicAdd(&ptr[base_f + 3], scalar * v.w);
    }
};

// --- VW=2, half/bf16: vec2 loads ---
template<typename cuda_t>
struct TileOps<2, cuda_t> {
    using Ops   = Vec2Ops<cuda_t>;
    using vec2_t = typename Ops::vec2_t;
    using vec_t = vec2_t;
    using ns_t  = vec2_t;
    static constexpr int ELEM_PER_VEC = 2;

    static __device__ __forceinline__ vec2_t get_zero() { return Ops::from_float(0.0f); }

    static __device__ __forceinline__ vec_t load(const cuda_t* ptr, int vec_idx) {
        return *reinterpret_cast<const vec2_t*>(&ptr[vec_idx * ELEM_PER_VEC]);
    }
    static __device__ __forceinline__ ns_t make_ns(float ns) { return Ops::from_float(ns); }

    static __device__ __forceinline__ float gatv2_dot_leaky_relu(vec_t l, vec_t r, vec_t a, ns_t ns) {
        vec2_t sum = Ops::add(l, r);
        vec2_t act = Ops::leaky_relu(sum, ns);
        vec2_t prod = Ops::mul(act, a);
        float2 pf = Ops::to_float2(prod); // TODO maybe do it in vec2_t and than cast to float after the summation?
        return pf.x + pf.y;
    }
    static __device__ __forceinline__ float dot_product(vec_t a, vec_t b) {
        vec2_t prod = Ops::mul(a, b);
        float2 pf = Ops::to_float2(prod);
        return pf.x + pf.y;
    }
    static __device__ __forceinline__ void weighted_accum(float* acc, float w, vec_t r) {
        // maybe cast weight to vec_t?
        float2 rf = Ops::to_float2(r);
        acc[0] = fmaf(w, rf.x, acc[0]);
        acc[1] = fmaf(w, rf.y, acc[1]);
    }
    static __device__ __forceinline__ void gatv2_accum_grad_al(float* ga, float* gl, float ge, vec_t l, vec_t r, vec_t a, float ns) {
        float2 lf = Ops::to_float2(l);
        float2 rf = Ops::to_float2(r);
        float2 af = Ops::to_float2(a);
        float edge0 = lf.x + rf.x;
        float edge1 = lf.y + rf.y;
        float tder0 = leaky_relu_der_elementwise(edge0, ns);
        float tder1 = leaky_relu_der_elementwise(edge1, ns);
        ga[0] = fmaf(ge, tder0 * edge0, ga[0]);
        ga[1] = fmaf(ge, tder1 * edge1, ga[1]);
        gl[0] = fmaf(ge * tder0, af.x, gl[0]);
        gl[1] = fmaf(ge * tder1, af.y, gl[1]);
    }
    static __device__ __forceinline__ void gatv2_accum_grad_r(float* gr, float alpha, vec_t gh, float ge, vec_t l, vec_t r, vec_t a, float ns) {
        float2 lf = Ops::to_float2(l);
        float2 rf = Ops::to_float2(r);
        float2 af = Ops::to_float2(a);
        float2 ghf = Ops::to_float2(gh);
        float edge0 = lf.x + rf.x;
        float edge1 = lf.y + rf.y;
        float tder0 = leaky_relu_der_elementwise(edge0, ns);
        float tder1 = leaky_relu_der_elementwise(edge1, ns);
        gr[0] = fmaf(alpha, ghf.x, gr[0]);
        gr[0] = fmaf(ge * tder0, af.x, gr[0]);
        gr[1] = fmaf(alpha, ghf.y, gr[1]);
        gr[1] = fmaf(ge * tder1, af.y, gr[1]);
    }
    static __device__ __forceinline__ void write(cuda_t* out, int vec_idx, const float* acc, float inv_sum) {
        float2 of = make_float2(acc[0] * inv_sum, acc[1] * inv_sum);
        *reinterpret_cast<vec2_t*>(&out[vec_idx * 2]) = Ops::from_float2(of);
    }
    static __device__ __forceinline__ void write_typed(cuda_t* out, int vec_idx, const float* acc) {
        *reinterpret_cast<vec2_t*>(&out[vec_idx * 2]) = Ops::from_float2(make_float2(acc[0], acc[1]));
    }
    static __device__ __forceinline__ void write_float(float* out, int vec_idx, const float* acc) {
        *reinterpret_cast<float2*>(&out[vec_idx * 2]) = make_float2(acc[0], acc[1]);
    }
    static __device__ __forceinline__ void write_zero(cuda_t* out, int vec_idx) {
        *reinterpret_cast<vec2_t*>(&out[vec_idx * 2]) = get_zero();
    }

    // --- generic element access ---
    static __device__ __forceinline__ cuda_t extract(vec_t v, int i) {
        return reinterpret_cast<const cuda_t*>(&v)[i];
    }
    static __device__ __forceinline__ float extract_float(vec_t v, int i) {
        return cuda_to_float(reinterpret_cast<const cuda_t*>(&v)[i]);
    }
    static __device__ __forceinline__ void store_vec(cuda_t* ptr, int vec_idx, vec_t v) {
        *reinterpret_cast<vec2_t*>(&ptr[vec_idx * ELEM_PER_VEC]) = v;
    }
    static __device__ __forceinline__ vec_t build(const cuda_t* arr) {
        return *reinterpret_cast<const vec2_t*>(arr);
    }
    static __device__ __forceinline__ vec_t build_from_float(const float* arr) {
        return Ops::from_float2(make_float2(arr[0], arr[1]));
    }

    // --- GT backward: typed atomic add of scalar * vec ---
    static __device__ __forceinline__ void gt_atomic_add_scaled(
        cuda_t* ptr, int base_f, float scalar, vec_t v
    ) {
        vec2_t sv = Ops::from_float(scalar);
        vec2_t prod = Ops::mul(sv, v);
        const cuda_t* elems = reinterpret_cast<const cuda_t*>(&prod);
        atomicAdd(&ptr[base_f],     elems[0]);
        atomicAdd(&ptr[base_f + 1], elems[1]);
    }
};

// --- VW=8, half/bf16: Vec8 (128-bit) loads ---
template<typename cuda_t>
struct TileOps<8, cuda_t> {
    using Ops    = Vec2Ops<cuda_t>;
    using vec2_t = typename Ops::vec2_t;
    using vec_t  = Vec8<cuda_t>;
    using ns_t   = vec2_t;
    static constexpr int ELEM_PER_VEC = 8;
    static constexpr float4 zero_bits = {0.f, 0.f, 0.f, 0.f};

    static __device__ __forceinline__ vec_t load(const cuda_t* ptr, int vec_idx) {
        return load_vec8(&ptr[vec_idx * ELEM_PER_VEC]);
    }
    static __device__ __forceinline__ ns_t make_ns(float ns) { return Ops::from_float(ns); }

    static __device__ __forceinline__ float gatv2_dot_leaky_relu(vec_t l, vec_t r, vec_t a, ns_t ns) {
        float dot = 0.0f;
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            vec2_t sum = Ops::add(l.v[p], r.v[p]);
            vec2_t act = Ops::leaky_relu(sum, ns);
            vec2_t prod = Ops::mul(act, a.v[p]);
            float2 pf = Ops::to_float2(prod);
            dot += pf.x + pf.y;
        }
        return dot;
    }
    static __device__ __forceinline__ float dot_product(vec_t a, vec_t b) {
        float dot = 0.0f;
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            vec2_t prod = Ops::mul(a.v[p], b.v[p]);
            float2 pf = Ops::to_float2(prod);
            dot += pf.x + pf.y;
        }
        return dot;
    }
    static __device__ __forceinline__ void weighted_accum(float* acc, float w, vec_t r) {
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            float2 rf = Ops::to_float2(r.v[p]);
            acc[p * 2]     = fmaf(w, rf.x, acc[p * 2]);
            acc[p * 2 + 1] = fmaf(w, rf.y, acc[p * 2 + 1]);
        }
    }
    static __device__ __forceinline__ void gatv2_accum_grad_al(float* ga, float* gl, float ge, vec_t l, vec_t r, vec_t a, float ns) {
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            float2 lf = Ops::to_float2(l.v[p]);
            float2 rf = Ops::to_float2(r.v[p]);
            float2 af = Ops::to_float2(a.v[p]);
            float edge0 = lf.x + rf.x;
            float edge1 = lf.y + rf.y;
            float tder0 = leaky_relu_der_elementwise(edge0, ns);
            float tder1 = leaky_relu_der_elementwise(edge1, ns);
            ga[p * 2]     = fmaf(ge, tder0 * edge0, ga[p * 2]);
            ga[p * 2 + 1] = fmaf(ge, tder1 * edge1, ga[p * 2 + 1]);
            gl[p * 2]     = fmaf(ge * tder0, af.x, gl[p * 2]);
            gl[p * 2 + 1] = fmaf(ge * tder1, af.y, gl[p * 2 + 1]);
        }
    }
    static __device__ __forceinline__ void gatv2_accum_grad_r(float* gr, float alpha, vec_t gh, float ge, vec_t l, vec_t r, vec_t a, float ns) {
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            float2 lf = Ops::to_float2(l.v[p]);
            float2 rf = Ops::to_float2(r.v[p]);
            float2 af = Ops::to_float2(a.v[p]);
            float2 ghf = Ops::to_float2(gh.v[p]);
            float edge0 = lf.x + rf.x;
            float edge1 = lf.y + rf.y;
            float tder0 = leaky_relu_der_elementwise(edge0, ns);
            float tder1 = leaky_relu_der_elementwise(edge1, ns);
            gr[p * 2]     = fmaf(alpha, ghf.x, gr[p * 2]);
            gr[p * 2]     = fmaf(ge * tder0, af.x, gr[p * 2]);
            gr[p * 2 + 1] = fmaf(alpha, ghf.y, gr[p * 2 + 1]);
            gr[p * 2 + 1] = fmaf(ge * tder1, af.y, gr[p * 2 + 1]);
        }
    }
    static __device__ __forceinline__ void write(cuda_t* out, int vec_idx, const float* acc, float inv_sum) {
        Vec8<cuda_t> out_v8;
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            out_v8.v[p] = Ops::from_float2(make_float2(
                acc[p * 2] * inv_sum, acc[p * 2 + 1] * inv_sum));
        }
        store_vec8(&out[vec_idx * 8], out_v8);
    }
    static __device__ __forceinline__ void write_typed(cuda_t* out, int vec_idx, const float* acc) {
        Vec8<cuda_t> out_v8;
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            out_v8.v[p] = Ops::from_float2(make_float2(acc[p * 2], acc[p * 2 + 1]));
        }
        store_vec8(&out[vec_idx * 8], out_v8);
    }
    static __device__ __forceinline__ void write_float(float* out, int vec_idx, const float* acc) {
        reinterpret_cast<float4*>(&out[vec_idx * 8])[0] = make_float4(acc[0], acc[1], acc[2], acc[3]);
        reinterpret_cast<float4*>(&out[vec_idx * 8])[1] = make_float4(acc[4], acc[5], acc[6], acc[7]);
    }
    static __device__ __forceinline__ void write_zero(cuda_t* out, int vec_idx) {
        *reinterpret_cast<float4*>(&out[vec_idx * 8]) = zero_bits;
    }

    // --- generic element access ---
    static __device__ __forceinline__ cuda_t extract(vec_t v, int i) {
        int pair = i / 2, elem = i % 2;
        return reinterpret_cast<const cuda_t*>(&v.v[pair])[elem];
    }
    static __device__ __forceinline__ float extract_float(vec_t v, int i) {
        int pair = i / 2, elem = i % 2;
        return cuda_to_float(reinterpret_cast<const cuda_t*>(&v.v[pair])[elem]);
    }
    static __device__ __forceinline__ void store_vec(cuda_t* ptr, int vec_idx, vec_t v) {
        store_vec8(&ptr[vec_idx * ELEM_PER_VEC], v);
    }
    static __device__ __forceinline__ vec_t build(const cuda_t* arr) {
        vec_t result;
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            result.v[p] = *reinterpret_cast<const vec2_t*>(&arr[p * 2]);
        }
        return result;
    }
    static __device__ __forceinline__ vec_t build_from_float(const float* arr) {
        vec_t result;
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            result.v[p] = Ops::from_float2(make_float2(arr[p * 2], arr[p * 2 + 1]));
        }
        return result;
    }

    // --- GT backward: typed atomic add of scalar * vec ---
    static __device__ __forceinline__ void gt_atomic_add_scaled(
        cuda_t* ptr, int base_f, float scalar, vec_t v
    ) {
        vec2_t sv = Ops::from_float(scalar);
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            vec2_t prod = Ops::mul(sv, v.v[p]);
            const cuda_t* elems = reinterpret_cast<const cuda_t*>(&prod);
            atomicAdd(&ptr[base_f + p * 2],     elems[0]);
            atomicAdd(&ptr[base_f + p * 2 + 1], elems[1]);
        }
    }
};
