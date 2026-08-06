#pragma once

#include "common/misc.cuh"
#include "common/traits.cuh"

template <FloatingNum T>
struct AdOps {
    using packed_t = deduce_packed_type_t<T>;

    // Singluar functions

    static constexpr __device__ __forceinline__ T log(T a) {
        if constexpr (is_half_fp_v<T>) {
            return hlog(a);
        } else if constexpr (std::is_same_v<std::remove_cvref_t<T>, float> && is_device_pass) {
            return __logf(a);
        } else {
            return cuda::std::log(a);
        }
    }

    static constexpr __device__ __forceinline__ T exp(T a) {
        if constexpr (is_half_fp_v<T>) {
            return hexp(a);
        } else if constexpr (std::is_same_v<std::remove_cvref_t<T>, float> && is_device_pass) {
            return __expf(a);
        } else {
            return cuda::std::exp(a);
        }
    }

    static constexpr __device__ __forceinline__ T min(T a, T b) {
        if constexpr (is_half_fp_v<T>) {
            return __hmin(a, b);
        } else if constexpr (std::is_same_v<std::remove_cvref_t<T>, float> && is_device_pass) {
            return fminf(a, b);
        } else {
            return cuda::std::min(a, b);
        }
    }

    static constexpr __device__ __forceinline__ T max(T a, T b) {
        if constexpr (is_half_fp_v<T>) {
            return __hmax(a, b);
        } else if constexpr (std::is_same_v<std::remove_cvref_t<T>, float> && is_device_pass) {
            return fmaxf(a, b);
        } else {
            return cuda::std::max(a, b);
        }
    }

    static constexpr __device__ __forceinline__ T fma(T a, T b, T c) {
        if constexpr (is_half_fp_v<T>) {
            return __hfma(a, b, c);
        } else if constexpr (std::is_same_v<std::remove_cvref_t<T>, float> && is_device_pass) {
            return fmaf(a, b, c);
        } else {
            return cuda::std::fma(a, b, c);
        }
    }

    // Packed functions
    static constexpr __device__ __forceinline__ packed_t packed_neg(packed_t a) {
        if constexpr (is_half_fp_v<T>) {
            return __hneg2(a);
        } else {
            return {-a.x, -a.y};
        }
    }

    static constexpr __device__ __forceinline__ packed_t packed_log(packed_t a) {
        if constexpr (is_half_fp_v<T>) {
            return h2log(a);
        } else {
            return {AdOps<T>::log(a.x), AdOps<T>::log(a.y)};
        }
    }

    static constexpr __device__ __forceinline__ packed_t packed_exp(packed_t a) {
        if constexpr (is_half_fp_v<T>) {
            return h2exp(a);
        } else {
            return {AdOps<T>::exp(a.x), AdOps<T>::exp(a.y)};
        }
    }

    static constexpr __device__ __forceinline__ packed_t packed_min(packed_t a, packed_t b) {
        if constexpr (is_half_fp_v<T>) {
            return __hmin2(a, b);
        } else if constexpr (std::is_same_v<std::remove_cvref_t<T>, float> && is_device_pass) {
            return {fminf(a.x, b.x), fminf(a.y, b.y)};
        } else {
            return {cuda::std::min(a.x, b.x), cuda::std::min(a.y, b.y)};
        }
    }

    static constexpr __device__ __forceinline__ packed_t packed_max(packed_t a, packed_t b) {
        if constexpr (is_half_fp_v<T>) {
            return __hmax2(a, b);
        } else if constexpr (std::is_same_v<std::remove_cvref_t<T>, float> && is_device_pass) {
            return {fmaxf(a.x, b.x), fmaxf(a.y, b.y)};
        } else {
            return {cuda::std::max(a.x, b.x), cuda::std::max(a.y, b.y)};
        }
    }

    static constexpr __device__ __forceinline__ packed_t packed_add(packed_t a, packed_t b) {
        if constexpr (is_half_fp_v<T>) {
            return __hadd2(a, b);
        } else if constexpr (std::is_same_v<std::remove_cvref_t<T>, float> && kCudaArch >= 1000) {
            __fadd2_rn(a, b);
        } else {
            return {a.x + b.x, a.y + b.y};
        }
    }

    static constexpr __device__ __forceinline__ packed_t packed_sub(packed_t a, packed_t b) {
        if constexpr (is_half_fp_v<T>) {
            return __hsub2(a, b);
        } else {
            return {a.x - b.x, a.y - b.y};
        }
    }

    static constexpr __device__ __forceinline__ packed_t packed_mul(packed_t a, packed_t b) {
        if constexpr (is_half_fp_v<T>) {
            return __hmul2(a, b);
        } else if constexpr (std::is_same_v<std::remove_cvref_t<T>, float> && kCudaArch >= 1000) {
            __fmul2_rn(a, b);
        } else {
            return {a.x * b.x, a.y * b.y};
        }
    }

    static constexpr __device__ __forceinline__ packed_t packed_div(packed_t a, packed_t b) {
        if constexpr (is_half_fp_v<T>) {
            return __h2div(a, b);
        } else {
            return {a.x / b.x, a.y / b.y};
        }
    }

    static constexpr __device__ __forceinline__ packed_t packed_fma(packed_t a, packed_t b, packed_t c) {
        if constexpr (is_half_fp_v<T>) {
            return __hfma2(a, b, c);
        } else if constexpr (std::is_same_v<std::remove_cvref_t<T>, float> && kCudaArch >= 1000) {
            __ffma2_rn(a, b, c);
        } else {
            return {a.x * b.x + c.x, a.y * b.y + c.y};
        }
    }
};

namespace adops {
template <
    FloatingNum dst_type, FloatingNum src_type, typename packed_dst_type = deduce_packed_type_t<dst_type>,
    typename packed_src_type = deduce_packed_type_t<src_type>
>
constexpr __device__ __forceinline__ packed_dst_type packed_convert(packed_src_type a) {
    if constexpr (std::is_same_v<std::remove_cvref_t<src_type>, float>) {
        if constexpr (std::is_same_v<std::remove_cvref_t<dst_type>, nv_bfloat16>) {
            return __float22bfloat162_rn(a);
        } else if constexpr (std::is_same_v<std::remove_cvref_t<dst_type>, half>) {
            return __float22half2_rn(a);
        } else {
            __builtin_unreachable();
        }
    } else if constexpr (std::is_same_v<std::remove_cvref_t<src_type>, nv_bfloat16>) {
        if constexpr (std::is_same_v<std::remove_cvref_t<dst_type>, float>) {
            return __bfloat1622float2(a);
        } else if constexpr (std::is_same_v<std::remove_cvref_t<dst_type>, half>) {
            return __float22half2_rn(__bfloat1622float2(a));
        } else {
            __builtin_unreachable();
        }
    } else if constexpr (std::is_same_v<std::remove_cvref_t<dst_type>, half>) {
        if constexpr (std::is_same_v<std::remove_cvref_t<dst_type>, float>) {
            return __half22float2(a);
        } else if constexpr (std::is_same_v<std::remove_cvref_t<dst_type>, nv_bfloat16>) {
            return __float22bfloat162_rn(__half22float2(a));
        } else {
            __builtin_unreachable();
        }
    } else {
        __builtin_unreachable();
    }
}

template <FloatingNum L, FloatingNum S>
constexpr __device__ auto broadcast_scalar_to_packed(L val) {
    if constexpr (is_half_fp_v<L>) {
        if constexpr (std::is_same_v<std::remove_cvref_t<S>, half>) {
            return half2{val, val};
        } else if constexpr (std::is_same_v<std::remove_cvref_t<S>, nv_bfloat16>) {
            return nv_bfloat162{val, val};
        } else {
            __builtin_unreachable();
        }
    } else {
        if constexpr (std::is_same_v<std::remove_cvref_t<S>, half>) {
            return __float2half2_rn(static_cast<float>(val));
        } else if constexpr (std::is_same_v<std::remove_cvref_t<S>, nv_bfloat16>) {
            return __float2bfloat162_rn(static_cast<float>(val));
        } else {
            __builtin_unreachable();
        }
    }
}
}  // namespace adops
