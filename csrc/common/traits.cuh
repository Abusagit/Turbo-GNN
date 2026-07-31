#pragma once

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>

#include <type_traits>
#include <variant>

// Dispatch and datatype Traits TODO move to the separate file in the final
// version

template <typename T>
struct TTypeTraits;

// Spec for float
template <>
struct TTypeTraits<float> {
    using TorchType                             = float;
    using CudaType                              = float;
    static constexpr c10::ScalarType ScalarType = c10::ScalarType::Float;
};

// Spec for double
template <>
struct TTypeTraits<double> {
    using TorchType                             = double;
    using CudaType                              = double;
    static constexpr c10::ScalarType ScalarType = c10::ScalarType::Double;
};

// Spec for at::Half
template <>
struct TTypeTraits<at::Half> {
    using TorchType                             = at::Half;
    using CudaType                              = __half;
    static constexpr c10::ScalarType ScalarType = c10::ScalarType::Half;
};

// Spec for at::BFloat16
template <>
struct TTypeTraits<at::BFloat16> {
    using TorchType                             = at::BFloat16;
    using CudaType                              = __nv_bfloat16;
    static constexpr c10::ScalarType ScalarType = c10::ScalarType::BFloat16;
};

// Helper for obtaining CUDA type from PyTorch  type
template <typename TorchT>
using ToCudaType = typename TTypeTraits<TorchT>::CudaType;

template <int... Values>
std::variant<std::integral_constant<int, Values>...> MakeIntVariant(int value) {
    std::variant<std::integral_constant<int, Values>...> result;
    bool found = false;
    (
        [&] {
            if (value == Values) {
                result.template emplace<std::integral_constant<int, Values>>();
                found = true;
            }
        }(),
        ...);
    if (!found) {
        throw std::runtime_error("Wrong int value: " + std::to_string(value));
    }
    return result;
}

template <bool... Values>
std::variant<std::integral_constant<bool, Values>...> MakeBoolVariant(bool value) {
    std::variant<std::integral_constant<bool, Values>...> result;
    bool found = false;
    (
        [&] {
            if (value == Values) {
                result.template emplace<std::integral_constant<bool, Values>>();
                found = true;
            }
        }(),
        ...);
    if (!found) {
        throw std::runtime_error("Wrong bool value");
    }
    return result;
}

template <typename T>
struct TTypeInfo {
    using Traits    = TTypeTraits<T>;
    using TorchType = typename Traits::TorchType;
    using CudaType  = typename Traits::CudaType;

    static constexpr c10::ScalarType ScalarType = Traits::ScalarType;
};

template <typename... T>
inline std::variant<TTypeInfo<T>...> MakeTypeVariant(at::ScalarType type) {
    std::variant<TTypeInfo<T>...> result;
    bool found = false;
    (
        [&] {
            if (TTypeInfo<T>::ScalarType == type) {
                result.template emplace<TTypeInfo<T>>();
                found = true;
            }
        }(),
        ...);
    if (!found) {
        throw std::runtime_error("Unsupported scalar type");
    }
    return result;
}

// =============================================================================
// Index type dispatch infrastructure
// =============================================================================

// Index type info: maps C++ integer type -> c10::ScalarType
template <typename T>
struct IndexTypeInfo {
    using Type = T;
};

template <>
struct IndexTypeInfo<int32_t> {
    using Type                                  = int32_t;
    static constexpr c10::ScalarType ScalarType = c10::ScalarType::Int;
};

template <>
struct IndexTypeInfo<int64_t> {
    using Type                                  = int64_t;
    static constexpr c10::ScalarType ScalarType = c10::ScalarType::Long;
};

template <>
struct IndexTypeInfo<uint32_t> {
    using Type                                  = uint32_t;
    static constexpr c10::ScalarType ScalarType = c10::ScalarType::UInt32;
};

template <>
struct IndexTypeInfo<uint64_t> {
    using Type                                  = uint64_t;
    static constexpr c10::ScalarType ScalarType = c10::ScalarType::UInt64;
};

// Sentinel traits: universal "invalid index" for all types
// For signed: -1. For unsigned: max value (all-ones bit pattern).
// cast(-1) gives all-ones for both signed and unsigned.
template <typename index_t>
struct IndexSentinel {
    static constexpr index_t INVALID = static_cast<index_t>(-1);
    static __device__ __forceinline__ bool is_valid(index_t idx) { return idx != INVALID; }
};

// Runtime dispatch to compile-time index type
template <typename... IndexTypes>
std::variant<IndexTypeInfo<IndexTypes>...> MakeIndexVariant(at::ScalarType type) {
    std::variant<IndexTypeInfo<IndexTypes>...> result;
    bool found = false;
    (
        [&] {
            if (IndexTypeInfo<IndexTypes>::ScalarType == type) {
                result.template emplace<IndexTypeInfo<IndexTypes>>();
                found = true;
            }
        }(),
        ...);
    if (!found) {
        throw std::runtime_error("Unsupported index scalar type");
    }
    return result;
}

// Is floating point trait

template <typename T>
struct is_floating_point_cuda {
   private:
    // Strip const/volatile, but intentionally keep references/pointers
    // so they correctly evaluate to false, matching std:: behavior.
    using U = std::remove_cvref_t<T>;

   public:
    static constexpr bool value = std::is_floating_point_v<U> ||       // Standard: float, double, long double
                                  std::is_same_v<U, __half> ||         // CUDA: FP16
                                  std::is_same_v<U, __nv_bfloat16> ||  // CUDA: BF16
                                  std::is_same_v<U, __nv_fp8_e4m3> ||  // CUDA: FP8 (E4M3)
                                  std::is_same_v<U, __nv_fp8_e5m2> ||  // CUDA: FP8 (E5M2)
                                  std::is_same_v<U, __float128>        // CUDA: FP128
        ;
};

template <typename T>
inline constexpr bool is_floating_point_cuda_v = is_floating_point_cuda<T>::value;

template <typename T>
concept FloatingNum = is_floating_point_cuda_v<T>;

template <typename T>
inline constexpr bool is_half_fp_v = std::is_same_v<std::remove_cv_t<T>, half> || std::is_same_v<std::remove_cv_t<T>, nv_bfloat16>;

template <FloatingNum L, FloatingNum S>
static constexpr __host__ __device__ auto broadcast_scalar_to_packed(L val) {
    if constexpr (sizeof(L) == 2) {
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