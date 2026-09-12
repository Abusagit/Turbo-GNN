#pragma once

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>

#include <string>
#include <type_traits>
#include <utility>
#include <variant>

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

// Runtime -> compile-time dispatch: returns a Variant whose active alternative
// matches the runtime key. Alternatives expose their key as either ::value
// (std::integral_constant) or ::ScalarType (the info structs below); what() is
// evaluated only on failure, so building the message costs nothing on the
// dispatch path.
template <typename Variant, typename Key, typename MsgFn>
Variant MakeVariantOf(Key key, MsgFn&& what) {
    Variant result;
    bool found = false;
    [&]<size_t... I>(std::index_sequence<I...>) {
        (
            [&] {
                using Alt = std::variant_alternative_t<I, Variant>;
                bool matched;
                if constexpr (requires { Alt::value; }) {
                    matched = Alt::value == key;
                } else {
                    static_assert(requires { Alt::ScalarType; }, "MakeVariantOf alternatives must expose ::value or ::ScalarType");
                    matched = Alt::ScalarType == key;
                }
                if (matched) {
                    result.template emplace<I>();
                    found = true;
                }
            }(),
            ...);
    }(std::make_index_sequence<std::variant_size_v<Variant>>{});
    if (!found) {
        throw std::runtime_error(what());
    }
    return result;
}

template <int... Values>
std::variant<std::integral_constant<int, Values>...> MakeIntVariant(int value) {
    return MakeVariantOf<std::variant<std::integral_constant<int, Values>...>>(value, [&] {
        return "Wrong int value: " + std::to_string(value);
    });
}

template <bool... Values>
std::variant<std::integral_constant<bool, Values>...> MakeBoolVariant(bool value) {
    return MakeVariantOf<std::variant<std::integral_constant<bool, Values>...>>(value, [] { return "Wrong bool value"; });
}

// Runtime -> compile-time for an enum: the returned variant's active
// alternative carries the matching enumerator as a template argument.
template <typename EnumT, EnumT... Values>
std::variant<std::integral_constant<EnumT, Values>...> MakeEnumVariant(EnumT value) {
    return MakeVariantOf<std::variant<std::integral_constant<EnumT, Values>...>>(value, [] { return "enum value not in the dispatch set"; });
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
    return MakeVariantOf<std::variant<TTypeInfo<T>...>>(type, [] { return "Unsupported scalar type"; });
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

// Is integer trait

template <typename T>
struct is_integral_cuda {
   private:
    // Strip const/volatile, but intentionally keep references/pointers
    // so they correctly evaluate to false, matching std:: behavior.
    using U = std::remove_cvref_t<T>;

   public:
    static constexpr bool value = std::is_integral_v<U> ||              // Standard: car, short, int, long long , e.t.c.
                                  std::is_same_v<U, __int128> ||        // CUDA: i128
                                  std::is_same_v<U, unsigned __int128>  // CUDA: ui128
        ;
};

template <typename T>
inline constexpr bool is_integral_cuda_v = is_integral_cuda<T>::value;

template <typename T>
concept IntegralNum = is_integral_cuda_v<T>;

template <IntegralNum T>
inline constexpr T ceil_div(T num, T den) {
    return (num + den - 1) / den;
}

// Sentinel traits: universal "invalid index" for all types
// For signed: -1. For unsigned: max value (all-ones bit pattern).
// cast(-1) gives all-ones for both signed and unsigned.
template <IntegralNum index_t>
struct IndexSentinel {
    static constexpr index_t INVALID = static_cast<index_t>(-1);
    static __device__ __forceinline__ bool is_valid(index_t idx) { return idx != INVALID; }
};

// Runtime dispatch to compile-time index type
template <typename... IndexTypes>
std::variant<IndexTypeInfo<IndexTypes>...> MakeIndexVariant(at::ScalarType type) {
    return MakeVariantOf<std::variant<IndexTypeInfo<IndexTypes>...>>(type, [] { return "Unsupported index scalar type"; });
}
