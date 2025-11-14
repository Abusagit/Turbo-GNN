#pragma once

#include <cuda_runtime.h>
#include <vector>

// Core C10 types (lightweight)
#include <c10/core/ScalarType.h>
#include <c10/core/Device.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/Optional.h>

// ATen core (tensor definition, no heavy templates)
#include <ATen/core/Tensor.h>

// Tensor creation functions (empty, ones, zeros, etc.)
#include <ATen/Functions.h>

// AT_DISPATCH_FLOATING_TYPES and similar macros
#include <ATen/Dispatch.h>

// PackedTensorAccessor - needed for efficient tensor access in CUDA kernels
#include <ATen/core/TensorAccessor.h>

// CUDA stream
#include <c10/cuda/CUDAStream.h>

// For TORCH_CHECK macro (already defined in Exception.h)
#include <c10/util/Exception.h>

// Convenience namespace aliases
namespace torch {
    using Tensor = at::Tensor;
    using TensorOptions = at::TensorOptions;

    template<typename T>
    using optional = c10::optional<T>;

    // PackedTensorAccessor with correct template template parameter
    template<typename T, size_t N, template <typename U> class PtrTraits = at::DefaultPtrTraits, typename index_t = int64_t>
    using PackedTensorAccessor = at::PackedTensorAccessor<T, N, PtrTraits, index_t>;

    // Pointer traits templates
    template<typename T>
    using RestrictPtrTraits = at::RestrictPtrTraits<T>;

    template<typename T>
    using DefaultPtrTraits = at::DefaultPtrTraits<T>;

    // Scalar type constants - these are actually enum values in c10::ScalarType
    constexpr auto kFloat32 = c10::ScalarType::Float;
    constexpr auto kFloat64 = c10::ScalarType::Double;
    constexpr auto kFloat = c10::ScalarType::Float;
    constexpr auto kDouble = c10::ScalarType::Double;
    constexpr auto kInt32 = c10::ScalarType::Int;
    constexpr auto kInt64 = c10::ScalarType::Long;
    constexpr auto kInt = c10::ScalarType::Int;
    constexpr auto kLong = c10::ScalarType::Long;

    // Tensor factory functions
    using at::empty;
    using at::empty_like;
    using at::zeros;
    using at::zeros_like;
    using at::ones;
    using at::ones_like;
}

// Helper function for getting CUDA stream
inline cudaStream_t getCurrentCUDAStream() {
    return c10::cuda::getCurrentCUDAStream().stream();
}

// TORCH_CHECK is already defined in c10/util/Exception.h
