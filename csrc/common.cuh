#pragma once

#include "common/misc.cuh"
#include "common/traits.cuh"
#include "common/tile.cuh"
#include "common/common_.cuh"

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
__device__ __forceinline__ bool operator<(const __half& a, const __half& b) { return __hlt(a, b); }
__device__ __forceinline__ bool operator>(const __half& a, const __half& b) { return __hgt(a, b); }
__device__ __forceinline__ bool operator<=(const __half& a, const __half& b) { return __hle(a, b); }
__device__ __forceinline__ bool operator>=(const __half& a, const __half& b) { return __hge(a, b); }
__device__ __forceinline__ bool operator==(const __half& a, const __half& b) { return __heq(a, b); }
__device__ __forceinline__ bool operator!=(const __half& a, const __half& b) { return __hne(a, b); }
#endif