#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <torch/torch.h>

#include <cstddef>

inline size_t max_dynamic_smem_per_block(int device = -1) {
    if (device < 0) {
        C10_CUDA_CHECK(cudaGetDevice(&device));
    }
    int value = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&value, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    return static_cast<size_t>(value);
}

template <typename KernelPtr>
inline void enable_dynamic_smem(KernelPtr kernel, size_t bytes) {
    constexpr size_t kDefaultLimit = 48 * 1024;
    if (bytes <= kDefaultLimit) {
        return;
    }

    const size_t limit = max_dynamic_smem_per_block();
    TORCH_CHECK(
        bytes <= limit,
        "kernel needs ",
        bytes,
        " bytes of dynamic shared memory, but this device allows at most ",
        limit,
        " per block. Lower the tile width, the warp count, or the number of pipeline stages."
    );

    C10_CUDA_CHECK(
        cudaFuncSetAttribute(reinterpret_cast<void const *>(kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(bytes))
    );
}
