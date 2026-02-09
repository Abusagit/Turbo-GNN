#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include "mean_aggr.cuh"

#include <vector>
#include <cstdint>

namespace py = pybind11;

// 0: feature-coalesced (warp reads consecutive features of one neighbor)
// 1: neighbor-parallel (warp lanes read different neighbors for same feature tile)
enum class MeanAggrKernelKind : int {
    kFeatureCoalesced = 0,
    kNeighborParallel = 1,
};


template <typename scalar_t, bool kUseSecondAccess, bool kUseVectorizedLoads>
void mean_aggr_launch_kernel_feature_coalesced( // feature-coalesced
    const int32_t* edge_ptr,   // [N+1]
    const int32_t* edge_idx,   // [E]
    const scalar_t* X,         // [N, D]
    scalar_t* out,             // [N, D]
    float* scratch,            // [N] or nullptr
    int32_t N,
    int32_t D,
    cudaStream_t stream
);

template <typename scalar_t, bool kUseSecondAccess, bool kUseVectorizedLoads>
void mean_aggr_launch_kernel_neighbor_parallel( // neighbor-parallel
    const int32_t* edge_ptr,   // [N+1]
    const int32_t* edge_idx,   // [E]
    const scalar_t* X,         // [N, D]
    scalar_t* out,             // [N, D]
    float* scratch,            // [N] or nullptr
    int32_t N,
    int32_t D,
    cudaStream_t stream
);


template <typename scalar_t>
static void mean_aggr_dispatch_typed(
    const at::Tensor& edge_ptr,
    const at::Tensor& edge_idx,
    const at::Tensor& X,
    at::Tensor& out,
    at::Tensor* scratch, // nullable
    MeanAggrKernelKind kernel_kind,
    bool use_second_access,
    bool use_vectorized_loads
) {
    const int32_t N = static_cast<int32_t>(edge_ptr.numel() - 1);
    const int32_t D = static_cast<int32_t>(X.size(1));

    const auto stream = at::cuda::getDefaultCUDAStream().stream();

    const int32_t* edge_ptr_ptr = edge_ptr.data_ptr<int32_t>();
    const int32_t* edge_idx_ptr = edge_idx.data_ptr<int32_t>();
    const scalar_t* X_ptr       = X.data_ptr<scalar_t>();
    scalar_t* out_ptr           = out.data_ptr<scalar_t>();
    float* scratch_ptr          = (scratch ? scratch->data_ptr<float>() : nullptr);

    // Runtime -> compile-time dispatch
    if (kernel_kind == MeanAggrKernelKind::kFeatureCoalesced) {
        if (use_second_access) {
            if (use_vectorized_loads) {
                mean_aggr_launch_kernel_feature_coalesced<scalar_t, true, true>(
                    edge_ptr_ptr, edge_idx_ptr, X_ptr, out_ptr, scratch_ptr, N, D, stream);
            } else {
                mean_aggr_launch_kernel_feature_coalesced<scalar_t, true, false>(
                    edge_ptr_ptr, edge_idx_ptr, X_ptr, out_ptr, scratch_ptr, N, D, stream);
            }
        } else {
            if (use_vectorized_loads) {
                mean_aggr_launch_kernel_feature_coalesced<scalar_t, false, true>(
                    edge_ptr_ptr, edge_idx_ptr, X_ptr, out_ptr, scratch_ptr, N, D, stream);
            } else {
                mean_aggr_launch_kernel_feature_coalesced<scalar_t, false, false>(
                    edge_ptr_ptr, edge_idx_ptr, X_ptr, out_ptr, scratch_ptr, N, D, stream);
            }
        }
    } else { // kNeighborParallel
        if (use_second_access) {
            if (use_vectorized_loads) {
                mean_aggr_launch_kernel_neighbor_parallel<scalar_t, true, true>(
                    edge_ptr_ptr, edge_idx_ptr, X_ptr, out_ptr, scratch_ptr, N, D, stream);
            } else {
                mean_aggr_launch_kernel_neighbor_parallel<scalar_t, true, false>(
                    edge_ptr_ptr, edge_idx_ptr, X_ptr, out_ptr, scratch_ptr, N, D, stream);
            }
        } else {
            if (use_vectorized_loads) {
                mean_aggr_launch_kernel_neighbor_parallel<scalar_t, false, true>(
                    edge_ptr_ptr, edge_idx_ptr, X_ptr, out_ptr, scratch_ptr, N, D, stream);
            } else {
                mean_aggr_launch_kernel_neighbor_parallel<scalar_t, false, false>(
                    edge_ptr_ptr, edge_idx_ptr, X_ptr, out_ptr, scratch_ptr, N, D, stream);
            }
        }
    }
}


std::vector<at::Tensor> mean_aggr_forward(
    at::Tensor edge_ptr,                // int32 [N+1]
    at::Tensor edge_idx,                // int32 [E]
    at::Tensor X,                       // (float32 or bfloat16) [N, D]
    int64_t kernel_kind,                // 0 or 1
    bool use_second_access,
    bool use_vectorized_loads
) {
    TORCH_CHECK(edge_ptr.is_cuda() && edge_idx.is_cuda() && X.is_cuda(), "inputs must be CUDA");
    c10::cuda::CUDAGuard device_guard(X.device());

    TORCH_CHECK(edge_ptr.scalar_type() == at::kInt, "edge_ptr must be int32");
    TORCH_CHECK(edge_idx.scalar_type() == at::kInt, "edge_idx must be int32");
    TORCH_CHECK(edge_ptr.dim() == 1, "edge_ptr must be 1D [N+1]");
    TORCH_CHECK(edge_idx.dim() == 1, "edge_idx must be 1D [E]");
    TORCH_CHECK(X.dim() == 2, "X must be 2D [N, D]");

    const int64_t N = edge_ptr.numel() - 1;
    TORCH_CHECK(N >= 0, "edge_ptr must have at least 1 element");
    TORCH_CHECK(X.size(0) == N, "X.size(0) must match num_nodes = edge_ptr.numel()-1");

    TORCH_CHECK(
        X.scalar_type() == at::kFloat || X.scalar_type() == at::kBFloat16,
        "X must be float32 or bfloat16"
    );

    // Keep the benchmark honest: require contiguous row-major [N, D]
    TORCH_CHECK(X.is_contiguous(), "X must be contiguous");
    TORCH_CHECK(edge_ptr.is_contiguous(), "edge_ptr must be contiguous");
    TORCH_CHECK(edge_idx.is_contiguous(), "edge_idx must be contiguous");

    const int64_t D = X.size(1);
    TORCH_CHECK(D > 0, "X.size(1) must be > 0");


    if (use_vectorized_loads) {
        constexpr uintptr_t kAlign = 16;
        const uintptr_t x_addr = reinterpret_cast<uintptr_t>(X.data_ptr());
        TORCH_CHECK((x_addr % kAlign) == 0, "X data_ptr must be 16B-aligned for vectorized loads");

        if (X.scalar_type() == at::kFloat) {
            TORCH_CHECK((D % 4) == 0, "For float32 vectorized loads, D must be multiple of 4 (float4)");
        } else { // bfloat16
            TORCH_CHECK((D % 8) == 0, "For bfloat16 vectorized loads, D must be multiple of 8 (16B pack)");
        }
    }

    auto out = at::empty({N, D}, X.options());

    // optional scratch to force the second pass loads to be executed
    at::Tensor scratch;
    at::Tensor* scratch_ptr = nullptr;
    if (use_second_access) {
        scratch = at::empty({N}, X.options().dtype(at::kFloat));
        scratch_ptr = &scratch;
    }

    const auto kk = static_cast<MeanAggrKernelKind>(kernel_kind);
    TORCH_CHECK(kk == MeanAggrKernelKind::kFeatureCoalesced || kk == MeanAggrKernelKind::kNeighborParallel,
                "kernel_kind must be 0 (feature-coalesced) or 1 (neighbor-parallel)");

    if (X.scalar_type() == at::kFloat) {
        mean_aggr_dispatch_typed<float>(edge_ptr, edge_idx, X, out, scratch_ptr, kk,
                                        use_second_access, use_vectorized_loads);
    } else {
        mean_aggr_dispatch_typed<at::BFloat16>(edge_ptr, edge_idx, X, out, scratch_ptr, kk,
                                              use_second_access, use_vectorized_loads);
    }

    if (use_second_access) return {out, scratch};
    return {out};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mean_aggr_forward", &mean_aggr_forward,
          "Mean aggregation (reordering sensitivity microbenchmark)",
          py::arg("edge_ptr"),
          py::arg("edge_idx"),
          py::arg("X"),
          py::arg("kernel_kind") = 0,
          py::arg("use_second_access") = false,
          py::arg("use_vectorized_loads") = false);
}
