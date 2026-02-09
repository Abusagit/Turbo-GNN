#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <vector>
#include <cstdint>

#include "dot_aggr_kernels.cuh"

namespace py = pybind11;

enum class DotAggrKernelKind : int {
    kFeatureParallel = 0,  // feature-coalesced
    kNeighborParallel = 1, // layout-sensitive
};

template <typename scalar_t>
static void dot_aggr_dispatch_typed(
    const at::Tensor& edge_ptr,
    const at::Tensor& edge_idx,
    const at::Tensor& X,
    at::Tensor& out,              // float32 [N]
    at::Tensor* scratch,          // float32 [N] or nullptr
    DotAggrKernelKind kernel_kind,
    bool use_second_access,
    bool use_vectorized_loads
) {
    const int32_t N = static_cast<int32_t>(edge_ptr.numel() - 1);
    const int32_t D = static_cast<int32_t>(X.size(1));

    const auto stream = at::cuda::getDefaultCUDAStream().stream();

    const int32_t* edge_ptr_ptr = edge_ptr.data_ptr<int32_t>();
    const int32_t* edge_idx_ptr = edge_idx.data_ptr<int32_t>();
    const scalar_t* X_ptr       = X.data_ptr<scalar_t>();
    float* out_ptr              = out.data_ptr<float>();
    float* scratch_ptr          = (scratch ? scratch->data_ptr<float>() : nullptr);

    // Runtime -> compile-time dispatch
    if (kernel_kind == DotAggrKernelKind::kFeatureParallel) {
        if (use_second_access) {
            if (use_vectorized_loads) {
                dot_aggr_launch_kernel_feature_parallel<scalar_t, true, true>(
                    edge_ptr_ptr, edge_idx_ptr, X_ptr, out_ptr, scratch_ptr, N, D, stream);
            } else {
                dot_aggr_launch_kernel_feature_parallel<scalar_t, true, false>(
                    edge_ptr_ptr, edge_idx_ptr, X_ptr, out_ptr, scratch_ptr, N, D, stream);
            }
        } else {
            if (use_vectorized_loads) {
                dot_aggr_launch_kernel_feature_parallel<scalar_t, false, true>(
                    edge_ptr_ptr, edge_idx_ptr, X_ptr, out_ptr, scratch_ptr, N, D, stream);
            } else {
                dot_aggr_launch_kernel_feature_parallel<scalar_t, false, false>(
                    edge_ptr_ptr, edge_idx_ptr, X_ptr, out_ptr, scratch_ptr, N, D, stream);
            }
        }
    } else { // neighbor-parallel
        if (use_second_access) {
            if (use_vectorized_loads) {
                dot_aggr_launch_kernel_neighbor_parallel<scalar_t, true, true>(
                    edge_ptr_ptr, edge_idx_ptr, X_ptr, out_ptr, scratch_ptr, N, D, stream);
            } else {
                dot_aggr_launch_kernel_neighbor_parallel<scalar_t, true, false>(
                    edge_ptr_ptr, edge_idx_ptr, X_ptr, out_ptr, scratch_ptr, N, D, stream);
            }
        } else {
            if (use_vectorized_loads) {
                dot_aggr_launch_kernel_neighbor_parallel<scalar_t, false, true>(
                    edge_ptr_ptr, edge_idx_ptr, X_ptr, out_ptr, scratch_ptr, N, D, stream);
            } else {
                dot_aggr_launch_kernel_neighbor_parallel<scalar_t, false, false>(
                    edge_ptr_ptr, edge_idx_ptr, X_ptr, out_ptr, scratch_ptr, N, D, stream);
            }
        }
    }
}

std::vector<at::Tensor> dot_aggr_forward(
    at::Tensor edge_ptr,               // int32 [N+1]
    at::Tensor edge_idx,               // int32 [E]
    at::Tensor X,                      // float32 or bfloat16 [N, D]
    int64_t kernel_kind,               // 0 or 1
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

    TORCH_CHECK(X.is_contiguous(), "X must be contiguous");
    TORCH_CHECK(edge_ptr.is_contiguous(), "edge_ptr must be contiguous");
    TORCH_CHECK(edge_idx.is_contiguous(), "edge_idx must be contiguous");

    TORCH_CHECK(X.scalar_type() == at::kFloat || X.scalar_type() == at::kBFloat16,
                "X must be float32 or bfloat16");

    const int64_t D = X.size(1);
    TORCH_CHECK(D > 0, "X.size(1) must be > 0");

    // Vectorized load constraints (16B loads)
    if (use_vectorized_loads) {
        constexpr uintptr_t kAlign = 16;
        const uintptr_t x_addr = reinterpret_cast<uintptr_t>(X.data_ptr());
        TORCH_CHECK((x_addr % kAlign) == 0, "X data_ptr must be 16B-aligned for vectorized loads");

        if (X.scalar_type() == at::kFloat) {
            TORCH_CHECK((D % 4) == 0, "For float32 vectorized loads, D must be multiple of 4 (16B pack)");
        } else {
            TORCH_CHECK((D % 8) == 0, "For bfloat16 vectorized loads, D must be multiple of 8 (16B pack)");
        }
    }

    auto out = at::empty({N}, X.options().dtype(at::kFloat));

    at::Tensor scratch;
    at::Tensor* scratch_ptr = nullptr;
    if (use_second_access) {
        scratch = at::empty({N}, X.options().dtype(at::kFloat));
        scratch_ptr = &scratch;
    }

    const auto kk = static_cast<DotAggrKernelKind>(kernel_kind);
    TORCH_CHECK(kk == DotAggrKernelKind::kFeatureParallel || kk == DotAggrKernelKind::kNeighborParallel,
                "kernel_kind must be 0 (feature-parallel) or 1 (neighbor-parallel)");

    if (X.scalar_type() == at::kFloat) {
        dot_aggr_dispatch_typed<float>(edge_ptr, edge_idx, X, out, scratch_ptr, kk,
                                       use_second_access, use_vectorized_loads);
    } else {
        dot_aggr_dispatch_typed<at::BFloat16>(edge_ptr, edge_idx, X, out, scratch_ptr, kk,
                                              use_second_access, use_vectorized_loads);
    }

    if (use_second_access) return {out, scratch};
    return {out};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dot_aggr_forward", &dot_aggr_forward,
          "Dot-product aggregation microbenchmark",
          py::arg("edge_ptr"),
          py::arg("edge_idx"),
          py::arg("X"),
          py::arg("kernel_kind") = 0,
          py::arg("use_second_access") = false,
          py::arg("use_vectorized_loads") = false);
}
