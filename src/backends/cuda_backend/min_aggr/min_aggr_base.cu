#include <cuda_runtime.h>
#include <cmath>
#include <torch/extension.h>
#include <torch/torch.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>

namespace py = pybind11;


void min_aggr_backward_cuda(
    const at::Tensor& grad_out,
    const at::Tensor& argmin,
    at::Tensor& grad_x,
    int warps_per_block = 8
);

void min_aggr_forward_partitioned_cuda(
    const at::Tensor& edge_ptr,
    const at::Tensor& edge_idx,
    const at::Tensor& X,
    const at::Tensor& light_nodes,
    const at::Tensor& heavy_nodes,
    int max_degree,
    at::Tensor& out,
    at::Tensor& argmin,
    int warps_per_block = 8,
    int edges_per_block_heavy_nodes = 128,
    bool use_2d_kernel = false,
    int features_per_block = 32,
    int tiles_y = 8
);

at::Tensor min_aggr_backward_torch(
    at::Tensor grad_out,
    at::Tensor argmin,
    int64_t num_src_nodes,
    int warps_per_block = 8
) {
    TORCH_CHECK(grad_out.is_cuda(), "grad_out must be CUDA");
    TORCH_CHECK(argmin.is_cuda(), "argmin must be CUDA");
    TORCH_CHECK(
        grad_out.scalar_type() == at::kFloat ||
        grad_out.scalar_type() == at::kDouble ||
        grad_out.scalar_type() == at::kHalf ||
        grad_out.scalar_type() == at::kBFloat16,
        "grad_out must be float32/float64/float16/bfloat16"
    );

    TORCH_CHECK(grad_out.dim() == 2, "grad_out must be 2D");
    TORCH_CHECK(argmin.sizes() == grad_out.sizes(), "argmin and grad_out shapes must match");
    const int64_t num_nodes = grad_out.size(0);
    const int64_t d = grad_out.size(1);

    auto grad_x = torch::zeros({num_src_nodes, d}, grad_out.options());

    min_aggr_backward_cuda(grad_out, argmin, grad_x, warps_per_block);
    return grad_x;
}

std::vector<at::Tensor> min_aggr_forward_partitioned_torch(
    at::Tensor edge_ptr,
    at::Tensor edge_idx,
    at::Tensor X,
    at::Tensor light_nodes,
    at::Tensor heavy_nodes,
    int max_degree,
    int warps_per_block = 8,
    int edges_per_block_heavy_nodes = 128,
    bool use_2d_kernel = false,
    int features_per_block = 32,
    int tiles_y = 8
) {
    TORCH_CHECK(edge_ptr.is_cuda() && edge_idx.is_cuda() && X.is_cuda(), "inputs must be CUDA");
    TORCH_CHECK(light_nodes.is_cuda() && heavy_nodes.is_cuda(), "node lists must be CUDA");
    TORCH_CHECK(edge_ptr.dtype() == torch::kInt32, "edge_ptr must be int32");
    TORCH_CHECK(edge_idx.dtype() == torch::kInt32, "edge_idx must be int32");
    TORCH_CHECK(light_nodes.dtype() == torch::kInt32, "light_nodes must be int32");
    TORCH_CHECK(heavy_nodes.dtype() == torch::kInt32, "heavy_nodes must be int32");
    TORCH_CHECK(X.scalar_type() == at::kFloat || X.scalar_type() == at::kHalf || X.scalar_type() == at::kBFloat16 || X.scalar_type() == at::kDouble, "X must be float32/float16/bfloat16/float64");
    TORCH_CHECK(X.dim() == 2, "X must be 2D");

    if (use_2d_kernel) {
        TORCH_CHECK(features_per_block > 0 && features_per_block <= 1024,  "features_per_block must be in range [1, 1024]");
        TORCH_CHECK(tiles_y > 0 && tiles_y <= 32, "tiles_y must be in range [1, 32]");
        TORCH_CHECK((tiles_y & (tiles_y - 1)) == 0, "tiles_y must be a power of 2 (e.g., 1, 2, 4, 8, 16, 32)");
        int total_threads = features_per_block * tiles_y;
        TORCH_CHECK(total_threads <= 1024,  "features_per_block * tiles_y must be <= 1024, got " +  std::to_string(total_threads));
    }

    const auto num_nodes = X.size(0);
    const auto d = X.size(1);

    auto out = torch::empty({num_nodes, d}, X.options());
    auto argmin = torch::empty({num_nodes, d}, edge_ptr.options());

    min_aggr_forward_partitioned_cuda(edge_ptr, edge_idx, X, light_nodes, heavy_nodes, max_degree, out, argmin, warps_per_block, edges_per_block_heavy_nodes, use_2d_kernel, features_per_block, tiles_y);
    return {out, argmin};
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("min_aggr_backward", &min_aggr_backward_torch,
          "Min aggregation backward",
          py::arg("grad_out"), py::arg("argmin"), py::arg("num_src_nodes"),
          py::arg("warps_per_block") = 8
        );
    m.def("min_aggr_forward_partitioned", &min_aggr_forward_partitioned_torch,
          "Min aggregation forward (partitioned)",
          py::arg("edge_ptr"), py::arg("edge_idx"), py::arg("X"),
          py::arg("light_nodes"), py::arg("heavy_nodes"), py::arg("max_degree"),
          py::arg("warps_per_block") = 8, py::arg("edges_per_block_heavy_nodes") = 128,
          py::arg("use_2d_kernel") = false, py::arg("features_per_block") = 32,
          py::arg("tiles_y") = 8
        );
}
