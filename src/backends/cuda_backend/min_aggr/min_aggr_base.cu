#include <torch/extension.h>
#include <vector>

namespace py = pybind11;

void min_aggr_forward_cuda(
    const at::Tensor& edge_ptr,
    const at::Tensor& edge_idx,
    const at::Tensor& X,
    at::Tensor& out,
    at::Tensor& argmin
);

void min_aggr_backward_cuda(
    const at::Tensor& grad_out,
    const at::Tensor& argmin,
    at::Tensor& grad_x
);

std::vector<at::Tensor> min_aggr_forward_torch(
    at::Tensor edge_ptr,
    at::Tensor edge_idx,
    at::Tensor X
) {
    TORCH_CHECK(edge_ptr.is_cuda(), "edge_ptr must be CUDA");
    TORCH_CHECK(edge_idx.is_cuda(), "edge_idx must be CUDA");
    TORCH_CHECK(X.is_cuda(), "X must be CUDA");
    TORCH_CHECK(edge_ptr.dtype() == torch::kInt32, "edge_ptr must be int32");
    TORCH_CHECK(edge_idx.dtype() == torch::kInt32, "edge_idx must be int32");
    TORCH_CHECK(X.dtype() == torch::kFloat32, "X must be float32");


    const int64_t num_nodes = X.size(0);
    const int64_t d = X.size(1);
    TORCH_CHECK(num_nodes > 0, "num_nodes must be > 0");
    TORCH_CHECK(X.dim() == 2, "X must be 2D");

    // auto out = torch::empty({num_nodes, d}, X.options());
    // auto argmin = torch::empty({num_nodes, d}, edge_ptr.options());
    auto out = torch::full({num_nodes, d}, INFINITY, X.options());
    auto argmin = torch::full({num_nodes, d}, -1, edge_ptr.options());
    min_aggr_forward_cuda(edge_ptr, edge_idx, X, out, argmin);
    return {out, argmin};
}

at::Tensor min_aggr_backward_torch(
    at::Tensor grad_out,
    at::Tensor argmin,
    int64_t num_src_nodes
) {
    TORCH_CHECK(grad_out.is_cuda(), "grad_out must be CUDA");
    TORCH_CHECK(argmin.is_cuda(), "argmin must be CUDA");
    TORCH_CHECK(grad_out.dtype() == torch::kFloat32, "grad_out must be float32");
    TORCH_CHECK(argmin.dtype() == torch::kInt32, "argmin must be int32");

    TORCH_CHECK(grad_out.dim() == 2, "grad_out must be 2D");
    TORCH_CHECK(argmin.sizes() == grad_out.sizes(), "argmin and grad_out shapes must match");
    const int64_t num_nodes = grad_out.size(0);
    const int64_t d = grad_out.size(1);

    auto grad_x = torch::zeros({num_src_nodes, d}, grad_out.options());
    min_aggr_backward_cuda(grad_out, argmin, grad_x);

    return grad_x;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("min_aggr_forward",  &min_aggr_forward_torch,
          "Min aggregation forward",
          py::arg("edge_ptr"), py::arg("edge_idx"), py::arg("X"));
    m.def("min_aggr_backward", &min_aggr_backward_torch,
          "Min aggregation backward",
          py::arg("grad_out"), py::arg("argmin"), py::arg("num_src_nodes"));
}
