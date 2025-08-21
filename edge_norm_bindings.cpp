#include <torch/extension.h>

enum class NormType {
    NONE = 0,
    RIGHT = 1,
    LEFT = 2,
    BOTH = 3
};

void launch_compute_degrees(const torch::Tensor& indptr, const torch::Tensor& indices,
                            torch::Tensor& in_degrees, torch::Tensor& out_degrees);

void launch_compute_normalized_weights(const torch::Tensor& indptr, const torch::Tensor& indices,
                                       const torch::Tensor& edge_weights, torch::Tensor& normalized_weights,
                                       const torch::Tensor& in_degrees, const torch::Tensor& out_degrees,
                                       NormType norm);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::enum_<NormType>(m, "NormType")
        .value("NONE",  NormType::NONE)
        .value("RIGHT", NormType::RIGHT)
        .value("LEFT",  NormType::LEFT)
        .value("BOTH",  NormType::BOTH);

    m.def("compute_degrees", &launch_compute_degrees,
          "Compute in/out degrees (TRANSPOSED CSR)");
    m.def("compute_normalized_weights", &launch_compute_normalized_weights,
          "Compute normalized edge weights (TRANSPOSED CSR)");
}
