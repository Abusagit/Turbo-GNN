#include <torch/extension.h>
#include <vector>

// Forward declarations from other files

// aggregate_kernel.cu
torch::Tensor fused_gar_f_cuda(
    torch::Tensor feature,
    torch::Tensor src_index,
    torch::Tensor tar_ptr,
    torch::Tensor edge_weight
);


torch::Tensor fused_gar_f(
    torch::Tensor feature,
    torch::Tensor src_index,
    torch::Tensor tar_ptr,
    torch::Tensor edge_weight
){
    return fused_gar_f_cuda(feature, src_index, tar_ptr, edge_weight);
}


std::vector<torch::Tensor> fused_gar_b_cuda(
    torch::Tensor grad_out,
    torch::Tensor feature,
    torch::Tensor tar_index,
    torch::Tensor src_ptr,
    torch::Tensor edge_weight,
    bool require_edge_weight
);


std::vector<torch::Tensor> fused_gar_b(
    torch::Tensor grad_out,
    torch::Tensor feature,
    torch::Tensor tar_index,
    torch::Tensor src_ptr,
    torch::Tensor edge_weight,
    bool require_edge_weight
){
    return fused_gar_b_cuda(grad_out, feature, tar_index, src_ptr, edge_weight, require_edge_weight);
}


torch::Tensor fused_gas_f_cuda(
    torch::Tensor feature,
    torch::Tensor src_index,
    torch::Tensor tar_index,
    torch::Tensor edge_weight
);


torch::Tensor fused_gas_f(
    torch::Tensor feature,
    torch::Tensor src_index,
    torch::Tensor tar_index,
    torch::Tensor edge_weight
){
    return fused_gas_f_cuda(feature, src_index, tar_index, edge_weight);
}


std::vector<torch::Tensor> fused_gas_b_cuda(
    torch::Tensor grad_out,
    torch::Tensor feature,
    torch::Tensor src_index,
    torch::Tensor tar_index,
    torch::Tensor edge_weight,
    bool require_edge_weight
);


std::vector<torch::Tensor> fused_gas_b(
    torch::Tensor grad_out,
    torch::Tensor feature,
    torch::Tensor src_index,
    torch::Tensor tar_index,
    torch::Tensor edge_weight,
    bool require_edge_weight
){
    return fused_gas_b_cuda(grad_out, feature, src_index, tar_index, edge_weight, require_edge_weight);
}


// format_kernel.cu
std::vector<torch::Tensor> csr2csc_cuda(
    torch::Tensor inPtr,
    torch::Tensor inInd,
    torch::Tensor inVal,
    int num_row
);


std::vector<torch::Tensor> csr2csc(
    torch::Tensor inPtr,
    torch::Tensor inInd,
    torch::Tensor inVal,
    int num_row
){
    return csr2csc_cuda(inPtr, inInd, inVal, num_row);
}

torch::Tensor coo2csr_cuda(
    torch::Tensor cooRowInd,
    int num_row
);

torch::Tensor coo2csr(
    torch::Tensor cooRowInd,
    int num_row
){
    return coo2csr_cuda(cooRowInd, num_row);
}


// gat_kernel.cu
std::vector<torch::Tensor> gat_gar_edge_weight_cuda(
    torch::Tensor e_pre,
    torch::Tensor src_ptr,
    torch::Tensor tar_index,
    float negative_slope
);


std::vector<torch::Tensor> gat_gar_edge_weight(
    torch::Tensor e_pre,
    torch::Tensor src_ptr,
    torch::Tensor tar_index,
    float negative_slope
){
    return gat_gar_edge_weight_cuda(e_pre, src_ptr, tar_index, negative_slope);
}


std::vector<torch::Tensor> gat_gas_edge_weight_cuda(
    torch::Tensor e_pre,
    torch::Tensor src_index,
    torch::Tensor tar_index,
    float negative_slope
);


std::vector<torch::Tensor> gat_gas_edge_weight(
    torch::Tensor e_pre,
    torch::Tensor src_index,
    torch::Tensor tar_index,
    float negative_slope
){
    return gat_gas_edge_weight_cuda(e_pre, src_index, tar_index, negative_slope);
}


std::vector<torch::Tensor> gat_gar_edge_weight_b_cuda(
    torch::Tensor grad_alpha_self,
    torch::Tensor grad_alpha,
    torch::Tensor src_index,
    torch::Tensor tar_index,
    torch::Tensor mask_lrelu,
    torch::Tensor mask_lrelu_self,
    torch::Tensor e,
    torch::Tensor e_self,
    torch::Tensor e_sum,
    torch::Tensor alpha_self,
    torch::Tensor alpha
);

std::vector<torch::Tensor> gat_gar_edge_weight_b(
    torch::Tensor grad_alpha_self,
    torch::Tensor grad_alpha,
    torch::Tensor src_index,
    torch::Tensor tar_index,
    torch::Tensor mask_lrelu,
    torch::Tensor mask_lrelu_self,
    torch::Tensor e,
    torch::Tensor e_self,
    torch::Tensor e_sum,
    torch::Tensor alpha_self,
    torch::Tensor alpha
){
    return gat_gar_edge_weight_b_cuda(grad_alpha_self, grad_alpha, src_index, tar_index, mask_lrelu, mask_lrelu_self, e, e_self, e_sum, alpha_self, alpha);
}


// gcn_kernel.cu
torch::Tensor gcn_gar_edge_weight_cuda(
    torch::Tensor src_index,
    torch::Tensor tar_ptr,
    torch::Tensor tar_index,
    int num_nodes,
    torch::optional<torch::Tensor> optional_edge_weight
);

torch::Tensor gcn_gar_edge_weight(
    torch::Tensor src_index,
    torch::Tensor tar_ptr,
    torch::Tensor tar_index,
    int num_nodes,
    torch::optional<torch::Tensor> optional_edge_weight
){
    return gcn_gar_edge_weight_cuda(src_index, tar_ptr, tar_index, num_nodes, optional_edge_weight);
}

torch::Tensor gcn_gas_edge_weight_cuda(
    torch::Tensor src_index,
    torch::Tensor tar_index,
    int num_nodes,
    torch::optional<torch::Tensor> optional_edge_weight
);


torch::Tensor gcn_gas_edge_weight(
    torch::Tensor src_index,
    torch::Tensor tar_index,
    int num_nodes,
    torch::optional<torch::Tensor> optional_edge_weight
){
    return gcn_gas_edge_weight_cuda(src_index, tar_index, num_nodes, optional_edge_weight);
}


// Single pybind11 module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // aggregate
    m.def("fused_gar_f", &fused_gar_f, "fused AGG GAR forward");
    m.def("fused_gar_b", &fused_gar_b, "fused AGG GAR backward");
    m.def("fused_gas_f", &fused_gas_f, "fused AGG GAS forward");
    m.def("fused_gas_b", &fused_gas_b, "fused AGG GAS backward");

    // format
    m.def("csr2csc", &csr2csc, "Converter between CSC and CSR");
    m.def("coo2csr", &coo2csr, "Convert COO to CSR");

    // gat
    m.def("gat_gar_edge_weight", &gat_gar_edge_weight, "gat_gar_edge_weight");
    m.def("gat_gas_edge_weight", &gat_gas_edge_weight, "gat_gas_edge_weight");
    m.def("gat_gar_edge_weight_b", &gat_gar_edge_weight_b, "gat_gar_edge_weight_b");

    // gcn
    m.def("gcn_gar_edge_weight", &gcn_gar_edge_weight, "gcn_gar_edge_weight");
    m.def("gcn_gas_edge_weight", &gcn_gas_edge_weight, "gcn_gas_edge_weight");
}
