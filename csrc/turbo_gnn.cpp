#include "kernels.cuh"
#include "spmm/cusparse_spmm.h"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Reduction aggregation
    m.def("reduction_aggr_forward_partitioned", &reduction_aggr_forward_partitioned_torch,
          "Reduction aggregation forward (partitioned)",
          py::arg("edge_ptr"), py::arg("edge_idx"), py::arg("X"),
          py::arg("light_nodes"), py::arg("heavy_nodes"), py::arg("max_degree"),
          py::arg("warps_per_block") = 8, py::arg("edges_per_block_heavy_nodes") = 128,
          py::arg("use_2d_kernel") = false, py::arg("features_per_block") = 32,
          py::arg("tiles_y") = 8, py::arg("reduce") = "min");

    m.def("reduction_aggr_backward", &reduction_aggr_backward_torch,
          "Reduction aggregation backward",
          py::arg("grad_out"), py::arg("arg_idx"), py::arg("num_src_nodes"),
          py::arg("warps_per_block") = 8);

    // GATv2 aggregation
    m.def("gatv2_forward", &gatv2_forward_cuda,
          "GATv2 forward pass (CUDA)",
          py::arg("l"), py::arg("r"), py::arg("row_ptr"), py::arg("col_idx"),
          py::arg("attn_vec"), py::arg("negative_slope") = 0.2f);

    m.def("gatv2_backward", &gatv2_backward_cuda,
          "GATv2 backward pass (CUDA)",
          py::arg("grad_h"), py::arg("l"), py::arg("r"),
          py::arg("row_ptr"), py::arg("col_idx"),
          py::arg("row_ptr_T"), py::arg("col_idx_T"),
          py::arg("attn_vec"), py::arg("logsumexp"),
          py::arg("negative_slope") = 0.2f,
          py::arg("grad_A_reduce_row_chunk_size") = 512);

    // Graph Transformer aggregation
    m.def("gt_forward_csr_mh", &graph_attention_forward_csr_mh_cuda,
          "Graph Transformer forward (CSR, multi-head)",
          py::arg("row_ptr"), py::arg("col_idx"),
          py::arg("Q"), py::arg("K"), py::arg("V"), py::arg("scale"));

    m.def("gt_backward_csr_mh", &graph_attention_backward_csr_mh_cuda,
          "Graph Transformer backward (CSR^T, multi-head)",
          py::arg("row_ptr_T"), py::arg("col_idx_T"),
          py::arg("Q"), py::arg("K"), py::arg("V"),
          py::arg("O"), py::arg("dO"), py::arg("logsumexp"),
          py::arg("scale"));

    // SpMM
    m.def("csr_SPMM_normalized", &csr_SPMM_normalized,
          "Optimized and cached csr_SPMM with normalization",
          py::arg("indptr"), py::arg("indices"), py::arg("features"),
          py::arg("edge_weights"), py::arg("norm") = "none",
          py::arg("algorithm") = -1, py::arg("use_cache") = true,
          py::arg("do_transpose_a") = false, py::arg("block_dim") = 256);

    m.def("csr_SPMM", &csr_SPMM,
          "Optimized and cached csr_SPMM (backward compatibility)",
          py::arg("indptr"), py::arg("indices"), py::arg("features"),
          py::arg("algorithm") = -1, py::arg("use_cache") = true,
          py::arg("do_transpose_a") = false, py::arg("block_dim") = 256);

    m.def("find_best_algorithm_normalized", &find_best_algorithm_normalized,
          "Find best cuSPARSE algorithm for given graph with normalization",
          py::arg("indptr"), py::arg("indices"), py::arg("features"),
          py::arg("edge_weights"), py::arg("norm") = "none",
          py::arg("block_dim") = 256);

    m.def("find_best_algorithm", &find_best_algorithm,
          "Find best cuSPARSE algorithm for given graph (backward compatibility)",
          py::arg("indptr"), py::arg("indices"), py::arg("features"),
          py::arg("block_dim") = 256);

    m.def("clear_graph_cache", &clear_graph_cache, "Clear graph cache");

    // Edge normalization
    m.def("compute_degrees", &launch_compute_degrees,
          "Compute in-degrees and out-degrees from CSR",
          py::arg("indptr"), py::arg("indices"),
          py::arg("in_degrees"), py::arg("out_degrees"),
          py::arg("block_dim") = 256);
}
