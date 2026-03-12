#pragma once
#include <torch/extension.h>
#include <string>

torch::Tensor csr_SPMM(
    const torch::Tensor &indptr,
    const torch::Tensor &indices,
    const torch::Tensor &features,
    int algorithm = -1,
    bool use_cache = true,
    bool do_transpose_a = false,
    int block_dim = 256
);

torch::Tensor csr_SPMM_normalized(
    const torch::Tensor &indptr,
    const torch::Tensor &indices,
    const torch::Tensor &features,
    const torch::Tensor &edge_weights,
    const std::string &norm_str = "none",
    int algorithm = -1,
    bool use_cache = true,
    bool do_transpose_a = false,
    int block_dim = 256
);

int find_best_algorithm(
    const torch::Tensor &indptr,
    const torch::Tensor &indices,
    const torch::Tensor &features,
    int block_dim = 256
);

int find_best_algorithm_normalized(
    const torch::Tensor &indptr,
    const torch::Tensor &indices,
    const torch::Tensor &features,
    const torch::Tensor &edge_weights,
    const std::string &norm_str = "none",
    int block_dim = 256
);

void clear_graph_cache();
