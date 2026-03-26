#pragma once
#include <torch/extension.h>
#include <vector>
#include <tuple>
#include <string>

// ============================================================================
// Reduction aggregation
// ============================================================================

std::vector<at::Tensor> reduction_aggr_forward_partitioned_torch(
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
    int tiles_y = 8,
    std::string reduce = "min"
);

at::Tensor reduction_aggr_backward_torch(
    at::Tensor grad_out,
    at::Tensor arg_idx,
    int64_t num_src_nodes,
    int warps_per_block = 8
);

// ============================================================================
// GATv2 aggregation
// ============================================================================

std::vector<torch::Tensor> gatv2_forward_cuda(
    torch::Tensor l,
    torch::Tensor r,
    torch::Tensor row_ptr,
    torch::Tensor col_idx,
    torch::Tensor attn_vec,
    float negative_slope
);

std::vector<torch::Tensor> gatv2_backward_cuda(
    torch::Tensor grad_h,
    torch::Tensor l,
    torch::Tensor r,
    torch::Tensor row_ptr,
    torch::Tensor col_idx,
    torch::Tensor row_ptr_T,
    torch::Tensor col_idx_T,
    torch::Tensor attn_vec,
    torch::Tensor logsumexp,
    float negative_slope,
    int grad_A_reduce_row_chunk_size,
    bool is_directed = true
);

// ============================================================================
// Graph Transformer aggregation
// ============================================================================

std::tuple<torch::Tensor, torch::Tensor>
graph_attention_forward_csr_mh_cuda(
    torch::Tensor row_ptr,
    torch::Tensor col_idx,
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    float scale
);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
graph_attention_backward_csr_mh_cuda(
    torch::Tensor row_ptr,       // forward CSR [N+1]
    torch::Tensor col_idx,       // forward CSR [E]
    torch::Tensor row_ptr_T,     // backward CSR^T [N+1]
    torch::Tensor col_idx_T,     // backward CSR^T [E]
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor O,
    torch::Tensor dO,
    torch::Tensor logsumexp,
    float scale,
    bool is_directed = true
);

// ============================================================================
// Edge normalization kernels
// ============================================================================

void launch_compute_degrees(
    const torch::Tensor& indptr,
    const torch::Tensor& indices,
    torch::Tensor& in_degrees,
    torch::Tensor& out_degrees,
    int block_dim
);
