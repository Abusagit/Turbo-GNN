#pragma once
#include <torch/extension.h>

#include <string>
#include <vector>

// out[v] = reduce over the edges e = (u -> v) of CSR row v of op(lhs[u], rhs[e]), for any feature
// width d. Returns {out, arg}: arg[v, f] is, for min/max, the CSR position of the winning edge (-1 for
// none), and empty for sum.
//
// slice_size: edges per slice of the heavy rows; chunk_node / chunk_start / node_chunk_offset are
// their slice table at that size (graph.walk_partition), built here at the cost of a host sync when
// empty. A table of other rows or another size is not detected and leaves outputs unwritten. light_degree_bound: an upper bound on the light
// rows' degrees, or -1 if unknown; it picks the walk's build (see reduction/reduction_launch.cuh).
//
// edge_map: backward-CSR edge position -> forward-CSR edge position. An empty tensor means the edge
// operand is indexed by the CSR being walked, which is what a forward pass wants; the sum backward of
// 'mul' / 'div' walks the transposed CSR and passes the map so it can read the operand in place.
std::vector<torch::Tensor> gspmm_forward(
    const torch::Tensor& edge_ptr,
    const torch::Tensor& edge_idx,
    const torch::Tensor& lhs,
    const torch::Tensor& rhs,
    const torch::Tensor& light_nodes,
    const torch::Tensor& heavy_nodes,
    const std::string& op,
    const std::string& reduce,
    int warps_per_block,
    int64_t slice_size,
    const torch::Tensor& chunk_node,
    const torch::Tensor& chunk_start,
    const torch::Tensor& node_chunk_offset,
    int64_t light_degree_bound,
    const torch::Tensor& edge_map
);

std::vector<torch::Tensor> gspmm_backward_arg(
    const torch::Tensor& grad_out,
    const torch::Tensor& arg_eid,
    const torch::Tensor& edge_idx,
    const torch::Tensor& bwd_edge_ptr,
    const torch::Tensor& bwd_edge_idx,
    const torch::Tensor& bwd_edge_map,
    const torch::Tensor& lhs,
    const torch::Tensor& rhs,
    const torch::Tensor& light_nodes,
    const torch::Tensor& heavy_nodes,
    const std::string& op,
    int warps_per_block    = 8,
    int features_per_block = 32,
    int tiles_y            = 8,
    // Largest out-degree of the transposed CSR (bwd_edge_ptr), gating the
    // heavy-bucket slicing of this walk; -1 means "unknown" and leaves slicing off.
    int max_degree = -1,
    // cp.async prefetch depth of the arg_eid gather, 0 disables.
    int pipeline_stages = 0
);

// Gradient w.r.t. the edge operand of a sum reduction, on the forward CSR's light rows and heavy
// slices (slice table as for gspmm_forward).
torch::Tensor gspmm_backward_edge(
    const torch::Tensor& edge_ptr,
    const torch::Tensor& edge_idx,
    const torch::Tensor& grad_out,
    const torch::Tensor& lhs,
    const torch::Tensor& rhs,
    const torch::Tensor& light_nodes,
    const torch::Tensor& heavy_nodes,
    const std::string& op,
    int warps_per_block,
    int64_t slice_size,
    const torch::Tensor& chunk_node,
    const torch::Tensor& chunk_start,
    const torch::Tensor& node_chunk_offset
);
