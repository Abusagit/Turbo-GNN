#pragma once
#include <torch/extension.h>

#include <string>
#include <vector>

std::vector<torch::Tensor> gspmm_forward(
    const torch::Tensor& edge_ptr,
    const torch::Tensor& edge_idx,
    const torch::Tensor& lhs,
    const torch::Tensor& rhs,
    const torch::Tensor& light_nodes,
    const torch::Tensor& heavy_nodes,
    const std::string& op,
    const std::string& reduce,
    int warps_per_block            = 8,
    int features_per_block         = 32,
    int tiles_y                    = 8,
    int pipeline_stages            = 0,
    // Backward-CSR edge position -> forward-CSR edge position.  An empty (or
    // undefined) tensor means the edge operand is indexed by the CSR being
    // walked, which is what a forward pass wants; a transposed pass passes the
    // map so it can read the operand in place.  Only 'mul'/'div' with
    // reduce='sum' accept one.
    //
    // The pybind binding hands over an *empty* tensor rather than relying on
    // this default: an undefined one renders as a Python None that pybind then
    // cannot cast back to a Tensor reference.
    const torch::Tensor& edge_map  = {},
    // Largest in-degree of the CSR being walked, which decides whether the
    // heavy bucket is worth slicing across blocks.  -1 means "unknown", and
    // slicing is then left off -- reading it off the device here would cost a
    // sync on every call.
    int max_degree                 = -1
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
    // heavy-bucket slicing exactly as max_degree does in gspmm_forward; -1
    // means "unknown" and leaves slicing off.
    int max_degree         = -1,
    // cp.async prefetch depth of the arg_eid gather, 0 disables.
    int pipeline_stages    = 0
);

// Gradient w.r.t. the edge operand of a sum reduction, walking the forward CSR
// with its light/heavy buckets.  max_degree sizes the heavy grid's chunk axis;
// -1 means "unknown" and the blocks stride over the chunks instead.
torch::Tensor gspmm_backward_edge(
    const torch::Tensor& edge_ptr,
    const torch::Tensor& edge_idx,
    const torch::Tensor& grad_out,
    const torch::Tensor& lhs,
    const torch::Tensor& rhs,
    const torch::Tensor& light_nodes,
    const torch::Tensor& heavy_nodes,
    const std::string& op,
    int warps_per_block    = 8,
    int features_per_block = 32,
    int tiles_y            = 8,
    int max_degree         = -1,
    int pipeline_stages    = 0
);
