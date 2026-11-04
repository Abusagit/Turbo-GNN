#include "common.cuh"

// The Python-facing entry points, with the argument lists turbo_gnn has always exposed. The
// implementation (reduction_aggr.cu) takes only what it uses.

std::vector<at::Tensor> reduction_aggr_forward_cuda(
    at::Tensor const& edge_ptr,
    at::Tensor const& edge_idx,
    at::Tensor X,
    at::Tensor const& light_nodes,
    at::Tensor const& heavy_nodes,
    int warps_per_block,
    int64_t slice_size,
    std::string const& reduce,
    at::Tensor const& chunk_node,
    at::Tensor const& chunk_start,
    at::Tensor const& node_chunk_offset,
    int64_t light_degree_bound
);

at::Tensor reduction_aggr_backward_cuda(at::Tensor grad_out, at::Tensor const& arg_idx, int64_t num_src_nodes);

at::Tensor reduction_aggr_backward_torch(at::Tensor grad_out, at::Tensor arg_idx, int64_t num_src_nodes, int /*warps_per_block*/) {
    return reduction_aggr_backward_cuda(grad_out, arg_idx, num_src_nodes);
}

// max_degree bounds the degree of the light rows -- any larger value is correct, -1 means unknown --
// and picks the forward's build (see reduction_aggr_forward_cuda). edges_per_block_heavy_nodes is
// the slice size; chunk_node / chunk_start / node_chunk_offset, the heavy bucket's slice table at
// that size, are built here when not given. Arguments of the earlier kernels that no longer mean
// anything are accepted and ignored: use_2d_kernel / features_per_block / tiles_y (the tiled heavy
// kernel; heavy rows are now always edge-sliced) and pipeline_stages (the per-thread cp.async scan,
// measured slower than none).
std::vector<at::Tensor> reduction_aggr_forward_partitioned_torch(
    at::Tensor edge_ptr,
    at::Tensor edge_idx,
    at::Tensor X,
    at::Tensor light_nodes,
    at::Tensor heavy_nodes,
    int max_degree,
    int warps_per_block,
    int edges_per_block_heavy_nodes,
    bool /*use_2d_kernel*/,
    int /*features_per_block*/,
    int /*tiles_y*/,
    std::string reduce,
    int /*pipeline_stages*/,
    at::Tensor chunk_node,
    at::Tensor chunk_start,
    at::Tensor node_chunk_offset
) {
    return reduction_aggr_forward_cuda(
        edge_ptr, edge_idx, X, light_nodes, heavy_nodes, warps_per_block, edges_per_block_heavy_nodes, reduce, chunk_node, chunk_start,
        node_chunk_offset, max_degree
    );
}
