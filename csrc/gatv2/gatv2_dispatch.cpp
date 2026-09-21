// Routes each gatv2 entry point to the shard compiled for the input's dtype.
#include <torch/extension.h>

#include <tuple>
#include <vector>

std::vector<torch::Tensor> gatv2_forward_cuda_f32(
    torch::Tensor l,
    torch::Tensor r,
    torch::Tensor row_ptr,
    torch::Tensor col_idx,
    torch::Tensor attn_vec,
    float negative_slope,
    torch::Tensor light_nodes,
    torch::Tensor heavy_nodes,
    int light_warps_per_block,
    int heavy_warps_per_block,
    int schedule,
    int blocks_per_sm,
    int sched_chunk,
    int bucket_launch,
    torch::Tensor chunk_node,
    torch::Tensor chunk_start,
    torch::Tensor node_chunk_offset,
    int heavy_edge_slice,
    int pipeline_stages,
    int heavy_pipeline_stages
);
std::vector<torch::Tensor> gatv2_forward_cuda_f16(
    torch::Tensor l,
    torch::Tensor r,
    torch::Tensor row_ptr,
    torch::Tensor col_idx,
    torch::Tensor attn_vec,
    float negative_slope,
    torch::Tensor light_nodes,
    torch::Tensor heavy_nodes,
    int light_warps_per_block,
    int heavy_warps_per_block,
    int schedule,
    int blocks_per_sm,
    int sched_chunk,
    int bucket_launch,
    torch::Tensor chunk_node,
    torch::Tensor chunk_start,
    torch::Tensor node_chunk_offset,
    int heavy_edge_slice,
    int pipeline_stages,
    int heavy_pipeline_stages
);
std::vector<torch::Tensor> gatv2_forward_cuda_bf16(
    torch::Tensor l,
    torch::Tensor r,
    torch::Tensor row_ptr,
    torch::Tensor col_idx,
    torch::Tensor attn_vec,
    float negative_slope,
    torch::Tensor light_nodes,
    torch::Tensor heavy_nodes,
    int light_warps_per_block,
    int heavy_warps_per_block,
    int schedule,
    int blocks_per_sm,
    int sched_chunk,
    int bucket_launch,
    torch::Tensor chunk_node,
    torch::Tensor chunk_start,
    torch::Tensor node_chunk_offset,
    int heavy_edge_slice,
    int pipeline_stages,
    int heavy_pipeline_stages
);

std::vector<torch::Tensor> gatv2_forward_cuda(
    torch::Tensor l,
    torch::Tensor r,
    torch::Tensor row_ptr,
    torch::Tensor col_idx,
    torch::Tensor attn_vec,
    float negative_slope,
    torch::Tensor light_nodes,
    torch::Tensor heavy_nodes,
    int light_warps_per_block,
    int heavy_warps_per_block,
    int schedule,
    int blocks_per_sm,
    int sched_chunk,
    int bucket_launch,
    torch::Tensor chunk_node,
    torch::Tensor chunk_start,
    torch::Tensor node_chunk_offset,
    int heavy_edge_slice,
    int pipeline_stages,
    int heavy_pipeline_stages
) {
    switch (l.scalar_type()) {
        case at::kFloat:
            return gatv2_forward_cuda_f32(l, r, row_ptr, col_idx, attn_vec, negative_slope, light_nodes, heavy_nodes, light_warps_per_block, heavy_warps_per_block, schedule, blocks_per_sm, sched_chunk, bucket_launch, chunk_node, chunk_start, node_chunk_offset, heavy_edge_slice, pipeline_stages, heavy_pipeline_stages);
        case at::kHalf:
            return gatv2_forward_cuda_f16(l, r, row_ptr, col_idx, attn_vec, negative_slope, light_nodes, heavy_nodes, light_warps_per_block, heavy_warps_per_block, schedule, blocks_per_sm, sched_chunk, bucket_launch, chunk_node, chunk_start, node_chunk_offset, heavy_edge_slice, pipeline_stages, heavy_pipeline_stages);
        case at::kBFloat16:
            return gatv2_forward_cuda_bf16(l, r, row_ptr, col_idx, attn_vec, negative_slope, light_nodes, heavy_nodes, light_warps_per_block, heavy_warps_per_block, schedule, blocks_per_sm, sched_chunk, bucket_launch, chunk_node, chunk_start, node_chunk_offset, heavy_edge_slice, pipeline_stages, heavy_pipeline_stages);
        default:
            TORCH_CHECK(false, "gatv2_forward_cuda: unsupported dtype ", l.scalar_type());
    }
}

std::vector<torch::Tensor> gatv2_backward_cuda_f32(
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
    torch::Tensor fwd_light_nodes,
    torch::Tensor fwd_heavy_nodes,
    torch::Tensor bwd_light_nodes,
    torch::Tensor bwd_heavy_nodes,
    int light_warps_per_block,
    int heavy_warps_per_block,
    bool is_directed,
    int schedule,
    int blocks_per_sm,
    int sched_chunk,
    int bucket_launch,
    torch::Tensor chunk_node,
    torch::Tensor chunk_start,
    torch::Tensor node_chunk_offset,
    int backward_heavy_edge_slice,
    int pipeline_stages,
    int backward_heavy_pipeline_stages
);
std::vector<torch::Tensor> gatv2_backward_cuda_f16(
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
    torch::Tensor fwd_light_nodes,
    torch::Tensor fwd_heavy_nodes,
    torch::Tensor bwd_light_nodes,
    torch::Tensor bwd_heavy_nodes,
    int light_warps_per_block,
    int heavy_warps_per_block,
    bool is_directed,
    int schedule,
    int blocks_per_sm,
    int sched_chunk,
    int bucket_launch,
    torch::Tensor chunk_node,
    torch::Tensor chunk_start,
    torch::Tensor node_chunk_offset,
    int backward_heavy_edge_slice,
    int pipeline_stages,
    int backward_heavy_pipeline_stages
);
std::vector<torch::Tensor> gatv2_backward_cuda_bf16(
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
    torch::Tensor fwd_light_nodes,
    torch::Tensor fwd_heavy_nodes,
    torch::Tensor bwd_light_nodes,
    torch::Tensor bwd_heavy_nodes,
    int light_warps_per_block,
    int heavy_warps_per_block,
    bool is_directed,
    int schedule,
    int blocks_per_sm,
    int sched_chunk,
    int bucket_launch,
    torch::Tensor chunk_node,
    torch::Tensor chunk_start,
    torch::Tensor node_chunk_offset,
    int backward_heavy_edge_slice,
    int pipeline_stages,
    int backward_heavy_pipeline_stages
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
    torch::Tensor fwd_light_nodes,
    torch::Tensor fwd_heavy_nodes,
    torch::Tensor bwd_light_nodes,
    torch::Tensor bwd_heavy_nodes,
    int light_warps_per_block,
    int heavy_warps_per_block,
    bool is_directed,
    int schedule,
    int blocks_per_sm,
    int sched_chunk,
    int bucket_launch,
    torch::Tensor chunk_node,
    torch::Tensor chunk_start,
    torch::Tensor node_chunk_offset,
    int backward_heavy_edge_slice,
    int pipeline_stages,
    int backward_heavy_pipeline_stages
) {
    switch (l.scalar_type()) {
        case at::kFloat:
            return gatv2_backward_cuda_f32(grad_h, l, r, row_ptr, col_idx, row_ptr_T, col_idx_T, attn_vec, logsumexp, negative_slope, grad_A_reduce_row_chunk_size, fwd_light_nodes, fwd_heavy_nodes, bwd_light_nodes, bwd_heavy_nodes, light_warps_per_block, heavy_warps_per_block, is_directed, schedule, blocks_per_sm, sched_chunk, bucket_launch, chunk_node, chunk_start, node_chunk_offset, backward_heavy_edge_slice, pipeline_stages, backward_heavy_pipeline_stages);
        case at::kHalf:
            return gatv2_backward_cuda_f16(grad_h, l, r, row_ptr, col_idx, row_ptr_T, col_idx_T, attn_vec, logsumexp, negative_slope, grad_A_reduce_row_chunk_size, fwd_light_nodes, fwd_heavy_nodes, bwd_light_nodes, bwd_heavy_nodes, light_warps_per_block, heavy_warps_per_block, is_directed, schedule, blocks_per_sm, sched_chunk, bucket_launch, chunk_node, chunk_start, node_chunk_offset, backward_heavy_edge_slice, pipeline_stages, backward_heavy_pipeline_stages);
        case at::kBFloat16:
            return gatv2_backward_cuda_bf16(grad_h, l, r, row_ptr, col_idx, row_ptr_T, col_idx_T, attn_vec, logsumexp, negative_slope, grad_A_reduce_row_chunk_size, fwd_light_nodes, fwd_heavy_nodes, bwd_light_nodes, bwd_heavy_nodes, light_warps_per_block, heavy_warps_per_block, is_directed, schedule, blocks_per_sm, sched_chunk, bucket_launch, chunk_node, chunk_start, node_chunk_offset, backward_heavy_edge_slice, pipeline_stages, backward_heavy_pipeline_stages);
        default:
            TORCH_CHECK(false, "gatv2_backward_cuda: unsupported dtype ", l.scalar_type());
    }
}
