// Routes each gt entry point to the shard compiled for the input's dtype.
#include <torch/extension.h>

#include <tuple>
#include <vector>

std::tuple<torch::Tensor, torch::Tensor> graph_attention_forward_csr_mh_cuda_f32(
    torch::Tensor row_ptr,
    torch::Tensor col_idx,
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    float scale,
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
std::tuple<torch::Tensor, torch::Tensor> graph_attention_forward_csr_mh_cuda_f16(
    torch::Tensor row_ptr,
    torch::Tensor col_idx,
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    float scale,
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
std::tuple<torch::Tensor, torch::Tensor> graph_attention_forward_csr_mh_cuda_bf16(
    torch::Tensor row_ptr,
    torch::Tensor col_idx,
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    float scale,
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

std::tuple<torch::Tensor, torch::Tensor> graph_attention_forward_csr_mh_cuda(
    torch::Tensor row_ptr,
    torch::Tensor col_idx,
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    float scale,
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
    switch (Q.scalar_type()) {
        case at::kFloat:
            return graph_attention_forward_csr_mh_cuda_f32(row_ptr, col_idx, Q, K, V, scale, light_nodes, heavy_nodes, light_warps_per_block, heavy_warps_per_block, schedule, blocks_per_sm, sched_chunk, bucket_launch, chunk_node, chunk_start, node_chunk_offset, heavy_edge_slice, pipeline_stages, heavy_pipeline_stages);
        case at::kHalf:
            return graph_attention_forward_csr_mh_cuda_f16(row_ptr, col_idx, Q, K, V, scale, light_nodes, heavy_nodes, light_warps_per_block, heavy_warps_per_block, schedule, blocks_per_sm, sched_chunk, bucket_launch, chunk_node, chunk_start, node_chunk_offset, heavy_edge_slice, pipeline_stages, heavy_pipeline_stages);
        case at::kBFloat16:
            return graph_attention_forward_csr_mh_cuda_bf16(row_ptr, col_idx, Q, K, V, scale, light_nodes, heavy_nodes, light_warps_per_block, heavy_warps_per_block, schedule, blocks_per_sm, sched_chunk, bucket_launch, chunk_node, chunk_start, node_chunk_offset, heavy_edge_slice, pipeline_stages, heavy_pipeline_stages);
        default:
            TORCH_CHECK(false, "graph_attention_forward_csr_mh_cuda: unsupported dtype ", Q.scalar_type());
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> graph_attention_backward_csr_mh_cuda_f32(
    torch::Tensor row_ptr,
    torch::Tensor col_idx,
    torch::Tensor row_ptr_T,
    torch::Tensor col_idx_T,
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor O,
    torch::Tensor dO,
    torch::Tensor logsumexp,
    float scale,
    torch::Tensor light_nodes,
    torch::Tensor heavy_nodes,
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
    int heavy_edge_slice,
    int pipeline_stages,
    int backward_heavy_pipeline_stages
);
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> graph_attention_backward_csr_mh_cuda_f16(
    torch::Tensor row_ptr,
    torch::Tensor col_idx,
    torch::Tensor row_ptr_T,
    torch::Tensor col_idx_T,
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor O,
    torch::Tensor dO,
    torch::Tensor logsumexp,
    float scale,
    torch::Tensor light_nodes,
    torch::Tensor heavy_nodes,
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
    int heavy_edge_slice,
    int pipeline_stages,
    int backward_heavy_pipeline_stages
);
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> graph_attention_backward_csr_mh_cuda_bf16(
    torch::Tensor row_ptr,
    torch::Tensor col_idx,
    torch::Tensor row_ptr_T,
    torch::Tensor col_idx_T,
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor O,
    torch::Tensor dO,
    torch::Tensor logsumexp,
    float scale,
    torch::Tensor light_nodes,
    torch::Tensor heavy_nodes,
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
    int heavy_edge_slice,
    int pipeline_stages,
    int backward_heavy_pipeline_stages
);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> graph_attention_backward_csr_mh_cuda(
    torch::Tensor row_ptr,
    torch::Tensor col_idx,
    torch::Tensor row_ptr_T,
    torch::Tensor col_idx_T,
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor O,
    torch::Tensor dO,
    torch::Tensor logsumexp,
    float scale,
    torch::Tensor light_nodes,
    torch::Tensor heavy_nodes,
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
    int heavy_edge_slice,
    int pipeline_stages,
    int backward_heavy_pipeline_stages
) {
    switch (Q.scalar_type()) {
        case at::kFloat:
            return graph_attention_backward_csr_mh_cuda_f32(row_ptr, col_idx, row_ptr_T, col_idx_T, Q, K, V, O, dO, logsumexp, scale, light_nodes, heavy_nodes, light_warps_per_block, heavy_warps_per_block, is_directed, schedule, blocks_per_sm, sched_chunk, bucket_launch, chunk_node, chunk_start, node_chunk_offset, heavy_edge_slice, pipeline_stages, backward_heavy_pipeline_stages);
        case at::kHalf:
            return graph_attention_backward_csr_mh_cuda_f16(row_ptr, col_idx, row_ptr_T, col_idx_T, Q, K, V, O, dO, logsumexp, scale, light_nodes, heavy_nodes, light_warps_per_block, heavy_warps_per_block, is_directed, schedule, blocks_per_sm, sched_chunk, bucket_launch, chunk_node, chunk_start, node_chunk_offset, heavy_edge_slice, pipeline_stages, backward_heavy_pipeline_stages);
        case at::kBFloat16:
            return graph_attention_backward_csr_mh_cuda_bf16(row_ptr, col_idx, row_ptr_T, col_idx_T, Q, K, V, O, dO, logsumexp, scale, light_nodes, heavy_nodes, light_warps_per_block, heavy_warps_per_block, is_directed, schedule, blocks_per_sm, sched_chunk, bucket_launch, chunk_node, chunk_start, node_chunk_offset, heavy_edge_slice, pipeline_stages, backward_heavy_pipeline_stages);
        default:
            TORCH_CHECK(false, "graph_attention_backward_csr_mh_cuda: unsupported dtype ", Q.scalar_type());
    }
}
