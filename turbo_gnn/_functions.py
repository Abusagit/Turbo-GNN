"""All torch.autograd.Function subclasses wrapping turbo_gnn._C kernels."""

from __future__ import annotations

import warnings
from math import ceil

import torch

import turbo_gnn._C as _C

WARP_SIZE = 32
FOUR_BYTES_CONSTANT = 4


def _next_power_of_two(x):
    x -= 1
    x |= x >> 1
    x |= x >> 2
    x |= x >> 4
    x |= x >> 8
    x |= x >> 16
    x += 1
    return x


class ReductionAggrFunction(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx,
        edge_ptr,
        edge_idx,
        X,
        light,
        heavy,
        max_degree,
        warps_per_block,
        edges_per_block_heavy_nodes,
        use_2d_kernel=False,
        features_per_block=32,
        tiles_y=8,
        reduce="min",
    ):
        if torch.is_autocast_enabled():
            X = X.to(torch.get_autocast_gpu_dtype())

        num_of_threads_invoked = WARP_SIZE * warps_per_block
        num_features_per_thread = FOUR_BYTES_CONSTANT // X.dtype.itemsize

        num_threads_needed = ceil(X.shape[-1] / num_features_per_thread)

        if num_threads_needed < num_of_threads_invoked:
            warps_per_block_needed = ceil(num_threads_needed / WARP_SIZE)
            warnings.warn(
                f"Number of threads involved for ReductionAggr is {num_of_threads_invoked} "
                f"({warps_per_block} warps per thread block requested). "
                f"However, number of threads needed is {num_threads_needed} "
                f"({warps_per_block_needed} warps). Setting this value instead."
            )

            warps_per_block = warps_per_block_needed
            if warps_per_block not in {1, 2, 4, 8, 16, 32, 64}:
                warps_per_block = _next_power_of_two(warps_per_block)

        out, arg_idx = _C.reduction_aggr_forward_partitioned(
            edge_ptr,
            edge_idx,
            X,
            light,
            heavy,
            max_degree,
            warps_per_block,
            edges_per_block_heavy_nodes,
            use_2d_kernel,
            features_per_block,
            tiles_y,
            reduce,
        )
        ctx.save_for_backward(arg_idx)
        ctx.num_src_nodes = X.size(0)
        ctx.warps_per_block = warps_per_block
        return out

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_out):
        (arg_idx,) = ctx.saved_tensors
        num_src_nodes = ctx.num_src_nodes
        grad_x = _C.reduction_aggr_backward(grad_out, arg_idx, num_src_nodes, ctx.warps_per_block)
        return None, None, grad_x, None, None, None, None, None, None, None, None, None


class gatv2_function(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx,
        indptr_forward,
        indices_forward,
        indptr_backward,
        indices_backward,
        x_left,
        x_right,
        attention_weights,
        negative_slope,
        grad_A_reduce_row_chunk_size,
    ):
        if torch.is_autocast_enabled():
            attention_weights = attention_weights.to(torch.get_autocast_gpu_dtype())

        output, logsumexp = _C.gatv2_forward(
            x_left,
            x_right,
            indptr_forward,
            indices_forward,
            attention_weights,
            negative_slope,
        )
        ctx.negative_slope = negative_slope
        ctx.grad_A_reduce_row_chunk_size = grad_A_reduce_row_chunk_size
        ctx.heads = x_left.shape[1]
        ctx.head_dim = x_left.shape[2]

        ctx.save_for_backward(
            x_left,
            x_right,
            indptr_forward,
            indices_forward,
            indptr_backward,
            indices_backward,
            attention_weights,
            logsumexp,
        )

        return output

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        (
            x_left,
            x_right,
            indptr_forward,
            indices_forward,
            indptr_backward,
            indices_backward,
            attention_weights,
            logsumexp,
        ) = ctx.saved_tensors

        num_heads = ctx.heads
        head_dim = ctx.head_dim

        grad_output = grad_output.view(-1, num_heads, head_dim)

        negative_slope = ctx.negative_slope
        grad_A_reduce_row_chunk_size = ctx.grad_A_reduce_row_chunk_size
        grad_x_left, grad_x_right, grad_attention = _C.gatv2_backward(
            grad_output,
            x_left,
            x_right,
            indptr_forward,
            indices_forward,
            indptr_backward,
            indices_backward,
            attention_weights,
            logsumexp,
            negative_slope,
            grad_A_reduce_row_chunk_size,
        )

        return None, None, None, None, grad_x_left, grad_x_right, grad_attention, None, None


class _FusedGraphAttention(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, edge_ptr, edge_idx, edge_ptr_T, edge_idx_T, Q, K, V, scale):
        out, logsumexp = _C.gt_forward_csr_mh(edge_ptr, edge_idx, Q, K, V, scale)

        ctx.scale = scale
        ctx.num_heads = Q.shape[1]
        ctx.head_dim = Q.shape[2]
        ctx.save_for_backward(edge_ptr_T, edge_idx_T, Q, K, V, out, logsumexp)
        return out

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        edge_ptr_T, edge_idx_T, Q, K, V, out, logsumexp = ctx.saved_tensors
        scale = ctx.scale
        num_heads = ctx.num_heads
        head_dim = ctx.head_dim
        grad_output = grad_output.view(-1, num_heads, head_dim)

        dQ, dK, dV = _C.gt_backward_csr_mh(edge_ptr_T, edge_idx_T, Q, K, V, out, grad_output, logsumexp, scale)
        return None, None, None, None, dQ, dK, dV, None


class _CudaSpMMConvFn(torch.autograd.Function):
    """cuSPARSE SpMM with AdjacencyForwardBackwardWithNodeBuckets graph format."""

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, x, forward_indptr, forward_indices, norm_type, cu_sparse_algorithm_id, block_dim):
        ctx.save_for_backward(forward_indptr, forward_indices)
        ctx.norm_type = norm_type
        ctx.cu_sparse_algorithm_id = cu_sparse_algorithm_id
        ctx.block_dim = block_dim

        return csr_SPMM_normalized(
            indptr=forward_indptr,
            indices=forward_indices,
            features=x,
            edge_weights=None,
            norm=norm_type,
            algorithm=cu_sparse_algorithm_id,
            do_transpose_a=False,
            block_dim=block_dim,
        )

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, *grad_outputs):
        forward_indptr, forward_indices = ctx.saved_tensors
        grad_x = csr_SPMM_normalized(
            indptr=forward_indptr,
            indices=forward_indices,
            features=grad_outputs[0],
            edge_weights=None,
            norm=ctx.norm_type,
            algorithm=ctx.cu_sparse_algorithm_id,
            do_transpose_a=True,
            block_dim=ctx.block_dim,
        )
        return grad_x, None, None, None, None, None


def csr_SPMM_normalized(
    indptr,
    indices,
    features,
    edge_weights=None,
    norm="none",
    algorithm=-1,
    use_cache=True,
    do_transpose_a=False,
    block_dim=256,
):
    """Normalized SpMM operation supporting different GCN normalization schemes."""
    if edge_weights is None:
        edge_weights_gpu = torch.empty(0, device=features.device, dtype=torch.float32)
    else:
        edge_weights_gpu = edge_weights.to(device=features.device, dtype=torch.float32)

    out = _C.csr_SPMM_normalized(
        indptr, indices, features.contiguous(), edge_weights_gpu, norm, algorithm, use_cache, do_transpose_a, block_dim
    )

    return out
