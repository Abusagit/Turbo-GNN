"""Re-export shim: imports from turbo_gnn."""

import torch

import turbo_gnn._C as reduction_aggr_cuda
from turbo_gnn._autotune import TunableKernel, TunableParam, with_autotune
from turbo_gnn._functions import ReductionAggrFunction, csr_SPMM_normalized
from turbo_gnn._kernels import ReductionAggrKernel
from turbo_gnn.graph import AdjacencyForwardBackwardWithNodeBuckets
from turbo_gnn.ops import reduction_aggr


def _chunk_offsets(edge_ptr, heavy, edges_per_block):
    offsets = torch.zeros(heavy.numel() + 1, dtype=torch.int32, device=heavy.device)
    if heavy.numel() > 0:
        indptr = edge_ptr.long()
        degrees = indptr[1:] - indptr[:-1]
        chunks = (degrees[heavy.long()] + (edges_per_block - 1)).div(edges_per_block, rounding_mode="floor")
        offsets[1:] = torch.cumsum(chunks, 0, dtype=torch.int32)
    return offsets, int(offsets[-1].item())


def reduction_aggr_forward_partitioned(
    edge_ptr,
    edge_idx,
    X,
    light,
    heavy,
    warps_per_block,
    edges_per_block_heavy_nodes,
    use_2d_kernel=False,
    features_per_block=32,
    tiles_y=8,
    reduce="min",
):
    chunk_offsets, total_chunks = _chunk_offsets(edge_ptr, heavy, edges_per_block_heavy_nodes)
    return reduction_aggr_cuda.reduction_aggr_forward_partitioned(
        edge_ptr,
        edge_idx,
        X,
        light,
        heavy,
        131070,
        chunk_offsets,
        total_chunks,
        warps_per_block,
        edges_per_block_heavy_nodes,
        use_2d_kernel,
        features_per_block,
        tiles_y,
        reduce,
    )
