"""Re-export shim: imports from turbo_gnn."""

import turbo_gnn._C as reduction_aggr_cuda
from turbo_gnn._autotune import TunableKernel, TunableParam, with_autotune
from turbo_gnn._functions import ReductionAggrFunction, csr_SPMM_normalized
from turbo_gnn._kernels import ReductionAggrKernel
from turbo_gnn.graph import AdjacencyForwardBackwardWithNodeBuckets
from turbo_gnn.ops import reduction_aggr


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
    return reduction_aggr_cuda.reduction_aggr_forward_partitioned(
        edge_ptr,
        edge_idx,
        X,
        light,
        heavy,
        131070,
        warps_per_block,
        edges_per_block_heavy_nodes,
        use_2d_kernel,
        features_per_block,
        tiles_y,
        reduce,
    )
