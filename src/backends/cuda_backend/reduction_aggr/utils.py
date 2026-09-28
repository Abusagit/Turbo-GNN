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
    pipeline_stages=0,
):
    # Keyword, not positional: the binding takes schedule/blocks_per_sm/sched_chunk/
    # bucket_launch/chunk_node/chunk_start/heavy_edge_slice between `reduce` and
    # `pipeline_stages`, so passing the depth positionally landed it in `schedule` six
    # slots early. Depth 2 is a legal schedule id, so it silently ran the baseline at a
    # different schedule and the test comparing it against depth 0 passed while proving
    # nothing; depth 6 is out of range and raised.
    return reduction_aggr_cuda.reduction_aggr_forward_partitioned(
        edge_ptr,
        edge_idx,
        X,
        light,
        heavy,
        max_degree=131070,
        warps_per_block=warps_per_block,
        edges_per_block_heavy_nodes=edges_per_block_heavy_nodes,
        use_2d_kernel=use_2d_kernel,
        features_per_block=features_per_block,
        tiles_y=tiles_y,
        reduce=reduce,
        pipeline_stages=pipeline_stages,
    )
