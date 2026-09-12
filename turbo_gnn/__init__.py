"""turbo_gnn -- High-performance CUDA kernels for GNN aggregation.

Provides fused, autotunable CUDA kernels for common GNN operations:

- **reduction_aggr**: Min/max neighbor aggregation with node bucketing.
- **gatv2_aggr**: GATv2 attention-weighted aggregation (LeakyReLU + edge softmax).
- **graph_transformer_aggr**: Fused multi-head graph attention (Q*K dot + edge softmax + V aggregation).
- **spmm_aggr**: cuSPARSE-based SpMM with GCN/mean/sum normalization.
- **gsddmm**: Per-edge binary ops (add/sub/mul/div/dot/copy) over node/edge
  feature rows, plus DGL-style prefilled aliases (``u_sub_v``, ``copy_u``, ...).
  Two CUDA kernels implement it; ``gsddmm`` times both once per graph and uses
  the faster one, or pins one with ``variant="node"`` / ``variant="edge"``.

All kernels operate on CSR graphs wrapped in
:class:`AdjacencyForwardBackwardWithNodeBuckets`, which stores forward and
backward adjacency plus light/heavy node partitions for load-balanced execution.

Quick start::

    import torch
    from turbo_gnn import reduction_aggr, AdjacencyForwardBackwardWithNodeBuckets

    edge_index = torch.tensor([[0,1,2],[1,2,0]], device="cuda")
    graph = AdjacencyForwardBackwardWithNodeBuckets.from_edge_list(
        edge_index, num_nodes=3, index_dtype=torch.int32,
    ).to("cuda")
    x = torch.randn(3, 64, device="cuda")
    out = reduction_aggr(graph, x, reduce="min")  # [3, 64]
"""

from turbo_gnn._autotune import AutotuneConfig, TunableKernel, TunableParam, with_autotune
from turbo_gnn._gsddmm import EdgeBlockParams, GsddmmLaunchPlan, GsddmmSpec, NodeBlockParams, TraversalOrder
from turbo_gnn._kernels import (
    GATv2AggrKernel,
    GraphTransformerAggrKernel,
    GSDDMMEdgeKernel,
    GSDDMMKernel,
    ReductionAggrKernel,
)
from turbo_gnn.graph import AdjacencyForwardBackwardWithNodeBuckets
from turbo_gnn.ops import (
    _GSDDMM_EDGE_PREFILLED_OPS,
    _GSDDMM_PREFILLED_OPS,
    csr_SPMM_normalized,
    gatv2_aggr,
    graph_transformer_aggr,
    gsddmm,
    gsddmm_edge,
    reduction_aggr,
    spmm_aggr,
)

# DGL-style prefilled gsddmm ops (u_add_v, v_dot_u, copy_u, ...), generated in
# ops.py. The ``*_edge`` variants are bound here too, but deliberately left out
# of __all__: they pin the edge-parallel kernel AND expose its traversal edge
# order, which only the benchmarks and the correctness tests want. One op per
# operation is the public surface; ``variant=`` picks the kernel.
globals().update(_GSDDMM_PREFILLED_OPS)
globals().update(_GSDDMM_EDGE_PREFILLED_OPS)

__all__ = [
    "AdjacencyForwardBackwardWithNodeBuckets",
    "TunableParam",
    "AutotuneConfig",
    "TunableKernel",
    "with_autotune",
    "ReductionAggrKernel",
    "GATv2AggrKernel",
    "GraphTransformerAggrKernel",
    "GSDDMMKernel",
    "GsddmmLaunchPlan",
    "GsddmmSpec",
    "NodeBlockParams",
    "EdgeBlockParams",
    "TraversalOrder",
    "reduction_aggr",
    "gatv2_aggr",
    "graph_transformer_aggr",
    "spmm_aggr",
    "csr_SPMM_normalized",
    "gsddmm",
    *_GSDDMM_PREFILLED_OPS,
]
