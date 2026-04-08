"""turbo_gnn -- High-performance CUDA kernels for GNN aggregation.

Provides fused, autotunable CUDA kernels for common GNN operations:

- **reduction_aggr**: Min/max neighbor aggregation with node bucketing.
- **gatv2_aggr**: GATv2 attention-weighted aggregation (LeakyReLU + edge softmax).
- **graph_transformer_aggr**: Fused multi-head graph attention (Q*K dot + edge softmax + V aggregation).
- **spmm_aggr**: cuSPARSE-based SpMM with GCN/mean/sum normalization.

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
from turbo_gnn._kernels import GATv2AggrKernel, GraphTransformerAggrKernel, ReductionAggrKernel
from turbo_gnn.graph import AdjacencyForwardBackwardWithNodeBuckets
from turbo_gnn.ops import (
    csr_SPMM_normalized,
    gatv2_aggr,
    graph_transformer_aggr,
    reduction_aggr,
    spmm_aggr,
)

__all__ = [
    "AdjacencyForwardBackwardWithNodeBuckets",
    "TunableParam",
    "AutotuneConfig",
    "TunableKernel",
    "with_autotune",
    "ReductionAggrKernel",
    "GATv2AggrKernel",
    "GraphTransformerAggrKernel",
    "reduction_aggr",
    "gatv2_aggr",
    "graph_transformer_aggr",
    "spmm_aggr",
    "csr_SPMM_normalized",
]
