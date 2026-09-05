"""turbo_gnn -- High-performance CUDA kernels for GNN aggregation.

Provides fused, autotunable CUDA kernels for common GNN operations:

- **reduction_aggr**: Min/max neighbor aggregation with node bucketing.
- **gatv2_aggr**: GATv2 attention-weighted aggregation (LeakyReLU + edge softmax).
- **graph_transformer_aggr**: Fused multi-head graph attention (Q*K dot + edge softmax + V aggregation).
- **spmm_aggr**: cuSPARSE-based SpMM with GCN/mean/sum normalization.
- **gspmm**: generalized SpMM -- {copy_u, copy_e, add, sub, mul, div} x {sum, min, max},
  the ``dgl.ops.gspmm`` operator family, also exposed under those 18 names.

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
from turbo_gnn._kernels import GATv2AggrKernel, GraphTransformerAggrKernel, GSpMMKernel, ReductionAggrKernel
from turbo_gnn.graph import AdjacencyForwardBackwardWithNodeBuckets
from turbo_gnn.ops import (
    copy_e_max,
    copy_e_min,
    copy_e_sum,
    copy_u_max,
    copy_u_min,
    copy_u_sum,
    csr_SPMM_normalized,
    gatv2_aggr,
    graph_transformer_aggr,
    gspmm,
    reduction_aggr,
    spmm_aggr,
    u_add_e_max,
    u_add_e_min,
    u_add_e_sum,
    u_div_e_max,
    u_div_e_min,
    u_div_e_sum,
    u_mul_e_max,
    u_mul_e_min,
    u_mul_e_sum,
    u_sub_e_max,
    u_sub_e_min,
    u_sub_e_sum,
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
    "GSpMMKernel",
    "reduction_aggr",
    "gatv2_aggr",
    "graph_transformer_aggr",
    "spmm_aggr",
    "csr_SPMM_normalized",
    # generalized SpMM (dgl.ops.gspmm family)
    "gspmm",
    "copy_u_sum",
    "copy_u_min",
    "copy_u_max",
    "copy_e_sum",
    "copy_e_min",
    "copy_e_max",
    "u_add_e_sum",
    "u_add_e_min",
    "u_add_e_max",
    "u_sub_e_sum",
    "u_sub_e_min",
    "u_sub_e_max",
    "u_mul_e_sum",
    "u_mul_e_min",
    "u_mul_e_max",
    "u_div_e_sum",
    "u_div_e_min",
    "u_div_e_max",
]
