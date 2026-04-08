"""Public API: autotunable kernel functions.

Each function takes an :class:`AdjacencyForwardBackwardWithNodeBuckets` graph
and node features, dispatches to fused CUDA kernels, and supports an optional
``autotune=True`` kwarg that runs a grid search over kernel/graph parameters
on first call, then caches the best configuration.
"""

from __future__ import annotations

import torch

from turbo_gnn._autotune import with_autotune
from turbo_gnn._functions import (
    ReductionAggrFunction,
    _CudaSpMMConvFn,
    _FusedGraphAttention,
    csr_SPMM_normalized,
    gatv2_function,
)
from turbo_gnn._kernels import (
    GATv2AggrKernel,
    GraphTransformerAggrKernel,
    ReductionAggrKernel,
)
from turbo_gnn.graph import AdjacencyForwardBackwardWithNodeBuckets


@with_autotune(ReductionAggrKernel, init_params=("reduce",))
def reduction_aggr(
    graph: AdjacencyForwardBackwardWithNodeBuckets,
    X: torch.Tensor,
    warps_per_block: int = 8,
    edges_per_block_heavy_nodes: int = 128,
    use_2d_kernel: bool = False,
    features_per_block: int = 32,
    tiles_y: int = 8,
    reduce: str = "min",
) -> torch.Tensor:
    """Element-wise min or max aggregation over incoming neighbors.

    For each destination node *v*, computes::

        out[v] = reduce_{u in N(v)} X[u]   (reduce = "min" or "max")

    Uses a partitioned kernel: "light" nodes (low degree) use an atomic-based
    kernel; "heavy" nodes (high degree) use a tiled reduction kernel for better
    load balance.

    Args:
        graph: CSR graph with forward adjacency and light/heavy node buckets.
        X: Node features, shape ``[N, F]``.
        warps_per_block: Warps per CUDA thread block (light-node kernel).
        edges_per_block_heavy_nodes: Edges processed per block (heavy-node kernel).
        use_2d_kernel: Use the 2-D tiled kernel variant for the heavy-node path.
        features_per_block: Feature-dimension tile size (2-D kernel only).
        tiles_y: Number of row tiles (2-D kernel only).
        reduce: ``"min"`` or ``"max"``.

    Returns:
        Aggregated features, shape ``[N, F]``. Nodes with no incoming edges
        receive zeros (infinities are clamped internally).
    """
    return ReductionAggrFunction.apply(
        graph.forward_indptr,
        graph.forward_indices,
        X,
        graph.light_nodes,
        graph.heavy_nodes,
        graph.max_degree,
        warps_per_block,
        edges_per_block_heavy_nodes,
        use_2d_kernel,
        features_per_block,
        tiles_y,
        reduce,
    )


@with_autotune(GATv2AggrKernel)
def gatv2_aggr(
    graph: AdjacencyForwardBackwardWithNodeBuckets,
    x: torch.Tensor,
    x_neighbors: torch.Tensor,
    attention_weights: torch.Tensor,
    negative_slope: float = 0.2,
    grad_A_reduce_row_chunk_size: int = 512,
) -> torch.Tensor:
    """GATv2 attention-weighted aggregation.

    Computes multi-head GATv2 attention over the graph::

        e_{uv,h} = attn_h^T * LeakyReLU(x[v, h, :] + x_neighbors[u, h, :])
        alpha_{uv} = softmax_u(e_{uv})        (over incoming neighbors of v)
        out[v] = sum_{u in N(v)} alpha_{uv} * x_neighbors[u]

    The forward pass fuses edge score computation, numerically stable softmax
    (via log-sum-exp), and weighted aggregation into a single kernel.

    Args:
        graph: CSR graph with forward + backward adjacency for fwd/bwd passes.
        x: Destination (left) node features after projection, shape ``[N, H, D]``.
        x_neighbors: Source (right) node features after projection, shape ``[N, H, D]``.
        attention_weights: Learnable attention vector per head, shape ``[H, D]``.
        negative_slope: LeakyReLU negative slope (typically 0.2).
        grad_A_reduce_row_chunk_size: Row chunk size for backward attention gradient
            reduction. Larger values use more shared memory but fewer kernel launches.

    Returns:
        Aggregated features, shape ``[N, H*D]`` (heads concatenated).
    """
    return gatv2_function.apply(
        graph.forward_indptr,
        graph.forward_indices,
        graph.backward_indptr,
        graph.backward_indices,
        x,
        x_neighbors,
        attention_weights,
        negative_slope,
        grad_A_reduce_row_chunk_size,
        graph.is_directed,
    )


@with_autotune(GraphTransformerAggrKernel)
def graph_transformer_aggr(
    graph: AdjacencyForwardBackwardWithNodeBuckets,
    x: torch.Tensor,
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    """Fused multi-head graph transformer attention.

    Computes sparse multi-head attention over the graph structure::

        score_{uv,h} = (Q[u, h, :] . K[v, h, :]) * scale
        alpha_{uv}   = softmax_u(score_{uv})       (over incoming neighbors of v)
        out[v, h, :] = sum_{u in N(v)} alpha_{uv,h} * V[u, h, :]

    The entire forward pass (dot-product scores, numerically stable softmax,
    weighted value aggregation) is fused into a single CSR-based CUDA kernel.

    Args:
        graph: CSR graph with forward + backward adjacency for fwd/bwd passes.
        x: Original node features (unused by the kernel but passed through the
            autotuning wrapper for shape inference), shape ``[N, F]``.
        Q: Query tensor, shape ``[N, H, D]`` where ``H * D = F``.
        K: Key tensor, shape ``[N, H, D]``.
        V: Value tensor, shape ``[N, H, D]``.
        scale: Scaling factor, typically ``1 / sqrt(D)``.

    Returns:
        Attended features, shape ``[N, H, D]``.
    """
    return _FusedGraphAttention.apply(
        graph.forward_indptr,
        graph.forward_indices,
        graph.backward_indptr,
        graph.backward_indices,
        Q,
        K,
        V,
        scale,
        graph.is_directed,
    )


def spmm_aggr(x, forward_indptr, forward_indices, norm_type, cu_sparse_algorithm_id, block_dim):
    """Normalized sparse matrix-vector multiply via cuSPARSE.

    Computes ``out = norm(A) @ x`` where ``A`` is the adjacency in CSR format
    and the normalization is selected by *norm_type*:

    - ``"none"``: ``A @ x``  (sum aggregation)
    - ``"right"``: ``D_in^{-1} A @ x``  (mean aggregation)
    - ``"left"``: ``A D_out^{-1} @ x``  (random-walk normalization)
    - ``"both"``: ``D_out^{-1/2} A D_in^{-1/2} @ x``  (symmetric / GCN normalization)

    Degree matrices and normalization weights are computed inside the CUDA kernel.
    Supports autograd (backward transposes A and re-applies cuSPARSE).

    Args:
        x: Node features, shape ``[N, F]``.
        forward_indptr: CSR row pointers, shape ``[N+1]``, int32.
        forward_indices: CSR column indices, shape ``[E]``, int32.
        norm_type: One of ``"none"``, ``"right"``, ``"left"``, ``"both"``.
        cu_sparse_algorithm_id: cuSPARSE algorithm selector (-1 = auto).
        block_dim: CUDA block dimension for the normalization pre-pass.

    Returns:
        Aggregated features, shape ``[N, F]``.
    """
    return _CudaSpMMConvFn.apply(x, forward_indptr, forward_indices, norm_type, cu_sparse_algorithm_id, block_dim)
