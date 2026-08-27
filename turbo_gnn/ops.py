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
    DEFAULT_BLOCKS_PER_SM,
    DEFAULT_BUCKET_LAUNCH,
    DEFAULT_SCHED_CHUNK,
    DEFAULT_SCHEDULE,
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
    schedule: str = DEFAULT_SCHEDULE,
    blocks_per_sm: int = DEFAULT_BLOCKS_PER_SM,
    sched_chunk: int = DEFAULT_SCHED_CHUNK,
    forward_bucket_launch: str = DEFAULT_BUCKET_LAUNCH,
    backward_bucket_launch: str = DEFAULT_BUCKET_LAUNCH,
    forward_heavy_edge_slice: int = 0,
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
        schedule: Node-to-block scheduling policy. ``"one_per_block"`` reproduces the
            historical one-block-per-node launch; ``"grid_stride"``, ``"precomputed"`` and
            ``"dynamic"`` launch persistently with ``blocks_per_sm * SM_count`` blocks and
            loop. ``"dynamic"`` (the default) claims work from an atomic queue, which is
            what balances heavy-tailed degree distributions.
        blocks_per_sm: Target resident blocks per SM for the persistent policies. Ignored
            by ``"one_per_block"``.

    Returns:
        Aggregated features, shape ``[N, F]``. Nodes with no incoming edges
        receive zeros (infinities are clamped internally).
    """
    table = graph.heavy_edge_slices("forward", forward_heavy_edge_slice) if forward_heavy_edge_slice > 0 else None

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
        schedule,
        blocks_per_sm,
        sched_chunk,
        forward_bucket_launch,
        backward_bucket_launch,
        forward_heavy_edge_slice,
        table.chunk_node if table is not None else None,
        table.chunk_start if table is not None else None,
    )


@with_autotune(GATv2AggrKernel)
def gatv2_aggr(
    graph: AdjacencyForwardBackwardWithNodeBuckets,
    x: torch.Tensor,
    x_neighbors: torch.Tensor,
    attention_weights: torch.Tensor,
    negative_slope: float = 0.2,
    grad_A_reduce_row_chunk_size: int = 512,
    forward_light_warps: int = 1,
    forward_heavy_warps: int = 8,
    backward_light_warps: int = 1,
    backward_heavy_warps: int = 8,
    schedule: str = DEFAULT_SCHEDULE,
    blocks_per_sm: int = DEFAULT_BLOCKS_PER_SM,
    sched_chunk: int = DEFAULT_SCHED_CHUNK,
    forward_bucket_launch: str = DEFAULT_BUCKET_LAUNCH,
    backward_bucket_launch: str = DEFAULT_BUCKET_LAUNCH,
    forward_heavy_edge_slice: int = 0,
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
        schedule: Node-to-block scheduling policy. ``"one_per_block"`` reproduces the
            historical one-block-per-node launch; ``"grid_stride"``, ``"precomputed"`` and
            ``"dynamic"`` launch persistently with ``blocks_per_sm * SM_count`` blocks and
            loop. ``"dynamic"`` (the default) claims work from an atomic queue, which is
            what balances heavy-tailed degree distributions.
        blocks_per_sm: Target resident blocks per SM for the persistent policies. Ignored
            by ``"one_per_block"``.

    Returns:
        Aggregated features, shape ``[N, H*D]`` (heads concatenated).
    """
    table = graph.heavy_edge_slices("forward", forward_heavy_edge_slice) if forward_heavy_edge_slice > 0 else None

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
        graph.forward_light_nodes,
        graph.forward_heavy_nodes,
        graph.backward_light_nodes,
        graph.backward_heavy_nodes,
        forward_light_warps,
        forward_heavy_warps,
        backward_light_warps,
        backward_heavy_warps,
        graph.is_directed,
        schedule,
        blocks_per_sm,
        sched_chunk,
        forward_bucket_launch,
        backward_bucket_launch,
        forward_heavy_edge_slice,
        table.chunk_node if table is not None else None,
        table.chunk_start if table is not None else None,
        table.node_chunk_offset if table is not None else None,
    )


@with_autotune(GraphTransformerAggrKernel)
def graph_transformer_aggr(
    graph: AdjacencyForwardBackwardWithNodeBuckets,
    x: torch.Tensor,
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    scale: float | None = None,
    forward_light_warps: int = 4,
    forward_heavy_warps: int = 8,
    backward_light_warps: int = 1,
    backward_heavy_warps: int = 8,
    schedule: str = DEFAULT_SCHEDULE,
    blocks_per_sm: int = DEFAULT_BLOCKS_PER_SM,
    sched_chunk: int = DEFAULT_SCHED_CHUNK,
    forward_bucket_launch: str = DEFAULT_BUCKET_LAUNCH,
    backward_bucket_launch: str = DEFAULT_BUCKET_LAUNCH,
    forward_heavy_edge_slice: int = 0,
    backward_heavy_edge_slice: int = 0,
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
        schedule: Node-to-block scheduling policy. ``"one_per_block"`` reproduces the
            historical one-block-per-node launch; ``"grid_stride"``, ``"precomputed"`` and
            ``"dynamic"`` launch persistently with ``blocks_per_sm * SM_count`` blocks and
            loop. ``"dynamic"`` (the default) claims work from an atomic queue, which is
            what balances heavy-tailed degree distributions.
        blocks_per_sm: Target resident blocks per SM for the persistent policies. Ignored
            by ``"one_per_block"``.
        forward_heavy_edge_slice: Edges per block in the forward heavy bucket. ``0`` keeps
            one block per heavy node; a positive value splits each heavy node's edge list
            into slices of that size, one block each, merged by a second kernel. Balances
            the heavy bucket and sizes its grid by edge count rather than node count.

    Returns:
        Attended features, shape ``[N, H, D]``.
    """
    table = graph.heavy_edge_slices("forward", forward_heavy_edge_slice) if forward_heavy_edge_slice > 0 else None
    bwd_table = (
        graph.heavy_edge_slices("backward", backward_heavy_edge_slice) if backward_heavy_edge_slice > 0 else None
    )

    return _FusedGraphAttention.apply(
        graph.forward_indptr,
        graph.forward_indices,
        graph.backward_indptr,
        graph.backward_indices,
        Q,
        K,
        V,
        scale,
        graph.forward_light_nodes,
        graph.forward_heavy_nodes,
        graph.backward_light_nodes,
        graph.backward_heavy_nodes,
        forward_light_warps,
        forward_heavy_warps,
        backward_light_warps,
        backward_heavy_warps,
        graph.is_directed,
        schedule,
        blocks_per_sm,
        sched_chunk,
        forward_bucket_launch,
        backward_bucket_launch,
        forward_heavy_edge_slice,
        table.chunk_node if table is not None else None,
        table.chunk_start if table is not None else None,
        table.node_chunk_offset if table is not None else None,
        backward_heavy_edge_slice,
        bwd_table.chunk_node if bwd_table is not None else None,
        bwd_table.chunk_start if bwd_table is not None else None,
        bwd_table.node_chunk_offset if bwd_table is not None else None,
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
