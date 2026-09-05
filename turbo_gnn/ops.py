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
    GSpMMFunction,
    ReductionAggrFunction,
    _CudaSpMMConvFn,
    _FusedGraphAttention,
    csr_SPMM_normalized,
    gatv2_function,
)
from turbo_gnn._kernels import (
    GATv2AggrKernel,
    GSpMMKernel,
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
    pipeline_stages: int = 0,
) -> torch.Tensor:
    """Element-wise min, max or sum aggregation over incoming neighbors.

    For each destination node *v*, computes::

        out[v] = reduce_{u in N(v)} X[u]   (reduce = "min", "max" or "sum")

    With ``reduce="sum"`` this is plain SpMM against the unweighted adjacency
    (DGL's ``copy_u_sum``); the summation is accumulated in fp32 even for
    fp16/bf16 inputs.

    Uses a partitioned kernel: "light" nodes (low degree) use an atomic-based
    kernel; "heavy" nodes (high degree) use a tiled reduction kernel for better
    load balance.

    Args:
        graph: CSR graph with forward adjacency and light/heavy node buckets.
            ``reduce="sum"`` additionally uses the backward (transposed)
            adjacency for its gradient.
        X: Node features, shape ``[N, F]``.
        warps_per_block: Warps per CUDA thread block (light-node kernel).
        edges_per_block_heavy_nodes: Edges processed per block (heavy-node kernel).
        use_2d_kernel: Use the 2-D tiled kernel variant for the heavy-node path.
            Ignored for ``reduce="sum"``, which has no packed-atomics
            alternative and always takes the 2-D path.
        features_per_block: Feature-dimension tile size (2-D kernel only).
        tiles_y: Number of row tiles (2-D kernel only).
        reduce: ``"min"``, ``"max"`` or ``"sum"``.
        pipeline_stages: Number of async-copy pipeline stages for the light-node
            and packed-atomics heavy-node kernels' per-thread neighbor scan. 0
            disables the pipeline. Ignored when ``use_2d_kernel=True``.

    Returns:
        Aggregated features, shape ``[N, F]``. Nodes with no incoming edges
        receive zeros -- for min/max the identity (+-inf) is clamped
        internally, for sum it is already the empty-sum value.
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
        pipeline_stages,
        graph.backward_indptr,
        graph.backward_indices,
        graph.backward_light_nodes,
        graph.backward_heavy_nodes,
        graph.backward_max_degree,
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
    pipeline_stages: int = 0,
    backward_pipeline_stages: int = 0,
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
        pipeline_stages: Number of async-copy pipeline stages for the forward kernel's
            r[j] prefetch. 0 disables the pipeline (plain warp-strided loop).
        backward_pipeline_stages: Number of async-copy pipeline stages for the backward
            kernels' neighbor-row prefetch (AL/R when directed, G/ALR when undirected).
            0 disables the pipeline.

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
        graph.forward_light_nodes,
        graph.forward_heavy_nodes,
        graph.backward_light_nodes,
        graph.backward_heavy_nodes,
        forward_light_warps,
        forward_heavy_warps,
        backward_light_warps,
        backward_heavy_warps,
        graph.is_directed,
        pipeline_stages,
        backward_pipeline_stages,
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
    pipeline_stages: int = 0,
    backward_pipeline_stages: int = 0,
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
        pipeline_stages: Number of async-copy pipeline stages for the forward kernel's
            Q[j]/V[j] prefetch. 0 disables the pipeline.
        backward_pipeline_stages: Number of async-copy pipeline stages for the backward
            kernels' neighbor-row prefetch. 0 disables the pipeline.

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
        graph.forward_light_nodes,
        graph.forward_heavy_nodes,
        graph.backward_light_nodes,
        graph.backward_heavy_nodes,
        forward_light_warps,
        forward_heavy_warps,
        backward_light_warps,
        backward_heavy_warps,
        graph.is_directed,
        pipeline_stages,
        backward_pipeline_stages,
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



_GSPMM_OPS = ("copy_u", "copy_e", "add", "sub", "mul", "div")
_GSPMM_REDUCERS = ("sum", "min", "max")


def _gspmm_apply(
    graph: AdjacencyForwardBackwardWithNodeBuckets,
    lhs: torch.Tensor | None,
    rhs: torch.Tensor | None,
    op: str,
    reduce: str,
    warps_per_block: int,
    features_per_block: int,
    tiles_y: int,
) -> torch.Tensor:
    """Unpack the graph and hand off to :class:`GSpMMFunction`."""
    edge_map = graph.backward_edge_map if (op in ("mul", "div") and reduce == "sum") else None

    return GSpMMFunction.apply(
        lhs,
        rhs,
        op,
        reduce,
        graph.forward_indptr,
        graph.forward_indices,
        graph.forward_light_nodes,
        graph.forward_heavy_nodes,
        graph.backward_indptr,
        graph.backward_indices,
        graph.backward_light_nodes,
        graph.backward_heavy_nodes,
        edge_map,
        warps_per_block,
        features_per_block,
        tiles_y,
    )


@with_autotune(GSpMMKernel, init_params=("op", "reduce"))
def gspmm(
    graph: AdjacencyForwardBackwardWithNodeBuckets,
    lhs: torch.Tensor | None,
    rhs: torch.Tensor | None = None,
    op: str = "copy_u",
    reduce: str = "sum",
    warps_per_block: int = 8,
    features_per_block: int = 32,
    tiles_y: int = 8,
) -> torch.Tensor:
    """Generalized SpMM -- one message operation composed with one reduction.

    For every destination node *v*::

        out[v] = reduce_{(u, e) in in_edges(v)}  op( lhs[u], rhs[e] )

    This is the ``dgl.ops.gspmm`` contract: ``op`` selects how a source node's
    features combine with the edge's own data, ``reduce`` how the resulting
    messages collapse into the destination.

    Args:
        graph: CSR graph with forward and backward adjacency plus light/heavy
            node buckets.
        lhs: Node data, shape ``[N, d]``. Pass ``None`` for ``op="copy_e"``.
        rhs: Edge data, shape ``[E, d]`` (element-wise) or ``[E]`` / ``[E, 1]``
            (broadcast over the feature dimension). Pass ``None`` for
            ``op="copy_u"``.
        op: One of ``"copy_u"``, ``"copy_e"``, ``"add"``, ``"sub"``, ``"mul"``,
            ``"div"``.
        reduce: One of ``"sum"``, ``"min"``, ``"max"``.
        warps_per_block: Block size (in warps) of the light-node kernel.
        features_per_block: Feature tile width of the heavy-node kernel.
        tiles_y: Edge tiles reduced in parallel by the heavy-node kernel; must
            be a power of two.

    Returns:
        Aggregated features, shape ``[N, d]``. Nodes with no incoming edges get
        zeros: for ``sum`` that is the empty sum, for ``min``/``max`` the
        identity (+-inf) is clamped, which differs from DGL -- it returns the
        raw infinity there.

    Note:
        **Edge data must be in CSR order.** ``rhs[i]`` is the data of the edge
        at ``graph.forward_indices[i]``, not of ``edge_index[:, i]``.
        :meth:`AdjacencyForwardBackwardWithNodeBuckets.from_edge_list` sorts
        edges while building the CSR, so data aligned to the original
        ``edge_index`` must be permuted first via ``graph.to_csr_edge_order``.
        Getting this wrong is silent -- the shapes still match.
    """
    if op not in _GSPMM_OPS:
        raise ValueError(f"Unknown gspmm op {op!r}, expected one of {_GSPMM_OPS}")
    if reduce not in _GSPMM_REDUCERS:
        raise ValueError(f"Unknown gspmm reduce {reduce!r}, expected one of {_GSPMM_REDUCERS}")
    if op == "copy_u" and rhs is not None:
        raise ValueError("gspmm(op='copy_u') ignores edge data; pass rhs=None")
    if op == "copy_e" and lhs is not None:
        raise ValueError("gspmm(op='copy_e') ignores node data; pass lhs=None")

    return _gspmm_apply(graph, lhs, rhs, op, reduce, warps_per_block, features_per_block, tiles_y)




def copy_u_sum(graph, x, **kwargs):
    """``out[v] = sum_{u in N(v)} x[u]`` -- plain unweighted SpMM."""
    return reduction_aggr(graph, x, reduce="sum", **kwargs)


def copy_u_min(graph, x, **kwargs):
    """``out[v] = min_{u in N(v)} x[u]``."""
    return reduction_aggr(graph, x, reduce="min", **kwargs)


def copy_u_max(graph, x, **kwargs):
    """``out[v] = max_{u in N(v)} x[u]``."""
    return reduction_aggr(graph, x, reduce="max", **kwargs)


def copy_e_sum(graph, e, **kwargs):
    """``out[v] = sum_{edges into v} e[edge]`` -- node data is not read."""
    return gspmm(graph, None, e, op="copy_e", reduce="sum", **kwargs)


def copy_e_min(graph, e, **kwargs):
    """``out[v] = min_{edges into v} e[edge]``."""
    return gspmm(graph, None, e, op="copy_e", reduce="min", **kwargs)


def copy_e_max(graph, e, **kwargs):
    """``out[v] = max_{edges into v} e[edge]``."""
    return gspmm(graph, None, e, op="copy_e", reduce="max", **kwargs)


def u_add_e_sum(graph, x, e, **kwargs):
    """``out[v] = sum (x[u] + e[edge])``."""
    return gspmm(graph, x, e, op="add", reduce="sum", **kwargs)


def u_add_e_min(graph, x, e, **kwargs):
    """``out[v] = min (x[u] + e[edge])``."""
    return gspmm(graph, x, e, op="add", reduce="min", **kwargs)


def u_add_e_max(graph, x, e, **kwargs):
    """``out[v] = max (x[u] + e[edge])``."""
    return gspmm(graph, x, e, op="add", reduce="max", **kwargs)


def u_sub_e_sum(graph, x, e, **kwargs):
    """``out[v] = sum (x[u] - e[edge])``."""
    return gspmm(graph, x, e, op="sub", reduce="sum", **kwargs)


def u_sub_e_min(graph, x, e, **kwargs):
    """``out[v] = min (x[u] - e[edge])``."""
    return gspmm(graph, x, e, op="sub", reduce="min", **kwargs)


def u_sub_e_max(graph, x, e, **kwargs):
    """``out[v] = max (x[u] - e[edge])``."""
    return gspmm(graph, x, e, op="sub", reduce="max", **kwargs)


def u_mul_e_sum(graph, x, e, **kwargs):
    """``out[v] = sum (x[u] * e[edge])`` -- weighted SpMM."""
    return gspmm(graph, x, e, op="mul", reduce="sum", **kwargs)


def u_mul_e_min(graph, x, e, **kwargs):
    """``out[v] = min (x[u] * e[edge])``."""
    return gspmm(graph, x, e, op="mul", reduce="min", **kwargs)


def u_mul_e_max(graph, x, e, **kwargs):
    """``out[v] = max (x[u] * e[edge])``."""
    return gspmm(graph, x, e, op="mul", reduce="max", **kwargs)


def u_div_e_sum(graph, x, e, **kwargs):
    """``out[v] = sum (x[u] / e[edge])``."""
    return gspmm(graph, x, e, op="div", reduce="sum", **kwargs)


def u_div_e_min(graph, x, e, **kwargs):
    """``out[v] = min (x[u] / e[edge])``."""
    return gspmm(graph, x, e, op="div", reduce="min", **kwargs)


def u_div_e_max(graph, x, e, **kwargs):
    """``out[v] = max (x[u] / e[edge])``."""
    return gspmm(graph, x, e, op="div", reduce="max", **kwargs)
