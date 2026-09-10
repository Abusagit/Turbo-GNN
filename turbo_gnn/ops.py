"""Public API: autotunable kernel functions.

Each function takes an :class:`AdjacencyForwardBackwardWithNodeBuckets` graph
and node features, dispatches to fused CUDA kernels, and supports an optional
``autotune=True`` kwarg that runs a grid search over kernel/graph parameters
on first call, then caches the best configuration.
"""

from __future__ import annotations

import torch

import turbo_gnn._C as _C
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
    GSDDMMEdgeKernel,
    GSDDMMKernel,
    ReductionAggrKernel,
    _graph_edge_list,
    _graph_heavy_blocks,
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
        pipeline_stages: Number of async-copy pipeline stages for the light-node
            and packed-atomics heavy-node kernels' per-thread neighbor scan. 0
            disables the pipeline. Ignored when ``use_2d_kernel=True``.

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
        pipeline_stages,
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


# =============================================================================
# GSDDMM: per-edge binary ops over node/edge features
# =============================================================================

# Member name mapping for DGL-style op names: u = source, v = destination, e = edge.
_GSDDMM_MEMBER_TO_NAME = {"src": "u", "dst": "v", "edge": "e"}
_GSDDMM_NAME_TO_MEMBER = {v: k for k, v in _GSDDMM_MEMBER_TO_NAME.items()}
_GSDDMM_OPS = ("add", "sub", "mul", "div", "dot")
# Ordered member pairs with lhs != rhs (same-member ops are dense-data ops and
# are rejected by the kernel's static_assert).
_GSDDMM_MEMBER_PAIRS = (
    ("src", "dst"),
    ("dst", "src"),
    ("src", "edge"),
    ("edge", "src"),
    ("dst", "edge"),
    ("edge", "dst"),
)


@with_autotune(GSDDMMKernel, init_params=("op", "lhs_target", "rhs_target"))
def gsddmm(
    graph: AdjacencyForwardBackwardWithNodeBuckets,
    lhs: torch.Tensor,
    rhs: torch.Tensor | None = None,
    op: str = "mul",
    lhs_target: str = "src",
    rhs_target: str = "dst",
    light_warps_per_block: int = 4,
    heavy_warps_per_block: int = 32,
    pipeline_stages: int = 0,
    heavy_edges_per_block: int = 0,
    overlap_buckets: bool = False,
) -> torch.Tensor:
    """Generalized SDDMM: per-edge binary op over node/edge feature rows.

    For every edge ``e = (u -> v)`` of the graph (CSR order), computes::

        out[e] = op(lhs[sel_l], rhs[sel_r])

    where ``sel_*`` picks a row by source node ``u`` (``"src"``), destination
    node ``v`` (``"dst"``), or edge position ``e`` (``"edge"``). ``"dot"``
    reduces the feature axis (output ``[E]``); all other ops are elementwise
    (output ``[E, D]``). ``"copy"`` propagates ``lhs`` to the edges and never
    reads ``rhs``.

    Forward-only (no autograd): the CUDA kernel has no backward pass yet, so
    the output is detached from the autograd graph.

    Args:
        graph: CSR graph with forward adjacency and light/heavy node buckets.
        lhs: Left operand, ``[N, D]`` for ``"src"``/``"dst"`` targets or
            ``[E, D]`` for ``"edge"``. D must be in {32, 64, 128, 256}.
        rhs: Right operand, same layout rules as ``lhs``. May be omitted for
            ``op="copy"`` (it is never read by the kernel).
        op: ``"add"``, ``"sub"``, ``"mul"``, ``"div"``, ``"dot"``, or ``"copy"``.
        lhs_target: ``"src"``, ``"dst"``, or ``"edge"``.
        rhs_target: ``"src"``, ``"dst"``, or ``"edge"``. Forced to ``"edge"``
            for ``op="copy"`` (the value is irrelevant since rhs is unread).
        light_warps_per_block: Warps per block for the light-node bucket. Only
            the counts instantiated by the binding are accepted (currently 4).
        heavy_warps_per_block: Warps per block for the heavy-node bucket. Only
            the counts instantiated by the binding are accepted (currently 32).
        pipeline_stages: Async-copy pipeline stages for the per-edge Src/Edge
            row prefetch, one of 0, 1, 2, 3. 0 disables the pipeline; stage
            ``i + stages`` is prefetched while stage ``i`` is consumed, which
            costs ``stages + 1`` shared-memory row slots per warp. A deep
            pipeline on a wide D with two per-edge operands can exceed the
            GPU's opt-in shared memory per block (e.g. D=256, 32 heavy warps
            and both operands gathered per edge needs 192 KiB at stages=2);
            the kernel raises with the exact figures when it does.
        heavy_edges_per_block: Split every heavy node into chunks of this many
            edges and run one 32-warp block per chunk instead of one per node,
            so the heavy launch's tail is bounded by the chunk rather than the
            largest degree. 0 keeps one block per node. The per-block (node,
            chunk) descriptors are built on first use and cached on the graph.
        overlap_buckets: Run the light-node bucket on a side CUDA stream
            concurrently with the heavy bucket on the current stream. The side
            stream is forked from and joined back to the current stream with
            events, so ordering for the caller is unchanged.

    Returns:
        ``[E, D]`` for elementwise ops, ``[E]`` for ``"dot"``, in CSR edge order.
    """
    if op == "copy":
        rhs_target = "edge"
    if rhs is None:
        if op != "copy":
            raise ValueError(f"gsddmm: rhs is required for op={op!r}")
        # Copy never reads R, but the binding validates its shape: [E, D].
        rhs = lhs.new_empty((graph.forward_indices.numel(), lhs.shape[-1]))
    heavy_nodes, heavy_parts = graph.heavy_nodes, None
    if heavy_edges_per_block > 0 and heavy_nodes.numel() > 0:
        heavy_nodes, heavy_parts = _graph_heavy_blocks(graph, heavy_edges_per_block)
    return _C.gsddmm_forward(
        lhs,
        rhs,
        graph.forward_indptr,
        graph.forward_indices,
        op,
        lhs_target,
        rhs_target,
        graph.light_nodes,
        heavy_nodes,
        light_warps_per_block,
        heavy_warps_per_block,
        pipeline_stages,
        heavy_parts,
        heavy_edges_per_block,
        overlap_buckets,
    )


@with_autotune(GSDDMMEdgeKernel, init_params=("op", "lhs_target", "rhs_target"))
def gsddmm_edge(
    graph: AdjacencyForwardBackwardWithNodeBuckets,
    lhs: torch.Tensor,
    rhs: torch.Tensor | None = None,
    op: str = "mul",
    lhs_target: str = "src",
    rhs_target: str = "dst",
    pipeline_stages: int = 0,
    edges_per_warp: int = 4,
    warps_per_block: int = 4,
) -> torch.Tensor:
    """Generalized SDDMM, edge-parallel variant: per-edge binary op.

    Same semantics as :func:`gsddmm` — for every edge ``e = (u -> v)``::

        out[e] = op(lhs[sel_l], rhs[sel_r])

    — but parallelized one warp per edge over an explicit ``[E, 2]`` edge list
    of ``(src, dst)`` pairs instead of walking the CSR with node-bucketed
    blocks. The edge list is derived from the graph's CSR on first use and
    cached per graph (shared with the autotuning kernel instances), so
    repeated calls on the same graph never rebuild it.

    Edge grouping: when an operand reads the destination vertex, edges are
    grouped by destination (built from the forward CSR, CSR edge order);
    otherwise they are grouped by source (built from the backward CSR, CSC
    edge order) so consecutive warps share the Src_V operand row. The output
    rows follow the chosen grouping.

    Forward-only (no autograd): the CUDA kernel has no backward pass yet, so
    the output is detached from the autograd graph.

    Args:
        graph: CSR graph; its forward (dst-grouped) or backward (src-grouped)
            adjacency is read to build the cached edge list.
        lhs: Left operand, ``[N, D]`` for ``"src"``/``"dst"`` targets or
            ``[E, D]`` for ``"edge"``. D must be in {32, 64, 128, 256}.
        rhs: Right operand, same layout rules as ``lhs``. May be omitted for
            ``op="copy"`` (it is never read by the kernel).
        op: ``"add"``, ``"sub"``, ``"mul"``, ``"div"``, ``"dot"``, or ``"copy"``.
        lhs_target: ``"src"``, ``"dst"``, or ``"edge"``.
        rhs_target: ``"src"``, ``"dst"``, or ``"edge"``. Forced to ``"edge"``
            for ``op="copy"`` (the value is irrelevant since rhs is unread).
        pipeline_stages: cp.async prefetch depth of the operand rows, in
            {0, 2, 3}. 0 gathers rows with direct loads; ``s >= 1`` copies
            the rows of the next ``s`` edges of the warp's chunk into shared
            memory while the current edge is computed. Only useful with
            ``edges_per_warp > 1``.
        edges_per_warp: contiguous edges each warp processes, in [1, 32].
            The default 1 is the original one-edge-per-warp layout.
        warps_per_block: independent warps per thread block, in [1, 8].
            The default 1 is the original 32-thread block.

    Returns:
        ``[E, D]`` for elementwise ops, ``[E]`` for ``"dot"``. Edge order: CSR
        (grouped by destination) when a ``"dst"`` operand is involved, CSC
        (grouped by source) otherwise.
    """
    if op == "copy":
        rhs_target = "edge"
    if rhs is None:
        if op != "copy":
            raise ValueError(f"gsddmm_edge: rhs is required for op={op!r}")
        # Copy never reads R, but the binding validates its shape: [E, D].
        rhs = lhs.new_empty((graph.forward_indices.numel(), lhs.shape[-1]))
    # Group edges by source when no operand reads the destination vertex.
    by_src = "dst" not in (lhs_target, rhs_target)
    edge_list = _graph_edge_list(graph, by_src=by_src)
    return _C.gsddmm_forward_edge(
        lhs,
        rhs,
        edge_list,
        op,
        lhs_target,
        rhs_target,
        graph.forward_indptr.numel() - 1,
        pipeline_stages,
        edges_per_warp,
        warps_per_block,
    )


def _make_gsddmm_op(op: str, lhs_target: str, rhs_target: str, edge_variant: bool = False):
    """Build a DGL-style gsddmm op with pre-filled op/targets (e.g. u_sub_v)."""
    base_fn = gsddmm_edge if edge_variant else gsddmm
    name = f"{_GSDDMM_MEMBER_TO_NAME[lhs_target]}_{op}_{_GSDDMM_MEMBER_TO_NAME[rhs_target]}"
    if edge_variant:
        name = f"{name}_edge"

    def _gsddmm_op(
        graph: AdjacencyForwardBackwardWithNodeBuckets,
        lhs: torch.Tensor,
        rhs: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        return base_fn(graph, lhs, rhs, op=op, lhs_target=lhs_target, rhs_target=rhs_target, **kwargs)

    _gsddmm_op.__name__ = name
    _gsddmm_op.__qualname__ = name
    order_note = (
        "CSR edge order (grouped by destination)"
        if not edge_variant or "dst" in (lhs_target, rhs_target)
        else "CSC edge order (grouped by source)"
    )
    _gsddmm_op.__doc__ = f"""{name}(graph, lhs, rhs): per-edge ``{op}`` of {lhs_target!r} and {rhs_target!r} rows.

    Alias for :func:`{base_fn.__name__}` with ``op={op!r}, lhs_target={lhs_target!r}, rhs_target={rhs_target!r}``.
    Returns ``[E]`` if op is "dot" else ``[E, D]``, in {order_note}. Forward-only (no autograd).
    """
    return _gsddmm_op


def _make_gsddmm_copy(lhs_target: str, edge_variant: bool = False):
    """Build a DGL-style copy op (copy_u / copy_v): broadcast node rows to edges."""
    base_fn = gsddmm_edge if edge_variant else gsddmm
    name = f"copy_{_GSDDMM_MEMBER_TO_NAME[lhs_target]}"
    if edge_variant:
        name = f"{name}_edge"

    def _gsddmm_copy(
        graph: AdjacencyForwardBackwardWithNodeBuckets,
        x: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return base_fn(graph, x, None, op="copy", lhs_target=lhs_target, rhs_target="edge", **kwargs)

    _gsddmm_copy.__name__ = name
    _gsddmm_copy.__qualname__ = name
    order_note = (
        "CSR edge order (grouped by destination)"
        if not edge_variant or lhs_target == "dst"
        else "CSC edge order (grouped by source)"
    )
    _gsddmm_copy.__doc__ = f"""{name}(graph, x): copy {lhs_target!r}-node feature rows onto incident edges.

    Alias for :func:`{base_fn.__name__}` with ``op="copy", lhs_target={lhs_target!r}``. ``x`` is ``[N, D]``;
    returns ``[E, D]`` in {order_note}. Forward-only (no autograd).
    """
    return _gsddmm_copy


# DGL-style prefilled ops: u_sub_v(graph, x, y) == gsddmm(..., op="sub", lhs="src", rhs="dst").
# The ``*_edge`` variants call gsddmm_edge (the edge-parallel kernel) instead.
_GSDDMM_PREFILLED_OPS: dict[str, callable] = {}
_GSDDMM_EDGE_PREFILLED_OPS: dict[str, callable] = {}
for _ll, _rr in _GSDDMM_MEMBER_PAIRS:
    for _op in _GSDDMM_OPS:
        _fn = _make_gsddmm_op(_op, _ll, _rr)
        _GSDDMM_PREFILLED_OPS[_fn.__name__] = _fn
        _fn_edge = _make_gsddmm_op(_op, _ll, _rr, edge_variant=True)
        _GSDDMM_EDGE_PREFILLED_OPS[_fn_edge.__name__] = _fn_edge
for _target in ("src", "dst"):
    _fn = _make_gsddmm_copy(_target)
    _GSDDMM_PREFILLED_OPS[_fn.__name__] = _fn
    _fn_edge = _make_gsddmm_copy(_target, edge_variant=True)
    _GSDDMM_EDGE_PREFILLED_OPS[_fn_edge.__name__] = _fn_edge

globals().update(_GSDDMM_PREFILLED_OPS)
globals().update(_GSDDMM_EDGE_PREFILLED_OPS)
