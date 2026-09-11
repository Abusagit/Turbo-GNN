"""GSDDMM launch planning: operand spec, per-variant parameters, kernel selection.

``csrc/gsddmm/`` implements the same per-edge math twice, with different work
decompositions:

- ``GSDDMM_forward_normal`` -- one thread block per (bucketed) CSR row, warps of
  the block striding over the row's edges. The ``Dst_V`` operand row is staged
  once per block in shared memory, so it shines when rows are long; its runtime
  is set by the largest degree, so it starves when they are short.
- ``GSDDMM_forward_edge_block`` -- one warp per contiguous chunk of an explicit
  edge list. Perfectly load balanced regardless of the degree distribution, at
  the cost of re-gathering operand rows per edge.

Which one wins is a property of the workload (graph geometry, feature width,
dtype, which members the operands read), not of the caller's intent, so it is
decided here -- measured once per graph and memoized on the graph object.

Two invariants hold this module together:

1. **One output numbering.** Every plan returns rows numbered by *forward-CSR*
   edge position, whichever kernel runs. The edge kernel may still *traverse* a
   source-grouped list for locality; ``canonical_edge_idx`` (see
   :func:`_graph_canonical_edge_idx`) maps its traversal slots back to canonical
   ids, for the ``Edge`` operand reads as well as the output store.
2. **Only the selected kernel's parameters travel.** A resolved
   :class:`GsddmmPlan` carries exactly one variant's parameter dataclass, so no
   dead argument reaches ``_C``, and :meth:`GsddmmPlan.backward_context` names
   exactly the tensors that variant's backward pass will need.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Literal, Union

import torch

import turbo_gnn._C as _C
from turbo_gnn._timer import time_callable

logger = logging.getLogger(__name__)

#: Resolved kernel choice. ``"node"`` = ``GSDDMM_forward_normal``,
#: ``"edge"`` = ``GSDDMM_forward_edge_block``.
GsddmmVariant = Literal["node", "edge"]
#: What a caller may ask for: a pinned kernel, or ``"auto"`` to have one measured.
GsddmmVariantChoice = Literal["auto", "node", "edge"]

GSDDMM_OPS = ("add", "sub", "mul", "div", "dot", "copy")
GSDDMM_MEMBERS = ("src", "dst", "edge")

# Feature widths the kernels are instantiated for (see gsddmm_dispatch.cuh).
GSDDMM_FEATURE_DIMS = (32, 64, 128, 256)

# Timing budget for the variant probe when the caller supplied no AutotuneConfig.
_PROBE_WARMUP = 3
_PROBE_ITERS = 10
# Repeats per candidate, interleaved. This box's timings drift by 2-3x between
# runs (8 GPUs shared with other jobs), and a single measurement per candidate
# was observed to mis-rank two candidates that differed by 46 %. Interleaving
# repeats and keeping each candidate's MINIMUM cancels drift that a sequential
# single shot cannot -- noise only ever adds time. The whole probe still costs
# well under 100 ms, once per (op, graph, D, dtype).
_PROBE_REPEATS = 3

# Keys of the per-graph caches this module owns (all live in ``graph.__dict__``).
_EDGE_LIST_DST_KEY = "_gsddmm_edge_list_dst"
_EDGE_LIST_SRC_KEY = "_gsddmm_edge_list_src"
_CANONICAL_IDX_KEY = "_gsddmm_canonical_edge_idx"
_HEAVY_BLOCKS_KEY = "_gsddmm_heavy_blocks"
_VARIANT_MEMO_KEY = "_gsddmm_variant"


class TraversalOrder(Enum):
    """Which cached edge list the edge-parallel kernel walks.

    Internal, and *not* an output contract: the output is always numbered by
    forward-CSR edge position. This only decides which operand row consecutive
    warps share, i.e. what stays in L2.

    - ``CSR``: forward CSR, edges grouped by destination.
    - ``CSC``: backward CSR, edges grouped by source.
    """

    CSR = "csr"
    CSC = "csc"


@dataclass(frozen=True)
class GsddmmSpec:
    """What to compute: the op and the member each operand reads.

    ``op="copy"`` never reads the right operand, so ``rhs_target`` is normalized
    to ``"edge"`` (the only pairing the binding instantiates for it).
    """

    op: str
    lhs_target: str
    rhs_target: str

    def __post_init__(self) -> None:
        if self.op not in GSDDMM_OPS:
            raise ValueError(f"gsddmm: unsupported op {self.op!r}; supported: {', '.join(GSDDMM_OPS)}")
        for name, target in (("lhs_target", self.lhs_target), ("rhs_target", self.rhs_target)):
            if target not in GSDDMM_MEMBERS:
                raise ValueError(f"gsddmm: unsupported {name} {target!r}; supported: {', '.join(GSDDMM_MEMBERS)}")
        if self.op == "copy":
            # Frozen dataclass: normalize through object.__setattr__.
            object.__setattr__(self, "rhs_target", "edge")
            if self.lhs_target == "edge":
                raise ValueError("gsddmm: op='copy' cannot copy edge features onto the edges")
        elif self.lhs_target == self.rhs_target:
            raise ValueError(
                f"gsddmm: lhs_target and rhs_target are both {self.lhs_target!r}; an op between two dense tensors "
                "indexed the same way is a plain elementwise op, not a GSDDMM"
            )

    @property
    def uses_rhs(self) -> bool:
        """False for ``copy``, whose kernel never dereferences the right operand."""
        return self.op != "copy"

    @property
    def is_dot(self) -> bool:
        """True when the feature axis is reduced, so the output is ``[E]`` not ``[E, D]``."""
        return self.op == "dot"

    @property
    def targets(self) -> tuple[str, ...]:
        """The members actually read, in operand order."""
        return (self.lhs_target, self.rhs_target) if self.uses_rhs else (self.lhs_target,)

    @property
    def reads_dst(self) -> bool:
        return "dst" in self.targets

    @property
    def reads_src(self) -> bool:
        return "src" in self.targets

    @property
    def reads_edge(self) -> bool:
        return "edge" in self.targets

    @property
    def preferred_traversal(self) -> TraversalOrder:
        """Edge grouping that maximizes operand-row reuse across a warp's chunk.

        Grouping by destination only helps when an operand reads the destination
        vertex; otherwise grouping by source is what lets consecutive edges share
        the ``Src_V`` row.
        """
        return TraversalOrder.CSR if self.reads_dst else TraversalOrder.CSC

    @property
    def shape_class(self) -> tuple[int, int, bool]:
        """Coarse operand shape: ``(#dst operands, #edge operands, is_dot)``.

        Two specs in the same class move the same bytes in the same pattern, so
        they agree on the variant almost always (measured: the winner is ~97 %
        consistent across ops within a class at a fixed graph/D/dtype). Used only
        when ``AutotuneConfig.share_variant_probe`` is on.
        """
        return (
            sum(t == "dst" for t in self.targets),
            sum(t == "edge" for t in self.targets),
            self.is_dot,
        )

    @property
    def backward_needs_operands(self) -> tuple[str, ...]:
        """Which forward operands the (future) backward pass has to keep.

        ``add``/``sub``/``copy`` have constant partials, so nothing is saved;
        ``mul``/``div`` need the other operand; ``dot`` needs both.
        """
        if self.op in ("add", "sub", "copy"):
            return ()
        if self.op in ("mul", "div"):
            return ("rhs",)
        return ("lhs", "rhs")


@dataclass(frozen=True)
class NodeBlockParams:
    """Launch parameters of ``GSDDMM_forward_normal``.

    Args:
        light_warps: Warps per block for the light-node bucket.
        heavy_warps: Warps per block for the heavy-node bucket.
        pipeline_stages: cp.async prefetch depth for the per-edge rows, 0-3.
        heavy_edges_per_block: Split heavy nodes into chunks of this many edges,
            one block per chunk (0 = one block per node).
        overlap_buckets: Run the light bucket on a side stream concurrently with
            the heavy bucket.

    The light/heavy partition itself is not here: it lives on the graph, and the
    autotuner explores it by passing a repartitioned graph.
    """

    light_warps: int = 4
    heavy_warps: int = 32
    pipeline_stages: int = 0
    heavy_edges_per_block: int = 0
    overlap_buckets: bool = False


@dataclass(frozen=True)
class EdgeBlockParams:
    """Launch parameters of ``GSDDMM_forward_edge_block``.

    Args:
        pipeline_stages: cp.async prefetch depth of the operand rows, 0-3.
        edges_per_warp: Contiguous edges each warp walks, 1-32.
        warps_per_block: Independent warps packed per block, 1-8.
    """

    pipeline_stages: int = 0
    edges_per_warp: int = 4
    warps_per_block: int = 4


GsddmmParams = Union[NodeBlockParams, EdgeBlockParams]

_PARAMS_FOR_VARIANT: dict[str, type] = {"node": NodeBlockParams, "edge": EdgeBlockParams}


@dataclass(frozen=True)
class GsddmmPlan:
    """A resolved GSDDMM launch: one kernel, its own parameters, nothing else.

    Frozen and hashable, so the future ``torch.autograd.Function`` can stash it
    on ``ctx`` as-is and pair it with :meth:`backward_context`.

    Args:
        spec: What to compute.
        variant: Which kernel runs.
        params: That kernel's parameters -- ``NodeBlockParams`` for ``"node"``,
            ``EdgeBlockParams`` for ``"edge"``. A mismatch raises.
        traversal: Edge grouping for the edge variant (ignored by the node one).
        canonical_output: Number the output rows by forward-CSR edge position
            even when traversing a source-grouped list. Always True for the
            public API -- both variants must be interchangeable. The internal
            ``gsddmm_edge`` op sets it False to keep its legacy contract, where
            the output follows the traversal order.
    """

    spec: GsddmmSpec
    variant: GsddmmVariant
    params: GsddmmParams
    traversal: TraversalOrder = TraversalOrder.CSR
    canonical_output: bool = True

    def __post_init__(self) -> None:
        expected = _PARAMS_FOR_VARIANT.get(self.variant)
        if expected is None:
            raise ValueError(f"gsddmm: unknown variant {self.variant!r}; expected 'node' or 'edge'")
        if not isinstance(self.params, expected):
            raise TypeError(
                f"gsddmm: variant {self.variant!r} needs {expected.__name__}, got {type(self.params).__name__}"
            )

    def remaps_edge_ids(self, graph) -> bool:
        """True when this plan needs the traversal -> canonical id indirection."""
        return (
            self.variant == "edge"
            and self.canonical_output
            and not _edge_list_is_canonical(graph, by_src=self.traversal is TraversalOrder.CSC)
        )

    def launch(self, graph, lhs: torch.Tensor, rhs: torch.Tensor | None = None) -> torch.Tensor:
        """Run the planned kernel. The single point in the library that calls ``_C``."""
        rhs = _resolve_rhs(self.spec, graph, lhs, rhs)
        if self.variant == "node":
            return _launch_node(self.spec, graph, lhs, rhs, self.params)
        return _launch_edge(self.spec, graph, lhs, rhs, self.params, self.traversal, self.canonical_output)

    def backward_context(self, graph) -> dict[str, torch.Tensor]:
        """Index tensors this plan's backward pass will need, and nothing else.

        No backward kernel exists yet; this is the contract it will consume (and
        what a ``GsddmmFunction`` would hand to ``ctx.save_for_backward``), kept
        next to the forward so the two cannot drift apart. Pair it with
        :attr:`GsddmmSpec.backward_needs_operands` for the feature tensors --
        which for ``add``/``sub``/``copy`` is nothing at all.

        The gradient of a ``src``-targeted operand is a segment reduction over
        each node's *outgoing* edges, so it reads the backward (source-grouped)
        adjacency; a ``dst``-targeted operand reduces over the forward one.
        """
        ctx: dict[str, torch.Tensor] = {}
        if self.variant == "node":
            ctx["forward_indptr"] = graph.forward_indptr
            ctx["forward_indices"] = graph.forward_indices
            if self.spec.reads_dst:
                ctx["forward_light_nodes"] = graph.forward_light_nodes
                ctx["forward_heavy_nodes"] = graph.forward_heavy_nodes
            if self.spec.reads_src:
                ctx["backward_indptr"] = graph.backward_indptr
                ctx["backward_indices"] = graph.backward_indices
                ctx["backward_light_nodes"] = graph.backward_light_nodes
                ctx["backward_heavy_nodes"] = graph.backward_heavy_nodes
        else:
            ctx["edge_list"] = _graph_edge_list(graph, by_src=self.traversal is TraversalOrder.CSC)
            if self.remaps_edge_ids(graph):
                # Also the permutation that carries grad_out back from canonical
                # ids to this plan's traversal slots.
                ctx["canonical_edge_idx"] = _graph_canonical_edge_idx(graph)
        return ctx


# =============================================================================
# Per-graph derived data, cached on the graph object
# =============================================================================


def _graph_stamp(graph) -> tuple:
    """Identity of the graph's current buffers, cheap enough to check per call.

    The variant memo is validated against this on every ``variant="auto"`` call,
    so it is deliberately three pointer reads rather than a broad fingerprint:

    - the two CSR pointers catch a rebuilt CSR and a device move (``to()``
      reassigns every buffer);
    - the heavy-bucket pointer catches an in-place re-bucketing.

    Nothing else needs stamping, because :meth:`repartition` returns a *new*
    graph object, whose memo starts empty.
    """
    return (
        graph.forward_indptr.data_ptr(),
        graph.forward_indices.data_ptr(),
        graph.forward_heavy_nodes.data_ptr(),
    )


def _graph_heavy_blocks(graph, edges_per_block: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-block ``(node, chunk)`` descriptors that split every heavy node into
    ``edges_per_block``-wide edge chunks, cached on the graph object.

    The node-bucketed CSR kernel runs one thread block per heavy node, so its
    runtime is set by the single largest degree (SM active cycles max/avg of
    1.5-2.4x on power-law graphs). With chunking, block ``b`` handles edges
    ``[indptr[n] + part * K, +K)`` of node ``n = nodes[b]``, ``part = parts[b]``,
    and the kernel's tail is bounded by ``K`` edges instead of the max degree.

    Returns ``(nodes, parts)`` in the graph's index dtype. Cached per
    ``(indptr, heavy bucket, K)`` -- a repartition invalidates it.
    """
    heavy = graph.heavy_nodes
    stamp = (graph.forward_indptr.data_ptr(), heavy.data_ptr(), heavy.numel(), edges_per_block)
    cache = graph.__dict__.setdefault(_HEAVY_BLOCKS_KEY, {})
    if stamp in cache:
        return cache[stamp]

    indptr = graph._to_signed_view(graph.forward_indptr)
    h = graph._to_signed_view(heavy).long()
    deg = (indptr[h + 1] - indptr[h]).long()
    nblk = torch.clamp((deg + edges_per_block - 1) // edges_per_block, min=1)
    nodes = torch.repeat_interleave(h, nblk)
    first = torch.cumsum(nblk, 0) - nblk  # index of each node's first block
    parts = torch.arange(int(nblk.sum().item()), device=h.device) - torch.repeat_interleave(first, nblk)
    result = (nodes.to(heavy.dtype), parts.to(heavy.dtype))
    cache[stamp] = result
    return result


def _edge_list_is_canonical(graph, by_src: bool) -> bool:
    """True when the ``by_src`` edge list is in forward-CSR edge order.

    Trivially so for the destination-grouped list. Also so for the
    source-grouped one on an undirected graph, where the backward CSR is aliased
    to the forward CSR and :func:`_graph_edge_list` hands back the same tensor --
    no remapping needed, and no permutation worth materializing.
    """
    return (not by_src) or graph.backward_indptr.data_ptr() == graph.forward_indptr.data_ptr()


def _graph_edge_list(graph, by_src: bool = False) -> torch.Tensor:
    """Return the graph's [E, 2] (src, dst) edge list, cached per direction.

    The edge-parallel GSDDMM kernel consumes an explicit edge list instead of
    the CSR. ``by_src=False`` builds it from the forward CSR (edges grouped by
    destination, CSR order); ``by_src=True`` builds it from the backward CSR
    (edges grouped by source, CSC order) so consecutive warps share the
    ``Src_V`` operand row.

    Both directions are cached on the graph object, stamped with both CSRs'
    data pointers and sizes, so a repartitioned graph (same CSR, new buckets)
    still hits, while a mutated or re-created CSR rebuilds. Undirected graphs
    alias the backward CSR to the forward one, so both directions share a
    single list.
    """
    # Undirected graphs alias backward CSR to forward CSR: one list serves both.
    if by_src and _edge_list_is_canonical(graph, by_src=True):
        return _graph_edge_list(graph, by_src=False)

    stamp = (
        graph.forward_indptr.data_ptr(),
        graph.forward_indices.data_ptr(),
        graph.backward_indptr.data_ptr(),
        graph.backward_indices.data_ptr(),
        graph.forward_indptr.numel(),
        graph.forward_indices.numel(),
        graph.backward_indices.numel(),
    )
    cache_key = _EDGE_LIST_SRC_KEY if by_src else _EDGE_LIST_DST_KEY
    cached = graph.__dict__.get(cache_key)
    if cached is not None and cached[0] == stamp:
        return cached[1]

    indptr = graph.backward_indptr if by_src else graph.forward_indptr
    indices = graph.backward_indices if by_src else graph.forward_indices
    num_nodes = indptr.numel() - 1
    signed_indptr = graph._to_signed_view(indptr)
    degrees = signed_indptr[1:] - signed_indptr[:-1]
    # Rows of the chosen CSR are destinations (forward) or sources (backward).
    rows = torch.repeat_interleave(torch.arange(num_nodes, device=indptr.device, dtype=torch.int64), degrees)
    cols = indices.to(torch.int64)
    src, dst = (rows, cols) if by_src else (cols, rows)
    # The binding reinterprets the buffer as ulonglong2, so it must be a
    # uint64 tensor; the uint64 view keeps the same values bit-for-bit.
    edge_list = torch.stack([src, dst], dim=1).contiguous().view(torch.uint64)

    graph.__dict__[cache_key] = (stamp, edge_list)
    return edge_list


def _graph_canonical_edge_idx(graph) -> torch.Tensor:
    """uint64 ``[E]``: forward-CSR edge position of each source-grouped edge.

    This is the bridge that lets ``GSDDMM_forward_edge_block`` traverse the
    L2-friendly source-grouped list while numbering its ``Edge`` operand reads
    and its output rows the way ``GSDDMM_forward_normal`` does.

    Built by sorting both edge orders on the same ``(src, dst)`` key and pairing
    them position-wise. Both sorts are stable, so the j-th edge of a duplicated
    ``(src, dst)`` pair in one order maps to the j-th in the other: a well-defined
    bijection on multigraphs. Neither order is *assumed* sorted -- a graph built
    from :meth:`AdjacencyForwardBackwardWithNodeBuckets.from_csr` may carry
    unsorted column indices within a row.

    Cached on the graph like :func:`_graph_edge_list`; also the permutation the
    backward pass needs to move gradients between the two edge orders.
    """
    stamp = _graph_stamp(graph)
    cached = graph.__dict__.get(_CANONICAL_IDX_KEY)
    if cached is not None and cached[0] == stamp:
        return cached[1]

    num_nodes = graph.forward_indptr.numel() - 1
    csr_src, csr_dst = _edge_endpoints(graph, by_src=False)
    csc_src, csc_dst = _edge_endpoints(graph, by_src=True)
    # Sorting both orders by the same key aligns them edge for edge.
    order_csr = torch.argsort(csr_src * num_nodes + csr_dst, stable=True)
    order_csc = torch.argsort(csc_src * num_nodes + csc_dst, stable=True)

    canonical = torch.empty_like(order_csr)
    canonical[order_csc] = order_csr
    canonical = canonical.contiguous().view(torch.uint64)

    graph.__dict__[_CANONICAL_IDX_KEY] = (stamp, canonical)
    return canonical


def _edge_endpoints(graph, by_src: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """``(src, dst)`` int64 arrays in one CSR's edge order (rows expanded)."""
    indptr = graph.backward_indptr if by_src else graph.forward_indptr
    indices = graph.backward_indices if by_src else graph.forward_indices
    num_nodes = indptr.numel() - 1
    signed = graph._to_signed_view(indptr)
    degrees = signed[1:] - signed[:-1]
    rows = torch.repeat_interleave(torch.arange(num_nodes, device=indptr.device, dtype=torch.int64), degrees)
    cols = graph._to_signed_view(indices).to(torch.int64)
    return (rows, cols) if by_src else (cols, rows)


# =============================================================================
# Launchers -- each sees only its own kernel's parameters
# =============================================================================


def _resolve_rhs(spec: GsddmmSpec, graph, lhs: torch.Tensor, rhs: torch.Tensor | None) -> torch.Tensor:
    """The right operand to hand the binding, allocating a stand-in for ``copy``.

    ``copy`` never reads R, but the binding still validates its shape, so a
    single row expanded to ``[E, D]`` satisfies the check (``size(0) == E``,
    ``stride(1) == 1``) while allocating ``D`` elements instead of ``E * D`` --
    on a large graph the difference between a few hundred bytes and tens of
    gigabytes. Any ``rhs`` a caller passes for ``copy`` is therefore ignored
    rather than validated and forwarded.
    """
    if spec.uses_rhs:
        if rhs is None:
            raise ValueError(f"gsddmm: rhs is required for op={spec.op!r}")
        return rhs
    feat_dim = lhs.shape[-1]
    return lhs.new_empty((1, feat_dim)).expand(graph.forward_indices.numel(), feat_dim)


def _launch_node(
    spec: GsddmmSpec, graph, lhs: torch.Tensor, rhs: torch.Tensor, params: NodeBlockParams
) -> torch.Tensor:
    """Launch ``GSDDMM_forward_normal`` over the graph's light/heavy node buckets."""
    heavy_nodes, heavy_parts = graph.heavy_nodes, None
    if params.heavy_edges_per_block > 0 and heavy_nodes.numel() > 0:
        heavy_nodes, heavy_parts = _graph_heavy_blocks(graph, params.heavy_edges_per_block)
    return _C.gsddmm_forward(
        lhs,
        rhs,
        graph.forward_indptr,
        graph.forward_indices,
        spec.op,
        spec.lhs_target,
        spec.rhs_target,
        graph.light_nodes,
        heavy_nodes,
        params.light_warps,
        params.heavy_warps,
        params.pipeline_stages,
        heavy_parts,
        params.heavy_edges_per_block,
        params.overlap_buckets,
    )


def _launch_edge(
    spec: GsddmmSpec,
    graph,
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    params: EdgeBlockParams,
    traversal: TraversalOrder,
    canonical_output: bool,
) -> torch.Tensor:
    """Launch ``GSDDMM_forward_edge_block`` over the cached edge list.

    When the traversal list is source-grouped and canonical output is wanted, the
    canonical-id array goes along so the kernel's ``Edge`` reads and output rows
    stay numbered by forward-CSR position.
    """
    by_src = traversal is TraversalOrder.CSC
    edge_list = _graph_edge_list(graph, by_src=by_src)
    canonical_idx = None
    if canonical_output and not _edge_list_is_canonical(graph, by_src=by_src):
        canonical_idx = _graph_canonical_edge_idx(graph)
    return _C.gsddmm_forward_edge(
        lhs,
        rhs,
        edge_list,
        spec.op,
        spec.lhs_target,
        spec.rhs_target,
        graph.forward_indptr.numel() - 1,
        params.pipeline_stages,
        params.edges_per_warp,
        params.warps_per_block,
        canonical_idx,
    )


# =============================================================================
# Variant selection
# =============================================================================


def _geometry_prefers_node(num_nodes: int, num_edges: int, feat_dim: int) -> bool:
    """Measurement-free fallback: does the geometry favour the node kernel?

    Fitted on the A100 sweep in ``out/cuda_gsddmm/fp16_*_second_improvements``
    (3136 (op, dataset, D) cells): the node kernel wins on graphs too small to
    fill the machine, and on very dense rows at narrow features, where staging
    the shared ``Dst_V`` row once per block beats re-gathering it per edge.
    Scored against the per-cell oracle this rule costs gmean 1.011 / p90 1.019,
    versus 1.041 for always-edge and 1.403 for always-node -- good enough to
    stand in when measuring is impossible, not good enough to replace it.
    """
    if num_edges < 50_000:
        return True
    avg_degree = num_edges / max(num_nodes, 1)
    return avg_degree > 250 and feat_dim <= 64


def _default_traversal(spec: GsddmmSpec, feat_dim: int) -> TraversalOrder:
    """Traversal order to use when nothing is measured (pinned variant, or the
    measurement-free fallback).

    Source grouping shares the ``Src_V`` row across a warp's chunk, but for an
    op with no ``dst`` operand it also forces the canonical-id indirection, which
    costs an extra per-lane load, a shuffle, and scattered row stores. Measured
    on an A100 (mul(src, edge), fp16) that trade is worth it at D=128 (5-23 %
    faster than traversing CSR) and a loss at D=32 (6-29 % slower), so narrow
    features default to the CSR traversal, which needs no remap at all.
    ``variant="auto"`` does not use this -- it times both (see
    :func:`_measure_variant`).
    """
    return spec.preferred_traversal if feat_dim > 32 else TraversalOrder.CSR


def _time_plan(plan: GsddmmPlan, graph, lhs, rhs, warmup: int, iters: int) -> float:
    """Time one candidate in ms/iter; a candidate that cannot launch scores +inf.

    A candidate can legitimately fail to launch (a deep pipeline at ``D=256``
    over-subscribes shared memory; the edge candidate's edge list may not fit in
    memory), and the other variant is then simply the answer.
    """
    try:
        return time_callable(lambda: plan.launch(graph, lhs, rhs), warmup=warmup, iters=iters).ms_per_iter
    except RuntimeError as exc:  # includes torch.OutOfMemoryError
        logger.debug("gsddmm: %s candidate did not launch (%s)", plan.variant, exc)
        return float("inf")


def _plan_cache_keys(plan: GsddmmPlan, graph) -> frozenset[str]:
    """Per-graph caches the given plan reads on every launch."""
    if plan.variant == "node":
        return frozenset()
    by_src = plan.traversal is TraversalOrder.CSC
    keys = {_EDGE_LIST_SRC_KEY if by_src else _EDGE_LIST_DST_KEY}
    if plan.remaps_edge_ids(graph):
        keys.add(_CANONICAL_IDX_KEY)
    return frozenset(keys)


def _release_edge_caches(graph, keep: frozenset[str]) -> None:
    """Drop edge-variant scratch the probe materialized but the winner won't read.

    An edge list is 16 B/edge and the canonical index another 8 B/edge (1.8 GB
    and 0.9 GB on a 114 M-edge graph), and the probe may build both directions,
    so everything the winner does not read is worth releasing. Caches that
    predate the probe are left alone -- their owner asked for them.
    """
    for key in (_EDGE_LIST_DST_KEY, _EDGE_LIST_SRC_KEY, _CANONICAL_IDX_KEY):
        if key not in keep:
            graph.__dict__.pop(key, None)


def select_variant(
    spec: GsddmmSpec,
    graph,
    lhs: torch.Tensor,
    rhs: torch.Tensor | None = None,
    node_params: NodeBlockParams | None = None,
    edge_params: EdgeBlockParams | None = None,
    *,
    config=None,
    warmup: int | None = None,
    iters: int | None = None,
    force: bool = False,
    canonical_output: bool = True,
) -> GsddmmPlan:
    """Pick the faster kernel for this workload, measuring once per graph.

    The candidates (both kernels, and both traversal orders where they differ --
    see :func:`_measure_variant`) are timed at the parameters in effect on the
    first call for a given ``(spec, graph, feature width, dtype)``, and the
    verdict is memoized on the graph object, so it dies with the graph rather
    than outliving it in a process-wide table. Later parameter changes do not
    re-probe -- the decision is coarse, and re-probing per parameter tweak would
    defeat the point; pass ``force=True`` (as the autotuner's first stage does)
    to measure again with a different budget.

    ``dtype`` is part of the key, never a searched axis: it is a property of the
    caller's tensors.

    Args:
        spec: What to compute.
        graph: ``AdjacencyForwardBackwardWithNodeBuckets``.
        lhs: Left operand; its last dim and dtype are part of the cache key.
        rhs: Right operand (required unless ``spec.op == "copy"``).
        node_params: Candidate parameters for the node kernel.
        edge_params: Candidate parameters for the edge kernel.
        config: Optional ``AutotuneConfig``, read for its ``measure_variant`` and
            ``share_variant_probe`` flags.
        warmup: Warmup iterations per candidate (default :data:`_PROBE_WARMUP`).
        iters: Timed iterations per candidate (default :data:`_PROBE_ITERS`). The
            autotuner passes its own, larger budget; a plain ``variant="auto"``
            call keeps the cheap default, since it is paying for a side effect of
            its first call rather than for a tuning run it asked for.
        force: Re-measure and overwrite the memo.
        canonical_output: Passed through to the resulting plan.

    Returns:
        The resolved :class:`GsddmmPlan`.
    """
    node_params = node_params or NodeBlockParams()
    edge_params = edge_params or EdgeBlockParams()
    feat_dim = lhs.shape[-1] if lhs.ndim > 1 else 1

    share = bool(getattr(config, "share_variant_probe", False))
    key = (spec.shape_class if share else spec, feat_dim, lhs.dtype)
    # The stamp guards the memo as a whole rather than riding inside every key,
    # so a hit is one cheap comparison plus one dict lookup, and a graph whose
    # buffers changed drops its stale entries instead of accumulating them.
    stamp = _graph_stamp(graph)
    entry = graph.__dict__.get(_VARIANT_MEMO_KEY)
    if entry is None or entry[0] != stamp:
        entry = (stamp, {})
        graph.__dict__[_VARIANT_MEMO_KEY] = entry
    memo = entry[1]

    choice = None if force else memo.get(key)
    if choice is None:
        choice = _measure_variant(
            spec,
            graph,
            lhs,
            rhs,
            node_params,
            edge_params,
            config,
            feat_dim,
            canonical_output,
            warmup if warmup is not None else _PROBE_WARMUP,
            iters if iters is not None else _PROBE_ITERS,
        )
        memo[key] = choice

    variant, traversal = choice
    if variant == "node":
        return GsddmmPlan(spec, "node", node_params)
    return GsddmmPlan(
        spec=spec,
        variant="edge",
        params=edge_params,
        traversal=traversal,
        canonical_output=canonical_output,
    )


def _measure_variant(
    spec: GsddmmSpec,
    graph,
    lhs: torch.Tensor,
    rhs: torch.Tensor | None,
    node_params: NodeBlockParams,
    edge_params: EdgeBlockParams,
    config,
    feat_dim: int,
    canonical_output: bool,
    warmup: int,
    iters: int,
) -> tuple[GsddmmVariant, TraversalOrder]:
    """Time the candidate kernels and return the winning (variant, traversal).

    Three candidates, not two: the edge kernel's traversal order is also a real
    choice, and a measured one. Grouping edges by source lets a warp's chunk
    share the ``Src_V`` row, but then the canonical-id indirection has to carry
    the output back to forward-CSR numbering, which costs an extra
    8-byte-per-lane load, a shuffle, and scattered row stores. Measured on an
    A100 (mul(src, edge), fp16): that remap is worth it at D=128 (5-23 % faster
    than traversing CSR) and a loss at D=32 (32-45 % slower than the
    unrenumbered path, and 6-29 % slower than traversing CSR), and the crossover
    moves with the graph -- so it is timed rather than guessed.

    Candidates that cannot launch score +inf; only if all of them fail does this
    raise.
    """
    num_nodes = graph.forward_indptr.numel() - 1
    num_edges = graph.forward_indices.numel()

    measure = num_edges > 0 and getattr(config, "measure_variant", True)
    if not measure:
        variant: GsddmmVariant = "node" if _geometry_prefers_node(num_nodes, num_edges, feat_dim) else "edge"
        traversal = _default_traversal(spec, feat_dim)
        logger.debug(
            "gsddmm: %s -> %s/%s by geometry (N=%d, E=%d, D=%d)",
            spec.op,
            variant,
            traversal.value,
            num_nodes,
            num_edges,
            feat_dim,
        )
        return variant, traversal

    pre_existing = frozenset(graph.__dict__.keys())

    candidates = [(GsddmmPlan(spec, "node", node_params), "node", TraversalOrder.CSR)]
    for traversal in dict.fromkeys((spec.preferred_traversal, TraversalOrder.CSR)):
        plan = GsddmmPlan(spec, "edge", edge_params, traversal, canonical_output)
        candidates.append((plan, "edge", traversal))

    # Interleave the repeats rather than finishing one candidate before starting
    # the next, so clock drift or a neighbouring job cannot systematically
    # penalize whichever candidate is timed later (see _PROBE_REPEATS).
    best_per_candidate = [float("inf")] * len(candidates)
    for _ in range(_PROBE_REPEATS):
        for index, (plan, _variant, _traversal) in enumerate(candidates):
            best_per_candidate[index] = min(best_per_candidate[index], _time_plan(plan, graph, lhs, rhs, warmup, iters))

    timings = [(ms, plan, variant, traversal) for ms, (plan, variant, traversal) in zip(best_per_candidate, candidates)]
    best_ms, best_plan, best_variant, best_traversal = min(timings, key=lambda row: row[0])

    if best_ms == float("inf"):
        raise RuntimeError(
            f"gsddmm: no kernel could be launched for op={spec.op!r} with D={feat_dim} on a graph with "
            f"{num_nodes} nodes / {num_edges} edges; see the logged reasons"
        )

    _release_edge_caches(graph, keep=pre_existing | _plan_cache_keys(best_plan, graph))
    logger.debug(
        "gsddmm: %s(%s,%s) D=%d %s -> %s/%s (%s)",
        spec.op,
        spec.lhs_target,
        spec.rhs_target,
        feat_dim,
        lhs.dtype,
        best_variant,
        best_traversal.value,
        ", ".join(f"{variant}/{traversal.value} {ms:.4f} ms" for ms, _, variant, traversal in timings),
    )
    return best_variant, best_traversal


def resolve_plan(
    spec: GsddmmSpec,
    graph,
    lhs: torch.Tensor,
    rhs: torch.Tensor | None = None,
    variant: GsddmmVariantChoice = "auto",
    node_params: NodeBlockParams | None = None,
    edge_params: EdgeBlockParams | None = None,
    *,
    config=None,
    canonical_output: bool = True,
) -> GsddmmPlan:
    """Turn a caller's request into a launchable plan.

    ``variant="node"`` / ``"edge"`` pin a kernel and measure nothing;
    ``"auto"`` delegates to :func:`select_variant`.
    """
    if variant == "node":
        return GsddmmPlan(spec, "node", node_params or NodeBlockParams())
    if variant == "edge":
        return GsddmmPlan(
            spec,
            "edge",
            edge_params or EdgeBlockParams(),
            traversal=_default_traversal(spec, lhs.shape[-1] if lhs.ndim > 1 else 1),
            canonical_output=canonical_output,
        )
    if variant != "auto":
        raise ValueError(f"gsddmm: unknown variant {variant!r}; expected 'auto', 'node' or 'edge'")
    return select_variant(
        spec,
        graph,
        lhs,
        rhs,
        node_params,
        edge_params,
        config=config,
        canonical_output=canonical_output,
    )
