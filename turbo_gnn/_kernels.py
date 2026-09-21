"""All TunableKernel subclasses for turbo_gnn kernels.

Note on ``schedule`` / ``blocks_per_sm``: both are accepted as constructor kwargs and
forwarded to the kernels, but they are deliberately **not** ``TunableParam``s. The autotuner
takes the full Cartesian product of the declared parameters, so adding a 3-value and a
6-value axis multiplies the reduction grid from 1,344 to 24,192 combinations -- 120,960
timed trials once graph repartitioning is included, which does not finish. Sweep them
explicitly instead.

``forward_bucket_launch`` / ``backward_bucket_launch`` *are* tunable. Two values each only
doubles the relevant grid, and the right answer genuinely varies: concurrency is worth
1.11-1.14x on the forward buckets and 0.92-0.99 on the backward ones, and which side of that
a given graph lands on is not predictable from its shape. Forward and backward are separate
parameters so a search can take concurrency on one pass and decline it on the other.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, ClassVar

import torch

from turbo_gnn._autotune import TunableKernel, TunableParam
from turbo_gnn._functions import (
    DEFAULT_BLOCKS_PER_SM,
    DEFAULT_BUCKET_LAUNCH,
    DEFAULT_SCHED_CHUNK,
    DEFAULT_SCHEDULE,
    GsddmmFunction,
    GSpMMFunction,
    ReductionAggrFunction,
    _FusedGraphAttention,
    gatv2_function,
)
from turbo_gnn._gsddmm import (
    EdgeBlockParams,
    GsddmmSpec,
    GsddmmVariant,
    GsddmmVariantChoice,
    NodeBlockParams,
    _graph_canonical_edge_idx,
    _graph_edge_list,
    _graph_heavy_blocks,
    resolve_plan,
    select_variant,
)

# Re-exported for callers that import the per-graph caches from here (the
# research backends and the correctness tests); they live in _gsddmm now,
# next to the kernels that consume them.
__all__ = [
    "ReductionAggrKernel",
    "GATv2AggrKernel",
    "GraphTransformerAggrKernel",
    "GSDDMMKernel",
    "GSDDMMEdgeKernel",
    "GSpMMKernel",
    "_graph_edge_list",
    "_graph_heavy_blocks",
    "_graph_canonical_edge_idx",
]


class ReductionAggrKernel(TunableKernel):
    """Tunable kernel for min/max neighbor aggregation.

    Tunable forward parameters (grid-searched during autotuning):

    - ``forward_warps_per_block``: warps per CUDA block for the light-node
      atomic kernel. More warps = higher occupancy but diminishing returns
      when feature dim is small.
    - ``forward_edges_per_block_heavy_nodes``: edges processed per block in
      the heavy-node tiled kernel. Larger values amortize launch overhead
      but increase register pressure.
    - ``forward_use_2d_kernel``: whether to use the 2-D tiled kernel variant
      for heavy nodes (tiles over both edges and features).
    - ``forward_features_per_block``, ``forward_tiles_y``: tile dimensions
      for the 2-D kernel.
    - ``forward_pipeline_stages``: async-copy pipeline stage count for the
      light-node and packed-atomics heavy-node kernels' per-thread neighbor
      scan (0 disables the pipeline; ignored by the 2-D tiled heavy kernel,
      which already hides latency via shared-memory tree reduction).
      MEASURED REGRESSION: as of this writing, stages=1 makes these kernels
      strictly slower on H100 (0/27 and 1/27 swept configs improved; see
      results/bench_min_aggr_forward/, results/bench_max_aggr_forward/,
      results/ncu_min_aggr/) -- each block here is a single warp copying a
      tiny (<=16B) per-thread slice, so there's no other warp to hide the
      async-copy latency behind, and the pipeline's own sync overhead just
      adds cost. Kept as a tunable (default 0, off) only so autotune can
      revisit it if the kernel shape changes.

    Tunable graph parameter:

    - ``forward_huge_degree_threshold_quantile``: degree quantile for the
      light/heavy partition (-1 disables bucketing, all nodes go to light).
    """

    def __init__(self, reduce: str = "min", **kwargs):
        super().__init__()
        self.reduce = reduce
        self.forward_warps_per_block = kwargs.get("warps_per_block", 8)
        self.forward_edges_per_block_heavy_nodes = kwargs.get("edges_per_block_heavy_nodes", 128)
        self.forward_use_2d_kernel = kwargs.get("use_2d_kernel", False)
        self.forward_features_per_block = kwargs.get("features_per_block", 32)
        self.forward_tiles_y = kwargs.get("tiles_y", 8)
        self.forward_pipeline_stages = kwargs.get("pipeline_stages", 0)

    def _execute(self, graph, x, **kwargs):
        return ReductionAggrFunction.apply(
            graph.forward_indptr,
            graph.forward_indices,
            x,
            graph.light_nodes,
            graph.heavy_nodes,
            graph.max_degree,
            self.forward_warps_per_block,
            self.forward_edges_per_block_heavy_nodes,
            self.forward_use_2d_kernel,
            self.forward_features_per_block,
            self.forward_tiles_y,
            self.reduce,
            self.forward_pipeline_stages,
            graph.backward_indptr,
            graph.backward_indices,
            graph.backward_light_nodes,
            graph.backward_heavy_nodes,
            graph.backward_max_degree,
        )

    def get_tunable_forward_kernel_params(self) -> list[TunableParam]:
        return [
            TunableParam("forward_warps_per_block", [1, 2, 4, 8, 16, 32], default=8),
            TunableParam("forward_edges_per_block_heavy_nodes", [32, 64, 128, 256, 512, 1024, 2048], default=128),
            TunableParam("forward_use_2d_kernel", [True, False], default=False),
            TunableParam("forward_features_per_block", [32, 64, 128, 256], default=32),
            TunableParam("forward_tiles_y", [2, 4, 8, 16], default=128),
            TunableParam("forward_pipeline_stages", [0, 1], default=0),
        ]

    def get_tunable_forward_graph_params(self) -> list[TunableParam]:
        return [
            TunableParam("forward_huge_degree_threshold_quantile", [-1, 0.9, 0.95, 0.99, 0.999], default=-1),
        ]


class GATv2AggrKernel(TunableKernel):
    """Tunable kernel for GATv2 attention aggregation.

    Tunable backward parameter:

    - ``backward_grad_A_reduce_row_chunk_size``: number of destination-node
      rows reduced per shared-memory pass when computing attention gradients.
      Larger chunks reduce kernel launches but increase shared memory usage.
    - ``backward_pipeline_stages``: async-copy pipeline stage count for the
      backward AL/R (directed) and G/ALR (undirected) kernels, mirroring
      ``forward_pipeline_stages`` (0 disables the pipeline).

    Tunable graph parameters (forward and backward):

    - ``forward_huge_degree_threshold_quantile``: light/heavy partition for
      the forward adjacency.
    - ``backward_huge_degree_threshold_quantile``: light/heavy partition for
      the backward (transposed) adjacency used in the gradient kernel.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.backward_grad_A_reduce_row_chunk_size = kwargs.get("grad_A_reduce_row_chunk_size", 512)
        self.schedule = kwargs.get("schedule", DEFAULT_SCHEDULE)
        self.blocks_per_sm = kwargs.get("blocks_per_sm", DEFAULT_BLOCKS_PER_SM)
        self.sched_chunk = kwargs.get("sched_chunk", DEFAULT_SCHED_CHUNK)
        self.forward_bucket_launch = kwargs.get("forward_bucket_launch", DEFAULT_BUCKET_LAUNCH)
        self.backward_bucket_launch = kwargs.get("backward_bucket_launch", DEFAULT_BUCKET_LAUNCH)
        self.forward_light_warps = kwargs.get("forward_light_warps", 1)
        self.forward_heavy_warps = kwargs.get("forward_heavy_warps", 8)
        self.backward_light_warps = kwargs.get("backward_light_warps", 1)
        self.backward_heavy_warps = kwargs.get("backward_heavy_warps", 8)
        self.forward_heavy_edge_slice = kwargs.get("forward_heavy_edge_slice", 0)
        self.forward_heavy_slice_blocks_per_sm = kwargs.get("forward_heavy_slice_blocks_per_sm", 0.0)
        self.backward_heavy_slice_blocks_per_sm = kwargs.get("backward_heavy_slice_blocks_per_sm", 0.0)
        self.forward_pipeline_stages = kwargs.get("pipeline_stages", 0)
        self.forward_heavy_pipeline_stages = kwargs.get("heavy_pipeline_stages", 0)
        self.backward_pipeline_stages = kwargs.get("backward_pipeline_stages", 0)
        self.backward_heavy_pipeline_stages = kwargs.get("backward_heavy_pipeline_stages", 0)

    def _execute(self, graph, x, *, x_neighbors=None, attention_weights=None, negative_slope=None, **kwargs):
        # An explicit edge count wins; otherwise size the slice from the heavy-degree
        # threshold, so the parameter means the same thing on graphs of different density.
        slice_size = self.forward_heavy_edge_slice or graph.heavy_slice_for_blocks_per_sm(
            "forward", self.forward_heavy_slice_blocks_per_sm
        )
        table = graph.heavy_edge_slices("forward", slice_size) if slice_size > 0 else None
        bwd_slice = graph.heavy_slice_for_blocks_per_sm("forward", self.backward_heavy_slice_blocks_per_sm)
        bwd_table = graph.heavy_edge_slices("forward", bwd_slice) if bwd_slice > 0 else None

        return gatv2_function.apply(
            graph.forward_indptr,
            graph.forward_indices,
            graph.backward_indptr,
            graph.backward_indices,
            x,
            x_neighbors,
            attention_weights,
            negative_slope,
            self.backward_grad_A_reduce_row_chunk_size,
            graph.forward_light_nodes,
            graph.forward_heavy_nodes,
            graph.backward_light_nodes,
            graph.backward_heavy_nodes,
            self.forward_light_warps,
            self.forward_heavy_warps,
            self.backward_light_warps,
            self.backward_heavy_warps,
            graph.is_directed,
            self.schedule,
            self.blocks_per_sm,
            self.sched_chunk,
            self.forward_bucket_launch,
            self.backward_bucket_launch,
            slice_size,
            table.chunk_node if table is not None else None,
            table.chunk_start if table is not None else None,
            table.node_chunk_offset if table is not None else None,
            bwd_slice,
            bwd_table.chunk_node if bwd_table is not None else None,
            bwd_table.chunk_start if bwd_table is not None else None,
            bwd_table.node_chunk_offset if bwd_table is not None else None,
            self.forward_pipeline_stages,
            self.forward_heavy_pipeline_stages,
            self.backward_pipeline_stages,
            self.backward_heavy_pipeline_stages,
        )

    def get_tunable_forward_kernel_params(self) -> list[TunableParam]:
        return [
            # The node->block policy. Searched rather than swept externally: which policy wins
            # is graph-dependent -- grid_stride and precomputed each take cells the other loses
            # -- and it interacts with the warp counts, so tuning it in isolation misattributes
            # the gain. One value covers both passes, so it appears in each list.
            TunableParam(
                "schedule",
                ["one_per_block", "grid_stride", "precomputed", "dynamic"],
                default="one_per_block",
            ),
            # Concurrency helps the forward buckets (1.11-1.14x) and hurts the backward ones
            # (0.92-0.99), so the two are searched independently rather than tied together.
            TunableParam("forward_bucket_launch", ["sequential", "concurrent"], default="sequential"),
            TunableParam("forward_light_warps", [1, 2, 4], default=1),
            TunableParam("forward_heavy_warps", [8, 16, 32], default=8),
            # Edge-slice size for the heavy bucket; 0 keeps one block per heavy node, so the
            # old and new decompositions are compared inside a single search axis.
            # Slice sized to fill the device with N blocks per SM, from the heavy bucket's
            # edge count. Degree statistics do not predict the optimum (1125x spread for the
            # bucketing threshold); block count does (7x). 0 disables slicing.
            TunableParam("forward_heavy_slice_blocks_per_sm", [0, 8, 16, 32, 64], default=0),
            TunableParam("forward_pipeline_stages", [0, 2, 6, 12], default=0),
            # Separate from the light depth: the heavy bucket runs 8-32 warps per block, so
            # its staging buffer hits the shared-memory ceiling at a depth the light bucket
            # (1-4 warps) absorbs without losing occupancy.
            TunableParam("forward_heavy_pipeline_stages", [0, 2, 6, 12], default=0),
        ]

    def get_tunable_forward_graph_params(self) -> list[TunableParam]:
        return [
            TunableParam("forward_huge_degree_threshold_quantile", [-1, 0.9, 0.95, 0.99], default=-1),
        ]

    def get_tunable_backward_kernel_params(self) -> list[TunableParam]:
        return [
            # The node->block policy. Searched rather than swept externally: which policy wins
            # is graph-dependent -- grid_stride and precomputed each take cells the other loses
            # -- and it interacts with the warp counts, so tuning it in isolation misattributes
            # the gain. One value covers both passes, so it appears in each list.
            TunableParam(
                "schedule",
                ["one_per_block", "grid_stride", "precomputed", "dynamic"],
                default="one_per_block",
            ),
            TunableParam("backward_grad_A_reduce_row_chunk_size", [512, 1024], default=512),
            # Concurrency helps the forward buckets (1.11-1.14x) and hurts the backward ones
            # (0.92-0.99), so the two are searched independently rather than tied together.
            TunableParam("backward_bucket_launch", ["sequential", "concurrent"], default="sequential"),
            TunableParam("backward_light_warps", [1, 2, 4], default=1),
            TunableParam("backward_heavy_warps", [8, 16, 32], default=8),
            # The undirected backward's heavy bucket is ~81% of that pass at ~7% occupancy;
            # slicing it is the point of this axis. 0 keeps one block per heavy node.
            TunableParam("backward_heavy_slice_blocks_per_sm", [0, 8, 16, 32, 64], default=0),
            TunableParam("backward_pipeline_stages", [0, 2, 6, 12], default=0),
            # Separate from the light depth: the heavy bucket runs 8-32 warps per block, so
            # its staging buffer hits the shared-memory ceiling at a depth the light bucket
            # (1-4 warps) absorbs without losing occupancy.
            TunableParam("backward_heavy_pipeline_stages", [0, 2, 6, 12], default=0),
        ]

    def get_tunable_backward_graph_params(self) -> list[TunableParam]:
        return [
            TunableParam("backward_huge_degree_threshold_quantile", [-1, 0.9, 0.95, 0.99], default=-1),
        ]

    def make_forward_bench_fn(self, x, graph_repr, **kwargs):
        x_neighbors = kwargs["x_neighbors"]
        attention_weights = kwargs["attention_weights"]
        negative_slope = kwargs["negative_slope"]

        def _bench():
            return self._execute(
                graph_repr,
                x,
                x_neighbors=x_neighbors,
                attention_weights=attention_weights,
                negative_slope=negative_slope,
            )

        return _bench


class GSDDMMKernel(TunableKernel):
    """Tunable kernel for GSDDMM (generalized sampled dense-dense matmul).

    Computes a per-edge binary op over feature rows selected from source
    nodes, destination nodes, or edges::

        out[e] = op(lhs[sel_l], rhs[sel_r])   (elementwise; dot reduces the feature axis)

    The (op, lhs_target, rhs_target) triple is fixed at construction (one
    kernel instance per DGL-style op, e.g. ``u_sub_v`` == sub(src, dst)), as is
    ``variant``:

    - ``"node"``: always ``GSDDMM_forward_normal`` (one block per bucketed CSR row).
    - ``"edge"``: always ``GSDDMM_forward_edge_block`` (one warp per edge chunk).
    - ``"auto"`` (default): time both once per ``(graph, feature width, dtype)``
      and keep the faster one (see :func:`turbo_gnn._gsddmm.select_variant`).
      ``forward_variant`` then reports which kernel is in use.

    Output rows are numbered by forward-CSR edge position for every variant, so
    the choice is invisible to the caller.

    Tunable forward parameters, node variant:

    - ``forward_light_warps`` / ``forward_heavy_warps``: warps per block for
      the light/heavy node buckets. Only the counts the binding instantiates
      are searchable (see ``csrc/gsddmm/gsddmm_dispatch.cuh``); any other value
      raises from the dispatch.
    - ``forward_pipeline_stages``: async-copy pipeline depth for the per-edge
      Src/Edge row prefetch, in {0, 1, 2, 3} (0 disables the pipeline). Deeper
      pipelines buy more overlap but cost ``stages + 1`` shared-memory row
      slots per warp, so the whole range is searched.
    - ``forward_heavy_edges_per_block``: split each heavy node into chunks of
      this many edges, one block per chunk (0 = one block per node). Bounds
      the heavy launch's tail by the chunk instead of the maximum degree; the
      (node, chunk) descriptors are built once per graph and cached.
    - ``forward_overlap_buckets``: run the light bucket on a side CUDA stream
      concurrently with the heavy bucket (event fork/join around the pair), so
      each launch's tail is filled by the other's blocks.

    Tunable forward parameters, edge variant:

    - ``forward_edges_per_warp``: contiguous edges each warp walks, in [1, 32].
      1 is the original one-edge-per-warp layout; larger chunks amortize the
      edge-list load (one coalesced load per chunk) and are what gives the
      pipeline something to prefetch.
    - ``forward_pipeline_stages``: cp.async prefetch depth of the operand rows,
      in {0, 2, 3}. Costs ``stages + 1`` shared row slots per operand per warp;
      pointless with ``edges_per_warp == 1``.
    - ``forward_warps_per_block``: independent warps packed per thread block,
      in {1, 2, 4, 8}. 32-thread blocks cap residency at 32 blocks/SM (half the
      warp slots on sm_80); packing warps lifts that cap.

    Tunable graph parameter (node variant only -- the edge kernel ignores the
    node buckets):

    - ``forward_huge_degree_threshold_quantile``: light/heavy partition.

    Backward: ``backward_variant`` (default ``"node"``) picks which of the two
    reduction kernels ``.backward()`` runs, independently of ``variant`` -- the
    backward is a reduction rather than a map, so the forward's winner does not
    carry over. It is declared as a tunable (its own axis, searched only when
    tuning backward); the backward's other knobs are shared with the forward and
    read from the same attributes.

    Two ways to run the forward:

    - ``forward(graph, lhs, rhs)`` -- the differentiable entry. It resolves the
      plan from this kernel's current parameters, stamps ``backward_variant`` on
      it and runs it inside :class:`~turbo_gnn._functions.GsddmmFunction`, so
      the output carries a ``grad_fn`` and ``.backward()`` dispatches the
      selected backward kernel.
    - ``__call__`` / ``_execute`` -- the raw launches (detached output), which is
      what the autotuner and the variant probes time.
    """

    #: Subclasses may pin the variant; see :class:`GSDDMMEdgeKernel`.
    _PINNED_VARIANT: ClassVar[str | None] = None
    #: Number output rows by forward-CSR edge position (see :class:`GsddmmLaunchPlan`).
    _CANONICAL_OUTPUT: ClassVar[bool] = True

    def __init__(
        self,
        op: str,
        lhs_target: str,
        rhs_target: str,
        variant: GsddmmVariantChoice = "auto",
        backward_variant: GsddmmVariant = "node",
        **kwargs,
    ):
        super().__init__()
        self.spec = GsddmmSpec(op=op, lhs_target=lhs_target, rhs_target=rhs_target)
        if self._PINNED_VARIANT is not None:
            if variant not in ("auto", self._PINNED_VARIANT):
                raise ValueError(
                    f"{type(self).__name__} is pinned to variant {self._PINNED_VARIANT!r}, got {variant!r}"
                )
            variant = self._PINNED_VARIANT
        elif variant not in ("auto", "node", "edge"):
            raise ValueError(f"gsddmm: unknown variant {variant!r}; expected 'auto', 'node' or 'edge'")
        self.variant = variant
        if backward_variant not in ("node", "edge"):
            raise ValueError(f"gsddmm: unknown backward_variant {backward_variant!r}; expected 'node' or 'edge'")
        self.backward_variant = backward_variant

        # Node-block knobs.
        self.forward_light_warps = kwargs.get("light_warps_per_block", 4)
        self.forward_heavy_warps = kwargs.get("heavy_warps_per_block", 32)
        self.forward_heavy_edges_per_block = kwargs.get("heavy_edges_per_block", 0)
        self.forward_overlap_buckets = kwargs.get("overlap_buckets", False)
        # Edge-block knobs.
        self.forward_edges_per_warp = kwargs.get("edges_per_warp", 4)
        self.forward_warps_per_block = kwargs.get("warps_per_block", 4)
        # Both kernels take a cp.async prefetch depth.
        self.forward_pipeline_stages = kwargs.get("pipeline_stages", 0)
        # The resolved kernel; a searched axis only when variant == "auto".
        self.forward_variant = "node" if variant == "auto" else variant
        #: (feat_dim, plan) memo for the pinned-variant path; see __setattr__.
        self._plan_cache: tuple[int, object] | None = None

    # ---- spec passthrough, for callers that read the op off the kernel ----

    @property
    def op(self) -> str:
        return self.spec.op

    @property
    def lhs_target(self) -> str:
        return self.spec.lhs_target

    @property
    def rhs_target(self) -> str:
        return self.spec.rhs_target

    # ---- launch ----

    def _node_params(self) -> NodeBlockParams:
        return NodeBlockParams(
            light_warps=self.forward_light_warps,
            heavy_warps=self.forward_heavy_warps,
            pipeline_stages=self.forward_pipeline_stages,
            heavy_edges_per_block=self.forward_heavy_edges_per_block,
            overlap_buckets=self.forward_overlap_buckets,
        )

    def _edge_params(self) -> EdgeBlockParams:
        return EdgeBlockParams(
            pipeline_stages=self.forward_pipeline_stages,
            edges_per_warp=self.forward_edges_per_warp,
            warps_per_block=self.forward_warps_per_block,
        )

    #: Attributes a pinned-variant plan is built from. Assigning any of them
    #: drops the memo (see __setattr__); autotuning is the only writer.
    _PLAN_INPUTS: ClassVar[frozenset[str]] = frozenset(
        {
            "variant",
            "backward_variant",
            "forward_light_warps",
            "forward_heavy_warps",
            "forward_pipeline_stages",
            "forward_heavy_edges_per_block",
            "forward_overlap_buckets",
            "forward_edges_per_warp",
            "forward_warps_per_block",
        }
    )

    def __setattr__(self, name: str, value) -> None:
        """Invalidate the plan memo when a plan input is written.

        Checking the inputs on every *read* instead was measured at 4 us per
        launch in the real call path (a 10-element tuple of attribute reads),
        against the 5.8 us plan rebuild it was there to avoid -- barely a win on
        an op whose whole host cost is ~15 us. Writes happen only at
        construction and while autotuning, so paying there is free.
        """
        if name in self._PLAN_INPUTS:
            self.__dict__["_plan_cache"] = None
        object.__setattr__(self, name, value)

    def _plan(self, graph, lhs, rhs=None):
        """Resolve to one kernel carrying only that kernel's parameters.

        ``variant="auto"`` consults :func:`select_variant`, whose verdict is
        memoized on the graph -- except while autotuning, where stage A has
        already fixed ``forward_variant`` and stage B must vary only that
        kernel's own axes.
        """
        variant = self.variant
        if variant == "auto":
            if self._is_autotuning:
                variant = self.forward_variant
            else:
                plan = select_variant(
                    self.spec,
                    graph,
                    lhs,
                    rhs,
                    self._node_params(),
                    self._edge_params(),
                    config=self._autotune_config,
                    canonical_output=self._CANONICAL_OUTPUT,
                )
                # Record the decision so parameter dumps see which kernel ran.
                self.forward_variant = plan.variant
                # Stamped here too: the pinned path below does it when it builds
                # the plan, and select_variant does not know this kernel's
                # backward choice.
                return replace(plan, backward_variant=self.backward_variant)
        # Pinned variant: resolve_plan measures nothing, so the plan depends only
        # on the plan inputs (invalidated by __setattr__) plus, for the edge
        # kernel, the feature dim -- which is what picks its traversal order.
        #
        # The node kernel's plan does not depend on the feature dim, so it is not
        # read at all there. That read is not free: benchmark.py (and any caller
        # that wants a default device) runs under torch.set_default_device, which
        # puts a torch-function mode on the stack, and EVERY tensor attribute
        # access is then dispatched through it -- measured on an A100 host,
        # lhs.shape[-1] 0.29 -> 1.92 us and lhs.ndim 0.15 -> 1.73 us. Two reads
        # per launch is 3.5 us on an op whose whole host cost is ~12 us, and the
        # host is what bounds these kernels on a small graph.
        feat_dim = -1 if variant == "node" else (lhs.shape[-1] if lhs.ndim > 1 else 1)
        cached = self._plan_cache
        if cached is not None and cached[0] == feat_dim:
            return cached[1]
        plan = resolve_plan(
            self.spec,
            graph,
            lhs,
            rhs,
            variant,
            self._node_params(),
            self._edge_params(),
            canonical_output=self._CANONICAL_OUTPUT,
        )
        # backward_variant is stamped here rather than by a dataclasses.replace()
        # in forward(): replace() re-runs __init__ reflectively on every call, and
        # launch() ignores the field, so one plan serves both entry points.
        plan = replace(plan, backward_variant=self.backward_variant)
        # Direct dict write: self._plan_cache = ... would route through
        # __setattr__, which is harmless but pointless here.
        self.__dict__["_plan_cache"] = (feat_dim, plan)
        return plan

    def _execute(self, graph, x, *, rhs=None, **kwargs):
        return self._plan(graph, x, rhs).launch(graph, x, rhs)

    def forward(self, graph, lhs: torch.Tensor, rhs: torch.Tensor | None = None) -> torch.Tensor:
        """Differentiable launch, in canonical (forward-CSR) edge order.

        The counterpart of :meth:`_execute`, which stays raw so timing probes
        never pay autograd bookkeeping. This resolves the same plan from this
        kernel's current parameters -- variant probes included, and they stay
        outside the autograd graph -- stamps ``backward_variant`` on it, and
        runs it inside :class:`turbo_gnn._functions.GsddmmFunction`, so the
        output carries a ``grad_fn`` and ``.backward()`` dispatches the
        reduction kernel selected by ``backward_variant``.

        Well-defined because the plan's output is canonical: the backward
        kernels read ``d_out`` -- and number edge gradients -- by forward-CSR
        position. That is also why :class:`GSDDMMEdgeKernel` refuses this
        method: its traversal-order numbering cannot serve as ``d_out``.
        """
        plan = self._plan(graph, lhs, rhs)
        # Nothing to record: torch.autograd.Function.apply costs ~10 us of host
        # time (measured on this call, against ~12 us for the whole launch), and
        # an output with no grad_fn is what the caller would get from it anyway.
        if not (torch.is_grad_enabled() and (lhs.requires_grad or (rhs is not None and rhs.requires_grad))):
            return plan.launch(graph, lhs, rhs)
        return GsddmmFunction.apply(plan, graph, lhs, rhs)

    # ---- tunable parameter declarations ----

    @staticmethod
    def _variant_kernel_params(variant: str) -> list[TunableParam]:
        """The axes the given kernel actually reads."""
        if variant == "node":
            return [
                TunableParam("forward_light_warps", [4], default=4),
                TunableParam("forward_heavy_warps", [32], default=32),
                TunableParam("forward_pipeline_stages", [0, 1, 2, 3], default=0),
                TunableParam("forward_heavy_edges_per_block", [0, 512, 1024, 2048, 4096], default=0),
                TunableParam("forward_overlap_buckets", [False, True], default=False),
            ]
        return [
            TunableParam("forward_pipeline_stages", [0, 2, 3], default=0),
            TunableParam("forward_edges_per_warp", [1, 4, 8, 16, 32], default=4),
            TunableParam("forward_warps_per_block", [1, 2, 4, 8], default=4),
        ]

    @staticmethod
    def _variant_graph_params(variant: str) -> list[TunableParam]:
        """Graph partitioning axes -- the edge kernel ignores the node buckets."""
        if variant == "node":
            return [TunableParam("forward_huge_degree_threshold_quantile", [-1, 0.9, 0.95, 0.99], default=-1)]
        return []

    @staticmethod
    def _merge_params(params: list[TunableParam]) -> list[TunableParam]:
        """Union two variants' axes, merging the values of shared names.

        Both kernels take ``forward_pipeline_stages``, so a plain concatenation
        would declare it twice -- and a duplicated name collapses in the grid's
        config dict and would be registered twice by the Optuna driver.
        """
        merged: dict[str, TunableParam] = {}
        for param in params:
            existing = merged.get(param.name)
            if existing is None:
                merged[param.name] = param
            else:
                values = list(dict.fromkeys([*existing.values, *param.values]))
                merged[param.name] = TunableParam(param.name, values, existing.default)
        return list(merged.values())

    def get_tunable_forward_kernel_params(self) -> list[TunableParam]:
        if self.variant != "auto":
            return self._variant_kernel_params(self.variant)
        # "auto": the kernel choice is itself the first axis, and both parameter
        # sets follow so a dump of current values (scripts/benchmark.py) and the
        # Optuna driver see everything that was in play. The inline search does
        # not walk this product -- see _inline_autotune.
        return [
            TunableParam("forward_variant", ["node", "edge"], default="node"),
            *self._merge_params([*self._variant_kernel_params("node"), *self._variant_kernel_params("edge")]),
        ]

    def get_tunable_forward_graph_params(self) -> list[TunableParam]:
        return self._variant_graph_params("node" if self.variant == "auto" else self.variant)

    def get_tunable_backward_kernel_params(self) -> list[TunableParam]:
        """The backward's own axis: which of the two reduction kernels runs.

        Everything else the backward kernels read is shared with the forward and
        already declared under its names (the node-parallel backward reads the
        same ``forward_light_warps`` / ``forward_heavy_warps``; the edge-parallel
        one the same ``forward_edges_per_warp`` / ``forward_warps_per_block``),
        so declaring those again would be a dead axis. This getter is what makes
        ``kernel_params`` in the benchmark CSV name the kernel behind a
        ``--mode backward`` row.
        """
        return [TunableParam("backward_variant", ["node", "edge"], default="node")]

    # ---- autotuning ----

    def _inline_autotune(self, x, graph_repr, config=None, **kwargs):
        """Two-stage search: pick the kernel, then tune only that kernel's axes.

        A cartesian product over both variants' parameters would spend most of
        its time timing configurations the running kernel ignores (the node
        kernel never reads ``edges_per_warp``; the edge kernel never reads the
        light/heavy quantile). So stage A decides the variant and stage B
        searches that variant's grid alone: ``2 + |winner|`` timings instead of
        ``|node| x |quantiles| + |edge|``.
        """
        if self.variant != "auto":
            return super()._inline_autotune(x, graph_repr, config, **kwargs)

        config = config or self._autotune_config
        # Stage A: measure the variant with the tuning budget, overriding any
        # cheaper verdict a plain call may have memoized earlier.
        plan = select_variant(
            self.spec,
            graph_repr,
            x,
            kwargs.get("rhs"),
            self._node_params(),
            self._edge_params(),
            config=config,
            warmup=config.warmup,
            iters=config.iters,
            force=True,
            canonical_output=self._CANONICAL_OUTPUT,
        )
        self.forward_variant = plan.variant

        # Stage B: the generic grid, restricted to the winning kernel.
        result = self._grid_search(
            x,
            graph_repr,
            config,
            self._variant_kernel_params(plan.variant),
            self._variant_graph_params(plan.variant),
            **kwargs,
        )
        # The variant travels in the cached config so a cache hit reapplies it.
        result["kernel_config"] = {"forward_variant": plan.variant, **result["kernel_config"]}
        return result

    def make_forward_bench_fn(self, x, graph_repr, **kwargs):
        rhs = kwargs.get("rhs")

        def _bench():
            return self._execute(graph_repr, x, rhs=rhs)

        return _bench


class GSDDMMEdgeKernel(GSDDMMKernel):
    """GSDDMM pinned to the edge-parallel kernel, in *traversal* edge order.

    Internal: kept for benchmarking and for the correctness tests that compare
    the two decompositions directly. Unlike ``GSDDMMKernel(variant="edge")``
    this does **not** renumber its output -- the rows follow the traversal
    order, which is CSC (grouped by source) whenever no operand reads the
    destination vertex, and its ``Edge`` operand is indexed the same way.
    Anything that has to line up with the CSR kernel wants
    :class:`GSDDMMKernel`, whose output is always in forward-CSR edge order.

    Forward-only by construction: its numbering is exactly what the backward
    kernels cannot consume (they read ``d_out`` and number edge gradients by
    forward-CSR position), so :meth:`forward` refuses rather than returning a
    graph whose backward would be silently wrong. ``GSDDMMKernel(variant=
    "edge")`` is the differentiable edge-parallel op.
    """

    _PINNED_VARIANT: ClassVar[str | None] = "edge"
    _CANONICAL_OUTPUT: ClassVar[bool] = False

    def forward(self, graph, lhs: torch.Tensor, rhs: torch.Tensor | None = None) -> torch.Tensor:
        """Refused on purpose; see the class docstring."""
        raise NotImplementedError(
            "GSDDMMEdgeKernel numbers its output by traversal order, which has no correct "
            "backward (d_out would be CSC-ordered while the backward kernels read by "
            "forward-CSR position); use GSDDMMKernel(variant='edge') when you need gradients"
        )


class GraphTransformerAggrKernel(TunableKernel):
    """Tunable kernel for fused multi-head graph transformer attention.

    Tunable kernel parameters:

    - ``forward_pipeline_stages``: async-copy pipeline stage count for the
      forward kernel's Q[j]/V[j] prefetch. 0 disables the pipeline.
    - ``backward_pipeline_stages``: async-copy pipeline stage count for the
      backward kernels' neighbor-row prefetch (directed: K[i]/dO[i]; undirected:
      Q[s]/K[s]/V[s]/dO[s]). 0 disables the pipeline.

    Tunable graph partitioning:

    - ``forward_huge_degree_threshold_quantile``: light/heavy partition for
      the forward CSR.
    - ``backward_huge_degree_threshold_quantile``: light/heavy partition for
      the backward CSR.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.forward_light_warps = kwargs.get("forward_light_warps", 4)
        self.schedule = kwargs.get("schedule", DEFAULT_SCHEDULE)
        self.blocks_per_sm = kwargs.get("blocks_per_sm", DEFAULT_BLOCKS_PER_SM)
        self.sched_chunk = kwargs.get("sched_chunk", DEFAULT_SCHED_CHUNK)
        self.forward_bucket_launch = kwargs.get("forward_bucket_launch", DEFAULT_BUCKET_LAUNCH)
        self.backward_bucket_launch = kwargs.get("backward_bucket_launch", DEFAULT_BUCKET_LAUNCH)
        self.forward_heavy_warps = kwargs.get("forward_heavy_warps", 8)
        self.backward_light_warps = kwargs.get("backward_light_warps", 1)
        self.backward_heavy_warps = kwargs.get("backward_heavy_warps", 8)
        self.forward_heavy_edge_slice = kwargs.get("forward_heavy_edge_slice", 0)
        self.forward_heavy_slice_blocks_per_sm = kwargs.get("forward_heavy_slice_blocks_per_sm", 0.0)
        self.backward_heavy_edge_slice = kwargs.get("backward_heavy_edge_slice", 0)
        self.backward_heavy_slice_blocks_per_sm = kwargs.get("backward_heavy_slice_blocks_per_sm", 0.0)
        self.forward_pipeline_stages = kwargs.get("pipeline_stages", 0)
        self.backward_pipeline_stages = kwargs.get("backward_pipeline_stages", 0)
        self.forward_heavy_pipeline_stages = kwargs.get("heavy_pipeline_stages", 0)
        self.backward_heavy_pipeline_stages = kwargs.get("backward_heavy_pipeline_stages", 0)

    def _execute(self, graph, x, *, Q=None, K=None, V=None, scale=None, **kwargs):
        # A positive slice size switches the heavy bucket from one block per node to one block
        # per fixed-size run of edges. The table is cached on the graph, so it is built once per
        # (direction, slice size) rather than per call.
        # An explicit edge count wins; otherwise size the slice from the heavy-degree
        # threshold, so the parameter means the same thing on graphs of different density.
        slice_size = self.forward_heavy_edge_slice or graph.heavy_slice_for_blocks_per_sm(
            "forward", self.forward_heavy_slice_blocks_per_sm
        )
        table = graph.heavy_edge_slices("forward", slice_size) if slice_size > 0 else None
        bwd_slice = self.backward_heavy_edge_slice or graph.heavy_slice_for_blocks_per_sm(
            "backward", self.backward_heavy_slice_blocks_per_sm
        )
        bwd_table = graph.heavy_edge_slices("backward", bwd_slice) if bwd_slice > 0 else None

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
            self.forward_light_warps,
            self.forward_heavy_warps,
            self.backward_light_warps,
            self.backward_heavy_warps,
            graph.is_directed,
            self.schedule,
            self.blocks_per_sm,
            self.sched_chunk,
            self.forward_bucket_launch,
            self.backward_bucket_launch,
            slice_size,
            table.chunk_node if table is not None else None,
            table.chunk_start if table is not None else None,
            table.node_chunk_offset if table is not None else None,
            bwd_slice,
            bwd_table.chunk_node if bwd_table is not None else None,
            bwd_table.chunk_start if bwd_table is not None else None,
            bwd_table.node_chunk_offset if bwd_table is not None else None,
            self.forward_pipeline_stages,
            self.forward_heavy_pipeline_stages,
            self.backward_pipeline_stages,
            self.backward_heavy_pipeline_stages,
        )

    def get_tunable_forward_kernel_params(self) -> list[TunableParam]:
        return [
            # The node->block policy. Searched rather than swept externally: which policy wins
            # is graph-dependent -- grid_stride and precomputed each take cells the other loses
            # -- and it interacts with the warp counts, so tuning it in isolation misattributes
            # the gain. One value covers both passes, so it appears in each list.
            TunableParam(
                "schedule",
                ["one_per_block", "grid_stride", "precomputed", "dynamic"],
                default="one_per_block",
            ),
            # Concurrency helps the forward buckets (1.11-1.14x) and hurts the backward ones
            # (0.92-0.99), so the two are searched independently rather than tied together.
            TunableParam("forward_bucket_launch", ["sequential", "concurrent"], default="sequential"),
            TunableParam("forward_light_warps", [1, 2, 4], default=4),
            TunableParam("forward_heavy_warps", [8, 16, 32], default=8),
            # Edge-slice size for the heavy bucket. 0 keeps one block per heavy node, so the old
            # and new decompositions are compared inside a single search axis rather than behind
            # a separate flag. Larger slices mean fewer, longer blocks and less scratch memory.
            # Slice sized to fill the device with N blocks per SM, from the heavy bucket's
            # edge count. Degree statistics do not predict the optimum (1125x spread for the
            # bucketing threshold); block count does (7x). 0 disables slicing.
            TunableParam("forward_heavy_slice_blocks_per_sm", [0, 8, 16, 32, 64], default=0),
            TunableParam("forward_pipeline_stages", [0, 2, 6, 12], default=0),
            # Separate from the light depth: the heavy bucket runs 8-32 warps per block, so its
            # staging buffer hits the shared-memory ceiling at a depth the light bucket absorbs.
            TunableParam("forward_heavy_pipeline_stages", [0, 2, 6, 12], default=0),
        ]

    def get_tunable_forward_graph_params(self) -> list[TunableParam]:
        return [
            TunableParam("forward_huge_degree_threshold_quantile", [-1, 0.9, 0.95, 0.99], default=-1),
        ]

    def get_tunable_backward_kernel_params(self) -> list[TunableParam]:
        return [
            # The node->block policy. Searched rather than swept externally: which policy wins
            # is graph-dependent -- grid_stride and precomputed each take cells the other loses
            # -- and it interacts with the warp counts, so tuning it in isolation misattributes
            # the gain. One value covers both passes, so it appears in each list.
            TunableParam(
                "schedule",
                ["one_per_block", "grid_stride", "precomputed", "dynamic"],
                default="one_per_block",
            ),
            # Concurrency helps the forward buckets (1.11-1.14x) and hurts the backward ones
            # (0.92-0.99), so the two are searched independently rather than tied together.
            TunableParam("backward_bucket_launch", ["sequential", "concurrent"], default="sequential"),
            TunableParam("backward_light_warps", [1, 2, 4], default=1),
            TunableParam("backward_heavy_warps", [8, 16, 32], default=8),
            # Backward slices the transpose CSR, so it gets its own size. 0 keeps the
            # node-per-block heavy path.
            TunableParam("backward_heavy_slice_blocks_per_sm", [0, 8, 16, 32, 64], default=0),
            TunableParam("backward_pipeline_stages", [0, 2, 6, 12], default=0),
            # Separate from the light depth: the heavy bucket runs 8-32 warps per block, so
            # its staging buffer hits the shared-memory ceiling at a depth the light bucket
            # (1-4 warps) absorbs without losing occupancy.
            TunableParam("backward_heavy_pipeline_stages", [0, 2, 6, 12], default=0),
        ]

    def get_tunable_backward_graph_params(self) -> list[TunableParam]:
        return [
            TunableParam("backward_huge_degree_threshold_quantile", [-1, 0.9, 0.95, 0.99], default=-1),
        ]

    def make_forward_bench_fn(self, x, graph_repr, **kwargs):
        Q = kwargs["Q"]
        K = kwargs["K"]
        V = kwargs["V"]
        scale = kwargs["scale"]

        def _bench():
            return self._execute(
                graph_repr,
                x,
                Q=Q,
                K=K,
                V=V,
                scale=scale,
            )

        return _bench


class GSpMMKernel(TunableKernel):
    """Tunable kernel for generalized SpMM (the ``dgl.ops.gspmm`` family).

    One singleton per ``(op, reduce)`` pair -- different operations have
    different optimal launch shapes, and the autotune cache is keyed on the
    kernel instance.

    Tunable forward parameters:

    - ``forward_warps_per_block``: block size for the light-node kernel, as
      warps. Features are spread across threadIdx.x and nodes across
      threadIdx.y, so this trades feature parallelism against node parallelism.
    - ``forward_features_per_block``, ``forward_tiles_y``: tile shape of the
      heavy-node kernel. ``tiles_y`` is how many edge chunks are reduced in
      parallel through shared memory and must be a power of two.

    Tunable graph parameter:

    - ``forward_huge_degree_threshold_quantile``: degree quantile for the
      light/heavy partition (-1 disables bucketing, all nodes go to light).

    Unlike :class:`ReductionAggrKernel` there is no ``use_2d_kernel`` knob: the
    packed-atomics heavy variant does not generalize past min/max, so the tiled
    kernel is the only heavy path.
    """

    def __init__(self, op: str = "copy_u", reduce: str = "sum", **kwargs):
        super().__init__()
        self.op = op
        self.reduce = reduce
        self.forward_warps_per_block = kwargs.get("warps_per_block", 8)
        self.forward_features_per_block = kwargs.get("features_per_block", 32)
        self.forward_tiles_y = kwargs.get("tiles_y", 8)
        self.forward_pipeline_stages = kwargs.get("pipeline_stages", 0)

    def _execute(self, graph, x, *, rhs=None, **kwargs):
        from turbo_gnn.ops import _gspmm_apply

        return _gspmm_apply(
            graph,
            x,
            rhs,
            self.op,
            self.reduce,
            self.forward_warps_per_block,
            self.forward_features_per_block,
            self.forward_tiles_y,
            self.forward_pipeline_stages,
        )

    def get_tunable_forward_kernel_params(self) -> list[TunableParam]:
        return [
            TunableParam("forward_warps_per_block", [1, 2, 4, 8, 16, 32], default=8),
            TunableParam("forward_features_per_block", [32, 64, 128, 256], default=32),
            TunableParam("forward_tiles_y", [2, 4, 8, 16], default=8),
            TunableParam("forward_pipeline_stages", [0, 1], default=0),
        ]

    def get_tunable_forward_graph_params(self) -> list[TunableParam]:
        return [
            TunableParam("forward_huge_degree_threshold_quantile", [-1, 0.9, 0.95, 0.99, 0.999], default=-1),
        ]

    def make_forward_bench_fn(self, x, graph_repr, **kwargs):
        rhs = kwargs.get("rhs")

        def _bench():
            return self._execute(graph_repr, x, rhs=rhs)

        return _bench
