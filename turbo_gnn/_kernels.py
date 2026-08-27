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

from typing import Any

import torch

from turbo_gnn._autotune import TunableKernel, TunableParam
from turbo_gnn._functions import (
    DEFAULT_BLOCKS_PER_SM,
    DEFAULT_BUCKET_LAUNCH,
    DEFAULT_SCHED_CHUNK,
    DEFAULT_SCHEDULE,
    ReductionAggrFunction,
    _FusedGraphAttention,
    gatv2_function,
)


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

    Tunable graph parameter:

    - ``forward_huge_degree_threshold_quantile``: degree quantile for the
      light/heavy partition (-1 disables bucketing, all nodes go to light).
    """

    def __init__(self, reduce: str = "min", **kwargs):
        super().__init__()
        self.reduce = reduce
        self.schedule = kwargs.get("schedule", DEFAULT_SCHEDULE)
        self.blocks_per_sm = kwargs.get("blocks_per_sm", DEFAULT_BLOCKS_PER_SM)
        self.sched_chunk = kwargs.get("sched_chunk", DEFAULT_SCHED_CHUNK)
        self.forward_bucket_launch = kwargs.get("forward_bucket_launch", DEFAULT_BUCKET_LAUNCH)
        # The reduction backward is a single kernel over all nodes with no light/heavy split,
        # so there is nothing to overlap there and nothing to tune.
        self.backward_bucket_launch = DEFAULT_BUCKET_LAUNCH
        self.forward_warps_per_block = kwargs.get("warps_per_block", 8)
        self.forward_edges_per_block_heavy_nodes = kwargs.get("edges_per_block_heavy_nodes", 128)
        self.forward_heavy_edge_slice = kwargs.get("forward_heavy_edge_slice", 0)
        self.forward_use_2d_kernel = kwargs.get("use_2d_kernel", False)
        self.forward_features_per_block = kwargs.get("features_per_block", 32)
        self.forward_tiles_y = kwargs.get("tiles_y", 8)

    def _execute(self, graph, x, **kwargs):
        slice_size = self.forward_heavy_edge_slice
        table = graph.heavy_edge_slices("forward", slice_size) if slice_size > 0 else None

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
            self.schedule,
            self.blocks_per_sm,
            self.sched_chunk,
            self.forward_bucket_launch,
            self.backward_bucket_launch,
            slice_size,
            table.chunk_node if table is not None else None,
            table.chunk_start if table is not None else None,
        )

    def canonicalise_config(self, config: dict[str, Any]) -> dict[str, Any]:
        """`features_per_block` and `tiles_y` are read only by the 2-D heavy-node kernel.

        With ``forward_use_2d_kernel=False`` the two are dead, so their 16 combinations all
        describe one behaviour. Pinning them to their defaults there takes the reduction's
        forward kernel grid from 2,688 configurations to 1,428 -- only 1.9x, because the 2-D
        branch where they *do* matter is the larger half -- but it also stops the argmin being
        decided by measurement noise among sixteen candidates identical by construction, which
        is the more important of the two effects.
        """
        if config.get("forward_use_2d_kernel") is False:
            config = dict(config, forward_features_per_block=32, forward_tiles_y=8)
        # `edges_per_block_heavy_nodes` sizes the rectangular tiled grid, which a positive
        # `forward_heavy_edge_slice` replaces outright. Its three values then describe one
        # behaviour, so pin it and let the search spend its time on distinctions that exist.
        if config.get("forward_heavy_edge_slice", 0) > 0:
            config = dict(config, forward_edges_per_block_heavy_nodes=128)
        return config

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
            TunableParam("forward_warps_per_block", [1, 2, 4, 8, 16, 32], default=8),
            TunableParam("forward_edges_per_block_heavy_nodes", [128, 512, 1024], default=128),
            # Flat edge-slice grid for the heavy bucket, replacing the rectangular
            # (num_heavy, ceil(max_degree / EPB)) one, which over-launches badly when a single
            # hub dominates the degree spread. 0 keeps the rectangular path.
            TunableParam("forward_heavy_edge_slice", [0, 256, 1024], default=0),
            TunableParam("forward_use_2d_kernel", [True, False], default=False),
            TunableParam("forward_features_per_block", [32, 64, 128, 256], default=32),
            TunableParam("forward_tiles_y", [2, 4, 8, 16], default=128),
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

    def _execute(self, graph, x, *, x_neighbors=None, attention_weights=None, negative_slope=None, **kwargs):
        slice_size = self.forward_heavy_edge_slice
        table = graph.heavy_edge_slices("forward", slice_size) if slice_size > 0 else None

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
            TunableParam("forward_heavy_edge_slice", [0, 256, 1024], default=0),
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


class GraphTransformerAggrKernel(TunableKernel):
    """Tunable kernel for fused multi-head graph transformer attention.

    No tunable kernel parameters (the kernel is fully fused).  Only graph
    partitioning can be tuned:

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
        self.backward_heavy_edge_slice = kwargs.get("backward_heavy_edge_slice", 0)

    def _execute(self, graph, x, *, Q=None, K=None, V=None, scale=None, **kwargs):
        # A positive slice size switches the heavy bucket from one block per node to one block
        # per fixed-size run of edges. The table is cached on the graph, so it is built once per
        # (direction, slice size) rather than per call.
        slice_size = self.forward_heavy_edge_slice
        table = graph.heavy_edge_slices("forward", slice_size) if slice_size > 0 else None
        bwd_slice = self.backward_heavy_edge_slice
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
            TunableParam("forward_heavy_edge_slice", [0, 256, 1024], default=0),
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
            TunableParam("backward_heavy_edge_slice", [0, 256, 1024], default=0),
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
