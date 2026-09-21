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
    GSpMMFunction,
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
