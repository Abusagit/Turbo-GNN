"""All TunableKernel subclasses for turbo_gnn kernels."""

from __future__ import annotations

import torch

from turbo_gnn._autotune import TunableKernel, TunableParam
from turbo_gnn._functions import ReductionAggrFunction, _FusedGraphAttention, gatv2_function


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
        self.forward_light_warps = kwargs.get("forward_light_warps", 1)
        self.forward_heavy_warps = kwargs.get("forward_heavy_warps", 8)
        self.backward_light_warps = kwargs.get("backward_light_warps", 1)
        self.backward_heavy_warps = kwargs.get("backward_heavy_warps", 8)
        self.forward_pipeline_stages = kwargs.get("pipeline_stages", 0)
        self.backward_pipeline_stages = kwargs.get("backward_pipeline_stages", 0)

    def _execute(self, graph, x, *, x_neighbors=None, attention_weights=None, negative_slope=None, **kwargs):
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
            self.forward_pipeline_stages,
            self.backward_pipeline_stages,
        )

    def get_tunable_forward_kernel_params(self) -> list[TunableParam]:
        return [
            TunableParam("forward_light_warps", [1, 2, 4], default=1),
            TunableParam("forward_heavy_warps", [8, 16, 32], default=8),
            TunableParam("forward_pipeline_stages", [0, 1], default=0),
        ]

    def get_tunable_forward_graph_params(self) -> list[TunableParam]:
        return [
            TunableParam("forward_huge_degree_threshold_quantile", [-1, 0.9, 0.95, 0.99], default=-1),
        ]

    def get_tunable_backward_kernel_params(self) -> list[TunableParam]:
        return [
            TunableParam("backward_grad_A_reduce_row_chunk_size", [16, 32, 64, 128, 256, 512, 1024, 2048], default=512),
            TunableParam("backward_light_warps", [1, 2, 4], default=1),
            TunableParam("backward_heavy_warps", [8, 16, 32], default=8),
            TunableParam("backward_pipeline_stages", [0, 1], default=0),
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
        self.forward_heavy_warps = kwargs.get("forward_heavy_warps", 8)
        self.backward_light_warps = kwargs.get("backward_light_warps", 1)
        self.backward_heavy_warps = kwargs.get("backward_heavy_warps", 8)
        self.forward_pipeline_stages = kwargs.get("pipeline_stages", 0)
        self.backward_pipeline_stages = kwargs.get("backward_pipeline_stages", 0)

    def _execute(self, graph, x, *, Q=None, K=None, V=None, scale=None, **kwargs):
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
            self.forward_pipeline_stages,
            self.backward_pipeline_stages,
        )

    def get_tunable_forward_kernel_params(self) -> list[TunableParam]:
        return [
            TunableParam("forward_light_warps", [1, 2, 4], default=4),
            TunableParam("forward_heavy_warps", [8, 16, 32], default=8),
            TunableParam("forward_pipeline_stages", [0, 1], default=0),
        ]

    def get_tunable_forward_graph_params(self) -> list[TunableParam]:
        return [
            TunableParam("forward_huge_degree_threshold_quantile", [-1, 0.9, 0.95, 0.99], default=-1),
        ]

    def get_tunable_backward_kernel_params(self) -> list[TunableParam]:
        return [
            TunableParam("backward_light_warps", [1, 2, 4], default=1),
            TunableParam("backward_heavy_warps", [8, 16, 32], default=8),
            TunableParam("backward_pipeline_stages", [0, 1], default=0),
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
