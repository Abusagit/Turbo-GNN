"""All TunableKernel subclasses for turbo_gnn kernels."""

from __future__ import annotations

import torch

from turbo_gnn._autotune import TunableKernel, TunableParam
from turbo_gnn._functions import ReductionAggrFunction, _FusedGraphAttention, gatv2_function


class ReductionAggrKernel(TunableKernel):
    """Tunable kernel callable for reduction aggregation (min/max)."""

    def __init__(self, reduce: str = "min", **kwargs):
        super().__init__()
        self.reduce = reduce
        self.forward_warps_per_block = kwargs.get("warps_per_block", 8)
        self.forward_edges_per_block_heavy_nodes = kwargs.get("edges_per_block_heavy_nodes", 128)
        self.forward_use_2d_kernel = kwargs.get("use_2d_kernel", False)
        self.forward_features_per_block = kwargs.get("features_per_block", 32)
        self.forward_tiles_y = kwargs.get("tiles_y", 8)

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
        )

    def get_tunable_forward_kernel_params(self) -> list[TunableParam]:
        return [
            TunableParam("forward_warps_per_block", [1, 2, 4, 8, 16, 32], default=8),
            TunableParam("forward_edges_per_block_heavy_nodes", [32, 64, 128, 256, 512, 1024, 2048], default=128),
            TunableParam("forward_use_2d_kernel", [True, False], default=False),
            TunableParam("forward_features_per_block", [32, 64, 128, 256], default=32),
            TunableParam("forward_tiles_y", [2, 4, 8, 16], default=128),
        ]

    def get_tunable_forward_graph_params(self) -> list[TunableParam]:
        return [
            TunableParam("forward_huge_degree_threshold_quantile", [-1, 0.9, 0.95, 0.99, 0.999], default=-1),
        ]


class GATv2AggrKernel(TunableKernel):
    """Tunable kernel callable for GATv2 aggregation."""

    def __init__(self, **kwargs):
        super().__init__()
        self.backward_grad_A_reduce_row_chunk_size = kwargs.get("grad_A_reduce_row_chunk_size", 512)
        self.forward_light_warps = kwargs.get("forward_light_warps", 1)
        self.forward_heavy_warps = kwargs.get("forward_heavy_warps", 8)
        self.backward_light_warps = kwargs.get("backward_light_warps", 1)
        self.backward_heavy_warps = kwargs.get("backward_heavy_warps", 8)

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
        )

    def get_tunable_forward_kernel_params(self) -> list[TunableParam]:
        return [
            TunableParam("forward_light_warps", [1, 2, 4], default=1),
            TunableParam("forward_heavy_warps", [8, 16, 32], default=8),
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
    """Tunable kernel callable for fused graph transformer attention."""

    def __init__(self, **kwargs):
        super().__init__()
        self.forward_light_warps = kwargs.get("forward_light_warps", 4)
        self.forward_heavy_warps = kwargs.get("forward_heavy_warps", 8)
        self.backward_light_warps = kwargs.get("backward_light_warps", 1)
        self.backward_heavy_warps = kwargs.get("backward_heavy_warps", 8)

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
        )

    def get_tunable_forward_kernel_params(self) -> list[TunableParam]:
        return [
            TunableParam("forward_light_warps", [1, 2, 4], default=4),
            TunableParam("forward_heavy_warps", [8, 16, 32], default=8),
        ]

    def get_tunable_forward_graph_params(self) -> list[TunableParam]:
        return [
            TunableParam("forward_huge_degree_threshold_quantile", [-1, 0.9, 0.95, 0.99], default=-1),
        ]

    def get_tunable_backward_kernel_params(self) -> list[TunableParam]:
        return [
            TunableParam("backward_light_warps", [1, 2, 4], default=1),
            TunableParam("backward_heavy_warps", [8, 16, 32], default=8),
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
