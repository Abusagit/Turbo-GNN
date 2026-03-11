from typing import Any, Literal

import torch
from torch import nn

from src.data.converters import AdjacencyForwardBackwardWithNodeBuckets

from ..base import AutotuneConfig, BaseBackend, BaseConvolution, TunableParam
from ..registry import BackendRegistry
from .gatv2_aggr.utils import gatv2_aggr
from .gt_aggr.utils import graph_transformer_aggr
from .reduction_aggr.utils import reduction_aggr
from .spmm_aggr.utils import spmm_aggr

doc = """
CUDA backend: wraps cuda-written kernels .
"""


class _CudaReductionAggrConv(nn.Module):
    """
    Reduction-aggregation (min/max) convolution using custom CUDA extension.

    Expects:
      - x: [N, F] float32 cuda
      - graph: (edge_ptr, edge_idx) where
            edge_ptr: [N+1] int32 cuda
            edge_idx: [E]   int32 cuda
      - light/heavy node partitions are stored as buffers inside the module
    """

    def __init__(
        self,
        /,
        reduce: str = "min",
        **kwargs,
    ) -> None:
        super().__init__()
        warps_per_block = kwargs.get("warps_per_block", 8)
        edges_per_block_heavy_nodes = kwargs.get("edges_per_block_heavy_nodes", 128)
        use_2d_kernel = kwargs.get("use_2d_kernel", False)
        features_per_block = kwargs.get("features_per_block", 32)
        tiles_y = kwargs.get("tiles_y", 8)

        self.forward_warps_per_block = warps_per_block
        self.forward_edges_per_block_heavy_nodes = edges_per_block_heavy_nodes
        self.warps_per_block = warps_per_block
        self.edges_per_block_heavy_nodes = edges_per_block_heavy_nodes
        self.reduce = reduce
        self.forward_use_2d_kernel = use_2d_kernel
        self.forward_features_per_block = features_per_block
        self.forward_tiles_y = tiles_y

    def forward(
        self,
        x: torch.Tensor,
        graph: AdjacencyForwardBackwardWithNodeBuckets,
        *,
        edge_weight: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        edge_ptr = graph.forward_indptr
        edge_idx = graph.forward_indices
        light = graph.light_nodes
        heavy = graph.heavy_nodes

        return reduction_aggr(
            edge_ptr,
            edge_idx,
            x,
            light,
            heavy,
            graph.max_degree,
            self.forward_warps_per_block,
            self.forward_edges_per_block_heavy_nodes,
            self.forward_use_2d_kernel,
            self.forward_features_per_block,
            self.forward_tiles_y,
            reduce=self.reduce,
        )


class _CudaSimpleAggrConv(BaseConvolution):
    def __init__(
        self,
        aggr_type: Literal["min", "max"] = "min",
        *,
        bias: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(bias=bias, **kwargs)
        self.conv = _CudaReductionAggrConv(reduce=aggr_type, **kwargs)

    def forward(
        self,
        x: torch.Tensor,
        graph: AdjacencyForwardBackwardWithNodeBuckets,
        *,
        edge_weight: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        return self.conv(x, graph, edge_weight=edge_weight, **kwargs)

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


class _CUDAGATv2Conv(BaseConvolution):
    """CUDA-backed GATv2Conv wrapper."""

    def __init__(
        self,
        feature_dim: int,
        bias: bool = False,
        heads: int = 1,
        negative_slope: float = 0.2,
        **kwargs: Any,
    ) -> None:
        """Initialize a GATv2 layer using DGL.

        Args:
            feature_dim (int): Input (and output) feature size.
            bias (bool): Include bias.
            **kwargs (Any): DGL GraphConv kwargs (norm, weight, ...).
        """
        super().__init__(num_heads=heads, bias=bias, **kwargs)
        self.left_right_projection = nn.Linear(feature_dim, 2 * feature_dim * heads, bias=bias)
        self._outer_proj = torch.nn.Linear(feature_dim * heads, feature_dim, bias=bias)

        self.negative_slope = negative_slope
        self.heads = heads
        self.backward_grad_A_reduce_row_chunk_size = kwargs.get("grad_A_reduce_row_chunk_size", 512)

        self.feature_dim = feature_dim
        self.head_dim = feature_dim

        self.attn_weights = nn.Parameter(torch.FloatTensor(size=(heads, feature_dim)))

        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.attn_weights, gain=gain)

    def forward(
        self,
        x: torch.Tensor,
        graph: AdjacencyForwardBackwardWithNodeBuckets,
        *,
        edge_weight: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Apply GATv2Conv.

        Args:
            x (torch.Tensor): Node features [N, Fin].
            graph (Any): Graph repr
            edge_weight (Optional[torch.Tensor]): Edge weights [E].
            **kwargs (Any): Extra kwargs (ignored).

        Returns:
            torch.Tensor: Output features [N, Fout].
        """

        x_left, x_right = self.left_right_projection(x).split(self.heads * self.head_dim, -1)
        x_left = x_left.view(-1, self.heads, self.head_dim)
        x_right = x_right.view(-1, self.heads, self.head_dim)

        indptr_forward = graph.forward_indptr
        indices_forward = graph.forward_indices
        indptr_backward = graph.backward_indptr
        indices_backward = graph.backward_indices

        out = gatv2_aggr(
            indptr_forward,
            indices_forward,
            indptr_backward,
            indices_backward,
            x_left,
            x_right,
            self.attn_weights.data,
            self.negative_slope,
            self.backward_grad_A_reduce_row_chunk_size,
        ).view(-1, self.heads * self.head_dim)

        out = self._outer_proj(out)
        return out

    def get_tunable_forward_kernel_params(self) -> list[TunableParam]:
        return []

    def get_tunable_forward_graph_params(self) -> list[TunableParam]:
        return [
            TunableParam("forward_huge_degree_threshold_quantile", [-1, 0.9, 0.95, 0.99], default=-1),
        ]

    def get_tunable_backward_kernel_params(self) -> list[TunableParam]:
        return [
            TunableParam("backward_grad_A_reduce_row_chunk_size", [16, 32, 64, 128, 256, 512, 1024, 2048], default=512),
        ]

    def get_tunable_backward_graph_params(self) -> list[TunableParam]:
        return [
            TunableParam("backward_huge_degree_threshold_quantile", [-1, 0.9, 0.95, 0.99], default=-1),
        ]


class _CudaGraphTransformerConv(BaseConvolution):
    """CUDA-based Fused graph transformer"""

    def __init__(
        self,
        feature_dim: int,
        heads: int = 8,
        **kwargs,
    ):
        super().__init__(bias=False, dropout=0.0)

        self.feature_dim = feature_dim
        self.num_heads = heads
        self.qkv_proj = nn.Linear(self.feature_dim, 3 * self.feature_dim)

        self.head_dim = self.feature_dim // self.num_heads

        self.attn_scores_multiplier = torch.rsqrt(torch.tensor(self.head_dim)).item()

    def forward(
        self,
        x: torch.Tensor,
        graph: AdjacencyForwardBackwardWithNodeBuckets,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Apply GraphConv.

        Args:
            x (torch.Tensor): Node features [N, Fin].
            graph (Any): graph representation for current backend and convolution
            **kwargs (Any): Extra kwargs (ignored).

        Returns:
            torch.Tensor: Output features [N, Fout].
        """
        x = torch.nn.functional.layer_norm(x, (x.shape[-1],))
        qkv: torch.Tensor = self.qkv_proj(x)
        q, k, v = qkv.split(self.feature_dim, -1)

        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_heads, self.head_dim)
        v = v.view(-1, self.num_heads, self.head_dim)

        edge_ptr = graph.forward_indptr
        edge_idx = graph.forward_indices
        edge_ptr_T = graph.backward_indptr
        edge_idx_T = graph.backward_indices

        out = graph_transformer_aggr(
            edge_ptr=edge_ptr,
            edge_idx=edge_idx,
            edge_ptr_T=edge_ptr_T,
            edge_idx_T=edge_idx_T,
            Q=q,
            K=k,
            V=v,
            scale=self.attn_scores_multiplier,
        ).view(-1, self.feature_dim)
        return out

    def get_tunable_forward_graph_params(self) -> list[TunableParam]:
        return [
            TunableParam("forward_huge_degree_threshold_quantile", [-1, 0.9, 0.95, 0.99], default=-1),
        ]

    def get_tunable_backward_graph_params(self) -> list[TunableParam]:
        return [
            TunableParam("backward_huge_degree_threshold_quantile", [-1, 0.9, 0.95, 0.99], default=-1),
        ]


class _CudaSpMMConv(BaseConvolution):
    """cuSPARSE SpMM convolution using AdjacencyForwardBackwardWithNodeBuckets.

    Supports float32, float16, bfloat16 features via mixed-precision cuSPARSE.
    """

    def __init__(
        self,
        norm_type: str = "none",
        cu_sparse_algorithm_id: int = -1,
        block_dim: int = 256,
        **kwargs: Any,
    ) -> None:
        super().__init__(bias=False, dropout=0.0)
        self.norm_type = norm_type
        self.cu_sparse_algorithm_id = cu_sparse_algorithm_id
        self.block_dim = block_dim

    def forward(
        self,
        x: torch.Tensor,
        graph: AdjacencyForwardBackwardWithNodeBuckets,
        **kwargs: Any,
    ) -> torch.Tensor:
        return spmm_aggr(
            x,
            graph.forward_indptr,
            graph.forward_indices,
            self.norm_type,
            self.cu_sparse_algorithm_id,
            self.block_dim,
        )


@BackendRegistry.register_backend("cuda")
class CUDABackend(BaseBackend):
    """Backend that instantiates CUDA-based convolutions."""

    def create_conv(
        self,
        conv_type: str,
        **kwargs: Any,
    ):
        """Factory for CUDA convolution layers.

        Args:
            conv_type (str): 'gat_v2', 'min_aggr', 'max_aggr', 'gt', 'sum_aggr', 'mean_aggr', 'gcn'.
            feature_dim (int): Input (and output) feature size.
            **kwargs (Any): Extra arguments for CUDA layers.

        Returns:
            BaseConvolution: An instance of the requested CUDA conv.
        """
        autotune = kwargs.pop("autotune", False)
        autotune_config = kwargs.pop("autotune_config", None)

        feature_dim = kwargs.pop("feature_dim")

        ct = conv_type.lower()
        match ct:
            case "gat_v2":
                heads = kwargs.pop("heads")
                conv = _CUDAGATv2Conv(feature_dim=feature_dim, heads=heads, **kwargs)
            case "min_aggr":
                conv = _CudaSimpleAggrConv(
                    aggr_type="min",
                    **kwargs,
                )
            case "max_aggr":
                return _CudaSimpleAggrConv(
                    aggr_type="max",
                    **kwargs,
                )
            case "gt":
                heads = kwargs.pop("heads")
                conv = _CudaGraphTransformerConv(feature_dim=feature_dim, heads=heads, **kwargs)
            case "sum_aggr":
                return _CudaSpMMConv(
                    norm_type="none",
                    cu_sparse_algorithm_id=kwargs.get("cu_sparse_algorithm_id", -1),
                    block_dim=kwargs.get("block_dim", 256),
                )
            case "mean_aggr":
                return _CudaSpMMConv(
                    norm_type="right",
                    cu_sparse_algorithm_id=kwargs.get("cu_sparse_algorithm_id", -1),
                    block_dim=kwargs.get("block_dim", 256),
                )
            case "gcn":
                return _CudaSpMMConv(
                    norm_type="both",
                    cu_sparse_algorithm_id=kwargs.get("cu_sparse_algorithm_id", -1),
                    block_dim=kwargs.get("block_dim", 256),
                )
            case _:
                raise KeyError(f"Unsupported conv_type for CUDA backend: {conv_type}")

        if autotune:
            conv.enable_autotune(config=autotune_config)

        return conv
