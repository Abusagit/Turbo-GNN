from typing import Any, Literal

import torch
from torch import nn

from src.data.converters import AdjacencyForwardBackwardWithNodeBuckets

from ..base import BaseBackend, BaseConvolution
from ..registry import BackendRegistry
from .gatv2_aggr.utils import GATv2AggrKernel, gatv2_aggr
from .gt_aggr.utils import GraphTransformerAggrKernel, graph_transformer_aggr
from .reduction_aggr.utils import ReductionAggrKernel, reduction_aggr
from .spmm_aggr.utils import spmm_aggr

doc = """
CUDA backend: wraps cuda-written kernels .
"""


class _CudaSimpleAggrConv(BaseConvolution):
    def __init__(
        self,
        aggr_type: Literal["min", "max"] = "min",
        *,
        bias: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(bias=bias, **kwargs)
        self.aggr_type = aggr_type
        self.kernel = ReductionAggrKernel(reduce=aggr_type, **kwargs)
        self.register_kernel(self.kernel)

    def forward(
        self,
        x: torch.Tensor,
        graph: AdjacencyForwardBackwardWithNodeBuckets,
        *,
        edge_weight: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        return reduction_aggr(graph, x, reduce=self.aggr_type)


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
        super().__init__(num_heads=heads, bias=bias, **kwargs)
        self.left_right_projection = nn.Linear(feature_dim, 2 * feature_dim * heads, bias=bias)
        self._outer_proj = torch.nn.Linear(feature_dim * heads, feature_dim, bias=bias)

        self.negative_slope = negative_slope
        self.heads = heads

        self.feature_dim = feature_dim
        self.head_dim = feature_dim

        self.attn_weights = nn.Parameter(torch.FloatTensor(size=(heads, feature_dim)))

        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.attn_weights, gain=gain)

        self.kernel = GATv2AggrKernel(**kwargs)
        self.register_kernel(self.kernel)

    def forward(
        self,
        x: torch.Tensor,
        graph: AdjacencyForwardBackwardWithNodeBuckets,
        *,
        edge_weight: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        x_left, x_right = self.left_right_projection(x).split(self.heads * self.head_dim, -1)
        x_left = x_left.view(-1, self.heads, self.head_dim)
        x_right = x_right.view(-1, self.heads, self.head_dim)

        out = gatv2_aggr(
            graph,
            x_left,
            x_right,
            self.attn_weights.data,
            self.negative_slope,
        ).view(-1, self.heads * self.head_dim)

        out = self._outer_proj(out)
        return out


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

        self.kernel = GraphTransformerAggrKernel()
        self.register_kernel(self.kernel)

    def forward(
        self,
        x: torch.Tensor,
        graph: AdjacencyForwardBackwardWithNodeBuckets,
        **kwargs: Any,
    ) -> torch.Tensor:
        x = torch.nn.functional.layer_norm(x, (x.shape[-1],))
        qkv: torch.Tensor = self.qkv_proj(x)
        q, k, v = qkv.split(self.feature_dim, -1)

        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_heads, self.head_dim)
        v = v.view(-1, self.num_heads, self.head_dim)

        return graph_transformer_aggr(
            graph,
            x,
            q,
            k,
            v,
            self.attn_scores_multiplier,
        ).view(-1, self.feature_dim)


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
