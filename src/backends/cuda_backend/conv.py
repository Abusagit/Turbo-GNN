from typing import Any, Literal

import torch

from ..base import BaseBackend, BaseConvolution
from ..registry import BackendRegistry
from .min_aggr.utils import min_aggr

doc = """
CUDA Backend: custom CUDA extensions (min_aggr + backward).
Expects CSR graph representation: (edge_ptr, edge_idx).
"""


class _CudaMinAggrConv(BaseConvolution):
    """
    Min-aggregation convolution using custom CUDA extension.

    Expects:
      - x: [N, F] float32 cuda
      - graph: (edge_ptr, edge_idx) where
            edge_ptr: [N+1] int32 cuda
            edge_idx: [E]   int32 cuda
      - light/heavy node partitions are stored as buffers inside MinAggr module
    """

    def __init__(
        self,
        *,
        bias: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(bias=bias, **kwargs)
        if bias:
            raise NotImplementedError("bias=True is not supported for pure min aggregation in this backend.")
        warps_per_block = kwargs.get("warps_per_block", 8)
        edges_per_block_heavy_nodes = kwargs.get("edges_per_block_heavy_nodes", 128)

        self.warps_per_block = warps_per_block
        self.edges_per_block_heavy_nodes = edges_per_block_heavy_nodes

        self.forward_meta = {
            "warps_per_block": warps_per_block,
            "edges_per_block_heavy_nodes": edges_per_block_heavy_nodes,
        }
        self.backward_meta = {"warps_per_block": warps_per_block}

    def forward(
        self,
        x: torch.Tensor,
        graph: Any,
        *,
        edge_weight: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if len(graph) >= 4:
            edge_ptr, edge_idx, light, heavy = graph[0], graph[1], graph[2], graph[3]
        elif len(graph) == 3:
            edge_ptr, edge_idx, _w = graph
            raise ValueError("cuda_backend needs (edge_ptr, edge_idx, light, heavy) in graph.")
        else:
            edge_ptr, edge_idx = graph
            raise ValueError("cuda_backend needs (edge_ptr, edge_idx, light, heavy) in graph.")

        return min_aggr(edge_ptr, edge_idx, x, light, heavy, self.warps_per_block, self.edges_per_block_heavy_nodes)


class _CudaSimpleAggrConv(BaseConvolution):
    def __init__(
        self,
        aggr_type: Literal["min"] = "min",
        *,
        bias: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(bias=bias, **kwargs)
        if aggr_type != "min":
            raise NotImplementedError(f"Only aggr_type='min' is implemented, got {aggr_type}")
        self.conv = _CudaMinAggrConv(bias=bias)

    def forward(
        self,
        x: torch.Tensor,
        graph: Any,
        *,
        edge_weight: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        return self.conv(x, graph, edge_weight=edge_weight, **kwargs)


@BackendRegistry.register_backend("cuda")
class CudaBackend(BaseBackend):
    """Backend instantiating custom CUDA-powered convolutions."""

    def create_conv(self, conv_type: str, **kwargs: Any):
        """
        Factory for CUDA backend convs.

        Supported:
          - "min_aggr"
        """
        _ = kwargs.pop("feature_dim", None)

        if conv_type == "min_aggr":
            return _CudaSimpleAggrConv(
                aggr_type="min",
                bias=False,
            )

        raise KeyError(f"Unsupported conv_type for cuda backend: {conv_type}")
from typing import Any

import torch
from torch import nn

from ..base import BaseBackend, BaseConvolution
from ..registry import BackendRegistry
from .utils import gatv2_function

doc = """
CUDA backend: wraps cuda-written kernels .
"""


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

        self.left_projection = nn.Linear(feature_dim, feature_dim * heads, bias=bias)
        self.rgith_projection = nn.Linear(feature_dim, feature_dim * heads, bias=bias)

        self._outer_proj = torch.nn.Linear(feature_dim * heads, feature_dim, bias=bias)

        self.negative_slope = negative_slope
        self.heads = heads

        if heads > 1:
            raise NotImplementedError("Heads > 1 is not implemented yet!!!!!!")

        # self.attn_weights = nn.Parameter(torch.FloatTensor(size=(heads, feature_dim)))
        self.attn_weights = nn.Parameter(torch.FloatTensor(size=(1, feature_dim)))

        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.attn_weights, gain=gain)

    def forward(
        self,
        x: torch.Tensor,
        graph: Any,
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

        x_left = self.left_projection(x)
        x_right = self.rgith_projection(x)

        (
            indptr_forward,
            indices_forward,
            indptr_backward,
            indices_backward,
        ) = graph

        out = gatv2_function.apply(
            indptr_forward,
            indices_forward,
            indptr_backward,
            indices_backward,
            x_left,
            x_right,
            self.attn_weights.data.flatten(),
            self.negative_slope,
        )
        out = self._outer_proj(out)
        return out


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
            conv_type (str): 'gcn' or 'gat_v2' currently. (Extend with GIN/SAGE as needed.)
            feature_dim (int): Input (and output) feature size.
            **kwargs (Any): Extra arguments for CUDA layers.

        Returns:
            BaseConvolution: An instance of the requested CUDA conv.
        """
        feature_dim = kwargs.pop("feature_dim")

        ct = conv_type.lower()
        match ct:
            case "gat_v2":
                heads = kwargs.pop("heads")
                return _CUDAGATv2Conv(feature_dim=feature_dim, heads=heads)
        raise KeyError(f"Unsupported conv_type for DGL backend: {conv_type}")
