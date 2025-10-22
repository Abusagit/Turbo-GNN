from typing import Any, TypedDict

import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv as _GAT
from torch_geometric.nn import GCNConv as _GCN

from ..base import BaseBackend, BaseConvolution
from ..registry import BackendRegistry

doc = """
PyG backend: wraps torch_geometric.nn layers and exposes them via BaseBackend.
"""


class _PygGCNConv(BaseConvolution):
    """PyG-backed GCNConv wrapper."""

    def __init__(self, feature_dim: int, bias: bool = False, **kwargs: Any) -> None:
        """Initialize a GCN convolution using PyG.

        Args:
            bias (bool): Whether to include bias.
            **kwargs (Any): Any torch_geometric.nn.GCNConv kwargs (e.g., normalize).
        """
        super().__init__(bias=bias, **kwargs)

        self._conv = _GCN(in_channels=feature_dim, out_channels=feature_dim, bias=bias, **kwargs)
        self._conv.lin = torch.nn.Identity()  # disable weight

    def forward(
        self,
        x: torch.Tensor,
        graph: Any,
        *,
        edge_weight: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Apply GCNConv.

        Args:
            x (torch.Tensor): Node features [N, Fin].
            graph (Any): PyG Data or (edge_index, edge_weight).
            edge_weight (Optional[torch.Tensor]): Edge weights [E].
            **kwargs (Any): Extra kwargs ignored.

        Returns:
            torch.Tensor: Output features [N, Fout].
        """
        edge_index, edge_weight = graph
        return self._conv(x, edge_index, edge_weight=edge_weight)


class _PygGATv2Conv(BaseConvolution):
    """PyG-backed GATv2."""

    def __init__(self, feature_dim: int, bias: bool = False, heads: int = 1, **kwargs: Any) -> None:
        """Initialize a GAT convolution using PyG.

        Args:
            feature_dim (int): Input (and output) feature size.
            bias (bool): Include bias.
            heads (int): Number of attention heads.
            **kwargs (Any): PyG GAT conv kwargs (concat, dropout, etc.).
        """
        super().__init__(bias=bias, heads=heads, **kwargs)

        self._conv = _GAT(in_channels=feature_dim, out_channels=feature_dim, heads=heads, bias=bias, **kwargs)
        self._outer_proj = torch.nn.Linear(
            feature_dim * heads, feature_dim, bias=bias
        )  # NOTE GAT produces 3D tensor [*, heads, feature_dim] --> Need to project it to [*, feature_dim]

    def forward(
        self,
        x: torch.Tensor,
        graph: Any,
        *,
        edge_weight: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Apply GATv2 conv.

        Args:
            x (torch.Tensor): Node features [N, Fin].
            graph (Any): PyG Data or (edge_index, edge_weight).
            edge_weight (Optional[torch.Tensor]): Ignored by classic GATv2.
            **kwargs (Any): Extra kwargs ignored.

        Returns:
            torch.Tensor: Output features [N, Fout] (aggregated per PyG behavior).
        """
        edge_index, edge_weight = graph
        return self._outer_proj(self._conv(x, edge_index))


@BackendRegistry.register_backend("pyg")
class PygBackend(BaseBackend):
    """Backend that instantiates PyG-based convolutions."""

    def create_conv(
        self,
        conv_type: str,
        **kwargs: Any,
    ) -> BaseConvolution:
        """Factory for PyG convolution layers.

        Args:
            conv_type (str): 'gcn' | 'gat_v2' | 'sage' | 'gin'.
            feature_dim (int): Input (and output) feature size.
            **kwargs (Any): Extra arguments passed to the underlying PyG layer.

        Returns:
            BaseConvolution: An instance of the requested PyG conv.
        """
        feature_dim = kwargs.pop("feature_dim")

        ct = conv_type.lower()
        match ct:
            case "gcn":
                return _PygGCNConv(feature_dim)
            case "mean_aggr":
                return _PygGCNConv(feature_dim, aggr="mean", normalize=False)
            case "sum_aggr":
                return _PygGCNConv(feature_dim, normalize=False)
            case "gat_v2":
                heads = kwargs.pop("heads")
                return _PygGATv2Conv(feature_dim, heads=heads, **kwargs)
        raise KeyError(f"Unsupported conv_type for PyG backend: {conv_type}")
