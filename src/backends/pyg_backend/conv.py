from typing import Any, Optional

import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv as _GAT
from torch_geometric.nn import GCNConv, GINConv, SAGEConv

from ..base import BaseBackend, BaseConvolution
from ..registry import BackendRegistry

doc = """
PyG backend: wraps torch_geometric.nn layers and exposes them via BaseBackend.
"""


class _PygGCNConv(BaseConvolution):
    """PyG-backed GCNConv wrapper."""

    def __init__(self, in_channels: int, out_channels: int, bias: bool = True, **kwargs: Any) -> None:
        """Initialize a GCN convolution using PyG.

        Args:
            in_channels (int): Input feature size.
            out_channels (int): Output feature size.
            bias (bool): Whether to include bias.
            **kwargs (Any): Any torch_geometric.nn.GCNConv kwargs (e.g., normalize).
        """
        super().__init__(in_channels, out_channels, bias=bias, **kwargs)

        self._conv = GCNConv(in_channels, out_channels, bias=bias, **kwargs)

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


class _PygGATConv(BaseConvolution):
    """PyG-backed GAT (v2 if available)."""

    def __init__(self, in_channels: int, out_channels: int, bias: bool = True, heads: int = 1, **kwargs: Any) -> None:
        """Initialize a GAT convolution using PyG.

        Args:
            in_channels (int): Input feature size.
            out_channels (int): Output feature size per head or aggregated.
            bias (bool): Include bias.
            heads (int): Number of attention heads.
            **kwargs (Any): PyG GAT conv kwargs (concat, dropout, etc.).
        """
        super().__init__(in_channels, out_channels, bias=bias, heads=heads, **kwargs)

        self._conv = _GAT(in_channels, out_channels, heads=heads, bias=bias, **kwargs)

    def forward(
        self,
        x: torch.Tensor,
        graph: Any,
        *,
        edge_weight: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Apply GAT conv.

        Args:
            x (torch.Tensor): Node features [N, Fin].
            graph (Any): PyG Data or (edge_index, edge_weight).
            edge_weight (Optional[torch.Tensor]): Ignored by classic GAT.
            **kwargs (Any): Extra kwargs ignored.

        Returns:
            torch.Tensor: Output features [N, Fout] (aggregated per PyG behavior).
        """
        edge_index, edge_weight = graph
        return self._conv(x, edge_index)


class _PygSAGEConv(BaseConvolution):
    """PyG-backed GraphSAGE."""

    def __init__(self, in_channels: int, out_channels: int, bias: bool = True, **kwargs: Any) -> None:
        """Initialize a SAGE convolution using PyG.

        Args:
            in_channels (int): Input feature size.
            out_channels (int): Output feature size.
            bias (bool): Include bias.
            **kwargs (Any): PyG SAGEConv kwargs (aggr='mean'/'sum'/'max', etc.).
        """
        super().__init__(in_channels, out_channels, bias=bias, **kwargs)

        self._conv = SAGEConv(in_channels, out_channels, bias=bias, **kwargs)

    def forward(
        self,
        x: torch.Tensor,
        graph: Any,
        *,
        edge_weight: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Apply SAGE conv.

        Args:
            x (torch.Tensor): Node features [N, Fin].
            graph (Any): PyG Data or (edge_index, edge_weight).
            edge_weight (Optional[torch.Tensor]): Ignored by default SAGE.
            **kwargs (Any): Extra kwargs ignored.

        Returns:
            torch.Tensor: Output features [N, Fout].
        """
        edge_index, edge_weight = graph
        return self._conv(x, edge_index)


class _PygGINConv(BaseConvolution):
    """PyG-backed GIN (uses MLP/nn for `nn` argument)."""

    def __init__(self, in_channels: int, out_channels: int, bias: bool = True, **kwargs: Any) -> None:
        """Initialize a GIN convolution using PyG.

        Args:
            in_channels (int): Input feature size.
            out_channels (int): Output feature size.
            bias (bool): Include bias in internal MLP.
            **kwargs (Any): PyG GINConv kwargs (eps, train_eps, etc.).
        """
        super().__init__(in_channels, out_channels, bias=bias, **kwargs)

        mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels, bias=bias),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels, bias=bias),
        )
        self._conv = GINConv(nn=mlp, **kwargs)

    def forward(
        self,
        x: torch.Tensor,
        graph: Any,
        *,
        edge_weight: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Apply GIN conv.

        Args:
            x (torch.Tensor): Node features [N, Fin].
            graph (Any): PyG Data or (edge_index, edge_weight).
            edge_weight (Optional[torch.Tensor]): Ignored by default GIN.
            **kwargs (Any): Extra kwargs ignored.

        Returns:
            torch.Tensor: Output features [N, Fout].
        """
        edge_index, edge_weight = graph
        return self._conv(x, edge_index)


@BackendRegistry.register_backend("pyg")
class PygBackend(BaseBackend):
    """Backend that instantiates PyG-based convolutions."""

    def create_conv(
        self,
        conv_type: str,
        in_channels: int,
        out_channels: int,
        **kwargs: Any,
    ):
        """Factory for PyG convolution layers.

        Args:
            conv_type (str): 'gcn' | 'gat' | 'sage' | 'gin'.
            in_channels (int): Input feature size.
            out_channels (int): Output feature size.
            **kwargs (Any): Extra arguments passed to the underlying PyG layer.

        Returns:
            BaseConvolution: An instance of the requested PyG conv.
        """
        ct = conv_type.lower()

        if ct == "gcn":
            return _PygGCNConv(in_channels, out_channels, **kwargs)
        if ct == "gat":
            return _PygGATConv(in_channels, out_channels, **kwargs)
        if ct == "sage":
            return _PygSAGEConv(in_channels, out_channels, **kwargs)
        if ct == "gin":
            return _PygGINConv(in_channels, out_channels, **kwargs)
        raise KeyError(f"Unsupported conv_type for PyG backend: {conv_type}")
