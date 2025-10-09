from typing import Any, Optional

import torch
from dgl.nn.pytorch import GraphConv

from ..base import BaseBackend, BaseConvolution
from ..registry import BackendRegistry

doc = """
DGL backend: wraps dgl.nn layers behind the BaseBackend interface.
"""


class _DglGCNConv(BaseConvolution):
    """DGL-backed GCNConv wrapper."""

    def __init__(self, in_channels: int, out_channels: int, bias: bool = True, **kwargs: Any) -> None:
        """Initialize a GCN layer using DGL.

        Args:
            in_channels (int): Input feature size.
            out_channels (int): Output feature size.
            bias (bool): Include bias.
            **kwargs (Any): DGL GraphConv kwargs (norm, weight, ...).
        """
        super().__init__(in_channels, out_channels, bias=bias, **kwargs)
        self._conv = GraphConv(in_channels, out_channels, bias=bias, allow_zero_in_degree=True, **kwargs)

    def forward(
        self,
        x: torch.Tensor,
        graph: Any,
        *,
        edge_weight: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Apply GraphConv.

        Args:
            x (torch.Tensor): Node features [N, Fin].
            graph (Any): dgl.DGLGraph or (edge_index, edge_weight, num_nodes).
            edge_weight (Optional[torch.Tensor]): Edge weights [E].
            **kwargs (Any): Extra kwargs (ignored).

        Returns:
            torch.Tensor: Output features [N, Fout].
        """
        return self._conv(graph, x, edge_weight=graph.edata.get("w"))


@BackendRegistry.register_backend("dgl")
class DglBackend(BaseBackend):
    """Backend that instantiates DGL-based convolutions."""

    def create_conv(
        self,
        conv_type: str,
        in_channels: int,
        out_channels: int,
        **kwargs: Any,
    ):
        """Factory for DGL convolution layers.

        Args:
            conv_type (str): 'gcn' currently. (Extend with GAT/GIN/SAGE as needed.)
            in_channels (int): Input feature size.
            out_channels (int): Output feature size.
            **kwargs (Any): Extra arguments for DGL layers.

        Returns:
            BaseConvolution: An instance of the requested DGL conv.
        """
        ct = conv_type.lower()
        if ct == "gcn":
            return _DglGCNConv(in_channels, out_channels, **kwargs)
        # TODO: Add DGL GAT/SAGE/GIN when needed
        raise KeyError(f"Unsupported conv_type for DGL backend: {conv_type}")
