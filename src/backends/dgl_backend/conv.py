from typing import Any, Optional

import dgl
import torch
from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch.conv import GATv2Conv as _GAT

from ..base import BaseBackend, BaseConvolution
from ..registry import BackendRegistry

doc = """
DGL backend: wraps dgl.nn layers behind the BaseBackend interface.
"""


class _DglGraphConv(BaseConvolution):
    """DGL-backed GraphConv wrapper."""

    def __init__(self, in_channels: int, out_channels: int, norm: str, bias: bool = False, **kwargs: Any) -> None:
        """Initialize a GraphConv layer using DGL.

        Args:
            in_channels (int): Input feature size.
            out_channels (int): Output feature size.
            norm (str): How to apply the normalizer.
            bias (bool): Include bias.
            **kwargs (Any): DGL GraphConv kwargs (weight, ...).
        """
        super().__init__(in_channels, out_channels, bias=bias, **kwargs)
        self._conv = GraphConv(
            in_channels, out_channels, norm=norm, weight=False, bias=False, allow_zero_in_degree=True, **kwargs
        )

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


class _DGLMinAggrConv(BaseConvolution):
    """DGL-backed MinAggregation wrapper."""

    def __init__(self, in_channels: int, out_channels: int, bias: bool = True, **kwargs: Any) -> None:
        """Initialize a MinAggr layer using DGL.

        Args:
            in_channels (int): Input feature size.
            out_channels (int): Output feature size.
            bias (bool): Include bias.
            **kwargs (Any): Reserved for future options.
        """
        super().__init__(in_channels, out_channels, bias=bias, **kwargs)

    def forward(
        self,
        x: torch.Tensor,
        graph: Any,
        *,
        edge_weight: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Apply DglMinAggrOp.

        Args:
            x (torch.Tensor): Node features [N, Fin].
            graph (Any): dgl.DGLGraph or (edge_index, edge_weight, num_nodes).
            edge_weight (Optional[torch.Tensor]): Edge weights [E].
            **kwargs (Any): Extra kwargs (ignored).

        Returns:
            torch.Tensor: Output features [N, Fout].
        """
        x_aggregated = dgl.ops.copy_u_min(graph, x)
        x_aggregated[x_aggregated.isinf()] = 0
        return x_aggregated


class _DGLMaxAggrConv(BaseConvolution):
    """DGL-backed MinAggregation wrapper."""

    def __init__(self, in_channels: int, out_channels: int, bias: bool = True, **kwargs: Any) -> None:
        """Initialize a MaxAggr layer using DGL.

        Args:
            in_channels (int): Input feature size.
            out_channels (int): Output feature size.
            bias (bool): Include bias.
            **kwargs (Any): Reserved for future options.
        """
        super().__init__(in_channels, out_channels, bias=bias, **kwargs)

    def forward(
        self,
        x: torch.Tensor,
        graph: Any,
        *,
        edge_weight: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Apply DglMinAggrOp.

        Args:
            x (torch.Tensor): Node features [N, Fin].
            graph (Any): dgl.DGLGraph or (edge_index, edge_weight, num_nodes).
            edge_weight (Optional[torch.Tensor]): Edge weights [E].
            **kwargs (Any): Extra kwargs (ignored).

        Returns:
            torch.Tensor: Output features [N, Fout].
        """
        x_aggregated = dgl.ops.copy_u_max(graph, x)
        x_aggregated[x_aggregated.isinf()] = 0
        return x_aggregated


class _DGLGATv2Conv(BaseConvolution):
    """DGL-backed GATv2Conv wrapper."""

    def __init__(self, in_channels: int, out_channels: int, bias: bool = False, heads: int = 1, **kwargs: Any) -> None:
        """Initialize a GATv2 layer using DGL.

        Args:
            in_channels (int): Input feature size.
            out_channels (int): Output feature size.
            bias (bool): Include bias.
            **kwargs (Any): DGL GraphConv kwargs (norm, weight, ...).
        """
        super().__init__(in_channels, out_channels, num_heads=heads, bias=bias, **kwargs)

        self._conv = _GAT(in_channels, out_channels, num_heads=heads, bias=bias, allow_zero_in_degree=True, **kwargs)
        self._outer_proj = torch.nn.Linear(
            out_channels * heads, out_channels, bias=bias
        )  # NOTE GAT produces 3D tensor [*, heads, out_channels] --> Need to project it to [*, out_channels]

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
            graph (Any): dgl.DGLGraph or (edge_index, edge_weight, num_nodes).
            edge_weight (Optional[torch.Tensor]): Edge weights [E].
            **kwargs (Any): Extra kwargs (ignored).

        Returns:
            torch.Tensor: Output features [N, Fout].
        """
        x = self._conv(graph, x, get_attention=False)
        x = x.view(x.shape[0], -1)
        x = self._outer_proj(x)
        return x


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
            conv_type (str): 'gcn' or 'gat' currently. (Extend with GIN/SAGE as needed.)
            in_channels (int): Input feature size.
            out_channels (int): Output feature size.
            **kwargs (Any): Extra arguments for DGL layers.

        Returns:
            BaseConvolution: An instance of the requested DGL conv.
        """
        ct = conv_type.lower()
        match ct:
            case "min_aggr":
                return _DGLMinAggrConv(in_channels, out_channels, **kwargs)
            case "max_aggr":
                return _DGLMaxAggrConv(in_channels, out_channels, **kwargs)
            case "gcn":
                return _DglGraphConv(in_channels, out_channels, norm="both", **kwargs)
            case "mean_aggr":
                return _DglGraphConv(in_channels, out_channels, norm="right", **kwargs)
            case "sum_aggr":
                return _DglGraphConv(in_channels, out_channels, norm="none", **kwargs)
            case "gat":
                return _DGLGATv2Conv(in_channels, out_channels, **kwargs)
        raise KeyError(f"Unsupported conv_type for DGL backend: {conv_type}")
