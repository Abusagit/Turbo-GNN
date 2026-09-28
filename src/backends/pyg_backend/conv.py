from typing import Any, TypedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GATv2Conv
from torch_geometric.nn import GCNConv as _GCN
from torch_geometric.nn.inits import glorot
from torch_geometric.utils import scatter, softmax

from ..base import BaseAggr, BaseBackend, BaseConvolution, ConvAsAggr
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


class _PygGATv1Conv(BaseConvolution):
    """PyG-backed GATv1 (just GAT)."""

    def __init__(self, feature_dim: int, bias: bool = False, heads: int = 1, **kwargs: Any) -> None:
        """Initialize a GAT convolution using PyG.

        Args:
            feature_dim (int): Input (and output) feature size.
            bias (bool): Include bias.
            heads (int): Number of attention heads.
            **kwargs (Any): PyG GAT conv kwargs (concat, dropout, etc.).
        """
        super().__init__(bias=bias, heads=heads, **kwargs)

        self._conv = GATConv(in_channels=feature_dim, out_channels=feature_dim, heads=heads, bias=bias, **kwargs)
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
        """Apply GAT conv.

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


class _PygGATv2Conv(BaseConvolution):
    """PyG-backed GATv2."""

    def __init__(self, feature_dim: int, bias: bool = False, heads: int = 1, **kwargs: Any) -> None:
        """Initialize a GATv2 convolution using PyG.

        Args:
            feature_dim (int): Input (and output) feature size.
            bias (bool): Include bias.
            heads (int): Number of attention heads.
            **kwargs (Any): PyG GATv2 conv kwargs (concat, dropout, etc.).
        """
        super().__init__(bias=bias, heads=heads, **kwargs)

        self._conv = GATv2Conv(in_channels=feature_dim, out_channels=feature_dim, heads=heads, bias=bias, **kwargs)
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


class _PygGATv2Aggr(BaseAggr):
    """Aggregation-only PyG GATv2 (no linear projections).

    The message passing is lifted verbatim out of ``GATv2Conv``: its
    ``edge_update`` computes

        x = x_i + x_j; alpha = (leaky_relu(x) * att).sum(-1); alpha = softmax(alpha, index)

    and its ``message`` returns ``x_j * alpha``, summed at the destination.
    That is reproduced here on PyG's own ``softmax`` and ``scatter`` primitives,
    so no ``lin_l``/``lin_r`` exist to be bypassed and none of PyG's
    MessagePassing dispatch runs. Only ``att`` is kept, initialised with PyG's
    ``glorot`` exactly as ``GATv2Conv.reset_parameters`` does.
    """

    def __init__(self, heads: int, head_dim: int, negative_slope: float = 0.2, **kwargs: Any) -> None:
        super().__init__(conv_type="gat_v2")
        self.heads = heads
        self.head_dim = head_dim
        self.negative_slope = negative_slope
        self.att = nn.Parameter(torch.empty(1, heads, head_dim))
        glorot(self.att)

    def forward(self, x_left: torch.Tensor, x_right: torch.Tensor, graph: Any, **kwargs: Any) -> torch.Tensor:
        edge_index, _edge_weight = graph
        src, dst = edge_index[0], edge_index[1]
        num_nodes = x_left.size(0)

        # x_j is source-indexed, x_i destination-indexed; our x_right is the
        # source/neighbour tensor and x_left the destination one.
        x_j = x_right.index_select(0, src)
        x_i = x_left.index_select(0, dst)

        alpha = (F.leaky_relu(x_i + x_j, self.negative_slope) * self.att).sum(dim=-1)
        alpha = softmax(alpha, dst, num_nodes=num_nodes)

        out = scatter(x_j * alpha.unsqueeze(-1), dst, dim=0, dim_size=num_nodes, reduce="sum")
        return out.view(-1, self.heads * self.head_dim)


class _PygGTAggr(BaseAggr):
    """Aggregation-only PyG graph transformer (no QKV projection).

    ``TransformerConv.message`` computes

        alpha = (query_i * key_j).sum(-1) / sqrt(out_channels); alpha = softmax(alpha, index)

    and returns ``value_j * alpha``, summed at the destination. That is
    reproduced here on PyG's own ``softmax`` and ``scatter``, so the layer's
    ``lin_query``/``lin_key``/``lin_value`` (and the ``lin_skip`` PyG allocates
    whenever ``concat=True``) are never created. The aggregation holds no
    parameters at all.
    """

    def __init__(self, heads: int, head_dim: int, **kwargs: Any) -> None:
        super().__init__(conv_type="gt")
        self.heads = heads
        self.head_dim = head_dim
        # PyG divides by sqrt(out_channels), i.e. sqrt(head_dim).
        self.scale = head_dim**-0.5

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, graph: Any, **kwargs: Any) -> torch.Tensor:
        edge_index, _edge_weight = graph
        src, dst = edge_index[0], edge_index[1]
        num_nodes = Q.size(0)

        # PyG's query is destination-indexed and its key source-indexed, while
        # turbo_gnn puts Q on the source and K on the destination -- hence the swap.
        query_i = K.index_select(0, dst)
        key_j = Q.index_select(0, src)
        value_j = V.index_select(0, src)

        alpha = (query_i * key_j).sum(dim=-1) * self.scale
        alpha = softmax(alpha, dst, num_nodes=num_nodes)

        out = scatter(value_j * alpha.view(-1, self.heads, 1), dst, dim=0, dim_size=num_nodes, reduce="sum")
        return out.view(-1, self.heads * self.head_dim)


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
            case "gat":
                heads = kwargs.pop("heads")
                return _PygGATv1Conv(feature_dim, heads=heads, **kwargs)
            case "gat_v2":
                heads = kwargs.pop("heads")
                return _PygGATv2Conv(feature_dim, heads=heads, **kwargs)
        raise KeyError(f"Unsupported conv_type for PyG backend: {conv_type}")

    def create_aggr(self, conv_type: str, **kwargs: Any) -> BaseAggr:
        """Build a projection-free aggregation.

        GCN-style convs are already projection-free and are simply wrapped;
        the attention convs bypass PyG's internal projections and call the
        message-passing entry points directly. Head dimensions follow the
        shared convention: gat_v2 uses feature_dim per head, gt splits
        feature_dim across heads.
        """
        feature_dim = kwargs.pop("feature_dim", None)
        ct = conv_type.lower()
        match ct:
            case "gcn":
                conv = _PygGCNConv(feature_dim)
            case "mean_aggr":
                conv = _PygGCNConv(feature_dim, aggr="mean", normalize=False)
            case "sum_aggr":
                conv = _PygGCNConv(feature_dim, normalize=False)
            case "gat_v2":
                heads = kwargs.pop("heads", 1)
                return _PygGATv2Aggr(heads=heads, head_dim=feature_dim, **kwargs)
            case "gt":
                heads = kwargs.pop("heads", 8)
                return _PygGTAggr(heads=heads, head_dim=feature_dim // heads, **kwargs)
            case _:
                raise KeyError(f"Unsupported conv_type for PyG aggr: {conv_type}")
        # _PygGCNConv is already projection-free, wrap it as BaseAggr
        return ConvAsAggr(conv)
