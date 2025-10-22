from typing import Any, Optional

import dgl.nn.functional as F
import torch
import torch.nn as nn
from dgl import ops
from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch.conv import GATv2Conv as _GAT

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
        self._conv = GraphConv(in_channels, out_channels, weight=False, bias=False, allow_zero_in_degree=True, **kwargs)

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


class _DGLGATv2Conv(BaseConvolution):
    """DGL-backed GATv2Conv wrapper."""

    def __init__(self, in_channels: int, out_channels: int, bias: bool = True, heads: int = 1, **kwargs: Any) -> None:
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


class _DglGraphTransformer(BaseConvolution):
    """DGL-backed GraphTransformer wrapper."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_heads: int = 8,
        bias: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(in_channels, out_channels, **kwargs)
        self.hidden_dim = out_channels
        self.num_heads = num_heads

        assert self.hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.q_proj = nn.Linear(in_channels, self.hidden_dim, bias=bias)
        self.k_proj = nn.Linear(in_channels, self.hidden_dim, bias=bias)
        self.v_proj = nn.Linear(in_channels, self.hidden_dim, bias=bias)

        self.attn_scores_multiplier = 1 / torch.tensor(self.hidden_dim // num_heads).sqrt()

    def forward(self, x: torch.Tensor, graph: Any, **kwargs: Any) -> torch.Tensor:
        # get node features
        n = graph.num_nodes()

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(n, self.num_heads, -1)
        k = k.view(n, self.num_heads, -1)
        v = v.view(n, self.num_heads, -1)

        attn_scores = ops.u_dot_v(graph, q, k)
        attn_scores = attn_scores * self.attn_scores_multiplier
        attn_probs = F.edge_softmax(graph, attn_scores)

        hidden = ops.u_mul_e_sum(graph, v, attn_probs).view(n, -1)

        return hidden


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
        if ct == "gcn":
            return _DglGCNConv(in_channels, out_channels, **kwargs)
        if ct == "gat":
            return _DGLGATv2Conv(in_channels, out_channels, **kwargs)
        # TODO: Add DGL GAT/SAGE/GIN when needed
        elif ct == "gt":
            return _DglGraphTransformer(in_channels, out_channels, **kwargs)
        raise KeyError(f"Unsupported conv_type for DGL backend: {conv_type}")
