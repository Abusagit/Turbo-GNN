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
        edge_dim: bool = 1,
        residual: bool = True,
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
        self.edge_proj = nn.Linear(edge_dim, self.hidden_dim, bias=bias)

        self.layer_norm = nn.LayerNorm(self.hidden_dim)

        self.residual = residual

        if self.residual:
            self.residual_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=bias)
            self.gating = nn.Linear(3 * self.hidden_dim, 1, bias=False)

    def forward(self, x: torch.Tensor, graph: Any, **kwargs: Any) -> torch.Tensor:
        # get node features
        n = graph.num_nodes()
        e = graph.num_edges()

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(n, self.num_heads, -1)
        k = k.view(n, self.num_heads, -1)
        v = v.view(n, self.num_heads, -1)

        if "w" in graph.edata.keys():
            edge_w = graph.edata["w"]
            edge_w = self.edge_proj(edge_w)
            edge_w = edge_w.view(e, self.num_heads, -1)
        else:
            edge_w = torch.zeros(e, self.hidden_dim, dtype=x.dtype, device=x.device).view(e, self.num_heads, -1)

        attn_scores = ops.e_add_v(graph, edge_w, k)
        attn_scores = ops.e_dot_v(graph, attn_scores, v)
        attn_scores = F.edge_softmax(graph, attn_scores)
        values = ops.e_add_v(graph, edge_w, v)

        values = values * attn_scores
        hidden = ops.copy_e_sum(graph, values).view(n, -1)

        if self.residual:
            residual = self.residual_proj(x)
            gate = self.gating(torch.cat([residual, hidden, hidden - residual], dim=-1))
            beta = torch.nn.functional.sigmoid(gate)
            residual.mul_(beta)
            hidden.mul_(1 - beta)
            hidden.add_(residual)

        hidden = self.layer_norm(hidden)
        torch.nn.functional.relu(hidden, inplace=True)
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
