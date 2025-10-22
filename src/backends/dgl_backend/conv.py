from typing import Any, Optional

import dgl
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


class _DglGraphConv(BaseConvolution):
    """DGL-backed GraphConv wrapper."""

    def __init__(self, feature_dim: int, norm: str, bias: bool = False, **kwargs: Any) -> None:
        """Initialize a GraphConv layer using DGL.

        Args:
            feature_dim (int): Input (and output) feature size.
            norm (str): How to apply the normalizer.
            bias (bool): Include bias.
            **kwargs (Any): DGL GraphConv kwargs (weight, ...).
        """
        super().__init__(bias=bias, **kwargs)
        self._conv = GraphConv(
            in_feats=feature_dim,
            out_feats=feature_dim,
            norm=norm,
            weight=False,
            bias=False,
            allow_zero_in_degree=True,
            **kwargs,
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

    def __init__(self, bias: bool = True, **kwargs: Any) -> None:
        """Initialize a MinAggr layer using DGL.

        Args:
            bias (bool): Include bias.
            **kwargs (Any): Reserved for future options.
        """
        super().__init__(bias=bias, **kwargs)

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

    def __init__(self, bias: bool = True, **kwargs: Any) -> None:
        """Initialize a MaxAggr layer using DGL.

        Args:
            bias (bool): Include bias.
            **kwargs (Any): Reserved for future options.
        """
        super().__init__(bias=bias, **kwargs)

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

    def __init__(self, feature_dim: int, bias: bool = False, heads: int = 1, **kwargs: Any) -> None:
        """Initialize a GATv2 layer using DGL.

        Args:
            feature_dim (int): Input (and output) feature size.
            bias (bool): Include bias.
            **kwargs (Any): DGL GraphConv kwargs (norm, weight, ...).
        """
        super().__init__(num_heads=heads, bias=bias, **kwargs)

        self._conv = _GAT(feature_dim, feature_dim, num_heads=heads, bias=bias, allow_zero_in_degree=True, **kwargs)
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
        feature_dim: int,
        heads: int = 8,
        **kwargs: Any,
    ) -> None:
        super().__init__(feature_dim=feature_dim, heads=heads, **kwargs)

        assert feature_dim % heads == 0, "hidden_dim must be divisible by num_heads"
        self.feature_dim = feature_dim
        self.num_heads = heads

        self.q_proj = nn.Linear(self.feature_dim, self.feature_dim)
        self.k_proj = nn.Linear(self.feature_dim, self.feature_dim)
        self.v_proj = nn.Linear(self.feature_dim, self.feature_dim)

        self.attn_scores_multiplier = torch.rsqrt(torch.tensor(self.feature_dim // self.num_heads))

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
        attn_scores *= self.attn_scores_multiplier
        attn_probs = F.edge_softmax(graph, attn_scores)

        hidden = ops.u_mul_e_sum(graph, v, attn_probs).view(n, -1)

        return hidden


@BackendRegistry.register_backend("dgl")
class DglBackend(BaseBackend):
    """Backend that instantiates DGL-based convolutions."""

    def create_conv(
        self,
        conv_type: str,
        **kwargs: Any,
    ):
        """Factory for DGL convolution layers.

        Args:
            conv_type (str): 'gcn' or 'gat_v2' currently. (Extend with GIN/SAGE as needed.)
            feature_dim (int): Input (and output) feature size.
            **kwargs (Any): Extra arguments for DGL layers.

        Returns:
            BaseConvolution: An instance of the requested DGL conv.
        """
        feature_dim = kwargs.pop("feature_dim")

        ct = conv_type.lower()
        match ct:
            case "min_aggr":
                return _DGLMinAggrConv()
            case "max_aggr":
                return _DGLMaxAggrConv()
            case "gcn":
                return _DglGraphConv(feature_dim=feature_dim, norm="both")
            case "mean_aggr":
                return _DglGraphConv(feature_dim=feature_dim, norm="right")
            case "sum_aggr":
<<<<<<< HEAD
                return _DglGraphConv(feature_dim=feature_dim, norm="none")
            case "gat":
                heads = kwargs.pop("heads")
                return _DGLGATv2Conv(feature_dim=feature_dim, heads=heads)
            case "gt":
                heads = kwargs.pop("heads")
                return _DglGraphTransformer(feature_dim=feature_dim, heads=heads)
=======
                return _DglGraphConv(in_channels, out_channels, norm="none", **kwargs)
            case "gat_v2":
                return _DGLGATv2Conv(in_channels, out_channels, **kwargs)
>>>>>>> b663137 (RENAME)
        raise KeyError(f"Unsupported conv_type for DGL backend: {conv_type}")
