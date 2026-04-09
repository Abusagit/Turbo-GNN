from typing import Any, Literal, Optional

import torch
import torch.nn as nn
from pylibcugraphops.pytorch import operators

from ..base import BaseAggr, BaseBackend, BaseConvolution, ConvAsAggr
from ..registry import BackendRegistry

doc = """
Cugraph Backend: implementations using `pylibcugraph` library.
"""


class _CugraphGATv2Conv(BaseConvolution):
    """GATv2 conv with CuGraph backend"""

    def __init__(self, feature_dim: int, bias: bool = True, heads: int = 1, **kwargs: Any) -> None:
        """Initialize a Torch-native GCN.

        Args:
            feature_dim (int): Input (and output) feature size.
            bias (bool): Include bias in linear transform.
            **kwargs (Any): Reserved for future options.
        """
        super().__init__(bias=bias, **kwargs)
        self.linear_gat_projection = nn.Linear(feature_dim, heads * feature_dim, bias=False)  # NOTE init from PyG
        self.attn_weights = nn.Parameter(torch.empty(heads, feature_dim))

        self.outer_projection = nn.Linear(heads * feature_dim, feature_dim)
        self.heads = heads
        self.feature_dim = feature_dim

        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.attn_weights, gain=gain)

    def forward(
        self,
        x: torch.Tensor,
        graph: Any,
        *,
        edge_weight: torch.Tensor | None = None,  # ignored for baseline
        **kwargs: Any,
    ) -> torch.Tensor:
        """Apply GATv2 layer

        Args:
            x (torch.Tensor): Node features [N, Fin].
            graph (Any): Either (edge_index, num_nodes) or (edge_index, edge_weight) or (edge_index, ew, num_nodes).
            edge_weight (Optional[torch.Tensor]): Unused baseline.
            **kwargs (Any): Extra kwargs ignored.

        Returns:
            torch.Tensor: Output features [N, Fout].
        """
        csc_graph, _gcn_weights_unused = graph
        x = self.linear_gat_projection(x)

        out = operators.mha_gat_v2_n2n(
            feat=x,
            attn_weights=self.attn_weights.flatten(),
            graph=csc_graph,
            num_heads=self.heads,
            activation="LeakyReLU",
            negative_slope=0.2,
            concat_heads=True,
        )
        out = self.outer_projection(out)
        return out


class _CugraphGraphTransfomerConv(BaseConvolution):
    """Graph Transformer conv with CuGraph backend"""

    def __init__(self, feature_dim: int, bias: bool = True, heads: int = 1, **kwargs: Any) -> None:
        """Initialize a Torch-native GCN.

        Args:
            feature_dim (int): Input (and output) feature size.
            bias (bool): Include bias in linear transform.
            **kwargs (Any): Reserved for future options.
        """
        super().__init__(bias=bias, **kwargs)

        self.heads = heads
        self.feature_dim = self.head_dim = feature_dim
        self.hidden_dim = self.head_dim * self.heads

        self.qkv_proj = nn.Linear(feature_dim, 3 * heads * feature_dim)

        self.outer_proj = nn.Linear()

    def forward(
        self,
        x: torch.Tensor,
        graph: Any,
        *,
        edge_weight: torch.Tensor | None = None,  # ignored for baseline
        **kwargs: Any,
    ) -> torch.Tensor:
        """Apply Graph Transformer layer

        Args:
            x (torch.Tensor): Node features [N, Fin].
            graph (Any): Either (edge_index, num_nodes) or (edge_index, edge_weight) or (edge_index, ew, num_nodes).
            edge_weight (Optional[torch.Tensor]): Unused baseline.
            **kwargs (Any): Extra kwargs ignored.

        Returns:
            torch.Tensor: Output features [N, Fout].
        """
        csc_graph, _gcn_weights_unused = graph

        qkv: torch.Tensor = self.qkv_proj(x)
        query, key, value = qkv.split(self.hidden_dim, -1)

        out = operators.mha_simple_n2n(
            key_emb=key,
            query_emb=query,
            value_emb=value,
            graph=csc_graph,
            num_heads=self.heads,
            concat_heads=True,
            edge_emb=None,
            norm_by_dim=False,
            score_bias=None,
        )

        out = self.outer_proj(out)
        return out


class _SimpleAggrGraphConv(BaseConvolution):
    """
    Simple Graph convolutions with cugraph backend: everything which can be done with a single sparse aggregation
    (mean/max/GCN/etc.)
    """

    def __init__(
        self,
        bias: bool = True,
        use_edge_weights: bool = False,
        aggr_type: Literal["sum", "max", "min", "mean"] = "sum",
        **kwargs: Any,
    ) -> None:
        """Initialize a Torch-native GCN.

        Args:
            bias (bool): Include bias in linear transform.
            **kwargs (Any): Reserved for future options.
        """
        super().__init__(bias=bias, **kwargs)
        self.use_edge_weights = use_edge_weights
        self.aggr_type = aggr_type

    def forward(
        self,
        x: torch.Tensor,
        graph: Any,
        *,
        edge_weight: torch.Tensor | None = None,  # ignored for baseline
        **kwargs: Any,
    ) -> torch.Tensor:
        """Apply Graph Transformer layer

        Args:
            x (torch.Tensor): Node features [N, Fin].
            graph (Any): Either (edge_index, num_nodes) or (edge_index, edge_weight) or (edge_index, ew, num_nodes).
            edge_weight (Optional[torch.Tensor]): Unused baseline.
            **kwargs (Any): Extra kwargs ignored.

        Returns:
            torch.Tensor: Output features [N, Fout].
        """
        csc_graph, gcn_weights = graph
        weights = gcn_weights if self.use_edge_weights else None
        out = operators.agg_simple_n2n(
            feat=x,
            graph=csc_graph,
            aggr=self.aggr_type,
            edge_weight=weights,
        )

        return out


@BackendRegistry.register_backend("cugraph")
class CugraphBackend(BaseBackend):
    """Backend instantiating Cugraph-powered convolutions."""

    def create_conv(
        self,
        conv_type: str,
        **kwargs: Any,
    ):
        """Factory for Torch-native mean aggregation convs.

        Args:
            conv_type (str): supported convolution type.
            feature_dim (int): Input (and output) feature size.
            **kwargs (Any): Extra kwargs.

        Returns:
            BaseConvolution: Convolution layer for CuGraph backend
        """
        feature_dim = kwargs.pop("feature_dim")

        if conv_type == "mean_aggr":
            return _SimpleAggrGraphConv(aggr_type="mean", use_edge_weights=False)
        if conv_type == "sum_aggr":
            return _SimpleAggrGraphConv(aggr_type="sum", use_edge_weights=False)
        if conv_type == "min_aggr":
            return _SimpleAggrGraphConv(aggr_type="min", use_edge_weights=False)
        if conv_type == "max_aggr":
            return _SimpleAggrGraphConv(aggr_type="max", use_edge_weights=False)
        if conv_type == "gcn":
            return _SimpleAggrGraphConv(aggr_type="sum", use_edge_weights=True)
        if conv_type == "gat_v2":
            return _CugraphGATv2Conv(feature_dim)
        if conv_type == "graph_transformer":
            raise NotImplementedError("mha_simple_n2n is broken and doesn't work with correct inputs")

    def create_aggr(self, conv_type: str, **kwargs: Any) -> BaseAggr:
        feature_dim = kwargs.pop("feature_dim", None)
        ct = conv_type.lower()
        # Simple aggregations are already projection-free
        if ct in ("sum_aggr", "mean_aggr", "min_aggr", "max_aggr", "gcn"):
            return ConvAsAggr(self.create_conv(ct, feature_dim=feature_dim, **kwargs))
        raise KeyError(f"Unsupported conv_type for CuGraph aggr (projections not separable): {conv_type}")
