from typing import Any, Optional

import torch
import torch.nn as nn

from ..base import BaseBackend, BaseConvolution
from ..registry import BackendRegistry
from .kernels_impl import WSBGraphTransformer, WSBSpMM

doc = """
Triton backend currently support block-sparse format
"""


class _TritonBlockSparseGraphConv(BaseConvolution):
    """Triton-backed GraphConv wrapper."""

    def __init__(self, feature_dim: int, norm: str, bias: bool = False, **kwargs: Any) -> None:
        """Initialize a GraphConv layer similar to DGL.

        Args:
            feature_dim (int): Input (and output) feature size.
            norm (str): How to apply the normalizer.
            bias (bool): Include bias.
            **kwargs (Any): DGL GraphConv kwargs (weight, ...).
        """
        super().__init__(bias=bias, **kwargs)

        self.norm = norm
        self.feature_dim = feature_dim

    def forward(
        self,
        x: torch.Tensor,
        graph,
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
        return WSBSpMM.apply(x, graph)


class _TritonBlockSparseGraphTransformerConv(BaseConvolution):
    """Triton-backed GraphTransformer wrapper."""

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

        self.qkv_proj = nn.Linear(self.feature_dim, 3 * self.feature_dim)

        self.attn_scores_multiplier = torch.rsqrt(torch.tensor(self.feature_dim // self.num_heads))

    def forward(self, x: torch.Tensor, graph: Any, **kwargs: Any) -> torch.Tensor:
        qkv: torch.Tensor = self.qkv_proj(x)
        q, k, v = qkv.split(self.feature_dim, -1)
        # TODO add support for multiple heads

        return WSBGraphTransformer.apply(q, k, v, graph)


@BackendRegistry.register_backend("triton_block_sparse")
class TritonBlockSparseBackend(BaseBackend):
    """Backend that instantiates DGL-based convolutions."""

    def create_conv(
        self,
        conv_type: str,
        **kwargs: Any,
    ):
        """Factory for Triton convolution layers.

        Args:
            conv_type (str): Convolution type
            feature_dim (int): Input (and output) feature size.
            **kwargs (Any): Extra arguments for DGL layers.

        Returns:
            BaseConvolution: An instance of the requested DGL conv.
        """
        feature_dim = kwargs.pop("feature_dim")

        ct = conv_type.lower()
        match ct:
            case "gcn":
                return _TritonBlockSparseGraphConv(feature_dim=feature_dim, norm="both")
            case "mean_aggr":
                return _TritonBlockSparseGraphConv(feature_dim=feature_dim, norm="right")
            case "sum_aggr":
                return _TritonBlockSparseGraphConv(feature_dim=feature_dim, norm="none")
            case "gt":
                heads = kwargs.pop("heads")
                return _TritonBlockSparseGraphTransformerConv(feature_dim=feature_dim, heads=heads)
        raise KeyError(f"Unsupported conv_type for DGL backend: {conv_type}")
