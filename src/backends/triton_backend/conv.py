from typing import Any, Optional

import torch
import torch.nn as nn

from ..base import BaseAggr, BaseBackend, BaseConvolution, ConvAsAggr
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
        self.head_dim = self.feature_dim // self.num_heads
        self.qkv_proj = nn.Linear(self.feature_dim, 3 * self.feature_dim)

        self.attn_scores_multiplier = torch.rsqrt(torch.tensor(self.head_dim)).item()

    def forward(self, x: torch.Tensor, graph: Any, **kwargs: Any) -> torch.Tensor:
        x = torch.nn.functional.layer_norm(x, (self.feature_dim,))

        qkv: torch.Tensor = self.qkv_proj(x)
        q, k, v = qkv.split(self.feature_dim, -1)

        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_heads, self.head_dim)
        v = v.view(-1, self.num_heads, self.head_dim)

        out = WSBGraphTransformer.apply(q, k, v, graph, self.attn_scores_multiplier)
        out = out.view(-1, self.feature_dim)
        return out


class _TritonGTAggr(BaseAggr):
    """Aggregation-only Triton GT (no QKV projection)."""

    def __init__(self, heads: int, head_dim: int, **kwargs: Any) -> None:
        super().__init__(conv_type="gt")
        self.scale = torch.rsqrt(torch.tensor(float(head_dim))).item()

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, graph, **kwargs: Any) -> torch.Tensor:
        out = WSBGraphTransformer.apply(Q, K, V, graph, self.scale)
        return out.view(Q.shape[0], -1)


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
        raise KeyError(f"Unsupported conv_type for Triton backend: {conv_type}")

    def create_aggr(self, conv_type: str, **kwargs: Any) -> BaseAggr:
        feature_dim = kwargs.pop("feature_dim", None)
        ct = conv_type.lower()
        match ct:
            case "gcn" | "mean_aggr" | "sum_aggr":
                # SpMM convs are already projection-free
                return ConvAsAggr(self.create_conv(ct, feature_dim=feature_dim, **kwargs))
            case "gt":
                heads = kwargs.pop("heads", 8)
                head_dim = feature_dim // heads
                return _TritonGTAggr(heads=heads, head_dim=head_dim)
            case _:
                raise KeyError(f"Unsupported conv_type for Triton aggr: {conv_type}")
