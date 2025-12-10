from typing import Any

import torch
from torch import nn

from ..base import BaseBackend, BaseConvolution
from ..registry import BackendRegistry
from .utils import graph_transformer_kernels

doc = """
CUDA backend: custom CUDA kernels for optimized graph convolution [MAIN CONTRIBUTION]
"""


class _FusedGraphAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, edge_ptr, edge_idx, mid_nodes, huge_nodes, Q, K, V):
        """
        Forward pass wrapper.
        Returns: (output, logsumexp)
        """
        out, logsumexp, attn_logits = graph_transformer_kernels.forward_buckets(
            edge_ptr, edge_idx, mid_nodes, huge_nodes, Q, K, V
        )

        # Save for backward
        ctx.save_for_backward(edge_ptr, edge_idx, mid_nodes, huge_nodes, Q, K, V, out, logsumexp, attn_logits)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass wrapper.
        Returns: (None, None, None, None, dQ, dK, dV)
        """
        edge_ptr, edge_idx, mid_nodes, huge_nodes, Q, K, V, out, logsumexp, attn_logits = ctx.saved_tensors

        dQ, dK, dV = graph_transformer_kernels.backward_buckets(
            edge_ptr, edge_idx, mid_nodes, huge_nodes, Q, K, V, out, grad_output, logsumexp, attn_logits
        )

        return None, None, None, None, dQ, dK, dV


class _СuSparseMatMulConv(BaseConvolution):
    """CUDA-based Fused graph transformer"""

    def __init__(
        self,
        feature_dim: int,
        heads: int = 8,
    ):
        super().__init__(bias=False, dropout=0.0)
        if heads > 1:
            raise NotImplementedError("Currently only single head attention is supported, work in progress")

        self.feature_dim = feature_dim
        self.num_heads = heads
        self.qkv_proj = nn.Linear(self.feature_dim, 3 * self.feature_dim)

    def forward(
        self,
        x: torch.Tensor,
        graph: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        **kwargs: Any,
    ) -> torch.Tensor:
        """Apply GraphConv.

        Args:
            x (torch.Tensor): Node features [N, Fin].
            graph (Any): graph representation for current backend and convolution
            **kwargs (Any): Extra kwargs (ignored).

        Returns:
            torch.Tensor: Output features [N, Fout].
        """
        x = torch.nn.functional.layer_norm(x, (x.shape[-1],))
        qkv: torch.Tensor = self.qkv_proj(x)
        q, k, v = qkv.split(self.feature_dim, -1)
        edge_ptr, edge_idx, mid_nodes, huge_nodes = graph

        out = _FusedGraphAttention.apply(edge_ptr, edge_idx, mid_nodes, huge_nodes, q, k, v)
        return out


@BackendRegistry.register_backend("cuda")
class CUDABackend(BaseBackend):
    """Backend that instantiates CUDA-based convolutions"""

    def create_conv(
        self,
        conv_type: str,
        **kwargs: Any,
    ) -> BaseConvolution:
        """Factory for CUDA convolution layers.

        Args:
            conv_type (str)

        Returns:
            BaseConvolution: An instance of the requested CUDA conv.
        """

        conv_type = conv_type.lower()
        feature_dim = kwargs.pop("feature_dim")

        if conv_type == "gt":
            heads = kwargs.pop("heads")
            return _СuSparseMatMulConv(feature_dim=feature_dim, heads=heads)
        raise KeyError(f"Unsupported conv_type for DGL backend: {conv_type}")
