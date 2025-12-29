from typing import Any

import torch
import torch.nn as nn

from ..base import BaseBackend, BaseConvolution
from ..registry import BackendRegistry
from .bindings import F3S_forward


class _F3SATConv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, graph, size, block_h=16, block_w=8):
        return F3S_forward(q, k, v, graph, size, block_h, block_w)

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("Backward pass is not implemented for F3SATConv")


class F3SATConv(BaseConvolution):
    def __init__(self, feature_dim, block_h=16, block_w=8):
        super().__init__()
        in_channels = feature_dim
        out_channels = feature_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.q_proj = nn.Linear(in_channels, out_channels)
        self.k_proj = nn.Linear(in_channels, out_channels)
        self.v_proj = nn.Linear(in_channels, out_channels)
        self.block_h = block_h
        self.block_w = block_w

    def forward(self, x, graph, **kwargs):
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        return _F3SATConv.apply(q, k, v, graph, self.block_h, self.block_w)

    def backward(self, grad_output):
        raise NotImplementedError("Backward pass is not implemented for F3SATConv")


@BackendRegistry.register_backend("f3s")
class F3SBackend(BaseBackend):
    """Fused 3S backend implementation"""

    def create_conv(self, conv_type: str, **kwargs: Any) -> BaseConvolution:
        """Factory for cusparse convolution layers.

        Args:
            conv_type (str): "grap_transformer"
            cu_sparse_algorithm_id (int): algorithm for CuSparse to use: -1 (default), 0, 1, 2, 3.
            **kwargs (Any): ignored.
        Returns:
            BaseConvolution: An instance of the requested Fused3S conv.
        """

        feature_dim = kwargs.get("feature_dim")

        conv_type = conv_type.lower()

        if conv_type == "graph_transformer":
            return F3SATConv(feature_dim)
        raise ValueError("Unkown conv type")
