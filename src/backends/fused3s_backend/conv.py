from typing import Any

import torch
import torch.nn as nn

from ..base import BaseBackend, BaseConvolution
from ..registry import BackendRegistry
from .bindings import F3S_forward


class _F3SATConv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, graph, size, block_h=16, block_w=8):
        time, output, sddm_result = F3S_forward(q, k, v, graph, size, block_h, block_w)
        return output

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
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)
        self.block_h = block_h
        self.block_w = block_w

    def forward(self, x, graph, **kwargs):
        q, k, v = self.q_proj(x).to(torch.half), self.k_proj(x).to(torch.half), self.v_proj(x).to(torch.half)
        return _F3SATConv.apply(q, k, v, graph, self.block_h, self.block_w)


@BackendRegistry.register_backend("f3s")
class F3SBackend(BaseBackend):
    """Fused 3S backend implementation"""

    def create_conv(self, conv_type: str, **kwargs: Any) -> BaseConvolution:
        """Factory for fused3s convolution layers.

        Args:
            conv_type (str): "gt"
            **kwargs (Any): ignored.
        Returns:
            BaseConvolution: An instance of the requested Fused3S conv.
        """

        feature_dim = kwargs.get("feature_dim")

        conv_type = conv_type.lower()

        if conv_type == "gt":
            return F3SATConv(feature_dim)
        raise ValueError("Unkown conv type")
