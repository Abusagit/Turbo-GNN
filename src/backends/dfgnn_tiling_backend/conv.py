import math
from typing import Any

import torch
import torch.nn as nn

from ..base import BaseBackend, BaseConvolution
from ..registry import BackendRegistry
from .bindings import dfgnn_tiling_ops


class GTConfFuseTilingFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, indptr, indices, val, Q, K, V):
        smem_consume = (128 + 32 - 1) // 32 * 32  # noqa: F821
        out_feat = dfgnn_tiling_ops.gt_tiling_inference(
            indptr,
            indices,
            val,
            smem_consume,
            Q,
            K,
            V,
        )

        return out_feat[0]

    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError("Backward pass is not implemented for GTConfFuseTilingFunction")


class _DFGNN_GTConv(BaseConvolution):
    def __init__(self, feature_dim: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.feature_dim = feature_dim
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)
        self.scale = 1 / math.sqrt(feature_dim)

    def forward(self, x: torch.Tensor, graph: Any, **kwargs):
        (
            _,
            row_ptr,
            col_ind,
            val,
            _,
            _,
            _,
            _,
        ) = graph
        x = torch.nn.functional.layer_norm(x, (x.shape[-1],))
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(x.shape[0], self.num_heads, -1)
        k = k.view(x.shape[0], self.num_heads, -1)
        v = v.view(x.shape[0], self.num_heads, -1)

        output = GTConfFuseTilingFunction.apply(
            row_ptr,
            col_ind,
            val,
            q,
            k,
            v,
        )
        return output.view(x.shape[0], -1)


@BackendRegistry.register_backend("dfgnn_tiling")
class DFGNNTilingBackend(BaseBackend):
    def create_conv(self, conv_type: str, **kwargs: Any) -> BaseConvolution:
        """Factory for DFGNN Tiling convolution layers.

        Args:
            conv_type (str): "gt"
            **kwargs (Any): ignored.
        Returns:
            BaseConvolution: An instance of the requested DFGNN Tiling conv.
        """

        if conv_type == "gt":
            return _DFGNN_GTConv(kwargs["feature_dim"], num_heads=kwargs["heads"])
        raise ValueError(f"Unsupported conv_type for DFGNN Tiling backend: {conv_type}")
