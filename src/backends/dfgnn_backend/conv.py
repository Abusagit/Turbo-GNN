import math
from typing import Any

import torch
import torch.nn as nn

from ..base import BaseAggr, BaseBackend, BaseConvolution
from ..registry import BackendRegistry
from .bindings import dfgnn_ops


class GTConvFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        rows,
        row_ptr,
        col_ind,
        val,
        col_ptr,
        row_ind,
        val_idx,
        smem_consume,
        Q,
        K,
        V,
    ):
        out_feat, attn_edge = dfgnn_ops.gt_hyper_forward(
            row_ptr,
            col_ind,
            rows,
            val,
            col_ptr,
            row_ind,
            val_idx,
            smem_consume,
            Q,
            K,
            V,
        )
        ctx.smem = smem_consume
        ctx.save_for_backward(row_ptr, col_ind, rows, val, col_ptr, row_ind, val_idx, Q, K, V, attn_edge)
        return out_feat

    @staticmethod
    def backward(ctx, grad_out):
        (
            row_ptr,
            col_ind,
            rows,
            val,
            col_ptr,
            row_ind,
            val_idx,
            Q,
            K,
            V,
            attn_edge,
        ) = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        grad_Q, grad_K, grad_V = dfgnn_ops.gt_backward(
            row_ptr,
            col_ind,
            rows,
            val,
            col_ptr,
            row_ind,
            val_idx,
            ctx.smem,
            Q,
            K,
            V,
            attn_edge,
            grad_out,
        )

        return (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            grad_Q,
            grad_K,
            grad_V,
        )


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
            rows,
            row_ptr,
            col_ind,
            val,
            col_ptr,
            row_ind,
            val_idx,
            smem_consume,
        ) = graph
        x = torch.nn.functional.layer_norm(x, (x.shape[-1],))
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(x.shape[0], self.num_heads, -1)
        k = k.view(x.shape[0], self.num_heads, -1)
        v = v.view(x.shape[0], self.num_heads, -1)

        output = GTConvFunction.apply(
            rows,
            row_ptr,
            col_ind,
            val,
            col_ptr,
            row_ind,
            val_idx,
            smem_consume,
            q,
            k,
            v,
        ).view(x.shape[0], -1)

        return output


class _DFGNN_GTAggr(BaseAggr):
    """Aggregation-only DFGNN GT (no QKV projection)."""

    def __init__(self, feature_dim: int, num_heads: int = 8) -> None:
        super().__init__(conv_type="gt")
        self.num_heads = num_heads
        self.feature_dim = feature_dim

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, graph, **kwargs) -> torch.Tensor:
        rows, row_ptr, col_ind, val, col_ptr, row_ind, val_idx, smem_consume = graph
        output = GTConvFunction.apply(
            rows,
            row_ptr,
            col_ind,
            val,
            col_ptr,
            row_ind,
            val_idx,
            smem_consume,
            Q,
            K,
            V,
        )
        return output.view(Q.shape[0], -1)


@BackendRegistry.register_backend("dfgnn")
class DFGNNBackend(BaseBackend):
    def create_conv(self, conv_type: str, **kwargs: Any) -> BaseConvolution:
        """Factory for DFGNN convolution layers.

        Args:
            conv_type (str): "gt"
            **kwargs (Any): ignored.
        Returns:
            BaseConvolution: An instance of the requested DFGNN conv.
        """

        if conv_type == "gt":
            return _DFGNN_GTConv(kwargs["feature_dim"], num_heads=kwargs["heads"])
        raise ValueError(f"Unsupported conv_type for DFGNN backend: {conv_type}")

    def create_aggr(self, conv_type: str, **kwargs: Any) -> BaseAggr:
        if conv_type == "gt":
            heads = kwargs.get("heads", 8)
            feature_dim = kwargs["feature_dim"]
            return _DFGNN_GTAggr(feature_dim=feature_dim, num_heads=heads)
        raise KeyError(f"Unsupported conv_type for DFGNN aggr: {conv_type}")
