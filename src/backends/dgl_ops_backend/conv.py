from typing import Any

import dgl.nn.functional as dgl_F
import torch
import torch.nn as nn
from dgl import ops

from ..base import BaseAggr, BaseBackend, BaseConvolution
from ..registry import BackendRegistry

_REDUCE_OPS = {
    "min": ops.copy_u_min,
    "max": ops.copy_u_max,
    "sum": ops.copy_u_sum,
    "mean": ops.copy_u_mean,
}


def _broadcast_edge_weights(w: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return w.view(-1, *([1] * (x.dim() - 1)))


class OpsReduceAggr(BaseAggr):
    def __init__(self, reduce: str, **kwargs: Any) -> None:
        super().__init__(conv_type=f"{reduce}_aggr")
        self._op = _REDUCE_OPS[reduce]
        self._clamp_inf = reduce in ("min", "max")

    def forward(self, x: torch.Tensor, graph, **kwargs: Any) -> torch.Tensor:
        out = self._op(graph, x)
        if self._clamp_inf:
            out = torch.where(out.isinf(), torch.zeros_like(out), out)
        return out


class OpsSpMMAggr(BaseAggr):
    def __init__(self, norm: str = "both", precomputed_norm: bool = True, **kwargs: Any) -> None:
        super().__init__(conv_type=f"spmm_{norm}")
        self.norm = norm
        self.precomputed_norm = precomputed_norm
        self._cache: tuple[int, torch.Tensor] | None = None

    def _edge_norm(self, graph) -> torch.Tensor:
        src, dst = graph.edges()
        deg_dst = graph.in_degrees().clamp(min=1).float()
        if self.norm == "right":
            return deg_dst[dst].reciprocal()
        deg_src = graph.out_degrees().clamp(min=1).float()
        return (deg_src[src] * deg_dst[dst]).rsqrt()

    def _weights(self, graph) -> torch.Tensor:
        if not self.precomputed_norm:
            return self._edge_norm(graph)
        if self._cache is None or self._cache[0] != id(graph):
            self._cache = (id(graph), self._edge_norm(graph))
        return self._cache[1]

    def reset_cache(self) -> None:
        self._cache = None

    def forward(self, x: torch.Tensor, graph, **kwargs: Any) -> torch.Tensor:
        w = self._weights(graph).to(x.dtype)
        return ops.u_mul_e_sum(graph, x, _broadcast_edge_weights(w, x))


class OpsGATv2Aggr(BaseAggr):
    def __init__(self, heads: int, head_dim: int, negative_slope: float = 0.2, **kwargs: Any) -> None:
        super().__init__(conv_type="gat_v2")
        self.heads = heads
        self.head_dim = head_dim
        self.negative_slope = negative_slope
        self.attn_weights = nn.Parameter(torch.empty(heads, head_dim))
        nn.init.xavier_normal_(self.attn_weights, gain=nn.init.calculate_gain("relu"))

    def forward(self, x_left: torch.Tensor, x_right: torch.Tensor, graph, **kwargs: Any) -> torch.Tensor:
        e = ops.u_add_v(graph, x_right, x_left)
        e = torch.nn.functional.leaky_relu(e, self.negative_slope)
        e = (e * self.attn_weights.detach()).sum(-1, keepdim=True)
        alpha = dgl_F.edge_softmax(graph, e)
        return ops.u_mul_e_sum(graph, x_right, alpha).flatten(1)


class OpsGTAggr(BaseAggr):
    def __init__(self, heads: int, head_dim: int, **kwargs: Any) -> None:
        super().__init__(conv_type="gt")
        self.heads = heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, graph, **kwargs: Any) -> torch.Tensor:
        attn_scores = ops.u_dot_v(graph, Q, K) * self.scale
        attn_probs = dgl_F.edge_softmax(graph, attn_scores)
        return ops.u_mul_e_sum(graph, V, attn_probs).flatten(1)


@BackendRegistry.register_backend("dgl_ops")
class DglOpsBackend(BaseBackend):
    def create_conv(self, conv_type: str, **kwargs: Any) -> BaseConvolution:
        raise KeyError(
            f"dgl_ops backend is aggregation-only (use --mode aggr); no layer for conv_type={conv_type}. "
            "Use the 'dgl' backend for full dgl.nn layers."
        )

    def create_aggr(self, conv_type: str, **kwargs: Any) -> BaseAggr:
        feature_dim = kwargs.pop("feature_dim", None)
        precomputed_norm = kwargs.pop("precomputed_norm", True)
        ct = conv_type.lower()
        match ct:
            case "min_aggr" | "max_aggr" | "sum_aggr" | "mean_aggr":
                return OpsReduceAggr(reduce=ct.removesuffix("_aggr"), **kwargs)
            case "gcn":
                return OpsSpMMAggr(norm="both", precomputed_norm=precomputed_norm, **kwargs)
            case "gat_v2":
                heads = kwargs.pop("heads", 1)
                return OpsGATv2Aggr(
                    heads=heads, head_dim=feature_dim, negative_slope=kwargs.pop("negative_slope", 0.2), **kwargs
                )
            case "gt":
                heads = kwargs.pop("heads", 8)
                return OpsGTAggr(heads=heads, head_dim=feature_dim // heads, **kwargs)
            case _:
                raise KeyError(f"Unsupported conv_type for dgl_ops aggr: {conv_type}")
