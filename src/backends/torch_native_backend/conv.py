from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F_torch

from ..base import BaseAggr, BaseBackend, BaseConvolution
from ..registry import BackendRegistry

doc = """
Torch-native backend: reference implementations using PyTorch sparse/dense ops.

Includes both the legacy sparse-matmul backends (torch_native_gcn, etc.) and
the unified scatter-based backend ("torch_native") that uses COO edge lists
and torch.scatter_add_ / torch.scatter_reduce_ for all conv types.
"""


# ---------------------------------------------------------------------------
# Scatter helpers (pure torch, no DGL / torch_scatter)
# ---------------------------------------------------------------------------


def _scatter_add(src_values: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Scatter-add src_values into output of shape [dim_size, *src_values.shape[1:]]."""
    out = torch.zeros(dim_size, *src_values.shape[1:], device=src_values.device, dtype=src_values.dtype)
    idx = index.unsqueeze(1).expand_as(src_values) if src_values.ndim == 2 else index
    out.scatter_add_(0, idx, src_values)
    return out


def _edge_softmax(scores: torch.Tensor, dst: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Numerically stable softmax over incoming edges per destination node.

    Args:
        scores: [E] or [E, H] edge scores.
        dst: [E] destination node indices.
        num_nodes: total number of nodes.

    Returns:
        Normalized attention weights with same shape as *scores*.
    """
    if scores.ndim == 1:
        # max per destination for stability
        max_vals = torch.full((num_nodes,), float("-inf"), device=scores.device, dtype=scores.dtype)
        max_vals.scatter_reduce_(0, dst, scores, reduce="amax", include_self=True)
        scores_stable = scores - max_vals[dst]

        exp_scores = torch.exp(scores_stable)

        sum_exp = torch.zeros(num_nodes, device=scores.device, dtype=scores.dtype)
        sum_exp.scatter_add_(0, dst, exp_scores)

        return exp_scores / sum_exp[dst].clamp(min=1e-16)
    else:
        # [E, H] case
        H = scores.shape[1]
        max_vals = torch.full((num_nodes, H), float("-inf"), device=scores.device, dtype=scores.dtype)
        dst_exp = dst.unsqueeze(1).expand(-1, H)
        max_vals.scatter_reduce_(0, dst_exp, scores, reduce="amax", include_self=True)
        scores_stable = scores - max_vals[dst]

        exp_scores = torch.exp(scores_stable)

        sum_exp = torch.zeros(num_nodes, H, device=scores.device, dtype=scores.dtype)
        sum_exp.scatter_add_(0, dst_exp, exp_scores)

        return exp_scores / sum_exp[dst].clamp(min=1e-16)


class _TorchNativeMatMulConv(BaseConvolution):
    """Reference GraphConv using modified adjacency and sparse matmul."""

    def __init__(self, bias: bool = False, **kwargs: Any) -> None:
        """Initialize a Torch-native GraphConv.

        Args:
            bias (bool): Include bias in linear transform.
            **kwargs (Any): Reserved for future options.
        """
        super().__init__(bias=bias, **kwargs)

    def forward(
        self,
        x: torch.Tensor,
        graph: Any,
        *,
        edge_weight: torch.Tensor | None = None,  # ignored for baseline
        **kwargs: Any,
    ) -> torch.Tensor:
        """Apply GraphConv: X' = A_hat @ (X W).

        Args:
            x (torch.Tensor): Node features [N, Fin].
            graph (Any): Either (edge_index, num_nodes) or (edge_index, edge_weight) or (edge_index, ew, num_nodes).
            edge_weight (Optional[torch.Tensor]): Unused baseline.
            **kwargs (Any): Extra kwargs ignored.

        Returns:
            torch.Tensor: Output features [N, Fout].
        """
        modified_adgacency = graph
        return torch.sparse.mm(modified_adgacency, x)


class TorchNativeMatMulBackend(BaseBackend):
    """Backend instantiating simple Torch-native MatMul GNN convs."""

    CONV_TYPE: Optional[str] = None

    def create_conv(
        self,
        conv_type: str,
        **kwargs: Any,
    ):
        """Factory for Torch-native matmul convs.

        Args:
            conv_type (str): {CONV_TYPE} supported (extend as needed).
            **kwargs (Any): Extra kwargs.

        Returns:
            BaseConvolution: Torch-native convolution layer.
        """
        # guard unsupported backends
        if conv_type == self.CONV_TYPE:
            return _TorchNativeMatMulConv(**kwargs)
        raise NotImplementedError(f"Convolution `{conv_type}` is not implemented for backend {self.__class__.__name__}")


class _TorchNativeMinConv(BaseConvolution):
    """Min aggregation of incoming neighbors."""

    def __init__(self, bias: bool = True, **kwargs: Any) -> None:
        """Initialize a Torch-native min aggregation convolution.

        Args:
            bias (bool): Include bias in linear transform.
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
        """Apply min aggregation convolution

        Args:
            x (torch.Tensor): Node features [N, Fin].
            graph (Any):
                - adj_mat: sparse COO tensor [N, N] (A^T)
            edge_weight (Optional[torch.Tensor]): Unused for this baseline.
            **kwargs (Any): Extra kwargs ignored.

        Returns:
            torch.Tensor: Output features [N, Fout].
        """
        src, dst = graph.indices()
        # we don't care what the values inside are
        # we can avoid normalizing / re-normalizing prior to this layer for speedup
        messages = x[src]
        num_nodes, feature_dim = x.size()
        out = torch.full((num_nodes, feature_dim), float("inf"), device=x.device)
        index = dst.unsqueeze(1).expand(-1, feature_dim)
        out.scatter_reduce_(0, index, messages, reduce="amin", include_self=False)
        out[out == float("inf")] = 0.0
        return out


class _TorchNativeMaxConv(BaseConvolution):
    """Max aggregation of incoming neighbors."""

    def __init__(self, bias: bool = True, **kwargs: Any) -> None:
        """Initialize a Torch-native max aggregation convolution.

        Args:
            bias (bool): Include bias in linear transform.
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
        """Apply max aggregation convolution

        Args:
            x (torch.Tensor): Node features [N, Fin].
            graph (Any):
                - adj_mat: sparse COO tensor [N, N] (A^T)
            edge_weight (Optional[torch.Tensor]): Unused for this baseline.
            **kwargs (Any): Extra kwargs ignored.

        Returns:
            torch.Tensor: Output features [N, Fout].
        """
        src, dst = graph.indices()
        # we don't care what the values inside are
        # we can avoid normalizing / re-normalizing prior to this layer for speedup
        messages = x[src]
        num_nodes, feat_dim = x.size()
        out = torch.full((num_nodes, feat_dim), float("-inf"), device=x.device)
        index = dst.unsqueeze(1).expand(-1, feat_dim)
        out.scatter_reduce_(0, index, messages, reduce="amax", include_self=False)
        out[out.isinf()] = 0.0
        return out


@BackendRegistry.register_backend("torch_native_adj_mat")
class TorchNativeAdjMatBackend(BaseBackend):
    """Factory for Torch-native pooling GNN convs."""

    def create_conv(
        self,
        conv_type: str,
        **kwargs: Any,
    ):
        """Factory for Torch-native pooling convs.

        Args:
            conv_type (str): 'gcn' supported (extend as needed).
            **kwargs (Any): Extra kwargs.

        Returns:
            BaseConvolution: Torch-native convolution layer.
        """
        # guard unsupported backends
        if conv_type == "min_aggr":
            return _TorchNativeMinConv(**kwargs)
        if conv_type == "max_aggr":
            return _TorchNativeMaxConv(**kwargs)
        raise NotImplementedError(f"Convolution `{conv_type}` is not implemented for backend {self.__class__.__name__}")


@BackendRegistry.register_backend("torch_native_gcn")
class TorchNativeGCNBackend(TorchNativeMatMulBackend):
    """Backend instantiating simple Torch-native GCN convs."""

    CONV_TYPE = "gcn"


@BackendRegistry.register_backend("torch_native_mean_aggr")
class TorchNativeMeanAggrBackend(TorchNativeMatMulBackend):
    """Backend instantiating simple Torch-native mean aggregation convs."""

    CONV_TYPE = "mean_aggr"


@BackendRegistry.register_backend("torch_native_sum_aggr")
class TorchNativeSumAggrBackend(TorchNativeMatMulBackend):
    """Backend instantiating simple Torch-native sum aggregation convs."""

    CONV_TYPE = "sum_aggr"


# ---------------------------------------------------------------------------
# Unified scatter-based convolutions (COO / edge-list graph format)
# Graph = (edge_index [2, E], edge_weight [E] | None, num_nodes)
# ---------------------------------------------------------------------------


class _ScatterSumAggr(BaseConvolution):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(bias=False, **kwargs)

    def forward(self, x: torch.Tensor, graph: Any, **kwargs: Any) -> torch.Tensor:
        edge_index, edge_weight, num_nodes = graph
        src, dst = edge_index
        messages = x[src]
        if edge_weight is not None:
            messages = messages * edge_weight.unsqueeze(1)
        return _scatter_add(messages, dst, num_nodes)


class _ScatterMeanAggr(BaseConvolution):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(bias=False, **kwargs)

    def forward(self, x: torch.Tensor, graph: Any, **kwargs: Any) -> torch.Tensor:
        edge_index, edge_weight, num_nodes = graph
        src, dst = edge_index
        messages = x[src]
        if edge_weight is not None:
            messages = messages * edge_weight.unsqueeze(1)
        out_sum = _scatter_add(messages, dst, num_nodes)
        counts = torch.zeros(num_nodes, device=x.device, dtype=x.dtype)
        counts.scatter_add_(0, dst, torch.ones(dst.size(0), device=x.device, dtype=x.dtype))
        return out_sum / counts.unsqueeze(1).clamp(min=1)


class _ScatterGCN(BaseConvolution):
    """GCN with symmetric normalization: D_out^{-1/2} @ src, D_in^{-1/2} @ dst.

    Matches DGL GraphConv(norm="both") behavior which uses out-degree for source
    normalization and in-degree for destination normalization.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(bias=False, **kwargs)

    def forward(self, x: torch.Tensor, graph: Any, **kwargs: Any) -> torch.Tensor:
        edge_index, _edge_weight, num_nodes = graph
        src, dst = edge_index
        ones = torch.ones(src.size(0), device=x.device, dtype=x.dtype)
        # out-degree (from src) and in-degree (from dst)
        out_deg = torch.zeros(num_nodes, device=x.device, dtype=x.dtype)
        out_deg.scatter_add_(0, src, ones)
        in_deg = torch.zeros(num_nodes, device=x.device, dtype=x.dtype)
        in_deg.scatter_add_(0, dst, ones)
        out_deg_inv_sqrt = out_deg.pow(-0.5)
        out_deg_inv_sqrt[out_deg_inv_sqrt.isinf()] = 0
        in_deg_inv_sqrt = in_deg.pow(-0.5)
        in_deg_inv_sqrt[in_deg_inv_sqrt.isinf()] = 0
        # per-edge normalization: D_out^{-1/2}[src] * D_in^{-1/2}[dst]
        norm = out_deg_inv_sqrt[src] * in_deg_inv_sqrt[dst]
        messages = x[src] * norm.unsqueeze(1)
        return _scatter_add(messages, dst, num_nodes)


class _ScatterMinAggr(BaseConvolution):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(bias=False, **kwargs)

    def forward(self, x: torch.Tensor, graph: Any, **kwargs: Any) -> torch.Tensor:
        edge_index, _edge_weight, num_nodes = graph
        src, dst = edge_index
        messages = x[src]
        F = x.size(1)
        out = torch.full((num_nodes, F), float("inf"), device=x.device, dtype=x.dtype)
        idx = dst.unsqueeze(1).expand(-1, F)
        out.scatter_reduce_(0, idx, messages, reduce="amin", include_self=False)
        out[out.isinf()] = 0.0
        return out


class _ScatterMaxAggr(BaseConvolution):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(bias=False, **kwargs)

    def forward(self, x: torch.Tensor, graph: Any, **kwargs: Any) -> torch.Tensor:
        edge_index, _edge_weight, num_nodes = graph
        src, dst = edge_index
        messages = x[src]
        F = x.size(1)
        out = torch.full((num_nodes, F), float("-inf"), device=x.device, dtype=x.dtype)
        idx = dst.unsqueeze(1).expand(-1, F)
        out.scatter_reduce_(0, idx, messages, reduce="amax", include_self=False)
        out[out.isinf()] = 0.0
        return out


class _ScatterGATv2Conv(BaseConvolution):
    """Pure-torch GATv2 using scatter-based edge softmax.

    Parameter names match DGL GATv2Conv for weight-sharing compatibility:
        fc_dst, fc_src, attn, _outer_proj

    Convention matches DGL/CUDA: each head operates on `feature_dim` features,
    so fc_dst/fc_src project to `feature_dim * heads`, and _outer_proj maps
    `feature_dim * heads -> feature_dim`.
    """

    def __init__(
        self,
        feature_dim: int,
        heads: int = 1,
        bias: bool = False,
        negative_slope: float = 0.2,
        **kwargs: Any,
    ) -> None:
        super().__init__(bias=bias, **kwargs)
        self.heads = heads
        self.head_dim = feature_dim  # each head uses full feature_dim (DGL/CUDA convention)
        self.negative_slope = negative_slope

        hd = heads * self.head_dim
        self.fc_dst = nn.Linear(feature_dim, hd, bias=bias)
        self.fc_src = nn.Linear(feature_dim, hd, bias=bias)
        self.attn = nn.Parameter(torch.empty(1, heads, self.head_dim))
        nn.init.xavier_normal_(self.attn)
        self._outer_proj = nn.Linear(hd, feature_dim, bias=bias)

    def forward(self, x: torch.Tensor, graph: Any, **kwargs: Any) -> torch.Tensor:
        edge_index, _ew, num_nodes = graph
        src, dst = edge_index
        H, D = self.heads, self.head_dim

        x_dst = self.fc_dst(x).view(-1, H, D)  # [N, H, D]
        x_src = self.fc_src(x).view(-1, H, D)  # [N, H, D]

        # edge-level attention
        e = F_torch.leaky_relu(x_dst[dst] + x_src[src], negative_slope=self.negative_slope)  # [E, H, D]
        attn_scores = (e * self.attn).sum(-1)  # [E, H]

        # edge softmax per destination
        attn_probs = _edge_softmax(attn_scores, dst, num_nodes)  # [E, H]

        # weighted aggregation of source features
        messages = x_src[src] * attn_probs.unsqueeze(-1)  # [E, H, D]
        out = torch.zeros(num_nodes, H, D, device=x.device, dtype=x.dtype)
        dst_exp = dst.unsqueeze(1).unsqueeze(2).expand_as(messages)
        out.scatter_add_(0, dst_exp, messages)  # [N, H, D]

        out = out.reshape(num_nodes, -1)
        return self._outer_proj(out)


class _ScatterGraphTransformer(BaseConvolution):
    """Pure-torch Graph Transformer using scatter-based edge softmax.

    Parameter names match DGL GT for weight-sharing compatibility: qkv_proj
    """

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

        self.qkv_proj = nn.Linear(feature_dim, 3 * feature_dim)
        self.attn_scores_multiplier = torch.rsqrt(torch.tensor(feature_dim // heads))

    def forward(self, x: torch.Tensor, graph: Any, **kwargs: Any) -> torch.Tensor:
        edge_index, _ew, num_nodes = graph
        src, dst = edge_index
        H = self.num_heads
        D = self.feature_dim // H

        x = F_torch.layer_norm(x, (x.shape[-1],))
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(self.feature_dim, -1)

        q = q.view(num_nodes, H, D)
        k = k.view(num_nodes, H, D)
        v = v.view(num_nodes, H, D)

        # per-edge attention scores: dot(q[src], k[dst]) per head
        attn_scores = (q[src] * k[dst]).sum(-1) * self.attn_scores_multiplier  # [E, H]

        # edge softmax over incoming edges per destination
        attn_probs = _edge_softmax(attn_scores, dst, num_nodes)  # [E, H]

        # weighted aggregation of source values
        messages = v[src] * attn_probs.unsqueeze(-1)  # [E, H, D]
        out = torch.zeros(num_nodes, H, D, device=x.device, dtype=x.dtype)
        dst_exp = dst.unsqueeze(1).unsqueeze(2).expand_as(messages)
        out.scatter_add_(0, dst_exp, messages)

        return out.reshape(num_nodes, -1)


# ---------------------------------------------------------------------------
# Unified torch_native backend (all conv types, COO graph format)
# ---------------------------------------------------------------------------


class _ScatterSimpleAggOnly(BaseAggr):
    """Aggregation-only scatter min/max/sum/mean."""

    def __init__(self, reduce: str, **kwargs: Any) -> None:
        super().__init__(conv_type=f"{reduce}_aggr")
        self.reduce = reduce

    def forward(self, x: torch.Tensor, graph, **kwargs: Any) -> torch.Tensor:
        edge_index, _ew, num_nodes = graph
        src, dst = edge_index
        messages = x[src]
        F_dim = x.size(1)
        idx = dst.unsqueeze(1).expand(-1, F_dim)

        if self.reduce == "min":
            out = torch.full((num_nodes, F_dim), float("inf"), device=x.device, dtype=x.dtype)
            out.scatter_reduce_(0, idx, messages, reduce="amin", include_self=False)
            out[out.isinf()] = 0.0
        elif self.reduce == "max":
            out = torch.full((num_nodes, F_dim), float("-inf"), device=x.device, dtype=x.dtype)
            out.scatter_reduce_(0, idx, messages, reduce="amax", include_self=False)
            out[out.isinf()] = 0.0
        elif self.reduce == "sum":
            out = _scatter_add(messages, dst, num_nodes)
        elif self.reduce == "mean":
            out = _scatter_add(messages, dst, num_nodes)
            counts = torch.zeros(num_nodes, 1, device=x.device, dtype=x.dtype)
            counts.scatter_add_(0, dst.unsqueeze(1), torch.ones(dst.size(0), 1, device=x.device, dtype=x.dtype))
            out = out / counts.clamp(min=1)
        else:
            raise ValueError(f"Unknown reduce: {self.reduce}")
        return out


class _ScatterGATv2AggOnly(BaseAggr):
    """Aggregation-only scatter GATv2: takes pre-projected x_left, x_right."""

    def __init__(self, heads: int, head_dim: int, negative_slope: float = 0.2, **kwargs: Any) -> None:
        super().__init__(conv_type="gat_v2")
        self.heads = heads
        self.head_dim = head_dim
        self.negative_slope = negative_slope
        self.attn = nn.Parameter(torch.empty(1, heads, head_dim))
        nn.init.xavier_normal_(self.attn)

    def forward(self, x_left: torch.Tensor, x_right: torch.Tensor, graph, **kwargs: Any) -> torch.Tensor:
        edge_index, _ew, num_nodes = graph
        src, dst = edge_index

        e = F_torch.leaky_relu(x_left[dst] + x_right[src], negative_slope=self.negative_slope)
        attn_scores = (e * self.attn).sum(-1)
        attn_probs = _edge_softmax(attn_scores, dst, num_nodes)

        messages = x_right[src] * attn_probs.unsqueeze(-1)
        out = torch.zeros(num_nodes, self.heads, self.head_dim, device=x_left.device, dtype=x_left.dtype)
        dst_exp = dst.unsqueeze(1).unsqueeze(2).expand_as(messages)
        out.scatter_add_(0, dst_exp, messages)
        return out.reshape(num_nodes, -1)


class _ScatterGTAggOnly(BaseAggr):
    """Aggregation-only scatter Graph Transformer: takes pre-projected Q, K, V."""

    def __init__(self, heads: int, head_dim: int, **kwargs: Any) -> None:
        super().__init__(conv_type="gt")
        self.heads = heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, graph, **kwargs: Any) -> torch.Tensor:
        edge_index, _ew, num_nodes = graph
        src, dst = edge_index

        attn_scores = (Q[src] * K[dst]).sum(-1) * self.scale
        attn_probs = _edge_softmax(attn_scores, dst, num_nodes)

        messages = V[src] * attn_probs.unsqueeze(-1)
        out = torch.zeros(num_nodes, *V.shape[1:], device=V.device, dtype=V.dtype)
        dst_exp = dst.view(-1, *([1] * (V.ndim - 1))).expand_as(messages)
        out.scatter_add_(0, dst_exp, messages)
        return out.reshape(num_nodes, -1)


@BackendRegistry.register_backend("torch_native")
class TorchNativeBackend(BaseBackend):
    """Unified pure-torch backend using scatter ops on COO edge lists.

    Graph format: (edge_index [2, E], edge_weight [E] | None, num_nodes).
    Supports all conv types as reference implementations for correctness tests.
    """

    def create_conv(self, conv_type: str, **kwargs: Any) -> BaseConvolution:
        feature_dim = kwargs.pop("feature_dim", None)
        ct = conv_type.lower()
        match ct:
            case "sum_aggr":
                kwargs.pop("bias", None)
                return _ScatterSumAggr(**kwargs)
            case "mean_aggr":
                kwargs.pop("bias", None)
                return _ScatterMeanAggr(**kwargs)
            case "gcn":
                kwargs.pop("bias", None)
                return _ScatterGCN(**kwargs)
            case "min_aggr":
                kwargs.pop("bias", None)
                return _ScatterMinAggr(**kwargs)
            case "max_aggr":
                kwargs.pop("bias", None)
                return _ScatterMaxAggr(**kwargs)
            case "gat_v2":
                heads = kwargs.pop("heads", 1)
                return _ScatterGATv2Conv(feature_dim=feature_dim, heads=heads, **kwargs)
            case "gt":
                heads = kwargs.pop("heads", 8)
                return _ScatterGraphTransformer(feature_dim=feature_dim, heads=heads, **kwargs)
        raise KeyError(f"Unsupported conv_type for torch_native backend: {conv_type}")

    def create_aggr(self, conv_type: str, **kwargs: Any) -> BaseAggr:
        feature_dim = kwargs.pop("feature_dim", None)
        ct = conv_type.lower()
        match ct:
            case "min_aggr":
                return _ScatterSimpleAggOnly(reduce="min")
            case "max_aggr":
                return _ScatterSimpleAggOnly(reduce="max")
            case "sum_aggr":
                return _ScatterSimpleAggOnly(reduce="sum")
            case "mean_aggr":
                return _ScatterSimpleAggOnly(reduce="mean")
            case "gat_v2":
                heads = kwargs.pop("heads", 1)
                return _ScatterGATv2AggOnly(heads=heads, head_dim=feature_dim)
            case "gt":
                heads = kwargs.pop("heads", 8)
                head_dim = feature_dim // heads
                return _ScatterGTAggOnly(heads=heads, head_dim=head_dim)
            case _:
                raise KeyError(f"Unsupported conv_type for torch_native aggr: {conv_type}")
