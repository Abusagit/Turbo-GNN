from typing import Any, Optional

import torch
import torch.nn as nn

from ..base import BaseBackend, BaseConvolution
from ..registry import BackendRegistry

doc = """
Torch-native backend: reference implementations using PyTorch sparse/dense ops.
"""


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
