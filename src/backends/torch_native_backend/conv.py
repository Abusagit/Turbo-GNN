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

    def __init__(self, in_channels: int, out_channels: int, bias: bool = False, **kwargs: Any) -> None:
        """Initialize a Torch-native GraphConv.

        Args:
            in_channels (int): Input feature size.
            out_channels (int): Output feature size.
            bias (bool): Include bias in linear transform.
            **kwargs (Any): Reserved for future options.
        """
        super().__init__(in_channels, out_channels, bias=bias, **kwargs)
        # self.lin = nn.Linear(in_channels, out_channels, bias=bias)

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
        in_channels: int,
        out_channels: int,
        **kwargs: Any,
    ):
        """Factory for Torch-native matmul convs.

        Args:
            conv_type (str): {CONV_TYPE} supported (extend as needed).
            in_channels (int): Input feature size.
            out_channels (int): Output feature size.
            **kwargs (Any): Extra kwargs.

        Returns:
            BaseConvolution: Torch-native convolution layer.
        """
        # guard unsupported backends
        if conv_type == self.CONV_TYPE:
            return _TorchNativeMatMulConv(in_channels, out_channels, **kwargs)
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
