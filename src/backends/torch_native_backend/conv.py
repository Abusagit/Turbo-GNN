from typing import Any, Optional

import torch
import torch.nn as nn

from ..base import BaseBackend, BaseConvolution
from ..registry import BackendRegistry

doc = """
Torch-native backend: reference implementations using PyTorch sparse/dense ops.
"""


class _TorchNativeGCNConv(BaseConvolution):
    """Reference GCN using sym-normalized adjacency and sparse matmul."""
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True, **kwargs: Any) -> None:
        """Initialize a Torch-native GCN.

        Args:
            in_channels (int): Input feature size.
            out_channels (int): Output feature size.
            bias (bool): Include bias in linear transform.
            **kwargs (Any): Reserved for future options.
        """
        super().__init__(in_channels, out_channels, bias=bias, **kwargs)
        self.lin = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(
        self,
        x: torch.Tensor,
        graph: Any,
        *,
        edge_weight: Optional[torch.Tensor] = None,  # ignored for baseline
        **kwargs: Any,
    ) -> torch.Tensor:
        """Apply GCN: X' = A_hat @ (X W).

        Args:
            x (torch.Tensor): Node features [N, Fin].
            graph (Any): Either (edge_index, num_nodes) or (edge_index, edge_weight) or (edge_index, ew, num_nodes).
            edge_weight (Optional[torch.Tensor]): Unused baseline.
            **kwargs (Any): Extra kwargs ignored.

        Returns:
            torch.Tensor: Output features [N, Fout].
        """
        normalized_adgacency = graph
        out = self.lin(x)
        return torch.sparse.mm(normalized_adgacency, out)


class _TorchNativeMeanConv(BaseConvolution):
    """Reference GNN convolution using mean aggregation of incoming neighbors."""
    
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True, **kwargs: Any) -> None:
        """Initialize a Torch-native mean aggregation convolution.

        Args:
            in_channels (int): Input feature size.
            out_channels (int): Output feature size.
            bias (bool): Include bias in linear transform.
            **kwargs (Any): Reserved for future options.
        """
        super().__init__(in_channels, out_channels, bias=bias, **kwargs)
        self.lin = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(
        self,
        x: torch.Tensor,
        graph: Any,
        *,
        edge_weight: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Apply mean aggregation convolution: X' = D_in^{-1} @ A^T @ (X @ W).

        Args:
            x (torch.Tensor): Node features [N, Fin].
            graph (Any): Tuple of (adj_matrix_transposed, in_degree_inv_diag) where:
                - adj_mat_normalized_by_in_degree_transposed: sparse COO tensor [N, N] (A^T), normalized by in-degree before transposition
            edge_weight (Optional[torch.Tensor]): Unused for this baseline.
            **kwargs (Any): Extra kwargs ignored.

        Returns:
            torch.Tensor: Output features [N, Fout].
        """
        adj_mat_normalized_by_in_degree_transposed = graph
        
        out = self.lin(x)
        out = torch.sparse.mm(adj_mat_normalized_by_in_degree_transposed, out)

        return out


@BackendRegistry.register_backend("torch_native_gcn")
class TorchNativeBackend(BaseBackend):
    """Backend instantiating simple Torch-native GNN convs."""
    def create_conv(
        self,
        conv_type: str,
        in_channels: int,
        out_channels: int,
        **kwargs: Any,
    ):
        """Factory for Torch-native convs.

        Args:
            conv_type (str): 'gcn' supported (extend as needed).
            in_channels (int): Input feature size.
            out_channels (int): Output feature size.
            **kwargs (Any): Extra kwargs.

        Returns:
            BaseConvolution: Torch-native convolution layer.
        """
        # guard unsupported backends
        if conv_type == "gcn":
            return _TorchNativeGCNConv(in_channels, out_channels, **kwargs)
        raise NotImplementedError(f"Convolution `{conv_type}` is not implemented for backend {self.__class__.__name__}")

@BackendRegistry.register_backend("torch_native_meanaggr")
class TorchNativeMeanBackend(BaseBackend):
    """Backend instantiating simple Torch-native mean aggregation convs."""
    
    def create_conv(
        self,
        conv_type: str,
        in_channels: int,
        out_channels: int,
        **kwargs: Any,
    ):
        """Factory for Torch-native mean aggregation convs.

        Args:
            conv_type (str): 'gcn' supported (mean aggregation variant).
            in_channels (int): Input feature size.
            out_channels (int): Output feature size.
            **kwargs (Any): Extra kwargs.

        Returns:
            BaseConvolution: Torch-native mean aggregation convolution layer.
        """
        if conv_type == "mean_aggr":
            return _TorchNativeMeanConv(in_channels, out_channels, **kwargs)
        raise NotImplementedError(f"Convolution `{conv_type}` is not implemented for backend {self.__class__.__name__}")