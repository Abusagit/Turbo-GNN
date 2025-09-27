from typing import Any, Optional

import torch
import torch.nn as nn

from ..base import BaseBackend, BaseConvolution
from ..registry import BackendRegistry
from .utils import normalize_adj

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
        if isinstance(graph, (tuple, list)):
            edge_index = graph[0]
            num_nodes = graph[2] if len(graph) > 2 else int(edge_index.max().item()) + 1
        else:
            # assume dict-like or object with attributes
            edge_index = getattr(graph, "edge_index")
            num_nodes = getattr(graph, "num_nodes", int(edge_index.max().item()) + 1)
        a_hat = normalize_adj(edge_index.long(), int(num_nodes), how="both")
        out = self.lin(x)
        return torch.sparse.mm(a_hat, out)


@BackendRegistry.register_backend("torch_native")
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
        ct = conv_type.lower()
        if ct == "gcn":
            return _TorchNativeGCNConv(in_channels, out_channels, **kwargs)
        raise KeyError(f"Unsupported conv_type for torch_native backend: {conv_type}")
