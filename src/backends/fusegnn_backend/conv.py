from typing import Any, Optional

import torch

from src.backends.fusegnn_backend.convs import garGCNConv, gasGCNConv

from ..base import BaseBackend, BaseConvolution
from ..registry import BackendRegistry


class _FuseGNN_GCNConv(BaseConvolution):
    """FuseGNN-backend GCN convolution wrapper."""

    def __init__(self, fuse_type: str):
        super().__init__()

        assert fuse_type in ("gar", "gas")
        if fuse_type == "gar":
            self.conv = garGCNConv(flow="source_to_target")
        elif fuse_type == "gas":
            self.conv = gasGCNConv(flow="source_to_target")

    def forward(
        self,
        x: torch.Tensor,
        graph: tuple[torch.Tensor, torch.Tensor | None, int],
        **kwargs: Any,
    ) -> torch.Tensor:
        """Apply GCNConv.

        Args:
            x (torch.Tensor): Node features [N, Fin].
            graph (tuple[torch.Tensor, torch.Tensor | None, int]): (edge_index, edge_weight, num_nodes).
            **kwargs (Any): Extra kwargs (ignored).

        """

        edge_index, edge_weight, _ = graph

        return self.conv(x, edge_index=edge_index, edge_weight=edge_weight)


@BackendRegistry.register_backend("fusegnn")
class FuseGNNBackend(BaseBackend):
    """Backend that instantiates FuseGNN-based convolutions. GCN and GAT are supported."""

    def create_conv(
        self,
        conv_type: str,
        fusegnn_fuse_type: str = "gar",
        **kwargs: Any,
    ) -> BaseConvolution:
        """Factory for FuseGNN convolution layers.

        Args:
            conv_type (str): 'gcn', 'gat'.
            fusegnn_fuse_type (str): fuse algorithm for FuseGNN to use: 'gar' (default), 'gas'.
            **kwargs (Any): ignored.

        Returns:
            BaseConvolution: An instance of the requested FuseGNN conv.
        """

        conv_type = conv_type.lower()
        assert fusegnn_fuse_type in ("gar", "gas")

        if conv_type == "gcn":
            return _FuseGNN_GCNConv(fusegnn_fuse_type)
        # elif conv_type == 'gat':
        #     return _FuseGNN_GATConv(fusegnn_fuse_type)
        raise KeyError(f"Unsupported conv_type for FuseGNN backend: {conv_type}")
