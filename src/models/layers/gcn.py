from typing import Any, Optional

import torch
import torch.nn as nn

from .conv_dispatcher import create_conv_layer
from ..base import activation_factory, norm_factory

doc = """
GCN block: backend-agnostic wrapper adding norm/activation/dropout/residual.
"""


class GCNBlock(nn.Module):
    """Graph Convolutional Network (GCN) block with optional post-processing."""

    def __init__(
        self,
        *,
        backend: str,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        activation: str = "relu",
        norm: str = "none",
        dropout: float = 0.0,
        residual: bool = False,
        **conv_kwargs: Any,
    ) -> None:
        """Initialize a GCN block.

            Creates a backend-specific GCN convolution via the dispatcher, then
            attaches normalization, activation, and dropout in that order.

        Args:
            backend (str): Backend name ('pyg','dgl','torch_native',...).
            in_channels (int): Input feature dimension.
            out_channels (int): Output feature dimension.
            bias (bool): Whether to use bias in the convolution.
            activation (str): Activation after norm ('relu','gelu','prelu','none',...).
            norm (str): Normalization ('batch','layer','none').
            dropout (float): Dropout probability applied after activation.
            residual (bool): If True and shapes match, add residual connection.
            **conv_kwargs (Any): Extra kwargs forwarded to backend conv (e.g., normalize=True).

        Returns:
            None
        """
        super().__init__()
        self.conv = create_conv_layer("gcn", backend, in_channels, out_channels, bias=bias, **conv_kwargs)
        self.norm = norm_factory(norm, out_channels)
        self.act = activation_factory(activation, dim=out_channels)
        self.drop = nn.Dropout(p=dropout) if dropout and dropout > 0.0 else nn.Identity()
        self.use_residual = bool(residual and in_channels == out_channels)

    def forward(
        self,
        x: torch.Tensor,
        graph: Any,
        *,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply GCN block.

        Args:
            x (torch.Tensor): Node features [N, Fin].
            graph (Any): Graph container accepted by the backend conv.
            edge_weight (Optional[torch.Tensor]): Optional edge weights [E].

        Returns:
            torch.Tensor: Output features [N, Fout].
        """
        out = self.conv(x, graph, edge_weight=edge_weight)
        out = self.norm(out)
        out = self.act(out)
        out = self.drop(out)
        if self.use_residual:
            out = out + x
        return out
