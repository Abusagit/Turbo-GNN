from typing import Any, Optional

import torch
import torch.nn as nn

from ..base import activation_factory, norm_factory
from .conv_dispatcher import create_conv_layer


class ResidualBlock(torch.nn.Module):
    def __init__(
        self,
        *,
        conv_type: str,
        backend: str,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        bias: bool = True,
        activation: str = "relu",
        norm: str = "none",
        dropout: float = 0.0,
        residual: bool = False,
        **conv_kwargs: Any,
    ) -> None:
        """
        Initialize residual block for GNN model. It contains:
        1) Graph Convolution with specified backend
        2)

        Arguments:
            conv_type (str): Convolution type (GCN/MeanAggr/GAT/etc.)
            backend (str): Backend name.
            in_channels (int): Input feature dim.
            out_channels (int): Output feature dim (per-head if concat=True).
            heads (int): Number of attention heads. Applicable for architectures with attention
            bias (bool): Use bias.
            activation (str): Post-norm activation.
            norm (str): 'batch'|'layer'|'none'.
            dropout (float): Dropout after activation.
            residual (bool): Residual if dims match.
            **conv_kwargs (Any): Extra kwargs forwarded to backend attention conv (concat, dropout, etc.).

        """

        super().__init__()

        self.conv = create_conv_layer(
            conv_type,
            backend,
            in_channels,
            in_channels,
            heads=heads,
            bias=bias,
            **conv_kwargs,  # NOTE no projection, do it later
        )
        self.projection = nn.Linear(in_channels, out_channels)
        self.out_channels = out_channels
        self.norm = norm_factory(norm, self.out_channels)
        self.act = activation_factory(activation, dim=self.out_channels)
        self.drop = nn.Dropout(p=dropout) if dropout and dropout > 0.0 else nn.Identity()
        self.use_residual = bool(residual and in_channels == self.out_channels)

    def forward(
        self,
        x: torch.Tensor,
        graph: Any,
        *,
        edge_weight: torch.Tensor | None = None,  # edge_weight is legacy
    ) -> torch.Tensor:
        """Apply GAT block.

        Args:
            x (torch.Tensor): Node features [N, Fin].
            graph (Any): Graph accepted by backend.
            edge_weight (Optional[torch.Tensor]): Ignored for most GAT variants.

        Returns:
            torch.Tensor: Output features [N, Fout].
        """
        out = self.conv(x, graph)
        out = self.projection(x)
        out = self.norm(out)
        out = self.act(out)
        out = self.drop(out)
        if self.use_residual:
            out = out + x
        return out
