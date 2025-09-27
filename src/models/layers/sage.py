from typing import Any, Optional

import torch
import torch.nn as nn

from .conv_dispatcher import create_conv_layer
from ..base import activation_factory, norm_factory

doc = """
GraphSAGE block: mean/sum/max aggregation with post norm/activation/dropout/residual.
"""


class SAGEBlock(nn.Module):
    """GraphSAGE block."""

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
        """Initialize a GraphSAGE block.

        Args:
            backend (str): Backend name.
            in_channels (int): Input feature dim.
            out_channels (int): Output feature dim.
            bias (bool): Use bias.
            activation (str): Post-norm activation.
            norm (str): 'batch'|'layer'|'none'.
            dropout (float): Dropout after activation.
            residual (bool): Residual if dims match.
            **conv_kwargs (Any): Extra kwargs passed to SAGE conv (aggr='mean'|'sum'|'max').

        Returns:
            None
        """
        super().__init__()
        self.conv = create_conv_layer("sage", backend, in_channels, out_channels, bias=bias, **conv_kwargs)
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
        """Apply GraphSAGE block.

        Args:
            x (torch.Tensor): Node features [N, Fin].
            graph (Any): Graph accepted by backend conv.
            edge_weight (Optional[torch.Tensor]): Optional edge weights (unused by default SAGE).

        Returns:
            torch.Tensor: Output features [N, Fout].
        """
        out = self.conv(x, graph)
        out = self.norm(out)
        out = self.act(out)
        out = self.drop(out)
        if self.use_residual:
            out = out + x
        return out
