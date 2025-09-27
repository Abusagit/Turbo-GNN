from typing import Any, Optional

import torch
import torch.nn as nn

from .conv_dispatcher import create_conv_layer
from ..base import activation_factory, norm_factory

doc = """
GAT block: backend-agnostic attention conv with norm/activation/dropout/residual.
"""


class GATBlock(nn.Module):
    """Graph Attention Network (GAT) block."""

    def __init__(
        self,
        *,
        backend: str,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        bias: bool = True,
        activation: str = "elu",
        norm: str = "none",
        dropout: float = 0.0,
        residual: bool = False,
        **conv_kwargs: Any,
    ) -> None:
        """Initialize a GAT block.

        Args:
            backend (str): Backend name.
            in_channels (int): Input feature dim.
            out_channels (int): Output feature dim (per-head if concat=True).
            heads (int): Number of attention heads.
            bias (bool): Use bias.
            activation (str): Post-norm activation.
            norm (str): 'batch'|'layer'|'none'.
            dropout (float): Dropout after activation.
            residual (bool): Residual if dims match.
            **conv_kwargs (Any): Extra kwargs forwarded to backend attention conv (concat, dropout, etc.).

        Returns:
            None
        """
        super().__init__()
        self.conv = create_conv_layer(
            "gat", backend, in_channels, out_channels, heads=heads, bias=bias, **conv_kwargs
        )
        # Determine output dim from conv if needed (PyG concat may change channels)
        self.out_channels = getattr(self.conv, "out_channels", out_channels * getattr(self.conv, "heads", heads))
        self.norm = norm_factory(norm, self.out_channels)
        self.act = activation_factory(activation, dim=self.out_channels)
        self.drop = nn.Dropout(p=dropout) if dropout and dropout > 0.0 else nn.Identity()
        self.use_residual = bool(residual and in_channels == self.out_channels)

    def forward(
        self,
        x: torch.Tensor,
        graph: Any,
        *,
        edge_weight: Optional[torch.Tensor] = None,  # not used by classic GAT
    ) -> torch.Tensor:
        """Apply GAT block.

        Args:
            x (torch.Tensor): Node features [N, Fin].
            graph (Any): Graph accepted by backend.
            edge_weight (Optional[torch.Tensor]): Ignored for most GAT variants.

        Returns:
            torch.Tensor: Output features [N, Fout].
        """
        out = self.conv(x, graph)  # typical GAT ignores edge_weight
        out = self.norm(out)
        out = self.act(out)
        out = self.drop(out)
        if self.use_residual:
            out = out + x
        return out
