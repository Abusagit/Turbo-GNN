from typing import Any, List

import torch
import torch.nn as nn

from ..base import EncoderSpec
from ..layers import ResidualBlock

doc = """
GNNEncoder: stacks typed layer blocks (GCN/GATv2/SAGE/GIN) per EncoderSpec.
"""


_BLOCKS = {
    "residual_block": ResidualBlock,
}


class GNNEncoder(nn.Module):
    """Encoder that stacks graph conv blocks defined in an EncoderSpec."""

    def __init__(self, spec: EncoderSpec) -> None:
        """Construct a GNN encoder from an EncoderSpec.

        Args:
            spec (EncoderSpec): Ordered list of LayerSpec entries.

        Returns:
            None
        """
        super().__init__()
        self.spec = spec
        blocks: list[nn.Module] = []
        for layer in spec.layers:
            cls = _BLOCKS[layer.layer_type]
            blocks.append(
                cls(
                    conv_type=layer.conv_type,
                    backend=layer.backend,
                    in_channels=layer.in_channels,
                    out_channels=layer.out_channels,
                    heads=layer.heads,
                    bias=layer.bias,
                    activation=layer.activation,
                    norm=layer.norm,
                    dropout=layer.dropout,
                    residual=layer.residual,
                    **(layer.conv_kwargs or {}),
                )
            )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor, graph: Any) -> torch.Tensor:
        """Apply the stacked blocks to produce node embeddings.

        Args:
            x (torch.Tensor): Node features [N, Fin].
            graph (Any): Graph container accepted by backend blocks, or -- for mini-batch
                training -- a list with one sampled block per layer (outermost first).

        Returns:
            torch.Tensor: Node embeddings after final block [N, Fout].
        """
        if isinstance(graph, (list, tuple)):
            assert len(graph) == len(self.blocks), f"{len(graph)} blocks for {len(self.blocks)} layers"
            for block, g in zip(self.blocks, graph):
                x = block(x, g)
            return x
        for block in self.blocks:
            x = block(x, graph)
        return x
