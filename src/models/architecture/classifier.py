from typing import Any, Optional

import torch
import torch.nn as nn

from ..base import ClassifierSpec
from ..registry import register
from .encoder import GNNEncoder

doc = """
NodeClassifier: wraps a GNNEncoder and a linear classification head.

Forward accepts either:
- a batch dict with keys {'features','graph'} and returns logits [N, C], or
- (x, graph) positional inputs.
"""


class NodeClassifier(nn.Module):
    """Node classification model: encoder + linear head."""

    def __init__(self, spec: ClassifierSpec) -> None:
        """Initialize the classifier from a ClassifierSpec.

        Args:
            spec (ClassifierSpec): Includes an EncoderSpec and num_classes.

        Returns:
            None
        """
        super().__init__()
        self.spec = spec
        self.encoder = GNNEncoder(spec.encoder)

        # infer encoder output dim from last layer
        last = spec.encoder.layers[-1]
        out_dim = int(last.out_channels if last.conv_type != "gat" else last.out_channels * max(1, last.heads))
        self.dropout = nn.Dropout(p=spec.dropout) if spec.dropout and spec.dropout > 0.0 else nn.Identity()
        self.head = nn.Linear(out_dim, spec.num_classes, bias=True)

    def forward(self, batch_or_x: Any, graph: Any | None = None) -> torch.Tensor:
        """Compute logits for node classification.

        Args:
            batch_or_x (Any): Either a batch dict {'features','graph'} or feature tensor [N, F].
            graph (Optional[Any]): Graph container when passing x directly.

        Returns:
            torch.Tensor: Logits [N, num_classes].
        """
        if isinstance(batch_or_x, dict):
            x = batch_or_x["features"]
            g = batch_or_x["graph"]
        else:
            x = batch_or_x
            g = graph
        z = self.encoder(x, g)
        z = self.dropout(z)
        return self.head(z)

    def predict(self, batch_or_x: Any, graph: Any | None = None) -> torch.Tensor:
        """Return predicted class indices.

        Args:
            batch_or_x (Any): Batch dict or features tensor.
            graph (Optional[Any]): Graph when passing x directly.

        Returns:
            torch.Tensor: Predicted class indices [N].
        """
        logits = self.forward(batch_or_x, graph)
        return torch.argmax(logits, dim=-1)


# register a default entry-point with the models registry
@register("node_classifier")
def build_node_classifier(spec: ClassifierSpec) -> nn.Module:
    """Factory registered as 'node_classifier'.

    Args:
        spec (ClassifierSpec): Model specification.

    Returns:
        nn.Module: NodeClassifier instance.
    """
    return NodeClassifier(spec)
