from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

import torch.nn as nn

doc = """
Base utilities and typed configs for building GNN models with pluggable backends.

- LayerSpec: per-layer configuration (conv_type, backend, dims, etc.)
- EncoderSpec / ClassifierSpec: high-level architecture configs
- activation_factory / norm_factory: helpers for blocks
"""


# ------------------------- Typed configuration objects ------------------------ #


@dataclass
class LayerSpec:
    """Configuration for a single GNN layer/block.

    Attributes:
        layer_type (Literal['residual_block']): Type of encoder block
        conv_type (Literal['gcn','gat_v2','sage','gin']): Convolution type.
        backend (str): Backend name ('pyg','dgl','torch_native', ...).
        in_channels (int): Input feature size.
        out_channels (int): Output feature size.
        heads (int): Number of attention heads (for GAT-like layers).
        bias (bool): Whether to use bias.
        dropout (float): Dropout probability applied after activation.
        activation (str): 'relu', 'gelu', 'prelu', 'elu', 'tanh', 'sigmoid', 'none'.
        norm (str): 'batch', 'layer', or 'none'.
        residual (bool): Add residual connection when in_channels==out_channels.
        conv_kwargs (Dict[str, Any]): Extra kwargs passed to the backend conv.
    """

    layer_type: Literal["redisual_block"]
    conv_type: Literal["gcn", "gat_v2", "sage", "gin"]
    backend: str
    in_channels: int
    out_channels: int
    heads: int = 1
    bias: bool = True
    dropout: float = 0.0
    activation: str = "relu"
    norm: str = "none"
    residual: bool = False
    conv_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class EncoderSpec:
    """Configuration for an encoder composed of stacked LayerSpec blocks.

    Attributes:
        layers (List[LayerSpec]): Ordered list of layer specs.
    """

    layers: list[LayerSpec]


@dataclass
class ClassifierSpec:
    """Configuration for a node-level classifier model.

    Attributes:
        encoder (EncoderSpec): Encoder configuration.
        num_classes (int): Number of output classes.
        dropout (float): Dropout applied before the final linear head.
    """

    encoder: EncoderSpec
    num_classes: int
    dropout: float = 0.0


# --------------------------- Small factory helpers --------------------------- #


def activation_factory(name: str, *, dim: int | None = None) -> nn.Module:
    """Construct an activation module by name.

    Args:
        name (str): Activation name ('relu','gelu','prelu','elu','tanh','sigmoid','none').
        dim (Optional[int]): Optional feature dim (for PReLU num_parameters).

    Returns:
        nn.Module: Activation module (or nn.Identity for 'none').
    """
    key = (name or "relu").lower()
    if key == "relu":
        return nn.ReLU(inplace=False)
    if key == "gelu":
        return nn.GELU()
    if key == "prelu":
        return nn.PReLU(num_parameters=1 if dim is None else dim)
    if key == "elu":
        return nn.ELU()
    if key == "tanh":
        return nn.Tanh()
    if key == "sigmoid":
        return nn.Sigmoid()
    return nn.Identity()


def norm_factory(name: str, dim: int) -> nn.Module:
    """Construct a normalization layer by name.

    Args:
        name (str): 'batch', 'layer', 'rms', or 'none'.
        dim (int): Feature dimension for affine parameters.

    Returns:
        nn.Module: Normalization module (or nn.Identity).
    """
    key = (name or "none").lower()
    if key == "batch":
        return nn.BatchNorm1d(dim)
    if key == "layer":
        return nn.LayerNorm(dim)
    if key == "rms":
        return nn.RMSNorm(dim)
    return nn.Identity()
