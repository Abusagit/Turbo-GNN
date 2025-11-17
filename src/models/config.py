from typing import Any, Dict, List, Optional

import torch.nn as nn
import yaml

from .base import ClassifierSpec, EncoderSpec, LayerSpec
from .registry import build as build_registered

doc = """
Config-driven model construction helpers.

Reads YAML (or dicts) describing the model layers/architecture and converts it
into LayerSpec / EncoderSpec / ClassifierSpec. Optionally infers missing
`in_channels` from the dataset feature dimension.

Expected YAML schema (example):

architecture: node_classifier
num_classes: 7
dropout: 0.5

encoder:
  layers:
    - conv_type: null
      backend: null
      in_channels: auto          # or omit / <= 0 to infer
      out_channels: 64
      norm: batch
      activation: relu
      dropout: 0.5
      residual: false
      conv_kwargs:
        cached: true             # (backend-specific optional arg)
    - conv_type: null
      backend: null
      in_channels: auto
      out_channels: 64
      norm: batch
      activation: relu
      dropout: 0.5
      residual: true

Notes:
- If you specify `in_channels` <= 0 or "auto", the loader will infer it by:
    first layer: use `input_dim` arg from the dataset/features,
    next layers: previous layer *effective* output dim.
"""


# ---------------------------- Public entry points ---------------------------- #


def load_model_spec_from_yaml(
    path: str,
    *,
    backend_to_override: str,
    conv_type_to_override: str,
    input_dim: int | None = None,
    override_num_classes: int | None = None,
) -> ClassifierSpec:
    """Load a model spec from a YAML file.

    Args:
        path (str): Path to the YAML model file.
        input_dim (Optional[int]): Feature dimension (N, F) to infer first layer
            `in_channels` when not explicitly specified.
        override_num_classes (Optional[int]): If provided, overrides YAML's
            num_classes field.

    Returns:
        ClassifierSpec: Parsed, validated, and (optionally) in_channels-inferred spec.

    Raises:
        FileNotFoundError: If the YAML file cannot be opened.
        yaml.YAMLError: If the YAML is malformed.
        ValueError: If required fields are missing or invalid.
    """
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    return classifier_spec_from_config(
        cfg,
        backend_to_override=backend_to_override,
        conv_type_to_override=conv_type_to_override,
        input_dim=input_dim,
        override_num_classes=override_num_classes,
    )


def build_model_from_yaml(
    path: str,
    *,
    backend_to_override: str,
    conv_type_to_override: str,
    input_dim: int | None = None,
    override_num_classes: int | None = None,
) -> nn.Module:
    """Build a registered model from a YAML file.

    Args:
        path (str): Path to YAML model config.
        input_dim (Optional[int]): Feature dimension (N, F) for in_channels inference.
        override_num_classes (Optional[int]): Optional override for number of classes.

    Returns:
        nn.Module: Constructed model instance ready for training.
    """
    spec = load_model_spec_from_yaml(
        path,
        backend_to_override=backend_to_override,
        conv_type_to_override=conv_type_to_override,
        input_dim=input_dim,
        override_num_classes=override_num_classes,
    )

    arch_name = _read_architecture_name(
        {}, default="node_classifier"
    )  # spec defines the body; name controls registry key
    return build_registered(arch_name, spec=spec)


def classifier_spec_from_config(
    cfg: dict[str, Any],
    *,
    backend_to_override: str,
    conv_type_to_override: str,
    input_dim: int | None = None,
    override_num_classes: int | None = None,
) -> ClassifierSpec:
    """Create a ClassifierSpec from a config dict (as loaded from YAML).

    Args:
        cfg (Dict[str, Any]): Parsed YAML dictionary.
        input_dim (Optional[int]): If provided, used to infer first layer input size.
        override_num_classes (Optional[int]): If provided, overrides cfg['num_classes'].

    Returns:
        ClassifierSpec: A complete classifier specification.
    """
    enc_cfg = _read_encoder_config(cfg)
    layers = [
        _parse_layer_dict(
            ld,
            backend_to_override=backend_to_override,
            conv_type_to_override=conv_type_to_override,
        )
        for ld in enc_cfg.get("layers", [])
    ]

    if not layers:
        raise ValueError("encoder.layers must contain at least one layer")

    # infer missing in_channels if requested
    _infer_in_channels(layers, input_dim=input_dim)

    enc_spec = EncoderSpec(layers=layers)
    num_classes = int(override_num_classes) if override_num_classes is not None else int(cfg.get("num_classes"))  # type: ignore[arg-type]
    dropout = float(cfg.get("dropout", 0.0))
    return ClassifierSpec(encoder=enc_spec, num_classes=num_classes, dropout=dropout)


# ------------------------------ Internal helpers ----------------------------- #


def _read_architecture_name(cfg: dict[str, Any], default: str = "node_classifier") -> str:
    """Read the architecture name from config or return a default.

    Args:
        cfg (Dict[str, Any]): Config dictionary.
        default (str): Fallback architecture registry key.

    Returns:
        str: Architecture name (registry key).
    """
    name = cfg.get("architecture", default)
    return str(name)


def _read_encoder_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Read the encoder sub-config.

    Args:
        cfg (Dict[str, Any]): Full config dict.

    Returns:
        Dict[str, Any]: Encoder sub-dict with 'layers' key.
    """
    enc = cfg.get("encoder")
    if enc is None:
        # allow top-level "layers" as a shortcut
        enc = {"layers": cfg.get("layers", [])}
    if "layers" not in enc:
        raise ValueError("encoder config must include 'layers'")
    return enc


def _parse_layer_dict(d: dict[str, Any], backend_to_override: str, conv_type_to_override: str) -> LayerSpec:
    """Parse a single layer dictionary into a LayerSpec.

    Args:
        d (Dict[str, Any]): Layer configuration from YAML.

    Returns:
        LayerSpec: Parsed layer spec with defaults.

    Raises:
        KeyError: If required keys are missing.
        ValueError: If values are invalid.
    """
    required = ("conv_type", "backend", "out_channels")
    for k in required:
        if k not in d:
            raise KeyError(f"Layer is missing required key '{k}'")

    layer_type = str(d["layer_type"]).lower()
    conv_type = conv_type_to_override
    backend = backend_to_override
    out_channels = int(d["out_channels"])

    # in_channels may be provided or left for inference
    in_val = d.get("in_channels", "auto")
    if isinstance(in_val, str) and in_val.lower() in ("auto", "infer", ""):
        in_channels = -1  # sentinel for inference
    else:
        in_channels = int(in_val)

    # optional knobs
    bias = bool(d.get("bias", True))
    dropout = float(d.get("dropout", 0.0))
    activation = str(d.get("activation", "relu"))
    norm = str(d.get("norm", "none"))
    residual = bool(d.get("residual", False))
    heads = int(d.get("heads", 1))
    conv_kwargs = dict(d.get("conv_kwargs", {}))

    return LayerSpec(
        layer_type=layer_type,  # type: ignore[arg-type]
        conv_type=conv_type,  # type: ignore[arg-type]
        backend=backend,
        in_channels=in_channels,
        out_channels=out_channels,
        heads=heads,
        bias=bias,
        dropout=dropout,
        activation=activation,
        norm=norm,
        residual=residual,
        conv_kwargs=conv_kwargs,
    )


def _infer_in_channels(layers: list[LayerSpec], *, input_dim: int | None) -> None:
    """Fill missing `in_channels` in-place based on previous effective output.

    Args:
        layers (List[LayerSpec]): Sequence of LayerSpec to update in-place.
        input_dim (Optional[int]): First-layer input feature dimension.

    Returns:
        None

    Raises:
        ValueError: If the first layer needs inference but input_dim is None.
    """

    def effective_out(ls: LayerSpec) -> int:
        """Compute effective output features for chaining.

        Args:
            ls (LayerSpec): Layer spec (GAT may multiply by heads if concat).

        Returns:
            int: Effective output feature size for next layer's input.
        """
        return ls.out_channels

    prev_out: int | None = None
    for i, ls in enumerate(layers):
        if ls.in_channels is None or ls.in_channels <= 0:
            if i == 0:
                if input_dim is None:
                    raise ValueError("First layer requires 'in_channels' or a provided input_dim for inference.")
                ls.in_channels = int(input_dim)
            else:
                if prev_out is None:
                    raise ValueError("Internal error: prev_out unavailable during inference.")
                ls.in_channels = int(prev_out)
        prev_out = effective_out(ls)
