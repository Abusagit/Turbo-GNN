import json
import os
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import yaml

from ..data.datasets import (
    MODEL_BACKEND_TO_GRAPH_REPR,
    DatasetConfig,
    GraphBackendOption,
    SingleGraphDataset,
    load_single_graph,
)

doc = """
Common utilities for training/validation/benchmark scripts:
- YAML loading and deep-merge
- Seed setting for reproducibility
- Device helpers
- Output directory utilities
- JSON saving
"""


PathLike = Union[str, Path]


def read_yaml(path: PathLike) -> dict[str, Any]:
    """Load a YAML file into a Python dict.

    Args:
        path (PathLike): Path to the YAML file.

    Returns:
        Dict[str, Any]: Parsed YAML (empty dict if file is empty).
    """
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def deep_update(base: dict[str, Any], other: dict[str, Any]) -> dict[str, Any]:
    """Recursively update dictionary `base` with fields from `other`.

    Args:
        base (Dict[str, Any]): Dictionary to be mutated in place.
        other (Dict[str, Any]): Values to merge into `base`.

    Returns:
        Dict[str, Any]: The same `base` dictionary after merge.
    """
    for k, v in other.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def merge_yaml_files(paths: Sequence[PathLike]) -> dict[str, Any]:
    """Load multiple YAML files and deep-merge them in order.

    Args:
        paths (Sequence[PathLike]): List of YAML file paths. Later files override earlier ones.

    Returns:
        Dict[str, Any]: Merged configuration dictionary.
    """
    merged: dict[str, Any] = {}
    for p in paths:
        cfg = read_yaml(p)
        deep_update(merged, cfg)
    return merged


def set_global_seed(seed: int | None) -> None:
    """Set seeds for Python, NumPy, and PyTorch.

    Args:
        seed (Optional[int]): Seed value. If None, does nothing.

    Returns:
        None
    """
    seed = seed or 42

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def device_from_string(device_str: str | None) -> torch.device:
    """Construct a torch.device from a string or return a sensible default.

    Args:
        device_str (Optional[str]): e.g., "cuda", "cuda:0", or "cpu". If None, prefer CUDA if available.

    Returns:
        torch.device: Target device.
    """
    if device_str is None:
        return torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    return torch.device(device_str)


def ensure_outdir(path: PathLike) -> Path:
    """Create output directory if it does not exist.

    Args:
        path (PathLike): Directory path.

    Returns:
        Path: Absolute, existing directory Path.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p.resolve()


def save_json(path: PathLike, payload: dict[str, Any]) -> None:
    """Save a dictionary as pretty JSON.

    Args:
        path (PathLike): Output filepath.
        payload (Dict[str, Any]): Serializable dictionary.

    Returns:
        None
    """
    Path(path).write_text(json.dumps(payload, indent=4))


def create_split_datasets_from_config_dict(
    cfg: dict[str, Any],
) -> tuple[SingleGraphDataset, SingleGraphDataset, SingleGraphDataset]:
    """Load dataset per config dict and return split datasets.

    Expected dict keys:
        - dataset: { source: 'ogbn'|'pyg'|'dgl'|'auto', name: str, root: str }
    """
    ds_cfg = cfg.get("dataset") or {}
    source = str(ds_cfg.get("source", "auto"))
    name = str(ds_cfg.get("name"))
    root = str(ds_cfg.get("root", "data"))

    sample = load_single_graph(
        DatasetConfig(source=source, name=name, root=root, graph_backend=cfg.get("graph_backend", "edge_index"))
    )

    return (
        SingleGraphDataset(sample, split="train"),
        SingleGraphDataset(sample, split="val"),
        SingleGraphDataset(sample, split="test"),
    )


def create_split_datasets_from_yaml(
    path: str, graph_backend: GraphBackendOption = "pyg"
) -> tuple[SingleGraphDataset, SingleGraphDataset, SingleGraphDataset]:
    """Load a YAML config file (dataset) and return split datasets.

    Args:
        path (str): Path to YAML file.
        graph_backend (GraphBackendOption): Graph backend option.

    Returns:
        Tuple[SingleGraphDataset, SingleGraphDataset, SingleGraphDataset]:
            (train_ds, val_ds, test_ds)
    """
    import yaml

    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg["graph_backend"] = graph_backend
    return create_split_datasets_from_config_dict(cfg)


def infer_graph_backend(model_config_path: str) -> GraphBackendOption:
    """Infer graph representation from used backend in the model.
    Traverses model config, tries to find layers description containing backend information

    Args:
        model_config_path (str): path to model concig

    Raises:
        ValueError: If couldn't find graph representation for the backend
        RuntimeError: If couldn't find backend description

    Returns:
        GraphBackendOption: Description for graph representation
    """

    with open(model_config_path) as f:
        model_config_raw = yaml.safe_load(f)

    # search for the 'layers' key on the second level - its entries describe the layer backend
    # NOTE currently only a single backend is supported
    for value in model_config_raw.values():
        if isinstance(value, dict) and "layers" in value:
            layers = value["layers"]

            backends = [layer["backend"] for layer in layers]
            assert all(
                backends[i - 1] == backends[i] for i in range(1, len(backends))
            ), f"So far single backend per run is supported, got multiple backends: {backends}"

            graph_representation_backend = MODEL_BACKEND_TO_GRAPH_REPR.get(backends[0])
            if graph_representation_backend is None:
                raise ValueError(
                    f"Couldn't infer suitable graph representation for backend {graph_representation_backend}."
                    f"Current supporting mapping is: {MODEL_BACKEND_TO_GRAPH_REPR}"
                )
            return graph_representation_backend

    raise RuntimeError(f"Couldnt infer suitable graph representation from the model spec: {model_config_raw}")
