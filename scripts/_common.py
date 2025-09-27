from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union, Tuple

import json
import os
import random

import numpy as np
import torch
import yaml

from src.data.datasets import load_single_graph, DatasetConfig, SingleGraphDataset

doc = """
Common utilities for training/validation/benchmark scripts:
- YAML loading and deep-merge
- Seed setting for reproducibility
- Device helpers
- Output directory utilities
- JSON saving
"""



PathLike = Union[str, Path]


def read_yaml(path: PathLike) -> Dict[str, Any]:
    """Load a YAML file into a Python dict.

    Args:
        path (PathLike): Path to the YAML file.

    Returns:
        Dict[str, Any]: Parsed YAML (empty dict if file is empty).
    """
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def deep_update(base: Dict[str, Any], other: Dict[str, Any]) -> Dict[str, Any]:
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


def merge_yaml_files(paths: Sequence[PathLike]) -> Dict[str, Any]:
    """Load multiple YAML files and deep-merge them in order.

    Args:
        paths (Sequence[PathLike]): List of YAML file paths. Later files override earlier ones.

    Returns:
        Dict[str, Any]: Merged configuration dictionary.
    """
    merged: Dict[str, Any] = {}
    for p in paths:
        cfg = read_yaml(p)
        deep_update(merged, cfg)
    return merged


def set_global_seed(seed: Optional[int]) -> None:
    """Set seeds for Python, NumPy, and PyTorch.

    Args:
        seed (Optional[int]): Seed value. If None, does nothing.

    Returns:
        None
    """
    seed = seed or 42

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def device_from_string(device_str: Optional[str]) -> torch.device:
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


def save_json(path: PathLike, payload: Dict[str, Any]) -> None:
    """Save a dictionary as pretty JSON.

    Args:
        path (PathLike): Output filepath.
        payload (Dict[str, Any]): Serializable dictionary.

    Returns:
        None
    """
    Path(path).write_text(json.dumps(payload, indent=4))


def create_split_datasets_from_config_dict(cfg: Dict[str, Any]) -> Tuple[SingleGraphDataset, SingleGraphDataset, SingleGraphDataset]:
    """Load dataset per config dict and return split datasets.

    Expected dict keys:
        - dataset: { source: 'ogbn'|'pyg'|'dgl'|'auto', name: str, root: str }
        - transforms: {...}   # optional (see `apply_transforms_from_config_dict`)
    """
    ds_cfg = cfg.get("dataset") or {}
    source = str(ds_cfg.get("source", "auto"))
    name = str(ds_cfg.get("name"))
    root = str(ds_cfg.get("root", "data"))

    sample = load_single_graph(DatasetConfig(source=source, name=name, root=root))

    return (
        SingleGraphDataset(sample, split="train"),
        SingleGraphDataset(sample, split="val"),
        SingleGraphDataset(sample, split="test"),
    )


def create_split_datasets_from_yaml(path: str) -> Tuple[SingleGraphDataset, SingleGraphDataset, SingleGraphDataset]:
    """Load a YAML config file (dataset + transforms), apply transforms once, and return split datasets.

    Args:
        path (str): Path to YAML file. See `src/data/config_utils.py` docstring for schema.

    Returns:
        Tuple[SingleGraphDataset, SingleGraphDataset, SingleGraphDataset]:
            (train_ds, val_ds, test_ds)
    """
    import yaml
    
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    return create_split_datasets_from_config_dict(cfg)
# ==============================================================================
