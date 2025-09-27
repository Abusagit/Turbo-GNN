from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
from torch.utils.data import Dataset

from torch_geometric.datasets import Planetoid, Reddit
from torch_geometric.data import Data
import torch_geometric.datasets as pyg_datasets

from ogb.nodeproppred import NodePropPredDataset

import dgl.data as dgl_data

# from .config_utils import load_yaml_config



doc = """
Single-graph dataset loaders that normalize OGB (ogbn-*), PyG, and DGL datasets
to a canonical representation consumable by any backend.

Batch contract (used in src/training/trainer.py):
    {
        'features': torch.Tensor [N, F],
        'labels' : torch.Tensor [N] or [N, C],
        'graph'  : (edge_index [2, E], edge_weight [E] or None),
        'mask'   : torch.BoolTensor [N],
    }

Notes:
- We standardize to a tuple for 'graph': (edge_index, edge_weight). Backends in
  this repo accept that form and can infer num_nodes if needed.
- All tensors are kept on CPU; the trainer moves them to device via _batch_to_device.
"""

# NOTE the last one can be optimized -- graph tensors can be placed on GPU once during the training

# ------------------------- Canonical sample container ------------------------- #

@dataclass
class GraphSample:
    """Holds a single large-graph sample in canonical tensor form.

    Attributes:
        x (torch.Tensor): Node features [N, F].
        y (torch.Tensor): Node labels [N] or [N, C].
        edge_index (torch.Tensor): Long tensor [2, E] with (row, col) edges.
        edge_weight (Optional[torch.Tensor]): Optional edge weights [E].
        train_mask (Optional[torch.BoolTensor]): Training mask [N] (True for used nodes).
        val_mask (Optional[torch.BoolTensor]): Validation mask [N].
        test_mask (Optional[torch.BoolTensor]): Test mask [N].
    """
    x: torch.Tensor
    y: torch.Tensor
    edge_index: torch.Tensor
    edge_weight: Optional[torch.Tensor] = None
    train_mask: Optional[torch.BoolTensor] = None
    val_mask: Optional[torch.BoolTensor] = None
    test_mask: Optional[torch.BoolTensor] = None

    @property
    def num_nodes(self) -> int:
        """Number of nodes N."""
        return self.x.shape[0]

    @property
    def num_features(self) -> int:
        """Feature dimensionality F."""
        return self.x.shape[1]

    @property
    def num_classes(self) -> Optional[int]:
        """Number of classes if labels are class indices or one-hot."""
        if self.y.ndim == 1 and self.y.numel() > 0:
            # class indices -> infer max+1
            return self.y.max().item() + 1
        if self.y.ndim == 2:
            return self.y.shape[1]
        return None

    def graph_tuple(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Canonical graph tuple used by the trainer/model: (edge_index, edge_weight)."""
        return self.edge_index, self.edge_weight


# ------------------------- Dataset wrapper (per split) ------------------------ #

class SingleGraphDataset(Dataset):
    """Wrap a single large graph as a PyTorch Dataset, exposing one item.

    The dataset yields a dict compatible with the trainer:
        - features, labels, graph, mask
    The `split` argument selects which mask to emit ('train'|'val'|'test').

    Example:
        train_ds = SingleGraphDataset(sample, split='train')
        batch = train_ds[0]
        # batch['graph'] is (edge_index, edge_weight)
    """

    def __init__(self, sample: GraphSample, split: str) -> None:
        """Initialize the dataset.

        Args:
            sample (GraphSample): Canonical sample containing x/y/graph/masks.
            split (str): Split to expose ('train', 'val', or 'test').

        Raises:
            ValueError: If requested split mask is missing.
        """
        split = split.lower()
        if split not in ("train", "val", "test"):
            raise ValueError("split must be one of {'train','val','test'}")

        mask = {
            "train": sample.train_mask,
            "val": sample.val_mask,
            "test": sample.test_mask,
        }[split]

        if mask is None:
            raise ValueError(f"Requested split '{split}' is not available for this dataset.")

        self.sample = sample
        self.mask = mask
        self.split = split

    def __len__(self) -> int:
        """Dataset length (single-graph -> length 1)."""
        return 1

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return the single batch dict (trainer-compatible).

        Args:
            idx (int): Index (ignored; always returns the single graph).

        Returns:
            Dict[str, Any]: A batch dict with 'features', 'labels', 'graph', 'mask'.
        """

        # canonical 'graph' for all backends is a tuple (edge_index, edge_weight)
        graph = self.sample.graph_tuple()
        return {
            "features": self.sample.x,
            "labels": self.sample.y,
            "graph": graph,
            "mask": self.mask,
        }


# ------------------------------ Utility helpers ------------------------------ #

def _masks_from_indices(num_nodes: int, splits: Dict[str, torch.Tensor]) -> Tuple[torch.BoolTensor, torch.BoolTensor, torch.BoolTensor]:
    """Create boolean masks from split index tensors.

    Args:
        num_nodes (int): Number of nodes N.
        splits (Dict[str, torch.Tensor]): Dict with keys 'train', 'valid'|'val', 'test'
            mapping to 1D index tensors.

    Returns:
        Tuple[torch.BoolTensor, torch.BoolTensor, torch.BoolTensor]: (train, val, test) masks.
    """
    train_idx = splits.get("train")
    val_idx = splits.get("valid", None)
    if val_idx is None:
        val_idx = splits.get("val")
    test_idx = splits.get("test")

    if train_idx is None or val_idx is None or test_idx is None:
        raise ValueError("Splits dict must contain 'train', 'val'/'valid', and 'test' indices")

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros_like(train_mask)
    test_mask = torch.zeros_like(train_mask)
    train_mask[train_idx.long()] = True
    val_mask[val_idx.long()] = True
    test_mask[test_idx.long()] = True
    return train_mask, val_mask, test_mask


# ------------------------------- OGBN loaders -------------------------------- #

def load_ogbn(name: str, root: str = "data") -> GraphSample:
    """Load an ogbn-* node property prediction dataset as a single-graph sample.

    Args:
        name (str): OGBN dataset name (e.g., 'ogbn-arxiv', 'ogbn-products').
        root (str): Download/cache directory.

    Returns:
        GraphSample: Canonical sample (x, y, edge_index, masks).

    Raises:
        ImportError: If OGB is not installed.
    """

    dset = NodePropPredDataset(name=name, root=root)
    split_idx = dset.get_idx_split()
    graph, labels = dset[0]

    edge_index = torch.as_tensor(graph["edge_index"], dtype=torch.long)
    x = torch.as_tensor(graph["node_feat"], dtype=torch.float32)
    y = torch.as_tensor(labels, dtype=torch.long)

    if y.ndim > 1 and y.size(-1) == 1:
        y = y.view(-1)

    train_mask, val_mask, test_mask = _masks_from_indices(x.shape[0], split_idx)

    return GraphSample(
        x=x,
        y=y.long(),
        edge_index=edge_index,
        edge_weight=None,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )


# ------------------------------- PyG loaders --------------------------------- #

def load_pyg_single_graph(name: str, root: str = "data") -> GraphSample:
    """Load a single-graph dataset from PyTorch Geometric.

    Supported (common) names:
        - 'Cora', 'CiteSeer', 'PubMed'  -> Planetoid datasets
        - 'Reddit'                      -> Reddit
    For other names, attempts to import a dataset of that name from
    torch_geometric.datasets and expects a single-graph output.

    Args:
        name (str): Dataset name.
        root (str): Download/cache directory.

    Returns:
        GraphSample: Canonical sample with masks.

    Raises:
        ImportError: If PyG is not installed.
        ValueError: If dataset cannot be loaded as a single graph.
    """
    n = name.lower()
    if n in ("cora", "citeseer", "pubmed"):
        dset = Planetoid(root=root, name=name)
        data: Data = dset[0]
    elif n in ("reddit",):
        dset = Reddit(root=root)
        data: Data = dset[0]
    else:
        # try dynamic import by attribute name (PascalCase/Exact)
        if hasattr(pyg_datasets, name):
            D = getattr(pyg_datasets, name)
            dset = D(root=root)
            if len(dset) != 1:
                raise ValueError(f"Expected a single-graph dataset for '{name}', got {len(dset)} graphs")
            data: Data = dset[0]
        else:
            raise ValueError(f"Unknown PyG dataset '{name}'. Supported: Cora/CiteSeer/PubMed/Reddit or provide a known class in torch_geometric.datasets.")

    x = data.x.float()
    y = data.y
    if y.ndim > 1 and y.size(-1) == 1:
        y = y.view(-1)
    edge_index = data.edge_index.long()
    edge_weight = getattr(data, "edge_weight", None)
    if edge_weight is not None:
        edge_weight = edge_weight.float()

    train_mask = getattr(data, "train_mask", None)
    val_mask = getattr(data, "val_mask", None)
    test_mask = getattr(data, "test_mask", None)

    if train_mask is None or val_mask is None or test_mask is None:
        raise ValueError(f"Dataset '{name}' does not provide standard masks; please construct custom splits.")

    return GraphSample(
        x=x,
        y=y.long(),
        edge_index=edge_index,
        edge_weight=edge_weight,
        train_mask=train_mask.bool(),
        val_mask=val_mask.bool(),
        test_mask=test_mask.bool(),
    )


# -------------------------------- DGL loaders -------------------------------- #

def load_dgl_single_graph(name: str, root: str = "data") -> GraphSample:
    """Load a single-graph dataset from DGL.

    Supported (common) names:
        - 'cora', 'citeseer', 'pubmed' -> CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
        - 'reddit'                     -> RedditDataset

    Args:
        name (str): Dataset name (case-insensitive for common names).
        root (str): Download/cache directory.

    Returns:
        GraphSample: Canonical sample with masks.

    Raises:
        ImportError: If DGL is not installed.
        ValueError: If dataset is unknown or lacks standard masks.
    """

    n = name.lower()
    if n == "cora":
        dset = dgl_data.CoraGraphDataset(raw_dir=root)
    elif n == "citeseer":
        dset = dgl_data.CiteseerGraphDataset(raw_dir=root)
    elif n == "pubmed":
        dset = dgl_data.PubmedGraphDataset(raw_dir=root)
    elif n == "reddit":
        dset = dgl_data.RedditDataset(raw_dir=root)
    else:
        raise ValueError(f"Unknown DGL dataset '{name}'. Supported: cora/citeseer/pubmed/reddit.")

    g = dset[0]

    x = g.ndata["feat"].float()
    y = g.ndata["label"]
    if y.ndim > 1 and y.size(-1) == 1:
        y = y.view(-1)

    src, dst = g.edges()
    edge_index = torch.stack([src.long(), dst.long()], dim=0)
    edge_weight = g.edata["w"] if "w" in g.edata else None
    if edge_weight is not None:
        edge_weight = edge_weight.float()


    train_mask = g.ndata.get("train_mask", None)
    val_mask = g.ndata.get("val_mask", None)
    test_mask = g.ndata.get("test_mask", None)
    if train_mask is None or val_mask is None or test_mask is None:
        raise ValueError(f"DGL dataset '{name}' lacks standard masks; please construct custom splits.")

    return GraphSample(
        x=x,
        y=y.long() if y.dtype not in (torch.long, torch.int64) else y,
        edge_index=edge_index,
        edge_weight=edge_weight,
        train_mask=train_mask.bool(),
        val_mask=val_mask.bool(),
        test_mask=test_mask.bool(),
    )


# ------------------------------ Public factories ----------------------------- #

@dataclass
class DatasetConfig:
    """Configuration for selecting and loading a single-graph dataset.

    Attributes:
        source (str): 'ogbn' | 'pyg' | 'dgl' | 'auto'
        name (str): Dataset name (e.g., 'ogbn-arxiv', 'Cora', 'reddit').
        root (str): Download/cache directory.
    """
    source: str
    name: str
    root: str = "data"


def load_single_graph(cfg: DatasetConfig) -> GraphSample:
    """Load a canonical single-graph sample according to config.

    Args:
        cfg (DatasetConfig): Dataset configuration with source/name/root.

    Returns:
        GraphSample: Canonical large-graph sample.

    Raises:
        KeyError: If source is unsupported.
    """
    s = cfg.source.lower()
    if s == "ogbn":
        return load_ogbn(cfg.name, root=cfg.root)
    if s == "pyg":
        return load_pyg_single_graph(cfg.name, root=cfg.root)
    if s == "dgl":
        return load_dgl_single_graph(cfg.name, root=cfg.root)
    if s == "auto":

        # ogbn-* -> OGBN; else try PyG; then DGL.
        if cfg.name.lower().startswith("ogbn-"):
            return load_ogbn(cfg.name, root=cfg.root)
        try:
            return load_pyg_single_graph(cfg.name, root=cfg.root)
        except Exception:
            return load_dgl_single_graph(cfg.name, root=cfg.root)
    raise KeyError(f"Unsupported dataset source '{cfg.source}'")


def create_split_datasets(cfg: DatasetConfig) -> Tuple[SingleGraphDataset, SingleGraphDataset, SingleGraphDataset]:
    """Convenience to construct per-split datasets for trainer.

    Args:
        cfg (DatasetConfig): Dataset selection/configuration.

    Returns:
        Tuple[SingleGraphDataset, SingleGraphDataset, SingleGraphDataset]:
            (train_ds, val_ds, test_ds)
    """
    sample = load_single_graph(cfg)
    return (
        SingleGraphDataset(sample, split="train"),
        SingleGraphDataset(sample, split="val"),
        SingleGraphDataset(sample, split="test"),
    )


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
