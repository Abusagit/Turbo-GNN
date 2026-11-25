"""Heterophilous graphs of Platonov et al., "A critical look at the evaluation of GNNs under
heterophily: Are we really making progress?" (ICLR 2023) <https://arxiv.org/abs/2302.11640>.

Read straight from the authors' release <https://github.com/yandex-research/heterophilous-graphs>,
one ``.npz`` per graph, with numpy and torch only -- no PyG or DGL. Each archive holds

    node_features  [N, F] float32
    node_labels    [N]    int64
    edges          [E, 2] int64, every undirected edge listed once
    train_masks, val_masks, test_masks  [S, N] bool, S fixed splits

and is stored as ``<root>/<name>/raw/<file>.npz``, downloaded on first use.
"""

import os
import os.path as osp
import shutil
import urllib.request
from typing import Dict

import numpy as np
import torch

URL = "https://github.com/yandex-research/heterophilous-graphs/raw/main/data"

# dataset name (as written in dataset configs) -> archive in the authors' release
FILES: Dict[str, str] = {
    "amazon-ratings": "amazon_ratings.npz",
}


def _download(url: str, path: str) -> None:
    """Fetch ``url`` to ``path`` atomically, so an interrupted download never looks complete.

    urllib honours the ``http(s)_proxy`` environment variables.
    """
    os.makedirs(osp.dirname(path), exist_ok=True)
    print(f"Downloading {url}")
    tmp = f"{path}.part"
    with urllib.request.urlopen(url) as response, open(tmp, "wb") as f:
        shutil.copyfileobj(response, f)
    os.replace(tmp, path)


def _to_undirected(edges: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """[E, 2] pairs -> [2, E'] edge_index with both directions, deduplicated, sorted by (row, col).

    Same edge set as the authors' ``dgl.to_bidirected`` and PyG's ``to_undirected``.
    """
    both = torch.cat([edges, edges.flip(1)], dim=0)
    keys = torch.unique(both[:, 0] * num_nodes + both[:, 1])  # unique() also sorts
    return torch.stack([keys // num_nodes, keys % num_nodes])


def load_heterophilous_graph(root: str, name: str, split_id: int = 0) -> Dict[str, torch.Tensor]:
    """Load one heterophilous graph, downloading its archive into ``root`` if missing.

    Args:
        root (str): Data directory; the archive lives in ``<root>/<name>/raw/``.
        name (str): Dataset name, a key of ``FILES`` (e.g. 'amazon-ratings').
        split_id (int): Which of the release's fixed train/val/test splits to return.

    Returns:
        Dict[str, torch.Tensor]: x [N, F], y [N], edge_index [2, E'] (undirected), and
        train_mask / val_mask / test_mask [N] bool.

    Raises:
        ValueError: If ``name`` is not a known heterophilous dataset.
    """
    if name not in FILES:
        raise ValueError(f"Unknown heterophilous dataset '{name}'; expected one of {sorted(FILES)}.")

    path = osp.join(root, name, "raw", FILES[name])
    if not osp.exists(path):
        _download(f"{URL}/{FILES[name]}", path)

    with np.load(path) as raw:
        x = torch.from_numpy(raw["node_features"]).float()
        return {
            "x": x,
            "y": torch.from_numpy(raw["node_labels"]).long(),
            "edge_index": _to_undirected(torch.from_numpy(raw["edges"]).long(), x.shape[0]),
            "train_mask": torch.from_numpy(raw["train_masks"][split_id]).bool(),
            "val_mask": torch.from_numpy(raw["val_masks"][split_id]).bool(),
            "test_mask": torch.from_numpy(raw["test_masks"][split_id]).bool(),
        }
