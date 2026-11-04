"""Offline tests for the PyG-free heterophilous-graph loader (src/data/heterophilous_datasets.py)."""

import io

import numpy as np
import pytest
import torch

from src.data import heterophilous_datasets as hd
from src.data.datasets import DatasetConfig, load_single_graph

# (1, 2) is listed in both directions to check deduplication.
EDGES = np.array([[0, 1], [1, 2], [2, 1]], dtype=np.int64)
UNDIRECTED = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])


def _archive() -> bytes:
    rng = np.random.default_rng(0)
    masks = np.array([[1, 0, 0, 1], [0, 1, 1, 0]], dtype=bool)  # two splits
    buf = io.BytesIO()
    np.savez(
        buf,
        node_features=rng.standard_normal((4, 3)).astype(np.float32),
        node_labels=np.array([0, 1, 2, 1], dtype=np.int64),
        edges=EDGES,
        train_masks=masks,
        val_masks=~masks,
        test_masks=~masks,
    )
    return buf.getvalue()


@pytest.fixture
def root(tmp_path):
    path = tmp_path / "amazon-ratings" / "raw" / "amazon_ratings.npz"
    path.parent.mkdir(parents=True)
    path.write_bytes(_archive())
    return str(tmp_path)


def test_reads_archive_as_undirected_graph(root):
    g = hd.load_heterophilous_graph(root, "amazon-ratings")
    assert g["x"].shape == (4, 3) and g["x"].dtype == torch.float32
    assert g["y"].tolist() == [0, 1, 2, 1]
    assert torch.equal(g["edge_index"], UNDIRECTED)
    assert g["train_mask"].tolist() == [True, False, False, True]
    assert hd.load_heterophilous_graph(root, "amazon-ratings", split_id=1)["train_mask"].tolist() == [
        False,
        True,
        True,
        False,
    ]


def test_downloads_missing_archive(tmp_path, monkeypatch):
    urls = []

    def fake_urlopen(url):
        urls.append(url)
        return io.BytesIO(_archive())

    monkeypatch.setattr(hd.urllib.request, "urlopen", fake_urlopen)
    g = hd.load_heterophilous_graph(str(tmp_path), "amazon-ratings")
    assert urls == [f"{hd.URL}/amazon_ratings.npz"]
    assert (tmp_path / "amazon-ratings" / "raw" / "amazon_ratings.npz").exists()
    assert not list(tmp_path.rglob("*.part"))
    assert torch.equal(g["edge_index"], UNDIRECTED)


def test_unknown_name_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unknown heterophilous dataset"):
        hd.load_heterophilous_graph(str(tmp_path), "roman-empire")


def test_dataset_config_source(root):
    cfg = DatasetConfig(source="heterophilous", name="amazon-ratings", root=root, conv_backend="pyg")
    sample = load_single_graph(cfg)
    assert torch.equal(sample._original_edge_index, UNDIRECTED)
    assert sample.train_mask.tolist() == [True, False, False, True]
