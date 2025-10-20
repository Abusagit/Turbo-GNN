from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Literal, Mapping

import torch
from torch.utils.data import Dataset

from torch_geometric.datasets import Planetoid, Reddit
from torch_geometric.data import Data
import torch_geometric.datasets as pyg_datasets

from functools import wraps

from ogb.nodeproppred import NodePropPredDataset

from torch_geometric.edge_index import EdgeIndex
from torch_geometric.utils import add_self_loops as add_self_loops_pyg
from dgl import add_self_loop, graph as dgl_graph
import dgl.data as dgl_data

from .graphland_datasets import GraphLandDataset

try:  # pragma: no cover
    LEGACY_MODE = False
    from pylibcugraphops.pytorch import CSC, HeteroCSC
    HAS_PYLIBCUGRAPHOPS = True
except ImportError:
    HAS_PYLIBCUGRAPHOPS = False
    try:  # pragma: no cover
        from pylibcugraphops import make_fg_csr
        LEGACY_MODE = True
    except ImportError:
        pass

GraphBackendOption = Literal["pyg", "dgl", "edge_list", "coo", "csr", "csc", "normalized_adj_mat_gcn", "adj_mat", "adj_mat_in_degree_normalized_transposed", "cugraph"] # NOTE we can define cached formalizations via this option


# NOTE place representations here when you add new backend
MODEL_BACKEND_TO_GRAPH_REPR: Mapping[str, GraphBackendOption] = {  # NOTE this dict contains mapping for suitable graph representation for each convolution backend
    "pyg": "pyg",
    "dgl": "dgl",
    "torch_native_gcn": "normalized_adj_mat_gcn",
    "torch_native_mean_aggr": "adj_mat_in_degree_normalized_transposed",
    "torch_native_sum_aggr": "adj_mat",
    "cugraph": "cugraph",
}


doc = """
Single-graph dataset loaders that normalize OGB (ogbn-*), PyG, and DGL datasets
to a canonical representation consumable by any backend.

Batch contract (used in src/training/trainer.py):
    {
        'features': torch.Tensor [N, F],
        'labels' : torch.Tensor [N] or [N, C],
        'graph'  : backend-specific graph representation (see `GraphSample.__post_init__`),
        'mask'   : torch.BoolTensor [N],
    }

Notes:
- We standardize to a tuple for 'graph': (edge_index, edge_weight). Backends in
  this repo accept that form and can infer num_nodes if needed.
- All tensors are kept on CPU; the trainer moves them to device via _batch_to_device.
"""

# NOTE the last one can be optimized -- graph tensors can be placed on GPU once during the training

def ensure_cpu_device(func):
    """Wrap a function to ensure that default device is CPU.
    Returns back default device after the execution

    Some functions (e.g. Pytorch Geometric's ones) load tensors,
    and torch.load stores them on the default device
    """

    @wraps(func)
    def wrapper(*args, **kwargs):

        prev_default_device = torch.get_default_device()
        torch.set_default_device("cpu")
        res = func(*args, **kwargs)
        torch.set_default_device(prev_default_device)
        return res

    return wrapper

# ------------------------- Canonical sample container ------------------------- #

@dataclass
class GraphSample:
    """Holds a single large-graph sample in canonical tensor form.

    Attributes:
        graph_backend (GraphBackendOption): format for storing graph and its weights for different graph convolutions
        x (torch.Tensor): Node features [N, F].
        y (torch.Tensor): Node labels [N] or [N, C].
        edge_index (torch.Tensor): Long tensor [2, E] with (row, col) edges.
        edge_weight (Optional[torch.Tensor]): Optional edge weights [E].
        train_mask (Optional[torch.BoolTensor]): Training mask [N] (True for used nodes).
        val_mask (Optional[torch.BoolTensor]): Validation mask [N].
        test_mask (Optional[torch.BoolTensor]): Test mask [N].
    """
    backend: GraphBackendOption
    x: torch.Tensor
    y: torch.Tensor
    edge_index: torch.Tensor
    edge_weight: Optional[torch.Tensor] = None
    train_mask: Optional[torch.BoolTensor] = None
    val_mask: Optional[torch.BoolTensor] = None
    test_mask: Optional[torch.BoolTensor] = None
    _graph_repr: Any = None
    add_self_loops: bool = True

    def __post_init__(self):
        """
        Add self loops -- For correct benchmarking purposes (We aren't racing for the SOTA architectures here so we can do whatever we want)
            1) Store graph representation in _graph_repr field --> it will be used in the convolutions
            2) Place everything on a default device -- defined in scripts
        """
        graph = None
        if self.backend == "pyg":  # pyg eats standard edge index & weight
            if self.add_self_loops:
                self.edge_index, self.edge_weight = add_self_loops_pyg(self.edge_index, self.edge_weight)
            graph = (self._to_default_device(self.edge_index), self._to_default_device(self.edge_weight))
        elif self.backend == "dgl":
            graph = dgl_graph((self.edge_index[0], self.edge_index[1]), num_nodes=self.num_nodes)
            if self.edge_weight is not None:
                graph.edata["w"] = self.edge_weight  # type: ignore
            if self.add_self_loops:
                graph = add_self_loop(graph)
            graph = self._to_default_device(graph)
        elif self.backend == "normalized_adj_mat_gcn":
            graph = normalize_adj(edge_index=self.edge_index, num_nodes=self.num_nodes, how='both', add_self_loops=self.add_self_loops)
            graph = self._to_default_device(graph)
        elif self.backend == "adj_mat_in_degree_normalized_transposed":
            graph = normalize_adj(edge_index=self.edge_index, num_nodes=self.num_nodes, how='right', add_self_loops=self.add_self_loops)
            graph = self._to_default_device(graph)
        elif self.backend == "cugraph":
            normalized_gcn_adjacency = normalize_adj(edge_index=self.edge_index, num_nodes=self.num_nodes, how='both', add_self_loops=self.add_self_loops)
            edge_index = normalized_gcn_adjacency.indices().tolist()
            edge_weights_for_gcn = normalized_gcn_adjacency.values()
            edge_index_for_pyg = EdgeIndex(edge_index, sparse_size=(self.num_nodes, self.num_nodes), sort_order='row', is_undirected=False, device=torch.get_default_device())
            csc_graph = get_cugraph_with_gcn_weights(edge_index_for_pyg)  # edge index is already on GPU
            graph = (csc_graph, edge_weights_for_gcn)
        elif self.backend == "adj_mat":
            graph = normalize_adj(edge_index=self.edge_index, num_nodes=self.num_nodes, how='none', add_self_loops=self.add_self_loops)
            graph = self._to_default_device(graph)
        elif self.backend == "coo":
            ... # TODO
        elif self.backend == "csr":
            ... # TODO
        elif self.backend == "csc":
            ... # TODO
        elif self.backend == "edge_list":
            edge_list = self.edge_index.T
            graph = (self._to_default_device(edge_list), self._to_default_device(self.edge_weight))
            # TODO: add self-loops

        self._graph_repr = graph
        assert self._graph_repr is not None, f"The backend {self.backend} isn't supported"

        # place features, labels, masks on default device
        self.x = self._to_default_device(self.x)
        self.y = self._to_default_device(self.y)
        self.train_mask = self._to_default_device(self.train_mask)
        self.val_mask = self._to_default_device(self.val_mask)
        self.test_mask = self._to_default_device(self.test_mask)

    def _to_default_device(self, item: Any) -> Any:
        """If tensor, place on device"""
        try:
            item = item.to(torch.get_default_device())
        except Exception:
            pass
        return item

    @property
    def num_nodes(self) -> int:
        """Number of nodes N."""
        return self.x.shape[0] # type: ignore

    @property
    def num_features(self) -> int:
        """Feature dimensionality F."""
        return self.x.shape[1] # type: ignore

    @property
    def num_classes(self) -> int:
        """Number of classes if labels are class indices or one-hot."""
        if self.y.ndim == 1 and self.y.numel() > 0:
            # class indices -> infer max+1
            return self.y.max().item() + 1 # type: ignore
        if self.y.ndim == 2:
            return self.y.shape[1] # type: ignore
        assert False, "Unreachable"

    @property
    def graph_repr(self) -> Any:
        """Returns the representation of a graph with specified backend"""
        return self._graph_repr

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

        # backend-specific graph representation
        graph = self.sample.graph_repr
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
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    return train_mask, val_mask, test_mask


# ------------------------------- OGBN loaders -------------------------------- #

def load_ogbn(name: str, graph_backend: GraphBackendOption, root: str = "data") -> GraphSample:
    """Load an ogbn-* node property prediction dataset as a single-graph sample.

    Args:
        name (str): OGBN dataset name (e.g., 'ogbn-arxiv', 'ogbn-products').
        graph_backend (GraphBackendOption): format for storing graph and its weights for different graph convolutions.
        root (str): Download/cache directory.

    Returns:
        GraphSample: Canonical sample (x, y, edge_index, masks).

    Raises:
        ImportError: If OGB is not installed.
    """
    @ensure_cpu_device
    def _load_oggn_cpu():
        return NodePropPredDataset(name=name, root=root)

    dset = _load_oggn_cpu()
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
        backend=graph_backend,
    )


# ------------------------------- PyG loaders --------------------------------- #

def load_pyg_single_graph(name: str, graph_backend: GraphBackendOption, root: str = "data") -> GraphSample:
    """Load a single-graph dataset from PyTorch Geometric.

    Supported (common) names:
        - 'Cora', 'CiteSeer', 'PubMed'  -> Planetoid datasets
        - 'Reddit'                      -> Reddit
    For other names, attempts to import a dataset of that name from
    torch_geometric.datasets and expects a single-graph output.

    Args:
        name (str): Dataset name.
        graph_backend (GraphBackendOption): format for storing graph and its weights for different graph convolutions.
        root (str): Download/cache directory.

    Returns:
        GraphSample: Canonical sample with masks.

    Raises:
        ImportError: If PyG is not installed.
        ValueError: If dataset cannot be loaded as a single graph.
    """
    @ensure_cpu_device
    def _load_pyg_cpu():
        if name in ("cora", "citeseer", "pubmed"):
            dset = Planetoid(root=root, name=name)
            data: Data = dset[0]
        elif name in ("reddit",):
            dset = Reddit(root=root)
            data = dset[0]
        elif name in ('hm-categories', 'pokec-regions', 'web-topics', 'tolokers-2', 'city-reviews', 'artnet-exp', 'web-fraud',
                   'hm-prices', 'avazu-ctr','city-roads-M', 'city-roads-L', 'twitch-views', 'artnet-views', 'web-traffic'):
            dset = GraphLandDataset(root=root, name=name, split="RL")
            data = dset[0]
        else:
            raise ValueError(f"Unknown PyG dataset '{name}'. Supported: Cora/CiteSeer/PubMed/Reddit or provide a known class in torch_geometric.datasets.")
        return data

    data = _load_pyg_cpu()

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
        backend=graph_backend,
    )


# -------------------------------- DGL loaders -------------------------------- #

def load_dgl_single_graph(name: str, graph_backend: GraphBackendOption, root: str = "data") -> GraphSample:
    """Load a single-graph dataset from DGL.

    Supported (common) names:
        - 'cora', 'citeseer', 'pubmed' -> CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
        - 'reddit'                     -> RedditDataset

    Args:
        name (str): Dataset name (case-insensitive for common names).
        graph_backend (GraphBackendOption): format for storing graph and its weights for different graph convolutions.
        root (str): Download/cache directory.

    Returns:
        GraphSample: Canonical sample with masks.

    Raises:
        ImportError: If DGL is not installed.
        ValueError: If dataset is unknown or lacks standard masks.
    """

    @ensure_cpu_device
    def _load_dgl_cpu():
        if name == "cora":
            dset = dgl_data.CoraGraphDataset(raw_dir=root)
        elif name == "citeseer":
            dset = dgl_data.CiteseerGraphDataset(raw_dir=root)
        elif name == "pubmed":
            dset = dgl_data.PubmedGraphDataset(raw_dir=root)
        elif name == "reddit":
            dset = dgl_data.RedditDataset(raw_dir=root)
        else:
            raise ValueError(f"Unknown DGL dataset '{name}'. Supported: cora/citeseer/pubmed/reddit.")
        return dset

    dset = _load_dgl_cpu()
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
        backend=graph_backend,
    )

# ------------------------------ Public factories ----------------------------- #

@dataclass
class DatasetConfig:
    """Configuration for selecting and loading a single-graph dataset.

    Attributes:
        source (str): 'ogbn' | 'pyg' | 'dgl' | 'auto'
        name (str): Dataset name (e.g., 'ogbn-arxiv', 'Cora', 'reddit').
        graph_backend (GraphBackendOption): format for storing graph and its weights for different graph convolutions.
        root (str): Download/cache directory.
    """
    source: str
    name: str
    graph_backend: GraphBackendOption
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
        return load_ogbn(cfg.name, root=cfg.root, graph_backend=cfg.graph_backend)
    if s == "pyg":
        return load_pyg_single_graph(cfg.name, root=cfg.root, graph_backend=cfg.graph_backend)
    if s == "dgl":
        return load_dgl_single_graph(cfg.name, root=cfg.root, graph_backend=cfg.graph_backend)
    if s == "auto":

        # ogbn-* -> OGBN; else try PyG; then DGL.
        if cfg.name.lower().startswith("ogbn-"):
            return load_ogbn(cfg.name, root=cfg.root, graph_backend=cfg.graph_backend)
        try:
            return load_pyg_single_graph(cfg.name, root=cfg.root, graph_backend=cfg.graph_backend)
        except Exception:
            return load_dgl_single_graph(cfg.name, root=cfg.root, graph_backend=cfg.graph_backend)

    raise KeyError(f"Unsupported dataset source '{cfg.source}'")


def normalize_adj(edge_index: torch.Tensor, num_nodes: int, how: Literal["left", "right", "both", "none"],
                  add_self_loops: bool = True) -> torch.Tensor:
    """Compute symmetric normalized adjacency (A_hat) as sparse COO.

    Args:
        edge_index (torch.Tensor): [2, E] long tensor.
        num_nodes (int): Number of nodes.

    Returns:
        torch.Tensor: Sparse COO adjacency with added self-loops and:
            - D^{-1/2} A D^{-1/2} normalization if `how` == "both".
            - D_in^{-1} A^T normalization if `how` == "right" -- normalization for mean-aggregation
            - A if `how` == "none" -- normalization for adj-mat backend
    """
    device = edge_index.device
    idx = edge_index

    if add_self_loops:
        self_loops = torch.arange(num_nodes, device=device)
        loop_idx = torch.stack([self_loops, self_loops], dim=0)
        idx = torch.cat([idx, loop_idx], dim=1)
        edge_index = idx

    if how == "both":
        values = torch.ones(idx.size(1), device=device)
        adj = torch.sparse_coo_tensor(idx, values, (num_nodes, num_nodes)).T.coalesce().to(values.device)

        deg1 = torch.sparse.sum(adj, dim=1).to_dense()
        D_inv_sqrt1 = torch.pow(deg1.clamp(min=1.0), -0.5)

        deg0 = torch.sparse.sum(adj, dim=0).to_dense()
        D_inv_sqrt0 = torch.pow(deg0.clamp(min=1.0), -0.5)

        idx, values = adj.indices(), adj.values()
        row, col = idx
        norm_vals = D_inv_sqrt1[row] * values * D_inv_sqrt0[col]
        return torch.sparse_coo_tensor(idx, norm_vals, (num_nodes, num_nodes)).coalesce()
    elif how == "left":
        raise NotImplementedError()
    elif how == "right":
        """
            Computes A^T (transposed adjacency) and D_in^{-1} (inverse in-degree diagonal).
            This matches DGL's copy_u_mean operation.
        """
        device = edge_index.device
        src, dst = edge_index[0], edge_index[1]

        values = torch.ones(edge_index.size(1), device=device)
        adj_t_indices = torch.stack([dst, src], dim=0)
        adj_t = torch.sparse_coo_tensor(
            adj_t_indices,
            values,
            (num_nodes, num_nodes)
        ).coalesce()

        in_degrees = torch.zeros(num_nodes, device=device)
        in_degrees.scatter_add_(0, dst, torch.ones_like(dst, dtype=torch.float32))

        # handle isolated nodes (in_degree = 0) by setting to 1 to avoid division by zero
        in_degrees = in_degrees.clamp(min=1.0)

        in_degree_inv = 1.0 / in_degrees
        diag_indices = torch.arange(num_nodes, device=device).unsqueeze(0).repeat(2, 1)
        in_degree_inv_diag = torch.sparse_coo_tensor(
            diag_indices,
            in_degree_inv,
            (num_nodes, num_nodes)
        ).coalesce()

        adj_t_normalized = in_degree_inv_diag @ adj_t
        return adj_t_normalized
    elif how == "none":
        """
            Computes A^T (transposed adjacency).
            This matches DGL's copy_u_sum operation.
        """
        device = edge_index.device
        src, dst = edge_index[0], edge_index[1]

        values = torch.ones(edge_index.size(1), device=device)
        adj_t_indices = torch.stack([dst, src], dim=0)
        adj_t = torch.sparse_coo_tensor(
            adj_t_indices,
            values,
            (num_nodes, num_nodes)
        ).coalesce()

        return adj_t
    else:
        raise ValueError(f"Normalization type {how} is inappropriate")

def get_cugraph_with_gcn_weights(
    edge_index: EdgeIndex,
) -> CSC:
    """Constructs a :obj:`cugraph` graph object from CSC representation.
        NOTE

    Args:
        edge_index (EdgeIndex): The edge indices.

    Returns CSC graph and edge index which is used only in GCN computation

    """
    if not isinstance(edge_index, EdgeIndex):
        raise ValueError(f"'edge_index' needs to be of type 'EdgeIndex' "
                            f"(got {type(edge_index)})")

    edge_index = edge_index.sort_by('col')[0]
    num_src_nodes = edge_index.get_sparse_size(0)
    (colptr, row), _ = edge_index.get_csc()

    if not row.is_cuda:
        raise RuntimeError("'get_cugraph' requires GPU-based processing (got CPU tensor)")

    if LEGACY_MODE:
        return make_fg_csr(colptr, row)

    return CSC(colptr, row, num_src_nodes=num_src_nodes)
