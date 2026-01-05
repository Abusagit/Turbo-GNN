from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Dict, Literal, Mapping, Optional, Tuple

import dgl.data as dgl_data
import torch
import torch_geometric.datasets as pyg_datasets
from dgl import add_self_loop
from dgl import graph as dgl_graph
from ogb.nodeproppred import NodePropPredDataset
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.datasets import Amazon, Coauthor, Planetoid, Reddit
from torch_geometric.edge_index import EdgeIndex
from torch_geometric.utils import add_self_loops as add_self_loops_pyg

from src.backends.fused3s_backend.bindings import f3s_preprocess
from src.data.converters import (
    AdjacencyForwardBackwardWithNodeBuckets,
    WSBFormat,
    build_csr_as_is,
    get_cugraph_with_gcn_weights,
    normalize_adj,
    reorder_graph,
    to_tcgnn_data,
)

from .graphland_datasets import GraphLandDataset

GraphBackendOption = Literal[
    "pyg",
    "dgl",
    "edge_list",
    "coo",
    "csr",
    "csc",
    "normalized_adj_mat_gcn",
    "adj_mat",
    "adj_mat_in_degree_normalized_transposed",
    "adj_mat_transposed",
    "cugraph",
    "tcgnn",
    "weighted_sparse_block",
    "csr_and_csr_transposed",
    "f3s",
    "cuda",
    "cuda_weighted_sparse_block_with_meta",
    "csr_and_csr_T_for_cusparse",
]  # NOTE we can define cached formalizations via this option


# NOTE place representations here when you add new backend
MODEL_BACKEND_TO_GRAPH_REPR: Mapping[str, GraphBackendOption] = {  # NOTE this dict contains mapping for suitable graph
    # representation for each convolution backend
    "pyg": "pyg",
    "dgl": "dgl",
    "torch_native_gcn": "normalized_adj_mat_gcn",
    "torch_native_mean_aggr": "adj_mat_in_degree_normalized_transposed",
    "torch_native_sum_aggr": "adj_mat_transposed",
    "cugraph": "cugraph",
    "torch_native_adj_mat": "adj_mat",
    "cusparse": "csr",
    "cusparse_precomputed_bwd": "csr_and_csr_T_for_cusparse",
    "fusegnn": "coo",
    "tcgnn": "tcgnn",
    "triton_block_sparse": "weighted_sparse_block",
    "cuda": "csr_and_csr_transposed",
    "f3s": "f3s",
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
    kernel_related_kwargs: dict[str, Any] = field(default_factory=lambda: {})

    _original_edge_index: Optional[torch.Tensor] = None
    _original_edge_weight: Optional[torch.Tensor] = None

    def update_graph_repr_with_new_hyperparameters(self, new_kernel_related_kwargs):
        delattr(self, "_graph_repr")
        torch.cuda.empty_cache()
        setattr(self, "_graph_repr", None)

        self.kernel_related_kwargs = new_kernel_related_kwargs
        self.add_self_loops = False  # we have already added self-loops when needed, now we don' have to do it
        self.edge_index = self._original_edge_index
        self.edge_weight = self._original_edge_weight

        self.__post_init__()
        return self

    def _save_original_connectivity(self):
        if self._original_edge_index is None:
            self._original_edge_index = self.edge_index.cpu().clone()
        if self._original_edge_weight is None and isinstance(self.edge_weight, torch.Tensor):
            self._original_edge_weight = self.edge_weight.cpu().clone()

    def __post_init__(self):
        """
        Add self loops -- For correct benchmarking purposes (We aren't racing for the SOTA architectures here so we can
        do whatever we want)
            1) Store graph representation in _graph_repr field --> it will be used in the convolutions
            2) Place everything on a default device -- defined in scripts
        """
        self._save_original_connectivity()

        graph_reordering_partition_size = self.kernel_related_kwargs.get("graph_reordering_partition_size", -1)

        if graph_reordering_partition_size != -1:
            # perform graph reordering
            self.edge_index, self.edge_weight = reorder_graph(
                edge_index=self.edge_index,
                edge_weights=self.edge_weight,
                num_nodes=self.num_nodes,
                partition_size=graph_reordering_partition_size,
            )

        graph: Any = None
        if self.backend == "pyg":  # pyg eats standard edge index & weight
            if self.add_self_loops:
                self.edge_index, self.edge_weight = add_self_loops_pyg(self.edge_index, self.edge_weight)
            graph = (self._to_default_device(self.edge_index), self._to_default_device(self.edge_weight))
        elif self.backend == "dgl":
            graph = dgl_graph((self.edge_index[0], self.edge_index[1]), num_nodes=self.num_nodes)
            if self.edge_weight is not None:
                graph.edata["w"] = self.edge_weight
            if self.add_self_loops:
                graph = add_self_loop(graph)
            graph = self._to_default_device(graph)
        elif self.backend == "normalized_adj_mat_gcn":
            graph = normalize_adj(
                edge_index=self.edge_index, num_nodes=self.num_nodes, how="both", add_self_loops=self.add_self_loops
            ).to_sparse_csr()
            graph = self._to_default_device(graph)
        elif self.backend == "adj_mat_in_degree_normalized_transposed":
            graph = normalize_adj(
                edge_index=self.edge_index, num_nodes=self.num_nodes, how="right", add_self_loops=self.add_self_loops
            ).to_sparse_csr()
            graph = self._to_default_device(graph)
        elif self.backend == "cugraph":
            normalized_gcn_adjacency = normalize_adj(
                edge_index=self.edge_index, num_nodes=self.num_nodes, how="both", add_self_loops=self.add_self_loops
            )
            edge_index = normalized_gcn_adjacency.indices().tolist()
            edge_weights_for_gcn = normalized_gcn_adjacency.values()
            edge_index_for_pyg = EdgeIndex(
                edge_index,
                sparse_size=(self.num_nodes, self.num_nodes),
                sort_order="row",
                is_undirected=False,
                device=torch.get_default_device(),
            )
            csc_graph = get_cugraph_with_gcn_weights(edge_index_for_pyg)  # edge index is already on GPU
            graph = (csc_graph, edge_weights_for_gcn)
        elif self.backend == "adj_mat_transposed":
            graph = normalize_adj(
                edge_index=self.edge_index, num_nodes=self.num_nodes, how="none", add_self_loops=self.add_self_loops
            ).to_sparse_csr()
            graph = self._to_default_device(graph)
        elif self.backend == "adj_mat":
            graph = normalize_adj(
                edge_index=self.edge_index, num_nodes=self.num_nodes, how="none", add_self_loops=self.add_self_loops
            ).T.coalesce()  # TODO make it CSR and compatible with tests!
            graph = self._to_default_device(graph)
        elif self.backend == "coo":
            if self.add_self_loops:
                self.edge_index, self.edge_weight = add_self_loops_pyg(self.edge_index, self.edge_weight)
            graph = (
                self._to_default_device(self.edge_index),
                self._to_default_device(self.edge_weight),
                self.num_nodes,
            )
        elif self.backend == "csr":
            if self.add_self_loops:
                self.edge_index, self.edge_weight = add_self_loops_pyg(self.edge_index, self.edge_weight)

            row_ptr, cols, w, _ = build_csr_as_is(
                self.edge_index, self.edge_weight, num_nodes=self.num_nodes, do_transpose=True
            )

            # Store graph as (row_pointers, column_indices, edge_weight) on default device
            graph = (
                self._to_int32(self._to_default_device(row_ptr)),
                self._to_int32(self._to_default_device(cols)),
                self._to_int32(self._to_default_device(w)),
            )
        elif self.backend == "csr_and_csr_T_for_cusparse":
            if self.add_self_loops:
                self.edge_index, self.edge_weight = add_self_loops_pyg(self.edge_index, self.edge_weight)
            graph = []
            for do_transpose in [True, False]:
                row_ptr, cols, w, _ = build_csr_as_is(
                    self.edge_index, self.edge_weight, num_nodes=self.num_nodes, do_transpose=do_transpose
                )

                graph.extend(
                    [
                        self._to_int32(self._to_default_device(row_ptr)),
                        self._to_int32(self._to_default_device(cols)),
                        self._to_int32(self._to_default_device(w)),
                    ]
                )

        elif self.backend == "cuda":
            if self.add_self_loops:
                self.edge_index, self.edge_weight = add_self_loops_pyg(self.edge_index, self.edge_weight)

            row_ptr, cols, _w, counts = build_csr_as_is(
                self.edge_index,
                self.edge_weight,
                num_nodes=self.num_nodes,
                do_transpose=True,
            )

            deg = counts
            quantile = self.kernel_related_kwargs.get("huge_degree_threshold_quantile", -1)

            if quantile != -1:
                huge_degree_threshold_quantile = torch.quantile(deg.float(), quantile).item()
            else:
                huge_degree_threshold_quantile = max(counts) + 1

            light = (deg < huge_degree_threshold_quantile).nonzero(as_tuple=False).view(-1).to(torch.int32)
            heavy = (deg >= huge_degree_threshold_quantile).nonzero(as_tuple=False).view(-1).to(torch.int32)

            graph = (
                self._to_int32(self._to_default_device(row_ptr)),
                self._to_int32(self._to_default_device(cols)),
                self._to_int32(self._to_default_device(light)),
                self._to_int32(self._to_default_device(heavy)),
            )

        elif self.backend == "csc":
            ...  # TODO
        elif self.backend == "edge_list":
            edge_list = self.edge_index.T
            graph = (self._to_default_device(edge_list), self._to_default_device(self.edge_weight))
            # TODO: add self-loops
        elif self.backend == "tcgnn":
            row_pointer, col_indices, block_partition, edge_to_column, edge_to_row = to_tcgnn_data(
                self.edge_index, self.num_nodes, self.edge_weight
            )
            graph = (
                self._to_default_device(row_pointer),
                self._to_default_device(col_indices),
                self._to_default_device(block_partition),
                self._to_default_device(edge_to_column),
                self._to_default_device(edge_to_row),
            )
        elif self.backend == "weighted_sparse_block":
            adj_sparse_csr = (
                normalize_adj(
                    self.edge_index,
                    num_nodes=self.num_nodes,
                    how="both",  # TODO implement other normalization types for this backend
                    add_self_loops=self.add_self_loops,
                )
                .to_sparse_csr()
                .cpu()
            )

            graph = WSBFormat.build_wsb_format(adj=adj_sparse_csr).to(torch.get_default_device())
        elif self.backend == "csr_and_csr_transposed":
            if self.add_self_loops:
                self.edge_index, self.edge_weight = add_self_loops_pyg(self.edge_index, self.edge_weight)

            row_ptr_fwd, cols_fwd, _w, counts_fwd = build_csr_as_is(
                edge_index=self.edge_index,
                edge_weight=self.edge_weight,
                num_nodes=self.num_nodes,
                do_transpose=True,
            )

            row_ptr_bwd, cols_bwd, _w, counts_bwd = build_csr_as_is(
                edge_index=self.edge_index,
                edge_weight=self.edge_weight,
                num_nodes=self.num_nodes,
                do_transpose=False,
            )

            deg = counts_fwd
            quantile = self.kernel_related_kwargs.get("huge_degree_threshold_quantile", -1)

            if quantile != -1:
                huge_degree_threshold_quantile = torch.quantile(deg.float(), quantile).item()
            else:
                huge_degree_threshold_quantile = max(counts_fwd) + 1

            light = (deg < huge_degree_threshold_quantile).nonzero(as_tuple=False).view(-1)
            heavy = (deg >= huge_degree_threshold_quantile).nonzero(as_tuple=False).view(-1)

            graph = self._to_default_device(
                AdjacencyForwardBackwardWithNodeBuckets(
                    forward_indptr=self._to_int32(row_ptr_fwd),
                    forward_indices=self._to_int32(cols_fwd),
                    backward_indptr=self._to_int32(row_ptr_bwd),
                    backward_indices=self._to_int32(cols_bwd),
                    light_nodes=self._to_int32(light),
                    heavy_nodes=self._to_int32(heavy),
                )
            )

        elif self.backend == "cuda_weighted_sparse_block_with_meta":
            adj_sparse_csr = (
                normalize_adj(
                    self.edge_index,
                    num_nodes=self.num_nodes,
                    how="both",  # TODO implement other normalization types for this backend
                    add_self_loops=self.add_self_loops,
                )
                .to_sparse_csr()
                .cpu()
            )

            # TODO speedud construction via GPU-based operation
            graph = WSBFormat.build_wsb_format(adj=adj_sparse_csr).to(torch.get_default_device())
            # TODO add light & heavy vertices

        elif self.backend == "f3s":
            graph = f3s_preprocess(self.edge_index)

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

    def _to_int32(self, item: Any) -> Any:
        """If tensor, convert to int32"""
        try:
            item = item.to(torch.int32)
        except Exception:
            pass
        return item

    def edge_index_to_csr(self, edge_index: torch.Tensor, edge_weight: torch.Tensor | None, transposed: bool = True):
        if transposed:
            rows = edge_index[1]
            cols = edge_index[0]
        else:
            rows = edge_index[0]
            cols = edge_index[1]

        N = self.num_nodes

        # Sort edges by (row, col) for a canonical CSR
        perm = (rows * N + cols).argsort()
        rows = rows[perm]
        cols = cols[perm]
        w = edge_weight[perm] if edge_weight is not None else None

        # Build CSR row pointers
        counts = torch.bincount(rows, minlength=N)
        row_ptr = torch.zeros(N + 1, dtype=torch.long, device=rows.device)
        row_ptr[1:] = counts.cumsum(0)

        # Store graph as (row_pointers, column_indices, edge_weight) on default device
        graph = (
            self._to_int32(self._to_default_device(row_ptr)),
            self._to_int32(self._to_default_device(cols)),
            self._to_int32(self._to_default_device(w)),
        )
        return graph

    @property
    def num_nodes(self) -> int:
        """Number of nodes N."""
        return self.x.shape[0]  # type: ignore

    @property
    def num_features(self) -> int:
        """Feature dimensionality F."""
        return self.x.shape[1]  # type: ignore

    @property
    def num_classes(self) -> int:
        """Number of classes if labels are class indices or one-hot."""
        if self.y.ndim == 1 and self.y.numel() > 0:
            # class indices -> infer max+1
            return self.y.max().item() + 1  # type: ignore
        if self.y.ndim == 2:
            return self.y.shape[1]  # type: ignore
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


def _masks_from_indices(
    num_nodes: int, splits: Dict[str, torch.Tensor]
) -> Tuple[torch.BoolTensor, torch.BoolTensor, torch.BoolTensor]:
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


def _mask_from_indices_with_splits_creation(
    num_nodes: int,
) -> Tuple[torch.BoolTensor, torch.BoolTensor, torch.BoolTensor]:
    """Create boolean train/val/test masks by generating a random 60/20/20 split.

    This helper samples a random permutation of node indices in [0, num_nodes)
    and constructs split index tensors for train/val/test with ratios
    0.6 / 0.2 / 0.2. It then delegates to `_masks_from_indices` to convert
    those index tensors into boolean masks of shape [num_nodes].

    Args:
        num_nodes (int): Total number of nodes N in the single-graph dataset.

    Returns:
        Tuple[torch.BoolTensor, torch.BoolTensor, torch.BoolTensor]:
            (train_mask, val_mask, test_mask), each of shape [N] with dtype=bool.
    """
    perm = torch.randperm(num_nodes)
    n_train = int(0.6 * num_nodes)
    n_val = int(0.2 * num_nodes)
    splits = {
        "train": perm[:n_train],
        "val": perm[n_train : n_train + n_val],
        "test": perm[n_train + n_val :],
    }
    return _masks_from_indices(num_nodes, splits)


# ------------------------------- OGBN loaders -------------------------------- #


def load_ogbn(
    name: str,
    graph_backend: GraphBackendOption,
    root: str = "data",
    kernel_related_kwargs: dict[str, Any] = {},
) -> GraphSample:
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
        kernel_related_kwargs=kernel_related_kwargs,
    )


# ------------------------------- PyG loaders --------------------------------- #


def load_pyg_single_graph(
    name: str,
    graph_backend: GraphBackendOption,
    root: str = "data",
    allow_random_split: bool = False,
    kernel_related_kwargs: dict[str, Any] = {},
) -> GraphSample:
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
        elif name in (
            "hm-categories",
            "pokec-regions",
            "web-topics",
            "tolokers-2",
            "city-reviews",
            "artnet-exp",
            "web-fraud",
            "hm-prices",
            "avazu-ctr",
            "city-roads-M",
            "city-roads-L",
            "twitch-views",
            "artnet-views",
            "web-traffic",
        ):
            dset = GraphLandDataset(root=root, name=name, split="RL")
            data = dset[0]
        elif name in ("amazon-photo", "amazon-computers"):
            dset = Amazon(root=root, name=name.split("-")[1])
            data = dset[0]
        elif name in ("coauthor-cs", "coauthor-physics"):
            dset = Coauthor(root=root, name=name.split("-")[1])
            data = dset[0]
        else:
            try:
                cls = getattr(pyg_datasets, name)
                try:
                    dset = cls(root=root, name=name)
                except TypeError:
                    dset = cls(root=root)
                data = dset[0]
            except AttributeError:
                raise ValueError(f"Unknown PyG dataset '{name}'.")
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

    def _pick_split(mask: Optional[torch.Tensor], split_id: int = 0) -> Optional[torch.Tensor]:
        if mask is None:
            return None
        mask = mask.bool()
        if mask.ndim == 2:
            mask = mask[:, split_id]
        return mask

    split_id = 0
    train_mask = _pick_split(train_mask, split_id)
    val_mask = _pick_split(val_mask, split_id)
    test_mask = _pick_split(test_mask, split_id)

    if train_mask is None or val_mask is None or test_mask is None:
        if not allow_random_split:
            raise ValueError(
                f"Dataset '{name}' does not provide standard masks; "
                "set `allow_random_split: true` in your dataset YAML to auto-generate them."
            )
        print(f"Dataset '{name}' lacks masks -> creating random 60/20/20 split.")
        train_mask, val_mask, test_mask = _mask_from_indices_with_splits_creation(y.size(0))

    return GraphSample(
        x=x,
        y=y.long(),
        edge_index=edge_index,
        edge_weight=edge_weight,
        train_mask=train_mask.bool(),
        val_mask=val_mask.bool(),
        test_mask=test_mask.bool(),
        backend=graph_backend,
        kernel_related_kwargs=kernel_related_kwargs,
    )


# -------------------------------- DGL loaders -------------------------------- #


def load_dgl_single_graph(
    name: str, graph_backend: GraphBackendOption, root: str = "data", kernel_related_kwargs: dict[str, Any] = {}
) -> GraphSample:
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
            cls = None
            if hasattr(dgl_data, name):
                cls = getattr(dgl_data, name)
            else:
                for suffix in ("Dataset", "GraphDataset"):
                    cand = name + suffix
                    if hasattr(dgl_data, cand):
                        cls = getattr(dgl_data, cand)
                        break
            if cls is None:
                raise ValueError(f"Unknown DGL dataset '{name}'")

            dset = cls(raw_dir=root)
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
        kernel_related_kwargs=kernel_related_kwargs,
    )


# ------------------------------ Public factories ----------------------------- #


@dataclass
class DatasetConfig:
    """Configuration for selecting and loading a single-graph dataset.

    Attributes:
        source (str): 'ogbn' | 'pyg' | 'dgl' | 'auto'
        name (str): Dataset name (e.g., 'ogbn-arxiv', 'Cora', 'reddit').
        conv_backend (str): Conv backend type.
        root (str): Download/cache directory.
    """

    source: str
    name: str
    conv_backend: str
    root: str = "data"
    allow_random_split: bool = False
    kernel_related_kwargs: dict[str, Any] = field(default_factory=lambda: {})


def load_single_graph(cfg: DatasetConfig) -> GraphSample:
    """Load a canonical single-graph sample according to config.

    Args:
        cfg (DatasetConfig): Dataset configuration with source/name/root.

    Returns:
        GraphSample: Canonical large-graph sample.

    Raises:
        KeyError: If source is unsupported.
    """
    assert cfg.conv_backend in MODEL_BACKEND_TO_GRAPH_REPR, f"Unknown conv backend: {cfg.conv_backend}"
    graph_backend = MODEL_BACKEND_TO_GRAPH_REPR[cfg.conv_backend]

    kernel_related_kwargs = cfg.kernel_related_kwargs

    s = cfg.source.lower()
    if s == "ogbn":
        return load_ogbn(
            cfg.name, root=cfg.root, graph_backend=graph_backend, kernel_related_kwargs=kernel_related_kwargs
        )
    if s == "pyg":
        return load_pyg_single_graph(
            cfg.name,
            root=cfg.root,
            graph_backend=graph_backend,
            allow_random_split=getattr(cfg, "allow_random_split", False),
            kernel_related_kwargs=kernel_related_kwargs,
        )
    if s == "dgl":
        return load_dgl_single_graph(
            cfg.name, root=cfg.root, graph_backend=graph_backend, kernel_related_kwargs=kernel_related_kwargs
        )
    if s == "auto":
        # ogbn-* -> OGBN; else try PyG; then DGL.
        if cfg.name.lower().startswith("ogbn-"):
            return load_ogbn(
                cfg.name, root=cfg.root, graph_backend=graph_backend, kernel_related_kwargs=kernel_related_kwargs
            )
        try:
            return load_pyg_single_graph(
                cfg.name, root=cfg.root, graph_backend=graph_backend, kernel_related_kwargs=kernel_related_kwargs
            )
        except Exception:
            return load_dgl_single_graph(
                cfg.name, root=cfg.root, graph_backend=graph_backend, kernel_related_kwargs=kernel_related_kwargs
            )

    raise KeyError(f"Unsupported dataset source '{cfg.source}'")
