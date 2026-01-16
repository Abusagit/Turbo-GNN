import os

os.environ["CUDA_HOME"] = "/usr/local/cuda"
os.environ["CUDA_PATH"] = "/usr/local/cuda"
os.environ["PATH"] = f"/usr/local/cuda/bin:{os.environ['PATH']}"

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional, Tuple

import dgl
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.edge_index import EdgeIndex
from torch_geometric.utils import add_self_loops as add_self_loops_pyg

from src.utils.triton_constants import ROW_WINDOW_SIZE, TCB_SIZE, TCB_WIDTH

try:  # pragma: no cover
    LEGACY_MODE = False
    from pylibcugraphops.pytorch import CSC

    HAS_PYLIBCUGRAPHOPS = True
except ImportError:
    HAS_PYLIBCUGRAPHOPS = False
    try:  # pragma: no cover
        from pylibcugraphops import make_fg_csr

        LEGACY_MODE = True
    except ImportError:
        pass

from torch.utils.cpp_extension import load

repo_root_path = Path(__file__).parent.parent.parent

build_path = repo_root_path / "build/wsb"
if not build_path.is_dir():
    build_path.mkdir(parents=True)


wsb_cuda = load(
    name="wsb_cuda",
    sources=[__file__.replace("converters.py", "") + "wsb_format.cu"],
    extra_cflags=["-O3"],
    extra_cuda_cflags=[
        "-O3",
        "--use_fast_math",
        "--generate-line-info",
    ],
    extra_include_paths=[
        # *glob.glob(str(repo_root_path / ".venv/lib/python3.11/site-packages/**/include"), recursive=True),
        "/usr/local/cuda/include"
    ],
    verbose=True,
    build_directory=str(build_path),
)


doc = """
Graph format converters among edge list, CSR, and optional framework objects.

- to_csr_from_edge_list
- to_edge_list_from_csr
- to_pyg_data
- to_dgl_graph
"""

EdgeList = Tuple[torch.Tensor, Optional[torch.Tensor]]  # (edge_index [2,E], edge_weight [E] or None)
CSR = Tuple[
    torch.Tensor, torch.Tensor, Optional[torch.Tensor]
]  # (crow_indices [N+1], col_indices [E], values [E] or None)


def reorder_graph(
    edge_index: torch.Tensor,
    edge_weights: torch.Tensor | None,
    num_nodes: int,
    node_permute_algo="metis",
    partition_size=1024,
):
    graph = dgl.graph((edge_index[0], edge_index[1]), num_nodes=num_nodes)
    if edge_weights is not None:
        graph.edata["w"] = edge_weights

    graph_reordered = dgl.reorder_graph(
        graph, node_permute_algo=node_permute_algo, permute_config={"k": partition_size}
    )
    src, dst = graph_reordered.edges()
    new_edge_index = torch.vstack([src.long(), dst.long()])

    new_edge_weight = graph_reordered.edata["w"] if edge_weights is not None else None
    return new_edge_index, new_edge_weight


def to_csr_from_edge_list(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weight: Optional[torch.Tensor] = None,
) -> CSR:
    """Convert (edge_index, edge_weight) to CSR tensors.

    Args:
        edge_index (torch.Tensor): Long tensor of shape [2, E] with (row, col) indices.
        num_nodes (int): Number of nodes (CSR rows).
        edge_weight (Optional[torch.Tensor]): Optional edge weights of shape [E].

    Returns:
        CSR: Tuple (crow_indices [N+1], col_indices [E], values [E] or None).
    """
    if edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must be [2, E] long tensor")
    if edge_index.dtype != torch.long:
        edge_index = edge_index.long()

    order = torch.argsort(edge_index[0], stable=True)
    row_sorted = edge_index[0][order]
    col_sorted = edge_index[1][order]
    val_sorted = edge_weight[order] if edge_weight is not None else None

    counts = torch.bincount(row_sorted, minlength=num_nodes)
    crow = torch.zeros(num_nodes + 1, dtype=torch.long, device=edge_index.device)
    crow[1:] = torch.cumsum(counts, dim=0)
    return crow, col_sorted, val_sorted


def to_edge_list_from_csr(
    crow_indices: torch.Tensor,
    col_indices: torch.Tensor,
    values: Optional[torch.Tensor] = None,
) -> EdgeList:
    """Convert CSR tensors to (edge_index, edge_weight).

    Args:
        crow_indices (torch.Tensor): CSR row pointer [N+1].
        col_indices (torch.Tensor): CSR col indices [E].
        values (Optional[torch.Tensor]): Optional values [E].

    Returns:
        EdgeList: (edge_index [2, E], edge_weight [E] or None).
    """
    if crow_indices.ndim != 1:
        raise ValueError("crow_indices must be [N+1]")
    if col_indices.ndim != 1:
        raise ValueError("col_indices must be [E]")

    num_nodes = crow_indices.numel() - 1
    row = torch.repeat_interleave(torch.arange(num_nodes, device=crow_indices.device), crow_indices.diff())
    edge_index = torch.vstack([row.long(), col_indices.long()])
    return edge_index, values


def to_pyg_data(
    edge_index: torch.Tensor,
    x: torch.Tensor,
    y: Optional[torch.Tensor] = None,
    edge_weight: Optional[torch.Tensor] = None,
) -> Any:
    """Create a PyG `Data` object lazily.

    Args:
        edge_index (torch.Tensor): [2, E] long.
        x (torch.Tensor): Node features [N, F].
        y (Optional[torch.Tensor]): Labels [N] or [N, C].
        edge_weight (Optional[torch.Tensor]): Edge weights [E].

    Returns:
        Any: torch_geometric.data.Data instance.

    Raises:
        ImportError: If PyG is not installed.
    """
    return Data(x=x, edge_index=edge_index, y=y, edge_weight=edge_weight)


def to_dgl_graph(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weight: Optional[torch.Tensor] = None,
) -> Any:
    """Create a DGLGraph lazily.

    Args:
        edge_index (torch.Tensor): [2, E] long.
        num_nodes (int): Number of nodes.
        edge_weight (Optional[torch.Tensor]): Edge weights [E].

    Returns:
        Any: dgl.DGLGraph instance.

    Raises:
        ImportError: If DGL is not installed.
    """
    try:
        import dgl
    except Exception as exc:
        raise ImportError("DGL is required for to_dgl_graph()") from exc

    g = dgl.graph((edge_index[0], edge_index[1]), num_nodes=num_nodes)
    if edge_weight is not None:
        g.edata["w"] = edge_weight
    return g


def to_tcgnn_data(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weight: Optional[torch.Tensor] = None,
) -> Any:
    """Create a TC-GNN `Data` object lazily.

    Args:
        edge_index (torch.Tensor): [2, E] long.
        num_nodes (int): Number of nodes.
        edge_weight (Optional[torch.Tensor]): Edge weights [E].

    Returns:
        Any: tcgnn.Data instance.
    """

    try:
        import TCGNN
    except Exception as exc:
        raise ImportError("TC-GNN is required for to_tcgnn_data()") from exc

    row_pointer, col_indices, values = to_csr_from_edge_list(edge_index, num_nodes, edge_weight)
    BLK_H = 16
    BLK_W = 8

    num_row_windows = (num_nodes + BLK_H - 1) // BLK_H
    block_partition = torch.zeros(num_row_windows, dtype=torch.int).cpu()
    edge_to_column = torch.zeros(edge_index.size(1), dtype=torch.int).cpu()
    edge_to_row = torch.zeros(edge_index.size(1), dtype=torch.int).cpu()
    col_indices = col_indices.to(torch.int).cpu()
    row_pointer = row_pointer.to(torch.int).cpu()

    TCGNN.preprocess(
        col_indices.cpu(), row_pointer.cpu(), num_nodes, BLK_H, BLK_W, block_partition, edge_to_column, edge_to_row
    )
    return row_pointer, col_indices, block_partition, edge_to_column, edge_to_row


def splot_by_rows(
    src_indices: torch.Tensor, dst_indices: torch.Tensor, row_size: int
) -> list[tuple[int, torch.Tensor, torch.Tensor]]:
    """Split the edge index by block rows.

    Args:
        src_indices (torch.Tensor): [E] long.
        dst_indices (torch.Tensor): [E] long.
        row_size (int): Row size.

    Returns:
        list[tuple[int, torch.Tensor, torch.Tensor]]: List of (row_id, src_indices, dst_indices).
    """
    splitted = src_indices.clone() // row_size
    boundaries = torch.cat([torch.tensor([True], device=src_indices.device), splitted[1:] != splitted[:-1]])
    idx = boundaries.nonzero(as_tuple=True)[0]
    idx = torch.cat([idx, torch.tensor([len(splitted)], device=src_indices.device)])
    return [
        (splitted[idx[i]], src_indices[idx[i] : idx[i + 1]], dst_indices[idx[i] : idx[i + 1]])
        for i in range(len(idx) - 1)
    ]


def non_zero_column_ids(
    src_indices_block: torch.Tensor,
    dst_indices_block: torch.Tensor,
    num_nodes: int,
    row_index: int,
    block_row_size: int,
) -> torch.Tensor:
    """Calculate the column remapping for a block of edges.

    Args:
        src_indices_block (torch.Tensor): [E] long.
        dst_indices_block (torch.Tensor): [E] long.
        num_nodes (int): Number of nodes.

    Returns:
        torch.Tensor: Column remapping.
    """

    row_start = row_index * block_row_size
    src_indices_block = src_indices_block.clone() - row_start
    coordinates = src_indices_block * num_nodes + dst_indices_block
    column_index = coordinates / block_row_size

    column_remapping = torch.unique(column_index)
    return column_remapping


def to_dense_matrix(
    src_indices: torch.Tensor, dst_indices: torch.Tensor, row_index: int, num_nodes: int, block_row_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert CSR to dense matrix.

    Args:
        src_indices (torch.Tensor): [E] long.
        dst_indices (torch.Tensor): [E] long.
        row_index (int): Row index.
        num_nodes (int): Number of nodes.
        block_row_size (int): Block row size.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Dense matrix, column remapping.
    """
    non_zero_ids = non_zero_column_ids(src_indices, dst_indices, num_nodes, row_index, block_row_size)
    dense_shape = (block_row_size, non_zero_ids.shape[0])
    dense = torch.zeros(dense_shape, device=src_indices.device).view(-1)
    index_unwrapped = (src_indices - src_indices.min()) * num_nodes + dst_indices
    dense.scatter_(0, index_unwrapped, 1)
    return dense.view(block_row_size, non_zero_ids.shape[0]), non_zero_ids


def to_block_sparse_matrix(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weight: Optional[torch.Tensor] = None,
    block_row_size: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a block sparse matrix lazily.

    Args:
        edge_index (torch.Tensor): [2, E] long.
        num_nodes (int): Number of nodes.
        edge_weight (Optional[torch.Tensor]): Edge weights [E].
        block_row_size (int): Block row size.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Row pointer, column indices, values.
    """

    src_indices, dst_indices = edge_index[0], edge_index[1]
    blocks = splot_by_rows(src_indices, dst_indices, block_row_size)

    row_block_ids = torch.zeros(block_row_size, device=src_indices.device, dtype=torch.long)

    dense_blocks = []
    column_remappings = []

    for row_id, src_indices_block, dst_indices_block in blocks:
        dense_block, column_remapping = to_dense_matrix(
            src_indices_block, dst_indices_block, row_id, num_nodes, block_row_size
        )
        row_block_ids[row_id] = row_id
        dense_blocks.append(dense_block)
        column_remappings.append(column_remapping)

    dense_block = torch.cat(dense_blocks, dim=1)
    column_remapping = torch.cat(column_remappings, dim=0)

    return row_block_ids, dense_block, column_remapping


def get_block_sparse_for_cuda_backend(
    edge_index: torch.Tensor,
    num_nodes: int,
    how: Literal["left", "right", "both", "none"],
    add_self_loops: bool = True,
    deg_huge_thr: int = 128,
):
    """Compute block-sparse format used by our CUDA backend. Additionally,
        bin nodes into two categories based on the hyperparam:
    - mid_nodes - nodes with degree less then deg_huge threshold
    - huge_nodes - nodes with degree more then deg_huge threshold

    Args:
        edge_index (torch.Tensor): edge index in COO format
        num_nodes (int): number of nodes in a graph
        how (Literal[&quot;left&quot;, &quot;right&quot;, &quot;both&quot;, &quot;none&quot;]):
            type of normalization in adjacency matrix
        add_self_loops: torch.Tensor, num_nodes: int,
        how: Literal["left", "right", "both", "none"],
        add_self_loops: bool
            whether to add self_loops or not
        deg_huge_thr: int - threshold for binning nodes into mid and huge category
    """
    pass


def normalize_adj(
    edge_index: torch.Tensor, num_nodes: int, how: Literal["left", "right", "both", "none"], add_self_loops: bool = True
) -> torch.Tensor:
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
        adj_t = torch.sparse_coo_tensor(adj_t_indices, values, (num_nodes, num_nodes)).coalesce()

        in_degrees = torch.zeros(num_nodes, device=device)
        in_degrees.scatter_add_(0, dst, torch.ones_like(dst, dtype=torch.float32))

        # handle isolated nodes (in_degree = 0) by setting to 1 to avoid division by zero
        in_degrees = in_degrees.clamp(min=1.0)

        in_degree_inv = 1.0 / in_degrees
        diag_indices = torch.arange(num_nodes, device=device).unsqueeze(0).repeat(2, 1)
        in_degree_inv_diag = torch.sparse_coo_tensor(diag_indices, in_degree_inv, (num_nodes, num_nodes)).coalesce()

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
        adj_t = torch.sparse_coo_tensor(adj_t_indices, values, (num_nodes, num_nodes)).coalesce()

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
        raise ValueError(f"'edge_index' needs to be of type 'EdgeIndex' (got {type(edge_index)})")

    edge_index = edge_index.sort_by("col")[0]
    num_src_nodes = edge_index.get_sparse_size(0)
    (colptr, row), _ = edge_index.get_csc()

    if not row.is_cuda:
        raise RuntimeError("'get_cugraph' requires GPU-based processing (got CPU tensor)")

    if LEGACY_MODE:
        return make_fg_csr(colptr, row)

    return CSC(colptr, row, num_src_nodes=num_src_nodes)


def build_csr_as_is(
    edge_index: torch.Tensor,
    edge_weight: Optional[torch.Tensor],
    num_nodes: int,
    do_transpose: bool = False,
):
    """
    Build CSR exactly as in current code (diff = 0 in logic):
      rows = edge_index[1], cols = edge_index[0]
      perm = (rows * N + cols).argsort()
      counts = bincount(rows)
      row_ptr[1:] = counts.cumsum(0)

    Returns:
        row_ptr, cols, w, counts
    """

    if do_transpose:
        rows = edge_index[1]
        cols = edge_index[0]
    else:
        rows = edge_index[0]
        cols = edge_index[1]

    N = num_nodes

    # sort edges by (row, col) for a canonical CSR
    perm = (rows * N + cols).argsort()
    rows = rows[perm]
    cols = cols[perm]
    w = edge_weight[perm] if edge_weight is not None else None

    # build CSR row pointers
    counts = torch.bincount(rows, minlength=N)
    row_ptr = torch.zeros(N + 1, dtype=torch.long, device=rows.device)
    row_ptr[1:] = counts.cumsum(0)

    return row_ptr, cols, w, counts


@dataclass
class AdjacencyForwardBackwardCSR:
    """
    Dataclass containing adjacency matrix for forward and backward pass
    If matrix is symmetric (graph is undirected),
    `adj_mat_csr_backward` points to the same tensor as `adj_mat_csr_forward`.
        (When `to` is called, this doesn't hold)
    """

    adj_mat_csr_forward: torch.Tensor
    adj_mat_csr_backward: torch.Tensor

    _device: torch.device = torch.device("cpu")

    def __post__init__(self):
        self._device = self.adj_mat_csr_forward.device

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, device) -> "AdjacencyForwardBackwardCSR":
        adj_mat_csr_forward_device = self.adj_mat_csr_forward.to(device)
        if id(self.adj_mat_csr_forward) == id(self.adj_mat_csr_backward):
            adj_mat_csr_backward_device = adj_mat_csr_forward_device
        else:
            adj_mat_csr_backward_device = self.adj_mat_csr_backward.to(device)

        self.adj_mat_csr_forward = adj_mat_csr_forward_device
        self.adj_mat_csr_backward = adj_mat_csr_backward_device
        torch.cuda.empty_cache()
        self._device = device
        return self


@dataclass
class AdjacencyForwardBackwardWithNodeBuckets:
    forward_indptr: torch.Tensor
    forward_indices: torch.Tensor
    backward_indptr: torch.Tensor
    backward_indices: torch.Tensor
    light_nodes: torch.Tensor
    heavy_nodes: torch.Tensor

    _device: torch.device = torch.device("cpu")

    def __post__init__(self):
        self._device = self.forward_indptr.device

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, device) -> "AdjacencyForwardBackwardWithNodeBuckets":
        self.forward_indptr = self.forward_indptr.to(device)
        self.forward_indices = self.forward_indices.to(device)
        self.backward_indptr = self.backward_indptr.to(device)
        self.backward_indices = self.backward_indices.to(device)
        self.light_nodes = self.light_nodes.to(device)
        self.heavy_nodes = self.heavy_nodes.to(device)
        torch.cuda.empty_cache()

        return self


@dataclass
class WSBFormat:
    """WSB format tensors for CUDA kernel"""

    tcb_row_offset: torch.Tensor  # [num_row_windows + 1], int32
    col_idx: torch.Tensor  # [num_tcbs * 8], int32
    bitmap: torch.Tensor  # [num_tcbs * 2], int64 (uint64)
    weights: torch.Tensor  # [num_tcbs * 128], float16

    num_nodes: int
    num_edges: int
    num_row_windows: int
    num_tcbs: int

    adjacency_matrices_meta: AdjacencyForwardBackwardCSR
    light_nodes: torch.Tensor | None = None
    heavy_nodes: torch.Tensor | None = None

    def to(self, device: str | torch.device) -> "WSBFormat":
        """Move tensors to device"""
        return WSBFormat(
            tcb_row_offset=self.tcb_row_offset.to(device),
            col_idx=self.col_idx.to(device),
            bitmap=self.bitmap.to(device),
            weights=self.weights.to(device),
            num_nodes=self.num_nodes,
            num_edges=self.num_edges,
            num_row_windows=self.num_row_windows,
            num_tcbs=self.num_tcbs,
            adjacency_matrices_meta=self.adjacency_matrices_meta.to(device),
            light_nodes=self.light_nodes,
            heavy_nodes=self.heavy_nodes,
        )

    def cuda(self) -> "WSBFormat":
        return self.to("cuda")

    def memory_bytes(self) -> int:
        """Total memory footprint"""
        return self.tcb_row_offset.nbytes + self.col_idx.nbytes + self.bitmap.nbytes + self.weights.nbytes  # type: ignore

    def __repr__(self) -> str:
        return (
            f"WSBFormat(nodes={self.num_nodes}, edges={self.num_edges}, "
            f"row_windows={self.num_row_windows}, tcbs={self.num_tcbs}, "
            f"memory={self.memory_bytes() / 1024:.1f} KB)"
        )

    def to_dense(self) -> torch.Tensor:
        """
        Convert WSB format back to dense matrix for verification.

        Returns:
        [N, N] dense adjacency matrix
        """
        N = self.num_nodes
        dense = np.zeros((N, N), dtype=np.float32)

        tcb_row_offset = self.tcb_row_offset.numpy()
        col_idx = self.col_idx.numpy()
        bitmap = self.bitmap.numpy().view(np.uint64)
        weights = self.weights.float().numpy()

        for rw in range(self.num_row_windows):
            row_start = rw * ROW_WINDOW_SIZE
            tcb_start = tcb_row_offset[rw]
            tcb_end = tcb_row_offset[rw + 1]

            for tcb_idx in range(tcb_start, tcb_end):
                # get columns for this TCB
                cols = col_idx[tcb_idx * TCB_WIDTH : (tcb_idx + 1) * TCB_WIDTH]

                # get bitmap
                bm_lo = bitmap[tcb_idx * 2]
                bm_hi = bitmap[tcb_idx * 2 + 1]

                # get weights
                tcb_weights = weights[tcb_idx * TCB_SIZE : (tcb_idx + 1) * TCB_SIZE]

                # decode
                for local_row in range(ROW_WINDOW_SIZE):
                    global_row = row_start + local_row
                    if global_row >= N:
                        break

                    for local_col in range(TCB_WIDTH):
                        bit_pos = (local_row % 8) * TCB_WIDTH + local_col
                        bm = bm_lo if local_row < 8 else bm_hi

                        if bm & (np.uint64(1) << np.uint64(bit_pos)):
                            weight_idx = local_row * TCB_WIDTH + local_col
                            global_col = cols[local_col]
                            dense[global_row, global_col] = tcb_weights[weight_idx]

        return torch.from_numpy(dense)

    @classmethod
    def build_wsb_format(cls, adj: torch.Tensor, dtype: torch.dtype = torch.float16) -> "WSBFormat":
        """
        NOTE this is a prototype and it works very slowly on large graphs - subject for optimization,
        e.g. embarrasingly parallel approach

        Build WSB format from torch.sparse CSR tensor.

        Algorithm:
        1. Divide nodes into row windows of 16 rows each
        2. For each row window:
        a. Collect all unique column indices from the 16 rows
        b. Sort and partition into TCBs of 8 columns each
        c. For each TCB, build bitmap and weight array

        Args:
            adj: sparse СSR tensor of adjacency matrix
            dtype: Weight dtype (e.g. float16 for tensor cores)

        Returns:
            WSBFormat with all tensors ready for CUDA kernel
        """

        # N = adj.shape[0]
        # assert adj.shape[0] == adj.shape[1], "Adjacency must be square"

        # indptr = adj.crow_indices()
        # indices = adj.col_indices()
        # weights = adj.values()

        # num_row_windows = (N + ROW_WINDOW_SIZE - 1) // ROW_WINDOW_SIZE
        # num_edges = len(indices)

        # tcb_row_offset = [0]
        # all_col_idx = []  # 8 columns per TCB
        # all_bitmaps = []  # 2 uint64 per TCB
        # all_weights = []  # 128 floats per TCB

        # # process each row window
        # for rw in range(num_row_windows):
        #     row_start = rw * ROW_WINDOW_SIZE
        #     row_end = min(row_start + ROW_WINDOW_SIZE, N)
        #     num_rows_in_window = row_end - row_start

        #     # collect all (local_row, col, weight) for this row window
        #     edges_in_window = []
        #     for local_row in range(num_rows_in_window):
        #         global_row = row_start + local_row
        #         for idx in range(indptr[global_row], indptr[global_row + 1]):
        #             col = indices[idx].item()
        #             w = weights[idx].item()
        #             # print(f"{rw=} {idx=} {global_row=} {col=} {w=}")
        #             edges_in_window.append((local_row, col, w))

        #     if len(edges_in_window) == 0:
        #         tcb_row_offset.append(tcb_row_offset[-1])
        #         continue

        #     # get unique columns and sort them
        #     unique_cols = sorted({e[1] for e in edges_in_window})
        #     num_unique_cols = len(unique_cols)

        #     # column -> local index mapping
        #     col_to_local = {c: i for i, c in enumerate(unique_cols)}

        #     # edge lookup: (local_row, local_col) -> weight
        #     edge_map = {}
        #     for local_row, col, w in edges_in_window:
        #         local_col = col_to_local[col]
        #         edge_map[(local_row, local_col)] = w

        #     # number of TCBs for this row window
        #     num_tcbs_in_rw = (num_unique_cols + TCB_WIDTH - 1) // TCB_WIDTH

        #     # process each TCB
        #     for tcb_idx in range(num_tcbs_in_rw):
        #         col_start = tcb_idx * TCB_WIDTH
        #         col_end = min(col_start + TCB_WIDTH, num_unique_cols)

        #         # column indices for this TCB (pad with 0 if fewer than 8)
        #         tcb_cols = unique_cols[col_start:col_end]
        #         while len(tcb_cols) < TCB_WIDTH:
        #             tcb_cols.append(0)  # padding
        #         all_col_idx.extend(tcb_cols)

        #         # build bitmap and weights for this TCB
        #         # bitmap layout: bits 0-63 for rows 0-7 (first uint64)
        #         #                bits 0-63 for rows 8-15 (second uint64)
        #         # within each uint64: bit = row * 8 + col_in_tcb
        #         bitmap_lo = np.uint64(0)  # Rows 0-7
        #         bitmap_hi = np.uint64(0)  # Rows 8-15
        #         tcb_weights = np.zeros(TCB_SIZE, dtype=np.float32)

        #         for local_row in range(ROW_WINDOW_SIZE):
        #             for local_col_in_tcb in range(TCB_WIDTH):
        #                 global_local_col = col_start + local_col_in_tcb

        #                 if global_local_col >= num_unique_cols:
        #                     continue

        #                 key = (local_row, global_local_col)
        #                 if key in edge_map:
        #                     # set bit in bitmap
        #                     bit_pos = (local_row % 8) * TCB_WIDTH + local_col_in_tcb
        #                     if local_row < 8:
        #                         bitmap_lo |= np.uint64(1) << np.uint64(bit_pos)
        #                     else:
        #                         bitmap_hi |= np.uint64(1) << np.uint64(bit_pos)

        #                     # store weight (row-major within TCB)
        #                     weight_idx = local_row * TCB_WIDTH + local_col_in_tcb
        #                     tcb_weights[weight_idx] = edge_map[key]

        #         all_bitmaps.extend([bitmap_lo, bitmap_hi])
        #         all_weights.extend(tcb_weights.tolist())

        #     tcb_row_offset.append(tcb_row_offset[-1] + num_tcbs_in_rw)

        # # convert to tensors
        # num_tcbs = tcb_row_offset[-1]
        # bitmap_array = np.array(all_bitmaps, dtype=np.uint64)
        # bitmap_tensor = torch.from_numpy(bitmap_array.view(np.int64)).clone()

        # return cls(
        #     tcb_row_offset=torch.tensor(tcb_row_offset, dtype=torch.int32),
        #     col_idx=torch.tensor(all_col_idx, dtype=torch.int32),
        #     bitmap=bitmap_tensor,
        #     weights=torch.tensor(all_weights, dtype=dtype),
        #     num_nodes=N,
        #     num_edges=num_edges,
        #     num_row_windows=num_row_windows,
        #     num_tcbs=num_tcbs,
        #     adjacency_matrices_meta=AdjacencyForwardBackwardCSR(
        #         adj_mat_csr_forward=adj,
        #         adj_mat_csr_backward=adj.to_sparse_coo().T.to_sparse_csr().to(adj.device),
        #     ),
        #     light_nodes=...,
        #     heavy_nodes=...,
        # )

        # TODO ADD LIGHT-HEAVY PARTITION!

        N = adj.shape[0]
        assert adj.shape[0] == adj.shape[1], "Adjacency must be square"

        num_row_windows = (N + ROW_WINDOW_SIZE - 1) // ROW_WINDOW_SIZE
        num_edges = adj._nnz()

        tcb_row_offset, col_idx, bitmap, weights = wsb_cuda.build_wsb_format_cpu(adj, dtype)

        num_tcbs = tcb_row_offset[-1].item()
        return cls(
            tcb_row_offset=tcb_row_offset,
            col_idx=col_idx,
            bitmap=bitmap,
            weights=weights,
            num_nodes=N,
            num_edges=num_edges,
            num_row_windows=num_row_windows,
            num_tcbs=num_tcbs,
            adjacency_matrices_meta=AdjacencyForwardBackwardCSR(
                adj_mat_csr_forward=adj,
                adj_mat_csr_backward=adj.to_sparse_coo().T.to_sparse_csr(),
            ),
            light_nodes=...,
            heavy_nodes=...,
        )
