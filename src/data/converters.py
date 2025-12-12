from typing import Any, Optional, Tuple

import torch
from torch_geometric.data import Data

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
