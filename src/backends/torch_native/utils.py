import torch

from typing import Literal

doc = """
Utilities for the Torch-native backend using edge-index/CSR and sparse ops.
"""


def normalize_adj(edge_index: torch.Tensor, num_nodes: int, how: Literal["left", "right", "both"],
                  add_self_loops: bool = True) -> torch.Tensor:
    """Compute symmetric normalized adjacency (A_hat) as sparse COO.

    Args:
        edge_index (torch.Tensor): [2, E] long tensor.
        num_nodes (int): Number of nodes.

    Returns:
        torch.Tensor: Sparse COO adjacency with added self-loops and:
            - D^{-1/2} A D^{-1/2} normalization if `how` == "both".
            - ...
            - ...
    """
    device = edge_index.device
    idx = edge_index

    if add_self_loops:
        self_loops = torch.arange(num_nodes, device=device)
        loop_idx = torch.stack([self_loops, self_loops], dim=0)
        idx = torch.cat([idx, loop_idx], dim=1)

    if how == "both":
        # add self loops
        values = torch.ones(idx.size(1), device=device)
        adj = torch.sparse_coo_tensor(idx, values, (num_nodes, num_nodes))

        deg = torch.sparse.sum(adj, dim=1).to_dense()
        deg_inv_sqrt = torch.pow(deg.clamp(min=1.0), -0.5)
        D_inv_sqrt = deg_inv_sqrt
        row, col = idx
        norm_vals = D_inv_sqrt[row] * values * D_inv_sqrt[col]

    return torch.sparse_coo_tensor(idx, norm_vals, (num_nodes, num_nodes)).coalesce()
