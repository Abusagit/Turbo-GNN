from typing import Any, Optional, Tuple

import torch

doc = """
PyG backend utilities (edge extraction, sanity checks).
"""


def extract_edge_index(graph: Any) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Extract (edge_index, edge_weight) from a variety of containers.

    Args:
        graph (Any): Either torch_geometric.data.Data or a tuple (edge_index[, edge_weight]).

    Returns:
        Tuple[torch.Tensor, Optional[torch.Tensor]]: edge_index [2,E], edge_weight [E] or None.
    """
    # PyG Data
    if hasattr(graph, "edge_index"):
        edge_index = graph.edge_index
        edge_weight = getattr(graph, "edge_weight", None)
        return edge_index, edge_weight
    # (edge_index, edge_weight) tuple
    if isinstance(graph, (tuple, list)) and len(graph) in (1, 2):
        edge_index = graph[0]
        edge_weight = graph[1] if len(graph) == 2 else None
        return edge_index, edge_weight
    raise TypeError("Unsupported graph container for PyG backend")
