from typing import Any, Optional, Tuple, Union

import torch
from dgl import DGLGraph
from torch import Tensor

doc = """
DGL backend utilities (edge extraction, conversions).
"""


def extract_graph_edges(
    graph: DGLGraph | tuple[Tensor, Tensor | None, int | None],
) -> tuple[Tensor, Tensor | None, int]:
    """Extract edges and num_nodes from DGLGraph or tuple.

    Args:
        graph (Any): dgl.DGLGraph or (edge_index[, edge_weight, num_nodes])

    Returns:
        Tuple[torch.Tensor, Optional[torch.Tensor], int]:
            (edge_index [2, E], edge_weight [E] or None, num_nodes)
    """

    # dgl graph
    if hasattr(graph, "num_nodes") and hasattr(graph, "edges"):
        src, dst = graph.edges()
        edge_index = torch.vstack([src.long(), dst.long()])
        w = graph.edata["w"] if "w" in graph.edata else None  # type: ignore
        return edge_index, w, int(graph.num_nodes())
    # (edge_index, [edge_weight], [num_nodes])
    if isinstance(graph, (tuple, list)) and len(graph) in (1, 2, 3):
        edge_index = graph[0]
        edge_weight = graph[1] if len(graph) >= 2 else None
        num_nodes = (
            int(graph[2].item())  # type: ignore
            if (len(graph) == 3 and torch.is_tensor(graph[2]))
            else (int(graph[2]) if len(graph) == 3 else int(edge_index.max().item() + 1))  # type: ignore
        )
        return edge_index, edge_weight, num_nodes
    raise TypeError("Unsupported graph container for DGL backend")
