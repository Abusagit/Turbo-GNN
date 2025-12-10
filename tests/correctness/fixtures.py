import sys
from pathlib import Path

import pytest
import torch
from torch_geometric.utils import add_self_loops

import src.backends.dgl_backend  # noqa: F401
import src.backends.pyg_backend  # noqa: F401
import src.backends.torch_native_backend  # noqa: F401
from src.backends.registry import BackendRegistry
from src.data.datasets import MODEL_BACKEND_TO_GRAPH_REPR, GraphSample

"""
Pytest configuration and shared fixtures for GNN backend tests.
"""


@pytest.fixture(scope="session")
def device():
    """Get the device to use for testing (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def set_default_device(device):
    """Set the default device for the test session."""
    torch.set_default_device(device)
    yield device
    torch.set_default_device("cpu")  # Reset after tests


@pytest.fixture
def small_graph_data(device):
    """
    Create a simple star graph for testing.

    Structure: Node 0 is the center, connected to nodes 1-9.
    This makes it easy to verify mean aggregation:
    - Node 0 should receive messages from nodes 1-9
    - Other nodes have no incoming edges (except self-loops if added)

    Returns:
        dict: Contains num_nodes, edge_index, features, and expected results
    """
    num_nodes = 10
    in_channels = 16

    # Star graph: nodes 1-9 -> node 0
    src = torch.arange(1, num_nodes, device=device)
    dst = torch.zeros(num_nodes - 1, dtype=torch.long, device=device)
    edge_index = torch.stack([src, dst], dim=0)

    # Node i has feature value i everywhere (for easy verification)
    features = torch.arange(num_nodes, device=device, dtype=torch.float32).unsqueeze(1).repeat(1, in_channels)

    # Expected mean for center node (receives from nodes 1-9)
    expected_center_mean = torch.arange(1, num_nodes, device=device, dtype=torch.float32).mean()
    expected_min = torch.zeros((num_nodes, in_channels), device=device, dtype=torch.float32)
    expected_min[0, :] = 1
    expected_max = torch.zeros((num_nodes, in_channels), device=device, dtype=torch.float32)
    expected_max[0, :] = num_nodes - 1

    return {
        "num_nodes": num_nodes,
        "edge_index": edge_index,
        "features": features,
        "in_channels": in_channels,
        "expected_center_mean": expected_center_mean,
        "expected_min": expected_min,
        "expected_max": expected_max,
        "device": device,
    }


@pytest.fixture
def fully_connected_on_3_vertices_data(device):
    """
    Create a simple fully connected graph (Kn) for testing.

    Structure: Nodes 0, 1, ... , n are all interconnected and pass messages to each other

    Returns:
        dict: Contains num_nodes, edge_index, features, and expected results
    """

    num_nodes = 3
    in_channels = 16

    row, col = torch.meshgrid(
        torch.arange(num_nodes, device=device), torch.arange(num_nodes, device=device), indexing="ij"
    )

    mask = row != col
    edge_index = torch.stack([row[mask], col[mask]], dim=0)

    features = torch.arange(num_nodes, device=device, dtype=torch.float32).unsqueeze(1).repeat(1, in_channels)

    expected_min = torch.zeros((num_nodes, in_channels), device=device, dtype=torch.float32)
    expected_min[0, :] = 1
    expected_max = torch.ones((num_nodes, in_channels), device=device, dtype=torch.float32) * (num_nodes - 1)
    expected_max[num_nodes - 1, :] = num_nodes - 2

    return {
        "num_nodes": num_nodes,
        "edge_index": edge_index,
        "features": features,
        "in_channels": in_channels,
        "expected_min": expected_min,
        "expected_max": expected_max,
        "device": device,
    }


@pytest.fixture
def empty_graph_data(device):
    """
    Create an empty graph for crash testing.

    Structure: No nodes. No edges.

    Returns:
        dict: Contains num_nodes, edge_index, features, and expected results
    """

    num_nodes = 0
    in_channels = 16

    edge_index = (
        [torch.tensor([], device=device, dtype=torch.float32)],
        [torch.tensor([], device=device, dtype=torch.float32)],
    )

    features = torch.empty((num_nodes, in_channels), device=device, dtype=torch.float32)

    expected_min = torch.tensor([], device=device, dtype=torch.float32)
    expected_max = torch.tensor([], device=device, dtype=torch.float32)

    return {
        "num_nodes": num_nodes,
        "edge_index": edge_index,
        "features": features,
        "in_channels": in_channels,
        "expected_min": expected_min,
        "expected_max": expected_max,
        "device": device,
    }


@pytest.fixture
def connectivity_component_and_isolated_vertice_data(device):
    """
    Create a simple graph to test nothing leaks to isolated vertices

    Structure: Nodes 0, 1 are bidirectionally connected to each other, node 2 is isolated.

    Returns:
        dict: Contains num_nodes, edge_index, features, and expected results
    """

    num_nodes = 3
    in_channels = 16

    # Star graph: nodes 1-9 -> node 0
    src = torch.tensor([0, 1], dtype=torch.long, device=device)
    dst = torch.tensor([1, 0], dtype=torch.long, device=device)
    edge_index = torch.stack([src, dst], dim=0)

    # Node i has feature value i everywhere (for easy verification)
    features = torch.arange(1, num_nodes + 1, device=device, dtype=torch.float32).unsqueeze(1).repeat(1, in_channels)

    # Expected mean for center node (receives from nodes 1-9)
    expected_min = torch.tensor([2.0, 1.0, 0.0], device=device, dtype=torch.float32).unsqueeze(1).repeat(1, in_channels)
    expected_max = expected_min

    return {
        "num_nodes": num_nodes,
        "edge_index": edge_index,
        "features": features,
        "in_channels": in_channels,
        "expected_min": expected_min,
        "expected_max": expected_max,
        "device": device,
    }


@pytest.fixture
def small_undirected_cylce_data(device):
    """
    Create a simple cycle graph for testing.

    Returns:
        dict: Contains num_nodes, edge_index, features
    """
    num_nodes = 5
    in_channels = 2

    src1 = torch.arange(0, num_nodes, device=device)
    dst1 = (src1 + 1) % num_nodes
    src2 = src1.clone()
    dst2 = (src2 + num_nodes - 1) % num_nodes
    edge_index = torch.stack([torch.concat((src1, src2)), torch.concat((dst1, dst2))], dim=0)

    # Node i has feature value i everywhere (for easy verification)
    features = torch.arange(num_nodes, device=device, dtype=torch.float32).unsqueeze(1).repeat(1, in_channels)

    return {
        "num_nodes": num_nodes,
        "edge_index": edge_index,
        "features": features,
        "in_channels": in_channels,
        "device": device,
    }


@pytest.fixture
def random_graph_data(device):
    """
    Create a random graph for general testing.

    Returns:
        dict: Contains num_nodes, num_edges, edge_index, and features
    """
    num_nodes = 100
    num_edges = 500
    in_channels = 32

    edge_index = torch.randint(0, num_nodes, (2, num_edges), device=device)
    features = torch.randn(num_nodes, in_channels, device=device)

    return {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "edge_index": edge_index,
        "features": features,
        "in_channels": in_channels,
        "device": device,
    }


@pytest.fixture
def simple_graph_data(device):
    """Simple graph with easily-traceable conections

    Returns:
        dict: Contains num_nodes, num_edges, edge_index, and features
    """

    num_nodes = 3
    in_channels = 64
    edge_index = torch.tensor([[0, 0, 1, 2, 1, 2], [1, 2, 2, 0, 0, 1]]).to(device)

    num_edges = edge_index.shape[1]

    features = (torch.arange(1, num_nodes + 1)[:, None] * torch.eye(num_nodes, in_channels)).to(device)

    return {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "edge_index": edge_index,
        "features": features,
        "in_channels": in_channels,
        "device": device,
    }


@pytest.fixture
def karate_like_club_graph(device):
    """
    Create Zachary's Karate Club graph (classic small social network).
    34 nodes, 78 edges (undirected, so 156 directed edges)
    + added self-loops for batter consistency with GraphSample dataclass

    Returns:
        dict: Contains num_nodes, edge_index, and features
    """
    # Karate club edges (undirected)
    edges = [
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (0, 5),
        (0, 6),
        (0, 7),
        (0, 8),
        (0, 10),
        (0, 11),
        (0, 12),
        (0, 13),
        (0, 17),
        (0, 19),
        (0, 21),
        (0, 31),
        (1, 2),
        (1, 3),
        (1, 7),
        (1, 13),
        (1, 17),
        (1, 19),
        (1, 21),
        (1, 30),
        (2, 3),
        (2, 7),
        (2, 8),
        (2, 9),
        (2, 13),
        (2, 27),
        (2, 28),
        (2, 32),
        (3, 7),
        (3, 12),
        (3, 13),
        (4, 6),
        (4, 10),
        (5, 6),
        (5, 10),
        (5, 16),
        (6, 16),
        (8, 30),
        (8, 32),
        (8, 33),
        (9, 33),
        (13, 33),
        (14, 32),
        (14, 33),
        (15, 32),
        (15, 33),
        (18, 32),
        (18, 33),
        (19, 33),
        (20, 32),
        (20, 33),
        (22, 32),
        (22, 33),
        (23, 25),
        (23, 27),
        (23, 29),
        (23, 32),
        (23, 33),
        (24, 25),
        (24, 27),
        (24, 31),
        (25, 31),
        (26, 29),
        (26, 33),
        (27, 33),
        (28, 31),
        (28, 33),
        (29, 32),
        (29, 33),
        (30, 32),
        (30, 33),
        (31, 32),
        (31, 33),
        (32, 33),
    ]

    # Convert to undirected
    src_list = []
    dst_list = []
    for u, v in edges:
        src_list.extend([u, v])
        dst_list.extend([v, u])

    num_nodes = 34
    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_index = edge_index.to(device=device)
    # Random features
    in_channels = 16
    features = torch.randn(num_nodes, in_channels, device=device)

    return {
        "num_nodes": num_nodes,
        "edge_index": edge_index,
        "features": features,
        "in_channels": in_channels,
        "device": device,
    }


@pytest.fixture(params=["torch_native_mean_aggr", "pyg", "dgl"])
def graph_backend(request):
    """Parametrized fixture for different graph backends."""
    return request.param


@pytest.fixture
def create_graph_sample(set_default_device):
    """
    Factory fixture to create GraphSample with specified backend.

    Usage:
        graph_sample = create_graph_sample(
            edge_index=edge_index,
            features=features,
            backend="mean_agg"
        )
    """

    def _create(edge_index, features, backend, num_nodes=None, add_self_loops=True):
        if num_nodes is None:
            num_nodes = features.shape[0]

        return GraphSample(
            backend=MODEL_BACKEND_TO_GRAPH_REPR[backend],
            x=features,
            y=torch.zeros(num_nodes, device=features.device),
            edge_index=edge_index,
            add_self_loops=add_self_loops,
        )

    return _create


@pytest.fixture
def create_conv_layer(set_default_device):
    """
    Factory fixture to create convolution layer with specified backend.

    Usage:
        conv = create_conv_layer(
            backend_name="torch_native_mean_aggr",
            conv_type="gcn",
            feature_dim=16,
        )
    """

    def _create(backend_name, conv_type, feature_dim, **kwargs):
        backend = BackendRegistry.get_backend(backend_name)
        conv = backend.create_conv(conv_type, feature_dim=feature_dim, **kwargs)
        device = torch.get_default_device()
        return conv.to(device)

    return _create


@pytest.fixture
def identity_weight_conv(create_conv_layer, set_default_device):
    """
    Factory fixture to create a convolution with identity weight matrix.
    Useful for testing pure aggregation behavior.

    Usage:
        conv = identity_weight_conv(
            backend_name="torch_native_mean_aggr",
            conv_type="gcn",
            feature_dim=16,
        )
    """

    def _create(backend_name, conv_type, feature_dim, **kwargs):
        conv = create_conv_layer(backend_name, conv_type, feature_dim=feature_dim, bias=False, **kwargs)

        # set weight to identity
        device = torch.get_default_device()
        with torch.no_grad():
            identity = torch.eye(feature_dim, feature_dim, device=device)
            if hasattr(conv, "lin"):
                conv.lin.weight.data = identity
                conv.lin.bias.data = torch.zeros_like(conv.lin.bias.data)
            elif hasattr(conv, "_conv") and hasattr(conv._conv, "lin"):
                conv._conv.lin.weight.data = identity
                conv._conv.bias.data = torch.zeros_like(conv.lin.bias.data)

        return conv

    return _create


@pytest.fixture(scope="session")
def dgl_available():
    """Check if DGL is available for comparison tests."""
    try:
        import dgl  # noqa: F401

        return True
    except ImportError:
        return False
