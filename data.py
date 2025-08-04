import torch
import scipy.sparse as sp
import dgl
from enum import Enum

from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from ogb.nodeproppred import DglNodePropPredDataset

class DatasetName(Enum):
    CORA = "Cora"
    CITESEER = "Citeseer"
    PUBMED = "PubMed"
    OGB_ARXIV = "Arxiv"
    OGB_PRODUCTS = "Products"


def generate_random_graph(num_nodes, avg_degree, feature_dim, device='cuda'):
    """
    Generates a random directed graph with given parameters,
    returns both DGL graph and CSR arrays, plus random node features.

    Args:
        num_nodes (int): number of nodes
        avg_degree (int): average out-degree per node
        feature_dim (int): dimension of node feature vectors
        device (str): device to place node features ('cuda' or 'cpu')

    Returns:
        dgl_graph: DGLGraph object (on CPU)
        csr_row_ptr: torch.LongTensor (on CPU)
        csr_col_ind: torch.LongTensor (on CPU)
        node_features: torch.FloatTensor (on device)
    """
    # Total edges roughly
    num_edges = num_nodes * avg_degree

    # Generate random adjacency matrix using scipy
    # Use directed Erdos-Renyi model: each edge exists with p = avg_degree / num_nodes
    p = avg_degree / num_nodes
    adj = sp.random(num_nodes, num_nodes, density=p, format='csr', dtype='float32', data_rvs=lambda n: torch.ones(n).numpy())

    # Extract CSR arrays (row_ptr and col_ind)

        # Build DGL graph from scipy adjacency
    dgl_graph: dgl.DGLGraph = dgl.from_scipy(adj)
        # Generate random node features
    node_features = torch.randn(num_nodes, feature_dim, device=device)
    dgl_graph = dgl.add_self_loop(dgl_graph)

    src, dst = dgl_graph.edges()

    adj_transposed = sp.csr_matrix(
        (torch.ones(len(src)).numpy(), (dst.cpu().numpy(), src.cpu().numpy())),
        shape=(num_nodes, num_nodes)
    )

    indptr = torch.from_numpy(adj_transposed.indptr).int().cuda()
    indices = torch.from_numpy(adj_transposed.indices).int().cuda()


    return dgl_graph, indptr, indices, node_features



def get_real_graph(dataset: DatasetName, hidden_dim: int = 64):
    if dataset is DatasetName.CITESEER:
        graph: dgl.DGLGraph = CiteseerGraphDataset(verbose=False)[0]
    elif dataset is DatasetName.PUBMED:
        graph = PubmedGraphDataset(verbose=False)[0]
    elif dataset is DatasetName.CORA:
        graph = CoraGraphDataset(verbose=False)[0]
    elif dataset is DatasetName.OGB_ARXIV:
        graph = DglNodePropPredDataset(name="ogbn-arxiv")[0][0]
    elif dataset is DatasetName.OGB_PRODUCTS:
        graph = DglNodePropPredDataset(name="ogbn-products")[0][0]
    else:
        raise NotImplementedError


    features = torch.randn((graph.num_nodes(), hidden_dim)).cuda()
    graph = dgl.add_self_loop(graph)
    src, dst = graph.edges()

    adj_transposed = sp.csr_matrix(
        (torch.ones(len(src)).numpy(), (dst.cpu().numpy(), src.cpu().numpy())),
        shape=(graph.num_nodes(), graph.num_nodes())
    )

    indptr = torch.from_numpy(adj_transposed.indptr).int().cuda()
    indices = torch.from_numpy(adj_transposed.indices).int().cuda()

    # breakpoint()

    return graph, indptr, indices, features
