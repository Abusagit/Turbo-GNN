import torch
import scipy.sparse as sp
import dgl
from enum import Enum, auto

class DatasetName(Enum):
    CORA = auto()
    CITESEER = auto()
    PUBMED = auto()
    OGB_ARXIV = auto()
    OGB_PRODUCTS = auto()


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

    src, dst = dgl_graph.edges()

    adj_transposed = sp.csr_matrix(
        (torch.ones(len(src)).numpy(), (dst.cpu().numpy(), src.cpu().numpy())),
        shape=(num_nodes, num_nodes)
    )

    indptr = torch.from_numpy(adj_transposed.indptr).int().cuda()
    indices = torch.from_numpy(adj_transposed.indices).int().cuda()


    return dgl_graph, indptr, indices, node_features



def get_real_graph(dataset: DatasetName):
    raise NotImplementedError



if __name__ == "__main__":
    num_nodes = 10000
    avg_degree = 20
    feature_dim = 64
    device = 'cuda'

    g, indptr, indices, feats = generate_random_graph(num_nodes, avg_degree, feature_dim, device)
    print(f"Generated graph with {g.number_of_nodes()} nodes and {g.number_of_edges()} edges")
    print(f"CSR row_ptr shape: {indptr.shape}, CSR col_ind shape: {indices.shape}")
    print(f"Node features shape: {feats.shape}, device: {feats.device}")
