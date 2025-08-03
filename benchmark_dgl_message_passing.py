import torch
import dgl
import time
from dgl.nn.pytorch import GraphConv
from data import generate_random_graph

def benchmark_dgl_message_passing(dgl_graph, node_features, num_iters=100):
    """
    Runs and times message passing (GraphConv) on given DGL graph and node features.

    Args:
        dgl_graph: DGLGraph object (can be on CPU; DGL moves data internally)
        node_features: torch.Tensor of shape (N, F) on CUDA device
        num_iters: how many iterations to average time over

    Returns:
        avg_time_ms: average time per forward pass in milliseconds
        output_feats: resulting tensor after message passing (for correctness check)
    """
    device = node_features.device

    # Move graph to device (DGL supports this)
    dgl_graph = dgl_graph.to(device)

    in_feats = node_features.shape[1]
    gcn_layer = GraphConv(in_feats, in_feats, norm='right', weight=True, bias=True).to(device) # average incoming nodes

    # Warm up
    for _ in range(10):
        out = gcn_layer(dgl_graph, node_features)

    # CUDA events for precise timing
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = []

    for _ in range(num_iters):
        starter.record()
        out = gcn_layer(dgl_graph, node_features)
        ender.record()

        # Wait for event to complete
        torch.cuda.synchronize()
        elapsed = starter.elapsed_time(ender)  # milliseconds
        timings.append(elapsed)

    avg_time_ms = sum(timings) / len(timings)
    return avg_time_ms, out

# Example usage:
if __name__ == "__main__":

    # Assume you have the generate_random_graph() from before
    num_nodes, avg_degree, feature_dim = 10000, 20, 64
    device = 'cuda'

    g, _, _, feats = generate_random_graph(num_nodes, avg_degree, feature_dim, device)

    avg_time, out_feats = benchmark_dgl_message_passing(g, feats)
    print(f"DGL GraphConv message passing average time: {avg_time:.3f} ms")
