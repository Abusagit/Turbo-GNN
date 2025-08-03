import torch
from torch import nn
from statistics import mean, stdev, median

import dgl.function as fn

from data import generate_random_graph
from cusparse_spmm import csr_SPMM, find_best_algorithm



class GraphConvSimple(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    
    def forward(self, graph, feat):
        with graph.local_scope():
            graph.ndata['h'] = feat
            graph.update_all(fn.copy_u('h', 'm'), fn.sum(msg="m", out="h"))
            return graph.ndata['h']


@torch.inference_mode()
def benchmark_dgl_message_passing(dgl_graph, node_features, num_iters=100):
    """
    Runs and times message passing (GraphConvSimple) on given DGL graph and node features.

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
    gcn_layer = GraphConvSimple(dgl_graph, in_feats) # average incoming nodes

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

    avg_time_ms = mean(timings)
    std_time_ms = stdev(timings)
    print(f"DGL timings: {timings}")
    return avg_time_ms, std_time_ms, out



def benchmark_cusparse_spmm(indptr, indices, feats, num_iters=100):


    torch.cuda.synchronize()
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = []

    best_alg_id = find_best_algorithm(indptr, indices, feats)

    print(f"Best algo fir this setup is {best_alg_id}")

    for _ in range(10):
        out = csr_SPMM(indptr, indices, feats, algorithm=best_alg_id)

    for _ in range(num_iters):
        starter.record()
        out = csr_SPMM(indptr, indices, feats, algorithm=best_alg_id)
        ender.record()
        torch.cuda.synchronize()
        timings.append(starter.elapsed_time(ender))

    avg_time_ms = mean(timings)
    std_time_ms = stdev(timings)
    print(f"cusparse timings: {timings}")

    return avg_time_ms, std_time_ms, out





if __name__ == "__main__":
    
    print(f"CUDA version: {torch.version.cuda}. CUDA is available: {torch.cuda.is_available()}. Device is: {torch.cuda.get_device_name(0)}")       # CUDA version PyTorch was built with

    num_nodes, avg_degree, feature_dim = 10000, 20, 64
    device = 'cuda'

    g, indptr, indices, feats = generate_random_graph(num_nodes, avg_degree, feature_dim, device)

    avg_time, std_time, out_feats_1 = benchmark_dgl_message_passing(g, feats)
    print(f"DGL GraphConv message passing mean time: {avg_time:.3f} +- {std_time:.3f} ms")

    avg_time, std_time, out_feats_2 = benchmark_cusparse_spmm(indptr, indices, feats)
    print(f"cuSPARSE SpMM message passing mean time: {avg_time:.3f} +- {std_time:.3f} ms")

    torch.testing.assert_close(out_feats_1, out_feats_2)
