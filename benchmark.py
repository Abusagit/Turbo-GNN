import torch
from torch import nn
from statistics import mean, stdev, median
import pandas as pd
import dgl.function as fn

from data import generate_random_graph
from cusparse_spmm import csr_SPMM, find_best_algorithm, clear_cache



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

    timings = sorted(timings)[10:-10]  # Remove top and bottom 10%
    # print(f"dgl timings: {timings}")

    return mean(timings), stdev(timings), out


def benchmark_cusparse_spmm(indptr, indices, feats, num_iters=100):


    torch.cuda.synchronize()
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = []

    best_alg_id = 2 # find_best_algorithm(indptr, indices, feats)

    # print(f"Best algo fir this setup is {best_alg_id}")

    for _ in range(10):
        out = csr_SPMM(indptr, indices, feats, algorithm=best_alg_id)

    for _ in range(num_iters):
        starter.record()
        out = csr_SPMM(indptr, indices, feats, algorithm=best_alg_id)
        ender.record()
        torch.cuda.synchronize()
        timings.append(starter.elapsed_time(ender))



    timings = sorted(timings)[10:-10]  # Remove top and bottom 10%
    # print(f"cusparse timings: {timings}")

    return mean(timings), stdev(timings), out





if __name__ == "__main__":
    
    print(f"CUDA version: {torch.version.cuda}. CUDA is available: {torch.cuda.is_available()}. Device is: {torch.cuda.get_device_name(0)}")       # CUDA version PyTorch was built with


    df = []
    # Test configurations
    configs = [
        (10000, 20, 64),    # Original
        (10000, 20, 128),   # More features
        (10000, 20, 1024),   # even More features
        (50000, 20, 64),    # Larger graph
        (50000, 20, 512),    # Larger graph * larger dim
        (10000, 50, 64),    # Higher degree
        (10000, 50, 256),    # Higher degree * larger dim
        (10000, 3, 256),    # Higher degree * larger dim
        (10000, 3, 512),    # Higher degree * larger dim
        (100000, 3, 1024),    # Higher degree * larger dim
        (100000, 5, 32),    # Higher degree * larger dim
        (100000, 5, 64),    # Higher degree * larger dim
    ]
    
    for num_nodes, avg_degree, feature_dim in configs:
        print(f"\n{'='*60}")
        print(f"Config: nodes={num_nodes}, degree={avg_degree}, features={feature_dim}")
        
        g, indptr, indices, feats = generate_random_graph(num_nodes, avg_degree, feature_dim, 'cuda')
        
        # Run benchmarks multiple times and take best
        dgl_times = []
        cusparse_times = []
        
        for run in range(3):
            print(f"\nRun {run+1}/3:")
            
            avg_time_dgl, std_time_dgl, out_dgl = benchmark_dgl_message_passing(g, feats, 100)
            print(f"  DGL: {avg_time_dgl:.4f} ± {std_time_dgl:.4f} ms")
            dgl_times.append(avg_time_dgl)
            
            avg_time_csr, std_time_csr, out_csr = benchmark_cusparse_spmm(indptr, indices, feats, 100)
            print(f"  cuSPARSE: {avg_time_csr:.4f} ± {std_time_csr:.4f} ms")
            cusparse_times.append(avg_time_csr)
            
            # Verify correctness
            torch.testing.assert_close(out_dgl, out_csr)
            clear_cache()
            
        
        best_dgl = min(dgl_times)
        best_cusparse = min(cusparse_times)
        
        print(f"\nBest times:")
        print(f"  DGL: {best_dgl:.4f} ms")
        print(f"  cuSPARSE: {best_cusparse:.4f} ms")
        print(f"  Speedup: {best_dgl/best_cusparse:.2f}x")

        df.append(dict(speedup=best_dgl/best_cusparse, num_nodes=num_nodes, avg_degree=avg_degree, feature_dim=feature_dim, graph="synthetic"))
    df = pd.DataFrame(df)
    df.to_csv("benchmark_results.csv")
