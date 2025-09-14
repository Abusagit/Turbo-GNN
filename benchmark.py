import torch
from torch import nn
from statistics import mean, stdev, median
import pandas as pd
import dgl.function as fn

from data import generate_random_graph, DatasetName, get_real_graph
from cusparse_spmm import csr_SPMM_normalized, csr_SPMM, find_best_algorithm, clear_cache, find_best_algorithm_normalized

from itertools import product
import argparse


from dgl.nn.pytorch import GraphConv

class GraphConvSimple(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    
    def forward(self, graph, feat):
        with graph.local_scope():
            graph.ndata['h'] = feat
            graph.update_all(fn.copy_u('h', 'm'), fn.sum(msg="m", out="h"))
            return graph.ndata['h']


@torch.inference_mode()
def benchmark_dgl_message_passing(dgl_graph, node_features, num_iters=100, norm=None):
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
    if norm is None:
        gcn_layer = GraphConvSimple(dgl_graph, in_feats) # average incoming nodes
    else:
        gcn_layer = GraphConv(in_feats, in_feats, norm=norm, weight=False, bias=False)

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


def benchmark_cusparse_spmm(indptr, indices, feats, num_iters=100, norm=None):


    torch.cuda.synchronize()
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = []

    # best_alg_id = 2 # find_best_algorithm(indptr, indices, feats)
    best_alg_id = find_best_algorithm_normalized(indptr, indices, feats, norm=norm)

    # print(f"Best algo fir this setup is {best_alg_id}")

    for _ in range(10):
        out = csr_SPMM_normalized(indptr, indices, feats, algorithm=best_alg_id, use_cache=True, norm=norm)

    for _ in range(num_iters):
        starter.record()
        out = csr_SPMM_normalized(indptr, indices, feats, algorithm=best_alg_id, use_cache=True, norm=norm)
        ender.record()
        torch.cuda.synchronize()
        timings.append(starter.elapsed_time(ender))



    timings = sorted(timings)[10:-10]  # Remove top and bottom 10%
    # print(f"cusparse timings: {timings}")

    return mean(timings), stdev(timings), out


def benchmark_torch_matmul(indptr, indices, feats, num_iters=100, norm=None):
    # create sparse matrix
    pass


def _benchmark(g, indptr, indices, feats, num_iters=200, norm=None):
    dgl_times = []
    cusparse_times = []

    max_diff = float("-inf")

    for run in range(3):
        print(f"\nRun {run+1}/3:")
        
        avg_time_dgl, std_time_dgl, out_dgl = benchmark_dgl_message_passing(g, feats, num_iters, norm=norm)
        print(f"  DGL: {avg_time_dgl:.4f} ± {std_time_dgl:.4f} ms")
        dgl_times.append(avg_time_dgl)
        
        avg_time_csr, std_time_csr, out_csr = benchmark_cusparse_spmm(indptr, indices, feats, num_iters, norm=norm)
        print(f"  cuSPARSE: {avg_time_csr:.4f} ± {std_time_csr:.4f} ms")
        cusparse_times.append(avg_time_csr)

        # Verify correctness
        # torch.testing.assert_close(out_dgl, out_csr, atol=1e-4, rtol=1e-4)
        diff = torch.abs(out_dgl - out_csr).max().item()
        max_diff = max(diff, max_diff)
        clear_cache()
    
    best_dgl = min(dgl_times)
    best_cusparse = min(cusparse_times)

    print( "\nBest times:")
    print(f"  DGL: {best_dgl:.4f} ms")
    print(f"  cuSPARSE: {best_cusparse:.4f} ms")
    print(f"  Speedup: {best_dgl/best_cusparse:.2f}x")
    print(f"  Max diff: {max_diff:.4f}")

    results = dict(
        best_dgl=best_dgl,
        best_cusparse=best_cusparse,
        speedup=best_dgl/best_cusparse,
        max_diff=max_diff,
    )

    return results


if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("--norm", default="none")
    parser.add_argument("--iters", default=200, type=int)

    args = parser.parse_args()
    
    print(f"CUDA version: {torch.version.cuda}. CUDA is available: {torch.cuda.is_available()}. Device is: {torch.cuda.get_device_name(0)}")       # CUDA version PyTorch was built with
    norm = args.norm or "none"


    df = []
    # Test configurations
    configs = [
        # (250000, 2, 1024),    # Higher degree * larger dim
        # (250000, 10, 1024),    # Higher degree * larger dim

        # (100000, 5, 32),    
        # # (100000, 5, 64),    # Higher degree * larger dim
        # (100000, 5, 128),    # Higher degree * larger dim
        # # (100000, 3, 1024),    # Higher degree * larger dim
        # # (100000, 20, 1024),    # Higher degree * larger dim

        # (10000, 3, 64),    # Original
        # (10000, 3, 128),   # More features
        # (10000, 5, 1024),   # even More features
        # (10000, 10, 1024),   # even More features
        # (10000, 20, 1024),   # even More features

    ]
    
    for num_nodes, avg_degree, feature_dim in configs:
        print(f"\n{'='*60}")
        print(f"Config: nodes={num_nodes}, degree={avg_degree}, features={feature_dim}")
        
        g, indptr, indices, feats = generate_random_graph(num_nodes, avg_degree, feature_dim, 'cuda')
        
        results = _benchmark(g, indptr, indices, feats, args.iters, args.norm)

        df.append(dict(num_nodes=num_nodes, avg_degree=avg_degree, feature_dim=feature_dim, graph="synthetic", norm=norm) | results)

    real_datasets_configs = product(
        [DatasetName.CORA, DatasetName.CITESEER, DatasetName.PUBMED, DatasetName.OGB_ARXIV, DatasetName.OGB_PRODUCTS],
        [32, 64, 128, 256, 512, 1024]
    )
    for dataset, hidden_dim in real_datasets_configs:
        print(f"\n{'='*60}")
        print(f"Config: dataset={dataset.value}, features={hidden_dim}")
        
        g, indptr, indices, feats = get_real_graph(dataset, hidden_dim)
        n_nodes = g.num_nodes()
        avg_degree = 2 * g.num_edges() / g.num_nodes()
        # Run benchmarks multiple times and take best
        results = _benchmark(g, indptr, indices, feats, args.iters, args.norm)

        df.append(dict(num_nodes=n_nodes, avg_degree=avg_degree, feature_dim=hidden_dim, graph=dataset.value, norm=norm) | results)


    df = pd.DataFrame(df)
    df.to_csv(f"data/benchmark_results_norm_{norm}_{args.iters}_iters.csv")

    print(df)
