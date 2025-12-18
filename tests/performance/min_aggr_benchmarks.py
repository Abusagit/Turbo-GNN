import dgl
import dgl.function as fn
import numpy as np
import ogb
import torch

from src.backends.cuda_backend.min_aggr.utils import min_aggr, min_aggr_forward, min_aggr_forward_partitioned
from src.data.datasets import load_pyg_single_graph


def create_random_graph(num_nodes, avg_degree=10, seed=42):
    """
    Create a random graph in CSR format on CUDA.

    Returns:
        edge_ptr      [num_nodes + 1] (int32, cuda)
        edge_indices  [num_edges]     (int32, cuda)
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Poisson-ish degrees, clipped to [1, num_nodes-1]
    degrees = np.random.poisson(avg_degree, num_nodes)
    degrees = np.clip(degrees, 1, num_nodes - 1)

    edge_ptr = np.concatenate([[0], np.cumsum(degrees)])
    num_edges = int(edge_ptr[-1])

    edge_indices = []
    for i in range(num_nodes):
        # sample neighbors w/o replacement
        nbrs = np.random.choice(num_nodes, size=degrees[i], replace=False)
        edge_indices.extend(nbrs)

    edge_indices = np.array(edge_indices, dtype=np.int32)
    edge_ptr = edge_ptr.astype(np.int32)

    edge_ptr_t = torch.from_numpy(edge_ptr).cuda()
    edge_idx_t = torch.from_numpy(edge_indices).cuda()
    return edge_ptr_t, edge_idx_t


def dgl_to_csr(g):
    """
    Convert DGL graph to CSR (indptr / indices),
    returned on CUDA as int32.
    """
    g = g.int().to("cuda")
    indptr, indices, _ = g.adj_tensors("csr")
    edge_ptr = indptr.int().contiguous()
    edge_idx = indices.int().contiguous()
    return edge_ptr, edge_idx


def csr_to_dgl_graph(edge_ptr, edge_indices, num_nodes):
    """
    Convert CSR back to a DGL graph (mostly for sanity / correctness).
    In CSR here, edge_ptr[i]:edge_ptr[i+1] are *incoming* neighbors of node i
    (nbr -> i). We'll reconstruct edges accordingly.
    """
    src_list = []
    dst_list = []

    edge_ptr_cpu = edge_ptr.cpu()
    edge_idx_cpu = edge_indices.cpu()

    for i in range(num_nodes):
        start = edge_ptr_cpu[i].item()
        end = edge_ptr_cpu[i + 1].item()
        nbrs = edge_idx_cpu[start:end]  # neighbors that point into i
        src_list.extend(nbrs.tolist())
        dst_list.extend([i] * (end - start))

    g = dgl.graph((src_list, dst_list), num_nodes=num_nodes)
    return g.to("cuda")


def split_nodes_by_degree(edge_ptr, threshold=32):
    deg = (edge_ptr[1:] - edge_ptr[:-1]).to(torch.int32)
    nodes = torch.arange(deg.numel(), device=edge_ptr.device, dtype=torch.int32)
    light = nodes[deg < threshold]
    heavy = nodes[deg >= threshold]
    return light, heavy


def dgl_min_aggr(g, x):
    out = dgl.ops.copy_u_min(g, x)
    out[out.isinf()] = 0
    return out


def load_real_graphs():
    """
    Load real-world benchmark graphs.

    Returns dict:
        name -> {
            "graph": DGLGraph on CUDA,
            "num_nodes": int,
            "num_edges": int
        }
    """
    graphs = {}
    import sys

    sys.path.append("/home/fominada/Accelerating-GNNs/data")
    from src.data.graphland_datasets import GraphLandDataset

    for dataset_name in [
        "hm-categories",
        # "pokec-regions",
        # "web-topics",
        "tolokers-2",
        "city-reviews",
        "artnet-exp",
        # "web-fraud",
        "hm-prices",
        "avazu-ctr",
        "city-roads-M",
        "city-roads-L",
        "twitch-views",
        "artnet-views",
        # "web-traffic",
    ]:
        dataset = GraphLandDataset(root="/home/fominada/Accelerating-GNNs/data", name=dataset_name, split="RL")
        g = dgl.graph((dataset[0].edge_index[0], dataset[0].edge_index[1]))
        g = dgl.add_self_loop(g)
        g = dgl.to_bidirected(g)
        graphs[dataset_name] = {
            "graph": g,
            "num_nodes": g.num_nodes(),
            "num_edges": g.num_edges(),
        }

    # Cora
    try:
        from dgl.data import CoraGraphDataset

        dataset = CoraGraphDataset("/home/fominada/Accelerating-GNNs/data")
        g = dataset[0]
        g = dgl.add_self_loop(g)
        g = dgl.to_bidirected(g)
        graphs["cora"] = {
            "graph": g,
            "num_nodes": g.num_nodes(),
            "num_edges": g.num_edges(),
        }
    except Exception as e:
        print(f"    Failed to load Cora: {e}")

    # Citeseer
    try:
        from dgl.data import CiteseerGraphDataset

        dataset = CiteseerGraphDataset("/home/fominada/Accelerating-GNNs/data")
        g = dataset[0]
        g = dgl.add_self_loop(g)
        g = dgl.to_bidirected(g)
        graphs["citeseer"] = {
            "graph": g,
            "num_nodes": g.num_nodes(),
            "num_edges": g.num_edges(),
        }
    except Exception as e:
        print(f"    Failed to load Citeseer: {e}")

    # Pubmed
    try:
        from dgl.data import PubmedGraphDataset

        dataset = PubmedGraphDataset("/home/fominada/Accelerating-GNNs/data")
        g = dataset[0]
        g = dgl.add_self_loop(g)
        g = dgl.to_bidirected(g)
        graphs["pubmed"] = {
            "graph": g,
            "num_nodes": g.num_nodes(),
            "num_edges": g.num_edges(),
        }
    except Exception as e:
        print(f"    Failed to load Pubmed: {e}")

    # ogbn-arxiv
    try:
        from ogb.nodeproppred import DglNodePropPredDataset

        dataset = DglNodePropPredDataset(name="ogbn-arxiv", root="/home/fominada/Accelerating-GNNs/data")
        g, _ = dataset[0]
        g = dgl.add_self_loop(g)
        g = dgl.to_bidirected(g)
        graphs["ogbn-arxiv"] = {
            "graph": g,
            "num_nodes": g.num_nodes(),
            "num_edges": g.num_edges(),
        }
    except ImportError:
        print("    ogbn-arxiv: OGB not installed (pip install ogb)")
    except Exception as e:
        print(f"    Failed to load ogbn-arxiv: {e}")

    # ogbn-products
    # try:
    #     from ogb.nodeproppred import DglNodePropPredDataset

    #     dataset = DglNodePropPredDataset(name="ogbn-products", root="/home/fominada/Accelerating-GNNs/data")
    #     g, _ = dataset[0]
    #     g = dgl.add_self_loop(g)
    #     g = dgl.to_bidirected(g)
    #     graphs["ogbn-products"] = {
    #         "graph": g,
    #         "num_nodes": g.num_nodes(),
    #         "num_edges": g.num_edges(),
    #     }
    # except ImportError:
    #     print("    ogbn-products: OGB not installed (pip install ogb)")
    # except Exception as e:
    #     print(f"    Failed to load ogbn-products: {e}")

    print()
    return graphs


def benchmark_performance(device):
    """
    Benchmarks CUDA kernels vs DGL on synthetic graphs.
    Measures forward, backward, and combined times separately.
    """
    import time

    print("=" * 160)
    print("PERFORMANCE BENCHMARK: CUDA vs DGL (Synthetic Graphs)")
    print("=" * 160)

    # torch.set_default_device(device)
    torch.manual_seed(0)
    np.random.seed(0)

    configs = [
        (50, 64, 10),
        (5000, 64, 20),
        (10000, 64, 30),
        (10000, 128, 30),
    ]

    header = (
        f"{'Nodes':<8} {'Dim':<6} {'Deg':<6} {'Edges':<10} | "
        f"{'FWD CUDA':<12} {'FWD DGL':<12} {'Speedup':<10} | "
        f"{'BWD CUDA':<12} {'BWD DGL':<12} {'Speedup':<10} | "
        f"{'TOTAL CUDA':<12} {'TOTAL DGL':<12} {'Speedup':<10}"
    )
    print(header)
    print("-" * 160)

    for num_nodes, d, avg_degree in configs:
        edge_ptr, edge_idx = create_random_graph(num_nodes, avg_degree)
        num_edges = edge_idx.shape[0]

        g = csr_to_dgl_graph(edge_ptr, edge_idx, num_nodes)

        x_ours = torch.randn(num_nodes, d, device=device, requires_grad=True)
        x_dgl = x_ours.detach().clone().requires_grad_(True)

        grad_output = torch.randn(num_nodes, d, device=device, dtype=torch.float32)

        light, heavy = split_nodes_by_degree(edge_ptr, threshold=32)

        # Warm-up
        for _ in range(10):
            x_ours.grad = None
            out_ours = min_aggr(edge_ptr, edge_idx, x_ours, light, heavy)
            out_ours.backward(grad_output)

        torch.cuda.synchronize()

        num_iters = 100

        torch.cuda.synchronize()
        start = time.time()
        for _ in range(num_iters):
            with torch.no_grad():
                _ = min_aggr_forward_partitioned(edge_ptr, edge_idx, x_ours.detach(), light, heavy)
        torch.cuda.synchronize()
        cuda_fwd_time = (time.time() - start) / num_iters * 1000.0

        x_ours.grad = None
        out_ours = min_aggr(edge_ptr, edge_idx, x_ours, light, heavy)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(num_iters):
            x_ours.grad = None
            out_ours.backward(grad_output, retain_graph=True)
        torch.cuda.synchronize()
        cuda_bwd_time = (time.time() - start) / num_iters * 1000.0

        torch.cuda.synchronize()
        start = time.time()
        for _ in range(num_iters):
            x_ours.grad = None
            out_ours = min_aggr(edge_ptr, edge_idx, x_ours, light, heavy)
            out_ours.backward(grad_output)
        torch.cuda.synchronize()
        cuda_total_time = (time.time() - start) / num_iters * 1000.0

        # ==================== DGL TIMING ====================

        # Warm-up
        for _ in range(10):
            x_dgl.grad = None
            out_dgl = dgl_min_aggr(g, x_dgl)
            out_dgl.backward(grad_output)
        torch.cuda.synchronize()

        # Forward only
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(num_iters):
            with torch.no_grad():
                _ = dgl_min_aggr(g, x_dgl.detach())
        torch.cuda.synchronize()
        dgl_fwd_time = (time.time() - start) / num_iters * 1000.0

        # Backward only
        x_dgl.grad = None
        out_dgl = dgl_min_aggr(g, x_dgl)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(num_iters):
            x_dgl.grad = None
            out_dgl.backward(grad_output, retain_graph=True)
        torch.cuda.synchronize()
        dgl_bwd_time = (time.time() - start) / num_iters * 1000.0

        # Forward + Backward combined
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(num_iters):
            x_dgl.grad = None
            out_dgl = dgl_min_aggr(g, x_dgl)
            out_dgl.backward(grad_output)
        torch.cuda.synchronize()
        dgl_total_time = (time.time() - start) / num_iters * 1000.0

        fwd_speedup = dgl_fwd_time / cuda_fwd_time
        bwd_speedup = dgl_bwd_time / cuda_bwd_time
        total_speedup = dgl_total_time / cuda_total_time

        print(
            f"{num_nodes:<8} {d:<6} {avg_degree:<6} {num_edges:<10} | "
            f"{cuda_fwd_time:<12.3f} {dgl_fwd_time:<12.3f} {fwd_speedup:<10.2f} | "
            f"{cuda_bwd_time:<12.3f} {dgl_bwd_time:<12.3f} {bwd_speedup:<10.2f} | "
            f"{cuda_total_time:<12.3f} {dgl_total_time:<12.3f} {total_speedup:<10.2f}"
        )

    print("\nTime units: milliseconds\n")


def benchmark_real_graphs(device):
    """
    Benchmark CUDA min_aggr kernels vs DGL on real datasets with memory usage tracking.
    """
    import time

    print("=" * 200)
    print("PERFORMANCE BENCHMARK: MIN-AGGR CUDA vs DGL (Real-World Graphs)")
    print("=" * 200)

    # torch.set_default_device(device)
    torch.manual_seed(0)
    np.random.seed(0)

    graphs = load_real_graphs()
    test_dims = [32, 64, 128, 256]

    header = (
        f"{'Dataset':<15} {'Nodes':<10} {'Edges':<10} {'Dim':<6} {'Thr':<6}| "
        f"{'FWD CUDA':<12} {'FWD DGL':<12} {'Speedup':<10} | "
        f"{'BWD CUDA':<12} {'BWD DGL':<12} {'Speedup':<10} | "
        f"{'TOTAL CUDA':<12} {'TOTAL DGL':<12} {'Speedup':<10} | "
        f"{'MEM CUDA':<12} {'MEM DGL':<12} {'Ratio':<10}"
    )
    print(header)
    print("-" * 200)

    for name, info in graphs.items():
        # sample = load_pyg_single_graph(
        #     name=name,
        #     graph_backend="csr",
        #     root="data",
        #     allow_random_split=True,
        # )

        sample = info["graph"].to("cuda")
        sample = info["graph"].to(device)
        indptr, indices = dgl_to_csr(sample)
        g = sample

        N = info["num_nodes"]
        num_edges = info["num_edges"]

        light, heavy = split_nodes_by_degree(indptr, threshold=64)
        for thr in [32, 64, 128, 256]:
            for d in test_dims:
                x = torch.randn(N, d, device=device, dtype=torch.float32)
                x_cuda = x.detach().clone().requires_grad_(True)
                x_dgl = x.detach().clone().requires_grad_(True)

                grad_output = torch.randn(N, d, device=device, dtype=torch.float32)

                # ==================== CUDA TIMING ====================

                # Warm-up
                for _ in range(5):
                    x_cuda.grad = None
                    out_cuda = min_aggr(indptr, indices, x_cuda, light, heavy)
                    out_cuda.backward(grad_output)
                torch.cuda.synchronize()

                num_iters = 10

                # Forward only
                torch.cuda.synchronize()
                start = time.time()
                for _ in range(num_iters):
                    with torch.no_grad():
                        _ = min_aggr_forward_partitioned(indptr, indices, x_cuda.detach(), light, heavy)
                torch.cuda.synchronize()
                cuda_fwd_time = (time.time() - start) / num_iters * 1000.0

                # Backward only
                x_cuda.grad = None
                out_cuda = min_aggr(indptr, indices, x_cuda, light, heavy)

                torch.cuda.synchronize()
                start = time.time()
                for _ in range(num_iters):
                    x_cuda.grad = None
                    out_cuda.backward(grad_output, retain_graph=True)
                torch.cuda.synchronize()
                cuda_bwd_time = (time.time() - start) / num_iters * 1000.0

                # Forward + Backward combined (with memory tracking)
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()

                start = time.time()
                for _ in range(num_iters):
                    x_cuda.grad = None
                    out_cuda = min_aggr(indptr, indices, x_cuda, light, heavy)
                    out_cuda.backward(grad_output)
                torch.cuda.synchronize()
                cuda_total_time = (time.time() - start) / num_iters * 1000.0

                cuda_peak_mem = torch.cuda.max_memory_allocated()
                cuda_mem_usage = cuda_peak_mem / (1024**3)  # GB

                # ==================== DGL TIMING ====================

                # Warm-up
                for _ in range(5):
                    x_dgl.grad = None
                    out_dgl = dgl_min_aggr(g, x_dgl)
                    out_dgl.backward(grad_output)
                torch.cuda.synchronize()

                # Forward only
                torch.cuda.synchronize()
                start = time.time()
                for _ in range(num_iters):
                    with torch.no_grad():
                        _ = dgl_min_aggr(g, x_dgl.detach())
                torch.cuda.synchronize()
                dgl_fwd_time = (time.time() - start) / num_iters * 1000.0

                # Backward only
                x_dgl.grad = None
                out_dgl = dgl_min_aggr(g, x_dgl)

                torch.cuda.synchronize()
                start = time.time()
                for _ in range(num_iters):
                    x_dgl.grad = None
                    out_dgl.backward(grad_output, retain_graph=True)
                torch.cuda.synchronize()
                dgl_bwd_time = (time.time() - start) / num_iters * 1000.0

                # Forward + Backward combined (with memory tracking)
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()

                start = time.time()
                for _ in range(num_iters):
                    x_dgl.grad = None
                    out_dgl = dgl_min_aggr(g, x_dgl)
                    out_dgl.backward(grad_output)
                torch.cuda.synchronize()
                dgl_total_time = (time.time() - start) / num_iters * 1000.0

                dgl_peak_mem = torch.cuda.max_memory_allocated()
                dgl_mem_usage = dgl_peak_mem / (1024**3)  # GB

                # Compute speedups and memory ratio
                fwd_speedup = dgl_fwd_time / cuda_fwd_time
                bwd_speedup = dgl_bwd_time / cuda_bwd_time
                total_speedup = dgl_total_time / cuda_total_time
                mem_ratio = dgl_mem_usage / cuda_mem_usage if cuda_mem_usage > 0 else 0.0

                print(
                    f"{name:<15} {N:<10} {num_edges:<10} {d:<6} {thr:<6} | "
                    f"{cuda_fwd_time:<12.3f} {dgl_fwd_time:<12.3f} {fwd_speedup:<10.2f} | "
                    f"{cuda_bwd_time:<12.3f} {dgl_bwd_time:<12.3f} {bwd_speedup:<10.2f} | "
                    f"{cuda_total_time:<12.3f} {dgl_total_time:<12.3f} {total_speedup:<10.2f} | "
                    f"{cuda_mem_usage:<12.3f} {dgl_mem_usage:<12.3f} {mem_ratio:<10.2f}x"
                )

    print("\nTime units: milliseconds | Memory units: GB | Ratio: DGL/CUDA memory usage\n")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # for name in ["cora"]:
    #     check_dataset(name, device)

    # check_synthetic(num_nodes=50, d=64, avg_degree=10, device=device)

    # mini_benchmark_synthetic(device)

    benchmark_performance(device)
    benchmark_real_graphs(device)


if __name__ == "__main__":
    main()
