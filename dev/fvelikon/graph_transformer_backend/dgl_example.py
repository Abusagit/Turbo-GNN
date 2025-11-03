import os
import time
import math
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

import dgl
import dgl.function as fn

# =====================================================
# User-tunable threshold: what we call a "huge" node.
# Must match the kernel's DEG_HUGE in C++.
# =====================================================
DEG_HUGE = 256

# =====================================================
# JIT compile the CUDA extension
# (assumes the .cu file exposes forward_buckets and forward_buckets_half)
# =====================================================
print("Compiling CUDA extension...")
current_dir = os.path.dirname(os.path.abspath(__file__))
graph_attention_cuda = load(
    name="src/backends/cuda_backend/graph_attention_cuda",
    sources=[os.path.join(current_dir, "graph_transformer.cu")],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-arch=sm_80"],
    verbose=True,
)
print("Compilation complete!\n")


# =====================================================
# Utility functions
# =====================================================


def measure_memory(func, *args, **kwargs):
    """
    Measure GPU memory usage of a function call.

    Returns:
        result: function output
        memory_allocated (MB): delta allocated during the call
        peak_memory (MB): max allocated during the call
    """
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    start_memory = torch.cuda.memory_allocated() / 1024**2

    result = func(*args, **kwargs)

    torch.cuda.synchronize()
    end_memory = torch.cuda.memory_allocated() / 1024**2
    peak_memory = torch.cuda.max_memory_allocated() / 1024**2

    memory_allocated = end_memory - start_memory
    return result, memory_allocated, peak_memory


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
    sys.path.append("/home/fvelikon/projects/cuda_exp/src")
    from src.data.graphland_datasets import GraphLandDataset

    for dataset_name in ["hm-categories",
            "pokec-regions",
            "web-topics",
            "tolokers-2",
            "city-reviews",
            "artnet-exp",
            "web-fraud",
            "hm-prices",
            "avazu-ctr",
            "city-roads-M",
            "city-roads-L",
            "twitch-views",
            "artnet-views",
            "web-traffic",
]:
        dataset  = GraphLandDataset(root="/home/fvelikon/projects/cuda_exp/data", name=dataset_name, split="RL")
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

        dataset = CoraGraphDataset("/home/fvelikon/projects/cuda_exp/data")
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

        dataset = CiteseerGraphDataset("/home/fvelikon/projects/cuda_exp/data")
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

        dataset = PubmedGraphDataset("/home/fvelikon/projects/cuda_exp/data")
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

        dataset = DglNodePropPredDataset(name="ogbn-arxiv", root="/home/fvelikon/projects/cuda_exp/data")
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
    try:
        from ogb.nodeproppred import DglNodePropPredDataset

        dataset = DglNodePropPredDataset(name="ogbn-products", root="/home/fvelikon/projects/cuda_exp/data")
        g, _ = dataset[0]
        g = dgl.add_self_loop(g)
        g = dgl.to_bidirected(g)
        graphs["ogbn-products"] = {
            "graph": g,
            "num_nodes": g.num_nodes(),
            "num_edges": g.num_edges(),
        }
    except ImportError:
        print("    ogbn-products: OGB not installed (pip install ogb)")
    except Exception as e:
        print(f"    Failed to load ogbn-products: {e}")

    print()
    return graphs


def bucket_nodes(edge_ptr, deg_huge=DEG_HUGE):
    """
    Split nodes into buckets:
      - mid_nodes: 0 < deg <= deg_huge
      - huge_nodes: deg > deg_huge
    Returns (mid_nodes:int32[cuda], huge_nodes:int32[cuda])
    """
    # deg[i] = number of inbound edges for node i
    deg = (edge_ptr[1:] - edge_ptr[:-1]).to(torch.int32)  # cuda int32

    mid_mask = (deg > 0) & (deg <= deg_huge)
    huge_mask = deg > deg_huge

    mid_nodes = torch.nonzero(mid_mask, as_tuple=False).view(-1).to(torch.int32)
    huge_nodes = torch.nonzero(huge_mask, as_tuple=False).view(-1).to(torch.int32)

    # ensure contiguous for kernel argument passing
    return mid_nodes.contiguous(), huge_nodes.contiguous()


def dgl_graph_attention(g, Q, K, V):
    """
    DGL implementation of graph attention using message passing:
        attn_scores = (Q ⋅ K^T on edges) * scale
        softmax over incoming edges
        aggregate V with those weights
    """
    num_nodes, d = Q.shape
    scale = 1.0 / math.sqrt(d)

    # edge attention scores: u_dot_v computes <src, dst> per edge
    attn_scores = dgl.ops.u_dot_v(g, Q, K)  # [E]
    attn_scores *= scale
    attn_probs = dgl.nn.functional.edge_softmax(g, attn_scores)

    # weighted sum of V over edges with attention weights
    hidden = dgl.ops.u_mul_e_sum(g, V, attn_probs)
    return hidden


def naive_graph_attention(edge_ptr, edge_indices, Q, K, V):
    """
    Reference PyTorch (CPU-ish logic on GPU tensors),
    for correctness checking.

    For each dst node i:
      neighbors = edge_indices[edge_ptr[i] : edge_ptr[i+1]]
      score[nbr] = <K[i], Q[nbr]> * scale
      attn = softmax(score)
      out[i] = sum(attn[nbr] * V[nbr])
    """
    num_nodes, d = Q.shape
    out = torch.zeros_like(V)
    scale = 1.0 / math.sqrt(d)

    for i in range(num_nodes):
        start = edge_ptr[i].item()
        end = edge_ptr[i + 1].item()
        if start == end:
            continue

        nbrs = edge_indices[start:end]

        # score = <K[i], Q[nbr]> * scale
        # Q[nbrs] shape: [deg, d]; K[i] shape: [d]
        scores = torch.matmul(Q[nbrs], K[i]) * scale  # [deg]

        attn = torch.softmax(scores, dim=0)  # [deg]

        out[i] = torch.matmul(attn, V[nbrs])  # [d]

    return out


def calculate_memory_bandwidth(num_bytes, time_ms):
    """
    Approximate memory bandwidth in GB/s, assuming time_ms.
    We keep the same formula (float32 assumptions) for comparison
    even when half path is active.
    """
    return (num_bytes / 1e9) / (time_ms / 1000.0)


def run_attention(
    edge_ptr,
    edge_indices,
    mid_nodes,
    huge_nodes,
    Q_fp32,
    K_fp32,
    V_fp32,
    Q_fp16=None,
    K_fp16=None,
    V_fp16=None,
    use_half=False,
):
    """
    Wrapper that calls the right CUDA kernel:
      - FP32 path (forward_buckets)
      - half-IO path for huge nodes (forward_buckets_half) -- IT TURNED OUT TO BE SLOWER SO WE DONT USE IT!!!
    """
    # return graph_attention_cuda.forward_edge_parallel(
    #     edge_ptr, edge_indices, Q_fp32, K_fp32, V_fp32
    # )

    return graph_attention_cuda.forward_buckets(
        edge_ptr,
        edge_indices,
        mid_nodes,
        huge_nodes,
        Q_fp32,
        K_fp32,
        V_fp32,
    )


# =====================================================
# Correctness test
# =====================================================


def test_correctness():
    print("=" * 60)
    print("CORRECTNESS TEST")
    print("=" * 60)

    num_nodes = 100
    d = 64
    avg_degree = 8
    print(f"Graph size: {num_nodes} nodes, dim {d}")

    edge_ptr, edge_idx = create_random_graph(num_nodes, avg_degree)
    num_edges = edge_idx.shape[0]
    print(f"Number of edges: {num_edges}")
    print(f"Average degree: {num_edges / num_nodes:.2f}\n")

    # features
    Q = torch.randn(num_nodes, d, device="cuda", dtype=torch.float32)
    K = torch.randn(num_nodes, d, device="cuda", dtype=torch.float32)
    V = torch.randn(num_nodes, d, device="cuda", dtype=torch.float32)

    # bucket nodes
    mid_nodes, huge_nodes = bucket_nodes(edge_ptr, deg_huge=DEG_HUGE)

    # CUDA FP32 path
    print("Running CUDA kernel (fp32 path)...")
    cuda_out_fp32 = run_attention(edge_ptr, edge_idx, mid_nodes, huge_nodes, Q, K, V, use_half=False)

    # CUDA half path (convert once)
    print("Running CUDA kernel (half path)...")
    Qh = Q.half()
    Kh = K.half()
    Vh = V.half()
    cuda_out_half = run_attention(
        edge_ptr, edge_idx, mid_nodes, huge_nodes, Q, K, V, Q_fp16=Qh, K_fp16=Kh, V_fp16=Vh, use_half=True
    )

    # Reference: naive PyTorch
    print("Running PyTorch naive reference...")
    ref_out = naive_graph_attention(edge_ptr, edge_idx, Q, K, V)

    # DGL reference
    print("Running DGL implementation...")
    g = csr_to_dgl_graph(edge_ptr, edge_idx, num_nodes)
    dgl_out = dgl_graph_attention(g, Q, K, V)

    # Compare fp32 kernel vs ref
    diff_max = (cuda_out_fp32 - ref_out).abs().max().item()
    diff_mean = (cuda_out_fp32 - ref_out).abs().mean().item()

    print("\nCUDA fp32 vs Naive:")
    print(f"  Max abs diff:  {diff_max:.6e}")
    print(f"  Mean abs diff: {diff_mean:.6e}")
    print("  PASS" if diff_max < 1e-3 else "  FAIL")

    # Compare fp32 kernel vs DGL
    diff_max_dgl = (cuda_out_fp32 - dgl_out).abs().max().item()
    diff_mean_dgl = (cuda_out_fp32 - dgl_out).abs().mean().item()
    print("\nCUDA fp32 vs DGL:")
    print(f"  Max abs diff:  {diff_max_dgl:.6e}")
    print(f"  Mean abs diff: {diff_mean_dgl:.6e}")
    print("  PASS" if diff_max_dgl < 1e-3 else "  FAIL")

    # Compare half kernel vs ref (looser tolerance)
    diff_max_half = (cuda_out_half - ref_out).abs().max().item()
    diff_mean_half = (cuda_out_half - ref_out).abs().mean().item()
    print("\nCUDA half vs Naive:")
    print(f"  Max abs diff:  {diff_max_half:.6e}")
    print(f"  Mean abs diff: {diff_mean_half:.6e}")
    print("  PASS (half tolerance)" if diff_max_half < 5e-3 else "  WARN (half deviation >5e-3)")

    print()


# =====================================================
# Synthetic benchmark
# =====================================================


def benchmark_performance():
    """
    Benchmarks CUDA kernels vs DGL on synthetic graphs.
    We auto-enable the half path for "large" cases to match
    target use (big hubs / big dim).
    """
    print("=" * 120)
    print("PERFORMANCE BENCHMARK: CUDA vs DGL (Synthetic Graphs)")
    print("=" * 120)

    configs = [
        (1000, 64, 10),
        (5000, 64, 20),
        (10000, 64, 30),
        (10000, 128, 30),
    ]

    print(
        f"{'Nodes':<8} {'Dim':<6} {'Deg':<6} {'Edges':<10} "
        f"{'CUDA (ms)':<12} {'DGL (ms)':<12} {'Speedup':<10} "
        f"{'CUDA TF/s':<12} {'DGL TF/s':<12} "
        f"{'CUDA Mem':<12} {'DGL Mem':<12} {'CUDA BW':<12} {'DGL BW':<12}"
    )
    print("-" * 120)

    torch.cuda.synchronize()

    for num_nodes, d, avg_degree in configs:
        # build graph
        edge_ptr, edge_idx = create_random_graph(num_nodes, avg_degree)
        num_edges = edge_idx.shape[0]

        # bucket nodes once
        mid_nodes, huge_nodes = bucket_nodes(edge_ptr, deg_huge=DEG_HUGE)

        # features
        Q = torch.randn(num_nodes, d, device="cuda", dtype=torch.float32)
        K = torch.randn(num_nodes, d, device="cuda", dtype=torch.float32)
        V = torch.randn(num_nodes, d, device="cuda", dtype=torch.float32)

        # precompute half tensors (so conversion is not part of timing)
        Qh = Q.half()
        Kh = K.half()
        Vh = V.half()

        # heuristic: use half path if graph is "large" or dim is wide
        auto_use_half = (num_nodes >= 1_000_000) or (d >= 128)

        # DGL graph
        g = csr_to_dgl_graph(edge_ptr, edge_idx, num_nodes)

        # theoretical memory footprint (same formula as before,
        # assumes float32 sizes for consistency in reporting)
        input_memory = 3 * num_nodes * d * 4  # Q,K,V
        output_memory = num_nodes * d * 4
        edge_memory = (num_nodes + 1) * 4 + num_edges * 4
        total_memory_bytes = input_memory + output_memory + edge_memory

        # ---------------- CUDA warm-up ----------------
        for _ in range(10):
            _ = run_attention(
                edge_ptr,
                edge_idx,
                mid_nodes,
                huge_nodes,
                Q,
                K,
                V,
                Q_fp16=Qh,
                K_fp16=Kh,
                V_fp16=Vh,
                use_half=auto_use_half,
            )
        torch.cuda.synchronize()

        # measure CUDA memory
        _, cuda_mem_alloc, cuda_peak_mem = measure_memory(
            run_attention,
            edge_ptr,
            edge_idx,
            mid_nodes,
            huge_nodes,
            Q,
            K,
            V,
            Q_fp16=Qh,
            K_fp16=Kh,
            V_fp16=Vh,
            use_half=auto_use_half,
        )

        # benchmark CUDA runtime
        num_iters = 100
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(num_iters):
            cuda_out = run_attention(
                edge_ptr,
                edge_idx,
                mid_nodes,
                huge_nodes,
                Q,
                K,
                V,
                Q_fp16=Qh,
                K_fp16=Kh,
                V_fp16=Vh,
                use_half=auto_use_half,
            )
        torch.cuda.synchronize()
        cuda_time = (time.time() - start) / num_iters * 1000.0  # ms

        # ---------------- DGL warm-up ----------------
        for _ in range(10):
            _ = dgl_graph_attention(g, Q, K, V)
        torch.cuda.synchronize()

        # measure DGL memory
        _, dgl_mem_alloc, dgl_peak_mem = measure_memory(dgl_graph_attention, g, Q, K, V)

        # benchmark DGL runtime
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(num_iters):
            dgl_out = dgl_graph_attention(g, Q, K, V)
        torch.cuda.synchronize()
        dgl_time = (time.time() - start) / num_iters * 1000.0  # ms

        # FLOPs estimate: per-edge
        #   dot(K[i],Q[j]) ~ d mul+add ~ 2d
        #   exp/softmax bookkeeping ~ ~2
        #   weighted V agg ~ d mul+add ~ 2d
        # ~4d + 2 per edge
        flops = num_edges * (4 * d + 2)

        cuda_tflops = (flops / (cuda_time * 1e-3)) / 1e12
        dgl_tflops = (flops / (dgl_time * 1e-3)) / 1e12
        speedup = dgl_time / cuda_time

        cuda_bandwidth = calculate_memory_bandwidth(total_memory_bytes, cuda_time)
        dgl_bandwidth = calculate_memory_bandwidth(total_memory_bytes, dgl_time)

        print(
            f"{num_nodes:<8} {d:<6} {avg_degree:<6} {num_edges:<10} "
            f"{cuda_time:<12.3f} {dgl_time:<12.3f} {speedup:<10.2f} "
            f"{cuda_tflops:<12.4f} {dgl_tflops:<12.4f} "
            f"{cuda_peak_mem:<12.1f} {dgl_peak_mem:<12.1f} "
            f"{cuda_bandwidth:<12.1f} {dgl_bandwidth:<12.1f}"
        )

    print("\nMemory units: MB | Bandwidth units: GB/s\n")


# =====================================================
# Real-world benchmark
# =====================================================


def benchmark_real_graphs():
    """
    Benchmark CUDA kernels vs DGL on real datasets.
    We'll automatically use the half path for ogbn-products
    (and generally for very large graphs).
    """
    print("=" * 120)
    print("PERFORMANCE BENCHMARK: CUDA vs DGL (Real-World Graphs)")
    print("=" * 120)

    graphs = load_real_graphs()

    print(
        f"{'Dataset':<15} {'Nodes':<10} {'Edges':<10} {'Dim':<6} "
        f"{'CUDA (ms)':<12} {'DGL (ms)':<12} {'Speedup':<10} "
        f"{'CUDA TF/s':<12} {'DGL TF/s':<12} "
        f"{'CUDA Mem':<12} {'DGL Mem':<12} {'CUDA BW':<12} {'DGL BW':<12}"
    )
    print("-" * 120)

    test_dims = [32, 64, 128, 256]

    for name, info in graphs.items():
        g = info["graph"].to("cuda")
        num_nodes = info["num_nodes"]
        num_edges = info["num_edges"]

        # convert to CSR
        edge_ptr, edge_idx = dgl_to_csr(g)

        # bucket nodes
        mid_nodes, huge_nodes = bucket_nodes(edge_ptr, deg_huge=DEG_HUGE)

        for d in test_dims:
            # init features
            Q = torch.randn(num_nodes, d, device="cuda", dtype=torch.float32)
            K = torch.randn(num_nodes, d, device="cuda", dtype=torch.float32)
            V = torch.randn(num_nodes, d, device="cuda", dtype=torch.float32)

            # half tensors (precompute once)
            Qh = Q.half()
            Kh = K.half()
            Vh = V.half()

            # heuristic for half path:
            # use_half if graph is giant (e.g. ogbn-products) or dim is wide
            auto_use_half = (num_nodes >= 1_000_000) or (d >= 128)

            # same memory model as before (float32 assumption)
            input_memory = 3 * num_nodes * d * 4
            output_memory = num_nodes * d * 4
            edge_memory = (num_nodes + 1) * 4 + num_edges * 4
            total_memory_bytes = input_memory + output_memory + edge_memory

            # # warm-up CUDA
            for _ in range(5):
                _ = run_attention(
                    edge_ptr,
                    edge_idx,
                    mid_nodes,
                    huge_nodes,
                    Q,
                    K,
                    V,
                    Q_fp16=Qh,
                    K_fp16=Kh,
                    V_fp16=Vh,
                    use_half=auto_use_half,
                )
            torch.cuda.synchronize()

            # measure CUDA memory
            _, cuda_mem_alloc, cuda_peak_mem = measure_memory(
                run_attention,
                edge_ptr,
                edge_idx,
                mid_nodes,
                huge_nodes,
                Q,
                K,
                V,
                Q_fp16=Qh,
                K_fp16=Kh,
                V_fp16=Vh,
                use_half=auto_use_half,
            )

            # benchmark CUDA
            num_iters = 10
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(num_iters):
                cuda_out = run_attention(
                    edge_ptr,
                    edge_idx,
                    mid_nodes,
                    huge_nodes,
                    Q,
                    K,
                    V,
                    Q_fp16=Qh,
                    K_fp16=Kh,
                    V_fp16=Vh,
                    use_half=auto_use_half,
                )
            torch.cuda.synchronize()
            cuda_time = (time.time() - start) / num_iters * 1000.0

            # # warm-up DGL
            for _ in range(5):
                _ = dgl_graph_attention(g, Q, K, V)
            torch.cuda.synchronize()

            # measure DGL memory
            _, dgl_mem_alloc, dgl_peak_mem = measure_memory(dgl_graph_attention, g, Q, K, V)

            # benchmark DGL
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(num_iters):
                dgl_out = dgl_graph_attention(g, Q, K, V)
            torch.cuda.synchronize()
            dgl_time = (time.time() - start) / num_iters * 1000.0

            # FLOPs
            flops = num_edges * (4 * d + 2)
            cuda_tflops = (flops / (cuda_time * 1e-3)) / 1e12
            dgl_tflops = (flops / (dgl_time * 1e-3)) / 1e12
            speedup = dgl_time / cuda_time

            cuda_bandwidth = calculate_memory_bandwidth(total_memory_bytes, cuda_time)
            dgl_bandwidth = calculate_memory_bandwidth(total_memory_bytes, dgl_time)

            print(
                f"{name:<15} {num_nodes:<10} {num_edges:<10} {d:<6} "
                f"{cuda_time:<12.3f} {dgl_time:<12.3f} {speedup:<10.2f} "
                f"{cuda_tflops:<12.4f} {dgl_tflops:<12.4f} "
                f"{cuda_peak_mem:<12.1f} {dgl_peak_mem:<12.1f} "
                f"{cuda_bandwidth:<12.1f} {dgl_bandwidth:<12.1f}"
            )

    print("\nMemory units: MB | Bandwidth units: GB/s\n")


# =====================================================
# Main
# =====================================================

if __name__ == "__main__":
    torch.set_grad_enabled(False)

    print("\n" + "=" * 60)
    print("GRAPH ATTENTION CUDA KERNEL - TESTS & BENCHMARKS")
    print("=" * 60 + "\n")

    test_correctness()
    # benchmark_performance()
    benchmark_real_graphs()

    print("=" * 60)
    print("DONE")
    print("=" * 60)
