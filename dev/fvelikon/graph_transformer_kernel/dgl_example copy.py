import math
import os
import time

import dgl
import dgl.function as fn
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Function
from torch.utils.cpp_extension import load

# =====================================================
# User-tunable threshold: what we call a "huge" node.
# Must match the kernel's DEG_HUGE in C++.
# =====================================================
DEG_HUGE = 256

# =====================================================
# JIT compile the CUDA extension
# =====================================================
print("Compiling CUDA extension...")
current_dir = os.path.dirname(os.path.abspath(__file__))
graph_attention_cuda = load(
    name="graph_attention_cuda",
    sources=[os.path.join("/home/fvelikon/projects/cuda_exp/src/backends/cuda_backend", "graph_transformer copy.cu")],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-arch=sm_80"],
    verbose=True,
)
print("Compilation complete!\n")


# =====================================================
# PyTorch Autograd Function Wrapper
# =====================================================


class GraphAttentionFunction(Function):
    @staticmethod
    def forward(ctx, edge_ptr, edge_idx, mid_nodes, huge_nodes, Q, K, V):
        """
        Forward pass wrapper.
        Returns: (output, logsumexp)
        """
        out, logsumexp = graph_attention_cuda.forward_buckets(edge_ptr, edge_idx, mid_nodes, huge_nodes, Q, K, V)

        # Save for backward
        ctx.save_for_backward(edge_ptr, edge_idx, mid_nodes, huge_nodes, Q, K, V, out, logsumexp)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass wrapper.
        Returns: (None, None, None, None, dQ, dK, dV)
        """
        edge_ptr, edge_idx, mid_nodes, huge_nodes, Q, K, V, out, logsumexp = ctx.saved_tensors

        dQ, dK, dV = graph_attention_cuda.backward_buckets(
            edge_ptr, edge_idx, mid_nodes, huge_nodes, Q, K, V, out, grad_output, logsumexp
        )

        # Return gradients for all inputs (None for non-tensor args)
        return None, None, None, None, dQ, dK, dV


def graph_attention_forward_backward(edge_ptr, edge_idx, mid_nodes, huge_nodes, Q, K, V):
    """
    Wrapper that enables autograd for the CUDA kernel.
    """
    return GraphAttentionFunction.apply(edge_ptr, edge_idx, mid_nodes, huge_nodes, Q, K, V)


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

    for dataset_name in [
        "hm-categories",
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
        dataset = GraphLandDataset(root="/home/fvelikon/projects/cuda_exp/data", name=dataset_name, split="RL")
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
    """
    return (num_bytes / 1e9) / (time_ms / 1000.0)


# =====================================================
# Correctness test - Forward + Backward
# =====================================================


def test_correctness():
    print("=" * 80)
    print("CORRECTNESS TEST - FORWARD + BACKWARD")
    print("=" * 80)

    num_nodes = 100
    d = 64
    avg_degree = 8
    print(f"Graph size: {num_nodes} nodes, dim {d}")

    edge_ptr, edge_idx = create_random_graph(num_nodes, avg_degree)
    num_edges = edge_idx.shape[0]
    print(f"Number of edges: {num_edges}")
    print(f"Average degree: {num_edges / num_nodes:.2f}\n")

    # features (require grad for backward)
    Q = torch.randn(num_nodes, d, device="cuda", dtype=torch.float32, requires_grad=True)
    K = torch.randn(num_nodes, d, device="cuda", dtype=torch.float32, requires_grad=True)
    V = torch.randn(num_nodes, d, device="cuda", dtype=torch.float32, requires_grad=True)

    # DGL copies (separate computation graph)
    Q_dgl = Q.detach().clone().requires_grad_(True)
    K_dgl = K.detach().clone().requires_grad_(True)
    V_dgl = V.detach().clone().requires_grad_(True)

    # bucket nodes
    mid_nodes, huge_nodes = bucket_nodes(edge_ptr, deg_huge=DEG_HUGE)

    # ========================================
    # Forward pass comparison
    # ========================================
    print("=" * 80)
    print("FORWARD PASS")
    print("=" * 80)

    # CUDA forward
    print("Running CUDA kernel...")
    cuda_out = graph_attention_forward_backward(edge_ptr, edge_idx, mid_nodes, huge_nodes, Q, K, V)

    # DGL forward
    print("Running DGL implementation...")
    g = csr_to_dgl_graph(edge_ptr, edge_idx, num_nodes)
    dgl_out = dgl_graph_attention(g, Q_dgl, K_dgl, V_dgl)

    # Compare outputs
    diff_max = (cuda_out - dgl_out).abs().max().item()
    diff_mean = (cuda_out - dgl_out).abs().mean().item()

    print("\nForward Output Comparison (CUDA vs DGL):")
    print(f"  Max abs diff:  {diff_max:.6e}")
    print(f"  Mean abs diff: {diff_mean:.6e}")
    print(f"  Status: {'✓ PASS' if diff_max < 1e-3 else '✗ FAIL'}")

    # ========================================
    # Backward pass comparison
    # ========================================
    print("\n" + "=" * 80)
    print("BACKWARD PASS")
    print("=" * 80)

    # Create identical upstream gradient
    grad_output = torch.randn_like(cuda_out)

    # CUDA backward
    print("Running CUDA backward...")
    cuda_out.backward(grad_output)
    cuda_dQ = Q.grad.clone()
    cuda_dK = K.grad.clone()
    cuda_dV = V.grad.clone()

    # DGL backward
    print("Running DGL backward...")
    dgl_out.backward(grad_output)
    dgl_dQ = Q_dgl.grad.clone()
    dgl_dK = K_dgl.grad.clone()
    dgl_dV = V_dgl.grad.clone()

    # Compare gradients
    print("\nGradient Comparison (CUDA vs DGL):")

    dQ_diff_max = (cuda_dQ - dgl_dQ).abs().max().item()
    dQ_diff_mean = (cuda_dQ - dgl_dQ).abs().mean().item()
    print("\ndQ:")
    print(f"  Max abs diff:  {dQ_diff_max:.6e}")
    print(f"  Mean abs diff: {dQ_diff_mean:.6e}")
    print(f"  Status: {'✓ PASS' if dQ_diff_max < 1e-3 else '✗ FAIL'}")

    dK_diff_max = (cuda_dK - dgl_dK).abs().max().item()
    dK_diff_mean = (cuda_dK - dgl_dK).abs().mean().item()
    print("\ndK:")
    print(f"  Max abs diff:  {dK_diff_max:.6e}")
    print(f"  Mean abs diff: {dK_diff_mean:.6e}")
    print(f"  Status: {'✓ PASS' if dK_diff_max < 1e-3 else '✗ FAIL'}")

    dV_diff_max = (cuda_dV - dgl_dV).abs().max().item()
    dV_diff_mean = (cuda_dV - dgl_dV).abs().mean().item()
    print("\ndV:")
    print(f"  Max abs diff:  {dV_diff_max:.6e}")
    print(f"  Mean abs diff: {dV_diff_mean:.6e}")
    print(f"  Status: {'✓ PASS' if dV_diff_max < 1e-3 else '✗ FAIL'}")

    # Overall verdict
    all_pass = diff_max < 1e-3 and dQ_diff_max < 1e-3 and dK_diff_max < 1e-3 and dV_diff_max < 1e-3

    print("\n" + "=" * 80)
    print(f"Overall: {'✓ ALL TESTS PASSED' if all_pass else '✗ SOME TESTS FAILED'}")
    print("=" * 80 + "\n")


# =====================================================
# Performance benchmark - Synthetic graphs
# =====================================================


def benchmark_performance():
    """
    Benchmarks CUDA kernels vs DGL on synthetic graphs.
    Measures forward, backward, and combined times separately.
    """
    print("=" * 160)
    print("PERFORMANCE BENCHMARK: CUDA vs DGL (Synthetic Graphs)")
    print("=" * 160)

    configs = [
        (1000, 64, 10),
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
        # build graph
        edge_ptr, edge_idx = create_random_graph(num_nodes, avg_degree)
        num_edges = edge_idx.shape[0]

        # bucket nodes once
        mid_nodes, huge_nodes = bucket_nodes(edge_ptr, deg_huge=DEG_HUGE)

        # features with gradients
        Q = torch.randn(num_nodes, d, device="cuda", dtype=torch.float32, requires_grad=True)
        K = torch.randn(num_nodes, d, device="cuda", dtype=torch.float32, requires_grad=True)
        V = torch.randn(num_nodes, d, device="cuda", dtype=torch.float32, requires_grad=True)

        Q_dgl = Q.detach().clone().requires_grad_(True)
        K_dgl = K.detach().clone().requires_grad_(True)
        V_dgl = V.detach().clone().requires_grad_(True)

        # DGL graph
        g = csr_to_dgl_graph(edge_ptr, edge_idx, num_nodes)

        grad_output = torch.randn(num_nodes, d, device="cuda", dtype=torch.float32)

        # ==================== CUDA TIMING ====================

        # Warm-up
        for _ in range(10):
            Q.grad = None
            K.grad = None
            V.grad = None
            out = graph_attention_forward_backward(edge_ptr, edge_idx, mid_nodes, huge_nodes, Q, K, V)
            out.backward(grad_output)
        torch.cuda.synchronize()

        # Forward only
        num_iters = 100
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(num_iters):
            with torch.no_grad():
                out = graph_attention_forward_backward(
                    edge_ptr, edge_idx, mid_nodes, huge_nodes, Q.detach(), K.detach(), V.detach()
                )
        torch.cuda.synchronize()
        cuda_fwd_time = (time.time() - start) / num_iters * 1000.0

        # Backward only (forward already computed)
        Q.grad = None
        K.grad = None
        V.grad = None
        out = graph_attention_forward_backward(edge_ptr, edge_idx, mid_nodes, huge_nodes, Q, K, V)

        torch.cuda.synchronize()
        start = time.time()
        for _ in range(num_iters):
            Q.grad = None
            K.grad = None
            V.grad = None
            out.backward(grad_output, retain_graph=True)
        torch.cuda.synchronize()
        cuda_bwd_time = (time.time() - start) / num_iters * 1000.0

        # Forward + Backward combined
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(num_iters):
            Q.grad = None
            K.grad = None
            V.grad = None
            out = graph_attention_forward_backward(edge_ptr, edge_idx, mid_nodes, huge_nodes, Q, K, V)
            out.backward(grad_output)
        torch.cuda.synchronize()
        cuda_total_time = (time.time() - start) / num_iters * 1000.0

        # ==================== DGL TIMING ====================

        # Warm-up
        for _ in range(10):
            Q_dgl.grad = None
            K_dgl.grad = None
            V_dgl.grad = None
            out = dgl_graph_attention(g, Q_dgl, K_dgl, V_dgl)
            out.backward(grad_output)
        torch.cuda.synchronize()

        # Forward only
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(num_iters):
            with torch.no_grad():
                out = dgl_graph_attention(g, Q_dgl.detach(), K_dgl.detach(), V_dgl.detach())
        torch.cuda.synchronize()
        dgl_fwd_time = (time.time() - start) / num_iters * 1000.0

        # Backward only
        Q_dgl.grad = None
        K_dgl.grad = None
        V_dgl.grad = None
        out = dgl_graph_attention(g, Q_dgl, K_dgl, V_dgl)

        torch.cuda.synchronize()
        start = time.time()
        for _ in range(num_iters):
            Q_dgl.grad = None
            K_dgl.grad = None
            V_dgl.grad = None
            out.backward(grad_output, retain_graph=True)
        torch.cuda.synchronize()
        dgl_bwd_time = (time.time() - start) / num_iters * 1000.0

        # Forward + Backward combined
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(num_iters):
            Q_dgl.grad = None
            K_dgl.grad = None
            V_dgl.grad = None
            out = dgl_graph_attention(g, Q_dgl, K_dgl, V_dgl)
            out.backward(grad_output)
        torch.cuda.synchronize()
        dgl_total_time = (time.time() - start) / num_iters * 1000.0

        # Compute speedups
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


# =====================================================
# Real-world benchmark
# =====================================================


def benchmark_real_graphs():
    """
    Benchmark CUDA kernels vs DGL on real datasets with memory usage tracking.
    """
    print("=" * 200)
    print("PERFORMANCE BENCHMARK: CUDA vs DGL (Real-World Graphs)")
    print("=" * 200)

    graphs = load_real_graphs()

    header = (
        f"{'Dataset':<15} {'Nodes':<10} {'Edges':<10} {'Dim':<6} | "
        f"{'FWD CUDA':<12} {'FWD DGL':<12} {'Speedup':<10} | "
        f"{'BWD CUDA':<12} {'BWD DGL':<12} {'Speedup':<10} | "
        f"{'TOTAL CUDA':<12} {'TOTAL DGL':<12} {'Speedup':<10} | "
        f"{'MEM CUDA':<12} {'MEM DGL':<12} {'Ratio':<10}"
    )
    print(header)
    print("-" * 200)

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
            # init features with gradients
            Q = torch.randn(num_nodes, d, device="cuda", dtype=torch.float32, requires_grad=True)
            K = torch.randn(num_nodes, d, device="cuda", dtype=torch.float32, requires_grad=True)
            V = torch.randn(num_nodes, d, device="cuda", dtype=torch.float32, requires_grad=True)

            Q_dgl = Q.detach().clone().requires_grad_(True)
            K_dgl = K.detach().clone().requires_grad_(True)
            V_dgl = V.detach().clone().requires_grad_(True)

            grad_output = torch.randn(num_nodes, d, device="cuda", dtype=torch.float32)

            # ==================== CUDA TIMING ====================

            # Warm-up
            for _ in range(5):
                Q.grad = None
                K.grad = None
                V.grad = None
                out = graph_attention_forward_backward(edge_ptr, edge_idx, mid_nodes, huge_nodes, Q, K, V)
                out.backward(grad_output)
            torch.cuda.synchronize()

            num_iters = 10

            # Forward only
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(num_iters):
                with torch.no_grad():
                    out = graph_attention_forward_backward(
                        edge_ptr, edge_idx, mid_nodes, huge_nodes, Q.detach(), K.detach(), V.detach()
                    )
            torch.cuda.synchronize()
            cuda_fwd_time = (time.time() - start) / num_iters * 1000.0

            # Backward only
            Q.grad = None
            K.grad = None
            V.grad = None
            out = graph_attention_forward_backward(edge_ptr, edge_idx, mid_nodes, huge_nodes, Q, K, V)

            torch.cuda.synchronize()
            start = time.time()
            for _ in range(num_iters):
                Q.grad = None
                K.grad = None
                V.grad = None
                out.backward(grad_output, retain_graph=True)
            torch.cuda.synchronize()
            cuda_bwd_time = (time.time() - start) / num_iters * 1000.0

            # Forward + Backward combined (with memory tracking)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

            start = time.time()
            for _ in range(num_iters):
                Q.grad = None
                K.grad = None
                V.grad = None
                out = graph_attention_forward_backward(edge_ptr, edge_idx, mid_nodes, huge_nodes, Q, K, V)
                out.backward(grad_output)
            torch.cuda.synchronize()
            cuda_total_time = (time.time() - start) / num_iters * 1000.0

            cuda_peak_mem = torch.cuda.max_memory_allocated()
            cuda_mem_usage = cuda_peak_mem / (1024**3)  # Convert to Gb

            # ==================== DGL TIMING ====================

            # Warm-up
            for _ in range(5):
                Q_dgl.grad = None
                K_dgl.grad = None
                V_dgl.grad = None
                out = dgl_graph_attention(g, Q_dgl, K_dgl, V_dgl)
                out.backward(grad_output)
            torch.cuda.synchronize()

            # Forward only
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(num_iters):
                with torch.no_grad():
                    out = dgl_graph_attention(g, Q_dgl.detach(), K_dgl.detach(), V_dgl.detach())
            torch.cuda.synchronize()
            dgl_fwd_time = (time.time() - start) / num_iters * 1000.0

            # Backward only
            Q_dgl.grad = None
            K_dgl.grad = None
            V_dgl.grad = None
            out = dgl_graph_attention(g, Q_dgl, K_dgl, V_dgl)

            torch.cuda.synchronize()
            start = time.time()
            for _ in range(num_iters):
                Q_dgl.grad = None
                K_dgl.grad = None
                V_dgl.grad = None
                out.backward(grad_output, retain_graph=True)
            torch.cuda.synchronize()
            dgl_bwd_time = (time.time() - start) / num_iters * 1000.0

            # Forward + Backward combined (with memory tracking)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

            start = time.time()
            for _ in range(num_iters):
                Q_dgl.grad = None
                K_dgl.grad = None
                V_dgl.grad = None
                out = dgl_graph_attention(g, Q_dgl, K_dgl, V_dgl)
                out.backward(grad_output)
            torch.cuda.synchronize()
            dgl_total_time = (time.time() - start) / num_iters * 1000.0

            dgl_peak_mem = torch.cuda.max_memory_allocated()
            dgl_mem_usage = dgl_peak_mem / (1024**3)  # Convert to MB

            # Compute speedups and memory ratio
            fwd_speedup = dgl_fwd_time / cuda_fwd_time
            bwd_speedup = dgl_bwd_time / cuda_bwd_time
            total_speedup = dgl_total_time / cuda_total_time
            mem_ratio = dgl_mem_usage / cuda_mem_usage if cuda_mem_usage > 0 else 0

            print(
                f"{name:<15} {num_nodes:<10} {num_edges:<10} {d:<6} | "
                f"{cuda_fwd_time:<12.3f} {dgl_fwd_time:<12.3f} {fwd_speedup:<10.2f} | "
                f"{cuda_bwd_time:<12.3f} {dgl_bwd_time:<12.3f} {bwd_speedup:<10.2f} | "
                f"{cuda_total_time:<12.3f} {dgl_total_time:<12.3f} {total_speedup:<10.2f} | "
                f"{cuda_mem_usage:<12.1f} {dgl_mem_usage:<12.1f} {mem_ratio:<10.2f}x"
            )

    print("\nTime units: milliseconds | Memory units: GB | Ratio: DGL/CUDA memory usage\n")


# =====================================================
# Main
# =====================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("GRAPH ATTENTION CUDA KERNEL - TESTS & BENCHMARKS")
    print("=" * 80 + "\n")

    test_correctness()
    benchmark_performance()
    benchmark_real_graphs()

    print("=" * 80)
    print("DONE")
    print("=" * 80)
