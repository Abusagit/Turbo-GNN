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
# JIT compile the CUDA extension
# =====================================================
from src.backends.cuda_backend.min_aggr.utils import min_aggr, min_aggr_forward  # , MinAggr

# min_aggr = MinAggr()

# =====================================================
# PyTorch Autograd Function Wrapper
# =====================================================


# =====================================================
# Utility functions
# =====================================================


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


def dgl_min_aggr(g, x):
    out = dgl.ops.copy_u_min(g, x)
    out[out.isinf()] = 0
    return out


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
        (50, 64, 10),
        # (5000, 64, 20),
        # (10000, 64, 30),
        # (10000, 128, 30),
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

        g = csr_to_dgl_graph(edge_ptr, edge_idx, num_nodes).to("cuda")

        x_ours = torch.randn(num_nodes, d, device="cuda", requires_grad=True)
        x_dgl = x_ours.detach().clone().requires_grad_(True)

        grad_output = torch.randn(num_nodes, d, device="cuda", dtype=torch.float32)

        # # ==================== CUDA TIMING ====================

        # Warm-up
        for _ in range(10):
            x_ours.grad = None
            out_ours = min_aggr(edge_ptr, edge_idx, x_ours)
            out_ours.backward(grad_output)
        torch.cuda.synchronize()

        # Forward only
        num_iters = 100
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(num_iters):
            with torch.no_grad():
                _ = min_aggr_forward(edge_ptr, edge_idx, x_ours.detach())
        torch.cuda.synchronize()
        cuda_fwd_time = (time.time() - start) / num_iters * 1000.0

        # Backward only (forward already computed)
        x_ours.grad = None
        out_ours = min_aggr(edge_ptr, edge_idx, x_ours)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(num_iters):
            x_ours.grad = None
            out_ours.backward(grad_output, retain_graph=True)
        torch.cuda.synchronize()
        cuda_bwd_time = (time.time() - start) / num_iters * 1000.0

        # Forward + Backward combined
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(num_iters):
            x_ours.grad = None
            out_ours = min_aggr(edge_ptr, edge_idx, x_ours)
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


# =====================================================
# Real-world benchmark
# =====================================================


# def benchmark_real_graphs():
#     """
#     Benchmark CUDA kernels vs DGL on real datasets with memory usage tracking.
#     """
#     print("=" * 200)
#     print("PERFORMANCE BENCHMARK: CUDA vs DGL (Real-World Graphs)")
#     print("=" * 200)

#     graphs = load_real_graphs()

#     header = (
#         f"{'Dataset':<15} {'Nodes':<10} {'Edges':<10} {'Dim':<6} | "
#         f"{'FWD CUDA':<12} {'FWD DGL':<12} {'Speedup':<10} | "
#         f"{'BWD CUDA':<12} {'BWD DGL':<12} {'Speedup':<10} | "
#         f"{'TOTAL CUDA':<12} {'TOTAL DGL':<12} {'Speedup':<10} | "
#         f"{'MEM CUDA':<12} {'MEM DGL':<12} {'Ratio':<10}"
#     )
#     print(header)
#     print("-" * 200)

#     test_dims = [32, 64, 128, 256]

#     for name, info in graphs.items():
#         g = info["graph"].to("cuda")
#         num_nodes = info["num_nodes"]
#         num_edges = info["num_edges"]

#         # convert to CSR
#         edge_ptr, edge_idx = dgl_to_csr(g)

#         # bucket nodes
#         mid_nodes, huge_nodes = bucket_nodes(edge_ptr, deg_huge=DEG_HUGE)

#         for d in test_dims:
#             # init features with gradients
#             Q = torch.randn(num_nodes, d, device="cuda", dtype=torch.float32, requires_grad=True)
#             K = torch.randn(num_nodes, d, device="cuda", dtype=torch.float32, requires_grad=True)
#             V = torch.randn(num_nodes, d, device="cuda", dtype=torch.float32, requires_grad=True)

#             Q_dgl = Q.detach().clone().requires_grad_(True)
#             K_dgl = K.detach().clone().requires_grad_(True)
#             V_dgl = V.detach().clone().requires_grad_(True)

#             grad_output = torch.randn(num_nodes, d, device="cuda", dtype=torch.float32)

#             # ==================== CUDA TIMING ====================

#             # Warm-up
#             for _ in range(5):
#                 Q.grad = None
#                 K.grad = None
#                 V.grad = None
#                 out = graph_attention_forward_backward(edge_ptr, edge_idx, mid_nodes, huge_nodes, Q, K, V)
#                 out.backward(grad_output)
#             torch.cuda.synchronize()

#             num_iters = 10

#             # Forward only
#             torch.cuda.synchronize()
#             start = time.time()
#             for _ in range(num_iters):
#                 with torch.no_grad():
#                     out = graph_attention_forward_backward(
#                         edge_ptr, edge_idx, mid_nodes, huge_nodes, Q.detach(), K.detach(), V.detach()
#                     )
#             torch.cuda.synchronize()
#             cuda_fwd_time = (time.time() - start) / num_iters * 1000.0

#             # Backward only
#             Q.grad = None
#             K.grad = None
#             V.grad = None
#             out = graph_attention_forward_backward(edge_ptr, edge_idx, mid_nodes, huge_nodes, Q, K, V)

#             torch.cuda.synchronize()
#             start = time.time()
#             for _ in range(num_iters):
#                 Q.grad = None
#                 K.grad = None
#                 V.grad = None
#                 out.backward(grad_output, retain_graph=True)
#             torch.cuda.synchronize()
#             cuda_bwd_time = (time.time() - start) / num_iters * 1000.0

#             # Forward + Backward combined (with memory tracking)
#             torch.cuda.reset_peak_memory_stats()
#             torch.cuda.synchronize()

#             start = time.time()
#             for _ in range(num_iters):
#                 Q.grad = None
#                 K.grad = None
#                 V.grad = None
#                 out = graph_attention_forward_backward(edge_ptr, edge_idx, mid_nodes, huge_nodes, Q, K, V)
#                 out.backward(grad_output)
#             torch.cuda.synchronize()
#             cuda_total_time = (time.time() - start) / num_iters * 1000.0

#             cuda_peak_mem = torch.cuda.max_memory_allocated()
#             cuda_mem_usage = cuda_peak_mem / (1024**3)  # Convert to Gb

#             # ==================== DGL TIMING ====================

#             # Warm-up
#             for _ in range(5):
#                 Q_dgl.grad = None
#                 K_dgl.grad = None
#                 V_dgl.grad = None
#                 out = dgl_graph_attention(g, Q_dgl, K_dgl, V_dgl)
#                 out.backward(grad_output)
#             torch.cuda.synchronize()

#             # Forward only
#             torch.cuda.synchronize()
#             start = time.time()
#             for _ in range(num_iters):
#                 with torch.no_grad():
#                     out = dgl_graph_attention(g, Q_dgl.detach(), K_dgl.detach(), V_dgl.detach())
#             torch.cuda.synchronize()
#             dgl_fwd_time = (time.time() - start) / num_iters * 1000.0

#             # Backward only
#             Q_dgl.grad = None
#             K_dgl.grad = None
#             V_dgl.grad = None
#             out = dgl_graph_attention(g, Q_dgl, K_dgl, V_dgl)

#             torch.cuda.synchronize()
#             start = time.time()
#             for _ in range(num_iters):
#                 Q_dgl.grad = None
#                 K_dgl.grad = None
#                 V_dgl.grad = None
#                 out.backward(grad_output, retain_graph=True)
#             torch.cuda.synchronize()
#             dgl_bwd_time = (time.time() - start) / num_iters * 1000.0

#             # Forward + Backward combined (with memory tracking)
#             torch.cuda.reset_peak_memory_stats()
#             torch.cuda.synchronize()

#             start = time.time()
#             for _ in range(num_iters):
#                 Q_dgl.grad = None
#                 K_dgl.grad = None
#                 V_dgl.grad = None
#                 out = dgl_graph_attention(g, Q_dgl, K_dgl, V_dgl)
#                 out.backward(grad_output)
#             torch.cuda.synchronize()
#             dgl_total_time = (time.time() - start) / num_iters * 1000.0

#             dgl_peak_mem = torch.cuda.max_memory_allocated()
#             dgl_mem_usage = dgl_peak_mem / (1024**3)  # Convert to MB

#             # Compute speedups and memory ratio
#             fwd_speedup = dgl_fwd_time / cuda_fwd_time
#             bwd_speedup = dgl_bwd_time / cuda_bwd_time
#             total_speedup = dgl_total_time / cuda_total_time
#             mem_ratio = dgl_mem_usage / cuda_mem_usage if cuda_mem_usage > 0 else 0

#             print(
#                 f"{name:<15} {num_nodes:<10} {num_edges:<10} {d:<6} | "
#                 f"{cuda_fwd_time:<12.3f} {dgl_fwd_time:<12.3f} {fwd_speedup:<10.2f} | "
#                 f"{cuda_bwd_time:<12.3f} {dgl_bwd_time:<12.3f} {bwd_speedup:<10.2f} | "
#                 f"{cuda_total_time:<12.3f} {dgl_total_time:<12.3f} {total_speedup:<10.2f} | "
#                 f"{cuda_mem_usage:<12.3f} {dgl_mem_usage:<12.3f} {mem_ratio:<10.2f}x"
#             )

#     print("\nTime units: milliseconds | Memory units: GB | Ratio: DGL/CUDA memory usage\n")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("GRAPH ATTENTION CUDA KERNEL - BENCHMARKS")
    print("=" * 80 + "\n")

    benchmark_performance()

    print("=" * 80)
    print("DONE")
    print("=" * 80)
