import math
import os
import sys

import dgl
import numpy as np
import torch
from torch.autograd import Function
from torch.utils.cpp_extension import load

sys.path.append("/home/fvelikon/projects/cuda_exp")

from src.data.datasets import DatasetConfig, load_single_graph

# =====================================================
# User-tunable threshold: what we call a "huge" node.
# Must match the kernel's DEG_HUGE in C++.
# =====================================================
DEG_HUGE = 128

# =====================================================
# JIT compile the CUDA extension
# =====================================================
print("Compiling CUDA extension...")
current_dir = os.path.dirname(os.path.abspath(__file__))
graph_attention_cuda = load(
    name="graph_attention_cuda",
    sources=[os.path.join("/home/fvelikon/projects/cuda_exp/src/backends/cuda_backend", "graph_transformer.cu")],
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
        print(f"Nodes with logsumexp=-inf: {(logsumexp == -float('inf')).sum().item()}")
        print(f"Nodes with non-finite logsumexp: {(~torch.isfinite(logsumexp)).sum().item()}")
        print(f"Mid nodes count: {len(mid_nodes)}, Huge nodes count: {len(huge_nodes)}")

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

        return None, None, None, None, dQ, dK, dV


def graph_attention_forward_backward(edge_ptr, edge_idx, mid_nodes, huge_nodes, Q, K, V):
    """
    Wrapper that enables autograd for the CUDA kernel.
    """
    return GraphAttentionFunction.apply(edge_ptr, edge_idx, mid_nodes, huge_nodes, Q, K, V)


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


def bucket_nodes(edge_ptr, deg_huge=DEG_HUGE):
    """
    Split nodes into buckets:
      - mid_nodes: 0 < deg <= deg_huge
      - huge_nodes: deg > deg_huge
    Returns (mid_nodes:int32[cuda], huge_nodes:int32[cuda])
    """
    # deg[i] = number of inbound edges for node i
    deg = (edge_ptr[1:] - edge_ptr[:-1]).to(torch.int32)  # cuda int32

    mid_mask = (deg >= 0) & (deg <= deg_huge)
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


# =====================================================
# Correctness test - Forward + Backward
# =====================================================


def edge_index_to_csr(
    num_nodes: int, edge_index: torch.Tensor, edge_weight: torch.Tensor | None, transposed: bool = True
):
    if transposed:
        rows = edge_index[1]
        cols = edge_index[0]
    else:
        rows = edge_index[0]
        cols = edge_index[1]

    N = num_nodes
    # Sort edges by (row, col) for a canonical CSR
    perm = (rows * N + cols).argsort()
    rows = rows[perm]
    cols = cols[perm]
    w = edge_weight[perm] if edge_weight is not None else None

    # Build CSR row pointers
    counts = torch.bincount(rows, minlength=N)
    row_ptr = torch.zeros(N + 1, dtype=torch.long, device=rows.device)
    row_ptr[1:] = counts.cumsum(0)

    # Store graph as (row_pointers, column_indices, edge_weight) on default device
    graph = (
        row_ptr,
        cols,
        w,
    )
    return graph


def test_correctness():
    print("=" * 80)
    print("CORRECTNESS TEST - FORWARD + BACKWARD")
    print("=" * 80)

    for num_nodes in [8, 64, 100, 128, 150, 256]:
        d = 128
        print(f"Graph size: {num_nodes} nodes, dim {d}")

        src = torch.arange(num_nodes)
        dst = torch.tensor([0] * num_nodes)

        g = dgl.graph(
            (
                src,
                dst,
            ),
            num_nodes=num_nodes,
        ).to("cuda")

        edge_index = torch.stack([src, dst])

        row_ptr, cols, _edge_weights = edge_index_to_csr(
            num_nodes=num_nodes, edge_index=edge_index, edge_weight=None, transposed=True
        )
        del _edge_weights
        mid_nodes_, huge_nodes_ = bucket_nodes(row_ptr, deg_huge=DEG_HUGE)
        graph = (row_ptr.int().cuda(), cols.int().cuda(), mid_nodes_.int().cuda(), huge_nodes_.int().cuda())

        num_edges = edge_index.shape[1]
        print(f"Number of edges: {num_edges}")
        print(f"Average degree: {num_edges / num_nodes:.2f}\n")

        # features (require grad for backward)
        Q = torch.ones((num_nodes, d), device="cuda", dtype=torch.float32, requires_grad=True)
        K = torch.ones((num_nodes, d), device="cuda", dtype=torch.float32, requires_grad=True)
        V = torch.ones((num_nodes, d), device="cuda", dtype=torch.float32, requires_grad=True)

        print(f"{Q=}\n{K=}\n{V=}")

        # DGL copies (separate computation graph)
        Q_dgl = Q.detach().clone().requires_grad_(True)
        K_dgl = K.detach().clone().requires_grad_(True)
        V_dgl = V.detach().clone().requires_grad_(True)

        # ========================================
        # Forward pass comparison
        # ========================================
        print("=" * 80)
        print("FORWARD PASS")
        print("=" * 80)

        # CUDA forward
        print("Running CUDA kernel...")
        cuda_out = graph_attention_forward_backward(*graph, Q, K, V)

        # DGL forward
        print("Running DGL implementation...")
        dgl_out = dgl_graph_attention(g, Q_dgl, K_dgl, V_dgl)
        print(f"{cuda_out=}")
        print(f"{dgl_out=}")

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
        grad_output = torch.ones_like(cuda_out)

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

        print(f"{dgl_dV=}")
        print(f"{cuda_dV=}")

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


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("GRAPH ATTENTION CUDA KERNEL - TESTS & BENCHMARKS")
    print("=" * 80 + "\n")

    test_correctness()

    print("=" * 80)
    print("DONE")
    print("=" * 80)
