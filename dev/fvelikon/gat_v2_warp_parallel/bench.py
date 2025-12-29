"""
Benchmark GATv2 Custom Kernel vs DGL Baseline
==============================================

DGL Baseline implements GATv2 using dgl.ops primitives:
1. u_add_v: compute l_i + r_j for each edge
2. LeakyReLU activation
3. Dot product with attention vector
4. edge_softmax: softmax over incoming edges
5. u_mul_e + sum: weighted aggregation
"""

import torch
import torch.nn.functional as F
import dgl
import dgl.ops as ops
import dgl.function as fn
import time
import numpy as np
from torch.utils.cpp_extension import load, load_inline
import os

# =============================================================================
# DGL Baseline Implementation
# =============================================================================


class GATv2_DGL_Ops:
    """GATv2 using DGL's low-level ops for fair comparison"""

    def __init__(self, attn_vec: torch.Tensor, negative_slope: float = 0.2):
        self.attn_vec = attn_vec  # [z]
        self.negative_slope = negative_slope

    def forward(self, g: dgl.DGLGraph, feat_l: torch.Tensor, feat_r: torch.Tensor) -> tuple:
        """
        Args:
            g: DGL graph
            feat_l: Left features [N, z] (typically Wl @ h)
            feat_r: Right features [N, z] (typically Wr @ h)

        Returns:
            h_out: Output features [N, z]
            alpha: Attention weights [E]
        """

        with g.local_scope():
            # Store features on nodes
            g.srcdata["l"] = feat_l
            g.dstdata["r"] = feat_r

            # 1. Compute l_i + r_j for each edge using message passing
            # u_add_v computes src + dst for each edge
            g.apply_edges(fn.u_add_v("l", "r", "lr_sum"))

            # 2. LeakyReLU
            lr_sum = g.edata["lr_sum"]  # [E, z]
            lr_activated = F.leaky_relu(lr_sum, negative_slope=self.negative_slope)

            # 3. Dot product with attention vector: e_ij = a^T @ LeakyReLU(l_i + r_j)
            # attn_vec is [z], lr_activated is [E, z]
            e = (lr_activated * self.attn_vec.unsqueeze(0)).sum(dim=-1)  # [E]

            # 4. Edge softmax (per destination node)
            alpha = ops.edge_softmax(g, e)  # [E]

            # 5. Weighted aggregation: h_i = sum_j alpha_ij * r_j
            g.edata["alpha"] = alpha.unsqueeze(-1)  # [E, 1]
            g.update_all(
                fn.u_mul_e("r", "alpha", "m"),  # message: r_j * alpha_ij
                fn.sum("m", "h"),  # aggregate: sum over neighbors
            )

            h_out = g.dstdata["h"]  # [N, z]

            return h_out, alpha

# =============================================================================
# Custom Kernel Wrapper
# =============================================================================


def load_custom_kernel():
    """JIT compile and load the custom GATv2 kernel"""
    cuda_source = open("/home/fvelikon/projects/cuda_exp/dev/fvelikon/gat_v2_warp_parallel/kernel_draft.cu").read()

    # Add PyTorch bindings
    cpp_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

void GATv2Forward_CSR(
    size_t N, size_t z,
    const float* d_l,
    const float* d_r,
    const int* d_row_ptr,
    const int* d_col_idx,
    const float* d_attn_vec,
    float* d_h_out,
    float* d_alpha_out,
    float negative_slope,
    int max_neighbors,
    cudaStream_t stream);

std::tuple<torch::Tensor, torch::Tensor> gatv2_forward(
    torch::Tensor feat_l,      // [N, z]
    torch::Tensor feat_r,      // [N, z]
    torch::Tensor row_ptr,     // [N+1]
    torch::Tensor col_idx,     // [E]
    torch::Tensor attn_vec,    // [z]
    float negative_slope,
    int max_neighbors
) {
    TORCH_CHECK(feat_l.is_cuda(), "feat_l must be CUDA tensor");
    TORCH_CHECK(feat_l.is_contiguous(), "feat_l must be contiguous");

    int64_t N = feat_l.size(0);
    int64_t z = feat_l.size(1);
    int64_t E = col_idx.size(0);

    auto h_out = torch::zeros({N, z}, feat_l.options());
    auto alpha_out = torch::zeros({E}, feat_l.options());

    GATv2Forward_CSR(
        N, z,
        feat_l.data_ptr<float>(),
        feat_r.data_ptr<float>(),
        row_ptr.data_ptr<int>(),
        col_idx.data_ptr<int>(),
        attn_vec.data_ptr<float>(),
        h_out.data_ptr<float>(),
        alpha_out.data_ptr<float>(),
        negative_slope,
        max_neighbors,
        at::cuda::getCurrentCUDAStream()
    );
    return std::make_tuple(h_out, alpha_out);
}
"""

    module = load_inline(
        name="gatv2_custom",
        cpp_sources=[cpp_source],
        cuda_sources=[cuda_source],
        functions=["gatv2_forward"],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-arch=sm_80"],
        extra_cflags=["-O3"],
        verbose=True,
    )

    return module

CACHE = {}




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
        dataset  = GraphLandDataset(root="/home/fvelikon/projects/cuda_exp/data", name=dataset_name, split="RL")
        g = dgl.graph((dataset[0].edge_index[0], dataset[0].edge_index[1]))
        g = dgl.add_self_loop(g)
        # g = dgl.to_bidirected(g)
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
        # g = dgl.to_bidirected(g)
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
        # g = dgl.to_bidirected(g)
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
        # g = dgl.to_bidirected(g)
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
        # g = dgl.to_bidirected(g)
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
        # g = dgl.to_bidirected(g)
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


def generate_random_graph(N: int, avg_degree: int, seed: int = 42) -> dgl.DGLGraph:
    """Generate random graph matching the CUDA kernel's generation"""
    global CACHE
    if (N, avg_degree) in CACHE:
        return CACHE[(N, avg_degree)]

    np.random.seed(seed)

    src_list = []
    dst_list = []

    for i in range(N):
        # Poisson-distributed degree, clamped
        degree = np.random.poisson(avg_degree)
        degree = max(1, min(avg_degree * 3, degree))

        # Random neighbors (no self-loops)
        neighbors = np.random.choice(N, size=degree, replace=False)
        neighbors = neighbors[neighbors != i]
        neighbors = np.unique(neighbors)

        for j in neighbors:
            src_list.append(j)  # Source (neighbor)
            dst_list.append(i)  # Destination (center node)

    g = dgl.graph((src_list, dst_list), num_nodes=N)
    CACHE[(N, avg_degree)] = g
    return g


def dgl_to_csr(g: dgl.DGLGraph) -> tuple:
    """Convert DGL graph to CSR format for custom kernel"""
    # DGL stores graphs in CSC by default for message passing
    # We need CSR (row = destination, col = source neighbors)
    indptr, indices, _ = g.adj_tensors("csr")

    row_ptr = indptr.int().cuda()
    col_idx = indices.int().cuda()

    # Compute max degree
    degrees = indptr[1:] - indptr[:-1]
    max_degree = int(degrees.max().item())

    return row_ptr, col_idx, max_degree


# =============================================================================
# Benchmark
# =============================================================================


def benchmark_comparison():
    print("\n" + "=" * 100)
    print("GATv2 Benchmark: Custom Kernel vs DGL Baseline")
    print("=" * 100)

    # Try to load custom kernel
    try:
        custom_kernel = load_custom_kernel()
        has_custom = True
        print("✓ Custom kernel loaded successfully")
    except Exception as e:
        has_custom = False
        print(f"✗ Custom kernel failed to load: {e}")


    test_cases = [
        # (N, avg_degree, z)
        (10000, 4, 128),
        (10000, 8, 128),
        (10000, 16, 128),
        (10000, 32, 128),
        (100000, 4, 128),
        (100000, 8, 128),
        (100000, 16, 128),
        (1000000, 32, 128),
        (1000000, 32, 128),
        # Varying z
        (100000, 8, 64),
        (100000, 8, 128),
        (100000, 8, 256),
        (100000, 8, 512),
        (100000, 16, 32),
        (100000, 16, 64),
        (100000, 16, 128),
        (100000, 16, 256),
    ]


    test_cases = [
        # (N, avg_degree, z)
        # Varying z
        (100000, 8, 32),
        (100000, 8, 64),
        (100000, 8, 128),
        (100000, 8, 256),
        (100000, 8, 512),
    ]

    num_warmup = 3
    num_iters = 3
    negative_slope = 0.2

    graphs = load_real_graphs()

    print("\n" + "-" * 100)
    print(
        f"{'Graph':>10} {'N':>10} {'AvgDeg':>8} {'Edges':>10} {'z':>6} | "
        f"{'DGL Ops (ms)':>14} {'Custom (ms)':>14} | "
        f"{'Speedup':>10}"
    )
    print("-" * 100)

    for name, info in graphs.items():
        g = info["graph"].to("cuda")
        num_nodes = info["num_nodes"]
        num_edges = info["num_edges"]

        for _, _, z in test_cases:
            torch.cuda.empty_cache()
            N = num_nodes

            # Generate graph
            # g = generate_random_graph(N, avg_degree)
            E = g.num_edges()
            avg_degree = int(E / N * 2)

            # Generate features
            torch.manual_seed(42)
            feat_l = torch.randn(N, z, device="cuda", dtype=torch.float32) * 0.1
            feat_r = torch.randn(N, z, device="cuda", dtype=torch.float32) * 0.1
            attn_vec = torch.randn(z, device="cuda", dtype=torch.float32) * 0.1

            # =====================================================================
            # DGL Ops Baseline
            # =====================================================================
            dgl_ops = GATv2_DGL_Ops(attn_vec, negative_slope)

            # Warmup

            try:
                for _ in range(num_warmup):
                    _ = dgl_ops.forward(g, feat_l, feat_r)
                torch.cuda.synchronize()

                # Benchmark
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)

                start.record()
                for _ in range(num_iters):
                    h_dgl, alpha_dgl = dgl_ops.forward(g, feat_l, feat_r)
                end.record()
                torch.cuda.synchronize()
                dgl_ops_time = start.elapsed_time(end) / num_iters
            except Exception as e:
                dgl_ops_time = float("nan")
                torch.cuda.empty_cache()

            # =====================================================================
            # Custom Kernel
            # =====================================================================
            if has_custom:
                row_ptr, col_idx, max_degree = dgl_to_csr(g)

                # Warmup
                for _ in range(num_warmup):
                    _ = custom_kernel.gatv2_forward(feat_l, feat_r, row_ptr, col_idx, attn_vec, negative_slope, max_degree)
                torch.cuda.synchronize()

                # Benchmark
                start.record()
                for _ in range(num_iters):
                    h_custom, alpha_custom = custom_kernel.gatv2_forward(
                        feat_l, feat_r, row_ptr, col_idx, attn_vec, negative_slope, max_degree
                    )
                end.record()
                torch.cuda.synchronize()
                torch.cuda.synchronize()  # Force wait for kernel
                # Check for errors
                if torch.cuda.is_available():
                    torch.cuda.current_stream().synchronize()


                custom_time = start.elapsed_time(end) / num_iters
                speedup = dgl_ops_time / custom_time if dgl_ops_time != float('nan') else float('nan')
            else:
                custom_time = float("nan")
                speedup = float("nan")

            # Print results
            print(
                f"{name:>10} {N:>10} {avg_degree:>8} {E:>10} {z:>6} | "
                f"{dgl_ops_time:>14.4f} {custom_time:>14.4f} | "
                f"{speedup:>10.2f}x"
            )

            # Verify correctness (compare attention weights)
            if has_custom and not torch.isnan(torch.tensor(custom_time)):
                # Need to reorder alpha to match DGL's edge ordering
                # This is tricky because DGL and our CSR might have different edge orders
                # For now, just verify softmax property
                alpha_sum_dgl = ops.copy_e_sum(g, alpha_dgl.unsqueeze(-1)).squeeze(-1)

                # Check that alphas sum to 1 for each node with neighbors
                in_degrees = g.in_degrees().float()
                mask = in_degrees > 0
                alpha_sum_check = alpha_sum_dgl[mask]

                if torch.allclose(alpha_sum_check, torch.ones_like(alpha_sum_check), atol=1e-3):
                    verify_status = "✓"
                else:
                    verify_status = "✗"
                # print(f"  Verification: {verify_status}")

        print("-" * 100)


def benchmark_dgl_ops_breakdown():
    """Profile individual DGL operations"""
    print("\n" + "=" * 80)
    print("DGL Ops Breakdown (N=100k, degree=16, z=128)")
    print("=" * 80)

    N, avg_degree, z = 100000, 16, 128
    negative_slope = 0.2

    g = generate_random_graph(N, avg_degree)
    g = g.to("cuda")

    torch.manual_seed(42)
    feat_l = torch.randn(N, z, device="cuda", dtype=torch.float32)
    feat_r = torch.randn(N, z, device="cuda", dtype=torch.float32)
    attn_vec = torch.randn(z, device="cuda", dtype=torch.float32)

    num_warmup = 10
    num_iters = 100

    timings = {}

    with g.local_scope():
        g.srcdata["l"] = feat_l
        g.dstdata["r"] = feat_r

        # 1. u_add_v
        for _ in range(num_warmup):
            g.apply_edges(fn.u_add_v("l", "r", "lr_sum"))
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(num_iters):
            g.apply_edges(fn.u_add_v("l", "r", "lr_sum"))
        end.record()
        torch.cuda.synchronize()
        timings["u_add_v"] = start.elapsed_time(end) / num_iters

        lr_sum = g.edata["lr_sum"]

        # 2. LeakyReLU
        for _ in range(num_warmup):
            _ = F.leaky_relu(lr_sum, negative_slope=negative_slope)
        torch.cuda.synchronize()

        start.record()
        for _ in range(num_iters):
            lr_activated = F.leaky_relu(lr_sum, negative_slope=negative_slope)
        end.record()
        torch.cuda.synchronize()
        timings["leaky_relu"] = start.elapsed_time(end) / num_iters

        # 3. Dot product
        for _ in range(num_warmup):
            _ = (lr_activated * attn_vec.unsqueeze(0)).sum(dim=-1)
        torch.cuda.synchronize()

        start.record()
        for _ in range(num_iters):
            e = (lr_activated * attn_vec.unsqueeze(0)).sum(dim=-1)
        end.record()
        torch.cuda.synchronize()
        timings["dot_product"] = start.elapsed_time(end) / num_iters

        # 4. Edge softmax
        for _ in range(num_warmup):
            _ = ops.edge_softmax(g, e)
        torch.cuda.synchronize()

        start.record()
        for _ in range(num_iters):
            alpha = ops.edge_softmax(g, e)
        end.record()
        torch.cuda.synchronize()
        timings["edge_softmax"] = start.elapsed_time(end) / num_iters

        g.edata["alpha"] = alpha.unsqueeze(-1)

        # 5. Weighted aggregation
        for _ in range(num_warmup):
            g.update_all(fn.u_mul_e("r", "alpha", "m"), fn.sum("m", "h"))
        torch.cuda.synchronize()

        start.record()
        for _ in range(num_iters):
            g.update_all(fn.u_mul_e("r", "alpha", "m"), fn.sum("m", "h"))
        end.record()
        torch.cuda.synchronize()
        timings["aggregation"] = start.elapsed_time(end) / num_iters

    total = sum(timings.values())
    print(f"\n{'Operation':<20} {'Time (ms)':>12} {'Percentage':>12}")
    print("-" * 46)
    for op, t in timings.items():
        print(f"{op:<20} {t:>12.4f} {100 * t / total:>11.1f}%")
    print("-" * 46)
    print(f"{'Total':<20} {total:>12.4f} {100.0:>11.1f}%")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--breakdown", action="store_true", help="Show DGL ops breakdown")
    parser.add_argument("--no-custom", action="store_true", help="Skip custom kernel")
    args = parser.parse_args()

    # Print GPU info
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"DGL version: {dgl.__version__}")
    print(f"PyTorch version: {torch.__version__}")

    if args.breakdown:
        benchmark_dgl_ops_breakdown()

    benchmark_comparison()
