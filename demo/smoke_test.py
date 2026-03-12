#!/usr/bin/env python3
"""Smoke test for turbo_gnn: verifies all kernels on random graphs in fp32 and fp16."""

import sys
import traceback

import torch

from turbo_gnn import (
    AdjacencyForwardBackwardWithNodeBuckets,
    csr_SPMM_normalized,
    gatv2_aggr,
    graph_transformer_aggr,
    reduction_aggr,
    spmm_aggr,
)

# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------


def make_random_edge_index(num_nodes: int, avg_degree: int = 6, seed: int = 42) -> torch.Tensor:
    """Generate a random undirected graph with self-loops as a COO edge_index [2, E]."""
    gen = torch.Generator().manual_seed(seed)
    num_directed = num_nodes * avg_degree
    src = torch.randint(0, num_nodes, (num_directed,), generator=gen)
    dst = torch.randint(0, num_nodes, (num_directed,), generator=gen)
    # Make undirected
    src_all = torch.cat([src, dst])
    dst_all = torch.cat([dst, src])
    # Add self-loops
    self_loop = torch.arange(num_nodes)
    src_all = torch.cat([src_all, self_loop])
    dst_all = torch.cat([dst_all, self_loop])
    # Deduplicate
    edge_index = torch.stack([src_all, dst_all], dim=0)
    edge_index = torch.unique(edge_index, dim=1)
    return edge_index


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

results: list[tuple[str, bool]] = []


def run_test(name: str, fn):
    """Run a single test, catch exceptions, record result."""
    try:
        fn()
        results.append((name, True))
        print(f"  PASS  {name}")
    except Exception:
        results.append((name, False))
        print(f"  FAIL  {name}")
        traceback.print_exc()
        print()


def check_output(out: torch.Tensor, expected_shape: tuple, label: str):
    """Assert shape is correct and no NaNs."""
    assert out.shape == expected_shape, f"{label}: expected shape {expected_shape}, got {out.shape}"
    assert not torch.isnan(out).any(), f"{label}: output contains NaN"


def check_grad(tensor: torch.Tensor, label: str):
    """Assert gradient exists and has no NaNs."""
    assert tensor.grad is not None, f"{label}: no gradient"
    assert not torch.isnan(tensor.grad).any(), f"{label}: gradient contains NaN"


# ---------------------------------------------------------------------------
# Kernel tests
# ---------------------------------------------------------------------------


def test_reduction_aggr(graph, N, F, dtype, reduce):
    tag = f"reduction_aggr(reduce={reduce!r}, {dtype})"

    def _run():
        X = torch.randn(N, F, device="cuda", dtype=dtype, requires_grad=True)
        out = reduction_aggr(graph, X, reduce=reduce)
        check_output(out, (N, F), tag)
        out.sum().backward()
        check_grad(X, tag)

    run_test(tag, _run)


def test_gatv2_aggr(graph, N, H, D, dtype):
    tag = f"gatv2_aggr({dtype})"

    def _run():
        x = torch.randn(N, H, D, device="cuda", dtype=dtype, requires_grad=True)
        x_nb = torch.randn(N, H, D, device="cuda", dtype=dtype, requires_grad=True)
        attn = torch.randn(H, D, device="cuda", dtype=dtype, requires_grad=True)
        out = gatv2_aggr(graph, x, x_neighbors=x_nb, attention_weights=attn, negative_slope=0.2)
        check_output(out, (N, H, D), tag)
        out.backward(torch.ones_like(out))
        check_grad(x, tag)
        check_grad(x_nb, tag)
        check_grad(attn, tag)

    run_test(tag, _run)


def test_graph_transformer_aggr(graph, N, H, D, dtype):
    tag = f"graph_transformer_aggr({dtype})"

    def _run():
        Q = torch.randn(N, H, D, device="cuda", dtype=dtype, requires_grad=True)
        K = torch.randn(N, H, D, device="cuda", dtype=dtype, requires_grad=True)
        V = torch.randn(N, H, D, device="cuda", dtype=dtype, requires_grad=True)
        scale = 1.0 / (D**0.5)
        # x is unused by the kernel but required by the API
        x = torch.randn(N, H, D, device="cuda", dtype=dtype)
        out = graph_transformer_aggr(graph, x, Q=Q, K=K, V=V, scale=scale, autotune=True)
        check_output(out, (N, H, D), tag)
        out.backward(torch.ones_like(out))
        check_grad(Q, tag)
        check_grad(K, tag)
        check_grad(V, tag)

    run_test(tag, _run)


def test_spmm_aggr(graph, N, F, dtype):
    tag = f"spmm_aggr({dtype})"

    def _run():
        x = torch.randn(N, F, device="cuda", dtype=dtype, requires_grad=True)
        out = spmm_aggr(
            x,
            graph.forward_indptr,
            graph.forward_indices,
            norm_type="none",
            cu_sparse_algorithm_id=-1,
            block_dim=256,
        )
        check_output(out, (N, F), tag)
        out.sum().backward()
        check_grad(x, tag)

    run_test(tag, _run)


def test_csr_SPMM_normalized(graph, N, F, dtype, norm):
    tag = f"csr_SPMM_normalized(norm={norm!r}, {dtype})"

    def _run():
        x = torch.randn(N, F, device="cuda", dtype=dtype)
        out = csr_SPMM_normalized(
            indptr=graph.forward_indptr,
            indices=graph.forward_indices,
            features=x,
            norm=norm,
        )
        check_output(out, (N, F), tag)

    run_test(tag, _run)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 60)
    print("turbo_gnn smoke test")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available — aborting.")
        sys.exit(1)

    # Parameters
    N = 512  # nodes
    F = 64  # feature dim for 2-D kernels
    H = 4  # attention heads
    D = 32  # head dim
    dtypes = [torch.float32, torch.float16]

    # Build graphs
    print(f"\nGenerating random graph (N={N}) ...")
    edge_index = make_random_edge_index(N)
    graph = AdjacencyForwardBackwardWithNodeBuckets.from_edge_list(edge_index, num_nodes=N).to("cuda")
    # cuSPARSE requires int32 indices
    graph_i32 = AdjacencyForwardBackwardWithNodeBuckets.from_edge_list(
        edge_index, num_nodes=N, index_dtype=torch.int32
    ).to("cuda")
    num_edges = edge_index.shape[1]
    print(f"  nodes={N}, edges={num_edges}")

    # Run tests
    print()
    for dtype in dtypes:
        print(f"--- dtype={dtype} ---")

        for reduce in ("min", "max"):
            test_reduction_aggr(graph, N, F, dtype, reduce)

        test_gatv2_aggr(graph, N, H, D, dtype)
        test_graph_transformer_aggr(graph, N, H, D, dtype)
        # cuSPARSE only supports fp32
        if dtype == torch.float32:
            test_spmm_aggr(graph_i32, N, F, dtype)

            for norm in ("none", "left", "right", "both"):
                test_csr_SPMM_normalized(graph_i32, N, F, dtype, norm)

        print()

    # Summary
    passed = sum(ok for _, ok in results)
    total = len(results)
    failed = total - passed
    print("=" * 60)
    print(f"Results: {passed}/{total} passed, {failed} failed")
    print("=" * 60)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
