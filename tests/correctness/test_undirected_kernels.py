"""
Tests for undirected graph kernel variants:
  - is_directed detection in graph.py
  - GT backward: directed vs undirected kernel output matching
  - GATv2 backward: directed vs undirected kernel output matching
"""

import sys
from pathlib import Path

import pytest
import torch

_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from turbo_gnn.graph import (  # noqa: E402
    AdjacencyForwardBackwardWithNodeBuckets,
    build_csr_as_is,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_undirected_edge_index(N: int, E_approx: int, device: str = "cuda", seed: int = 42):
    """Random undirected graph with self-loops, deduplicated."""
    gen = torch.Generator(device=device).manual_seed(seed)
    src = torch.randint(0, N, (E_approx,), device=device, generator=gen)
    dst = torch.randint(0, N, (E_approx,), device=device, generator=gen)
    src_all = torch.cat([src, dst])
    dst_all = torch.cat([dst, src])
    self_nodes = torch.arange(N, device=device)
    src_all = torch.cat([src_all, self_nodes])
    dst_all = torch.cat([dst_all, self_nodes])
    edge_index = torch.stack([src_all, dst_all], dim=0)
    flat = edge_index[0] * N + edge_index[1]
    flat_unique = torch.unique(flat)
    row = flat_unique // N
    col = flat_unique % N
    return torch.stack([row, col], dim=0)


def make_directed_edge_index(N: int, E_approx: int, device: str = "cuda", seed: int = 42):
    """Random directed (non-symmetric) graph."""
    gen = torch.Generator(device=device).manual_seed(seed)
    src = torch.randint(0, N, (E_approx,), device=device, generator=gen)
    dst = torch.randint(0, N, (E_approx,), device=device, generator=gen)
    edge_index = torch.stack([src, dst], dim=0)
    flat = edge_index[0] * N + edge_index[1]
    flat_unique = torch.unique(flat)
    row = flat_unique // N
    col = flat_unique % N
    return torch.stack([row, col], dim=0)


def build_graph(edge_index, num_nodes, is_directed=None):
    """Build AdjacencyForwardBackwardWithNodeBuckets."""
    fwd_indptr, fwd_indices, _, _ = build_csr_as_is(
        edge_index,
        edge_weight=None,
        num_nodes=num_nodes,
        do_transpose=True,
    )
    bwd_indptr, bwd_indices, _, _ = build_csr_as_is(
        edge_index,
        edge_weight=None,
        num_nodes=num_nodes,
        do_transpose=False,
    )
    all_nodes = torch.arange(num_nodes, device=edge_index.device, dtype=torch.int32)
    empty_nodes = torch.tensor([], dtype=torch.int32, device=edge_index.device)
    return AdjacencyForwardBackwardWithNodeBuckets(
        forward_indptr=fwd_indptr.int(),
        forward_indices=fwd_indices.int(),
        backward_indptr=bwd_indptr.int(),
        backward_indices=bwd_indices.int(),
        forward_light_nodes=all_nodes,
        forward_heavy_nodes=empty_nodes,
        backward_light_nodes=all_nodes,
        backward_heavy_nodes=empty_nodes,
        is_directed=is_directed,
    )


# ---------------------------------------------------------------------------
# is_directed detection tests
# ---------------------------------------------------------------------------


class TestIsDirectedDetection:
    def test_undirected_graph_auto_detected(self):
        """Undirected graph should be auto-detected as is_directed=False."""
        device = "cuda"
        edge_index = make_undirected_edge_index(32, 100, device=device)
        graph = build_graph(edge_index, 32)
        assert graph.is_directed is False

    def test_directed_graph_auto_detected(self):
        """Directed graph should be auto-detected as is_directed=True."""
        device = "cuda"
        edge_index = make_directed_edge_index(32, 100, device=device)
        graph = build_graph(edge_index, 32)
        assert graph.is_directed is True

    def test_explicit_is_directed_true(self):
        """Explicit is_directed=True should skip detection."""
        device = "cuda"
        edge_index = make_undirected_edge_index(32, 100, device=device)
        graph = build_graph(edge_index, 32, is_directed=True)
        assert graph.is_directed is True

    def test_explicit_is_directed_false(self):
        """Explicit is_directed=False should alias backward to forward CSR."""
        device = "cuda"
        edge_index = make_undirected_edge_index(32, 100, device=device)
        graph = build_graph(edge_index, 32, is_directed=False)
        assert graph.is_directed is False
        assert graph.backward_indptr.data_ptr() == graph.forward_indptr.data_ptr()
        assert graph.backward_indices.data_ptr() == graph.forward_indices.data_ptr()

    def test_undirected_csr_aliasing(self):
        """Auto-detected undirected graph should alias backward CSR to forward CSR."""
        device = "cuda"
        edge_index = make_undirected_edge_index(32, 100, device=device)
        graph = build_graph(edge_index, 32)
        assert graph.backward_indptr.data_ptr() == graph.forward_indptr.data_ptr()
        assert graph.backward_indices.data_ptr() == graph.forward_indices.data_ptr()

    def test_from_edge_list_auto_detect(self):
        """from_edge_list should auto-detect undirected graphs."""
        device = "cuda"
        edge_index = make_undirected_edge_index(32, 100, device=device)
        graph = AdjacencyForwardBackwardWithNodeBuckets.from_edge_list(
            edge_index,
            32,
            index_dtype=torch.int32,
        )
        assert graph.is_directed is False

    def test_from_edge_list_explicit_false_skips_backward_csr(self):
        """from_edge_list with is_directed=False should alias and skip backward CSR build."""
        device = "cuda"
        edge_index = make_undirected_edge_index(32, 100, device=device)
        graph = AdjacencyForwardBackwardWithNodeBuckets.from_edge_list(
            edge_index,
            32,
            index_dtype=torch.int32,
            is_directed=False,
        )
        assert graph.is_directed is False
        assert graph.backward_indptr.data_ptr() == graph.forward_indptr.data_ptr()

    def test_repartition_preserves_is_directed(self):
        """repartition() should preserve is_directed flag."""
        device = "cuda"
        edge_index = make_undirected_edge_index(32, 100, device=device)
        graph = build_graph(edge_index, 32)
        repartitioned = graph.repartition(forward_huge_degree_threshold_quantile=0.9)
        assert repartitioned.is_directed == graph.is_directed

    def test_to_device_preserves_aliasing(self):
        """to(device) should preserve CSR aliasing for undirected graphs."""
        device = "cuda"
        edge_index = make_undirected_edge_index(32, 100, device=device)
        graph = build_graph(edge_index, 32)
        graph_moved = graph.to(device)
        if not graph_moved.is_directed:
            assert graph_moved.backward_indptr.data_ptr() == graph_moved.forward_indptr.data_ptr()


# ---------------------------------------------------------------------------
# GT backward: directed vs undirected kernel matching
# ---------------------------------------------------------------------------


class TestGTUndirectedKernel:
    @pytest.mark.parametrize("num_nodes", [32, 64])
    @pytest.mark.parametrize("D", [32, 64])
    @pytest.mark.parametrize("H", [1, 4])
    def test_gt_backward_directed_vs_undirected(self, num_nodes, D, H):
        """On a symmetric graph, directed and undirected GT backward kernels must produce matching gradients."""
        device = "cuda"
        torch.manual_seed(42)

        edge_index = make_undirected_edge_index(num_nodes, num_nodes * 3, device=device)

        # Build graph with is_directed=True (force directed kernel)
        graph_directed = build_graph(edge_index, num_nodes, is_directed=True)
        # Build graph with is_directed=False (force undirected kernel)
        graph_undirected = build_graph(edge_index, num_nodes, is_directed=False)

        Q = torch.randn(num_nodes, H, D, device=device, requires_grad=True)
        K = torch.randn(num_nodes, H, D, device=device, requires_grad=True)
        V = torch.randn(num_nodes, H, D, device=device, requires_grad=True)
        scale = D**-0.5

        try:
            from turbo_gnn._functions import _FusedGraphAttention
        except ImportError:
            pytest.skip("turbo_gnn C extension not available")

        all_nodes = torch.arange(num_nodes, device=device, dtype=torch.int32)
        empty_nodes = torch.tensor([], dtype=torch.int32, device=device)

        # Directed backward
        Q_d = Q.detach().clone().requires_grad_(True)
        K_d = K.detach().clone().requires_grad_(True)
        V_d = V.detach().clone().requires_grad_(True)
        out_d = _FusedGraphAttention.apply(
            graph_directed.forward_indptr,
            graph_directed.forward_indices,
            graph_directed.backward_indptr,
            graph_directed.backward_indices,
            Q_d,
            K_d,
            V_d,
            scale,
            all_nodes,
            empty_nodes,  # fwd light/heavy
            all_nodes,
            empty_nodes,  # bwd light/heavy
            1,
            8,
            1,
            8,  # light/heavy warps fwd/bwd
            True,  # is_directed
        )
        grad_out = torch.randn_like(out_d)
        out_d.backward(grad_out)

        # Undirected backward
        Q_u = Q.detach().clone().requires_grad_(True)
        K_u = K.detach().clone().requires_grad_(True)
        V_u = V.detach().clone().requires_grad_(True)
        out_u = _FusedGraphAttention.apply(
            graph_undirected.forward_indptr,
            graph_undirected.forward_indices,
            graph_undirected.backward_indptr,
            graph_undirected.backward_indices,
            Q_u,
            K_u,
            V_u,
            scale,
            all_nodes,
            empty_nodes,  # fwd light/heavy
            all_nodes,
            empty_nodes,  # bwd light/heavy
            1,
            8,
            1,
            8,  # light/heavy warps fwd/bwd
            False,  # is_directed
        )
        out_u.backward(grad_out)

        # Forward outputs should match exactly
        assert torch.allclose(
            out_d, out_u, atol=1e-5, rtol=1e-5
        ), f"Forward mismatch: max|diff|={(out_d - out_u).abs().max().item():.3e}"

        # Gradients should match within tolerance
        atol, rtol = 1e-3, 1e-3
        for name, gd, gu in [("dQ", Q_d.grad, Q_u.grad), ("dK", K_d.grad, K_u.grad), ("dV", V_d.grad, V_u.grad)]:
            assert torch.allclose(gd, gu, atol=atol, rtol=rtol), (
                f"{name} mismatch: max|diff|={(gd - gu).abs().max().item():.3e}, "
                f"mean|diff|={(gd - gu).abs().mean().item():.3e}"
            )


# ---------------------------------------------------------------------------
# GATv2 backward: directed vs undirected kernel matching
# ---------------------------------------------------------------------------


class TestGATv2UndirectedKernel:
    @pytest.mark.parametrize("num_nodes", [32, 64])
    @pytest.mark.parametrize("D", [32, 64])
    @pytest.mark.parametrize("H", [1, 4])
    def test_gatv2_backward_directed_vs_undirected(self, num_nodes, D, H):
        """On a symmetric graph, directed and undirected GATv2 backward must produce matching gradients."""
        device = "cuda"
        torch.manual_seed(42)

        edge_index = make_undirected_edge_index(num_nodes, num_nodes * 3, device=device)

        graph_directed = build_graph(edge_index, num_nodes, is_directed=True)
        graph_undirected = build_graph(edge_index, num_nodes, is_directed=False)

        x_left = torch.randn(num_nodes, H, D, device=device, requires_grad=True)
        x_right = torch.randn(num_nodes, H, D, device=device, requires_grad=True)
        attn_weights = torch.randn(H, D, device=device, requires_grad=True)
        negative_slope = 0.2

        try:
            from turbo_gnn._functions import gatv2_function
        except ImportError:
            pytest.skip("turbo_gnn C extension not available")

        all_nodes = torch.arange(num_nodes, device=device, dtype=torch.int32)
        empty_nodes = torch.tensor([], dtype=torch.int32, device=device)

        # Directed backward
        xl_d = x_left.detach().clone().requires_grad_(True)
        xr_d = x_right.detach().clone().requires_grad_(True)
        aw_d = attn_weights.detach().clone().requires_grad_(True)
        out_d = gatv2_function.apply(
            graph_directed.forward_indptr,
            graph_directed.forward_indices,
            graph_directed.backward_indptr,
            graph_directed.backward_indices,
            xl_d,
            xr_d,
            aw_d,
            negative_slope,
            512,
            all_nodes,
            empty_nodes,  # fwd light/heavy
            all_nodes,
            empty_nodes,  # bwd light/heavy
            1,
            8,
            1,
            8,  # light/heavy warps fwd/bwd
            True,  # is_directed
        )
        grad_out = torch.randn_like(out_d)
        out_d.backward(grad_out)

        # Undirected backward
        xl_u = x_left.detach().clone().requires_grad_(True)
        xr_u = x_right.detach().clone().requires_grad_(True)
        aw_u = attn_weights.detach().clone().requires_grad_(True)
        out_u = gatv2_function.apply(
            graph_undirected.forward_indptr,
            graph_undirected.forward_indices,
            graph_undirected.backward_indptr,
            graph_undirected.backward_indices,
            xl_u,
            xr_u,
            aw_u,
            negative_slope,
            512,
            all_nodes,
            empty_nodes,  # fwd light/heavy
            all_nodes,
            empty_nodes,  # bwd light/heavy
            1,
            8,
            1,
            8,  # light/heavy warps fwd/bwd
            False,  # is_directed
        )
        out_u.backward(grad_out)

        # Forward outputs should match exactly
        assert torch.allclose(
            out_d, out_u, atol=1e-5, rtol=1e-5
        ), f"Forward mismatch: max|diff|={(out_d - out_u).abs().max().item():.3e}"

        # Gradients should match within tolerance
        atol, rtol = 1e-3, 1e-3
        for name, gd, gu in [
            ("grad_l", xl_d.grad, xl_u.grad),
            ("grad_r", xr_d.grad, xr_u.grad),
            ("grad_a", aw_d.grad, aw_u.grad),
        ]:
            assert torch.allclose(gd, gu, atol=atol, rtol=rtol), (
                f"{name} mismatch: max|diff|={(gd - gu).abs().max().item():.3e}, "
                f"mean|diff|={(gd - gu).abs().mean().item():.3e}"
            )
