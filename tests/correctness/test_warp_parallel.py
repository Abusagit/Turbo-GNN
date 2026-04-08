"""
Warp-parallel kernel correctness tests:
  - GT forward/backward: different warp counts produce same results
  - GATv2 forward/backward: different warp counts produce same results
  - Light/heavy node dispatch: split results match all-in-one-bucket baseline
"""

import sys
from pathlib import Path

import pytest
import torch

_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from turbo_gnn._functions import _FusedGraphAttention, gatv2_function  # noqa: E402
from turbo_gnn.graph import (  # noqa: E402
    AdjacencyForwardBackwardWithNodeBuckets,
    build_csr_as_is,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_graph(N, E_approx, device="cuda", seed=42):
    """Random undirected graph with self-loops, deduplicated."""
    gen = torch.Generator(device=device).manual_seed(seed)
    src = torch.randint(0, N, (E_approx,), device=device, generator=gen)
    dst = torch.randint(0, N, (E_approx,), device=device, generator=gen)
    src_all = torch.cat([src, dst, torch.arange(N, device=device)])
    dst_all = torch.cat([dst, src, torch.arange(N, device=device)])
    edge_index = torch.stack([src_all, dst_all])
    flat = edge_index[0] * N + edge_index[1]
    flat_unique = torch.unique(flat)
    return torch.stack([flat_unique // N, flat_unique % N])


def build_graph_with_buckets(edge_index, num_nodes, quantile=-1):
    """Build graph with node bucketing. quantile=-1 puts all nodes in light."""
    return AdjacencyForwardBackwardWithNodeBuckets.from_edge_list(
        edge_index,
        num_nodes,
        quantile=quantile,
        index_dtype=torch.int32,
    )


def run_gt_forward_backward(graph, N, H, D, light_w, heavy_w, seed=42):
    """Run GT forward+backward with given warp config. Returns (out, dQ, dK, dV)."""
    torch.manual_seed(seed)
    Q = torch.randn(N, H, D, device="cuda", requires_grad=True)
    K = torch.randn(N, H, D, device="cuda", requires_grad=True)
    V = torch.randn(N, H, D, device="cuda", requires_grad=True)
    scale = D**-0.5

    out = _FusedGraphAttention.apply(
        graph.forward_indptr,
        graph.forward_indices,
        graph.backward_indptr,
        graph.backward_indices,
        Q,
        K,
        V,
        scale,
        graph.forward_light_nodes,
        graph.forward_heavy_nodes,
        graph.backward_light_nodes,
        graph.backward_heavy_nodes,
        light_w,
        heavy_w,
        light_w,
        heavy_w,
        True,  # is_directed
    )

    torch.manual_seed(seed + 1000)
    grad_out = torch.randn_like(out)
    out.backward(grad_out)

    return out.detach(), Q.grad.detach(), K.grad.detach(), V.grad.detach()


def run_gatv2_forward_backward(graph, N, H, D, light_w, heavy_w, seed=42):
    """Run GATv2 forward+backward with given warp config. Returns (out, gl, gr, ga)."""
    torch.manual_seed(seed)
    xl = torch.randn(N, H, D, device="cuda", requires_grad=True)
    xr = torch.randn(N, H, D, device="cuda", requires_grad=True)
    aw = torch.randn(H, D, device="cuda", requires_grad=True)

    out = gatv2_function.apply(
        graph.forward_indptr,
        graph.forward_indices,
        graph.backward_indptr,
        graph.backward_indices,
        xl,
        xr,
        aw,
        0.2,
        512,
        graph.forward_light_nodes,
        graph.forward_heavy_nodes,
        graph.backward_light_nodes,
        graph.backward_heavy_nodes,
        light_w,
        heavy_w,
        light_w,
        heavy_w,
        True,  # is_directed
    )

    torch.manual_seed(seed + 1000)
    grad_out = torch.randn_like(out)
    out.backward(grad_out)

    return out.detach(), xl.grad.detach(), xr.grad.detach(), aw.grad.detach()


# ---------------------------------------------------------------------------
# GT: warp count sweep (all nodes in light bucket)
# ---------------------------------------------------------------------------


class TestGTWarpSweep:
    """Compare GT outputs across different warp counts with all nodes in one bucket."""

    @pytest.mark.parametrize("warps", [1, 2, 4])
    @pytest.mark.parametrize("D", [32, 64])
    def test_gt_forward_warp_sweep(self, warps, D):
        N, H = 48, 2
        edge_index = make_graph(N, N * 5)
        graph = build_graph_with_buckets(edge_index, N, quantile=-1)  # all light

        out_ref, dQ_ref, dK_ref, dV_ref = run_gt_forward_backward(graph, N, H, D, 1, 8)
        out_test, dQ_test, dK_test, dV_test = run_gt_forward_backward(graph, N, H, D, warps, 8)

        atol_fwd, atol_bwd = 1e-4, 1e-3
        assert torch.allclose(
            out_ref, out_test, atol=atol_fwd, rtol=atol_fwd
        ), f"GT fwd W={warps}: max diff {(out_ref - out_test).abs().max():.3e}"
        for name, ref, test in [("dQ", dQ_ref, dQ_test), ("dK", dK_ref, dK_test), ("dV", dV_ref, dV_test)]:
            assert torch.allclose(
                ref, test, atol=atol_bwd, rtol=atol_bwd
            ), f"GT {name} W={warps}: max diff {(ref - test).abs().max():.3e}"

    @pytest.mark.parametrize("warps", [8, 16, 32])
    def test_gt_forward_heavy_warp_sweep(self, warps):
        """Test heavy-range warp counts with all nodes in heavy bucket."""
        N, H, D = 48, 2, 64
        edge_index = make_graph(N, N * 5)
        # Put all nodes in heavy bucket by using quantile=0 (threshold=0, all degrees >= 0)
        fwd_indptr, fwd_indices, _, _ = build_csr_as_is(edge_index, None, N, do_transpose=True)
        bwd_indptr, bwd_indices, _, _ = build_csr_as_is(edge_index, None, N, do_transpose=False)
        empty = torch.tensor([], dtype=torch.int32, device="cuda")
        all_nodes = torch.arange(N, device="cuda", dtype=torch.int32)

        graph = AdjacencyForwardBackwardWithNodeBuckets(
            forward_indptr=fwd_indptr.int(),
            forward_indices=fwd_indices.int(),
            backward_indptr=bwd_indptr.int(),
            backward_indices=bwd_indices.int(),
            forward_light_nodes=empty,
            forward_heavy_nodes=all_nodes,
            backward_light_nodes=empty,
            backward_heavy_nodes=all_nodes,
        )

        out_ref, dQ_ref, dK_ref, dV_ref = run_gt_forward_backward(graph, N, H, D, 1, 8)
        out_test, dQ_test, dK_test, dV_test = run_gt_forward_backward(graph, N, H, D, 1, warps)

        atol_fwd, atol_bwd = 1e-4, 1e-3
        assert torch.allclose(
            out_ref, out_test, atol=atol_fwd, rtol=atol_fwd
        ), f"GT fwd heavy W={warps}: max diff {(out_ref - out_test).abs().max():.3e}"
        for name, ref, test in [("dQ", dQ_ref, dQ_test), ("dK", dK_ref, dK_test), ("dV", dV_ref, dV_test)]:
            assert torch.allclose(
                ref, test, atol=atol_bwd, rtol=atol_bwd
            ), f"GT {name} heavy W={warps}: max diff {(ref - test).abs().max():.3e}"


# ---------------------------------------------------------------------------
# GATv2: warp count sweep
# ---------------------------------------------------------------------------


class TestGATv2WarpSweep:
    """Compare GATv2 outputs across different warp counts."""

    @pytest.mark.parametrize("warps", [1, 2, 4])
    @pytest.mark.parametrize("D", [32, 64])
    def test_gatv2_forward_warp_sweep(self, warps, D):
        N, H = 48, 2
        edge_index = make_graph(N, N * 5)
        graph = build_graph_with_buckets(edge_index, N, quantile=-1)

        out_ref, gl_ref, gr_ref, ga_ref = run_gatv2_forward_backward(graph, N, H, D, 1, 8)
        out_test, gl_test, gr_test, ga_test = run_gatv2_forward_backward(graph, N, H, D, warps, 8)

        atol_fwd, atol_bwd = 1e-4, 1e-3
        assert torch.allclose(
            out_ref, out_test, atol=atol_fwd, rtol=atol_fwd
        ), f"GATv2 fwd W={warps}: max diff {(out_ref - out_test).abs().max():.3e}"
        for name, ref, test in [("grad_l", gl_ref, gl_test), ("grad_r", gr_ref, gr_test), ("grad_a", ga_ref, ga_test)]:
            assert torch.allclose(
                ref, test, atol=atol_bwd, rtol=atol_bwd
            ), f"GATv2 {name} W={warps}: max diff {(ref - test).abs().max():.3e}"

    @pytest.mark.parametrize("warps", [8, 16, 32])
    def test_gatv2_heavy_warp_sweep(self, warps):
        N, H, D = 48, 2, 64
        edge_index = make_graph(N, N * 5)
        fwd_indptr, fwd_indices, _, _ = build_csr_as_is(edge_index, None, N, do_transpose=True)
        bwd_indptr, bwd_indices, _, _ = build_csr_as_is(edge_index, None, N, do_transpose=False)
        empty = torch.tensor([], dtype=torch.int32, device="cuda")
        all_nodes = torch.arange(N, device="cuda", dtype=torch.int32)

        graph = AdjacencyForwardBackwardWithNodeBuckets(
            forward_indptr=fwd_indptr.int(),
            forward_indices=fwd_indices.int(),
            backward_indptr=bwd_indptr.int(),
            backward_indices=bwd_indices.int(),
            forward_light_nodes=empty,
            forward_heavy_nodes=all_nodes,
            backward_light_nodes=empty,
            backward_heavy_nodes=all_nodes,
        )

        out_ref, gl_ref, gr_ref, ga_ref = run_gatv2_forward_backward(graph, N, H, D, 1, 8)
        out_test, gl_test, gr_test, ga_test = run_gatv2_forward_backward(graph, N, H, D, 1, warps)

        atol_fwd, atol_bwd = 1e-4, 1e-3
        assert torch.allclose(
            out_ref, out_test, atol=atol_fwd, rtol=atol_fwd
        ), f"GATv2 fwd heavy W={warps}: max diff {(out_ref - out_test).abs().max():.3e}"
        for name, ref, test in [("grad_l", gl_ref, gl_test), ("grad_r", gr_ref, gr_test), ("grad_a", ga_ref, ga_test)]:
            assert torch.allclose(
                ref, test, atol=atol_bwd, rtol=atol_bwd
            ), f"GATv2 {name} heavy W={warps}: max diff {(ref - test).abs().max():.3e}"


# ---------------------------------------------------------------------------
# Light/heavy split: split dispatch matches all-in-one-bucket baseline
# ---------------------------------------------------------------------------


class TestLightHeavySplit:
    """Verify that splitting nodes into light+heavy buckets produces the same output
    as putting all nodes in a single bucket."""

    def test_gt_light_heavy_split(self):
        N, H, D = 64, 2, 64
        edge_index = make_graph(N, N * 8, seed=123)

        # Baseline: all nodes in light (W=1)
        graph_all_light = build_graph_with_buckets(edge_index, N, quantile=-1)
        out_ref, dQ_ref, dK_ref, dV_ref = run_gt_forward_backward(
            graph_all_light,
            N,
            H,
            D,
            light_w=1,
            heavy_w=8,
            seed=99,
        )

        # Split: quantile=0.9 separates light from heavy
        graph_split = build_graph_with_buckets(edge_index, N, quantile=0.9)
        n_light = graph_split.forward_light_nodes.numel()
        n_heavy = graph_split.forward_heavy_nodes.numel()
        assert n_light > 0 and n_heavy > 0, "Need both buckets non-empty for this test"

        out_test, dQ_test, dK_test, dV_test = run_gt_forward_backward(
            graph_split,
            N,
            H,
            D,
            light_w=1,
            heavy_w=8,
            seed=99,
        )

        atol_fwd, atol_bwd = 1e-4, 1e-3
        assert torch.allclose(
            out_ref, out_test, atol=atol_fwd, rtol=atol_fwd
        ), f"GT fwd split: max diff {(out_ref - out_test).abs().max():.3e}"
        for name, ref, test in [("dQ", dQ_ref, dQ_test), ("dK", dK_ref, dK_test), ("dV", dV_ref, dV_test)]:
            assert torch.allclose(
                ref, test, atol=atol_bwd, rtol=atol_bwd
            ), f"GT {name} split: max diff {(ref - test).abs().max():.3e}"

    def test_gatv2_light_heavy_split(self):
        N, H, D = 64, 2, 64
        edge_index = make_graph(N, N * 8, seed=123)

        graph_all_light = build_graph_with_buckets(edge_index, N, quantile=-1)
        out_ref, gl_ref, gr_ref, ga_ref = run_gatv2_forward_backward(
            graph_all_light,
            N,
            H,
            D,
            light_w=1,
            heavy_w=8,
            seed=99,
        )

        graph_split = build_graph_with_buckets(edge_index, N, quantile=0.9)
        n_light = graph_split.forward_light_nodes.numel()
        n_heavy = graph_split.forward_heavy_nodes.numel()
        assert n_light > 0 and n_heavy > 0, "Need both buckets non-empty for this test"

        out_test, gl_test, gr_test, ga_test = run_gatv2_forward_backward(
            graph_split,
            N,
            H,
            D,
            light_w=1,
            heavy_w=8,
            seed=99,
        )

        atol_fwd, atol_bwd = 1e-4, 1e-3
        assert torch.allclose(
            out_ref, out_test, atol=atol_fwd, rtol=atol_fwd
        ), f"GATv2 fwd split: max diff {(out_ref - out_test).abs().max():.3e}"
        for name, ref, test in [("grad_l", gl_ref, gl_test), ("grad_r", gr_ref, gr_test), ("grad_a", ga_ref, ga_test)]:
            assert torch.allclose(
                ref, test, atol=atol_bwd, rtol=atol_bwd
            ), f"GATv2 {name} split: max diff {(ref - test).abs().max():.3e}"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases: empty buckets, single-node graphs, isolated nodes."""

    def test_gt_all_heavy_empty_light(self):
        """All nodes heavy, light bucket empty."""
        N, H, D = 32, 2, 64
        edge_index = make_graph(N, N * 5)
        fwd_indptr, fwd_indices, _, _ = build_csr_as_is(edge_index, None, N, do_transpose=True)
        bwd_indptr, bwd_indices, _, _ = build_csr_as_is(edge_index, None, N, do_transpose=False)
        empty = torch.tensor([], dtype=torch.int32, device="cuda")
        all_nodes = torch.arange(N, device="cuda", dtype=torch.int32)

        graph = AdjacencyForwardBackwardWithNodeBuckets(
            forward_indptr=fwd_indptr.int(),
            forward_indices=fwd_indices.int(),
            backward_indptr=bwd_indptr.int(),
            backward_indices=bwd_indices.int(),
            forward_light_nodes=empty,
            forward_heavy_nodes=all_nodes,
            backward_light_nodes=empty,
            backward_heavy_nodes=all_nodes,
        )
        out, dQ, dK, dV = run_gt_forward_backward(graph, N, H, D, 1, 8)
        assert not out.isnan().any(), "NaN in output"
        assert not dQ.isnan().any(), "NaN in dQ"

    def test_gt_all_light_empty_heavy(self):
        """All nodes light, heavy bucket empty."""
        N, H, D = 32, 2, 64
        edge_index = make_graph(N, N * 5)
        graph = build_graph_with_buckets(edge_index, N, quantile=-1)
        out, dQ, dK, dV = run_gt_forward_backward(graph, N, H, D, 2, 8)
        assert not out.isnan().any(), "NaN in output"
        assert not dQ.isnan().any(), "NaN in dQ"

    def test_gt_small_graph(self):
        """Very small graph (4 nodes)."""
        N, H, D = 4, 1, 32
        edge_index = make_graph(N, 6)
        graph = build_graph_with_buckets(edge_index, N, quantile=-1)
        out, dQ, dK, dV = run_gt_forward_backward(graph, N, H, D, 1, 8)
        assert out.shape == (N, H, D)
        assert not out.isnan().any()

    def test_gatv2_small_graph(self):
        """Very small GATv2 graph (4 nodes)."""
        N, H, D = 4, 1, 32
        edge_index = make_graph(N, 6)
        graph = build_graph_with_buckets(edge_index, N, quantile=-1)
        out, gl, gr, ga = run_gatv2_forward_backward(graph, N, H, D, 1, 8)
        assert out.shape == (N, H, D)
        assert not out.isnan().any()
