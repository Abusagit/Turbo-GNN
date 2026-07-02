import sys
from pathlib import Path

import pytest
import torch

_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import turbo_gnn._C as _C  # noqa: E402
from turbo_gnn.graph import AdjacencyForwardBackwardWithNodeBuckets  # noqa: E402


def make_graph(N, avg_degree=6, seed=42, device="cuda"):
    gen = torch.Generator(device=device).manual_seed(seed)
    E = N * avg_degree
    src = torch.randint(0, N, (E,), device=device, generator=gen)
    dst = torch.randint(0, N, (E,), device=device, generator=gen)
    src_all = torch.cat([src, dst, torch.arange(N, device=device)])
    dst_all = torch.cat([dst, src, torch.arange(N, device=device)])
    edge_index = torch.stack([src_all, dst_all])
    flat = edge_index[0] * N + edge_index[1]
    flat_unique = torch.unique(flat)
    return torch.stack([flat_unique // N, flat_unique % N])


def build_bucketed(edge_index, N, quantile):
    return AdjacencyForwardBackwardWithNodeBuckets.from_edge_list(
        edge_index, N, quantile=quantile, index_dtype=torch.int32
    )


def run_forward(graph, xl, xr, aw, grid_size_override):
    return _C.gatv2_forward(
        xl,
        xr,
        graph.forward_indptr,
        graph.forward_indices,
        aw,
        0.2,
        graph.forward_light_nodes,
        graph.forward_heavy_nodes,
        1,
        8,
        grid_size_override,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestGATv2GridStrided:
    @pytest.mark.parametrize("D", [32, 64, 128])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    @pytest.mark.parametrize("grid_size", [1, 32, 132, 264, 1024])
    def test_all_light_bucket_matches_legacy(self, D, dtype, grid_size):
        N, H = 512, 4
        torch.manual_seed(0)
        edge_index = make_graph(N, avg_degree=6)
        graph = build_bucketed(edge_index, N, quantile=-1)

        xl = torch.randn(N, H, D, device="cuda", dtype=dtype)
        xr = torch.randn(N, H, D, device="cuda", dtype=dtype)
        aw = torch.randn(H, D, device="cuda", dtype=dtype)

        out_legacy, lse_legacy = run_forward(graph, xl, xr, aw, grid_size_override=0)
        out_gs, lse_gs = run_forward(graph, xl, xr, aw, grid_size_override=grid_size)

        assert torch.equal(out_legacy, out_gs), (
            f"D={D} dtype={dtype} grid_size={grid_size}: "
            f"max|Δout|={(out_legacy.float() - out_gs.float()).abs().max().item():.3e}"
        )
        assert torch.equal(lse_legacy, lse_gs), (
            f"D={D} dtype={dtype} grid_size={grid_size}: "
            f"max|Δlse|={(lse_legacy - lse_gs).abs().max().item():.3e}"
        )

    @pytest.mark.parametrize("D", [32, 64])
    @pytest.mark.parametrize("grid_size", [1, 32, 132])
    def test_mixed_light_heavy_matches_legacy(self, D, grid_size):
        N, H = 1024, 2
        torch.manual_seed(1)
        edge_index = make_graph(N, avg_degree=8)
        graph = build_bucketed(edge_index, N, quantile=0.9)

        assert graph.forward_light_nodes.numel() > 0
        assert graph.forward_heavy_nodes.numel() > 0

        xl = torch.randn(N, H, D, device="cuda", dtype=torch.float32)
        xr = torch.randn(N, H, D, device="cuda", dtype=torch.float32)
        aw = torch.randn(H, D, device="cuda", dtype=torch.float32)

        out_legacy, lse_legacy = run_forward(graph, xl, xr, aw, grid_size_override=0)
        out_gs, lse_gs = run_forward(graph, xl, xr, aw, grid_size_override=grid_size)

        assert torch.equal(out_legacy, out_gs), (
            f"D={D} grid_size={grid_size}: "
            f"max|Δout|={(out_legacy - out_gs).abs().max().item():.3e}"
        )
        assert torch.equal(lse_legacy, lse_gs)

    def test_grid_size_larger_than_bucket_is_capped(self):
        N, H, D = 64, 2, 32
        torch.manual_seed(2)
        edge_index = make_graph(N, avg_degree=4)
        graph = build_bucketed(edge_index, N, quantile=-1)

        xl = torch.randn(N, H, D, device="cuda", dtype=torch.float32)
        xr = torch.randn(N, H, D, device="cuda", dtype=torch.float32)
        aw = torch.randn(H, D, device="cuda", dtype=torch.float32)

        out_ref, _ = run_forward(graph, xl, xr, aw, grid_size_override=0)
        out_big, _ = run_forward(graph, xl, xr, aw, grid_size_override=100_000)
        assert torch.equal(out_ref, out_big)
