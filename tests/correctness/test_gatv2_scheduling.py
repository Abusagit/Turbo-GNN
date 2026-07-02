import sys
from pathlib import Path

import pytest
import torch

_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import turbo_gnn._C as _C  # noqa: E402
from turbo_gnn.graph import AdjacencyForwardBackwardWithNodeBuckets  # noqa: E402
from turbo_gnn.scheduling import sort_by_degree_desc  # noqa: E402


def make_graph(N, avg_degree=8, seed=42, device="cuda"):
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
class TestSchedulingPreservesOutput:
    @pytest.mark.parametrize("grid_size", [0, 132, 264])
    @pytest.mark.parametrize("D", [32, 64, 128])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_sorted_matches_original(self, grid_size, D, dtype):
        N, H = 2048, 4
        torch.manual_seed(0)
        edge_index = make_graph(N, avg_degree=8)
        graph = build_bucketed(edge_index, N, quantile=-1)

        xl = torch.randn(N, H, D, device="cuda", dtype=dtype)
        xr = torch.randn(N, H, D, device="cuda", dtype=dtype)
        aw = torch.randn(H, D, device="cuda", dtype=dtype)

        out_orig, lse_orig = run_forward(graph, xl, xr, aw, grid_size_override=grid_size)

        graph.forward_light_nodes = sort_by_degree_desc(
            graph.forward_light_nodes, graph.forward_indptr,
        )
        out_sorted, lse_sorted = run_forward(graph, xl, xr, aw, grid_size_override=grid_size)

        assert torch.equal(out_orig, out_sorted), (
            f"grid_size={grid_size} D={D} dtype={dtype}: "
            f"max|Δ|={(out_orig.float() - out_sorted.float()).abs().max().item():.3e}"
        )
        assert torch.equal(lse_orig, lse_sorted)

    def test_sort_actually_reorders(self):
        N = 1024
        torch.manual_seed(1)
        edge_index = make_graph(N, avg_degree=6)
        graph = build_bucketed(edge_index, N, quantile=-1)

        original = graph.forward_light_nodes.clone()
        sorted_nodes = sort_by_degree_desc(original, graph.forward_indptr)

        degrees_full = graph.forward_indptr[1:] - graph.forward_indptr[:-1]
        degrees_sorted = degrees_full[sorted_nodes.long()]

        assert torch.all(degrees_sorted[:-1] >= degrees_sorted[1:]), \
            "sort_by_degree_desc did not produce descending degrees"
        assert not torch.equal(original, sorted_nodes), \
            "sort was a no-op — the input was already sorted, pick another seed"
