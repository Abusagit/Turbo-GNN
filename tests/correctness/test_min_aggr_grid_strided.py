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


def max_degree_of(indptr):
    signed = indptr
    if indptr.dtype == torch.uint32:
        signed = indptr.view(torch.int32)
    elif indptr.dtype == torch.uint64:
        signed = indptr.view(torch.int64)
    return int((signed[1:] - signed[:-1]).max().item())


def run_forward(graph, x, reduce, grid_size_override):
    return _C.reduction_aggr_forward_partitioned(
        graph.forward_indptr,
        graph.forward_indices,
        x,
        graph.forward_light_nodes,
        graph.forward_heavy_nodes,
        max_degree_of(graph.forward_indptr),
        8,
        128,
        False,
        32,
        8,
        reduce,
        grid_size_override,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestMinAggrGridStrided:
    @pytest.mark.parametrize("reduce", ["min", "max"])
    @pytest.mark.parametrize("d", [32, 64, 128])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    @pytest.mark.parametrize("grid_size", [1, 32, 132, 264, 1024])
    def test_all_light_bucket_matches_legacy(self, reduce, d, dtype, grid_size):
        N = 512
        torch.manual_seed(0)
        edge_index = make_graph(N, avg_degree=6)
        graph = build_bucketed(edge_index, N, quantile=-1)

        x = torch.randn(N, d, device="cuda", dtype=dtype)

        out_legacy, arg_legacy = run_forward(graph, x, reduce, grid_size_override=0)
        out_gs, arg_gs = run_forward(graph, x, reduce, grid_size_override=grid_size)

        assert torch.equal(out_legacy, out_gs), (
            f"reduce={reduce} d={d} dtype={dtype} grid_size={grid_size}: "
            f"max|Δout|={(out_legacy.float() - out_gs.float()).abs().max().item():.3e}"
        )
        assert torch.equal(arg_legacy, arg_gs)

    @pytest.mark.parametrize("reduce", ["min", "max"])
    @pytest.mark.parametrize("d", [32, 64])
    @pytest.mark.parametrize("grid_size", [1, 32, 132])
    def test_mixed_light_heavy_matches_legacy(self, reduce, d, grid_size):
        N = 1024
        torch.manual_seed(1)
        edge_index = make_graph(N, avg_degree=8)
        graph = build_bucketed(edge_index, N, quantile=0.9)

        assert graph.forward_light_nodes.numel() > 0
        assert graph.forward_heavy_nodes.numel() > 0

        x = torch.randn(N, d, device="cuda", dtype=torch.float32)

        out_legacy, arg_legacy = run_forward(graph, x, reduce, grid_size_override=0)
        out_gs, arg_gs = run_forward(graph, x, reduce, grid_size_override=grid_size)

        assert torch.equal(out_legacy, out_gs), (
            f"reduce={reduce} d={d} grid_size={grid_size}: "
            f"max|Δout|={(out_legacy - out_gs).abs().max().item():.3e}"
        )
        assert torch.equal(arg_legacy, arg_gs)

    def test_grid_size_larger_than_bucket_is_capped(self):
        N, d = 64, 32
        torch.manual_seed(2)
        edge_index = make_graph(N, avg_degree=4)
        graph = build_bucketed(edge_index, N, quantile=-1)

        x = torch.randn(N, d, device="cuda", dtype=torch.float32)

        out_ref, arg_ref = run_forward(graph, x, "min", grid_size_override=0)
        out_big, arg_big = run_forward(graph, x, "min", grid_size_override=100_000)
        assert torch.equal(out_ref, out_big)
        assert torch.equal(arg_ref, arg_big)
