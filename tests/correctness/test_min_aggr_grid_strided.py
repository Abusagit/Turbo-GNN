import sys
from pathlib import Path

import pytest
import torch

_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import turbo_gnn._C as _C  # noqa: E402
from turbo_gnn.graph import AdjacencyForwardBackwardWithNodeBuckets  # noqa: E402
from turbo_gnn.scheduling import edge_balanced_partition  # noqa: E402


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestMinAggrBalancedPartition:
    @pytest.mark.parametrize("reduce", ["min", "max"])
    @pytest.mark.parametrize("d", [32, 64, 128])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    @pytest.mark.parametrize("num_blocks", [16, 64, 132])
    def test_balanced_matches_legacy(self, reduce, d, dtype, num_blocks):
        N = 512
        torch.manual_seed(0)
        edge_index = make_graph(N, avg_degree=6)
        graph = build_bucketed(edge_index, N, quantile=-1)

        x = torch.randn(N, d, device="cuda", dtype=dtype)

        out_ref, arg_ref = run_forward(graph, x, reduce, grid_size_override=0)

        sorted_nodes, offsets = edge_balanced_partition(
            graph.forward_light_nodes, graph.forward_indptr, num_blocks
        )
        graph.forward_light_nodes = sorted_nodes
        out_bal, arg_bal = _C.reduction_aggr_forward_partitioned(
            graph.forward_indptr,
            graph.forward_indices,
            x,
            graph.forward_light_nodes,
            graph.forward_heavy_nodes,
            max_degree_of(graph.forward_indptr),
            8, 128, False, 32, 8,
            reduce, 0,
            offsets,
        )

        assert torch.equal(out_ref, out_bal), (
            f"reduce={reduce} d={d} dtype={dtype} num_blocks={num_blocks}: "
            f"max|Δ|={(out_ref.float() - out_bal.float()).abs().max().item():.3e}"
        )
        assert torch.equal(arg_ref, arg_bal)

    @pytest.mark.parametrize("num_blocks", [16, 64])
    def test_balanced_mixed_light_heavy_matches_legacy(self, num_blocks):
        N, d = 1024, 64
        torch.manual_seed(3)
        edge_index = make_graph(N, avg_degree=8)
        graph = build_bucketed(edge_index, N, quantile=0.9)

        assert graph.forward_light_nodes.numel() > 0
        assert graph.forward_heavy_nodes.numel() > 0

        x = torch.randn(N, d, device="cuda", dtype=torch.float32)
        out_ref, arg_ref = run_forward(graph, x, "min", grid_size_override=0)

        sorted_nodes, offsets = edge_balanced_partition(
            graph.forward_light_nodes, graph.forward_indptr, num_blocks
        )
        graph.forward_light_nodes = sorted_nodes
        out_bal, arg_bal = _C.reduction_aggr_forward_partitioned(
            graph.forward_indptr,
            graph.forward_indices,
            x,
            graph.forward_light_nodes,
            graph.forward_heavy_nodes,
            max_degree_of(graph.forward_indptr),
            8, 128, False, 32, 8,
            "min", 0,
            offsets,
        )

        assert torch.equal(out_ref, out_bal), (
            f"max|Δ|={(out_ref - out_bal).abs().max().item():.3e}"
        )
        assert torch.equal(arg_ref, arg_bal)

    @pytest.mark.parametrize("reduce", ["min", "max"])
    @pytest.mark.parametrize("d", [32, 64])
    @pytest.mark.parametrize("num_blocks", [16, 64, 132])
    def test_balanced_dynamic_schedule_matches(self, reduce, d, num_blocks):
        N = 512
        torch.manual_seed(11)
        edge_index = make_graph(N, avg_degree=6)
        graph = build_bucketed(edge_index, N, quantile=-1)

        x = torch.randn(N, d, device="cuda", dtype=torch.float32)

        out_ref, arg_ref = run_forward(graph, x, reduce, grid_size_override=0)

        sorted_nodes, offsets = edge_balanced_partition(
            graph.forward_light_nodes, graph.forward_indptr, num_blocks
        )
        graph.forward_light_nodes = sorted_nodes
        out_dyn, arg_dyn = _C.reduction_aggr_forward_partitioned(
            graph.forward_indptr,
            graph.forward_indices,
            x,
            graph.forward_light_nodes,
            graph.forward_heavy_nodes,
            max_degree_of(graph.forward_indptr),
            8, 128, False, 32, 8,
            reduce, 0,
            offsets,
            True,
        )

        assert torch.equal(out_ref, out_dyn), (
            f"reduce={reduce} d={d} num_blocks={num_blocks}: "
            f"max|Δ|={(out_ref.float() - out_dyn.float()).abs().max().item():.3e}"
        )
        assert torch.equal(arg_ref, arg_dyn)

    @pytest.mark.parametrize("reduce", ["min", "max"])
    @pytest.mark.parametrize("grid_size", [32, 132])
    def test_gsl_dynamic_schedule_matches(self, reduce, grid_size):
        N, d = 512, 32
        torch.manual_seed(12)
        edge_index = make_graph(N, avg_degree=6)
        graph = build_bucketed(edge_index, N, quantile=-1)

        x = torch.randn(N, d, device="cuda", dtype=torch.float32)

        out_ref, arg_ref = run_forward(graph, x, reduce, grid_size_override=0)

        empty_offsets = torch.empty(0, dtype=torch.int32, device="cuda")
        out_dyn, arg_dyn = _C.reduction_aggr_forward_partitioned(
            graph.forward_indptr,
            graph.forward_indices,
            x,
            graph.forward_light_nodes,
            graph.forward_heavy_nodes,
            max_degree_of(graph.forward_indptr),
            8, 128, False, 32, 8,
            reduce, grid_size,
            empty_offsets,
            True,
        )

        assert torch.equal(out_ref, out_dyn), (
            f"reduce={reduce} grid_size={grid_size}: "
            f"max|Δ|={(out_ref.float() - out_dyn.float()).abs().max().item():.3e}"
        )
        assert torch.equal(arg_ref, arg_dyn)

    def test_balanced_dynamic_mixed_light_heavy(self):
        N, d = 1024, 64
        torch.manual_seed(13)
        edge_index = make_graph(N, avg_degree=8)
        graph = build_bucketed(edge_index, N, quantile=0.9)

        assert graph.forward_light_nodes.numel() > 0
        assert graph.forward_heavy_nodes.numel() > 0

        x = torch.randn(N, d, device="cuda", dtype=torch.float32)
        out_ref, arg_ref = run_forward(graph, x, "min", grid_size_override=0)

        sorted_nodes, offsets = edge_balanced_partition(
            graph.forward_light_nodes, graph.forward_indptr, 64
        )
        graph.forward_light_nodes = sorted_nodes
        out_dyn, arg_dyn = _C.reduction_aggr_forward_partitioned(
            graph.forward_indptr,
            graph.forward_indices,
            x,
            graph.forward_light_nodes,
            graph.forward_heavy_nodes,
            max_degree_of(graph.forward_indptr),
            8, 128, False, 32, 8,
            "min", 0,
            offsets,
            True,
        )

        assert torch.equal(out_ref, out_dyn)
        assert torch.equal(arg_ref, arg_dyn)

    @pytest.mark.parametrize("num_blocks", [8, 32, 128])
    def test_partition_edge_balance(self, num_blocks):
        N = 2048
        torch.manual_seed(4)
        edge_index = make_graph(N, avg_degree=16)
        graph = build_bucketed(edge_index, N, quantile=-1)

        _, offsets = edge_balanced_partition(
            graph.forward_light_nodes, graph.forward_indptr, num_blocks
        )

        indptr = graph.forward_indptr
        if indptr.dtype == torch.uint32:
            indptr = indptr.view(torch.int32)
        degrees = (indptr[1:] - indptr[:-1]).float()
        sorted_nodes, _ = edge_balanced_partition(
            graph.forward_light_nodes, graph.forward_indptr, num_blocks
        )
        block_degrees = degrees[sorted_nodes.long()]

        offsets_cpu = offsets.cpu().tolist()
        loads = [
            block_degrees[offsets_cpu[b]:offsets_cpu[b + 1]].sum().item()
            for b in range(num_blocks)
            if offsets_cpu[b + 1] > offsets_cpu[b]
        ]
        if len(loads) > 1:
            imbalance = (max(loads) - min(loads)) / (sum(loads) / len(loads))
            assert imbalance < 1.0, f"Block loads too imbalanced: {imbalance:.2f}"
