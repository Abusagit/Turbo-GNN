"""Every scheduler policy must compute the same answer as the historical launch.

The scheduler only changes *which block* handles *which node*. Each (node, head) output row
is owned exclusively by one block and the intra-block reduction order is untouched, so the
forward results must be **bit-identical** across policies -- `torch.equal`, not `allclose`.
That makes these tests a very sharp detector of a scheduling bug.

The two exceptions are documented below: GT's `dK` and GATv2's reduced attention gradient are
float `atomicAdd` accumulations, so reordering nodes reorders the summation.
"""

from __future__ import annotations

import pytest
import torch

from turbo_gnn import AdjacencyForwardBackwardWithNodeBuckets, gatv2_aggr, graph_transformer_aggr, reduction_aggr

BASELINE = "one_per_block"
PERSISTENT = ["grid_stride", "precomputed", "dynamic"]

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="scheduler tests need CUDA")


def _graph(n, deg, quantile=0.9, *, index_dtype=torch.int32, directed=None, seed=0):
    torch.manual_seed(seed)
    dev = torch.device("cuda")
    src = torch.arange(n, device=dev).repeat_interleave(deg)
    dst = torch.randint(0, n, (n * deg,), device=dev)
    ei = torch.stack([src, dst])
    if directed is False:  # symmetrise so the undirected kernels are exercised
        ei = torch.cat([ei, ei.flip(0)], dim=1)
    return AdjacencyForwardBackwardWithNodeBuckets.from_edge_list(
        ei, n, quantile=quantile, index_dtype=index_dtype, is_directed=directed
    ).to(dev)


# ---------------------------------------------------------------------------------------
# reduction_aggr
# ---------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------
# gatv2_aggr / graph_transformer_aggr
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("schedule", PERSISTENT)
@pytest.mark.parametrize("heads,head_dim", [(1, 64), (4, 32), (2, 128)])
@pytest.mark.parametrize("directed", [True, False], ids=["directed", "undirected"])
def test_gatv2_forward_bit_exact(schedule, heads, head_dim, directed):
    n = 3000
    g = _graph(n, 6, directed=directed)
    xl = torch.randn(n, heads, head_dim, device="cuda")
    xr = torch.randn(n, heads, head_dim, device="cuda")
    a = torch.randn(heads, head_dim, device="cuda")
    ref = gatv2_aggr(g, xl, xr, a, 0.2, schedule=BASELINE)
    got = gatv2_aggr(g, xl, xr, a, 0.2, schedule=schedule)
    assert torch.equal(got, ref)


@pytest.mark.parametrize("schedule", PERSISTENT)
@pytest.mark.parametrize("heads,head_dim", [(1, 64), (4, 32)])
@pytest.mark.parametrize("directed", [True, False], ids=["directed", "undirected"])
def test_gt_forward_bit_exact(schedule, heads, head_dim, directed):
    n = 3000
    g = _graph(n, 6, directed=directed)
    q, k, v = (torch.randn(n, heads, head_dim, device="cuda") for _ in range(3))
    ref = graph_transformer_aggr(g, q, q, k, v, None, schedule=BASELINE)
    got = graph_transformer_aggr(g, q, q, k, v, None, schedule=schedule)
    assert torch.equal(got, ref)


@pytest.mark.parametrize("schedule", PERSISTENT)
@pytest.mark.parametrize("directed", [True, False], ids=["directed", "undirected"])
def test_gt_backward(schedule, directed):
    """dQ/dV are exclusively owned; dK uses float atomicAdd in the directed path only."""
    n, heads, head_dim = 2000, 4, 32
    g = _graph(n, 6, directed=directed)

    def run(sched):
        gen = torch.Generator("cuda").manual_seed(11)
        q, k, v = (torch.randn(n, heads, head_dim, device="cuda", generator=gen, requires_grad=True) for _ in range(3))
        out = graph_transformer_aggr(g, q, q, k, v, None, schedule=sched)
        out.backward(torch.ones_like(out))
        return q.grad, k.grad, v.grad

    for got, ref in zip(run(schedule), run(BASELINE)):
        torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("schedule", PERSISTENT)
@pytest.mark.parametrize("directed", [True, False], ids=["directed", "undirected"])
def test_gatv2_backward(schedule, directed):
    n, heads, head_dim = 2000, 4, 32
    g = _graph(n, 6, directed=directed)

    def run(sched):
        gen = torch.Generator("cuda").manual_seed(13)
        xl = torch.randn(n, heads, head_dim, device="cuda", generator=gen, requires_grad=True)
        xr = torch.randn(n, heads, head_dim, device="cuda", generator=gen, requires_grad=True)
        a = torch.randn(heads, head_dim, device="cuda", generator=gen)
        out = gatv2_aggr(g, xl, xr, a, 0.2, schedule=sched)
        out.backward(torch.ones_like(out))
        return xl.grad, xr.grad

    for got, ref in zip(run(schedule), run(BASELINE)):
        torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------------------
# LPT ordering
# ---------------------------------------------------------------------------------------


def test_sorted_by_degree_is_a_permutation_and_descending():
    g = _graph(4000, 6)
    s = g.sorted_by_degree()
    indptr = g._to_signed_view(g.forward_indptr)
    deg = indptr[1:] - indptr[:-1]
    for orig, srt in ((g.forward_light_nodes, s.forward_light_nodes), (g.forward_heavy_nodes, s.forward_heavy_nodes)):
        assert torch.equal(orig.sort().values, srt.sort().values), "sorting must not add or drop nodes"
        d = deg[srt.long()]
        assert torch.all(d[:-1] >= d[1:]), "bucket is not in descending-degree order"


def test_sorted_by_locality_is_a_permutation():
    """RCM reorders which node a block visits; it must not add, drop or duplicate one."""
    pytest.importorskip("scipy")
    g = _graph(4000, 6)
    s = g.sorted_by_locality()
    for orig, srt in ((g.forward_light_nodes, s.forward_light_nodes), (g.forward_heavy_nodes, s.forward_heavy_nodes)):
        assert torch.equal(orig.sort().values, srt.sort().values), "reordering must not add or drop nodes"
    assert not torch.equal(g.forward_light_nodes, s.forward_light_nodes) or g.forward_light_nodes.numel() <= 1


@pytest.mark.parametrize("order", ["sorted_by_degree", "sorted_by_locality"])
def test_node_order_does_not_change_the_answer(order):
    """Visit order is the largest performance lever here, so pin that it is only that.

    The scheduler's `nodes` array chooses which node a block visits; the result is still
    written to that node's own row, so every order must be bit-identical to the natural one.
    """
    if order == "sorted_by_locality":
        pytest.importorskip("scipy")
    g = _graph(5000, 8)
    x = torch.randn(5000, 128, device="cuda")
    ref = reduction_aggr(g, x)
    got = reduction_aggr(getattr(g, order)(), x)
    assert torch.equal(got, ref), f"{order} differs: max|d|={(got - ref).abs().max()}"


def test_lpt_order_does_not_change_the_answer():
    """LPT reorders *when* nodes are processed, never what is computed."""
    g = _graph(5000, 8)
    x = torch.randn(5000, 128, device="cuda")
    ref = reduction_aggr(g, x)
    assert torch.equal(reduction_aggr(g.sorted_by_degree(), x), ref)


# ---------------------------------------------------------------------------------------
# API surface
# ---------------------------------------------------------------------------------------
