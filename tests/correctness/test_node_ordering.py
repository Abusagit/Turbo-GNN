"""Reordering the node buckets must not change the answer.

Degree ordering (LPT) and locality ordering only change *which block* handles *which node*.
Each output row is still owned by exactly one block and the intra-block reduction order is
untouched, so results must be **bit-identical** to the natural order -- `torch.equal`, not
`allclose`.
"""

from __future__ import annotations

import pytest
import torch

from turbo_gnn import AdjacencyForwardBackwardWithNodeBuckets, reduction_aggr

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="ordering tests need CUDA")


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
