"""The edge-sliced heavy path must agree with the node-per-block one it replaces.

`forward_heavy_edge_slice=0` runs one block per heavy node; a positive value splits each heavy
node's edge list into slices of that size, one block each, and merges the partials in a second
kernel. Both compute the same attention, so the only question these tests ask is whether the
decomposition changed the answer.

Agreement is close but not bit-exact by construction: the online-softmax merge sums a node's
contributions in a different association order once the edges are grouped differently, and
floating-point addition is not associative. The tolerances below are for that regrouping alone,
not for a different algorithm.

The cases that matter are the boundaries -- a degree that is an exact multiple of the slice
size, one that is a single edge short, a node with no edges at all, and an empty heavy bucket.
"""

from __future__ import annotations

import pytest
import torch

from turbo_gnn import gatv2_aggr, graph_transformer_aggr
from turbo_gnn.graph import AdjacencyForwardBackwardWithNodeBuckets as Adjacency

SLICES = [128, 256, 1024]


def _graph(edge_index, n, quantile=0.9, index_dtype=torch.int32):
    return Adjacency.from_edge_list(edge_index, n, quantile=quantile, index_dtype=index_dtype).to("cuda")


def _hub_graph(n=4000, deg=4, hubs=6, hub_deg=1000, seed=0, index_dtype=torch.int32, quantile=0.9):
    """Heavy-tailed graph: a few hubs land in the heavy bucket, the rest stay light."""
    g = torch.Generator().manual_seed(seed)
    src = torch.arange(n).repeat_interleave(deg)
    dst = torch.randint(0, n, (n * deg,), generator=g)
    hub_src = torch.randint(0, n, (hubs * hub_deg,), generator=g)
    hub_dst = torch.randint(0, hubs, (hubs * hub_deg,), generator=g)
    ei = torch.stack([torch.cat([src, hub_src]), torch.cat([dst, hub_dst])])
    return _graph(ei, n, quantile=quantile, index_dtype=index_dtype)


def _qkv(n, heads, dim, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return tuple(torch.randn(n, heads, dim, device="cuda", generator=g) for _ in range(3))


def _run(graph, q, k, v, slice_size, conv="gt", **kw):
    """Run one convolution with the heavy bucket decomposed either way.

    Both convolutions take the same slice parameter and both merge online-softmax partials, so
    the tests below cover them by parametrising this one call.
    """
    if conv == "gt":
        return graph_transformer_aggr(graph, q, q, k, v, None, forward_heavy_edge_slice=slice_size, **kw)
    # gatv2_aggr takes (graph, x, x_neighbors, attention_weights, negative_slope).
    a = torch.ones(q.shape[1], q.shape[2], device=q.device) * 0.05
    return gatv2_aggr(graph, q, k, a, 0.2, forward_heavy_edge_slice=slice_size, **kw)


CONVS = ["gt", "gat_v2"]


def _assert_close(ref, got, what):
    assert torch.isfinite(got).all(), f"{what}: non-finite values in sliced output"
    torch.testing.assert_close(got, ref, rtol=2e-4, atol=2e-5, msg=lambda m: f"{what}\n{m}")


@pytest.mark.parametrize("conv", CONVS)
@pytest.mark.parametrize("slice_size", SLICES)
@pytest.mark.parametrize("heads,dim", [(1, 128), (1, 256), (4, 64)])
def test_matches_node_per_block(conv, slice_size, heads, dim):
    g = _hub_graph()
    n = g.forward_indptr.numel() - 1
    q, k, v = _qkv(n, heads, dim)
    _assert_close(
        _run(g, q, k, v, 0, conv=conv),
        _run(g, q, k, v, slice_size, conv=conv),
        f"{conv} slice={slice_size} H={heads} D={dim}",
    )


@pytest.mark.parametrize("conv", CONVS)
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64, torch.uint32, torch.uint64])
def test_index_dtypes(conv, index_dtype):
    g = _hub_graph(index_dtype=index_dtype)
    n = g.forward_indptr.numel() - 1
    q, k, v = _qkv(n, 1, 128)
    _assert_close(_run(g, q, k, v, 0, conv=conv), _run(g, q, k, v, 128, conv=conv), f"{conv} {index_dtype}")


@pytest.mark.parametrize("conv", CONVS)
@pytest.mark.parametrize("slice_size", [64, 128])
@pytest.mark.parametrize("delta", [0, -1, 1])
def test_degree_at_slice_boundary(conv, slice_size, delta):
    """A hub whose degree is an exact multiple of the slice size, or one edge either side.

    This is where an off-by-one in the slice table shows up: the final slice is empty, or
    overruns the row, or a whole slice goes missing.
    """
    n, hub_deg = 2000, slice_size * 3 + delta
    g_ = torch.Generator().manual_seed(1)
    src = torch.arange(n).repeat_interleave(2)
    dst = torch.randint(0, n, (n * 2,), generator=g_)
    hub_src = torch.randint(0, n, (hub_deg,), generator=g_)
    hub_dst = torch.zeros(hub_deg, dtype=torch.long)  # every extra edge points at node 0
    ei = torch.stack([torch.cat([src, hub_src]), torch.cat([dst, hub_dst])])
    g = _graph(ei, n, quantile=0.999)
    q, k, v = _qkv(n, 1, 128)
    _assert_close(
        _run(g, q, k, v, 0, conv=conv),
        _run(g, q, k, v, slice_size, conv=conv),
        f"{conv} deg={hub_deg} slice={slice_size}",
    )


@pytest.mark.parametrize("conv", CONVS)
def test_isolated_nodes_present(conv):
    """Nodes with no edges must still get a zeroed row and -inf logsumexp through the merge."""
    n = 3000
    g_ = torch.Generator().manual_seed(2)
    # only the first half of the nodes emit edges, so the rest are isolated
    src = torch.randint(0, n // 2, (6000,), generator=g_)
    dst = torch.randint(0, n // 2, (6000,), generator=g_)
    hub_src = torch.randint(0, n, (3000,), generator=g_)
    hub_dst = torch.zeros(3000, dtype=torch.long)
    ei = torch.stack([torch.cat([src, hub_src]), torch.cat([dst, hub_dst])])
    g = _graph(ei, n, quantile=0.99)
    q, k, v = _qkv(n, 1, 128)
    _assert_close(_run(g, q, k, v, 0, conv=conv), _run(g, q, k, v, 128, conv=conv), f"{conv} isolated nodes")


@pytest.mark.parametrize("conv", CONVS)
def test_empty_heavy_bucket(conv):
    """quantile=-1 puts every node in the light bucket; the split path must be a no-op."""
    g = _hub_graph(quantile=-1)
    assert g.forward_heavy_nodes.numel() == 0
    n = g.forward_indptr.numel() - 1
    q, k, v = _qkv(n, 1, 128)
    _assert_close(_run(g, q, k, v, 0, conv=conv), _run(g, q, k, v, 128, conv=conv), f"{conv} empty heavy bucket")


@pytest.mark.parametrize("conv", CONVS)
@pytest.mark.parametrize("bucket_launch", ["sequential", "concurrent"])
def test_bucket_launch_modes(conv, bucket_launch):
    """The partial buffers are allocated on the heavy stream, so concurrent mode must be safe."""
    g = _hub_graph()
    n = g.forward_indptr.numel() - 1
    q, k, v = _qkv(n, 1, 128)
    ref = _run(g, q, k, v, 0, conv=conv, forward_bucket_launch=bucket_launch)
    got = _run(g, q, k, v, 128, conv=conv, forward_bucket_launch=bucket_launch)
    _assert_close(ref, got, f"{conv} bucket_launch={bucket_launch}")


# --- min_aggr -------------------------------------------------------------------------------
#
# The reduction path differs from the attention ones in a way worth testing separately: its
# per-slice partials fold together through a 64-bit atomicMin/atomicMax on a packed
# (value, index) pair. That merge is order-independent, so regrouping the edges cannot change
# the result at all -- these assertions are for exact equality, not a tolerance.


def _run_min(graph, x, slice_size, **kw):
    from turbo_gnn import reduction_aggr

    return reduction_aggr(graph, x, forward_heavy_edge_slice=slice_size, **kw)


@pytest.mark.parametrize("slice_size", SLICES)
@pytest.mark.parametrize("dim", [128, 256])
def test_min_aggr_bit_exact(slice_size, dim):
    g = _hub_graph()
    n = g.forward_indptr.numel() - 1
    x = torch.randn(n, dim, device="cuda", generator=torch.Generator(device="cuda").manual_seed(3))
    ref, got = _run_min(g, x, 0), _run_min(g, x, slice_size)
    torch.testing.assert_close(got, ref, rtol=0, atol=0, msg=lambda m: f"slice={slice_size} D={dim}\n{m}")


@pytest.mark.parametrize("delta", [0, -1, 1])
def test_min_aggr_degree_at_slice_boundary(delta):
    slice_size = 128
    n, hub_deg = 2000, slice_size * 3 + delta
    g_ = torch.Generator().manual_seed(4)
    src = torch.arange(n).repeat_interleave(2)
    dst = torch.randint(0, n, (n * 2,), generator=g_)
    hub_src = torch.randint(0, n, (hub_deg,), generator=g_)
    hub_dst = torch.zeros(hub_deg, dtype=torch.long)
    ei = torch.stack([torch.cat([src, hub_src]), torch.cat([dst, hub_dst])])
    g = _graph(ei, n, quantile=0.999)
    x = torch.randn(n, 128, device="cuda", generator=torch.Generator(device="cuda").manual_seed(5))
    torch.testing.assert_close(_run_min(g, x, slice_size), _run_min(g, x, 0), rtol=0, atol=0)


def test_min_aggr_empty_heavy_bucket():
    g = _hub_graph(quantile=-1)
    n = g.forward_indptr.numel() - 1
    x = torch.randn(n, 128, device="cuda", generator=torch.Generator(device="cuda").manual_seed(6))
    torch.testing.assert_close(_run_min(g, x, 128), _run_min(g, x, 0), rtol=0, atol=0)


# --- GT backward ----------------------------------------------------------------------------
#
# Only the directed path is bucketed, so only it has a heavy bucket to split. Backward partials
# are plain sums (alpha is recomputed from the saved logsumexp rather than tracked online), and
# dK is left to its existing atomic scatter, so slicing changes only the grouping of additions.


def _gt_backward_grads(graph, q, k, v, slice_size):
    q = q.detach().requires_grad_(True)
    k = k.detach().requires_grad_(True)
    v = v.detach().requires_grad_(True)
    out = graph_transformer_aggr(graph, q, q, k, v, None, backward_heavy_edge_slice=slice_size)
    out.backward(torch.ones_like(out))
    return q.grad, k.grad, v.grad


@pytest.mark.parametrize("slice_size", [128, 512])
@pytest.mark.parametrize("dim", [128, 256])
def test_gt_backward_matches_node_per_block(slice_size, dim):
    # directed=True so the CSR^T path (the bucketed one) is exercised
    g = _hub_graph()
    g.is_directed = True
    n = g.forward_indptr.numel() - 1
    q, k, v = _qkv(n, 1, dim)
    ref = _gt_backward_grads(g, q, k, v, 0)
    got = _gt_backward_grads(g, q, k, v, slice_size)
    for name, r, t in zip(("dQ", "dK", "dV"), ref, got):
        assert torch.isfinite(t).all(), f"{name}: non-finite with slice={slice_size}"
        torch.testing.assert_close(t, r, rtol=2e-4, atol=2e-5, msg=lambda m: f"{name} slice={slice_size} D={dim}\n{m}")


def test_gt_backward_empty_heavy_bucket():
    g = _hub_graph(quantile=-1)
    g.is_directed = True
    n = g.forward_indptr.numel() - 1
    q, k, v = _qkv(n, 1, 128)
    for r, t in zip(_gt_backward_grads(g, q, k, v, 0), _gt_backward_grads(g, q, k, v, 128)):
        torch.testing.assert_close(t, r, rtol=2e-4, atol=2e-5)


# --- device-relative slice sizing ---------------------------------------------------------------
#
# The slice can be given as a raw edge count or sized to fill the device with N blocks per SM,
# derived from the heavy bucket's edge count. The second form is what the autotuner searches: a
# degree statistic does not predict the optimum (the bucketing threshold's implied constant spans
# 1125x across graphs) whereas block count does (7x).


def test_blocks_per_sm_maps_to_expected_slice():
    g = _hub_graph()
    e = g.heavy_edge_count("forward")
    sm = torch.cuda.get_device_properties(g.device).multi_processor_count
    assert e > 0
    assert g.heavy_slice_for_blocks_per_sm("forward", 0) == 0, "0 must disable slicing"
    for bps in (8, 16, 32, 64):
        assert g.heavy_slice_for_blocks_per_sm("forward", bps) == max(1, round(e / (bps * sm)))


def test_blocks_per_sm_matches_equivalent_absolute_slice():
    """Same decomposition either way, so the two spellings must agree exactly."""
    g = _hub_graph()
    n = g.forward_indptr.numel() - 1
    q, k, v = _qkv(n, 1, 128)
    for bps in (16, 64):
        absolute = g.heavy_slice_for_blocks_per_sm("forward", bps)
        by_count = graph_transformer_aggr(g, q, q, k, v, None, forward_heavy_edge_slice=absolute)
        by_target = graph_transformer_aggr(g, q, q, k, v, None, forward_heavy_slice_blocks_per_sm=bps)
        torch.testing.assert_close(by_target, by_count, rtol=0, atol=0, msg=lambda m: f"bps={bps}\n{m}")


def test_absolute_slice_overrides_blocks_per_sm():
    """An explicit edge count wins, so a pinned configuration stays pinned."""
    g = _hub_graph()
    n = g.forward_indptr.numel() - 1
    q, k, v = _qkv(n, 1, 128)
    pinned = graph_transformer_aggr(g, q, q, k, v, None, forward_heavy_edge_slice=128)
    both = graph_transformer_aggr(
        g, q, q, k, v, None, forward_heavy_edge_slice=128, forward_heavy_slice_blocks_per_sm=64
    )
    torch.testing.assert_close(both, pinned, rtol=0, atol=0)


def test_heavy_edge_count_matches_csr():
    g = _hub_graph()
    ip = g._to_signed_view(g.forward_indptr).to(torch.int64)
    deg = (ip[1:] - ip[:-1]).index_select(0, g.forward_heavy_nodes.to(torch.int64))
    assert g.heavy_edge_count("forward") == int(deg.sum())


def test_empty_heavy_bucket_sizing_is_noop():
    g = _hub_graph(quantile=-1)
    assert g.heavy_edge_count("forward") == 0
    assert g.heavy_slice_for_blocks_per_sm("forward", 32) == 0


# --- undirected GATv2 backward bucketing ---------------------------------------------------------
#
# The undirected backward path (G kernel + ALR kernel, both over the forward CSR) used to run one
# launch across every node, with no light/heavy split at all. On twitch-views that was 99% of the
# backward pass at ~25% achieved occupancy. Bucketing it splits the launch in two, which changes
# only *which block* computes a node -- never the arithmetic -- so results must be unchanged.
#
# The ordering constraint worth guarding: ALR reads G[neighbor_j] for arbitrary neighbours, not
# just those in its own bucket, so both G launches have to finish before either ALR launch starts.
# A bug there shows up as wrong gradients for nodes whose neighbours sit in the other bucket, which
# is exactly what a hub-heavy graph exercises.


def _undirected_graph(n=4000, deg=4, hubs=8, hub_deg=900, seed=11, quantile=0.9):
    """Symmetric adjacency with hubs, so light and heavy buckets are both populated."""
    g = torch.Generator().manual_seed(seed)
    src = torch.arange(n).repeat_interleave(deg)
    dst = torch.randint(0, n, (n * deg,), generator=g)
    hub_src = torch.randint(0, n, (hubs * hub_deg,), generator=g)
    hub_dst = torch.randint(0, hubs, (hubs * hub_deg,), generator=g)
    s = torch.cat([src, hub_src])
    d = torch.cat([dst, hub_dst])
    # symmetrise: the undirected backward path is only selected when both CSRs match
    ei = torch.stack([torch.cat([s, d]), torch.cat([d, s])])
    graph = Adjacency.from_edge_list(ei, n, quantile=quantile, index_dtype=torch.int32).to("cuda")
    assert not graph.is_directed, "test needs the undirected backward path"
    return graph


def _gatv2_backward_grads(graph, x, xn, a, **kw):
    x = x.detach().requires_grad_(True)
    xn = xn.detach().requires_grad_(True)
    a = a.detach().requires_grad_(True)
    out = gatv2_aggr(graph, x, xn, a, 0.2, **kw)
    out.backward(torch.ones_like(out))
    return x.grad, xn.grad, a.grad


@pytest.mark.parametrize("quantile", [0.9, 0.99])
@pytest.mark.parametrize("dim", [128, 256])
def test_undirected_backward_bucketing_matches_unbucketed(quantile, dim):
    """quantile=-1 puts every node in the light bucket, i.e. the old single-launch behaviour."""
    g_split = _undirected_graph(quantile=quantile)
    g_whole = _undirected_graph(quantile=-1)
    assert g_split.forward_heavy_nodes.numel() > 0, "no heavy bucket to test"
    assert g_whole.forward_heavy_nodes.numel() == 0

    n = g_split.forward_indptr.numel() - 1
    q, k, _ = _qkv(n, 1, dim)
    a = torch.ones(1, dim, device="cuda") * 0.05
    for name, ref, got in zip(
        ("grad_x", "grad_x_neighbors", "grad_attn"),
        _gatv2_backward_grads(g_whole, q, k, a),
        _gatv2_backward_grads(g_split, q, k, a),
    ):
        assert torch.isfinite(got).all(), f"{name}: non-finite with quantile={quantile}"
        torch.testing.assert_close(got, ref, rtol=2e-4, atol=2e-5, msg=lambda m: f"{name} q={quantile} D={dim}\n{m}")


@pytest.mark.parametrize("bucket_launch", ["sequential", "concurrent"])
def test_undirected_backward_bucket_launch_modes(bucket_launch):
    """Concurrent mode runs light and heavy on separate streams; the G/ALR join must still hold."""
    g = _undirected_graph()
    n = g.forward_indptr.numel() - 1
    q, k, _ = _qkv(n, 1, 128)
    a = torch.ones(1, 128, device="cuda") * 0.05
    ref = _gatv2_backward_grads(g, q, k, a, backward_bucket_launch="sequential")
    got = _gatv2_backward_grads(g, q, k, a, backward_bucket_launch=bucket_launch)
    for name, r, t in zip(("grad_x", "grad_x_neighbors", "grad_attn"), ref, got):
        torch.testing.assert_close(t, r, rtol=2e-4, atol=2e-5, msg=lambda m: f"{name} {bucket_launch}\n{m}")


@pytest.mark.parametrize("bps", [8, 32, 64])
@pytest.mark.parametrize("dim", [128, 256])
def test_undirected_backward_slicing_matches_node_per_block(bps, dim):
    """Slicing the undirected backward's heavy bucket must not change the gradients.

    Both kernels accumulate plain sums over a node's own edges, so a slice's partial is a partial
    sum and the merge is addition -- the only thing that changes is which block adds what, and in
    what order, which costs floating-point associativity and nothing else.
    """
    g = _undirected_graph()
    n = g.forward_indptr.numel() - 1
    q, k, _ = _qkv(n, 1, dim)
    a = torch.ones(1, dim, device="cuda") * 0.05
    ref = _gatv2_backward_grads(g, q, k, a)
    got = _gatv2_backward_grads(g, q, k, a, backward_heavy_slice_blocks_per_sm=bps)
    for name, r, t in zip(("grad_x", "grad_x_neighbors", "grad_attn"), ref, got):
        assert torch.isfinite(t).all(), f"{name}: non-finite at bps={bps}"
        torch.testing.assert_close(t, r, rtol=2e-4, atol=2e-5, msg=lambda m: f"{name} bps={bps} D={dim}\n{m}")


def test_undirected_backward_slicing_empty_heavy_bucket():
    g = _undirected_graph(quantile=-1)
    assert g.forward_heavy_nodes.numel() == 0
    n = g.forward_indptr.numel() - 1
    q, k, _ = _qkv(n, 1, 128)
    a = torch.ones(1, 128, device="cuda") * 0.05
    ref = _gatv2_backward_grads(g, q, k, a)
    got = _gatv2_backward_grads(g, q, k, a, backward_heavy_slice_blocks_per_sm=32)
    for r, t in zip(ref, got):
        torch.testing.assert_close(t, r, rtol=2e-4, atol=2e-5)
