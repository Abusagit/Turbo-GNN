"""g-SpMM for any feature width, against the pure-torch reference of test_gspmm_vs_torch.py.

The kernels lay a feature row across lane groups of 1-32 lanes -- or whole warps, in passes, for rows
of more than 32 vectors -- in 16-byte vectors when every row the walk reads is 16-byte aligned and a
whole number of them, and element by element otherwise. Heavy rows are cut into slices of 128 edges
whose results are merged: sums in a fixed order, min/max through packed 64-bit keys. These tests
reach each of those paths for every op, reducer and edge-operand form, in both directions.
"""

import pytest
import torch

from tests.correctness.test_gspmm_vs_torch import (
    BROADCASTABLE_OPS,
    OPS,
    REDUCERS,
    _assert_close,
    _backward_ref,
    _forward_ref,
    _operands,
)
from turbo_gnn.graph import AdjacencyForwardBackwardWithNodeBuckets as Adjacency
from turbo_gnn.ops import gspmm

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
]

DEVICE = "cuda"
TOL = {torch.float32: (1e-5, 1e-6), torch.float16: (4e-3, 1e-4), torch.bfloat16: (3e-2, 1e-3)}

#: Per dtype: one-vector shapes of 1-32 lanes in both layouts, rows ending mid-pass, the two-vector
#: whole-warp shape, and rows of several passes.
DIMS = {
    torch.float32: [1, 3, 4, 8, 12, 33, 64, 100, 128, 255, 260, 513, 1030],
    torch.float16: [1, 2, 7, 8, 16, 24, 48, 100, 136, 257, 520, 2056],
    torch.bfloat16: [5, 8, 56, 264, 1024],
}


def _graph(
    num_nodes=300, num_edges=2400, seed=0, quantile=0.9, index_dtype=torch.int32, hubs=2, hub_degree=600, isolated=10
):
    """Random multigraph with `hubs` destinations of degree ~hub_degree (several heavy slices each) and
    `isolated` nodes without in-edges."""
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    live = num_nodes - isolated
    src = torch.randint(0, num_nodes, (num_edges,), device=DEVICE, generator=gen)
    dst = torch.randint(0, live, (num_edges,), device=DEVICE, generator=gen)
    if hubs:
        hub_src = torch.randint(0, num_nodes, (hubs * hub_degree,), device=DEVICE, generator=gen)
        hub_dst = torch.arange(hubs, device=DEVICE).repeat_interleave(hub_degree)
        src, dst = torch.cat([src, hub_src]), torch.cat([dst, hub_dst])
    graph = Adjacency.from_edge_list(torch.stack([src, dst]), num_nodes, quantile=quantile, index_dtype=index_dtype).to(
        DEVICE
    )
    return graph, num_nodes


def _csr(graph):
    return graph._to_signed_view(graph.forward_indptr), graph._to_signed_view(graph.forward_indices)


def _check(graph, num_nodes, op, reduce, d, dtype, broadcast=False, operands=None, **kwargs):
    indptr, indices = _csr(graph)
    x, e = operands if operands is not None else _operands(op, indices.numel(), num_nodes, d, dtype, broadcast)
    out = gspmm(graph, x, e, op=op, reduce=reduce, **kwargs)
    ref, win = _forward_ref(indptr, indices, x, e, op, reduce, num_nodes, d, dtype)
    what = f"{op}/{reduce} d={d} {dtype} bcast={broadcast}"
    _assert_close_dtype(f"{what} forward", out, ref, dtype)
    grad_out = torch.randn_like(out)
    out.backward(grad_out)
    grad_x, grad_e = _backward_ref(indptr, indices, x, e, op, reduce, num_nodes, d, grad_out, win)
    _assert_close_dtype(f"{what} grad_x", x.grad if x is not None else None, grad_x, dtype)
    _assert_close_dtype(f"{what} grad_e", e.grad if e is not None else None, grad_e, dtype)


def _assert_close_dtype(name, got, want, dtype):
    if dtype in (torch.float32, torch.float16):
        _assert_close(name, got, want, dtype)
        return
    assert (got is None) == (want is None), f"{name}: one side is None"
    if got is None:
        return
    rtol, atol = TOL[dtype]
    diff = (got.float() - want.float()).abs().max().item()
    scale = max(want.float().abs().max().item(), 1e-6)
    assert diff <= atol + rtol * scale, f"{name}: max|diff|={diff:.3e}, scale={scale:.3e}"


@pytest.fixture(scope="module")
def hub_graph():
    return _graph()


DIM_CASES = [(d, dt) for dt in DIMS for d in DIMS[dt]]


@pytest.mark.parametrize("d,dtype", DIM_CASES, ids=[f"d{d}-{str(t)[6:]}" for d, t in DIM_CASES])
@pytest.mark.parametrize(
    "op,reduce", [("copy_u", "sum"), ("mul", "max"), ("div", "sum"), ("copy_e", "min"), ("sub", "max")]
)
def test_dims(hub_graph, op, reduce, d, dtype):
    graph, n = hub_graph
    _check(graph, n, op, reduce, d, dtype)


@pytest.mark.parametrize("d", [3, 64, 100, 520])
@pytest.mark.parametrize("reduce", REDUCERS)
@pytest.mark.parametrize("op", OPS)
def test_ops_and_reducers(hub_graph, op, reduce, d):
    graph, n = hub_graph
    _check(graph, n, op, reduce, d, torch.float32)


@pytest.mark.parametrize(
    "d,dtype", [(8, torch.float32), (100, torch.float16), (264, torch.float32), (1032, torch.float16)]
)
@pytest.mark.parametrize("reduce", REDUCERS)
@pytest.mark.parametrize("op", BROADCASTABLE_OPS)
@pytest.mark.parametrize("shape", ["flat", "column"])
def test_broadcast_edge_operand(hub_graph, op, reduce, d, dtype, shape):
    """Edge operand of one scalar per edge, as [E] or [E, 1], over rows of every shape."""
    graph, n = hub_graph
    x, e = _operands(op, graph.forward_indices.numel(), n, d, dtype, broadcast=True)
    if shape == "column":
        e = e.detach().unsqueeze(1).requires_grad_(True)
    _check(graph, n, op, reduce, d, dtype, broadcast=True, operands=(x, e))


@pytest.mark.parametrize("op,reduce", [("mul", "sum"), ("copy_u", "max"), ("add", "min")])
@pytest.mark.parametrize("d,dtype", [(16, torch.float32), (64, torch.float16), (256, torch.float32)])
def test_misaligned_rows_take_the_element_layout(hub_graph, op, reduce, d, dtype):
    """A contiguous view offset by one element passes through as is but cannot be read in 16-byte
    vectors: the element-wise layout must take it."""
    graph, n = hub_graph
    num_edges = graph.forward_indices.numel()
    gen = torch.Generator(device=DEVICE).manual_seed(3)
    x = torch.randn(n * d + 1, device=DEVICE, generator=gen).to(dtype)[1:].view(n, d)
    assert x.is_contiguous() and x.data_ptr() % 16 != 0
    e = None
    if op != "copy_u":
        e = (torch.rand(num_edges * d + 1, device=DEVICE, generator=gen) + 0.5).to(dtype)[1:].view(num_edges, d)
        e.requires_grad_(True)
    _check(graph, n, op, reduce, d, dtype, operands=(x.requires_grad_(True), e))


@pytest.mark.parametrize("index_dtype", [torch.int64, torch.uint32, torch.uint64])
@pytest.mark.parametrize(
    "op,reduce,bcast", [("mul", "sum", False), ("div", "max", True), ("copy_e", "sum", False), ("copy_u", "min", False)]
)
def test_index_dtypes(index_dtype, op, reduce, bcast):
    graph, n = _graph(index_dtype=index_dtype, seed=4)
    _check(graph, n, op, reduce, 40, torch.float32, broadcast=bcast)


@pytest.mark.parametrize("d", [4, 72, 300])
@pytest.mark.parametrize("op,reduce", [("mul", "min"), ("copy_u", "max"), ("sub", "max")])
def test_heavy_min_max_zero_results_keep_their_bits(op, reduce, d):
    """Heavy rows' min/max merge through packed keys under which every zero is +0; a zero result is
    rebuilt from the winning edge's own message, which must keep its sign."""
    graph, n = _graph(seed=5, hubs=3, hub_degree=900)
    num_edges = graph.forward_indices.numel()
    # Every message is the zero of the sign that wins -- -0 for max (as x - (+0) and x * 1 keep it), +0
    # for min -- so each result's bits are those of its winner.
    x = torch.full((n, d), -0.0 if reduce == "max" else 0.0, device=DEVICE)
    e = None if op == "copy_u" else torch.full((num_edges, d), 0.0 if op == "sub" else 1.0, device=DEVICE)
    out = gspmm(graph, x, e, op=op, reduce=reduce)
    want = torch.full_like(out, -0.0 if reduce == "max" else 0.0)
    has_in = graph._to_signed_view(graph.forward_indptr).diff() > 0
    assert torch.equal(torch.signbit(out[has_in]), torch.signbit(want[has_in]))
    assert torch.equal(out[~has_in], torch.zeros_like(out[~has_in]))


@pytest.mark.parametrize("reduce", ["min", "max"])
@pytest.mark.parametrize("d", [8, 64, 300])
def test_fp16_ties_go_to_the_earliest_edge(reduce, d):
    """Integer-valued fp16 messages tie constantly; the winner -- and so the gradient routing -- must be
    the earliest edge, in light rows and across the slices of heavy ones alike."""
    graph, n = _graph(seed=6, hub_degree=1000)
    num_edges = graph.forward_indices.numel()
    gen = torch.Generator(device=DEVICE).manual_seed(7)
    x = torch.randint(-2, 3, (n, d), device=DEVICE, generator=gen).half().requires_grad_(True)
    e = torch.randint(1, 3, (num_edges, d), device=DEVICE, generator=gen).half().requires_grad_(True)
    _check(graph, n, "mul", reduce, d, torch.float16, operands=(x, e))


@pytest.mark.parametrize("op,reduce", [("mul", "sum"), ("copy_u", "max"), ("copy_e", "sum")])
def test_empty_graph_and_rows(op, reduce):
    """A graph without edges: sums are 0, min/max 0 with no winner, and no gradient anywhere."""
    graph = Adjacency.from_edge_list(
        torch.zeros(2, 0, dtype=torch.long, device=DEVICE), 50, quantile=0.9, index_dtype=torch.int32
    ).to(DEVICE)
    x = torch.randn(50, 9, device=DEVICE, requires_grad=True) if op != "copy_e" else None
    e = torch.randn(0, 9, device=DEVICE, requires_grad=True) if op != "copy_u" else None
    out = gspmm(graph, x, e, op=op, reduce=reduce)
    assert out.shape == (50, 9) and torch.equal(out, torch.zeros_like(out))
    out.sum().backward()
    if x is not None:
        assert torch.equal(x.grad, torch.zeros_like(x))


def test_zero_width():
    graph, n = _graph(seed=8)
    x = torch.randn(n, 0, device=DEVICE, requires_grad=True)
    out = gspmm(graph, x, None, op="copy_u", reduce="sum")
    out.sum().backward()
    assert out.shape == (n, 0) and x.grad.shape == (n, 0)


@pytest.mark.parametrize("op,bcast", [("mul", False), ("div", True), ("copy_u", False)])
@pytest.mark.parametrize("d", [13, 256, 1000])
def test_sum_is_deterministic(hub_graph, op, bcast, d):
    """Sums -- and their gradients -- add heavy-row slices in a fixed order and write each edge
    gradient once: repeated runs agree bitwise."""
    graph, n = hub_graph
    x, e = _operands(op, graph.forward_indices.numel(), n, d, torch.float32, bcast)
    runs = []
    for _ in range(3):
        xi = x.detach().clone().requires_grad_(True)
        ei = e.detach().clone().requires_grad_(True) if e is not None else None
        out = gspmm(graph, xi, ei, op=op, reduce="sum")
        out.backward(torch.ones_like(out))
        runs.append((out, xi.grad, ei.grad if ei is not None else None))
    for other in runs[1:]:
        for a, b in zip(runs[0], other):
            assert (a is None and b is None) or torch.equal(a, b)


def test_arguments_are_validated(hub_graph):
    graph, n = hub_graph
    x = torch.randn(n, 8, device=DEVICE)
    e = torch.randn(graph.forward_indices.numel(), 5, device=DEVICE)
    with pytest.raises(RuntimeError, match="rhs.size\\(1\\)"):
        gspmm(graph, x, e, op="mul", reduce="sum")
    with pytest.raises(RuntimeError, match="dtype"):
        gspmm(graph, x, e[:, :1].expand(-1, 8).half(), op="mul", reduce="sum")


@pytest.mark.parametrize("slice_size", [64, 128])
@pytest.mark.parametrize(
    "op,reduce,bcast", [("mul", "sum", False), ("div", "max", True), ("copy_u", "min", False), ("copy_e", "sum", False)]
)
def test_slice_sizes_of_large_graphs(monkeypatch, slice_size, op, reduce, bcast):
    """The slice size grows with the edges per resident warp (``gspmm_slice_size``); these graphs are
    small enough for 16-edge slices, so the sizes of large graphs are forced here."""
    import turbo_gnn.ops

    monkeypatch.setattr(turbo_gnn.ops, "gspmm_slice_size", lambda graph: slice_size)
    graph, n = _graph(seed=9, hub_degree=700)
    _check(graph, n, op, reduce, 72, torch.float16, broadcast=bcast)


def test_edge_gradient_of_zero_width_rows():
    """d = 0 with a per-edge scalar: the edge gradient is the empty sum, 0 -- not uninitialized memory."""
    graph, n = _graph(seed=10)
    num_edges = graph.forward_indices.numel()
    for reduce in ("sum", "max"):
        x = torch.randn(n, 0, device=DEVICE, requires_grad=True)
        e = torch.full((num_edges,), 2.0, device=DEVICE, requires_grad=True)
        out = gspmm(graph, x, e, op="mul", reduce=reduce)
        out.backward(torch.ones_like(out))
        assert torch.equal(e.grad, torch.zeros_like(e))


@pytest.mark.parametrize(
    "d,dtype", [(7, torch.float32), (33, torch.float32), (63, torch.float32), (13, torch.float16), (50, torch.float16)]
)
@pytest.mark.parametrize("op", ["add", "sub", "copy_e"])
def test_edge_gradient_written_flat(hub_graph, op, d, dtype):
    """add/sub/copy_e sums on single-element rows of one pass write each row's edge gradients as one
    flat block; the lane's feature index wraps by G mod d."""
    graph, n = hub_graph
    _check(graph, n, op, "sum", d, dtype)
