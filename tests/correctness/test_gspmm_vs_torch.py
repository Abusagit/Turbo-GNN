"""g-SpMM against a pure-torch reference, with no DGL dependency.

test_gspmm_vs_dgl.py is the stricter check -- it pins the semantics to the
operator family turbo_gnn is meant to replace -- but it skips wholesale where
DGL is not installed, and its graph tops out around degree 25.  Two kernel
paths are invisible at that size:

* the heavy bucket is only sliced across blocks once some node's degree
  exceeds the slice width, and
* the edge gradient's load balance only matters when one node holds a large
  share of the edges.

A third one hides at the other end: the min/max backward walks the transposed
CSR only while the graph is sparse (E <= 10 N) and scatters over the winning
edges otherwise, so a dense graph is needed to reach the scatter at all.

So this module carries its own reference, a hub graph that reaches the first
two and a dense graph that reaches the third.
"""

import pytest
import torch

from turbo_gnn.graph import AdjacencyForwardBackwardWithNodeBuckets
from turbo_gnn.ops import gspmm

OPS = ["copy_u", "copy_e", "add", "sub", "mul", "div"]
REDUCERS = ["sum", "min", "max"]
BROADCASTABLE_OPS = ["add", "sub", "mul", "div"]

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
]

# Both sides accumulate in float32, so these bound one final rounding plus the
# reassociation the kernels' tiling introduces.  float32 is not tighter than
# 1e-5 because the extension is built with --use_fast_math, whose divide
# differs from torch's.
TOL = {torch.float32: (1e-5, 1e-6), torch.float16: (4e-3, 1e-4)}

BIG = float("inf")


def _csr_dst(indptr):
    """Destination node of every forward-CSR edge position."""
    deg = (indptr[1:] - indptr[:-1]).to(torch.int64)
    return torch.repeat_interleave(torch.arange(deg.numel(), device=indptr.device), deg)


def _edge_values(e, d):
    """Edge operand as [E, d] float32, expanding a per-edge (broadcast) one."""
    ev = (e if e.dim() > 1 else e.unsqueeze(1)).float()
    if ev.size(1) == 1 and d > 1:
        ev = ev.expand(-1, d)
    return ev


def _messages(op, x, e, src, d):
    if op == "copy_u":
        return x.float()[src]
    if op == "copy_e":
        return _edge_values(e, d)  # d comes from e itself
    xe = x.float()[src]
    ev = _edge_values(e, d)
    return {"add": xe + ev, "sub": xe - ev, "mul": xe * ev, "div": xe / ev}[op]


def _message_grads(op, x, e, src, d, g):
    """d(msg)/d(operand) * g, as (gradient wrt lhs slice, wrt rhs slice)."""
    if op == "copy_u":
        return g, None
    if op == "copy_e":
        return None, g
    if op == "add":
        return g, g
    if op == "sub":
        return g, -g
    ev = _edge_values(e, d)
    if op == "mul":
        return g * ev, g * x.float()[src]
    return g / ev, -g * x.float()[src] / (ev * ev)


def _forward_ref(indptr, indices, x, e, op, reduce, num_nodes, d, dtype):
    """Returns (out, winning CSR edge position per element; None for sum).

    Everything is computed in float32, which is what the kernels do -- they
    widen the accumulator and cast once at the end, so a half-precision
    reference would just measure a second rounding.  min/max compare the
    message *narrowed to the operand dtype*, though, because their accumulator
    is cuda_t (ReductionOps<MIN>::AccumType); in fp16 that rounding decides a
    great many ties, and comparing in float would pick different winners and
    route whole rows of gradient elsewhere.
    """
    src = indices.to(torch.int64)
    dst = _csr_dst(indptr)
    msg = _messages(op, x, e, src, d)
    index = dst.unsqueeze(1).expand(-1, d)

    if reduce == "sum":
        out = torch.zeros(num_nodes, d, dtype=torch.float32, device=msg.device)
        return out.index_add(0, dst, msg).to(dtype), None

    sign = 1.0 if reduce == "min" else -1.0
    msg_cmp = msg.to(dtype).float()
    acc = torch.full((num_nodes, d), BIG, dtype=torch.float32, device=msg.device)
    acc = acc.scatter_reduce(0, index, sign * msg_cmp, reduce="amin")
    empty = torch.isinf(acc)

    # First edge (lowest CSR position) reaching the extremum, which is the
    # tie-break turbo_gnn documents.
    eid = torch.arange(msg.size(0), dtype=torch.float32, device=msg.device).unsqueeze(1).expand(-1, d)
    hit = torch.where(sign * msg_cmp == acc[dst], eid, torch.full_like(eid, BIG))
    win = torch.full((num_nodes, d), BIG, dtype=torch.float32, device=msg.device)
    win = win.scatter_reduce(0, index, hit, reduce="amin")

    out = torch.where(empty, torch.zeros_like(acc), sign * acc)
    return out.to(dtype), torch.where(empty, torch.full_like(win, -1), win).long()


def _backward_ref(indptr, indices, x, e, op, reduce, num_nodes, d, grad_out, win):
    src = indices.to(torch.int64)
    dst = _csr_dst(indptr)
    num_edges = src.numel()
    g_out = grad_out.float()

    if reduce == "sum":
        g_edge = g_out[dst]  # every edge carries its destination's gradient
    else:
        valid = win >= 0  # only the winning edge of each element carries any
        g_edge = torch.zeros(num_edges, d, dtype=torch.float32, device=src.device)
        feat = torch.arange(d, device=src.device).unsqueeze(0).expand(num_nodes, -1)
        g_edge[win[valid], feat[valid]] = g_out[valid]

    g_lhs_slice, g_rhs_slice = _message_grads(op, x, e, src, d, g_edge)

    grad_x = None
    if g_lhs_slice is not None:
        acc = torch.zeros(num_nodes, d, dtype=torch.float32, device=src.device)
        grad_x = acc.index_add(0, src, g_lhs_slice).to(x.dtype)

    grad_e = None
    if g_rhs_slice is not None:
        if e.dim() == 1:
            grad_e = g_rhs_slice.sum(1).to(e.dtype)
        elif e.size(1) == 1 and d > 1:
            grad_e = g_rhs_slice.sum(1, keepdim=True).to(e.dtype)
        else:
            grad_e = g_rhs_slice.to(e.dtype)

    return grad_x, grad_e


def _assert_close(name, got, want, dtype):
    """Max-norm relative to the largest element, not elementwise.

    These are reductions: the rounding error of a high-degree sum scales with
    the magnitude of the terms, not of the result, so an output element that
    lands near zero would otherwise fail on an error that is tiny relative to
    everything that went into it.
    """
    assert (got is None) == (want is None), f"{name}: one side is None"
    if got is None:
        return
    rtol, atol = TOL[dtype]
    diff = (got.float() - want.float()).abs().max().item()
    scale = max(want.float().abs().max().item(), 1e-6)
    assert diff <= atol + rtol * scale, f"{name}: max|diff|={diff:.3e}, scale={scale:.3e}"


def _build(edge_index, num_nodes, device):
    # quantile below 1 so the heavy-node kernel is actually reached
    return AdjacencyForwardBackwardWithNodeBuckets.from_edge_list(
        edge_index, num_nodes=num_nodes, quantile=0.95, index_dtype=torch.int32
    ).to(device)


@pytest.fixture(scope="module")
def small_graph():
    """Uniform, ~degree 9. Self-loops keep every in-degree non-zero."""
    device = "cuda"
    num_nodes = 400
    g = torch.Generator(device=device).manual_seed(0)
    src = torch.randint(0, num_nodes, (3000,), device=device, generator=g)
    dst = torch.randint(0, num_nodes, (3000,), device=device, generator=g)
    loops = torch.arange(num_nodes, device=device)
    edge_index = torch.stack([torch.cat([src, loops]), torch.cat([dst, loops])])
    return _build(edge_index, num_nodes, device), num_nodes


@pytest.fixture(scope="module")
def hub_graph():
    """Three destinations of degree ~5000, everyone else ~3.

    Above the heavy-node slice width, so this is the fixture that reaches the
    sliced path and the skewed edge-gradient balance.
    """
    device = "cuda"
    num_nodes, hubs, hub_degree = 4000, 3, 5000
    g = torch.Generator(device=device).manual_seed(0)

    hub_src = torch.randint(0, num_nodes, (hubs * hub_degree,), device=device, generator=g)
    hub_dst = torch.arange(hubs, device=device).repeat_interleave(hub_degree)
    src = torch.randint(0, num_nodes, (num_nodes * 2,), device=device, generator=g)
    dst = torch.randint(0, num_nodes, (num_nodes * 2,), device=device, generator=g)
    loops = torch.arange(num_nodes, device=device)

    edge_index = torch.stack([torch.cat([hub_src, src, loops]), torch.cat([hub_dst, dst, loops])])
    graph = _build(edge_index, num_nodes, device)
    assert graph.max_degree > 1024, f"fixture does not reach the sliced path (max_degree={graph.max_degree})"
    return graph, num_nodes


@pytest.fixture(scope="module")
def source_hub_graph():
    """Three sources of out-degree ~5000, everyone else ~3.

    The mirror image of hub_graph: the hubs sit in the *transposed* CSR, which
    is what the min/max backward walks, so this is the fixture that reaches its
    sliced heavy path.  Sparse enough (E <= 10 N) to stay off the scatter
    fallback.
    """
    device = "cuda"
    num_nodes, hubs, hub_degree = 4000, 3, 5000
    g = torch.Generator(device=device).manual_seed(0)

    hub_src = torch.arange(hubs, device=device).repeat_interleave(hub_degree)
    hub_dst = torch.randint(0, num_nodes, (hubs * hub_degree,), device=device, generator=g)
    src = torch.randint(0, num_nodes, (num_nodes * 2,), device=device, generator=g)
    dst = torch.randint(0, num_nodes, (num_nodes * 2,), device=device, generator=g)
    loops = torch.arange(num_nodes, device=device)

    edge_index = torch.stack([torch.cat([hub_src, src, loops]), torch.cat([hub_dst, dst, loops])])
    graph = _build(edge_index, num_nodes, device)
    num_edges = graph.forward_indices.numel()
    assert graph.backward_max_degree > 1024, f"fixture does not reach the sliced path (backward_max_degree={graph.backward_max_degree})"
    assert graph.backward_max_degree * 16 > num_edges, "fixture does not pass the slice share gate"
    assert num_edges <= 10 * num_nodes, "fixture would take the scatter fallback"
    return graph, num_nodes


@pytest.fixture(scope="module")
def dense_graph():
    """Uniform, ~degree 25: above the average degree at which the min/max
    backward switches from the transposed walk to the scatter over arg_eid."""
    device = "cuda"
    num_nodes, num_edges = 400, 9600
    g = torch.Generator(device=device).manual_seed(0)
    src = torch.randint(0, num_nodes, (num_edges,), device=device, generator=g)
    dst = torch.randint(0, num_nodes, (num_edges,), device=device, generator=g)
    loops = torch.arange(num_nodes, device=device)
    edge_index = torch.stack([torch.cat([src, loops]), torch.cat([dst, loops])])
    graph = _build(edge_index, num_nodes, device)
    assert graph.forward_indices.numel() > 10 * num_nodes, "fixture does not reach the scatter path"
    return graph, num_nodes


def _operands(op, num_edges, num_nodes, d, dtype, broadcast, seed=1):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = None
    if op != "copy_e":
        x = torch.randn(num_nodes, d, device="cuda", dtype=dtype, generator=g).requires_grad_(True)
    e = None
    if op != "copy_u":
        shape = (num_edges,) if broadcast else (num_edges, d)
        # offset away from zero so that div stays well conditioned
        e = (torch.rand(shape, device="cuda", dtype=dtype, generator=g) + 0.5).requires_grad_(True)
    return x, e


def _check_cell(graph, num_nodes, op, reduce, d, dtype, broadcast=False, **gspmm_kwargs):
    indptr, indices = graph.forward_indptr, graph.forward_indices
    x, e = _operands(op, indices.numel(), num_nodes, d, dtype, broadcast)

    out = gspmm(graph, x, e, op=op, reduce=reduce, **gspmm_kwargs)
    ref, win = _forward_ref(indptr, indices, x, e, op, reduce, num_nodes, d, dtype)
    _assert_close(f"{op}/{reduce} d={d} forward", out, ref, dtype)

    grad_out = torch.randn_like(out)
    out.backward(grad_out)
    grad_x, grad_e = _backward_ref(indptr, indices, x, e, op, reduce, num_nodes, d, grad_out, win)
    _assert_close(f"{op}/{reduce} d={d} grad_x", x.grad if x is not None else None, grad_x, dtype)
    _assert_close(f"{op}/{reduce} d={d} grad_e", e.grad if e is not None else None, grad_e, dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("d", [1, 32, 65])
@pytest.mark.parametrize("reduce", REDUCERS)
@pytest.mark.parametrize("op", OPS)
def test_matches_torch_reference(small_graph, op, reduce, d, dtype):
    graph, num_nodes = small_graph
    _check_cell(graph, num_nodes, op, reduce, d, dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("reduce", REDUCERS)
@pytest.mark.parametrize("op", BROADCASTABLE_OPS)
def test_broadcast_edge_data_matches_torch_reference(small_graph, op, reduce, dtype):
    graph, num_nodes = small_graph
    _check_cell(graph, num_nodes, op, reduce, 32, dtype, broadcast=True)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("d", [1, 32, 65])
@pytest.mark.parametrize("reduce", REDUCERS)
@pytest.mark.parametrize("op", OPS)
def test_high_degree_matches_torch_reference(hub_graph, op, reduce, d, dtype):
    """Same table on a graph whose top degree reaches the sliced heavy path."""
    graph, num_nodes = hub_graph
    _check_cell(graph, num_nodes, op, reduce, d, dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("d", [1, 32, 65])
@pytest.mark.parametrize("reduce", ["min", "max"])
@pytest.mark.parametrize("op", OPS)
def test_dense_min_max_matches_torch_reference(dense_graph, op, reduce, d, dtype):
    """The min/max backward on a graph dense enough for the scatter fallback."""
    graph, num_nodes = dense_graph
    _check_cell(graph, num_nodes, op, reduce, d, dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("reduce", ["min", "max"])
@pytest.mark.parametrize("op", BROADCASTABLE_OPS)
def test_dense_broadcast_min_max_matches_torch_reference(dense_graph, op, reduce, dtype):
    graph, num_nodes = dense_graph
    _check_cell(graph, num_nodes, op, reduce, 32, dtype, broadcast=True)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("d", [1, 32, 65])
@pytest.mark.parametrize("reduce", ["min", "max"])
@pytest.mark.parametrize("op", OPS)
def test_source_hub_min_max_matches_torch_reference(source_hub_graph, op, reduce, d, dtype):
    """The min/max backward on a graph whose hubs are sources: its heavy
    bucket is sliced across blocks and folded by the reduce pass."""
    graph, num_nodes = source_hub_graph
    _check_cell(graph, num_nodes, op, reduce, d, dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("reduce", ["min", "max"])
@pytest.mark.parametrize("op", BROADCASTABLE_OPS)
def test_source_hub_broadcast_min_max_matches_torch_reference(source_hub_graph, op, reduce, dtype):
    graph, num_nodes = source_hub_graph
    _check_cell(graph, num_nodes, op, reduce, 32, dtype, broadcast=True)


@pytest.mark.parametrize("broadcast", [False, True])
@pytest.mark.parametrize("pipeline_stages", [1, 2])
@pytest.mark.parametrize("fixture", ["small_graph", "source_hub_graph"])
@pytest.mark.parametrize("reduce", ["min", "max"])
@pytest.mark.parametrize("op", OPS)
def test_min_max_pipeline_matches_torch_reference(request, fixture, op, reduce, pipeline_stages, broadcast):
    """The cp.async prefetch of the arg_eid gather in the min/max backward: on
    the small graph and on the sliced source-hub graph, full-width and in the
    warp-folded broadcast layout."""
    if broadcast and op not in BROADCASTABLE_OPS:
        pytest.skip("op has no broadcastable edge operand")
    graph, num_nodes = request.getfixturevalue(fixture)
    _check_cell(graph, num_nodes, op, reduce, 32, torch.float32, broadcast=broadcast, pipeline_stages=pipeline_stages)
