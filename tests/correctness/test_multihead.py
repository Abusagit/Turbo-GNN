"""Multi-head (DGL-layout) operands for gsddmm and gspmm.

``gsddmm`` accepts ``[rows, H, D]`` operands and computes every head in one
launch (the head is a grid dimension; rows are ``H * D`` apart). ``gspmm`` takes
``[N, H, D]`` node data with ``[E, H, D]`` (elementwise), ``[E, H]`` / ``[E, H, 1]``
(one scalar per edge and head, broadcast over that head's ``D`` features) or
``[E]`` edge data. Both are checked against an fp32 PyTorch reference (values
and gradients) and, for gsddmm, against the per-head 2-D calls they replace.
"""

import pytest
import torch

from turbo_gnn import gsddmm
from turbo_gnn._gsddmm import _edge_endpoints
from turbo_gnn.graph import AdjacencyForwardBackwardWithNodeBuckets
from turbo_gnn.ops import gspmm

if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)

DEVICE = torch.device("cuda")

GSDDMM_OPS = ["add", "sub", "mul", "div", "dot"]
MEMBER_PAIRS = [("src", "dst"), ("dst", "src"), ("src", "edge"), ("edge", "src"), ("dst", "edge"), ("edge", "dst")]
HEAD_SHAPES = [(1, 64), (2, 32), (4, 128), (8, 32), (3, 64), (2, 256)]

#: The graph has hub nodes with ~1000 in-edges. The backward rounds each edge's
#: partial to the operand dtype before the fp32 accumulation (as the 2-D path
#: does), so a hub's low-precision gradient carries ~sqrt(degree) roundings; the
#: atol below covers that. Exactness against the 2-D kernel is checked separately
#: (test_gsddmm_multihead_equals_per_head_calls).
_TOL = {
    torch.float32: {"rtol": 2e-4, "atol": 2e-4},
    torch.float16: {"rtol": 2e-2, "atol": 5e-2},
    torch.bfloat16: {"rtol": 6e-2, "atol": 2.5e-1},
}


def _graph(num_nodes=300, num_edges=4000, hubs=True, seed=0):
    """Random directed multigraph; with ``hubs`` a few nodes get hundreds of
    in-edges so both node buckets (light and heavy) are exercised."""
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    src = torch.randint(0, num_nodes, (num_edges,), device=DEVICE, generator=gen)
    dst = torch.randint(0, num_nodes, (num_edges,), device=DEVICE, generator=gen)
    if hubs:
        dst[: num_edges // 4] = torch.randint(0, 3, (num_edges // 4,), device=DEVICE, generator=gen)
    edge_index = torch.stack([src, dst])
    return AdjacencyForwardBackwardWithNodeBuckets.from_edge_list(
        edge_index, num_nodes, quantile=0.99, index_dtype=torch.int32, is_directed=True
    ).to(DEVICE)


@pytest.fixture(scope="module")
def graph():
    return _graph()


def _rows(graph, target):
    return graph.forward_indices.numel() if target == "edge" else graph.forward_indptr.numel() - 1


def _close(got, ref, dtype, what):
    assert got is not None, f"{what}: no gradient"
    torch.testing.assert_close(got.float(), ref, **_TOL[dtype], msg=lambda m: f"{what}\n{m}")


# =============================================================================
# gsddmm
# =============================================================================


def _sddmm_operands(graph, op, lt, rt, heads, dim, dtype, seed=0):
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    lhs = torch.randn(_rows(graph, lt), heads, dim, device=DEVICE, dtype=dtype, generator=gen)
    rhs = torch.randn(_rows(graph, rt), heads, dim, device=DEVICE, dtype=dtype, generator=gen).abs() + 1.0
    e = graph.forward_indices.numel()
    d_out = torch.randn((e, heads) if op == "dot" else (e, heads, dim), device=DEVICE, dtype=dtype, generator=gen)
    return lhs, rhs, d_out


def _sddmm_reference(graph, lhs, rhs, d_out, op, lt, rt):
    src, dst = _edge_endpoints(graph, by_src=False)
    l32 = lhs.detach().float().requires_grad_(True)
    r32 = rhs.detach().float().requires_grad_(True)
    pick = lambda t, m: t.index_select(0, src) if m == "src" else (t.index_select(0, dst) if m == "dst" else t)
    a, b = pick(l32, lt), pick(r32, rt)
    out = {"add": lambda: a + b, "sub": lambda: a - b, "mul": lambda: a * b, "div": lambda: a / b, "dot": lambda: (a * b).sum(-1), "copy": lambda: a}[op]()
    out.backward(d_out.detach().float())
    return out.detach(), l32.grad, (r32.grad if op != "copy" else None)


def _sddmm_kernel(graph, lhs, rhs, d_out, op, lt, rt, variant, backward_variant):
    l = lhs.detach().clone().requires_grad_(True)
    r = None if op == "copy" else rhs.detach().clone().requires_grad_(True)
    out = gsddmm(graph, l, r, op=op, lhs_target=lt, rhs_target=rt, variant=variant, backward_variant=backward_variant)
    out.backward(d_out)
    return out, l.grad, (r.grad if r is not None else None)


@pytest.mark.parametrize("backward_variant", ["node", "edge"])
@pytest.mark.parametrize("variant", ["node", "edge"])
@pytest.mark.parametrize("lt,rt", MEMBER_PAIRS)
@pytest.mark.parametrize("op", GSDDMM_OPS)
def test_gsddmm_multihead_matches_reference(graph, op, lt, rt, variant, backward_variant):
    dtype = torch.float32
    lhs, rhs, d_out = _sddmm_operands(graph, op, lt, rt, 4, 128, dtype)
    out, gl, gr = _sddmm_kernel(graph, lhs, rhs, d_out, op, lt, rt, variant, backward_variant)
    ref_out, ref_gl, ref_gr = _sddmm_reference(graph, lhs, rhs, d_out, op, lt, rt)
    assert out.shape == ref_out.shape and gl.shape == lhs.shape and gr.shape == rhs.shape
    _close(out, ref_out, dtype, "out")
    _close(gl, ref_gl, dtype, "d_lhs")
    _close(gr, ref_gr, dtype, "d_rhs")


@pytest.mark.parametrize("backward_variant", ["node", "edge"])
@pytest.mark.parametrize("variant", ["node", "edge"])
@pytest.mark.parametrize("lt", ["src", "dst"])
def test_gsddmm_multihead_copy(graph, lt, variant, backward_variant):
    lhs, rhs, d_out = _sddmm_operands(graph, "copy", lt, "edge", 4, 64, torch.float32)
    out, gl, _ = _sddmm_kernel(graph, lhs, rhs, d_out, "copy", lt, "edge", variant, backward_variant)
    ref_out, ref_gl, _ = _sddmm_reference(graph, lhs, rhs, d_out, "copy", lt, "edge")
    _close(out, ref_out, torch.float32, "out")
    _close(gl, ref_gl, torch.float32, "d_lhs")


@pytest.mark.parametrize("variant,backward_variant", [("node", "node"), ("edge", "edge")])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("heads,dim", HEAD_SHAPES)
@pytest.mark.parametrize("op,lt,rt", [("mul", "src", "dst"), ("dot", "src", "edge"), ("div", "edge", "dst")])
def test_gsddmm_multihead_shapes_and_dtypes(graph, op, lt, rt, heads, dim, dtype, variant, backward_variant):
    lhs, rhs, d_out = _sddmm_operands(graph, op, lt, rt, heads, dim, dtype)
    out, gl, gr = _sddmm_kernel(graph, lhs, rhs, d_out, op, lt, rt, variant, backward_variant)
    ref_out, ref_gl, ref_gr = _sddmm_reference(graph, lhs, rhs, d_out, op, lt, rt)
    _close(out, ref_out, dtype, "out")
    _close(gl, ref_gl, dtype, "d_lhs")
    _close(gr, ref_gr, dtype, "d_rhs")


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("lt,rt", [("src", "dst"), ("edge", "src"), ("dst", "edge")])
@pytest.mark.parametrize("variant,backward_variant", [("node", "node"), ("edge", "edge")])
@pytest.mark.parametrize("op", GSDDMM_OPS + ["copy"])
def test_gsddmm_multihead_equals_per_head_calls(graph, op, lt, rt, variant, backward_variant, dtype):
    """One 3-D launch computes what one 2-D call per head computes: the forward and
    the deterministic node-parallel backward bit for bit, the atomic edge-parallel
    backward up to its fp32 summation order."""
    if op == "copy" and lt == "edge":
        pytest.skip("copy reads a node operand")
    heads, dim = 4, 128
    lhs, rhs, d_out = _sddmm_operands(graph, op, lt, rt, heads, dim, dtype)
    fused, gl, gr = _sddmm_kernel(graph, lhs, rhs, d_out, op, lt, rt, variant, backward_variant)
    per_head = [
        _sddmm_kernel(graph, lhs[:, h].contiguous(), rhs[:, h].contiguous(), d_out[:, h].contiguous(), op, lt, rt, variant, backward_variant)
        for h in range(heads)
    ]
    loop = torch.stack([p[0] for p in per_head], 1)
    loop_gl = torch.stack([p[1] for p in per_head], 1)
    assert torch.equal(fused, loop)
    if backward_variant == "node":
        assert torch.equal(gl, loop_gl)
    else:
        torch.testing.assert_close(gl, loop_gl, rtol=1e-4, atol=1e-4) if dtype == torch.float32 else _close(gl, loop_gl.float(), dtype, "d_lhs")
    if op != "copy":
        loop_gr = torch.stack([p[2] for p in per_head], 1)
        if backward_variant == "node":
            assert torch.equal(gr, loop_gr)
        else:
            torch.testing.assert_close(gr, loop_gr, rtol=1e-4, atol=1e-4) if dtype == torch.float32 else _close(gr, loop_gr.float(), dtype, "d_rhs")


def test_gsddmm_multihead_heavy_chunks_and_overlap(graph):
    """Heavy-node chunking and the overlapped light bucket with 3-D operands."""
    lhs, rhs, d_out = _sddmm_operands(graph, "mul", "src", "dst", 4, 64, torch.float32)
    ref_out, _, _ = _sddmm_reference(graph, lhs, rhs, d_out, "mul", "src", "dst")
    for kwargs in ({"heavy_edges_per_block": 64}, {"overlap_buckets": True}, {"pipeline_stages": 2}):
        out = gsddmm(graph, lhs, rhs, op="mul", variant="node", **kwargs)
        _close(out, ref_out, torch.float32, f"out {kwargs}")
    for stages in (1, 2):
        out = gsddmm(graph, lhs, rhs, op="mul", variant="edge", pipeline_stages=stages)
        _close(out, ref_out, torch.float32, f"edge out stages={stages}")


def test_gsddmm_shape_errors_are_reported_as_shape_errors(graph):
    n = graph.forward_indptr.numel() - 1
    x = torch.randn(n, 4, 100, device=DEVICE)
    with pytest.raises(ValueError, match="per-head feature dim D=100"):
        gsddmm(graph, x, x, op="add")
    with pytest.raises(ValueError, match="share the per-row layout"):
        gsddmm(graph, torch.randn(n, 4, 64, device=DEVICE), torch.randn(n, 2, 64, device=DEVICE), op="add")
    with pytest.raises(ValueError, match="pass lhs/rhs as \\[rows, heads, D\\]"):
        gsddmm(graph, torch.randn(n, 512, device=DEVICE), torch.randn(n, 512, device=DEVICE), op="add")


# =============================================================================
# gspmm
# =============================================================================


def _spmm_reference(graph, lhs, rhs, op, heads, dim):
    """out[v] = sum_{e=(u->v)} op(lhs[u], rhs[e]) in fp32; rhs in CSR edge order."""
    src, dst = _edge_endpoints(graph, by_src=False)
    n = graph.forward_indptr.numel() - 1
    l32 = None if lhs is None else lhs.detach().float().requires_grad_(True)
    r32 = None if rhs is None else rhs.detach().float().requires_grad_(True)
    u = None if l32 is None else l32.index_select(0, src)  # [E, H, D]
    if r32 is not None:
        e = r32
        if e.ndim == 1:
            e = e[:, None, None]
        elif e.ndim == 2 and u is not None:
            e = e[:, :, None]  # [E, H] -> per head
        elif e.ndim == 2:
            pass  # copy_e on [E, H]
    msg = {"copy_u": lambda: u, "copy_e": lambda: e, "add": lambda: u + e, "sub": lambda: u - e, "mul": lambda: u * e, "div": lambda: u / e}[op]()
    out = torch.zeros((n, *msg.shape[1:]), device=DEVICE).index_add(0, dst, msg)
    return out, l32, r32


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("edge_layout", ["per_head", "per_head_1", "elementwise", "scalar"])
@pytest.mark.parametrize("op", ["mul", "add", "sub", "div"])
@pytest.mark.parametrize("heads,dim", [(4, 128), (2, 64), (8, 32)])
def test_gspmm_multihead_sum_matches_reference(graph, op, edge_layout, heads, dim, dtype):
    gen = torch.Generator(device=DEVICE).manual_seed(1)
    n, e = graph.forward_indptr.numel() - 1, graph.forward_indices.numel()
    lhs = torch.randn(n, heads, dim, device=DEVICE, dtype=dtype, generator=gen)
    shape = {"per_head": (e, heads), "per_head_1": (e, heads, 1), "elementwise": (e, heads, dim), "scalar": (e,)}[edge_layout]
    rhs = torch.rand(shape, device=DEVICE, dtype=dtype, generator=gen) + 0.5
    l = lhs.clone().requires_grad_(True)
    r = rhs.clone().requires_grad_(True)
    out = gspmm(graph, l, r, op=op, reduce="sum")
    ref, l32, r32 = _spmm_reference(graph, lhs, rhs.reshape(e, heads) if edge_layout == "per_head_1" else rhs, op, heads, dim)
    assert out.shape == (n, heads, dim)
    g = torch.randn(out.shape, device=DEVICE, dtype=dtype, generator=gen)
    out.backward(g)
    ref.backward(g.float())
    tol = {"rtol": 3e-2, "atol": 3e-1} if dtype == torch.bfloat16 else {"rtol": 1e-4, "atol": 1e-3}
    torch.testing.assert_close(out.float(), ref.detach(), **tol)
    torch.testing.assert_close(l.grad.float(), l32.grad, **tol)
    ref_gr = r32.grad.reshape(rhs.shape)
    assert r.grad.shape == rhs.shape
    torch.testing.assert_close(r.grad.float(), ref_gr, **tol)


@pytest.mark.parametrize("heads,dim", [(4, 128), (2, 64)])
def test_gspmm_multihead_copy_u(graph, heads, dim):
    n = graph.forward_indptr.numel() - 1
    lhs = torch.randn(n, heads, dim, device=DEVICE, requires_grad=True)
    out = gspmm(graph, lhs, None, op="copy_u", reduce="sum")
    ref, l32, _ = _spmm_reference(graph, lhs, None, "copy_u", heads, dim)
    assert out.shape == (n, heads, dim)
    torch.testing.assert_close(out, ref.detach(), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("reduce", ["sum", "max"])
def test_gspmm_copy_e_per_head_scores(graph, reduce):
    """Edge-softmax's reductions: [E, H] scores reduced per destination -> [N, H]."""
    n, e = graph.forward_indptr.numel() - 1, graph.forward_indices.numel()
    s = torch.randn(e, 4, device=DEVICE)
    out = gspmm(graph, None, s, op="copy_e", reduce=reduce)
    _, dst = _edge_endpoints(graph, by_src=False)
    if reduce == "sum":
        ref = torch.zeros(n, 4, device=DEVICE).index_add(0, dst, s)
    else:
        ref = torch.full((n, 4), float("-inf"), device=DEVICE).scatter_reduce(0, dst[:, None].expand(-1, 4), s, "amax")
        ref = torch.where(torch.isinf(ref), torch.zeros_like(ref), ref)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_gspmm_per_head_min_max_is_rejected(graph):
    n, e = graph.forward_indptr.numel() - 1, graph.forward_indices.numel()
    with pytest.raises(RuntimeError, match="reduce='sum' only"):
        gspmm(graph, torch.randn(n, 4, 64, device=DEVICE), torch.randn(e, 4, device=DEVICE), op="mul", reduce="max")


def test_gspmm_bad_multihead_edge_shape(graph):
    n, e = graph.forward_indptr.numel() - 1, graph.forward_indices.numel()
    with pytest.raises(ValueError, match="needs edge data"):
        gspmm(graph, torch.randn(n, 4, 64, device=DEVICE), torch.randn(e, 3, device=DEVICE), op="mul", reduce="sum")
