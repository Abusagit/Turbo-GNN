"""g-SDDMM for any feature dim D in [1, 65536], against a float64 autograd reference.

The kernels lay a row across lane groups of 4-32 lanes in 16-byte vectors when every row read is
16-byte aligned and a whole number of them, and element by element otherwise; rows of more than 32
vectors take whole warps, in passes of 64 vectors. The dims below cover each group size in both
layouts, rows that end mid-pass, and rows of several passes, for the forward kernels (node- and
edge-parallel) and both backward kernels.

Operand layouts the kernels accept as views -- column slices of wider tensors (row stride > D),
views offset by one element (unaligned), expanded rows (row stride 0) -- and graphs with uint32 /
uint64 indices, heavy rows (32-warp blocks, chunked), short heavy rows (the backward's solo path)
and isolated nodes are covered too.
"""

from __future__ import annotations

import pytest
import torch

from turbo_gnn import _C, gsddmm, gsddmm_edge
from turbo_gnn._gsddmm import _edge_endpoints
from turbo_gnn.graph import AdjacencyForwardBackwardWithNodeBuckets as Adjacency

if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)

DEVICE = torch.device("cuda")
DTYPES = [torch.float32, torch.float16, torch.bfloat16]
OPS = ["add", "sub", "mul", "div", "dot"]
MEMBER_PAIRS = [("src", "dst"), ("dst", "src"), ("src", "edge"), ("edge", "src"), ("dst", "edge"), ("edge", "dst")]

#: Max error relative to max |reference|. Low-precision kernels form products in the storage type
#: (as the reference does not) and round every output once.
TOL = {torch.float32: 2e-5, torch.float16: 4e-3, torch.bfloat16: 3e-2}

#: Per dtype: G = 4 (1 element / 1 vector), 8, 16, 32 lanes, the two-vector shape, rows ending mid
#: pass, and several passes; each list has dims that are whole 16-byte vectors and dims that are not.
DIMS = {
    torch.float32: [1, 3, 8, 12, 33, 64, 100, 128, 255, 256, 260, 513, 1000],
    torch.float16: [1, 2, 7, 16, 24, 48, 64, 100, 136, 257, 512, 520, 1032],
    torch.bfloat16: [5, 8, 31, 56, 128, 200, 264, 1024],
}


def _graph(num_nodes=300, num_edges=2500, seed=0, symmetric=False, quantile=0.9, index_dtype=torch.int32, hub=True):
    """Random multigraph with duplicates and isolated nodes; `hub` adds a node of high in- and
    out-degree so the heavy buckets hold a real 32-warp row."""
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    src = torch.randint(0, num_nodes - 10, (num_edges,), device=DEVICE, generator=gen)
    dst = torch.randint(0, num_nodes - 10, (num_edges,), device=DEVICE, generator=gen)
    if hub:
        spokes = torch.randint(0, num_nodes - 10, (400,), device=DEVICE, generator=gen)
        src = torch.cat([src, spokes, torch.full_like(spokes, 3)])
        dst = torch.cat([dst, torch.full_like(spokes, 3), spokes])
    if symmetric:
        src, dst = torch.cat([src, dst]), torch.cat([dst, src])
    graph = Adjacency.from_edge_list(
        torch.stack([src, dst]), num_nodes, quantile=quantile, index_dtype=index_dtype, is_directed=not symmetric
    ).to(DEVICE)
    return graph


@pytest.fixture(scope="module")
def directed():
    return _graph()


@pytest.fixture(scope="module")
def undirected():
    graph = _graph(symmetric=True, seed=1)
    assert not graph.is_directed
    return graph


def _rows(graph, target):
    return graph.forward_indices.numel() if target == "edge" else graph.forward_indptr.numel() - 1


def _operands(graph, op, lt, rt, dim, dtype, seed=0):
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    lhs = torch.randn(_rows(graph, lt), dim, device=DEVICE, generator=gen).to(dtype)
    rhs = torch.randn(_rows(graph, rt), dim, device=DEVICE, generator=gen)
    if op == "div":
        # Divisors away from zero: div's gradient carries 1/R^2. torch.where rather than sign(),
        # which maps a draw of exactly 0.0 -- it happens, one in a few million -- to 0.
        rhs = torch.where(rhs < 0, rhs - 0.5, rhs + 0.5)
    rhs = rhs.to(dtype)
    e = graph.forward_indices.numel()
    grad = torch.randn((e,) if op == "dot" else (e, dim), device=DEVICE, generator=gen).to(dtype)
    return lhs, rhs, grad


def _reference(graph, lhs, rhs, grad, op, lt, rt):
    """float64 forward and gradients, in forward-CSR edge order."""
    src, dst = _edge_endpoints(graph, by_src=False)
    l64 = lhs.detach().double().requires_grad_(True)
    r64 = rhs.detach().double().requires_grad_(True) if rhs is not None else None

    def pick(t, target):
        return t[src] if target == "src" else t[dst] if target == "dst" else t

    a = pick(l64, lt)
    if op == "copy":
        out = a
    else:
        b = pick(r64, rt)
        out = {"add": a + b, "sub": a - b, "mul": a * b, "div": a / b, "dot": (a * b).sum(-1)}[op]
    out.backward(grad.double())
    return out.detach(), l64.grad, (r64.grad if r64 is not None else None)


def _run(graph, lhs, rhs, grad, op, lt, rt, **kw):
    # detach(), not clone(): a clone of a strided or offset view is a dense copy, and the layout of
    # the caller's tensors is what some cases are about.
    lhs = lhs.detach().requires_grad_(True)
    rhs = rhs.detach().requires_grad_(True) if rhs is not None else None
    out = gsddmm(graph, lhs, rhs, op=op, lhs_target=lt, rhs_target=rt, **kw)
    out.backward(grad)
    return out.detach(), lhs.grad, (rhs.grad if rhs is not None else None)


def _assert_matches(got, ref, dtype, what):
    for name, g, r in zip(("out", "d_lhs", "d_rhs"), got, ref):
        if r is None:
            assert g is None, f"{what} {name}: unexpected gradient"
            continue
        assert g.shape == r.shape, f"{what} {name}: shape {tuple(g.shape)} != {tuple(r.shape)}"
        assert torch.isfinite(g).all(), f"{what} {name}: non-finite values"
        err = (g.double() - r).abs().max().item() / max(r.abs().max().item(), 1e-6)
        assert err <= TOL[dtype], f"{what} {name}: max error {err:.3e} (relative to max |ref|) exceeds {TOL[dtype]:.1e}"


def _check(graph, op, lt, rt, dim, dtype, operands=None, **kw):
    lhs, rhs, grad = operands if operands is not None else _operands(graph, op, lt, rt, dim, dtype)
    if op == "copy":
        rhs = None
    got = _run(graph, lhs, rhs, grad, op, lt, rt, **kw)
    ref = _reference(graph, lhs, rhs, grad, op, lt, rt)
    _assert_matches(got, ref, dtype, f"{op}({lt},{rt}) D={dim} {dtype} {kw}")


# ---------------------------------------------------------------------------
# Dims x dtypes x kernel variants
# ---------------------------------------------------------------------------

DIM_CASES = [(dim, dtype) for dtype in DTYPES for dim in DIMS[dtype]]
VARIANTS = [("node", "node"), ("edge", "edge"), ("node", "edge"), ("edge", "node")]


@pytest.mark.parametrize("variant,backward_variant", VARIANTS, ids=[f"{a}-{b}" for a, b in VARIANTS])
@pytest.mark.parametrize("dim,dtype", DIM_CASES, ids=[f"D{d}-{str(t)[6:]}" for d, t in DIM_CASES])
@pytest.mark.parametrize("op,lt,rt", [("mul", "src", "dst"), ("dot", "edge", "src"), ("div", "dst", "edge")])
def test_dims(directed, op, lt, rt, dim, dtype, variant, backward_variant):
    _check(directed, op, lt, rt, dim, dtype, variant=variant, backward_variant=backward_variant)


@pytest.mark.parametrize("backward_variant", ["node", "edge"])
@pytest.mark.parametrize("variant", ["node", "edge"])
@pytest.mark.parametrize("lt,rt", MEMBER_PAIRS)
@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("dim", [7, 100, 520])
def test_ops_and_members(directed, op, lt, rt, variant, backward_variant, dim):
    _check(directed, op, lt, rt, dim, torch.float32, variant=variant, backward_variant=backward_variant)


@pytest.mark.parametrize("backward_variant", ["node", "edge"])
@pytest.mark.parametrize("variant", ["node", "edge"])
@pytest.mark.parametrize("lt", ["src", "dst"])
@pytest.mark.parametrize("dim,dtype", [(3, torch.float32), (40, torch.float16), (1000, torch.bfloat16)])
def test_copy(directed, lt, dim, dtype, variant, backward_variant):
    _check(directed, "copy", lt, "edge", dim, dtype, variant=variant, backward_variant=backward_variant)


@pytest.mark.parametrize("backward_variant", ["node", "edge"])
@pytest.mark.parametrize("variant", ["node", "edge"])
@pytest.mark.parametrize(
    "op,lt,rt", [("mul", "src", "dst"), ("sub", "src", "edge"), ("dot", "dst", "src"), ("copy", "src", "edge")]
)
@pytest.mark.parametrize("dim", [5, 64, 300])
def test_undirected(undirected, op, lt, rt, dim, variant, backward_variant):
    _check(undirected, op, lt, rt, dim, torch.float16, variant=variant, backward_variant=backward_variant)


# ---------------------------------------------------------------------------
# Operand layouts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant,backward_variant", VARIANTS, ids=[f"{a}-{b}" for a, b in VARIANTS])
@pytest.mark.parametrize("dim,dtype", [(64, torch.float32), (100, torch.float16), (300, torch.float32)])
def test_column_slices(directed, dim, dtype, variant, backward_variant):
    """Row strides larger than D: column slices of wider tensors, read in place."""
    lhs, rhs, grad = _operands(directed, "mul", "src", "edge", dim + 24, dtype)
    lhs, rhs = lhs[:, :dim], rhs[:, 8 : 8 + dim]
    assert lhs.stride(0) == dim + 24
    _check(
        directed,
        "mul",
        "src",
        "edge",
        dim,
        dtype,
        operands=(lhs, rhs, grad[:, :dim].contiguous()),
        variant=variant,
        backward_variant=backward_variant,
    )


@pytest.mark.parametrize("variant,backward_variant", VARIANTS, ids=[f"{a}-{b}" for a, b in VARIANTS])
@pytest.mark.parametrize("op,lt,rt", [("mul", "src", "dst"), ("div", "edge", "dst"), ("dot", "dst", "src")])
@pytest.mark.parametrize("dim", [16, 128, 512])
def test_misaligned_rows(directed, op, lt, rt, dim, variant, backward_variant):
    """Views offset by one element are not 16-byte aligned: the element-wise layout must take them."""
    lhs, rhs, grad = _operands(directed, op, lt, rt, dim + 1, torch.float32)
    lhs, rhs = lhs[:, 1:], rhs[:, 1:]
    assert lhs.data_ptr() % 16 != 0
    grad = grad if op == "dot" else grad[:, 1:].contiguous()
    _check(
        directed,
        op,
        lt,
        rt,
        dim,
        torch.float32,
        operands=(lhs, rhs, grad),
        variant=variant,
        backward_variant=backward_variant,
    )


@pytest.mark.parametrize("variant", ["node", "edge"])
@pytest.mark.parametrize("dim", [32, 100])
def test_expanded_operand_forward(directed, dim, variant):
    """A row stride of 0 -- one row broadcast to every node -- is read in place."""
    lhs = torch.randn(1, dim, device=DEVICE).expand(_rows(directed, "src"), dim)
    rhs = torch.randn(_rows(directed, "dst"), dim, device=DEVICE)
    out = gsddmm(directed, lhs, rhs, op="mul", lhs_target="src", rhs_target="dst", variant=variant)
    ref, _, _ = _reference(directed, lhs, rhs, torch.zeros_like(out), "mul", "src", "dst")
    assert (out.double() - ref).abs().max().item() <= 1e-6 * ref.abs().max().item()


def test_feature_strided_operand_is_copied(directed):
    """A transposed operand (feature stride != 1) is made contiguous by the binding, not rejected."""
    n = _rows(directed, "src")
    lhs = torch.randn(48, n, device=DEVICE).t()
    rhs = torch.randn(n, 48, device=DEVICE)
    assert lhs.stride(1) != 1
    out = gsddmm(directed, lhs, rhs, op="add", lhs_target="src", rhs_target="dst", variant="node")
    ref, _, _ = _reference(directed, lhs, rhs, torch.zeros_like(out), "add", "src", "dst")
    assert (out.double() - ref).abs().max().item() <= 1e-6 * ref.abs().max().item()


# ---------------------------------------------------------------------------
# Graphs and kernel parameters
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64, torch.uint32, torch.uint64])
@pytest.mark.parametrize("variant,backward_variant", VARIANTS, ids=[f"{a}-{b}" for a, b in VARIANTS])
def test_index_dtypes(index_dtype, variant, backward_variant):
    graph = _graph(index_dtype=index_dtype, seed=2)
    _check(graph, "mul", "src", "dst", 40, torch.float32, variant=variant, backward_variant=backward_variant)
    _check(graph, "dot", "src", "edge", 9, torch.float16, variant=variant, backward_variant=backward_variant)


@pytest.mark.parametrize("edges_per_warp,warps_per_block", [(1, 1), (3, 2), (8, 4), (32, 8)])
@pytest.mark.parametrize("dim", [4, 24, 64, 300])
@pytest.mark.parametrize("op,lt,rt", [("mul", "src", "dst"), ("div", "dst", "edge"), ("dot", "src", "edge")])
def test_edge_chunk_shapes(directed, op, lt, rt, dim, edges_per_warp, warps_per_block):
    """Chunks of 1 to 32 edges per warp (edges_per_warp * 32 / G, capped), with partial last steps."""
    _check(
        directed,
        op,
        lt,
        rt,
        dim,
        torch.float32,
        variant="edge",
        backward_variant="edge",
        edges_per_warp=edges_per_warp,
        warps_per_block=warps_per_block,
    )


@pytest.mark.parametrize("dim", [6, 72, 600])
@pytest.mark.parametrize("op,lt,rt", [("mul", "src", "edge"), ("dot", "edge", "src"), ("copy", "src", "edge")])
def test_source_grouped_traversal(directed, op, lt, rt, dim):
    """Pairs without a dst operand traverse the source-grouped list: gsddmm(variant="edge") remaps its
    slots to canonical ids, while gsddmm_edge numbers its edge operand and output by traversal slot."""
    _check(directed, op, lt, rt, dim, torch.float32, variant="edge", backward_variant="edge")
    lhs, rhs, _ = _operands(directed, op, lt, rt, dim, torch.float32)
    rhs = None if op == "copy" else rhs
    out = gsddmm_edge(directed, lhs, rhs, op=op, lhs_target=lt, rhs_target=rt)
    src, _ = _edge_endpoints(directed, by_src=True)
    a = lhs.double()[src] if lt == "src" else lhs.double()
    if op == "copy":
        ref = a
    else:
        b = rhs.double()[src] if rt == "src" else rhs.double()
        ref = a * b if op == "mul" else (a * b).sum(-1)
    assert (out.double() - ref).abs().max().item() <= 2e-5 * ref.abs().max().item()


def test_remap_is_rejected_for_dst_pairs(directed):
    """The remap instantiation exists only for pairs without a dst operand, whose lists can be
    source-grouped; the others must traverse the destination-grouped list, which needs none."""
    lhs = torch.randn(_rows(directed, "src"), 32, device=DEVICE)
    edges = torch.zeros(directed.forward_indices.numel(), 2, dtype=torch.int32, device=DEVICE)
    canon = torch.zeros(directed.forward_indices.numel(), dtype=torch.int32, device=DEVICE)
    with pytest.raises(RuntimeError, match="dst member"):
        _C.gsddmm_forward_edge(lhs, lhs, edges, "mul", "src", "dst", lhs.shape[0], 0, 4, 4, canon)


@pytest.mark.parametrize("heavy_edges_per_block,overlap", [(0, False), (64, False), (64, True), (7, True)])
@pytest.mark.parametrize("dim", [3, 96, 700])
def test_heavy_chunking_and_overlap(directed, dim, heavy_edges_per_block, overlap):
    _check(
        directed,
        "sub",
        "dst",
        "src",
        dim,
        torch.float32,
        variant="node",
        backward_variant="node",
        heavy_edges_per_block=heavy_edges_per_block,
        overlap_buckets=overlap,
    )


@pytest.mark.parametrize("backward_variant", ["node", "edge"])
@pytest.mark.parametrize("dim", [8, 256, 1100])
def test_short_heavy_rows(backward_variant, dim):
    """A low quantile sends rows of a few edges to the 32-warp buckets (the backward's solo path)."""
    graph = _graph(num_nodes=400, num_edges=1200, quantile=0.3, hub=False, seed=3)
    assert graph.heavy_nodes.numel() > 0
    _check(graph, "div", "src", "dst", dim, torch.float32, variant="node", backward_variant=backward_variant)


@pytest.mark.parametrize("backward_variant", ["node", "edge"])
def test_isolated_nodes_get_zero_gradient(backward_variant):
    graph = _graph(num_nodes=300, num_edges=500, hub=False, seed=4)
    lhs, rhs, grad = _operands(graph, "mul", "src", "dst", 37, torch.float32)
    _, d_lhs, d_rhs = _run(
        graph, lhs, rhs, grad, "mul", "src", "dst", variant="node", backward_variant=backward_variant
    )
    indptr_t = graph._to_signed_view(graph.backward_indptr).long()
    no_out = (indptr_t[1:] == indptr_t[:-1]).nonzero().flatten()
    assert no_out.numel() > 0 and (d_lhs[no_out] == 0).all()
    indptr = graph._to_signed_view(graph.forward_indptr).long()
    no_in = (indptr[1:] == indptr[:-1]).nonzero().flatten()
    assert no_in.numel() > 0 and (d_rhs[no_in] == 0).all()


@pytest.mark.parametrize("dim,dtype", [(13, torch.float32), (200, torch.float16), (2000, torch.float32)])
def test_node_backward_is_deterministic(directed, dim, dtype):
    lhs, rhs, grad = _operands(directed, "mul", "src", "dst", dim, dtype)
    runs = [
        _run(directed, lhs, rhs, grad, "mul", "src", "dst", variant="node", backward_variant="node") for _ in range(3)
    ]
    for other in runs[1:]:
        for a, b in zip(runs[0], other):
            assert torch.equal(a, b)


@pytest.mark.parametrize("dim,dtype", [(4096, torch.float32), (8192, torch.float16), (65536, torch.float32)])
def test_huge_dims(dim, dtype):
    graph = _graph(num_nodes=60, num_edges=150, hub=False, seed=5)
    for variant, backward_variant in (("node", "node"), ("edge", "edge")):
        _check(graph, "dot", "src", "dst", dim, dtype, variant=variant, backward_variant=backward_variant)
        _check(graph, "mul", "edge", "dst", dim, dtype, variant=variant, backward_variant=backward_variant)
