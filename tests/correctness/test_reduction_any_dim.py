"""reduction_aggr (min / max / sum neighbour aggregation) for any feature dim.

The kernels take the feature dim D as a runtime value and lay a row out across lanes by shape
(csrc/common/row_shape.cuh): lane groups of 4..32 for short rows, two 16-byte vectors per lane for
longer ones, and rows longer than one pass swept in passes. "Wide" shapes read 16-byte vectors when
every row of X is 16-byte aligned, "narrow" ones single elements otherwise (odd D, offset views).
Heavy rows are cut into edge slices whose partial results are merged in edge order.

Every case is checked against a float64 reference computed from the graph's own CSR -- the values,
the arg of min/max (the first edge in CSR order attaining the extremum; -1 for none) and the
autograd gradient. Min and max select input values, so their outputs and args must match exactly;
sums accumulate in fp32 and are compared relative to the reference's largest magnitude.
"""

from __future__ import annotations

import pytest
import torch

import turbo_gnn._C as _C
from turbo_gnn import reduction_aggr
from turbo_gnn.graph import AdjacencyForwardBackwardWithNodeBuckets as Adjacency

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

DTYPES = [torch.float32, torch.float16, torch.bfloat16]
REDUCES = ["min", "max", "sum"]
# Sums: fp32 accumulation, output rounded to the input dtype. Min/max backward: atomics in the input
# dtype, several outputs adding into the same source.
SUM_TOL = {torch.float32: 1e-5, torch.float16: 1e-3, torch.bfloat16: 8e-3}
SCATTER_TOL = {torch.float32: 1e-6, torch.float16: 2e-3, torch.bfloat16: 1.6e-2}

# Straddles every shape boundary for 4-byte (16-byte vector = 4 elements) and 2-byte (8 elements)
# dtypes: lane groups of 4/8/16/32, two vectors per lane, more than one pass, and sizes that force
# the narrow layout.
DIMS = [1, 2, 3, 4, 5, 8, 12, 16, 17, 31, 32, 33, 48, 64, 100, 128, 129, 256, 257, 264, 512, 520, 1000, 1024, 2056]


# ---------------------------------------------------------------------------
# Graphs and reference
# ---------------------------------------------------------------------------


def _edge_index(n=600, deg=3, hubs=4, hub_deg=300, isolated=3, seed=0):
    """Heavy-tailed random graph: a few hubs land in the heavy bucket (several slices each), the
    rest stay light, and the last ``isolated`` nodes receive no edges."""
    g = torch.Generator().manual_seed(seed)
    m = n - isolated
    src = torch.cat([torch.randint(0, n, (m * deg,), generator=g), torch.randint(0, n, (hubs * hub_deg,), generator=g)])
    dst = torch.cat(
        [torch.randint(0, m, (m * deg,), generator=g), torch.randint(0, hubs, (hubs * hub_deg,), generator=g)]
    )
    key = torch.unique(src * n + dst)
    return torch.stack([key // n, key % n])


def _graph(ei, n, quantile=0.9, index_dtype=torch.int32):
    return Adjacency.from_edge_list(ei, n, quantile=quantile, index_dtype=index_dtype).to("cuda")


def _signed(t):
    if t.dtype == torch.uint32:
        return t.view(torch.int32)
    if t.dtype == torch.uint64:
        return t.view(torch.int64)
    return t


def _csr(graph, direction="forward"):
    indptr = _signed(getattr(graph, f"{direction}_indptr")).long()
    indices = _signed(getattr(graph, f"{direction}_indices")).long()
    rows = torch.repeat_interleave(torch.arange(indptr.numel() - 1, device=indptr.device), indptr[1:] - indptr[:-1])
    return rows, indices


def _reference(graph, x, reduce):
    """float64 reduction over the forward CSR: (out, arg); arg is None for sums. NaN candidates are
    skipped by min/max; a row left without a candidate gives 0 and arg -1."""
    rows, src = _csr(graph)
    n, dim = graph.forward_indptr.numel() - 1, x.shape[1]
    vals = x.double()[src]
    index = rows[:, None].expand_as(vals)
    if reduce == "sum":
        return torch.zeros(n, dim, dtype=torch.float64, device=x.device).index_add(0, rows, vals), None
    ok = ~torch.isnan(vals)
    ident = torch.inf if reduce == "min" else -torch.inf
    masked = torch.where(ok, vals, torch.full_like(vals, ident))
    best = torch.full((n, dim), ident, dtype=torch.float64, device=x.device)
    best = best.scatter_reduce(0, index, masked, reduce="a" + reduce, include_self=True)
    pos = torch.arange(rows.numel(), device=x.device)[:, None].expand_as(vals)
    hit = ok & (vals == best[rows])
    none = rows.numel()
    first = torch.full((n, dim), none, dtype=torch.long, device=x.device)
    first = first.scatter_reduce(
        0, index, torch.where(hit, pos, torch.full_like(pos, none)), reduce="amin", include_self=True
    )
    found = first < none
    arg = torch.where(found, src[first.clamp(max=max(none - 1, 0))], torch.full_like(first, -1))
    out = torch.where(found, best, torch.zeros_like(best))
    return out, arg


def _reference_grad(graph, reduce, arg, grad_out):
    """float64 gradient of the reduction with respect to X."""
    num_src = graph.forward_indptr.numel() - 1
    g = grad_out.double()
    if reduce == "sum":
        rows, src = _csr(graph)
        return torch.zeros(num_src, g.shape[1], dtype=torch.float64, device=g.device).index_add(0, src, g[rows])
    valid = arg >= 0
    cols = torch.arange(g.shape[1], device=g.device).expand_as(arg)
    flat = (arg.clamp(min=0) * g.shape[1] + cols)[valid]
    return (
        torch.zeros(num_src * g.shape[1], dtype=torch.float64, device=g.device)
        .index_add(0, flat, g[valid])
        .view(num_src, -1)
    )


def _inputs(n, dim, dtype, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(n, dim, device="cuda", generator=g).to(dtype)
    grad_out = torch.randn(n, dim, device="cuda", generator=g).to(dtype)
    return x, grad_out


def _run(graph, x, reduce, grad_out=None, **kw):
    # detach(), not clone(): a clone of a strided or offset view is a dense copy, and the layout of
    # the caller's tensor is what some cases are about.
    x = x.detach().requires_grad_(grad_out is not None)
    out = reduction_aggr(graph, x, reduce=reduce, **kw)
    if grad_out is None:
        return out, None
    out.backward(grad_out)
    return out.detach(), x.grad


def _raw_arg(graph, x, reduce, **kw):
    """The arg tensor itself, from the extension (the autograd op does not return it)."""
    table = graph.heavy_edge_slices("forward", kw.get("edges_per_block_heavy_nodes", 128))
    _, arg = _C.reduction_aggr_forward_partitioned(
        graph.forward_indptr,
        graph.forward_indices,
        x,
        graph.light_nodes,
        graph.heavy_nodes,
        graph.max_degree,
        reduce=reduce,
        edges_per_block_heavy_nodes=kw.get("edges_per_block_heavy_nodes", 128),
        chunk_node=table.chunk_node,
        chunk_start=table.chunk_start,
        node_chunk_offset=table.node_chunk_offset,
    )
    return _signed(arg).long()


def _rel_err(got, ref):
    return (got.double() - ref).abs().max().item() / max(ref.abs().max().item(), 1e-6)


def _check(graph, x, reduce, grad_out, what="", **kw):
    """Forward (values and, for min/max, args) and backward against the float64 reference."""
    out, grad = _run(graph, x, reduce, grad_out, **kw)
    ref, ref_arg = _reference(graph, x, reduce)
    assert out.shape == ref.shape and out.dtype == x.dtype, f"{what}: got {tuple(out.shape)} {out.dtype}"
    if reduce == "sum":
        err = _rel_err(out, ref)
        assert err <= SUM_TOL[x.dtype], f"{what}: forward error {err:.2e}"
        grad_ref = _reference_grad(graph, reduce, None, grad_out)
        err = _rel_err(grad, grad_ref)
        assert err <= SUM_TOL[x.dtype], f"{what}: backward error {err:.2e}"
        return
    assert torch.equal(out, ref.to(x.dtype)), (
        f"{what}: forward values differ, max |d| {(out.double() - ref).abs().max():.3e}"
    )
    arg = _raw_arg(graph, x, reduce, **kw)
    mismatch = (arg != ref_arg).sum().item()
    assert mismatch == 0, f"{what}: {mismatch} args differ from the first extremum in edge order"
    grad_ref = _reference_grad(graph, reduce, ref_arg, grad_out)
    err = _rel_err(grad, grad_ref)
    assert err <= SCATTER_TOL[x.dtype], f"{what}: backward error {err:.2e}"


@pytest.fixture(scope="module")
def hub_graph():
    ei = _edge_index()
    graph = _graph(ei, 600)
    assert graph.heavy_nodes.numel() > 0 and graph.max_degree > 128, "test needs multi-slice heavy rows"
    return graph


# ---------------------------------------------------------------------------
# Feature dims
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("reduce", REDUCES)
@pytest.mark.parametrize("dim", DIMS)
def test_matches_reference(hub_graph, dim, reduce, dtype):
    x, grad_out = _inputs(600, dim, dtype)
    _check(hub_graph, x, reduce, grad_out, f"D={dim} {reduce} {dtype}")


@pytest.mark.parametrize(
    ("dim", "dtype"), [(4099, torch.float32), (8192, torch.bfloat16), (65536, torch.float16), (70001, torch.float32)]
)
@pytest.mark.parametrize("reduce", REDUCES)
def test_large_dims(dim, dtype, reduce):
    ei = _edge_index(n=96, deg=3, hubs=1, hub_deg=60, isolated=2)
    graph = _graph(ei, 96)
    x, grad_out = _inputs(96, dim, dtype)
    _check(graph, x, reduce, grad_out, f"D={dim} {reduce} {dtype}", edges_per_block_heavy_nodes=16)


# ---------------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("dim", "dtype"), [(128, torch.float32), (256, torch.float32), (256, torch.float16)])
@pytest.mark.parametrize("reduce", REDUCES)
def test_many_waves(dim, reduce, dtype):
    """Enough rows for many waves of work items, so that whole-warp rows take the occupancy-bounded
    build rather than the unbounded one small graphs get (see kIlpWaves): 120k rows is above eight
    waves of any current GPU."""
    n = 120_000
    g = torch.Generator().manual_seed(9)
    dst = torch.randint(0, n, (3 * n,), generator=g)
    src = torch.randint(0, n, (3 * n,), generator=g)
    key = torch.unique(src * n + dst)
    graph = _graph(torch.stack([key // n, key % n]), n)
    x, grad_out = _inputs(n, dim, dtype)
    _check(graph, x, reduce, grad_out, f"many waves D={dim} {reduce} {dtype}")


@pytest.mark.parametrize("index_dtype", [torch.int64, torch.uint32])
@pytest.mark.parametrize("reduce", REDUCES)
@pytest.mark.parametrize("dim", [5, 64, 300])
def test_index_dtypes(dim, reduce, index_dtype):
    ei = _edge_index()
    graph = _graph(ei, 600, index_dtype=index_dtype)
    x, grad_out = _inputs(600, dim, torch.float32)
    _check(graph, x, reduce, grad_out, f"{index_dtype} D={dim} {reduce}")


@pytest.mark.parametrize("slice_size", [1, 7, 32, 300, 4096])
@pytest.mark.parametrize("reduce", REDUCES)
@pytest.mark.parametrize("dim", [3, 64, 520])
def test_slice_sizes(hub_graph, dim, reduce, slice_size):
    x, grad_out = _inputs(600, dim, torch.float16)
    _check(
        hub_graph, x, reduce, grad_out, f"slice={slice_size} D={dim} {reduce}", edges_per_block_heavy_nodes=slice_size
    )


@pytest.mark.parametrize("quantile", [-1, 0.5, 0.0])
@pytest.mark.parametrize("reduce", REDUCES)
def test_bucketings(quantile, reduce):
    """All rows light, half heavy, all heavy."""
    ei = _edge_index()
    graph = _graph(ei, 600, quantile=quantile)
    x, grad_out = _inputs(600, 40, torch.float32)
    _check(graph, x, reduce, grad_out, f"quantile={quantile} {reduce}")


@pytest.mark.parametrize("warps", [1, 2, 3, 8, 64])
@pytest.mark.parametrize("reduce", REDUCES)
def test_warps_per_block(hub_graph, reduce, warps):
    x, grad_out = _inputs(600, 72, torch.bfloat16)
    _check(hub_graph, x, reduce, grad_out, f"warps={warps} {reduce}", warps_per_block=warps)


@pytest.mark.parametrize(
    "compat", [{"use_2d_kernel": True, "features_per_block": 64, "tiles_y": 4}, {"pipeline_stages": 1}]
)
def test_compat_arguments_are_accepted(hub_graph, compat):
    x, grad_out = _inputs(600, 64, torch.float32)
    _check(hub_graph, x, "max", grad_out, f"{compat}", **compat)


# ---------------------------------------------------------------------------
# Layouts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("reduce", REDUCES)
@pytest.mark.parametrize(
    "layout",
    ["column_slice", "offset_by_one", "transposed", "expanded_rows", "row_stride_unaligned"],
)
def test_layouts(hub_graph, reduce, layout):
    """Views the kernels must read in place (any row stride) or reject into a copy (non-unit
    feature stride); unaligned ones take the narrow layout."""
    n, dim = 600, 64
    base, grad_out = _inputs(n, 3 * dim, torch.float16)
    if layout == "column_slice":
        x = base[:, dim : 2 * dim]  # row stride 3 * dim, 16-byte aligned
    elif layout == "offset_by_one":
        x = base[:, 1 : dim + 1]  # every row misaligned
    elif layout == "transposed":
        x = base[:, :dim].t().contiguous().t()  # feature stride n
    elif layout == "expanded_rows":
        x = base[:1, :dim].expand(n, dim)  # row stride 0
    else:
        x = torch.randn(n, dim + 3, device="cuda").half()[:, :dim]  # row stride dim + 3
    assert not x.is_contiguous()
    _check(hub_graph, x, reduce, grad_out[:, :dim], f"{layout} {reduce}")


@pytest.mark.parametrize("reduce", REDUCES)
def test_expanded_grad_out(hub_graph, reduce):
    """out.sum().backward() hands the backward an expanded, zero-stride gradient."""
    x, _ = _inputs(600, 48, torch.float32)
    x = x.requires_grad_(True)
    reduction_aggr(hub_graph, x, reduce=reduce).sum().backward()
    _, ref_arg = _reference(hub_graph, x.detach(), reduce)
    ref = _reference_grad(hub_graph, reduce, ref_arg, torch.ones(600, 48, device="cuda"))
    assert _rel_err(x.grad, ref) <= SCATTER_TOL[torch.float32] * 10


# ---------------------------------------------------------------------------
# Semantics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("reduce", ["min", "max"])
@pytest.mark.parametrize("dim", [3, 16, 100, 520])
def test_ties_take_first_edge(hub_graph, reduce, dim):
    """Values in {0, 1, 2}: nearly every output is a tie, on light rows, across lane groups and
    across the slices of heavy rows."""
    g = torch.Generator(device="cuda").manual_seed(3)
    x = torch.randint(0, 3, (600, dim), device="cuda", generator=g).float()
    grad_out = torch.randn(600, dim, device="cuda", generator=g)
    for slice_size in (7, 128):
        _check(
            hub_graph, x, reduce, grad_out, f"ties D={dim} slice={slice_size}", edges_per_block_heavy_nodes=slice_size
        )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("reduce", ["min", "max"])
def test_infinities_and_nan(hub_graph, reduce, dtype):
    """+-inf inputs are selectable (all-inf rows give inf, not 0); NaN is never selected, and rows
    whose every candidate is NaN give 0 with no gradient."""
    g = torch.Generator(device="cuda").manual_seed(4)
    x = torch.randn(600, 40, device="cuda", generator=g)
    pick = torch.rand(600, 40, device="cuda", generator=g)
    x[pick < 0.2] = torch.inf
    x[(pick >= 0.2) & (pick < 0.4)] = -torch.inf
    x[(pick >= 0.4) & (pick < 0.5)] = torch.nan
    x[:, :4] = torch.inf if reduce == "min" else -torch.inf  # whole columns at the identity
    x[:, 4:8] = torch.nan  # whole columns without a candidate
    x = x.to(dtype)
    grad_out = torch.randn(600, 40, device="cuda", generator=g).to(dtype)
    _check(hub_graph, x, reduce, grad_out, f"inf/nan {reduce} {dtype}", edges_per_block_heavy_nodes=7)


def test_empty_rows_and_edgeless_graph():
    ei = torch.empty(2, 0, dtype=torch.long)
    graph = _graph(ei, 50, quantile=-1)
    x, grad_out = _inputs(50, 24, torch.float32)
    for reduce in REDUCES:
        out, grad = _run(graph, x, reduce, grad_out)
        assert torch.equal(out, torch.zeros_like(out)) and torch.equal(grad, torch.zeros_like(grad)), reduce


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("reduce", REDUCES)
def test_deterministic(hub_graph, reduce):
    x, grad_out = _inputs(600, 130, torch.float32)
    first = _run(hub_graph, x, reduce, grad_out, edges_per_block_heavy_nodes=16)
    for _ in range(3):
        again = _run(hub_graph, x, reduce, grad_out, edges_per_block_heavy_nodes=16)
        assert torch.equal(first[0], again[0]), "forward is not bitwise reproducible"
        if reduce == "sum":  # min/max backward adds with atomics
            assert torch.equal(first[1], again[1]), "sum backward is not bitwise reproducible"


@pytest.mark.parametrize("reduce", REDUCES)
def test_extension_builds_slice_table(hub_graph, reduce):
    """Called without a slice table, the extension builds the one turbo_gnn caches on the graph."""
    x, _ = _inputs(600, 33, torch.float16)
    kw = {"reduce": reduce, "edges_per_block_heavy_nodes": 50}
    built, built_arg = _C.reduction_aggr_forward_partitioned(
        hub_graph.forward_indptr, hub_graph.forward_indices, x, hub_graph.light_nodes, hub_graph.heavy_nodes, 0, **kw
    )
    cached = reduction_aggr(hub_graph, x, reduce=reduce, edges_per_block_heavy_nodes=50)
    assert torch.equal(built, cached)
    if reduce != "sum":
        assert torch.equal(
            built_arg,
            _C.reduction_aggr_forward_partitioned(
                hub_graph.forward_indptr,
                hub_graph.forward_indices,
                x,
                hub_graph.light_nodes,
                hub_graph.heavy_nodes,
                0,
                **kw,
            )[1],
        )


@pytest.mark.parametrize("reduce", REDUCES)
def test_graph_capture(hub_graph, reduce):
    """Forward and backward captured into a CUDA graph replay what eager computes on new inputs:
    every launch is on the caller's stream and nothing synchronizes (the slice tables are cached on
    the graph by the warm-up)."""
    static_x, static_grad = (t.clone() for t in _inputs(600, 96, torch.float32, seed=5))
    static_x.requires_grad_(True)

    def step():
        static_x.grad = None
        out = reduction_aggr(hub_graph, static_x, reduce=reduce)
        out.backward(static_grad)
        return out

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            step()
    torch.cuda.current_stream().wait_stream(side)
    cuda_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(cuda_graph):
        static_out = step()

    fresh_x, fresh_grad = _inputs(600, 96, torch.float32, seed=6)
    with torch.no_grad():
        static_x.copy_(fresh_x)
        static_grad.copy_(fresh_grad)
    cuda_graph.replay()
    torch.cuda.synchronize()

    eager_out, eager_grad = _run(hub_graph, fresh_x, reduce, fresh_grad)
    assert torch.equal(static_out, eager_out), "forward: replay != eager"
    # min/max scatter their gradient with atomics, in whatever order they land.
    assert _rel_err(static_x.grad, eager_grad.double()) <= 1e-6, "backward: replay != eager"


def test_rejects_bad_arguments(hub_graph):
    x, _ = _inputs(600, 8, torch.float32)
    with pytest.raises(RuntimeError, match="unsupported reduce"):
        reduction_aggr(hub_graph, x, reduce="mean")
    with pytest.raises(RuntimeError, match="warps_per_block"):
        reduction_aggr(hub_graph, x, warps_per_block=0)
    with pytest.raises((RuntimeError, ValueError)):
        reduction_aggr(hub_graph, x, edges_per_block_heavy_nodes=0)
    with pytest.raises(RuntimeError, match="float32, float16 or bfloat16"):
        reduction_aggr(hub_graph, x.double())
