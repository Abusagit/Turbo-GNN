"""Graph transformer attention for any head dim.

The kernels take the head dim D as a runtime value in [1, 65536] and lay a row out across lanes
by shape, picked per launch (csrc/common/row_shape.cuh):

  * fused shapes keep a node's rows in registers -- lane groups of 4..32 for short rows, one or
    two 16-byte vectors per lane for longer ones;
  * the chunked path takes everything longer, in two passes over the edges;
  * "wide" shapes use 16-byte vectors when every row is 16-byte aligned, "narrow" ones single
    elements otherwise (odd D, offset views).

Every case below is checked against a float64 PyTorch reference -- forward and autograd
backward -- computed from the same edge list, so the head dims straddle each boundary between
those paths, and the configurations cover the light/heavy buckets, the edge-sliced heavy path,
the cp.async pipelines, concurrent bucket launches, both backward modes and all index dtypes.

Errors are reported relative to the reference's largest magnitude, so one tolerance per dtype
holds across head dims. Low-precision tolerances cover the kernels' products being rounded to
the input dtype before fp32 accumulation.
"""

from __future__ import annotations

import re

import pytest
import torch

from turbo_gnn import _functions, graph_transformer_aggr
from turbo_gnn.graph import AdjacencyForwardBackwardWithNodeBuckets as Adjacency

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

DTYPES = [torch.float32, torch.float16, torch.bfloat16]
TOL = {torch.float32: 2e-4, torch.float16: 1e-2, torch.bfloat16: 4e-2}

# Straddles every path boundary for both 4-byte (16-byte vector = 4 elements) and 2-byte
# (8 elements) dtypes: lane groups of 4/8/16/32, one and two vectors per lane, the chunked path,
# and odd sizes that force the narrow layout.
HEAD_DIMS = [
    1,
    3,
    4,
    7,
    8,
    12,
    16,
    24,
    32,
    33,
    48,
    64,
    96,
    100,
    128,
    160,
    200,
    256,
    264,
    384,
    512,
    520,
    1000,
    1024,
    2056,
]


# ---------------------------------------------------------------------------
# Graphs and reference
# ---------------------------------------------------------------------------


def _edge_index(n=600, deg=3, hubs=4, hub_deg=300, isolated=3, symmetric=False, seed=0):
    """Heavy-tailed random graph: a few hubs land in the heavy bucket, the rest stay light.

    The last ``isolated`` nodes receive no edges (and, when symmetric, send none), so rows with an
    empty neighbourhood are always exercised.
    """
    g = torch.Generator().manual_seed(seed)
    m = n - isolated
    src = torch.cat([torch.randint(0, m, (m * deg,), generator=g), torch.randint(0, m, (hubs * hub_deg,), generator=g)])
    dst = torch.cat(
        [torch.randint(0, m, (m * deg,), generator=g), torch.randint(0, hubs, (hubs * hub_deg,), generator=g)]
    )
    if symmetric:
        src, dst = torch.cat([src, dst]), torch.cat([dst, src])
    key = torch.unique(src * n + dst)
    return torch.stack([key // n, key % n])


def _graph(ei, n, quantile=0.9, index_dtype=torch.int32, directed=None):
    graph = Adjacency.from_edge_list(ei, n, quantile=quantile, index_dtype=index_dtype).to("cuda")
    if directed is not None:
        graph.is_directed = directed
    return graph


def _qkv(n, heads, dim, dtype, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return tuple(torch.randn(n, heads, dim, device="cuda", generator=g).to(dtype) for _ in range(3))


def _reference(ei, n, q, k, v, scale, grad_out):
    """float64 forward and backward of the attention, straight from the edge list (src -> dst)."""
    src, dst = ei[0].cuda(), ei[1].cuda()
    q64, k64, v64 = (t.detach().double().requires_grad_(True) for t in (q, k, v))
    heads = q.shape[1]
    s = (k64[dst] * q64[src]).sum(-1) * scale  # [E, H]
    m = torch.full((n, heads), -torch.inf, dtype=torch.float64, device="cuda")
    m = m.scatter_reduce(0, dst[:, None].expand(-1, heads), s, reduce="amax", include_self=True)
    p = torch.exp(s - m[dst])
    denom = torch.zeros(n, heads, dtype=torch.float64, device="cuda").index_add(0, dst, p)
    out = torch.zeros(n, heads, q.shape[2], dtype=torch.float64, device="cuda").index_add(
        0, dst, p[..., None] * v64[src]
    )
    out = out / torch.where(denom > 0, denom, torch.ones_like(denom))[..., None]
    out.backward(grad_out.double())
    return out.detach(), q64.grad, k64.grad, v64.grad


def _run(graph, q, k, v, scale, grad_out, **kw):
    # detach(), not clone(): a clone of a strided or offset view is a dense copy, and the layout
    # of the caller's tensors is what some cases are about.
    q, k, v = (t.detach().requires_grad_(True) for t in (q, k, v))
    out = graph_transformer_aggr(graph, q, q, k, v, scale, **kw)
    out.backward(grad_out)
    return out.detach(), q.grad, k.grad, v.grad


def _assert_matches(got, ref, dtype, what):
    for name, g, r in zip(("out", "dQ", "dK", "dV"), got, ref):
        assert g.shape == r.shape, f"{what} {name}: shape {tuple(g.shape)} != {tuple(r.shape)}"
        assert torch.isfinite(g).all(), f"{what} {name}: non-finite values"
        err = (g.double() - r).abs().max().item() / max(r.abs().max().item(), 1e-6)
        assert err <= TOL[dtype], f"{what} {name}: max error {err:.3e} (relative to max |ref|) exceeds {TOL[dtype]:.1e}"


def _check(ei, n, graph, heads, dim, dtype, seed=0, qkv=None, **kw):
    q, k, v = qkv if qkv is not None else _qkv(n, heads, dim, dtype, seed)
    scale = dim**-0.5
    grad_out = torch.randn(
        n, heads, dim, device="cuda", generator=torch.Generator(device="cuda").manual_seed(seed + 1)
    ).to(dtype)
    got = _run(graph, q, k, v, scale, grad_out, **kw)
    ref = _reference(ei, n, q, k, v, scale, grad_out)
    _assert_matches(got, ref, dtype, f"D={dim} {dtype} {kw}")


@pytest.fixture(scope="module")
def directed_graph():
    ei = _edge_index()
    return ei, 600, _graph(ei, 600)


@pytest.fixture(scope="module")
def undirected_graph():
    ei = _edge_index(symmetric=True)
    graph = _graph(ei, 600)
    assert not graph.is_directed, "test needs the undirected backward"
    return ei, 600, graph


# ---------------------------------------------------------------------------
# Head dims
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("dim", HEAD_DIMS)
def test_directed_matches_reference(directed_graph, dim, dtype):
    ei, n, graph = directed_graph
    assert graph.forward_heavy_nodes.numel() > 0 and graph.backward_heavy_nodes.numel() > 0
    _check(ei, n, graph, 2, dim, dtype)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("dim", [1, 7, 8, 32, 64, 100, 128, 256, 512, 1000, 2056])
def test_undirected_matches_reference(undirected_graph, dim, dtype):
    ei, n, graph = undirected_graph
    _check(ei, n, graph, 2, dim, dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("directed", [True, False])
def test_largest_head_dim(dtype, directed):
    """D = 65536: the chunked path with 2048 (fp32) or 1024 (bf16) vector chunks per row."""
    n = 40
    ei = _edge_index(n=n, deg=3, hubs=2, hub_deg=12, isolated=2, symmetric=not directed, seed=3)
    _check(ei, n, _graph(ei, n, quantile=0.8, directed=directed), 1, 65536, dtype)


def test_head_dim_out_of_range():
    n = 16
    ei = _edge_index(n=n, deg=2, hubs=1, hub_deg=4, isolated=1)
    graph = _graph(ei, n)
    q = torch.randn(n, 1, 65537, device="cuda")
    with pytest.raises(RuntimeError, match="head dim"):
        graph_transformer_aggr(graph, q, q, q, q, None)


# ---------------------------------------------------------------------------
# Row layouts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("dim", [8, 64, 256, 1024])
def test_misaligned_rows(directed_graph, dim, dtype):
    """Views offset by one element are not 16-byte aligned: the narrow layout must take them."""
    ei, n, graph = directed_graph
    g = torch.Generator(device="cuda").manual_seed(5)
    base = [torch.randn(n, 2, dim + 1, device="cuda", generator=g).to(dtype) for _ in range(3)]
    q, k, v = (b[..., 1:] for b in base)
    assert q.data_ptr() % 16 != 0
    _check(ei, n, graph, 2, dim, dtype, qkv=(q, k, v))


@pytest.mark.parametrize("dim", [16, 128, 1000])
def test_strided_heads(directed_graph, dim):
    """Q, K, V split from one fused projection: row strides of 3 * H * D, still aligned."""
    ei, n, graph = directed_graph
    qkv = torch.randn(n, 3, 2, dim, device="cuda", generator=torch.Generator(device="cuda").manual_seed(7))
    q, k, v = qkv.unbind(1)
    assert not q.is_contiguous()
    _check(ei, n, graph, 2, dim, torch.float32, qkv=(q, k, v))


# ---------------------------------------------------------------------------
# Launch configurations, per path: G=4 groups, one vector per lane, two, and chunked rows.
# ---------------------------------------------------------------------------

CONFIG_DIMS = [
    (8, torch.float32),
    (128, torch.float32),
    (256, torch.float32),
    (1000, torch.float32),
    (520, torch.bfloat16),
]

CONFIGS = [
    {"forward_light_warps": 1, "forward_heavy_warps": 16, "backward_light_warps": 2, "backward_heavy_warps": 32},
    {"forward_light_warps": 2, "forward_heavy_warps": 32, "backward_light_warps": 4, "backward_heavy_warps": 16},
    {"forward_heavy_edge_slice": 64, "backward_heavy_edge_slice": 64},
    {
        "forward_heavy_edge_slice": 37,
        "backward_heavy_edge_slice": 200,
        "forward_heavy_warps": 32,
        "backward_heavy_warps": 32,
    },
    {
        "pipeline_stages": 2,
        "heavy_pipeline_stages": 6,
        "backward_pipeline_stages": 6,
        "backward_heavy_pipeline_stages": 2,
    },
    {
        "pipeline_stages": 12,
        "heavy_pipeline_stages": 12,
        "backward_pipeline_stages": 12,
        "backward_heavy_pipeline_stages": 12,
        "forward_heavy_warps": 32,
        "backward_heavy_warps": 32,
    },
    {
        "forward_heavy_edge_slice": 64,
        "heavy_pipeline_stages": 6,
        "backward_heavy_edge_slice": 64,
        "backward_heavy_pipeline_stages": 6,
    },
    {"forward_bucket_launch": "concurrent", "backward_bucket_launch": "concurrent"},
]


@pytest.mark.parametrize("config", CONFIGS, ids=lambda c: "-".join(f"{k}={v}" for k, v in c.items()))
@pytest.mark.parametrize(("dim", "dtype"), CONFIG_DIMS, ids=[f"D={d}-{str(t)[6:]}" for d, t in CONFIG_DIMS])
@pytest.mark.parametrize("directed", [True, False], ids=["directed", "undirected"])
def test_configurations(directed_graph, undirected_graph, directed, dim, dtype, config):
    ei, n, graph = directed_graph if directed else undirected_graph
    _check(ei, n, graph, 2, dim, dtype, **config)


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64, torch.uint32, torch.uint64])
@pytest.mark.parametrize("dim", [8, 200, 1000])
@pytest.mark.parametrize("symmetric", [False, True], ids=["directed", "undirected"])
def test_index_dtypes(index_dtype, dim, symmetric):
    ei = _edge_index(n=300, hub_deg=150, symmetric=symmetric, seed=2)
    graph = _graph(ei, 300, index_dtype=index_dtype)
    _check(ei, 300, graph, 3, dim, torch.float32, forward_heavy_edge_slice=32, backward_heavy_edge_slice=32)


# ---------------------------------------------------------------------------
# Rows padded to whole 16-byte vectors (_functions._pad_attention_rows)
# ---------------------------------------------------------------------------

# Odd sizes in every dtype, and sizes that are whole vectors in fp32 only (12, 100), on both sides
# of the single-pass limits.
PADDED_DIMS = [1, 3, 7, 12, 33, 100, 257, 1001]


@pytest.fixture
def always_pad(monkeypatch):
    """Pad every call whose rows are not whole 16-byte vectors (and at least 16 bytes), whatever its
    size. With the default thresholds this module's small graphs stay on the element-wise kernels,
    which the tests above cover."""
    monkeypatch.setattr(_functions, "_PAD_MIN_GATHER_BYTES", 0)
    monkeypatch.setattr(_functions, "_PAD_MIN_GATHER_BYTES_SHORT_ROWS", 0)


def _launched_shapes(fn, family):
    """Row-shape codes (csrc/common/row_shape.cuh) of the forward and backward kernels of `family`
    ("gt" or "gatv2") that fn launches: 1000 and up read rows in 16-byte vectors, below 1000 element
    by element."""
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    names = [
        e.name for e in prof.events() if f"{family}_forward_kernel<" in e.name or f"{family}_backward_kernel<" in e.name
    ]
    assert names, f"no {family} kernels launched"
    return {int(re.search(r"ul, (\d+), \d+, (?:true|false),", name).group(1)) for name in names}


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("dim", PADDED_DIMS)
@pytest.mark.parametrize("directed", [True, False], ids=["directed", "undirected"])
def test_padded_rows_match_reference(directed_graph, undirected_graph, always_pad, directed, dim, dtype):
    ei, n, graph = directed_graph if directed else undirected_graph
    _check(ei, n, graph, 2, dim, dtype)


@pytest.mark.parametrize("dim", [7, 100])
def test_padding_keeps_the_scale_of_the_given_head_dim(directed_graph, always_pad, dim):
    """scale=None means 1/sqrt(D) of the caller's D, not of the padded one."""
    ei, n, graph = directed_graph
    q, k, v = _qkv(n, 2, dim, torch.float16)
    grad_out = torch.randn(n, 2, dim, device="cuda").to(torch.float16)
    got = _run(graph, q, k, v, None, grad_out)
    _assert_matches(got, _reference(ei, n, q, k, v, dim**-0.5, grad_out), torch.float16, f"D={dim} default scale")


def test_padding_copies_offset_views(directed_graph, always_pad):
    """A whole number of vectors but offset by one element: copied, not padded, then vector rows."""
    ei, n, graph = directed_graph
    base = [torch.randn(n, 2, 65, device="cuda") for _ in range(3)]
    q, k, v = (b[..., 1:] for b in base)
    _check(ei, n, graph, 2, 64, torch.float32, qkv=(q, k, v))


@pytest.mark.parametrize(("dim", "dtype"), [(7, torch.float32), (100, torch.float16), (1001, torch.bfloat16)])
def test_padding_takes_vector_rows(directed_graph, monkeypatch, dim, dtype):
    """Padded, every GT kernel reads 16-byte vectors; below the threshold the element-wise ones run."""
    ei, n, graph = directed_graph
    q, k, v = (t.requires_grad_(True) for t in _qkv(n, 2, dim, dtype))

    def step():
        graph_transformer_aggr(graph, q, q, k, v, None).sum().backward()

    monkeypatch.setattr(_functions, "_PAD_MIN_GATHER_BYTES", 0)
    monkeypatch.setattr(_functions, "_PAD_MIN_GATHER_BYTES_SHORT_ROWS", 0)
    assert min(_launched_shapes(step, "gt")) >= 1000
    monkeypatch.setattr(_functions, "_PAD_MIN_GATHER_BYTES", 1 << 62)
    monkeypatch.setattr(_functions, "_PAD_MIN_GATHER_BYTES_SHORT_ROWS", 1 << 62)
    assert max(_launched_shapes(step, "gt")) < 1000


def test_rows_under_16_bytes_stay_element_wise(directed_graph, always_pad):
    """fp16 head dim 7: 14-byte rows cost one sector a gather either way, and padding them measured
    slower on the backward, so even with no size threshold they are not copied."""
    ei, n, graph = directed_graph
    q, k, v = (t.requires_grad_(True) for t in _qkv(n, 2, 7, torch.float16))

    def step():
        graph_transformer_aggr(graph, q, q, k, v, None).sum().backward()

    assert max(_launched_shapes(step, "gt")) < 1000


def _road_edge_index(n=400, max_deg=5, symmetric=False, seed=7):
    """Low-degree graph, every node with 1..max_deg in-edges: a road network's degree profile."""
    g = torch.Generator().manual_seed(seed)
    deg = torch.randint(1, max_deg + 1, (n,), generator=g)
    dst = torch.repeat_interleave(torch.arange(n), deg)
    src = torch.randint(0, n, (int(deg.sum()),), generator=g)
    if symmetric:
        src, dst = torch.cat([src, dst]), torch.cat([dst, src])
    key = torch.unique(src * n + dst)
    return torch.stack([key // n, key % n])


@pytest.mark.parametrize("dim", [4, 32, 100, 128, 256, 1000])
@pytest.mark.parametrize("symmetric", [False, True], ids=["directed", "undirected"])
def test_short_heavy_rows(dim, symmetric):
    """A heavy bucket of rows with a few edges each, as quantile bucketing gives low-degree graphs:
    the heavy blocks hand such rows to one warp and skip the cross-warp merge (solo_row), with
    the default and the widest heavy blocks, and the light path for the rest."""
    ei = _road_edge_index(symmetric=symmetric)
    graph = _graph(ei, 400, quantile=0.5)
    assert graph.forward_heavy_nodes.numel() > 0 and graph.backward_heavy_nodes.numel() > 0
    for kw in ({}, {"forward_heavy_warps": 32, "backward_heavy_warps": 16}):
        _check(ei, 400, graph, 2, dim, torch.float32, **kw)


def test_all_light_and_all_heavy():
    """Empty heavy bucket (quantile -1) and a graph whose every node is heavy (quantile 0)."""
    ei = _edge_index(n=200, hub_deg=100, seed=4)
    for quantile in (-1, 0.0):
        for dim in (16, 300):
            _check(ei, 200, _graph(ei, 200, quantile=quantile), 2, dim, torch.float32)


# ---------------------------------------------------------------------------
# Determinism: the merges are in a fixed order; only the directed dK scatter uses atomics.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dim", [8, 256, 1000])
@pytest.mark.parametrize("symmetric", [False, True], ids=["directed", "undirected"])
def test_bitwise_deterministic(dim, symmetric):
    ei = _edge_index(symmetric=symmetric, seed=6)
    graph = _graph(ei, 600)
    q, k, v = _qkv(600, 2, dim, torch.float32, seed=6)
    grad_out = torch.randn_like(q)
    first = _run(graph, q, k, v, None, grad_out, forward_heavy_warps=16, backward_heavy_warps=16)
    second = _run(graph, q, k, v, None, grad_out, forward_heavy_warps=16, backward_heavy_warps=16)
    names = ("out", "dQ", "dK", "dV")
    for name, a, b in zip(names, first, second):
        if symmetric or name != "dK":
            assert torch.equal(a, b), f"{name} differs between identical runs (D={dim})"
