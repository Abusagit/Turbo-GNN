"""GATv2 attention for any head dim.

The kernels take the head dim D as a runtime value in [1, 65536] and lay a row out across lanes
by shape, picked per launch (csrc/common/row_shape.cuh):

  * fused shapes keep a node's rows in registers -- lane groups of 4..32 for short rows, one or
    two 16-byte vectors per lane for longer ones;
  * the chunked path takes everything longer, in two passes over the edges;
  * "wide" shapes use 16-byte vectors when every row is 16-byte aligned, "narrow" ones single
    elements otherwise (odd D, offset views).

Every case below is checked against a float64 PyTorch reference -- forward and autograd backward,
including the attention vector's gradient -- computed from the same edge list, so the head dims
straddle each boundary between those paths, and the configurations cover the light/heavy buckets,
the edge-sliced heavy paths, the cp.async pipelines, concurrent bucket launches, both backward
graph kinds and all index dtypes. Graph capture and replay checks that every launch is on the
caller's stream.

Errors are reported relative to the reference's largest magnitude, so one tolerance per dtype
holds across head dims. The low-precision tolerances cover the outputs and gradients being stored
in the input dtype, and the backward's <grad, out> term being taken from that stored output.
"""

from __future__ import annotations

import pytest
import torch

from turbo_gnn import gatv2_aggr
from turbo_gnn.graph import AdjacencyForwardBackwardWithNodeBuckets as Adjacency

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

DTYPES = [torch.float32, torch.float16, torch.bfloat16]
TOL = {torch.float32: 2e-4, torch.float16: 1e-2, torch.bfloat16: 4e-2}
SLOPE = 0.2

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


def _inputs(n, heads, dim, dtype, seed=0):
    """x_left, x_right and an attention vector scaled like a trained one, so scores stay O(1)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    left, right = (torch.randn(n, heads, dim, device="cuda", generator=g).to(dtype) for _ in range(2))
    attn = (torch.randn(heads, dim, device="cuda", generator=g) * dim**-0.5).to(dtype)
    return left, right, attn


def _reference(ei, n, left, right, attn, slope, grad_out):
    """float64 forward and backward of GATv2 attention, straight from the edge list (src -> dst)."""
    src, dst = ei[0].cuda(), ei[1].cuda()
    l64, r64, a64 = (t.detach().double().requires_grad_(True) for t in (left, right, attn))
    heads = left.shape[1]
    z = torch.nn.functional.leaky_relu(l64[dst] + r64[src], slope)
    e = (z * a64).sum(-1)  # [E, H]
    m = torch.full((n, heads), -torch.inf, dtype=torch.float64, device="cuda")
    m = m.scatter_reduce(0, dst[:, None].expand(-1, heads), e, reduce="amax", include_self=True)
    p = torch.exp(e - m[dst])
    denom = torch.zeros(n, heads, dtype=torch.float64, device="cuda").index_add(0, dst, p)
    out = torch.zeros(n, heads, left.shape[2], dtype=torch.float64, device="cuda").index_add(
        0, dst, p[..., None] * r64[src]
    )
    out = out / torch.where(denom > 0, denom, torch.ones_like(denom))[..., None]
    out.backward(grad_out.double())
    return out.detach(), l64.grad, r64.grad, a64.grad


def _run(graph, left, right, attn, slope, grad_out, **kw):
    # detach(), not clone(): a clone of a strided or offset view is a dense copy, and the layout
    # of the caller's tensors is what some cases are about.
    left, right, attn = (t.detach().requires_grad_(True) for t in (left, right, attn))
    out = gatv2_aggr(graph, left, right, attn, slope, **kw)
    out.backward(grad_out)
    return out.detach(), left.grad, right.grad, attn.grad


def _assert_matches(got, ref, dtype, what):
    for name, g, r in zip(("out", "dl", "dr", "da"), got, ref):
        assert g.shape == r.shape, f"{what} {name}: shape {tuple(g.shape)} != {tuple(r.shape)}"
        assert torch.isfinite(g).all(), f"{what} {name}: non-finite values"
        err = (g.double() - r).abs().max().item() / max(r.abs().max().item(), 1e-6)
        assert err <= TOL[dtype], f"{what} {name}: max error {err:.3e} (relative to max |ref|) exceeds {TOL[dtype]:.1e}"


def _assert_close_to_rounding(a, b, what):
    """Equal up to fp32 reordering: the attention vector's gradient sums thousands of edge terms
    with atomics, so its rounding error scales with the largest entry, not with each entry."""
    err = (a - b).abs().max().item() / max(b.abs().max().item(), 1e-6)
    assert err <= 1e-5, f"{what}: differs by {err:.2e} of max |{what}|"


def _grad_out(n, heads, dim, dtype, seed):
    g = torch.Generator(device="cuda").manual_seed(seed + 1)
    return torch.randn(n, heads, dim, device="cuda", generator=g).to(dtype)


def _check(ei, n, graph, heads, dim, dtype, seed=0, inputs=None, slope=SLOPE, **kw):
    left, right, attn = inputs if inputs is not None else _inputs(n, heads, dim, dtype, seed)
    grad_out = _grad_out(n, heads, dim, dtype, seed)
    got = _run(graph, left, right, attn, slope, grad_out, **kw)
    ref = _reference(ei, n, left, right, attn, slope, grad_out)
    _assert_matches(got, ref, dtype, f"D={dim} {dtype} slope={slope} {kw}")


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
@pytest.mark.parametrize("dim", [1, 7, 8, 32, 64, 100, 128, 256, 264, 512, 1000, 2056])
def test_undirected_matches_reference(undirected_graph, dim, dtype):
    ei, n, graph = undirected_graph
    _check(ei, n, graph, 2, dim, dtype)


@pytest.mark.parametrize("slope", [0.0, 0.01, 0.5, 1.5, -0.3])
@pytest.mark.parametrize("dim", [16, 128, 1000])
def test_negative_slopes(directed_graph, dim, slope):
    """Any slope, including ones outside [0, 1] where LeakyReLU is not max(s, slope * s).

    Not 1.0: LeakyReLU is then the identity, <a, l[i]> is constant over node i's softmax and the
    true gradient of l is exactly zero, so an error relative to it is meaningless.
    """
    ei, n, graph = directed_graph
    _check(ei, n, graph, 2, dim, torch.float32, slope=slope)


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
    x = torch.randn(n, 1, 65537, device="cuda")
    with pytest.raises(RuntimeError, match="head dim"):
        gatv2_aggr(graph, x, x, x[0], SLOPE)


# ---------------------------------------------------------------------------
# Row layouts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("dim", [8, 64, 256, 1024])
def test_misaligned_rows(directed_graph, dim, dtype):
    """Views offset by one element are not 16-byte aligned: the narrow layout must take them."""
    ei, n, graph = directed_graph
    g = torch.Generator(device="cuda").manual_seed(5)
    base = [torch.randn(n, 2, dim + 1, device="cuda", generator=g).to(dtype) for _ in range(2)]
    left, right = (b[..., 1:] for b in base)
    attn = (torch.randn(2, dim, device="cuda", generator=g) * dim**-0.5).to(dtype)
    assert left.data_ptr() % 16 != 0
    _check(ei, n, graph, 2, dim, dtype, inputs=(left, right, attn))


@pytest.mark.parametrize("dim", [16, 128, 1000])
def test_strided_heads(directed_graph, dim):
    """x_left and x_right split from one fused projection, as in the conv: row strides of 2 * H * D."""
    ei, n, graph = directed_graph
    g = torch.Generator(device="cuda").manual_seed(7)
    lr = torch.randn(n, 2 * 2 * dim, device="cuda", generator=g)
    left, right = (t.view(n, 2, dim) for t in lr.split(2 * dim, -1))
    assert not left.is_contiguous()
    attn = torch.randn(2, dim, device="cuda", generator=g) * dim**-0.5
    _check(ei, n, graph, 2, dim, torch.float32, inputs=(left, right, attn))


def test_non_contiguous_grad_output(directed_graph):
    """The output gradient is read through its strides."""
    ei, n, graph = directed_graph
    left, right, attn = _inputs(n, 2, 64, torch.float32, seed=8)
    grad_full = torch.randn(n, 2, 128, device="cuda", generator=torch.Generator(device="cuda").manual_seed(9))
    grad_out = grad_full[..., ::2]
    assert grad_out.stride(2) == 2
    got = _run(graph, left, right, attn, SLOPE, grad_out)
    ref = _reference(ei, n, left, right, attn, SLOPE, grad_out)
    _assert_matches(got, ref, torch.float32, "strided grad_out")


# ---------------------------------------------------------------------------
# Launch configurations, per path: G=4 groups, one vector per lane, two, and chunked rows.
# ---------------------------------------------------------------------------

CONFIG_DIMS = [
    (8, torch.float32),
    (128, torch.float32),
    (256, torch.float32),
    (1000, torch.float32),
    (512, torch.float16),
    (520, torch.bfloat16),
]

CONFIGS = [
    {"forward_light_warps": 1, "forward_heavy_warps": 16, "backward_light_warps": 2, "backward_heavy_warps": 32},
    {"forward_light_warps": 2, "forward_heavy_warps": 32, "backward_light_warps": 4, "backward_heavy_warps": 16},
    {"forward_light_warps": 4, "backward_light_warps": 4, "forward_heavy_warps": 8, "backward_heavy_warps": 8},
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


def test_all_light_and_all_heavy():
    """Empty heavy bucket (quantile -1) and a graph whose every node is heavy (quantile 0)."""
    for symmetric in (False, True):
        ei = _edge_index(n=200, hub_deg=100, symmetric=symmetric, seed=4)
        for quantile in (-1, 0.0):
            for dim in (16, 300):
                _check(ei, 200, _graph(ei, 200, quantile=quantile), 2, dim, torch.float32)


def test_node_list_width_must_match_csr():
    """Node lists are read through the CSR's index type: another width is an error, not garbage."""
    ei = _edge_index(n=100, hub_deg=40, seed=10)
    graph = _graph(ei, 100)
    graph.forward_light_nodes = graph.forward_light_nodes.long()
    left, right, attn = _inputs(100, 1, 16, torch.float32)
    with pytest.raises(RuntimeError, match="light_nodes"):
        gatv2_aggr(graph, left, right, attn, SLOPE)


# ---------------------------------------------------------------------------
# Determinism: row gradients and the forward are summed in a fixed order. The attention vector's
# gradient is accumulated with fp32 atomics across blocks, so it agrees to rounding only.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dim", [8, 256, 1000])
@pytest.mark.parametrize("symmetric", [False, True], ids=["directed", "undirected"])
def test_bitwise_deterministic(dim, symmetric):
    ei = _edge_index(symmetric=symmetric, seed=6)
    graph = _graph(ei, 600)
    left, right, attn = _inputs(600, 2, dim, torch.float32, seed=6)
    grad_out = torch.randn_like(left)
    kw = {
        "forward_heavy_warps": 16,
        "backward_heavy_warps": 16,
        "forward_heavy_edge_slice": 64,
        "backward_heavy_edge_slice": 64,
    }
    first = _run(graph, left, right, attn, SLOPE, grad_out, **kw)
    second = _run(graph, left, right, attn, SLOPE, grad_out, **kw)
    for name, a, b in zip(("out", "dl", "dr"), first, second):
        assert torch.equal(a, b), f"{name} differs between identical runs (D={dim})"
    _assert_close_to_rounding(first[3], second[3], "da")


# ---------------------------------------------------------------------------
# Streams: forward and backward captured into a CUDA graph and replayed on new inputs must match
# eager execution. A launch on another stream than the caller's -- the legacy default stream, or a
# raw cudaMalloc -- breaks the capture or leaves its kernel out of the replay.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dim", [64, 1000])
@pytest.mark.parametrize("symmetric", [False, True], ids=["directed", "undirected"])
def test_cuda_graph_replay(dim, symmetric):
    n = 500
    ei = _edge_index(n=n, hub_deg=200, symmetric=symmetric, seed=11)
    graph = _graph(ei, n)
    kw = {"forward_heavy_edge_slice": 64, "backward_heavy_edge_slice": 64}
    static = [t.detach().clone().requires_grad_(True) for t in _inputs(n, 2, dim, torch.float32, seed=12)]
    static_grad = torch.randn(n, 2, dim, device="cuda")

    def step():
        for t in static:
            t.grad = None
        out = gatv2_aggr(graph, *static, SLOPE, **kw)
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

    fresh = _inputs(n, 2, dim, torch.float32, seed=13)
    fresh_grad = torch.randn(n, 2, dim, device="cuda", generator=torch.Generator(device="cuda").manual_seed(14))
    with torch.no_grad():
        for t, f in zip(static, fresh):
            t.copy_(f)
        static_grad.copy_(fresh_grad)
    cuda_graph.replay()
    torch.cuda.synchronize()
    replayed = (static_out.detach().clone(), *(t.grad.detach().clone() for t in static))

    eager = _run(graph, *fresh, SLOPE, fresh_grad, **kw)
    for name, a, b in zip(("out", "dl", "dr"), replayed, eager):
        assert torch.equal(a, b), f"{name}: replay != eager"
    _assert_close_to_rounding(replayed[3], eager[3], "da")
