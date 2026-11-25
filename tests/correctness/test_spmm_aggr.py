"""The cuSPARSE SpMM path (``spmm_aggr`` / ``csr_SPMM_normalized``) against a dense float64 reference.

``out = norm(A) @ x`` with A the CSR adjacency (rows are destinations) and norm scaling edge u -> v by
1, 1 / in_deg(v), 1 / out_deg(u) or 1 / sqrt(out_deg(u) in_deg(v)), times an optional edge weight; its
backward is the product with A^T. Every dtype, normalization and algorithm, for any feature width, plus
the descriptor cache (which must notice new weights, modified graphs and graphs at a reused address)
and stream ordering.
"""

import pytest
import torch

import turbo_gnn
from turbo_gnn.graph import AdjacencyForwardBackwardWithNodeBuckets as Adjacency

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
]

DEVICE = "cuda"
NORMS = ["none", "right", "left", "both"]
DTYPES = [torch.float32, torch.float16, torch.bfloat16]
#: Max error relative to max |reference|: one rounding of the output, and of the edge values, which
#: cuSPARSE takes in the dtype of the features.
TOL = {torch.float32: 1e-5, torch.float16: 2e-3, torch.bfloat16: 1.5e-2}
N, E = 500, 4000


def _graph(seed=0, index_dtype=torch.int32, num_nodes=N, num_edges=E):
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    edge_index = torch.randint(0, num_nodes, (2, num_edges), device=DEVICE, generator=gen)
    return Adjacency.from_edge_list(edge_index, num_nodes, quantile=0.9, index_dtype=index_dtype).to(DEVICE)


def _csr(graph):
    return graph.forward_indptr, graph.forward_indices


def _dense(indptr, indices, norm, weights=None):
    """norm(A) as a dense float64 matrix."""
    ip = indptr.view(torch.int32 if indptr.dtype == torch.uint32 else indptr.dtype).long()
    ix = indices.view(torch.int32 if indices.dtype == torch.uint32 else indices.dtype).long()
    n = ip.numel() - 1
    dst = torch.repeat_interleave(torch.arange(n, device=DEVICE), ip.diff())
    w = torch.ones(ix.numel(), device=DEVICE, dtype=torch.float64) if weights is None else weights.double()
    in_deg = torch.bincount(dst, minlength=n).double().clamp_min(1)
    out_deg = torch.bincount(ix, minlength=n).double().clamp_min(1)
    if norm == "right":
        w = w / in_deg[dst]
    elif norm == "left":
        w = w / out_deg[ix]
    elif norm == "both":
        w = w / (out_deg[ix] * in_deg[dst]).sqrt()
    return torch.zeros(n, n, device=DEVICE, dtype=torch.float64).index_put_((dst, ix), w, accumulate=True)


def _rel(got, want):
    return (got.double() - want).abs().max().item() / max(want.abs().max().item(), 1e-12)


@pytest.fixture(scope="module")
def graph():
    return _graph()


@pytest.mark.parametrize("d", [1, 3, 33, 100, 256])
@pytest.mark.parametrize("norm", NORMS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_forward_and_backward(graph, dtype, norm, d):
    indptr, indices = _csr(graph)
    x = torch.randn(N, d, device=DEVICE).to(dtype).requires_grad_(True)
    out = turbo_gnn.spmm_aggr(x, indptr, indices, norm, -1, 256)
    A = _dense(indptr, indices, norm)
    assert out.dtype == dtype
    assert _rel(out, A @ x.detach().double()) <= TOL[dtype]
    g = torch.randn_like(out)
    out.backward(g)
    assert _rel(x.grad, A.t() @ g.double()) <= TOL[dtype]


@pytest.mark.parametrize("algorithm", [0, 1, 2, 3])
@pytest.mark.parametrize("dtype", DTYPES)
def test_every_algorithm_forward_and_backward(graph, dtype, algorithm):
    """Algorithm 3 has no transposed product in cuSPARSE: the backward of a forward that asked for it
    must still work (it used to end the process)."""
    indptr, indices = _csr(graph)
    x = torch.randn(N, 40, device=DEVICE).to(dtype).requires_grad_(True)
    out = turbo_gnn.spmm_aggr(x, indptr, indices, "both", algorithm, 256)
    A = _dense(indptr, indices, "both")
    assert _rel(out, A @ x.detach().double()) <= TOL[dtype]
    g = torch.randn_like(out)
    out.backward(g)
    assert _rel(x.grad, A.t() @ g.double()) <= TOL[dtype]


@pytest.mark.parametrize("use_cache", [True, False])
@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("dtype", DTYPES)
def test_edge_weights_are_honoured_on_every_call(graph, dtype, transpose, use_cache):
    """New weight values on a cached graph: the cache used to return the first call's weights."""
    indptr, indices = _csr(graph)
    x = torch.randn(N, 24, device=DEVICE).to(dtype)
    for seed in range(3):
        w = torch.rand(indices.numel(), device=DEVICE, generator=torch.Generator(device=DEVICE).manual_seed(seed)) * (
            seed + 1
        )
        out = turbo_gnn.csr_SPMM_normalized(
            indptr, indices, x, w, "left", use_cache=use_cache, do_transpose_a=transpose
        )
        A = _dense(indptr, indices, "left", w)
        assert _rel(out, (A.t() if transpose else A) @ x.double()) <= TOL[dtype], f"call {seed}"


def test_cached_graph_modified_in_place():
    """The cache keys a graph by its tensors; overwriting them with another graph of the same sizes
    must not reuse the first graph's normalization."""
    ga, gb = _csr(_graph(seed=0)), _csr(_graph(seed=1))
    assert ga[0].numel() == gb[0].numel() and ga[1].numel() == gb[1].numel()
    indptr, indices = ga[0].clone(), ga[1].clone()
    x = torch.randn(N, 16, device=DEVICE)
    turbo_gnn.csr_SPMM_normalized(indptr, indices, x, None, "both")
    indptr.copy_(gb[0])
    indices.copy_(gb[1])
    out = turbo_gnn.csr_SPMM_normalized(indptr, indices, x, None, "both")
    assert _rel(out, _dense(*gb, "both") @ x.double()) <= TOL[torch.float32]


def test_new_graph_at_a_cached_address():
    """A different tensor at the address and with the sizes of a cached one -- what the allocator hands
    out after the cached graph is freed -- is a different graph."""
    ga, gb = _csr(_graph(seed=0)), _csr(_graph(seed=1))
    buf_p = torch.empty(2 * ga[0].numel(), dtype=ga[0].dtype, device=DEVICE)
    buf_x = torch.empty(2 * ga[1].numel(), dtype=ga[1].dtype, device=DEVICE)
    x = torch.randn(N, 16, device=DEVICE)
    indptr, indices = buf_p[: ga[0].numel()], buf_x[: ga[1].numel()]
    indptr.copy_(ga[0])
    indices.copy_(ga[1])
    turbo_gnn.csr_SPMM_normalized(indptr, indices, x, None, "both")
    del indptr, indices
    indptr, indices = buf_p[: gb[0].numel()], buf_x[: gb[1].numel()]
    indptr.copy_(gb[0])
    indices.copy_(gb[1])
    out = turbo_gnn.csr_SPMM_normalized(indptr, indices, x, None, "both")
    assert _rel(out, _dense(*gb, "both") @ x.double()) <= TOL[torch.float32]


def test_side_stream_ordering(graph):
    """The CSR written on a side stream, behind a long kernel, then used on that stream: the
    normalization must wait for it (its kernels used to run on the legacy default stream)."""
    indptr0, indices0 = _csr(graph)
    x = torch.randn(N, 32, device=DEVICE)
    want = _dense(indptr0, indices0, "both") @ x.double()
    # The same call once beforehand: creating the cuSPARSE handle and the first allocations of these
    # sizes synchronize the device, which would hide a missing ordering inside the side-stream block.
    turbo_gnn.csr_SPMM_normalized(indptr0, indices0, x, None, "both", use_cache=False)
    indptr, indices = torch.zeros_like(indptr0), torch.zeros_like(indices0)
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        torch.cuda._sleep(100_000_000)
        indptr.copy_(indptr0)
        indices.copy_(indices0)
        out = turbo_gnn.csr_SPMM_normalized(indptr, indices, x, None, "both", use_cache=False)
    torch.cuda.synchronize()
    assert _rel(out, want) <= TOL[torch.float32]


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64, torch.uint32])
def test_index_dtypes(index_dtype):
    g = _graph(seed=2, index_dtype=index_dtype)
    indptr, indices = _csr(g)
    x = torch.randn(N, 20, device=DEVICE, requires_grad=True)
    out = turbo_gnn.spmm_aggr(x, indptr, indices, "right", -1, 256)
    A = _dense(indptr, indices, "right")
    assert _rel(out, A @ x.detach().double()) <= TOL[torch.float32]
    out.backward(torch.ones_like(out))
    assert _rel(x.grad, A.t() @ torch.ones_like(out).double()) <= TOL[torch.float32]


def test_empty_graph_and_width():
    empty = Adjacency.from_edge_list(
        torch.zeros(2, 0, dtype=torch.long, device=DEVICE), 10, quantile=0.9, index_dtype=torch.int32
    ).to(DEVICE)
    out = turbo_gnn.spmm_aggr(torch.randn(10, 8, device=DEVICE), *_csr(empty), "both", -1, 256)
    assert torch.equal(out, torch.zeros_like(out))
    g = _graph(seed=3)
    out = turbo_gnn.spmm_aggr(torch.randn(N, 0, device=DEVICE), *_csr(g), "both", -1, 256)
    assert out.shape == (N, 0)


def test_find_best_algorithm(graph):
    for dtype in DTYPES:
        alg = turbo_gnn._C.find_best_algorithm_normalized(
            *_csr(graph), torch.randn(N, 32, device=DEVICE).to(dtype), torch.empty(0, device=DEVICE), "both"
        )
        assert alg in (0, 1, 2, 3)


def test_invalid_normalization_raises(graph):
    with pytest.raises(RuntimeError, match="normalization"):
        turbo_gnn.spmm_aggr(torch.randn(N, 4, device=DEVICE), *_csr(graph), "sym", -1, 256)


def test_inference_mode_graph(graph):
    """Tensors created in inference mode track no version, so they bypass the cache (reading their
    version counter would raise)."""
    with torch.inference_mode():
        indptr, indices = (t.clone() for t in _csr(graph))
        x = torch.randn(N, 16, device=DEVICE)
        out = turbo_gnn.csr_SPMM_normalized(indptr, indices, x, None, "both")
        out_t = turbo_gnn.csr_SPMM_normalized(indptr, indices, x.half(), None, "both", do_transpose_a=True)
    A = _dense(indptr, indices, "both")
    assert _rel(out, A @ x.double()) <= TOL[torch.float32]
    assert _rel(out_t, A.t() @ x.half().double()) <= TOL[torch.float16]


def test_cache_shared_across_streams(graph):
    """An entry built on one stream and read on another: the reading stream waits for the build."""
    indptr0, indices0 = _csr(graph)
    indptr, indices = indptr0.clone(), indices0.clone()
    x = torch.randn(N, 64, device=DEVICE)
    want = _dense(indptr, indices, "both")
    built_on, read_on = torch.cuda.Stream(), torch.cuda.Stream()
    torch.cuda.synchronize()
    with torch.cuda.stream(built_on):
        torch.cuda._sleep(50_000_000)  # the build's normalization queues behind this
        turbo_gnn.csr_SPMM_normalized(indptr, indices, x, None, "both", do_transpose_a=True)
    with torch.cuda.stream(read_on):
        out = turbo_gnn.csr_SPMM_normalized(indptr, indices, x, None, "both", do_transpose_a=True)
    torch.cuda.synchronize()
    assert _rel(out, want.t() @ x.double()) <= TOL[torch.float32]
