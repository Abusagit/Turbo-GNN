import pytest
import torch

dgl = pytest.importorskip("dgl")

from skewgnn import gspmm  # noqa: E402
from skewgnn.graph import AdjacencyForwardBackwardWithNodeBuckets  # noqa: E402

OPS = ["copy_u", "copy_e", "add", "sub", "mul", "div"]
REDUCERS = ["sum", "min", "max"]

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
]


def _dgl_op_name(op: str, reduce: str) -> str:
    """"mul", "sum" -> "u_mul_e_sum"; copy_* keep their own naming."""
    if op in ("copy_u", "copy_e"):
        return f"{op}_{reduce}"
    return f"u_{op}_e_{reduce}"


@pytest.fixture(scope="module")
def graph_data():
    torch.manual_seed(0)
    device = "cuda"
    num_nodes, num_edges = 400, 3000

    src = torch.randint(0, num_nodes, (num_edges,), device=device)
    dst = torch.randint(0, num_nodes, (num_edges,), device=device)
    loops = torch.arange(num_nodes, device=device)
    src = torch.cat([src, loops])
    dst = torch.cat([dst, loops])
    edge_index = torch.stack([src, dst])

    return edge_index, num_nodes, device


@pytest.fixture(scope="module")
def graphs(graph_data):
    edge_index, num_nodes, device = graph_data

    g = dgl.graph((edge_index[0], edge_index[1]), num_nodes=num_nodes)

    # quantile below 1 so that the heavy-node kernel is actually exercised
    skewgnn = AdjacencyForwardBackwardWithNodeBuckets.from_edge_list(
        edge_index, num_nodes=num_nodes, quantile=0.95, index_dtype=torch.int32
    ).to(device)

    return g, skewgnn


def _make_inputs(op, num_nodes, num_edges, feat_dim, device, edge_broadcast=False):
    """Node/edge operands in COO (DGL) order; None for the operand op ignores."""
    x = None
    if op != "copy_e":
        x = torch.randn(num_nodes, feat_dim, device=device, requires_grad=True)

    e = None
    if op != "copy_u":
        shape = (num_edges,) if edge_broadcast else (num_edges, feat_dim)
        # offset away from zero so that div stays well conditioned
        e = (torch.rand(shape, device=device) + 0.5).requires_grad_(True)

    return x, e


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("reduce", REDUCERS)
@pytest.mark.parametrize("feat_dim", [1, 32, 65])
def test_forward_matches_dgl(graphs, graph_data, op, reduce, feat_dim):
    g, skewgnn = graphs
    _, num_nodes, device = graph_data
    num_edges = g.num_edges()

    x, e = _make_inputs(op, num_nodes, num_edges, feat_dim, device)

    ref_fn = getattr(dgl.ops, _dgl_op_name(op, reduce))
    ref_args = [a for a in (x, e) if a is not None]
    expected = ref_fn(g, *ref_args)

    # skewgnn wants edge data in CSR order, DGL in edge_index order
    e_csr = skewgnn.to_csr_edge_order(e) if e is not None else None
    got = gspmm(skewgnn, x, e_csr, op=op, reduce=reduce)

    assert got.shape == expected.reshape(got.shape).shape
    torch.testing.assert_close(
        got, expected.reshape(got.shape), rtol=1e-4, atol=1e-4, msg=f"forward mismatch for {op}/{reduce}, d={feat_dim}"
    )


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("reduce", REDUCERS)
def test_backward_matches_dgl(graphs, graph_data, op, reduce):
    g, skewgnn = graphs
    _, num_nodes, device = graph_data
    num_edges = g.num_edges()
    feat_dim = 32

    x, e = _make_inputs(op, num_nodes, num_edges, feat_dim, device)
    # separate leaves so the two runs accumulate into different .grad
    x_ref = x.detach().clone().requires_grad_(True) if x is not None else None
    e_ref = e.detach().clone().requires_grad_(True) if e is not None else None

    grad_seed = torch.randn(num_nodes, feat_dim, device=device)

    ref_fn = getattr(dgl.ops, _dgl_op_name(op, reduce))
    ref_args = [a for a in (x_ref, e_ref) if a is not None]
    out_ref = ref_fn(g, *ref_args)
    out_ref.reshape(num_nodes, feat_dim).backward(grad_seed)

    e_csr = skewgnn.to_csr_edge_order(e) if e is not None else None
    if e_csr is not None:
        e_csr.retain_grad()
    out = gspmm(skewgnn, x, e_csr, op=op, reduce=reduce)
    out.reshape(num_nodes, feat_dim).backward(grad_seed)

    if x is not None:
        torch.testing.assert_close(
            x.grad, x_ref.grad, rtol=1e-3, atol=1e-3, msg=f"grad wrt node data mismatch for {op}/{reduce}"
        )

    if e is not None:
        # our edge gradient is in CSR order; permute DGL's the same way
        expected_e_grad = skewgnn.to_csr_edge_order(e_ref.grad)
        torch.testing.assert_close(
            e_csr.grad,
            expected_e_grad,
            rtol=1e-3,
            atol=1e-3,
            msg=f"grad wrt edge data mismatch for {op}/{reduce}",
        )


@pytest.mark.parametrize("op", ["mul", "div", "add"])
@pytest.mark.parametrize("reduce", REDUCERS)
def test_broadcast_edge_data_matches_dgl(graphs, graph_data, op, reduce):
    """Edge data of shape [E] must broadcast over the feature dimension."""
    g, skewgnn = graphs
    _, num_nodes, device = graph_data
    num_edges = g.num_edges()
    feat_dim = 32

    x, e = _make_inputs(op, num_nodes, num_edges, feat_dim, device, edge_broadcast=True)

    ref_fn = getattr(dgl.ops, _dgl_op_name(op, reduce))
    # DGL wants an explicit trailing 1 to broadcast
    expected = ref_fn(g, x, e.unsqueeze(-1))

    got = gspmm(skewgnn, x, skewgnn.to_csr_edge_order(e), op=op, reduce=reduce)

    torch.testing.assert_close(
        got, expected.reshape(got.shape), rtol=1e-4, atol=1e-4, msg=f"broadcast forward mismatch for {op}/{reduce}"
    )


@pytest.mark.parametrize("reduce", REDUCERS)
def test_isolated_node_semantics(graph_data, reduce):
    _, _, device = graph_data
    num_nodes, feat_dim = 8, 4

    # node 0 has no incoming edge
    edge_index = torch.tensor([[1, 2, 3], [1, 2, 3]], device=device)
    skewgnn = AdjacencyForwardBackwardWithNodeBuckets.from_edge_list(
        edge_index, num_nodes=num_nodes, quantile=-1, index_dtype=torch.int32
    ).to(device)

    x = torch.randn(num_nodes, feat_dim, device=device)
    out = gspmm(skewgnn, x, None, op="copy_u", reduce=reduce)

    assert torch.isfinite(out).all(), "identity value leaked into the output"
    torch.testing.assert_close(out[0], torch.zeros(feat_dim, device=device))


def test_edge_order_contract_is_load_bearing(graphs, graph_data):
    g, skewgnn = graphs
    _, num_nodes, device = graph_data
    num_edges = g.num_edges()
    feat_dim = 16

    x, e = _make_inputs("mul", num_nodes, num_edges, feat_dim, device)

    correct = gspmm(skewgnn, x, skewgnn.to_csr_edge_order(e), op="mul", reduce="sum")
    unpermuted = gspmm(skewgnn, x, e, op="mul", reduce="sum")

    assert not torch.allclose(correct, unpermuted, rtol=1e-3, atol=1e-3)


def test_u_add_e_sum_decomposes(graphs, graph_data):
    _, skewgnn = graphs
    _, num_nodes, device = graph_data
    num_edges = skewgnn.forward_indices.numel()
    feat_dim = 32

    x = torch.randn(num_nodes, feat_dim, device=device)
    e = torch.randn(num_edges, feat_dim, device=device)

    fused = gspmm(skewgnn, x, e, op="add", reduce="sum")
    split = gspmm(skewgnn, x, None, op="copy_u", reduce="sum") + gspmm(skewgnn, None, e, op="copy_e", reduce="sum")

    torch.testing.assert_close(fused, split, rtol=1e-4, atol=1e-4)
