import pytest
import torch
from fixtures import device, karate_like_club_graph

from src.backends.registry import BackendRegistry
from src.data.datasets import MODEL_BACKEND_TO_GRAPH_REPR

dgl = pytest.importorskip("dgl")

import src.backends.dgl_backend
import src.backends.dgl_ops_backend

TOL = {
    torch.float32: {"atol": 1e-5, "rtol": 1e-4},
    torch.bfloat16: {"atol": 2e-2, "rtol": 2e-2},
}

SIMPLE_CONVS = ["min_aggr", "max_aggr", "sum_aggr", "mean_aggr", "gcn"]
DTYPES = [torch.float32, torch.bfloat16]

FEATURE_DIM = 32
HEADS = 4


def _to_dgl(edge_index, num_nodes, device):
    return dgl.graph((edge_index[0], edge_index[1]), num_nodes=num_nodes).to(device)


def _aggr(backend, conv_type, **kwargs):
    return BackendRegistry.get_backend(backend).create_aggr(conv_type, **kwargs)


@pytest.fixture
def isolated_node_graph(device):
    edge_index = torch.tensor([[0, 1, 2], [1, 0, 1]], device=device)
    return _to_dgl(edge_index, 4, device)


@pytest.fixture
def karate_dgl_graph(karate_like_club_graph):
    data = karate_like_club_graph
    return _to_dgl(data["edge_index"], data["num_nodes"], data["device"])


@pytest.fixture
def cuda_graph_repr():
    from turbo_gnn import AdjacencyForwardBackwardWithNodeBuckets

    def _create(g):
        return AdjacencyForwardBackwardWithNodeBuckets.from_dgl(g, index_dtype=torch.int32).to(g.device)

    return _create


@pytest.fixture
def cuda_backend():
    try:
        import src.backends.cuda_backend  # noqa: F401

        return BackendRegistry.get_backend("cuda")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"cuda backend unavailable: {exc}")


class TestRegistration:
    def test_backend_is_registered(self):
        assert "dgl_ops" in BackendRegistry.list_backends()

    def test_graph_representation_mapping(self):
        assert MODEL_BACKEND_TO_GRAPH_REPR["dgl_ops"] == "dgl"

    @pytest.mark.parametrize("conv_type", SIMPLE_CONVS + ["gat_v2", "gt"])
    def test_aggr_creation(self, conv_type):
        aggr = _aggr("dgl_ops", conv_type, feature_dim=FEATURE_DIM, heads=HEADS)
        assert callable(aggr)

    def test_create_conv_is_rejected(self):
        with pytest.raises(KeyError):
            BackendRegistry.get_backend("dgl_ops").create_conv("gcn", feature_dim=FEATURE_DIM)


class TestVsDglLayers:
    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("conv_type", SIMPLE_CONVS)
    def test_simple_aggr(self, karate_dgl_graph, conv_type, dtype, device):
        g = karate_dgl_graph
        x = torch.randn(g.num_nodes(), FEATURE_DIM, device=device, dtype=dtype)

        ref = _aggr("dgl", conv_type, feature_dim=FEATURE_DIM).to(device=device, dtype=dtype)(x, g)
        out = _aggr("dgl_ops", conv_type, feature_dim=FEATURE_DIM).to(device=device, dtype=dtype)(x, g)

        torch.testing.assert_close(out, ref, **TOL[dtype])

    @pytest.mark.parametrize("conv_type", ["min_aggr", "max_aggr", "sum_aggr", "mean_aggr", "gcn"])
    def test_isolated_node_stays_zero(self, isolated_node_graph, conv_type, device):
        g = isolated_node_graph
        x = torch.randn(g.num_nodes(), FEATURE_DIM, device=device)

        out = _aggr("dgl_ops", conv_type, feature_dim=FEATURE_DIM).to(device)(x, g)

        assert torch.isfinite(out).all(), "isolated node produced inf/nan"
        assert (out[3] == 0).all(), "value leaked into the isolated node"

    def test_gt(self, karate_dgl_graph, device):
        g = karate_dgl_graph
        n = g.num_nodes()
        qkv = [torch.randn(n, HEADS, FEATURE_DIM, device=device) for _ in range(3)]

        ref = _aggr("dgl", "gt", feature_dim=FEATURE_DIM, heads=HEADS).to(device)(*qkv, g)
        out = _aggr("dgl_ops", "gt", feature_dim=FEATURE_DIM, heads=HEADS).to(device)(*qkv, g)

        torch.testing.assert_close(out, ref, **TOL[torch.float32])

    def test_gat_v2_shape(self, karate_dgl_graph, device):
        g = karate_dgl_graph
        n = g.num_nodes()
        xl = torch.randn(n, HEADS, FEATURE_DIM, device=device)
        xr = torch.randn(n, HEADS, FEATURE_DIM, device=device)

        out = _aggr("dgl_ops", "gat_v2", feature_dim=FEATURE_DIM, heads=HEADS).to(device)(xl, xr, g)

        assert out.shape == (n, HEADS * FEATURE_DIM)


class TestGradients:
    @pytest.mark.parametrize("conv_type", SIMPLE_CONVS)
    def test_simple_aggr_grad(self, karate_dgl_graph, conv_type, device):
        g = karate_dgl_graph
        x = torch.randn(g.num_nodes(), FEATURE_DIM, device=device, requires_grad=True)

        out = _aggr("dgl_ops", conv_type, feature_dim=FEATURE_DIM).to(device)(x, g)
        (grad,) = torch.autograd.grad(out, x, torch.randn_like(out))

        assert torch.isfinite(grad).all()

    @pytest.mark.parametrize("conv_type", ["gat_v2", "gt"])
    def test_attention_aggr_grad(self, karate_dgl_graph, conv_type, device):
        g = karate_dgl_graph
        n = g.num_nodes()
        n_inputs = 3 if conv_type == "gt" else 2
        inputs = [torch.randn(n, HEADS, FEATURE_DIM, device=device, requires_grad=True) for _ in range(n_inputs)]

        out = _aggr("dgl_ops", conv_type, feature_dim=FEATURE_DIM, heads=HEADS).to(device)(*inputs, g)
        grads = torch.autograd.grad(out, inputs, torch.randn_like(out))

        assert all(torch.isfinite(grad).all() for grad in grads)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
class TestVsCudaKernels:
    @pytest.mark.parametrize("conv_type", SIMPLE_CONVS)
    def test_simple_aggr(self, karate_dgl_graph, cuda_graph_repr, cuda_backend, conv_type):
        g = karate_dgl_graph
        x = torch.randn(g.num_nodes(), FEATURE_DIM, device=g.device)

        ref = cuda_backend.create_aggr(conv_type, feature_dim=FEATURE_DIM).to(g.device)(x, cuda_graph_repr(g))
        out = _aggr("dgl_ops", conv_type, feature_dim=FEATURE_DIM).to(g.device)(x, g)

        torch.testing.assert_close(out, ref, **TOL[torch.float32])

    def test_gat_v2(self, karate_dgl_graph, cuda_graph_repr, cuda_backend):
        g = karate_dgl_graph
        n = g.num_nodes()
        xl = torch.randn(n, HEADS, FEATURE_DIM, device=g.device)
        xr = torch.randn(n, HEADS, FEATURE_DIM, device=g.device)

        cuda_aggr = cuda_backend.create_aggr("gat_v2", feature_dim=FEATURE_DIM, heads=HEADS).to(g.device)
        ops_aggr = _aggr("dgl_ops", "gat_v2", feature_dim=FEATURE_DIM, heads=HEADS).to(g.device)
        with torch.no_grad():
            ops_aggr.attn_weights.copy_(cuda_aggr.attn_weights)

        ref = cuda_aggr(xl, xr, cuda_graph_repr(g)).flatten(1)

        torch.testing.assert_close(ops_aggr(xl, xr, g), ref, **TOL[torch.float32])

    def test_gt(self, karate_dgl_graph, cuda_graph_repr, cuda_backend):
        g = karate_dgl_graph
        qkv = [torch.randn(g.num_nodes(), HEADS, FEATURE_DIM, device=g.device) for _ in range(3)]

        ref = (
            cuda_backend.create_aggr("gt", feature_dim=FEATURE_DIM, heads=HEADS)
            .to(g.device)(*qkv, cuda_graph_repr(g))
            .flatten(1)
        )
        out = _aggr("dgl_ops", "gt", feature_dim=FEATURE_DIM, heads=HEADS).to(g.device)(*qkv, g)

        torch.testing.assert_close(out, ref, **TOL[torch.float32])
