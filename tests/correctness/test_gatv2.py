import dgl as dgl_mod
import pytest
import torch
from fixtures import device, karate_like_club_graph
from torch import nn


def _set_identity_(W):
    with torch.no_grad():
        W.zero_()
        n = min(W.size(0), W.size(1))
        W[:n, :n].copy_(torch.eye(n, device=W.device))


def _leaky_relu(x, negative_slope=0.2):
    leaky_relu = nn.LeakyReLU(negative_slope)
    return leaky_relu(x)


@pytest.mark.parametrize("heads", [1, 2])
def test_pyg_vs_dgl_graph(karate_like_club_graph, heads):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(1234)

    data = karate_like_club_graph
    device = data["device"]
    x = data["features"]
    edge_index = data["edge_index"].long()
    N = data["num_nodes"]
    feature_dim = x.size(1)

    from src.backends.registry import BackendRegistry

    pyg_backend = BackendRegistry.get_backend("pyg")
    dgl_backend = BackendRegistry.get_backend("dgl")

    pyg_layer = pyg_backend.create_conv(
        "gat_v2",
        feature_dim=feature_dim,
        heads=heads,
        add_self_loops=False,
        dropout=0.0,
        share_weights=True,
        bias=False,
    ).to(device)
    # All projections to Identity:
    _set_identity_(pyg_layer._conv.lin_r.weight)
    _set_identity_(pyg_layer._conv.lin_l.weight)
    _set_identity_(pyg_layer._outer_proj.weight)
    pyg_layer._conv.att.data = torch.ones_like(pyg_layer._conv.att.data)

    dgl_layer = dgl_backend.create_conv(
        "gat_v2",
        feature_dim=feature_dim,
        heads=heads,
        feat_drop=0.0,
        attn_drop=0.0,
        residual=False,
        share_weights=True,
        bias=False,
    ).to(device)
    # All projections to Identity:
    _set_identity_(dgl_layer._conv.fc_src.weight)
    _set_identity_(dgl_layer._conv.fc_dst.weight)
    _set_identity_(dgl_layer._outer_proj.weight)
    with torch.no_grad():
        dgl_layer._conv.attn.data = torch.ones_like(dgl_layer._conv.attn.data)

    g = dgl_mod.graph((edge_index[0], edge_index[1]), num_nodes=N).to(device)
    y_pyg = pyg_layer(x, (edge_index, None))
    y_dgl = dgl_layer(x, g)

    assert y_pyg.shape == y_dgl.shape == (N, feature_dim)
    assert torch.allclose(
        y_pyg, y_dgl, atol=1e-6, rtol=1e-6
    ), f"PyG vs DGL: max|Δ|={(y_pyg - y_dgl).abs().max().item():.3e}"


def test_dgl_matches_tiny_graph():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(1234)

    N = 3
    feature_dim = 2
    heads = 1
    src = torch.tensor([0, 2], dtype=torch.long, device=device)
    dst = torch.tensor([1, 1], dtype=torch.long, device=device)

    x = torch.tensor([[1.0, 3.0], [-1.0, 0.0], [2.0, 1.0]], device=device)

    from src.backends.registry import BackendRegistry

    dgl_backend = BackendRegistry.get_backend("dgl")

    dgl_layer = dgl_backend.create_conv(
        "gat_v2",
        feature_dim=feature_dim,
        heads=heads,
        feat_drop=0.0,
        attn_drop=0.0,
        residual=False,
        share_weights=True,
        bias=False,
    ).to(device)
    # All projections to Identity:
    _set_identity_(dgl_layer._conv.fc_src.weight)
    _set_identity_(dgl_layer._conv.fc_dst.weight)
    _set_identity_(dgl_layer._outer_proj.weight)
    with torch.no_grad():
        dgl_layer._conv.attn.data = torch.ones_like(dgl_layer._conv.attn.data)

    g = dgl_mod.graph((src, dst), num_nodes=N).to(device)
    y_dgl = dgl_layer(x, g)

    with torch.no_grad():
        v_0_1 = _leaky_relu(x[0] + x[1])
        v_2_1 = _leaky_relu(x[2] + x[1])
        e_0_1 = v_0_1.sum()
        e_2_1 = v_2_1.sum()

        a_0_1 = torch.exp(e_0_1) / (torch.exp(e_0_1) + torch.exp(e_2_1))
        a_2_1 = torch.exp(e_2_1) / (torch.exp(e_0_1) + torch.exp(e_2_1))

        y_expected = torch.zeros(N, feature_dim, device=device)
        y_expected[1] = a_0_1 * x[0] + a_2_1 * x[2]

    assert y_dgl.shape == (N, feature_dim)
    assert torch.allclose(
        y_dgl, y_expected, atol=1e-6, rtol=1e-6
    ), f"manual vs DGL: max|Δ|={(y_dgl - y_expected).abs().max().item():.3e}"
