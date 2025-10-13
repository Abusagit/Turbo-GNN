import dgl as dgl_mod
import pytest
import torch
from fixtures import device, karate_like_club_graph


def _set_identity_(W):
    with torch.no_grad():
        W.zero_()
        n = min(W.size(0), W.size(1))
        W[:n, :n].copy_(torch.eye(n, device=W.device))


@pytest.mark.parametrize("heads", [1, 2])
def test_pyg_vs_dgl_graph(karate_like_club_graph, heads):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(1234)

    data = karate_like_club_graph
    device = data["device"]
    x = data["features"]
    edge_index = data["edge_index"].long()
    N = data["num_nodes"]
    Fin = Fout = x.size(1)

    from src.backends.registry import BackendRegistry

    pyg_backend = BackendRegistry.get_backend("pyg")
    dgl_backend = BackendRegistry.get_backend("dgl")

    pyg_layer = pyg_backend.create_conv(
        "gat",
        Fin,
        Fout,
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
        "gat",
        Fin,
        Fout,
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
    dgl_layer._conv.attn.data = torch.ones_like(dgl_layer._conv.attn.data)

    g = dgl_mod.graph((edge_index[0], edge_index[1]), num_nodes=N).to(device)
    y_pyg = pyg_layer(x, (edge_index, None))
    y_dgl = dgl_layer(x, g)

    assert y_pyg.shape == y_dgl.shape == (N, Fout)
    assert torch.allclose(
        y_pyg, y_dgl, atol=1e-6, rtol=1e-6
    ), f"PyG vs DGL: max|Δ|={(y_pyg - y_dgl).abs().max().item():.3e}"
