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


def test_torch_native_matches_tiny_graph():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(1234)

    N = 3
    feature_dim = 2
    heads = 1
    src = torch.tensor([0, 2], dtype=torch.long, device=device)
    dst = torch.tensor([1, 1], dtype=torch.long, device=device)

    x = torch.tensor([[1.0, 3.0], [-1.0, 0.0], [2.0, 1.0]], device=device)

    from src.backends.registry import BackendRegistry

    ref_backend = BackendRegistry.get_backend("torch_native")

    ref_layer = ref_backend.create_conv(
        "gat_v2",
        feature_dim=feature_dim,
        heads=heads,
        bias=False,
    ).to(device)
    # All projections to Identity:
    _set_identity_(ref_layer.fc_src.weight)
    _set_identity_(ref_layer.fc_dst.weight)
    _set_identity_(ref_layer._outer_proj.weight)
    with torch.no_grad():
        ref_layer.attn.data = torch.ones_like(ref_layer.attn.data)

    edge_index = torch.stack([src, dst], dim=0)
    ref_graph = (edge_index, None, N)
    y_ref = ref_layer(x, ref_graph)

    with torch.no_grad():
        v_0_1 = _leaky_relu(x[0] + x[1])
        v_2_1 = _leaky_relu(x[2] + x[1])
        e_0_1 = v_0_1.sum()
        e_2_1 = v_2_1.sum()

        a_0_1 = torch.exp(e_0_1) / (torch.exp(e_0_1) + torch.exp(e_2_1))
        a_2_1 = torch.exp(e_2_1) / (torch.exp(e_0_1) + torch.exp(e_2_1))

        y_expected = torch.zeros(N, feature_dim, device=device)
        y_expected[1] = a_0_1 * x[0] + a_2_1 * x[2]

    assert y_ref.shape == (N, feature_dim)
    assert torch.allclose(
        y_ref, y_expected, atol=1e-6, rtol=1e-6
    ), f"manual vs torch_native: max|D|={(y_ref - y_expected).abs().max().item():.3e}"
