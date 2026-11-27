"""prefix_grad="inplace" (fused DstSrcProject) must match plain autograd slicing."""

import pytest
import torch

from src.models.layers.prefix_grad import _DstSrcProject, dst_src_project, num_dst_of


@pytest.mark.parametrize("bias", [True, False])
def test_fused_matches_autograd(bias):
    torch.manual_seed(0)
    S, n, F, HD = 37, 11, 8, 12
    fc_dst = torch.nn.Linear(F, HD, bias=bias).double()
    fc_src = torch.nn.Linear(F, HD, bias=bias).double()
    params = [*fc_dst.parameters(), *fc_src.parameters()]

    def loss(l, r, skip):
        return (l.sin() * 1.3).sum() + r.cos().sum() * 0.7 + (skip**2).sum()

    x = torch.randn(S, F, dtype=torch.float64, requires_grad=True)
    out_fused = loss(*dst_src_project(x, fc_dst, fc_src, n))
    g_fused = torch.autograd.grad(out_fused, [x, *params])
    out_plain = loss(fc_dst(x[:n]), fc_src(x), x[:n])
    g_plain = torch.autograd.grad(out_plain, [x, *params])

    torch.testing.assert_close(out_fused, out_plain)
    for a, b in zip(g_fused, g_plain):
        torch.testing.assert_close(a, b)


def test_gradcheck():
    torch.manual_seed(0)
    x = torch.randn(9, 4, dtype=torch.float64, requires_grad=True)
    wd = torch.randn(6, 4, dtype=torch.float64, requires_grad=True)
    ws = torch.randn(6, 4, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x, wd, ws: _DstSrcProject.apply(x, wd, None, ws, None, 3), (x, wd, ws))


def test_unused_skip():
    fc = torch.nn.Linear(4, 6).double()
    x = torch.randn(9, 4, dtype=torch.float64, requires_grad=True)
    l, r, _ = dst_src_project(x, fc, fc, 3)
    (l.sum() + r.sum()).backward()
    assert x.grad.shape == x.shape


def test_num_dst_of():
    class Square:
        num_dst = num_src = 5

    class Block:
        num_dst, num_src = 3, 7

    assert num_dst_of(Square()) is None
    assert num_dst_of(Block()) == 3
    assert num_dst_of(object()) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for AMP autocast")
def test_mixed_precision_none_grad_right():
    """fp32 input under bf16 autocast with grad_right=None must not fail with dtype mismatch."""
    torch.manual_seed(0)
    fc_dst = torch.nn.Linear(8, 12).cuda()
    fc_src = torch.nn.Linear(8, 12).cuda()
    x = torch.randn(20, 8, device="cuda", dtype=torch.float32, requires_grad=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        left, _right, skip = dst_src_project(x, fc_dst, fc_src, 6)
        loss = left.sum() + skip.sum()
    loss.backward()
    assert x.grad is not None and x.grad.shape == x.shape and torch.isfinite(x.grad).all()
