"""GATv2 on rectangular (bipartite) graphs: sampled mini-batch blocks.

A block has num_dst destination rows and num_src >= num_dst source nodes; l is [num_dst, H, D]
and r is [num_src, H, D]. Checks the forward and all three gradients (l, r, attention vector)
against a pure-torch scatter reference, for light-only and heavy buckets in both directions,
and that a sampled block gives the same result as the same edges padded to a square graph.
"""

import pytest
import torch
from torch.testing import assert_close

from src.sampling.neighbor_sampler import GPUNeighborSampler
from turbo_gnn.graph import AdjacencyForwardBackwardWithNodeBuckets as G
from turbo_gnn.ops import gatv2_aggr

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

TOL = {
    torch.float32: dict(rtol=1e-4, atol=1e-4),
    torch.float16: dict(rtol=3e-2, atol=3e-2),
    torch.bfloat16: dict(rtol=8e-2, atol=8e-2),
}
# Gradients accumulate over edges and are stored in the input dtype; use the same low-precision
# backward tolerance as test_gatv2_correctness.py (isolated elements exceed the forward one).
GRAD_TOL = {
    torch.float32: TOL[torch.float32],
    torch.float16: dict(rtol=1e-5, atol=2e-1),
    torch.bfloat16: dict(rtol=1e-5, atol=2e-1),
}


def reference_gatv2(indptr, indices, l, r, attn, negative_slope=0.2):
    """out[v] = sum_u softmax_u(attn . leaky(l[v] + r[u])) r[u], over the CSR row of v (float32)."""
    num_dst = indptr.numel() - 1
    counts = (indptr[1:] - indptr[:-1]).long()
    dst = torch.repeat_interleave(torch.arange(num_dst, device=l.device), counts)
    src = indices.long()
    l, r, attn = l.float(), r.float(), attn.float()
    e = (torch.nn.functional.leaky_relu(l[dst] + r[src], negative_slope) * attn).sum(-1)  # [E, H]
    H = l.shape[1]
    emax = torch.full((num_dst, H), -float("inf"), device=l.device).scatter_reduce(
        0, dst[:, None].expand(-1, H), e, reduce="amax", include_self=True
    )
    p = torch.exp(e - emax[dst])
    denom = torch.zeros(num_dst, H, device=l.device).index_add_(0, dst, p)
    alpha = p / denom[dst]
    out = torch.zeros_like(l).index_add_(0, dst, alpha[..., None] * r[src])
    return out


def sampled_block(fanout, num_nodes=3000, avg_deg=40, seeds=200, index_dtype=torch.int32, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    E = num_nodes * avg_deg
    src = torch.randint(0, num_nodes, (E,), device="cuda", generator=g)
    # Skewed destinations so some rows are much heavier than others.
    dst = (torch.rand(E, device="cuda", generator=g) ** 3 * num_nodes).long()
    key = torch.unique(dst * num_nodes + src)
    dst, src = key // num_nodes, key % num_nodes
    indptr = torch.zeros(num_nodes + 1, dtype=torch.int64, device="cuda")
    indptr[1:] = torch.bincount(dst, minlength=num_nodes).cumsum(0)
    sampler = GPUNeighborSampler(indptr, src, [fanout], index_dtype=index_dtype, generator=g)
    batch = sampler.sample(torch.randperm(num_nodes, device="cuda", generator=g)[:seeds])
    return batch.blocks[0]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("fanout", [10, -1])
@pytest.mark.parametrize("heads,head_dim", [(1, 32), (4, 128)])
@pytest.mark.parametrize("buckets", ["light", "heavy"])
def test_gatv2_block_vs_reference(dtype, fanout, heads, head_dim, buckets):
    blk = sampled_block(fanout)
    assert blk.num_src > blk.num_dst
    thresholds = {} if buckets == "light" else dict(forward_heavy_threshold=8, backward_heavy_threshold=4)
    graph = G.from_block_csr(
        blk.indptr, blk.indices, blk.num_src, max_in_degree=fanout if fanout > 0 else None, **thresholds
    )
    assert graph.num_dst == blk.num_dst and graph.num_src == blk.num_src
    if buckets == "heavy":
        assert graph.forward_heavy_nodes.numel() > 0 and graph.backward_heavy_nodes.numel() > 0

    torch.manual_seed(0)
    l = torch.randn(blk.num_dst, heads, head_dim, device="cuda", dtype=dtype, requires_grad=True)
    r = torch.randn(blk.num_src, heads, head_dim, device="cuda", dtype=dtype, requires_grad=True)
    attn = (0.1 * torch.randn(heads, head_dim, device="cuda", dtype=dtype)).requires_grad_()
    grad_out = torch.randn(blk.num_dst, heads, head_dim, device="cuda", dtype=dtype)

    out = gatv2_aggr(graph, l, r, attn)
    assert out.shape == (blk.num_dst, heads, head_dim)
    out.backward(grad_out)

    l_ref, r_ref, a_ref = (t.detach().float().requires_grad_() for t in (l, r, attn))
    ref = reference_gatv2(blk.indptr, blk.indices, l_ref, r_ref, a_ref)
    ref.backward(grad_out.float())

    assert_close(out.float(), ref, **TOL[dtype])
    gtol = GRAD_TOL[dtype]
    assert_close(l.grad.float(), l_ref.grad, **gtol)
    assert_close(r.grad.float(), r_ref.grad, **gtol)
    # grad of the attention vector sums over all edges; compare relative to its scale.
    scale = a_ref.grad.abs().max().clamp_min(1.0)
    assert_close(attn.grad.float() / scale, a_ref.grad / scale, **gtol)


def test_block_matches_padded_square():
    """The rectangular path equals the same edges laid out as a square graph (empty extra rows)."""
    blk = sampled_block(10)
    rect = G.from_block_csr(blk.indptr, blk.indices, blk.num_src)
    pad_rows = blk.num_src - blk.num_dst
    sq_indptr = torch.cat([blk.indptr, blk.indptr[-1:].expand(pad_rows)])
    square = G.from_block_csr(sq_indptr, blk.indices, blk.num_src)
    assert square.num_dst == square.num_src

    torch.manual_seed(1)
    H, D = 2, 64
    r = torch.randn(blk.num_src, H, D, device="cuda", requires_grad=True)
    l_full = torch.randn(blk.num_src, H, D, device="cuda")
    attn = 0.1 * torch.randn(H, D, device="cuda")
    l = l_full[: blk.num_dst].clone().requires_grad_()
    lp = l_full.clone().requires_grad_()
    rp = r.detach().clone().requires_grad_()
    g = torch.randn(blk.num_dst, H, D, device="cuda")

    out = gatv2_aggr(rect, l, r, attn)
    out.backward(g)
    out_p = gatv2_aggr(square, lp, rp, attn)
    out_p[: blk.num_dst].backward(g)

    assert_close(out, out_p[: blk.num_dst])
    assert_close(l.grad, lp.grad[: blk.num_dst])
    assert_close(r.grad, rp.grad)


def test_undirected_backward_rejects_rectangular():
    blk = sampled_block(10)
    graph = G.from_block_csr(blk.indptr, blk.indices, blk.num_src)
    graph.is_directed = False
    l = torch.randn(blk.num_dst, 1, 32, device="cuda", requires_grad=True)
    r = torch.randn(blk.num_src, 1, 32, device="cuda", requires_grad=True)
    out = gatv2_aggr(graph, l, r, 0.1 * torch.randn(1, 32, device="cuda"))
    with pytest.raises(RuntimeError, match="square graph"):
        out.sum().backward()
