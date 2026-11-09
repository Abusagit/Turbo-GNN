"""AsyncNeighborSampler: valid uniform samples, consistent CSR pair and buckets, and no host syncs."""

import pytest
import torch

from src.sampling.async_sampler import AsyncNeighborSampler, GraphedNeighborSampler

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def random_graph(num_nodes=1500, num_edges=40000, seed=0):
    g = torch.Generator().manual_seed(seed)
    src = torch.randint(0, num_nodes, (num_edges,), generator=g)
    dst = (torch.rand(num_edges, generator=g) ** 3 * num_nodes).long()  # skewed in-degrees
    key = torch.unique(dst * num_nodes + src)
    dst, src = key // num_nodes, key % num_nodes
    indptr = torch.zeros(num_nodes + 1, dtype=torch.int64)
    indptr[1:] = torch.bincount(dst, minlength=num_nodes).cumsum(0)
    adj = [set(src[indptr[v] : indptr[v + 1]].tolist()) for v in range(num_nodes)]
    return indptr.cuda(), src.cuda(), adj


@pytest.mark.parametrize("graphed", [False, True])
@pytest.mark.parametrize("fanouts", [[5, 10], [3, 7, 4], [25]])
@pytest.mark.parametrize("thresholds", [(None, None), (4, 3)])
def test_blocks_are_valid(fanouts, thresholds, graphed):
    indptr, indices, adj = random_graph()
    sampler = AsyncNeighborSampler(indptr, indices, fanouts, fwd_heavy_threshold=thresholds[0], bwd_heavy_threshold=thresholds[1])
    if graphed:  # captured for 100 seeds, run with 64: exercises the padded-seed path
        sampler = GraphedNeighborSampler(sampler, batch_size=100)
        for _ in range(4):  # cycle through every graph instance
            sampler.finalize(sampler.launch(torch.randperm(indptr.numel() - 1)[:64].cuda()))
    seeds = torch.randperm(indptr.numel() - 1)[:64].cuda()
    batch = sampler.finalize(sampler.launch(seeds))

    # Validate on CPU copies: element-wise reads of GPU tensors are one sync each.
    gids = [batch.input_nodes.cpu()]
    seeds = seeds.cpu()
    for blk in batch.blocks[:-1]:
        gids.append(gids[-1][: blk.num_dst])
    assert torch.equal(gids[-1][: batch.blocks[-1].num_dst], seeds)

    for i, (blk, g) in enumerate(zip(batch.blocks, gids)):
        assert g.numel() == blk.num_src and torch.unique(g).numel() == g.numel()
        assert int(g.max()) < indptr.numel() - 1  # no padding id leaked
        if i + 1 < len(batch.blocks):
            assert blk.num_dst == batch.blocks[i + 1].num_src
        ip, ix = blk.indptr.long().cpu(), blk.indices.long().cpu()
        assert int(ip[-1]) == blk.num_edges and (blk.num_edges == 0 or int(ix.max()) < blk.num_src)
        for row in range(blk.num_dst):
            v = int(g[row])
            nbrs = g[ix[ip[row] : ip[row + 1]]].tolist()
            k = min(len(adj[v]), blk.fanout)
            assert len(nbrs) == k and len(set(nbrs)) == k and set(nbrs) <= adj[v]

        # Transposed CSR holds exactly the same (src, dst) pairs, rows sorted by source.
        dst_of = torch.repeat_interleave(torch.arange(blk.num_dst), ip[1:] - ip[:-1])
        fwd_pairs = sorted(zip(ix.tolist(), dst_of.tolist()))
        bp, bx = blk.bwd_indptr.long().cpu(), blk.bwd_indices.long().cpu()
        assert bp.numel() == blk.num_src + 1 and int(bp[-1]) == blk.num_edges
        src_of = torch.repeat_interleave(torch.arange(blk.num_src), bp[1:] - bp[:-1])
        assert sorted(zip(src_of.tolist(), bx.tolist())) == fwd_pairs

        # Buckets partition the rows and respect the thresholds.
        for light, heavy, deg, n, t in (
            (blk.fwd_light, blk.fwd_heavy, ip[1:] - ip[:-1], blk.num_dst, thresholds[0]),
            (blk.bwd_light, blk.bwd_heavy, bp[1:] - bp[:-1], blk.num_src, thresholds[1]),
        ):
            lt, hv = light.long().cpu(), heavy.long().cpu()
            assert sorted(torch.cat([lt, hv]).tolist()) == list(range(n))
            if t is None:
                assert hv.numel() == 0
            else:
                assert bool((deg[hv] >= t).all()) and bool((deg[lt] < t).all())


def test_sampling_is_uniform():
    indptr, indices, adj = random_graph()
    v = int(torch.argmax(indptr[1:] - indptr[:-1]))
    deg = len(adj[v])
    sampler = AsyncNeighborSampler(indptr, indices, [5])
    trials = 3000
    counts = torch.zeros(indptr.numel() - 1, device="cuda")
    seed = torch.tensor([v], device="cuda")
    for _ in range(trials):
        b = sampler.finalize(sampler.launch(seed))
        counts.index_add_(0, b.input_nodes[b.blocks[0].indices.long()], torch.ones(b.blocks[0].num_edges, device="cuda"))
    counts = counts.cpu()
    hit = counts[list(adj[v])]
    expected = trials * 5 / deg
    assert hit.sum().item() == trials * 5
    assert (hit > 0).float().mean() > 0.99
    assert hit.std().item() < 1.5 * expected**0.5


@pytest.mark.parametrize("graphed", [False, True])
def test_launch_and_finalize_never_sync(graphed):
    indptr, indices, _ = random_graph()
    sampler = AsyncNeighborSampler(indptr, indices, [10, 5], fwd_heavy_threshold=4, bwd_heavy_threshold=3)
    if graphed:
        sampler = GraphedNeighborSampler(sampler, batch_size=128)
    seeds = torch.randperm(indptr.numel() - 1)[:128].cuda()
    sampler.finalize(sampler.launch(seeds))  # warm-up: pinned-memory pool, kernels
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        pending = sampler.launch(seeds)
        batch = sampler.finalize(pending)
        # Downstream use of the views must not sync either.
        _ = [b.indptr[1:] - b.indptr[:-1] for b in batch.blocks]
    finally:
        torch.cuda.set_sync_debug_mode(0)


def test_graphed_sampler_leaves_default_generator_alone():
    """Replays must not consume the default CUDA generator's Philox range (dropout's)."""
    indptr, indices, _ = random_graph()
    sampler = GraphedNeighborSampler(AsyncNeighborSampler(indptr, indices, [10, 5]), batch_size=64)
    seeds = torch.randperm(indptr.numel() - 1)[:64].cuda()
    sampler.finalize(sampler.launch(seeds))  # capture
    gen = torch.cuda.default_generators[torch.cuda.current_device()]
    before = gen.get_offset()
    for _ in range(5):
        sampler.finalize(sampler.launch(seeds))
    assert gen.get_offset() == before
    # and the private stream still gives fresh draws per replay
    a = sampler.finalize(sampler.launch(seeds)).blocks[-1].indices.clone()
    b = sampler.finalize(sampler.launch(seeds)).blocks[-1].indices.clone()
    assert not torch.equal(a, b)
