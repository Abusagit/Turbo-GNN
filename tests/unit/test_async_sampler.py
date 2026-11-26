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


@pytest.mark.parametrize("order", ["natural", "degree"])
@pytest.mark.parametrize("graphed", [False, True])
@pytest.mark.parametrize("fanouts", [[5, 10], [3, 7, 4], [25], [-1], [-1, 4], [-1, -1]])
@pytest.mark.parametrize("thresholds", [(None, None), (4, 3)])
def test_blocks_are_valid(fanouts, thresholds, graphed, order):
    indptr, indices, adj = random_graph()
    sampler = AsyncNeighborSampler(
        indptr, indices, fanouts, fwd_heavy_threshold=thresholds[0], bwd_heavy_threshold=thresholds[1],
        fwd_order=order, bwd_order=order,
    )
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
            if order == "degree":  # inside each bucket: degree descending, ties in id order
                for rows in (lt, hv):
                    d = deg[rows]
                    assert bool((d[:-1] >= d[1:]).all())
                    same = d[:-1] == d[1:]
                    assert bool((rows[:-1][same] < rows[1:][same]).all())


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


@pytest.mark.parametrize("thresholds", [(None, None), (4, 3)])
def test_degree_order_only_permutes_rows(thresholds):
    """Degree ordering changes the order rows are listed in, nothing else: same CSR pair and
    the same bucket membership as natural order for the same random draws."""
    indptr, indices, _ = random_graph()
    seeds = torch.randperm(indptr.numel() - 1)[:64].cuda()
    out = {}
    for order in ("natural", "degree"):
        gen = torch.Generator(device="cuda").manual_seed(7)
        s = AsyncNeighborSampler(indptr, indices, [6, 4], fwd_heavy_threshold=thresholds[0],
                                 bwd_heavy_threshold=thresholds[1], generator=gen, fwd_order=order, bwd_order=order)
        out[order] = s.finalize(s.launch(seeds))
    for a, b in zip(out["natural"].blocks, out["degree"].blocks):
        for name in ("indptr", "indices", "bwd_indptr", "bwd_indices"):
            assert torch.equal(getattr(a, name), getattr(b, name)), name
        for name in ("fwd_light", "fwd_heavy", "bwd_light", "bwd_heavy"):
            assert sorted(getattr(a, name).tolist()) == sorted(getattr(b, name).tolist()), name


def test_bad_order_rejected():
    indptr, indices, _ = random_graph(num_nodes=50, num_edges=300)
    with pytest.raises(ValueError):
        AsyncNeighborSampler(indptr, indices, [3], bwd_order="random")


def test_full_neighbor_matches_exact_sampler():
    """fanout -1: the async sampler returns every in-edge, the same blocks as GPUNeighborSampler."""
    from src.sampling.neighbor_sampler import GPUNeighborSampler

    indptr, indices, _ = random_graph()
    seeds = torch.randperm(indptr.numel() - 1)[:64].cuda()
    a = AsyncNeighborSampler(indptr, indices, [-1, -1])
    got = a.finalize(a.launch(seeds))
    ref = GPUNeighborSampler(indptr, indices, [-1, -1]).sample(seeds)
    assert torch.equal(got.input_nodes, ref.input_nodes)
    for g, r in zip(got.blocks, ref.blocks):
        assert g.num_src == r.num_src
        assert torch.equal(g.indptr.long(), r.indptr.long())
        # rows list the same neighbors (in-row order may differ)
        for v in range(r.num_dst):
            s, e = int(r.indptr[v]), int(r.indptr[v + 1])
            assert sorted(g.indices[s:e].tolist()) == sorted(r.indices[s:e].tolist())


def test_async_sampler_edge_cases_and_compact_indices():
    # 1) Compact int32 indices stay int32 on GPU (no 2x upcast).
    indptr, indices, _ = random_graph(num_nodes=200, num_edges=2000)
    idx32 = indices.to(torch.int32)
    s = AsyncNeighborSampler(indptr, idx32, [5, 3])
    assert s.indices.dtype == torch.int32
    assert AsyncNeighborSampler.estimate_padded_edge_capacity(indptr, [5, 3], 64) == s.padded_edge_capacity(64)

    # 2) Empty seeds batch.
    empty = s.finalize(s.launch(torch.empty(0, dtype=torch.int64, device="cuda")))
    assert empty.input_nodes.numel() == 0
    assert all(b.num_dst == 0 and b.num_src == 0 and b.num_edges == 0 for b in empty.blocks)

    # 3) Empty graph (0 edges).
    z_indptr = torch.zeros(51, dtype=torch.int64, device="cuda")
    z_indices = torch.empty(0, dtype=torch.int32, device="cuda")
    s_empty = AsyncNeighborSampler(z_indptr, z_indices, [4, 2])
    b_empty = s_empty.finalize(s_empty.launch(torch.arange(10, device="cuda")))
    assert torch.equal(b_empty.input_nodes, torch.arange(10, device="cuda"))
    assert all(b.num_edges == 0 and b.num_dst == 10 and b.num_src == 10 for b in b_empty.blocks)

    # 4) GraphedNeighborSampler with batch_size > num_nodes (e.g. small graph with large batch_size).
    small_ip, small_ix, _ = random_graph(num_nodes=40, num_edges=120)
    gs = GraphedNeighborSampler(AsyncNeighborSampler(small_ip, small_ix.to(torch.int32), [2]), batch_size=128)
    b_small = gs.finalize(gs.launch(torch.arange(16, device="cuda")))
    assert b_small.blocks[0].num_dst == 16 and b_small.input_nodes.numel() == b_small.blocks[0].num_src
