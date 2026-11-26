"""GPUNeighborSampler: block layout and sampling semantics (runs on CPU, or CUDA if available)."""

import pytest
import torch

from src.sampling.neighbor_sampler import GPUNeighborSampler, seed_batches

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def random_graph(num_nodes=1500, num_edges=40000, device="cpu", seed=0):
    g = torch.Generator().manual_seed(seed)
    src = torch.randint(0, num_nodes, (num_edges,), generator=g)
    dst = (torch.rand(num_edges, generator=g) ** 3 * num_nodes).long()  # skewed in-degrees
    key = torch.unique(dst * num_nodes + src)
    dst, src = key // num_nodes, key % num_nodes
    indptr = torch.zeros(num_nodes + 1, dtype=torch.int64)
    indptr[1:] = torch.bincount(dst, minlength=num_nodes).cumsum(0)
    adj = [set(src[indptr[v] : indptr[v + 1]].tolist()) for v in range(num_nodes)]
    return indptr.to(device), src.to(device), adj


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fanouts", [[5, 10], [3, -1, 7], [25]])
def test_blocks_are_valid_samples(device, fanouts):
    indptr, indices, adj = random_graph(device=device)
    sampler = GPUNeighborSampler(indptr, indices, fanouts)
    seeds = torch.randperm(indptr.numel() - 1)[:64].to(device)
    batch = sampler.sample(seeds)
    assert len(batch.blocks) == len(fanouts)

    # Global ids of each block's sources: block 0 = input_nodes; block i+1 = prefix of block i.
    gids = [batch.input_nodes]
    for blk in batch.blocks[:-1]:
        gids.append(gids[-1][: blk.num_dst])
    assert torch.equal(gids[-1][: batch.blocks[-1].num_dst], seeds)

    for i, (blk, g) in enumerate(zip(batch.blocks, gids)):
        assert blk.fanout == fanouts[i]
        assert g.numel() == blk.num_src and torch.unique(g).numel() == g.numel()
        if i + 1 < len(batch.blocks):
            assert blk.num_dst == batch.blocks[i + 1].num_src
        assert blk.indptr.dtype == torch.int32 and blk.indices.dtype == torch.int32
        assert int(blk.indptr[-1]) == blk.num_edges
        assert blk.num_edges == 0 or int(blk.indices.max()) < blk.num_src
        for row in range(blk.num_dst):
            v = int(g[row])
            nbrs = g[blk.indices[blk.indptr[row] : blk.indptr[row + 1]].long()].tolist()
            k = len(adj[v]) if blk.fanout < 0 else min(len(adj[v]), blk.fanout)
            assert len(nbrs) == k and len(set(nbrs)) == k and set(nbrs) <= adj[v]


def test_sampling_is_uniform():
    indptr, indices, adj = random_graph()
    v = int(torch.argmax(indptr[1:] - indptr[:-1]))
    deg = len(adj[v])
    sampler = GPUNeighborSampler(indptr, indices, [5])
    trials = 3000
    counts = torch.zeros(indptr.numel() - 1)
    for _ in range(trials):
        b = sampler.sample(torch.tensor([v]))
        counts[b.input_nodes[b.blocks[0].indices.long()]] += 1
    hit = counts[list(adj[v])]
    expected = trials * 5 / deg
    assert hit.sum().item() == trials * 5  # exactly `fanout` distinct picks per draw
    assert (hit > 0).float().mean() > 0.99  # essentially every neighbour gets drawn
    # Uniform without replacement => per-neighbour counts ~ Binomial(trials, 5/deg); a biased
    # sampler (e.g. favouring low CSR positions) inflates the spread well past sqrt(expected).
    assert hit.std().item() < 1.5 * expected**0.5


def test_seed_batches_cover_once():
    idx = torch.arange(1000)
    batches = seed_batches(idx, 128)
    allseeds = torch.cat(batches)
    assert allseeds.numel() == 1000 and torch.equal(allseeds.sort().values, idx)
    assert [b.numel() for b in batches][:-1] == [128] * 7


@pytest.mark.parametrize("device", DEVICES)
def test_adaptive_frontier_dedup_sparse_and_dense_regimes(device):
    """Both branches (E_batch << N sparse unique vs E_batch >= N/8 dense mark scan) produce exact blocks."""
    # Sparse mini-batch regime: N=5000, non-hub seed set -> E_batch * 8 < N
    indptr_s, indices_s, adj_s = random_graph(num_nodes=5000, num_edges=25000, device=device, seed=1)
    seeds_s = torch.arange(4000, 4016, device=device)
    b_s = GPUNeighborSampler(indptr_s, indices_s, [-1]).sample(seeds_s)
    blk_s = b_s.blocks[0]
    assert blk_s.num_edges > 0 and blk_s.num_edges * 8 < 5000
    g_s = b_s.input_nodes
    assert torch.unique(g_s).numel() == g_s.numel()
    # New sources past dst must be strictly ascending
    assert torch.equal(g_s[16:], g_s[16:].sort().values)
    for row in range(16):
        v = int(seeds_s[row])
        nbrs = g_s[blk_s.indices[blk_s.indptr[row] : blk_s.indptr[row + 1]].long()].tolist()
        assert set(nbrs) == adj_s[v]

    # Dense full-neighbor regime: N=200, large seed set -> E_batch * 8 >= N
    indptr_d, indices_d, adj_d = random_graph(num_nodes=200, num_edges=6000, device=device, seed=2)
    seeds_d = torch.arange(64, device=device)
    b_d = GPUNeighborSampler(indptr_d, indices_d, [-1]).sample(seeds_d)
    blk_d = b_d.blocks[0]
    assert blk_d.num_edges * 8 >= 200
    g_d = b_d.input_nodes
    assert torch.unique(g_d).numel() == g_d.numel()
    assert torch.equal(g_d[64:], g_d[64:].sort().values)
    for row in range(64):
        nbrs = g_d[blk_d.indices[blk_d.indptr[row] : blk_d.indptr[row + 1]].long()].tolist()
        assert set(nbrs) == adj_d[row]
