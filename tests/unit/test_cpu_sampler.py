"""CPU neighbor sampling, CPU batching, pinned H2D transfer, and parallel-process loading."""

from __future__ import annotations

import pytest
import torch

from src.sampling.cpu_loader import CPUPrefetchLoader, pin_batch, transfer_batch_to_device
from src.sampling.cpu_sampler import CPUBatchMaker, CPUNeighborSampler
from src.sampling.loader import iter_tensors, resolve_gpu_ms
from src.sampling.neighbor_sampler import GPUNeighborSampler, seed_batches
from turbo_gnn.graph import AdjacencyForwardBackwardWithNodeBuckets as G

HAS_CUDA = torch.cuda.is_available()
TARGET_DEVICES = ["cpu"] + (["cuda"] if HAS_CUDA else [])


def make_random_graph(num_nodes: int = 1200, num_edges: int = 30000, feat_dim: int = 16, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    src = torch.randint(0, num_nodes, (num_edges,), generator=g)
    dst = (torch.rand(num_edges, generator=g) ** 3 * num_nodes).long()  # skewed in-degrees
    key = torch.unique(dst * num_nodes + src)
    dst, src = key // num_nodes, key % num_nodes
    indptr = torch.zeros(num_nodes + 1, dtype=torch.int64)
    indptr[1:] = torch.bincount(dst, minlength=num_nodes).cumsum(0)
    indices = src.to(torch.int32)
    adj = [set(src[indptr[v] : indptr[v + 1]].tolist()) for v in range(num_nodes)]
    x = torch.randn(num_nodes, feat_dim, generator=g)
    y = torch.randint(0, 7, (num_nodes,), generator=g)
    return indptr, indices, x, y, adj


class _FailingProducer:
    """Top-level picklable callable that raises on a target seed batch (for worker error tests)."""

    def __init__(self, fail_on_first_seed: int) -> None:
        self.fail_on_first_seed = fail_on_first_seed

    def __call__(self, seeds: torch.Tensor):
        if int(seeds[0].item()) == self.fail_on_first_seed:
            raise ValueError(f"boom on seed {self.fail_on_first_seed}")
        return {"seeds": seeds.clone()}


class _EchoProducer:
    """Top-level picklable callable echoing seeds to test ordering and IPC."""

    def __call__(self, seeds: torch.Tensor):
        return {"seeds": seeds.clone(), "tag": int(seeds[0].item())}


# ----------------------------------------------------------------- CPUNeighborSampler
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("fanouts", [[5, 10], [3, -1, 7], [25], [-1], [-1, -1]])
def test_cpu_sampler_blocks_are_valid(fanouts, index_dtype):
    indptr, indices, _, _, adj = make_random_graph()
    sampler = CPUNeighborSampler(indptr, indices, fanouts, index_dtype=index_dtype)
    seeds = torch.randperm(indptr.numel() - 1)[:64]
    batch = sampler.sample(seeds)

    assert len(batch.blocks) == len(fanouts)
    assert batch.seeds.device.type == "cpu" and batch.input_nodes.device.type == "cpu"

    gids = [batch.input_nodes]
    for blk in batch.blocks[:-1]:
        gids.append(gids[-1][: blk.num_dst])
    assert torch.equal(gids[-1][: batch.blocks[-1].num_dst], seeds)

    for i, (blk, g) in enumerate(zip(batch.blocks, gids)):
        assert blk.fanout == fanouts[i]
        assert g.numel() == blk.num_src and torch.unique(g).numel() == g.numel()
        if i + 1 < len(batch.blocks):
            assert blk.num_dst == batch.blocks[i + 1].num_src
        assert blk.indptr.dtype == index_dtype and blk.indices.dtype == index_dtype
        assert int(blk.indptr[-1]) == blk.num_edges
        assert blk.num_edges == 0 or int(blk.indices.max()) < blk.num_src
        for row in range(blk.num_dst):
            v = int(g[row])
            nbrs = g[blk.indices[blk.indptr[row] : blk.indptr[row + 1]].long()].tolist()
            k = len(adj[v]) if blk.fanout < 0 else min(len(adj[v]), blk.fanout)
            assert len(nbrs) == k and len(set(nbrs)) == k and set(nbrs) <= adj[v]


def test_cpu_sampler_edge_cases():
    # Graph with isolated nodes (nodes 0, 1, 2 have degree 0; node 3 has 2 edges).
    indptr = torch.tensor([0, 0, 0, 0, 2], dtype=torch.int64)
    indices = torch.tensor([0, 1], dtype=torch.int32)
    sampler = CPUNeighborSampler(indptr, indices, [5, 3])

    # Empty seeds
    empty = sampler.sample(torch.empty(0, dtype=torch.int64))
    assert empty.input_nodes.numel() == 0
    assert all(b.num_dst == 0 and b.num_src == 0 and b.num_edges == 0 for b in empty.blocks)

    # All-isolated seeds (0 sampled edges across all layers)
    iso = sampler.sample(torch.tensor([0, 2]))
    assert torch.equal(iso.input_nodes, torch.tensor([0, 2]))
    assert all(b.num_edges == 0 and b.num_dst == 2 and b.num_src == 2 for b in iso.blocks)

    # Reject invalid fanouts and CUDA tensors
    with pytest.raises(ValueError, match="fanouts"):
        CPUNeighborSampler(indptr, indices, [0])
    with pytest.raises(ValueError, match="fanouts"):
        CPUNeighborSampler(indptr, indices, [-2])
    if HAS_CUDA:
        with pytest.raises(ValueError, match="CPU"):
            CPUNeighborSampler(indptr.cuda(), indices.cuda(), [3])


def test_cpu_sampler_is_uniform():
    indptr, indices, _, _, adj = make_random_graph()
    v = int(torch.argmax(indptr[1:] - indptr[:-1]))
    deg = len(adj[v])
    gen = torch.Generator(device="cpu").manual_seed(42)
    sampler = CPUNeighborSampler(indptr, indices, [5], generator=gen)
    trials = 3000
    counts = torch.zeros(indptr.numel() - 1)
    seed = torch.tensor([v])
    for _ in range(trials):
        b = sampler.sample(seed)
        counts[b.input_nodes[b.blocks[0].indices.long()]] += 1
    hit = counts[list(adj[v])]
    expected = trials * 5 / deg
    assert hit.sum().item() == trials * 5
    assert (hit > 0).float().mean() > 0.99
    assert hit.std().item() < 1.5 * expected**0.5


def test_cpu_sampler_matches_gpu_sampler_full_neighbor():
    indptr, indices, _, _, _ = make_random_graph()
    seeds = torch.randperm(indptr.numel() - 1)[:64]
    cpu_b = CPUNeighborSampler(indptr, indices, [-1, -1]).sample(seeds)
    ref_b = GPUNeighborSampler(indptr, indices, [-1, -1]).sample(seeds)

    # Identical global node sets and per-destination global neighbor sets at every layer
    # (newly discovered non-destination nodes may appear in different order).
    assert torch.equal(cpu_b.input_nodes.sort().values, ref_b.input_nodes.sort().values)
    cpu_gids, ref_gids = [cpu_b.input_nodes], [ref_b.input_nodes]
    for blk in cpu_b.blocks[:-1]:
        cpu_gids.append(cpu_gids[-1][: blk.num_dst])
    for blk in ref_b.blocks[:-1]:
        ref_gids.append(ref_gids[-1][: blk.num_dst])
    assert torch.equal(cpu_gids[-1][: cpu_b.blocks[-1].num_dst], seeds)

    for g, r, cg, rg in zip(cpu_b.blocks, ref_b.blocks, cpu_gids, ref_gids):
        assert g.num_dst == r.num_dst and g.num_src == r.num_src and g.num_edges == r.num_edges
        cpu_map = {}
        for row in range(g.num_dst):
            v = int(cg[row])
            s, e = int(g.indptr[row]), int(g.indptr[row + 1])
            cpu_map[v] = sorted(cg[g.indices[s:e].long()].tolist())
        ref_map = {}
        for row in range(r.num_dst):
            v = int(rg[row])
            s, e = int(r.indptr[row]), int(r.indptr[row + 1])
            ref_map[v] = sorted(rg[r.indices[s:e].long()].tolist())
        assert cpu_map == ref_map


def test_cpu_sampler_parallel_partitioned_path():
    """Exercise the >= 65536-edge parallel 32-way zero-atomic partitioned dedup path in C++."""
    indptr, indices, _, _, adj = make_random_graph(num_nodes=8000, num_edges=320000, feat_dim=8, seed=11)
    sampler = CPUNeighborSampler(indptr, indices, [20, 15, 10], index_dtype=torch.int32, num_threads=8)
    seeds = torch.randperm(8000)[:256]
    batch = sampler.sample(seeds)
    # Confirm outermost layer exceeded the 65536 parallel threshold (choose_threads(..., 32768) >= 2).
    assert batch.blocks[0].num_edges >= 65536
    gids = [batch.input_nodes]
    for blk in batch.blocks[:-1]:
        gids.append(gids[-1][: blk.num_dst])
    assert torch.equal(gids[-1][: batch.blocks[-1].num_dst], seeds)
    for blk, g in zip(batch.blocks, gids):
        assert g.numel() == blk.num_src and torch.unique(g).numel() == g.numel()
        for row in range(min(blk.num_dst, 200)):
            v = int(g[row])
            s, e = int(blk.indptr[row]), int(blk.indptr[row + 1])
            nbrs = g[blk.indices[s:e].long()].tolist()
            k = min(len(adj[v]), blk.fanout)
            assert len(nbrs) == k and len(set(nbrs)) == k and set(nbrs) <= adj[v]


def test_cpu_sampler_parallel_duplicate_seeds():
    """Duplicate destination seeds must work cleanly and race-free in the parallel dense path."""
    indptr, indices, _, _, adj = make_random_graph(num_nodes=8000, num_edges=320000, feat_dim=8, seed=17)
    sampler = CPUNeighborSampler(indptr, indices, [20], index_dtype=torch.int32, num_threads=8)
    base = torch.arange(3500)
    # Place duplicates across distant chunks so multiple threads see them.
    seeds = torch.cat([base, base[:1000]])
    batch = sampler.sample(seeds)
    blk = batch.blocks[0]
    assert blk.num_edges >= 65536
    g = batch.input_nodes
    assert torch.equal(g[: seeds.numel()], seeds)
    for row in (0, 10, 3500, 3510):
        v = int(seeds[row])
        s, e = int(blk.indptr[row]), int(blk.indptr[row + 1])
        nbrs = g[blk.indices[s:e].long()].tolist()
        k = min(len(adj[v]), 20)
        assert len(nbrs) == k and len(set(nbrs)) == k and set(nbrs) <= adj[v]


def test_cpu_sampler_reproducible_with_generator():
    indptr, indices, _, _, _ = make_random_graph()
    seeds = torch.randperm(indptr.numel() - 1)[:64]
    b1 = CPUNeighborSampler(indptr, indices, [10, 5], generator=torch.Generator().manual_seed(123)).sample(seeds)
    b2 = CPUNeighborSampler(indptr, indices, [10, 5], generator=torch.Generator().manual_seed(123)).sample(seeds)
    assert torch.equal(b1.input_nodes, b2.input_nodes)
    for x, y in zip(b1.blocks, b2.blocks):
        assert torch.equal(x.indptr, y.indptr) and torch.equal(x.indices, y.indices)


# -------------------------------------------------------------------- CPUBatchMaker
@pytest.mark.parametrize("order", ["natural", "degree"])
@pytest.mark.parametrize("thresholds", [(None, None), (4, 3)])
def test_cpu_batch_maker_cuda_graphs_and_compact_storage(order, thresholds):
    indptr, indices, x, y, _ = make_random_graph()
    maker = CPUBatchMaker(
        indptr,
        indices,
        x,
        y,
        [10, 5],
        backend="cuda",
        fwd_heavy_threshold=thresholds[0],
        bwd_heavy_threshold=thresholds[1],
        fwd_order=order,
        bwd_order=order,
        heavy_edge_slices={"forward": 4, "backward": 4},
        seed=7,
    )
    seeds = torch.randperm(indptr.numel() - 1)[:48]
    batch = maker(seeds)

    assert batch["x"].device.type == "cpu" and batch["y"].device.type == "cpu"
    assert torch.equal(batch["y"], y[seeds])
    assert batch["x"].shape == (batch["graphs"][0].num_src, x.shape[1])
    assert len(batch["graphs"]) == 2

    # Every emitted tensor must own compact storage (never a view into the full graph/features).
    for t in iter_tensors(batch):
        assert t.untyped_storage().nbytes() == t.numel() * t.element_size()

    # Validate transposed CSR, buckets, degree ordering, and precomputed heavy edge slices.
    for g in batch["graphs"]:
        assert g.device.type == "cpu"
        ip, ix = g.forward_indptr.long(), g.forward_indices.long()
        bp, bx = g.backward_indptr.long(), g.backward_indices.long()
        dst_of = torch.repeat_interleave(torch.arange(g.num_dst), ip[1:] - ip[:-1])
        src_of = torch.repeat_interleave(torch.arange(g.num_src), bp[1:] - bp[:-1])
        assert sorted(zip(ix.tolist(), dst_of.tolist())) == sorted(zip(src_of.tolist(), bx.tolist()))

        for light, heavy, deg, n, thresh in (
            (g.forward_light_nodes, g.forward_heavy_nodes, ip[1:] - ip[:-1], g.num_dst, thresholds[0]),
            (g.backward_light_nodes, g.backward_heavy_nodes, bp[1:] - bp[:-1], g.num_src, thresholds[1]),
        ):
            lt, hv = light.long(), heavy.long()
            assert sorted(torch.cat([lt, hv]).tolist()) == list(range(n))
            if thresh is None:
                assert hv.numel() == 0
            else:
                assert bool((deg[hv] >= thresh).all()) and bool((deg[lt] < thresh).all())
            if order == "degree":
                for rows in (lt, hv):
                    d = deg[rows]
                    assert bool((d[:-1] >= d[1:]).all())
                    same = d[:-1] == d[1:]
                    assert bool((rows[:-1][same] < rows[1:][same]).all())

        assert ("forward", 4) in g._edge_slice_cache
        assert ("backward", 4) in g._edge_slice_cache


@pytest.mark.skipif(not HAS_CUDA, reason="CUDA required for pinned memory")
def test_graph_pin_memory_and_non_blocking_transfer():
    indptr, indices, x, y, _ = make_random_graph()
    maker = CPUBatchMaker(
        indptr,
        indices,
        x,
        y,
        [8, 4],
        fwd_heavy_threshold=4,
        bwd_heavy_threshold=3,
        heavy_edge_slices={"forward": 4, "backward": 4},
        pin_memory=True,
        seed=0,
    )
    seeds = torch.randperm(indptr.numel() - 1)[:64]
    batch = maker(seeds)
    assert batch["x"].is_pinned() and batch["y"].is_pinned()
    for t in iter_tensors(batch):
        # Empty 0-byte tensors have null data_ptr, so PyTorch reports is_pinned() == False for them.
        assert t.device.type == "cpu" and (t.numel() == 0 or t.is_pinned())

    dev = torch.device("cuda", torch.cuda.current_device())
    # Warm up allocator, then verify transfer_batch_to_device(non_blocking=True) does not sync.
    _ = transfer_batch_to_device(maker(seeds), dev, non_blocking=True)
    torch.cuda.synchronize()

    pinned_batch = maker(seeds)
    torch.cuda.set_sync_debug_mode("error")
    try:
        gpu_batch = transfer_batch_to_device(pinned_batch, dev, non_blocking=True)
    finally:
        torch.cuda.set_sync_debug_mode(0)
    torch.cuda.synchronize()

    for t in iter_tensors(gpu_batch):
        assert t.device == dev
    for g in gpu_batch["graphs"]:
        assert g.device == dev
        assert ("forward", 4) in g._edge_slice_cache
        assert g._edge_slice_cache[("forward", 4)].chunk_node.device == dev


# ---------------------------------------------------------------- CPUPrefetchLoader
@pytest.mark.parametrize("target_device", TARGET_DEVICES)
@pytest.mark.parametrize(
    "num_workers,prefetch,pin_memory,mp_context",
    [
        (0, False, False, "spawn"),
        (0, False, True, "spawn"),
        (0, True, True, "spawn"),
        (1, True, True, "spawn"),
        (2, True, True, "spawn"),
        (2, True, False, "spawn"),
        (2, True, True, "forkserver"),
    ],
)
def test_cpu_prefetch_loader_modes(target_device, num_workers, prefetch, pin_memory, mp_context):
    indptr, indices, x, y, _ = make_random_graph(num_nodes=800, num_edges=15000, feat_dim=16, seed=3)
    maker = CPUBatchMaker(indptr, indices, x, y, [8, 4], fwd_heavy_threshold=4, bwd_heavy_threshold=3, seed=5)
    train_idx = torch.arange(320)
    batches_seeds = seed_batches(train_idx, batch_size=64, shuffle=False)

    expected_dev = torch.device("cuda", torch.cuda.current_device()) if target_device == "cuda" else torch.device("cpu")
    with CPUPrefetchLoader(
        maker,
        batches_seeds,
        num_workers=num_workers,
        prefetch=prefetch,
        depth=2,
        pin_memory=pin_memory,
        device=target_device,
        persistent_workers=True,
        mp_context=mp_context,
        seed=5,
    ) as loader:
        # Run 2 epochs to verify persistent worker reuse and deterministic batch ordering.
        for _ in range(2):
            seen_y = []
            for batch, timing in loader.epoch(batches_seeds):
                assert batch["x"].device == expected_dev
                assert batch["y"].device == expected_dev
                assert all(g.device == expected_dev for g in batch["graphs"])
                assert timing["produce_host_ms"] >= 0.0
                assert timing["wait_host_ms"] >= 0.0
                seen_y.append(batch["y"].cpu())
                if target_device == "cuda":
                    torch.cuda.synchronize()
                    resolved = resolve_gpu_ms(timing)
                    assert resolved["produce_gpu_ms"] >= 0.0
            assert torch.equal(torch.cat(seen_y), y[train_idx])


def test_cpu_prefetch_loader_preserves_order_and_early_break():
    train_idx = torch.arange(200)
    batches_seeds = seed_batches(train_idx, batch_size=10, shuffle=False)

    with CPUPrefetchLoader(
        _EchoProducer(),
        num_workers=2,
        depth=2,
        pin_memory=False,
        device="cpu",
        persistent_workers=True,
        preserve_order=True,
        mp_context="spawn",
    ) as loader:
        # Early break after 3 batches (out of 20)
        partial = []
        for batch, _ in loader.epoch(batches_seeds):
            partial.append(batch["tag"])
            if len(partial) == 3:
                break
        assert partial == [0, 10, 20]

        # Full subsequent epoch on the same persistent workers: no stale batches leaked
        full = [batch["tag"] for batch, _ in loader.epoch(batches_seeds)]
        assert full == [i * 10 for i in range(20)]


def test_cpu_prefetch_loader_surfaces_worker_exception():
    batches_seeds = seed_batches(torch.arange(60), batch_size=10, shuffle=False)
    with CPUPrefetchLoader(
        _FailingProducer(fail_on_first_seed=20),
        batches_seeds,
        num_workers=2,
        depth=2,
        pin_memory=False,
        device="cpu",
        persistent_workers=False,
        mp_context="spawn",
    ) as loader:
        with pytest.raises(RuntimeError, match="boom on seed 20"):
            for _ in loader:
                pass


@pytest.mark.skipif(not HAS_CUDA, reason="CUDA required")
def test_cpu_prefetch_loader_consumer_never_syncs():
    """Consumer handoff (wait_event + record_stream) from parallel CPU workers must not sync GPU."""
    indptr, indices, x, y, _ = make_random_graph(num_nodes=800, num_edges=15000, feat_dim=32, seed=1)
    maker = CPUBatchMaker(indptr, indices, x, y, [8, 4], seed=1)
    batches_seeds = seed_batches(torch.arange(384), batch_size=64, shuffle=False)
    dev = torch.device("cuda", torch.cuda.current_device())

    with CPUPrefetchLoader(
        maker,
        num_workers=2,
        depth=2,
        pin_memory=True,
        device=dev,
        persistent_workers=True,
        mp_context="spawn",
        seed=1,
    ) as loader:
        # Warm up pinned host / CUDA allocators on epoch 0
        for batch, _ in loader.epoch(batches_seeds[:2]):
            _ = batch["x"].sum()
        torch.cuda.synchronize()

        torch.cuda.set_sync_debug_mode("error")
        try:
            acc = torch.zeros((), device=dev)
            for batch, _ in loader.epoch(batches_seeds):
                acc = acc + batch["x"].sum()
        finally:
            torch.cuda.set_sync_debug_mode(0)
        torch.cuda.synchronize()
        assert torch.isfinite(acc)


@pytest.mark.skipif(not HAS_CUDA, reason="CUDA required")
def test_cpu_sampling_end_to_end_gpu_step():
    """Verify full CPU-graph -> parallel CPU sampler -> pinned H2D -> GPU forward/backward step."""
    indptr, indices, x, y, _ = make_random_graph(num_nodes=600, num_edges=12000, feat_dim=32, seed=9)
    maker = CPUBatchMaker(indptr, indices, x, y, [10, 5], seed=9)
    batches_seeds = seed_batches(torch.arange(256), batch_size=64, shuffle=True)
    dev = torch.device("cuda", torch.cuda.current_device())

    # Simple pure-torch message-passing model consuming AdjacencyForwardBackwardWithNodeBuckets
    # (runs even without rebuilding the C++ extension).
    class SimpleBlockNet(torch.nn.Module):
        def __init__(self, in_dim: int, hidden: int, out_dim: int) -> None:
            super().__init__()
            self.w1 = torch.nn.Linear(in_dim, hidden)
            self.w2 = torch.nn.Linear(hidden, out_dim)

        def forward(self, h: torch.Tensor, graphs: list[G]) -> torch.Tensor:
            for lin, g in zip((self.w1, self.w2), graphs):
                ip = g.forward_indptr.long()
                ix = g.forward_indices.long()
                deg = (ip[1:] - ip[:-1]).clamp(min=1)
                dst = torch.repeat_interleave(torch.arange(g.num_dst, device=h.device), ip[1:] - ip[:-1])
                agg = torch.zeros(g.num_dst, h.shape[1], device=h.device, dtype=h.dtype)
                if ix.numel() > 0:
                    agg.index_add_(0, dst, h[ix])
                h = torch.relu(lin(agg / deg[:, None]))
            return h

    model = SimpleBlockNet(32, 32, 7).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)

    with CPUPrefetchLoader(
        maker,
        num_workers=2,
        depth=2,
        pin_memory=True,
        device=dev,
        persistent_workers=False,
        mp_context="spawn",
        seed=9,
    ) as loader:
        losses = []
        for batch, _ in loader.epoch(batches_seeds):
            opt.zero_grad(set_to_none=True)
            logits = model(batch["x"], batch["graphs"])
            loss = torch.nn.functional.cross_entropy(logits, batch["y"])
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        assert len(losses) == 4 and all(v > 0 for v in losses)
        # Confirm original graph and features never moved off CPU
        assert indptr.device.type == "cpu" and x.device.type == "cpu" and y.device.type == "cpu"


@pytest.mark.skipif(not HAS_CUDA, reason="CUDA required")
def test_fork_after_cuda_init_rejected():
    _ = torch.empty(1, device="cuda")
    batches_seeds = seed_batches(torch.arange(20), batch_size=10, shuffle=False)
    with CPUPrefetchLoader(
        _EchoProducer(),
        batches_seeds,
        num_workers=1,
        device="cpu",
        mp_context="fork",
    ) as loader:
        with pytest.raises(RuntimeError, match="mp_context='fork'"):
            for _ in loader:
                pass


def test_multiprocess_keeps_features_out_of_shared_memory():
    """Topology-only workers share indptr/indices, leaving large x/y matrices in the main process."""
    indptr, indices, x, y, _ = make_random_graph(num_nodes=400, num_edges=6000, feat_dim=16, seed=2)
    maker = CPUBatchMaker(indptr, indices, x, y, [5, 3], seed=2)
    batches_seeds = seed_batches(torch.arange(128), batch_size=32, shuffle=False)

    with CPUPrefetchLoader(
        maker,
        num_workers=2,
        depth=2,
        pin_memory=HAS_CUDA,
        device="cuda" if HAS_CUDA else "cpu",
        persistent_workers=False,
        mp_context="spawn",
        seed=2,
    ) as loader:
        seen = []
        for batch, _ in loader.epoch(batches_seeds):
            seen.append(batch["y"].cpu())
            assert batch["x"].shape == (batch["graphs"][0].num_src, x.shape[1])
        assert torch.equal(torch.cat(seen), y[:128])
    assert indptr.is_shared() and indices.is_shared()
    assert not x.is_shared() and not y.is_shared()


def test_cpu_batch_maker_dgl_raw_blocks_and_transfer():
    """backend='dgl' keeps worker output pure torch (raw SampledBlocks) until post_transfer."""
    indptr, indices, x, y, _ = make_random_graph(num_nodes=300, num_edges=4000, feat_dim=8, seed=4)
    maker = CPUBatchMaker(indptr, indices, x, y, [6, 4], backend="dgl", seed=4)
    seeds = torch.arange(32)
    raw = maker(seeds)
    assert raw["graphs"] is None and len(raw["_dgl_blocks"]) == 2
    assert torch.equal(raw["y"], y[seeds])
    if HAS_CUDA:
        pinned = pin_batch(raw)
        assert pinned["x"].is_pinned() and all(b.indptr.is_pinned() for b in pinned["_dgl_blocks"])
        dev = torch.device("cuda", torch.cuda.current_device())
        on_gpu = transfer_batch_to_device(pinned, dev, non_blocking=True)
        torch.cuda.synchronize()
        assert on_gpu["x"].device == dev and all(b.indptr.device == dev for b in on_gpu["_dgl_blocks"])


@pytest.mark.parametrize("quantile", [0.9, -1])
def test_empty_seeds_with_quantile_bucketing(quantile):
    indptr, indices, x, y, _ = make_random_graph(num_nodes=100, num_edges=500, feat_dim=8, seed=0)
    maker = CPUBatchMaker(indptr, indices, x, y, [5, 3], quantile=quantile, seed=0)
    batch = maker(torch.empty(0, dtype=torch.int64))
    assert batch["x"].shape == (0, 8) and batch["y"].shape == (0,)
    assert all(g.num_dst == 0 and g.num_src == 0 and g.forward_light_nodes.numel() == 0 for g in batch["graphs"])


def test_close_stops_active_iterator_cleanly():
    batches_seeds = seed_batches(torch.arange(200), batch_size=10, shuffle=False)
    loader = CPUPrefetchLoader(
        _EchoProducer(),
        num_workers=2,
        depth=2,
        pin_memory=False,
        device="cpu",
        persistent_workers=True,
        mp_context="spawn",
    )
    it = loader.epoch(batches_seeds)
    first, _ = next(it)
    assert first["tag"] == 0
    # Closing while `it` is still alive must stop & join the transfer thread cleanly.
    loader.close()
    assert not loader._active_threads
    it.close()


def test_single_process_clears_pin_memory_when_disabled():
    indptr, indices, x, y, _ = make_random_graph(num_nodes=100, num_edges=500, feat_dim=8, seed=0)
    maker = CPUBatchMaker(indptr, indices, x, y, [4], pin_memory=True, seed=0)
    maker.sampler.num_threads = 8
    seeds = seed_batches(torch.arange(32), batch_size=16, shuffle=False)
    with CPUPrefetchLoader(
        maker, seeds, num_workers=0, prefetch=False, pin_memory=False, device="cpu", worker_threads=2
    ) as loader:
        assert loader._single_produce.pin_memory is False
        assert loader._single_produce.sampler.num_threads == 2
        assert maker.sampler.num_threads == 8
        for batch, _ in loader:
            assert not batch["x"].is_pinned()


def test_cpu_sampler_large_fanout_and_hash_path_and_copy_isolation():
    import copy

    # 1) fanout > 64 exercises the heap hash-set Floyd path (sample_row_large_fanout).
    indptr = torch.tensor([0, 150] + [150] * 149, dtype=torch.int64)
    indices = torch.arange(150, dtype=torch.int32)
    s_big_f = CPUNeighborSampler(indptr, indices, [80], num_threads=4)
    b = s_big_f.sample(torch.tensor([0]))
    blk = b.blocks[0]
    assert blk.num_edges == 80
    nbrs = b.input_nodes[blk.indices.long()].tolist()
    assert len(set(nbrs)) == 80 and min(nbrs) >= 0 and max(nbrs) < 150

    # 2) Shallow copy must not share mutable C++ CpuSamplerState*.
    s_copy = copy.copy(s_big_f)
    assert s_copy._native is None
    _ = s_copy.sample(torch.tensor([0]))
    if s_big_f._native is not None and s_copy._native is not None:
        assert s_big_f._native._ptr != s_copy._native._ptr

    # 3) Hash-table frontier path (N > 4_000_000): verify exactness with zero O(N) table.
    big_n = 4_000_010
    big_ip = torch.zeros(big_n + 1, dtype=torch.int64)
    big_ip[1:] = 20  # node 0 has 20 neighbors; all others have degree 0
    big_ix = torch.tensor([big_n - 1 - k for k in range(20)], dtype=torch.int32)
    s_hash = CPUNeighborSampler(big_ip, big_ix, [10], num_threads=4)
    bh = s_hash.sample(torch.tensor([0, 5]))
    assert bh.blocks[0].num_edges == 10
    picked = set(bh.input_nodes[bh.blocks[0].indices.long()].tolist())
    assert len(picked) == 10 and picked <= set(big_ix.tolist())


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_parallel_transpose_exact_sorted_order(index_dtype):
    """Exercise the >= 65536-edge 32-way zero-atomic parallel CSR transpose in C++."""
    indptr, indices, _, _, _ = make_random_graph(num_nodes=4000, num_edges=200000, feat_dim=8, seed=13)
    sampler = CPUNeighborSampler(indptr, indices, [20, 15, 10], index_dtype=index_dtype, num_threads=8)
    seeds = torch.randperm(4000)[:256]
    batch = sampler.sample(seeds)
    blk = batch.blocks[0]
    assert blk.num_edges >= 65536

    g = sampler.build_cuda_graph(blk, fwd_heavy_threshold=15, bwd_heavy_threshold=10, fwd_order="degree", bwd_order="degree")
    ip, ix = g.forward_indptr.long(), g.forward_indices.long()
    bp, bx = g.backward_indptr.long(), g.backward_indices.long()
    assert bp.numel() == blk.num_src + 1 and int(bp[-1]) == blk.num_edges
    dst_of = torch.repeat_interleave(torch.arange(blk.num_dst), ip[1:] - ip[:-1])
    src_of = torch.repeat_interleave(torch.arange(blk.num_src), bp[1:] - bp[:-1])
    # Exact (src, dst) sorted order must match forward pairs.
    assert list(zip(src_of.tolist(), bx.tolist())) == sorted(zip(ix.tolist(), dst_of.tolist()))


def test_native_cpu_sampler_has_no_openmp_dependency():
    import subprocess
    from src.sampling._cpu_sampler_cpp import _so_path, load_native_lib

    lib = load_native_lib()
    if lib is None:
        pytest.skip("native C++ backend disabled or unavailable")
    out = subprocess.check_output(["ldd", str(_so_path())], text=True)
    assert "libgomp" not in out and "libiomp" not in out


def test_parallel_large_fanout_hash_indptr_and_gather():
    """Exercise parallel large fanout (>64), parallel hash path (N > 4M), parallel indptr (n >= 131072), and parallel gather (>= 2 MiB)."""
    # 1) Parallel large fanout (>64) with >= 65536 sampled edges.
    n_lf, deg_lf = 1000, 120
    ip_lf = torch.arange(0, (n_lf + 1) * deg_lf, deg_lf, dtype=torch.int64)
    ix_lf = (torch.arange(n_lf * deg_lf, dtype=torch.int32) % n_lf).contiguous()
    s_lf = CPUNeighborSampler(ip_lf, ix_lf, [80], num_threads=4)
    b_lf = s_lf.sample(torch.arange(n_lf))
    assert b_lf.blocks[0].num_edges == n_lf * 80 >= 65536

    # 2) Parallel hash-table path (N > 4_000_000) with >= 32768 sampled edges.
    big_n = 4_000_050
    deg_h = 16
    active_h = 2500
    ip_h = torch.zeros(big_n + 1, dtype=torch.int64)
    ip_h[1 : active_h + 1] = torch.arange(1, active_h + 1, dtype=torch.int64) * deg_h
    ip_h[active_h + 1 :] = active_h * deg_h
    ix_h = (big_n - 1 - (torch.arange(active_h * deg_h, dtype=torch.int32) % 50000)).contiguous()
    s_h = CPUNeighborSampler(ip_h, ix_h, [14], num_threads=4)
    b_h = s_h.sample(torch.arange(active_h))
    assert b_h.blocks[0].num_edges == active_h * 14 >= 32768
    assert torch.unique(b_h.input_nodes).numel() == b_h.input_nodes.numel()

    # 3) Parallel cpu_sample_layer_indptr (n >= 131072) + parallel cpu_gather_rows (>= 2 MiB).
    n_big = 140_000
    ip_big = torch.arange(n_big + 1, dtype=torch.int64)
    ix_big = torch.arange(n_big, dtype=torch.int32)
    x_big = torch.randn(n_big, 16, dtype=torch.float32)  # 140K * 64B = ~9 MiB (> 2 MiB parallel gather)
    y_big = torch.arange(n_big, dtype=torch.int64)
    maker = CPUBatchMaker(ip_big, ix_big, x_big, y_big, [1], seed=0)
    maker.sampler.num_threads = 4
    out = maker(torch.arange(n_big))
    assert out["graphs"][0].num_dst == n_big and out["graphs"][0].forward_indices.numel() == n_big
    assert torch.equal(out["x"], x_big) and torch.equal(out["y"], y_big)
    # Single-process maker must reuse sampler._native's pool rather than creating a second _gather_state.
    assert maker._gather_state is None


def test_fork_after_parent_threadpool_active():
    """mp_context='fork' must not deadlock or SIGABRT even after parent ThreadPool threads were spawned."""
    import os
    import subprocess
    import sys

    script = (
        "import torch\n"
        "torch.set_num_threads(1)\n"  # keep PyTorch's own OpenMP inactive; our C++17 ThreadPool runs with 4 threads
        "from src.sampling import CPUBatchMaker, CPUPrefetchLoader, seed_batches\n"
        "N, deg = 4000, 30\n"
        "ip = torch.arange(0, (N + 1) * deg, deg, dtype=torch.int64)\n"
        "ix = (torch.arange(N * deg, dtype=torch.int32) % N).contiguous()\n"
        "x = torch.randn(N, 256, dtype=torch.float32)\n"  # 4000*1KB = 4MB -> triggers parallel gather
        "y = torch.arange(N, dtype=torch.int64)\n"
        "maker = CPUBatchMaker(ip, ix, x, y, [20], seed=0)\n"
        "maker.sampler.num_threads = 4\n"
        "# Case A: spawn parent sampler ThreadPool BEFORE forking.\n"
        "_ = maker(torch.arange(3500))\n"
        "seeds = seed_batches(torch.arange(2000), 500, shuffle=False)\n"
        "# Case B: run 2 consecutive epochs with persistent_workers=False (forks again after gather pool).\n"
        "with CPUPrefetchLoader(maker, num_workers=2, device='cpu', persistent_workers=False, mp_context='fork', worker_threads=2, seed=0) as loader:\n"
        "    for ep in range(2):\n"
        "        ys = [b['y'] for b, _ in loader.epoch(seeds)]\n"
        "        assert torch.equal(torch.cat(ys), torch.arange(2000))\n"
        "print('FORK_OK')\n"
    )
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    res = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env, timeout=60, check=False)
    assert res.returncode == 0, f"fork test failed ({res.returncode}):\n{res.stderr}\n{res.stdout}"
    assert "FORK_OK" in res.stdout





