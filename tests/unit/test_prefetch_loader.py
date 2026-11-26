"""Tests for PrefetchLoader (single/two-stage, GPU timing, early break, errors) and train_sampled helpers."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src.sampling.async_sampler import AsyncNeighborSampler
from src.sampling.loader import PrefetchLoader, resolve_gpu_ms
from src.sampling.neighbor_sampler import GPUNeighborSampler

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _make_graph(num_nodes: int = 400, num_edges: int = 8000, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    src = torch.randint(0, num_nodes, (num_edges,), generator=g)
    dst = (torch.rand(num_edges, generator=g) ** 3 * num_nodes).long()
    key = torch.unique(dst * num_nodes + src)
    dst, src = key // num_nodes, key % num_nodes
    indptr = torch.zeros(num_nodes + 1, dtype=torch.int64)
    indptr[1:] = torch.bincount(dst, minlength=num_nodes).cumsum(0)
    return indptr.cuda(), src.to(torch.int32).cuda()


@pytest.mark.parametrize("prefetch", [False, True])
def test_prefetch_loader_single_stage(prefetch):
    dev = torch.device("cuda", torch.cuda.current_device())
    seeds = [torch.arange(i * 16, (i + 1) * 16, device=dev) for i in range(6)]
    loader = PrefetchLoader(lambda s: {"s": s + 1}, seeds, prefetch=prefetch, depth=2, device=dev)
    got = []
    timings = []
    for batch, timing in loader:
        got.append(batch["s"].clone())
        timings.append(timing)
    torch.cuda.synchronize()
    assert torch.equal(torch.cat(got), torch.arange(1, 97, device=dev))
    for t in timings:
        r = resolve_gpu_ms(t)
        assert r["produce_host_ms"] >= 0.0 and r["wait_host_ms"] >= 0.0 and r["produce_gpu_ms"] >= 0.0


@pytest.mark.parametrize("prefetch", [False, True])
def test_prefetch_loader_two_stage_gpu_timing_does_not_span_idle_gap(prefetch):
    """Two-stage _gpu_events must sum launch + finalize spans, not launch_start -> fin_end."""
    dev = torch.device("cuda", torch.cuda.current_device())

    class TwoStageMaker:
        def launch(self, s):
            return s * 2

        def finalize(self, pending):
            return {"val": pending + 1}

    seeds = [torch.arange(i * 16, (i + 1) * 16, device=dev) for i in range(5)]
    loader = PrefetchLoader(TwoStageMaker(), seeds, prefetch=prefetch, depth=1, device=dev)
    out = []
    timings = []
    for batch, timing in loader:
        # Insert a deliberate 30 ms GPU sleep on compute stream (which also stalls queue drain).
        torch.cuda._sleep(int(30e6))
        out.append(batch["val"].clone())
        timings.append(timing)
    torch.cuda.synchronize()
    assert len(out) == 5
    for idx, t in enumerate(timings):
        # Must record two separate event pairs: [launch_events, finalize_events]
        assert isinstance(t["_gpu_events"], list) and len(t["_gpu_events"]) == 2
        resolved = resolve_gpu_ms(t)
        # Each trivial kernel pair should take far less than 30 ms even with consumer stalls.
        assert 0.0 <= resolved["produce_gpu_ms"] < 15.0, f"batch {idx}: {resolved}"


def test_prefetch_loader_early_break_and_exception():
    dev = torch.device("cuda", torch.cuda.current_device())
    seeds = [torch.arange(i * 8, (i + 1) * 8, device=dev) for i in range(10)]

    # Early break must drain producer cleanly without hanging.
    loader = PrefetchLoader(lambda s: {"s": s + 1}, seeds, prefetch=True, depth=2, device=dev)
    seen = 0
    for _batch, _ in loader:
        seen += 1
        if seen == 2:
            break
    assert seen == 2

    # Producer exception must surface cleanly in consumer.
    def boom(s):
        if int(s[0]) == 16:
            raise ValueError("producer boom")
        return {"s": s}

    bad_loader = PrefetchLoader(boom, seeds, prefetch=True, depth=2, device=dev)
    with pytest.raises(ValueError, match="producer boom"):
        for _ in bad_loader:
            pass


def test_train_sampled_run_epoch_and_batch_makers():
    """Verify run_epoch records separate bwd/opt timing during training and batch makers prebuild slices."""
    scripts_dir = str(Path(__file__).resolve().parents[2] / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    import train_sampled as ts

    dev = torch.device("cuda", torch.cuda.current_device())
    indptr, indices = _make_graph()
    N = indptr.numel() - 1
    x = torch.randn(N, 16, device=dev)
    y = torch.randint(0, 5, (N,), device=dev)
    task = ts.Task("multiclass", y)

    # AsyncBatchMaker prebuilds heavy_edge_slices and clones compact slices out of padded buffers.
    async_s = AsyncNeighborSampler(indptr, indices, [8, 4], fwd_heavy_threshold=4, bwd_heavy_threshold=3)
    async_maker = ts.AsyncBatchMaker(async_s, x, y, "cuda", kernel_kwargs={"forward_heavy_edge_slice": 4})
    seeds = torch.arange(32, device=dev)
    batch = async_maker.finalize(async_maker.launch(seeds))
    for g in batch["graphs"]:
        assert ("forward", 4) in g._edge_slice_cache
        # Cloned exact slices must own compact storage, not padded E_cap storage.
        assert g.forward_indices.untyped_storage().nbytes() == g.forward_indices.numel() * g.forward_indices.element_size()

    # OursBatchMaker parses kernel_kwargs once and prebuilds heavy_edge_slices.
    args = SimpleNamespace(
        fwd_heavy_threshold=4,
        bwd_heavy_threshold=3,
        quantile=None,
        fwd_order="natural",
        bwd_order="natural",
        kernel_kwargs='{"forward_heavy_edge_slice": 4}',
    )
    gpu_s = GPUNeighborSampler(indptr, indices, [8, 4])
    ours_maker = ts.OursBatchMaker(gpu_s, x, y, "cuda", args)
    b2 = ours_maker(seeds)
    assert all(("forward", 4) in g._edge_slice_cache for g in b2["graphs"])

    # run_epoch: train=True must NOT enter the eval branch (opt_ms > 0, metrics is None).
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(16, 5)

        def forward(self, h, graphs):
            return self.lin(h[: graphs[-1].num_dst])

    model = TinyModel().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    batches = [(ours_maker(torch.arange(i * 32, (i + 1) * 32, device=dev)), {"wait_host_ms": 0.1}) for i in range(3)]
    rows, loss, metrics = ts.run_epoch(model, opt, None, batches, SimpleNamespace(), torch.float32, task, train=True)
    assert metrics is None and len(rows) == 3 and loss > 0
    assert all(r["opt_ms"] > 0.0 for r in rows)

    # Eval returns valid metrics.
    rows_ev, _, metrics_ev = ts.run_epoch(model, None, None, batches, SimpleNamespace(), torch.float32, task, train=False)
    assert len(rows_ev) == 3 and metrics_ev is not None and 0.0 <= metrics_ev["acc"] <= 1.0
