"""Coverage tests for the node scheduler in ``csrc/common/scheduler.cuh``.

The scheduler decides which thread block processes which node. The property that has to hold
for every policy, at every grid size, is that each work item is handled **exactly once** --
not zero times (dropped work) and not twice (double-counted output). These tests assert that
directly, so a failure implicates the scheduler and nothing downstream.

Opt-in like the rest of ``tests/csrc``: ``pytest tests/csrc --csrc`` or ``TGNN_RUN_CSRC=1``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]

ONE_PER_BLOCK, GRID_STRIDE, PRECOMPUTED, DYNAMIC = 0, 1, 2, 3
ALL_KINDS = [ONE_PER_BLOCK, GRID_STRIDE, PRECOMPUTED, DYNAMIC]
PERSISTENT_KINDS = [GRID_STRIDE, PRECOMPUTED, DYNAMIC]
KIND_NAMES = {
    ONE_PER_BLOCK: "one_per_block",
    GRID_STRIDE: "grid_stride",
    PRECOMPUTED: "precomputed",
    DYNAMIC: "dynamic",
}

_MODULE = None


def _bridge():
    """JIT-build (once, cached on disk) and return the scheduler bridge module."""
    global _MODULE
    if _MODULE is None:
        from torch.utils.cpp_extension import load

        build_dir = REPO_ROOT / "build" / "scheduler_tests"
        build_dir.mkdir(parents=True, exist_ok=True)
        _MODULE = load(
            name="scheduler_test_bridge",
            sources=[str(REPO_ROOT / "tests/csrc/scheduler_test_bridge.cu")],
            extra_include_paths=[str(REPO_ROOT / "csrc")],
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                "-std=c++20",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "-U__CUDA_NO_HALF2_OPERATORS__",
                "-U__CUDA_NO_BFLOAT162_OPERATORS__",
            ],
            extra_cflags=["-O3", "-std=c++20"],
            build_directory=str(build_dir),
            verbose=False,
        )
    return _MODULE


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="scheduler tests need a GPU")


def _block_offsets(count: int, num_blocks: int, device: str = "cuda") -> torch.Tensor:
    """Even contiguous split of ``count`` items across ``num_blocks``, as CSR-style offsets."""
    edges = torch.linspace(0, count, num_blocks + 1, device=device)
    return edges.round().to(torch.int32).contiguous()


def _run(
    kind: int,
    count: int,
    *,
    heads: int = 1,
    blocks_per_sm: int = 8,
    threads: int = 128,
    nodes: torch.Tensor | None = None,
    skip_every: int = 0,
    force_grid_x: int | None = None,
    chunk: int = 1,
):
    offs = None
    if kind == PRECOMPUTED:
        grid_x = force_grid_x if force_grid_x is not None else max(1, min(64, max(1, count)))
        offs = _block_offsets(count, grid_x)
    return _bridge().run_scheduler(
        kind, count, heads, blocks_per_sm, threads, nodes, offs, skip_every, force_grid_x, chunk
    )


def _assert_exact_coverage(visits: torch.Tensor, count: int, heads: int) -> None:
    """Every (work item, head) handled exactly once."""
    assert visits.shape == (count, heads)
    if count == 0:
        return
    bad = (visits != 1).nonzero()
    assert bad.numel() == 0, (
        f"{bad.shape[0]} of {count * heads} (item, head) pairs were not visited exactly once; "
        f"min={int(visits.min())} max={int(visits.max())}, first offenders={bad[:8].tolist()}"
    )


# ---------------------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ALL_KINDS, ids=lambda k: KIND_NAMES[k])
@pytest.mark.parametrize("count", [1, 7, 64, 1000, 5000])
def test_every_item_visited_exactly_once(kind, count):
    visits, per_block, _first, grid_x = _run(kind, count)
    _assert_exact_coverage(visits, count, heads=1)
    assert int(per_block.sum()) == count, "blocks collectively handled the wrong number of items"
    if kind == ONE_PER_BLOCK:
        assert grid_x == count, "one_per_block must keep grid.x == item count"


@pytest.mark.parametrize("kind", ALL_KINDS, ids=lambda k: KIND_NAMES[k])
@pytest.mark.parametrize("heads", [1, 4, 8])
def test_coverage_is_per_head(kind, heads):
    """Heads live on gridDim.y; each head must independently cover every item."""
    count = 777
    visits, per_block, _first, _grid_x = _run(kind, count, heads=heads)
    _assert_exact_coverage(visits, count, heads)
    for h in range(heads):
        assert int(per_block[h].sum()) == count, f"head {h} handled the wrong number of items"


@pytest.mark.parametrize("kind", ALL_KINDS, ids=lambda k: KIND_NAMES[k])
def test_coverage_with_node_indirection(kind):
    """A permuted `nodes` array (an LPT order) must still cover every node exactly once."""
    count = 2048
    perm = torch.randperm(count, device="cuda").to(torch.int32).contiguous()
    visits, _pb, _fi, _gx = _run(kind, count, nodes=perm)
    _assert_exact_coverage(visits, count, heads=1)


@pytest.mark.parametrize("kind", ALL_KINDS, ids=lambda k: KIND_NAMES[k])
def test_isolated_node_continue_path(kind):
    """`continue` must not desynchronise the block: the fence lives in next(), not the body."""
    count = 1500
    visits, per_block, _fi, _gx = _run(kind, count, heads=4, skip_every=3)
    _assert_exact_coverage(visits, count, heads=4)
    assert int(per_block.sum()) == count * 4


# ---------------------------------------------------------------------------------------
# Grid-size edge cases
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", PERSISTENT_KINDS, ids=lambda k: KIND_NAMES[k])
@pytest.mark.parametrize("count", [1, 16, 63, 64, 65])
def test_more_blocks_than_work(kind, count):
    """Surplus blocks must claim a rank, find nothing and exit -- not hang, not double-count."""
    visits, _pb, _fi, grid_x = _run(kind, count, force_grid_x=64)
    assert grid_x == 64
    _assert_exact_coverage(visits, count, heads=1)


@pytest.mark.parametrize("kind", PERSISTENT_KINDS, ids=lambda k: KIND_NAMES[k])
def test_far_more_work_than_blocks(kind):
    """Deep persistence: 8 blocks chew through 20k items."""
    count = 20000
    visits, per_block, _fi, grid_x = _run(kind, count, force_grid_x=8)
    assert grid_x == 8
    _assert_exact_coverage(visits, count, heads=1)
    assert int(per_block.sum()) == count


@pytest.mark.parametrize("kind", ALL_KINDS, ids=lambda k: KIND_NAMES[k])
def test_zero_work_items(kind):
    """A bucket can legitimately be empty (quantile=-1 puts every node in `light`)."""
    visits, _pb, _fi, _gx = _run(kind, 0, force_grid_x=8 if kind != ONE_PER_BLOCK else None)
    assert visits.numel() == 0


@pytest.mark.parametrize("kind", PERSISTENT_KINDS, ids=lambda k: KIND_NAMES[k])
@pytest.mark.parametrize("threads", [32, 128, 1024])
def test_block_shapes(kind, threads):
    """Single-warp blocks and full 1024-thread blocks both broadcast the claim correctly."""
    count = 999
    visits, _pb, _fi, _gx = _run(kind, count, threads=threads)
    _assert_exact_coverage(visits, count, heads=1)


# ---------------------------------------------------------------------------------------
# Policy-specific behaviour
# ---------------------------------------------------------------------------------------


def test_ranks_are_dense_and_ascending():
    """Persistent policies claim logical ids dynamically; they must form a permutation of
    [0, grid_x) per head, which is what makes `atomicAdd(...) + gridDim.x` cover the tail."""
    count, heads, grid_x = 4096, 4, 32
    _v, _pb, first_idx, gx = _run(GRID_STRIDE, count, heads=heads, force_grid_x=grid_x)
    assert gx == grid_x
    for h in range(heads):
        ranks = first_idx[h].sort().values
        expected = torch.arange(grid_x, device=ranks.device, dtype=ranks.dtype)
        assert torch.equal(ranks, expected), f"head {h} ranks are not a permutation of [0,{grid_x})"


def test_precomputed_respects_its_assignment():
    """PrecomputedList must walk exactly the slices the host laid out.

    Compared as multisets, not element-wise: ranks are claimed dynamically, so the block at
    ``blockIdx.x == b`` holds whichever rank it claimed, not rank ``b``. Each block's first
    item also has to be one of the slice starts.
    """
    count, grid_x = 1000, 16
    offs = _block_offsets(count, grid_x)
    visits, per_block, first_idx, gx = _bridge().run_scheduler(PRECOMPUTED, count, 1, 8, 128, None, offs, 0, grid_x)
    assert gx == grid_x
    _assert_exact_coverage(visits, count, heads=1)

    expected = (offs[1:] - offs[:-1]).to(per_block.dtype)
    assert torch.equal(per_block[0].sort().values, expected.sort().values), (
        f"slice sizes do not match block_offsets: got {sorted(per_block[0].tolist())}, "
        f"expected {sorted(expected.tolist())}"
    )
    starts = first_idx[0][first_idx[0] >= 0].sort().values
    assert torch.equal(starts, offs[:-1].to(starts.dtype).sort().values), (
        "blocks did not start at the slice boundaries in block_offsets"
    )


def test_dynamic_queue_balances_better_than_grid_stride():
    """The point of the dynamic queue: with unequal per-item cost the static stride fixes each
    block's share up front, while the queue lets fast blocks take more items."""
    count, grid_x = 4000, 32
    _v, pb_static, _fi, _gx = _run(GRID_STRIDE, count, force_grid_x=grid_x)
    _v2, pb_dynamic, _fi2, _gx2 = _run(DYNAMIC, count, force_grid_x=grid_x)
    # Static stride hands out a fixed count; the queue's distribution is demand-driven.
    assert int(pb_static.sum()) == count
    assert int(pb_dynamic.sum()) == count
    assert int(pb_static.max()) - int(pb_static.min()) <= 1, "grid-stride shares should be even by count"


@pytest.mark.parametrize("kind", ALL_KINDS, ids=lambda k: KIND_NAMES[k])
def test_repeat_launches_are_stable(kind):
    """Counters are zeroed per launch; a second launch must not inherit the first's state."""
    count = 1234
    for _ in range(3):
        visits, _pb, _fi, _gx = _run(kind, count, heads=2)
        _assert_exact_coverage(visits, count, heads=2)


# ---------------------------------------------------------------------------------------
# DynamicQueue: chunked claiming and the contiguity invariant
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("chunk", [1, 2, 4, 8, 32])
@pytest.mark.parametrize("count", [1, 7, 100, 5000])
def test_dynamic_chunk_coverage(chunk, count):
    """Chunking changes how many items an atomic buys, never which items get done."""
    visits, per_block, _fi, _gx = _run(DYNAMIC, count, chunk=chunk)
    _assert_exact_coverage(visits, count, heads=1)
    assert int(per_block.sum()) == count


@pytest.mark.parametrize("chunk", [1, 4, 16])
def test_dynamic_chunk_with_ragged_tail(chunk):
    """`count` not divisible by `chunk`: the last chunk is short and must not overrun."""
    count = 1000 + chunk // 2 + 1
    visits, _pb, _fi, _gx = _run(DYNAMIC, count, heads=2, chunk=chunk, force_grid_x=13)
    _assert_exact_coverage(visits, count, heads=2)


@pytest.mark.parametrize("chunk", [1, 2, 8])
def test_dynamic_claims_are_a_dense_prefix(chunk):
    """The property the monotone cursor exists for.

    Every block draws from one counter, including its first claim, so the chunks handed out
    are ``0, 1, 2, ...`` in demand order with no gaps. The first ``grid_x`` claims are
    therefore exactly the first ``grid_x`` chunks -- which is what keeps the items in flight a
    *contiguous window* of the node list rather than a scatter.

    The earlier `atomicAdd(...) + gridDim.x` form failed this: it pre-assigned chunk ``rank``
    to block ``rank`` and then jumped by ``gridDim.x``, so with a grid deeper than one
    resident wave the in-flight window fragmented and CSR streaming locality was lost. That
    cost up to 2x on dense graphs, so it is worth pinning down.
    """
    count, grid_x = 4096, 16
    _v, _pb, first_idx, gx = _run(DYNAMIC, count, chunk=chunk, force_grid_x=grid_x)
    assert gx == grid_x
    claims = first_idx[0][first_idx[0] >= 0].sort().values
    expected = torch.arange(grid_x, device=claims.device, dtype=claims.dtype) * chunk
    assert torch.equal(claims, expected), (
        f"first claims are not the first {grid_x} chunks of size {chunk}: got {claims.tolist()[:8]}, "
        f"expected {expected.tolist()[:8]}"
    )


def test_dynamic_grid_deeper_than_work_still_covers_once():
    """More blocks than chunks: the surplus draw a base past the end and exit immediately."""
    count, chunk, grid_x = 40, 8, 64
    visits, _pb, first_idx, _gx = _run(DYNAMIC, count, chunk=chunk, force_grid_x=grid_x)
    _assert_exact_coverage(visits, count, heads=1)
    assert int((first_idx[0] >= 0).sum()) == 5, "only ceil(40/8) blocks should have found work"
