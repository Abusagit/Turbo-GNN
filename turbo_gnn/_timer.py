"""CUDA-event-based timer for kernel benchmarking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch


@dataclass
class TimerResult:
    """Timing result for a callable."""

    iters: int
    ms_per_iter: float


def time_callable(
    fn: Callable[[], Any],
    warmup: int = 10,
    iters: int = 50,
) -> TimerResult:
    """Benchmark a zero-arg callable using CUDA events.

    Args:
        fn: Callable to benchmark.
        warmup: Warmup iterations (discarded).
        iters: Timed iterations.

    Returns:
        TimerResult with average time per iteration in ms.
    """
    for _ in range(warmup):
        fn()

    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        fn()
    end.record()

    torch.cuda.synchronize()
    ms_total = start.elapsed_time(end)

    return TimerResult(iters=iters, ms_per_iter=ms_total / iters)
