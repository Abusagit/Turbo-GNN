import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch

doc = """
Microbenchmark helper for timing only the layer kernel (forward/backward).
"""


@dataclass
class MicrobenchResult:
    """Timing result for a callable."""
    iters: int
    ms_per_iter: float
    device: str
    std_ms: Optional[float] = None


def _sync() -> None:
    """Synchronize device if CUDA is available."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_callable(fn: Callable[[], Any], warmup: int = 10, iters: int = 50) -> MicrobenchResult:
    """Benchmark a zero-arg callable with warmup and averaged iterations.

    Args:
        fn (Callable[[], Any]): Callable to benchmark.
        warmup (int): Warmup iterations (discarded).
        iters (int): Timed iterations.

    Returns:
        MicrobenchResult: Average time per iteration in ms.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for _ in range(warmup):
        fn()
    _sync()

    if torch.cuda.is_available():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        _sync()
        total_ms = start.elapsed_time(end)
        return MicrobenchResult(iters=iters, ms_per_iter=total_ms / iters, device=device)
    else:
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        ms = (time.perf_counter() - t0) * 1000.0
        return MicrobenchResult(iters=iters, ms_per_iter=ms / iters, device=device)
