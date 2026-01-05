import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Optional

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
    std_ms: float | None = None
    memory_allocated: float | None = None


# def _sync() -> None:
#     """Synchronize device if CUDA is available."""
#     if torch.cuda.is_available():
#         torch.cuda.synchronize()


def measure_memory(func):
    """
    Measure GPU memory usage of a function call.

    Returns:
        result: function output
        memory_allocated (MB): delta allocated during the call
        peak_memory (MB): max allocated during the call
    """
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    start_memory = torch.cuda.memory_allocated() / 1024**2

    result = func()

    torch.cuda.synchronize()
    end_memory = torch.cuda.memory_allocated() / 1024**2
    peak_memory = torch.cuda.max_memory_allocated() / 1024**2

    memory_allocated = end_memory - start_memory
    # return result, memory_allocated, peak_memory
    return result, peak_memory, peak_memory


def time_callable(
    fn: Callable[[], Any], warmup: int = 10, iters: int = 50, do_memory_profile: bool = True
) -> MicrobenchResult:
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
        torch.cuda.synchronize()

    if torch.cuda.is_available():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
            torch.cuda.synchronize()

        end.record()
        end.synchronize()
        total_ms = start.elapsed_time(end)

        if do_memory_profile:
            _, memory_allocated, peak_memory = measure_memory(func=fn)
        else:
            memory_allocated = None

        return MicrobenchResult(
            iters=iters,
            ms_per_iter=total_ms / iters,
            device=device,
            memory_allocated=memory_allocated,
        )

    else:
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        ms = (time.perf_counter() - t0) * 1000.0
        return MicrobenchResult(iters=iters, ms_per_iter=ms / iters, device=device)


def get_gpu_info(device=None) -> dict[str, Any]:
    """Return GPU info

    Returns:
        dict[str, Any]: gpu info for metrics
    """
    device = device or torch.get_default_device()

    if torch.cuda.is_available():
        device_properties = torch.cuda.get_device_properties(device)
        return {
            "device_name": device_properties.name,
            "device_total_memory_mb": device_properties.total_memory / 2**20,
            "sm_count": device_properties.multi_processor_count,
            "compute_capability": f"{device_properties.major}.{device_properties.minor}",
            "max_threads_per_sm": device_properties.max_threads_per_multi_processor,
            "registers_per_sm": device_properties.regs_per_multiprocessor,
        }
    return {}
