from contextlib import contextmanager
from typing import Any, Optional

import torch
from torch.profiler import ProfilerActivity, profile, schedule, tensorboard_trace_handler

doc = """
Small torch.profiler wrapper for training pipelines and scripted profiles.
"""


def make_schedule(wait: int = 1, warmup: int = 1, active: int = 3, repeat: int = 1):
    """Create common profiling schedule.

    Args:
        wait (int): Wait steps before warmup.
        warmup (int): Warmup steps before active profiling.
        active (int): Active steps to record.
        repeat (int): Repeat cycles.

    Returns:
        Any: torch.profiler.schedule object.
    """
    return schedule(wait=wait, warmup=warmup, active=active, repeat=repeat)


@contextmanager
def prof_ctx(
    log_dir: str,
    *,
    with_cpu: bool = True,
    with_cuda: bool = True,
    record_shapes: bool = True,
    profile_memory: bool = True,
    sched: Any | None = None,
):
    """Context manager for torch.profiler with sensible defaults.

    Args:
        log_dir (str): Output dir for TensorBoard traces.
        with_cpu (bool): Include CPU activities.
        with_cuda (bool): Include CUDA activities (if available).
        record_shapes (bool): Record input shapes.
        profile_memory (bool): Track memory usage.
        sched (Optional[Any]): Optional profiling schedule.

    Yields:
        profile: A torch.profiler.profile object.
    """
    activities = []
    if with_cpu:
        activities.append(ProfilerActivity.CPU)
    if with_cuda and torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)
    with profile(
        activities=activities,
        schedule=sched or make_schedule(),
        on_trace_ready=tensorboard_trace_handler(log_dir),
        record_shapes=record_shapes,
        profile_memory=profile_memory,
        with_stack=True,
        with_flops=True,
    ) as p:
        yield p
