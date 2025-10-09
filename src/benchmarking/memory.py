import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

try:
    import psutil  # optional, for RSS reporting
except Exception:  # pragma: no cover - optional dep
    psutil = None  # type: ignore[assignment]


doc = """
Memory benchmarking utilities (CPU + CUDA) for graph NN experiments.

Highlights:
- Human-readable byte formatting
- Parameter/buffer/optimizer-state memory accounting
- CUDA peak memory capture (allocated + reserved)
- One-call measurement of peak CUDA memory during a callable
- Forward/backward (single step) peak memory measurement helpers
- Optional CPU RSS deltas via psutil (if available)

This module has *no trainer assumptions*; you can call it from hooks, tests,
or scripts. All functions are typed and robust to CPU-only environments.
"""


# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------


@dataclass
class CudaMemorySnapshot:
    """A snapshot of CUDA memory metrics on a single device.

    Attributes:
        device (str): Device string (e.g., "cuda:0" or "cpu").
        allocated_bytes (int): Currently allocated bytes.
        reserved_bytes (int): Currently reserved bytes (CUDA memory pool).
        max_allocated_bytes (int): Peak allocated bytes since last reset.
        max_reserved_bytes (int): Peak reserved bytes since last reset.
        stats (Dict[str, int]): Optional extra stats from torch.cuda.memory_stats().
    """

    device: str
    allocated_bytes: int
    reserved_bytes: int
    max_allocated_bytes: int
    max_reserved_bytes: int
    stats: dict[str, int]


@dataclass
class PeakMemoryResult:
    """Peak memory measurement result (for a timed callable).

    Attributes:
        device (str): Device string ("cuda:X" or "cpu").
        duration_s (float): Wall-clock duration of the measured callable (seconds).
        start_allocated (int): Allocated bytes at start (CUDA only).
        end_allocated (int): Allocated bytes at end (CUDA only).
        peak_allocated (int): Peak allocated bytes observed during run (CUDA only).
        start_reserved (int): Reserved bytes at start (CUDA only).
        end_reserved (int): Reserved bytes at end (CUDA only).
        peak_reserved (int): Peak reserved bytes observed (CUDA only).
        cpu_rss_start (Optional[int]): Process RSS at start, if psutil available.
        cpu_rss_end (Optional[int]): Process RSS at end, if psutil available.
    """

    device: str
    duration_s: float
    start_allocated: int
    end_allocated: int
    peak_allocated: int
    start_reserved: int
    end_reserved: int
    peak_reserved: int
    cpu_rss_start: int | None = None
    cpu_rss_end: int | None = None


@dataclass
class ModelMemoryBreakdown:
    """Coarse memory breakdown for a model/setup.

    Attributes:
        param_bytes (int): Sum of parameter tensor bytes.
        buffer_bytes (int): Sum of registered buffer tensor bytes.
        optimizer_state_bytes (int): Bytes in optimizer state tensors.
        activation_bytes_estimate (Optional[int]): Heuristic forward-activation size.
        total_bytes (int): Sum of known components (params + buffers + optim + activation est).
    """

    param_bytes: int
    buffer_bytes: int
    optimizer_state_bytes: int
    activation_bytes_estimate: int | None
    total_bytes: int


# -----------------------------------------------------------------------------
# Formatting helpers
# -----------------------------------------------------------------------------


def human_bytes(nbytes: int, *, binary: bool = False, precision: int = 2) -> str:
    """Format bytes as a human-readable string.

    Args:
        nbytes (int): Number of bytes.
        binary (bool, optional): If True, use KiB/MiB/GiB. If False, use KB/MB/GB.
        precision (int, optional): Decimal places.

    Returns:
        str: Human-readable size string, e.g. "123.45 MiB".
    """
    step = 1024 if binary else 1000
    units = ["B", "KiB", "MiB", "GiB", "TiB"] if binary else ["B", "KB", "MB", "GB", "TB"]
    x = float(max(nbytes, 0))
    for u in units:
        if abs(x) < step:
            return f"{x:.{precision}f} {u}"
        x /= step
    return f"{x * step:.{precision}f} {units[-1]}"


def tensor_nbytes(t: torch.Tensor) -> int:
    """Compute number of bytes for a tensor.

    Args:
        t (torch.Tensor): Input tensor.

    Returns:
        int: t.nelement() * t.element_size().
    """
    return int(t.nelement()) * int(t.element_size())


# -----------------------------------------------------------------------------
# Model/optimizer memory accounting
# -----------------------------------------------------------------------------


def model_param_and_buffer_bytes(model: torch.nn.Module, *, trainable_only: bool = False) -> tuple[int, int]:
    """Compute bytes used by parameters and buffers.

    Args:
        model (torch.nn.Module): The model instance.
        trainable_only (bool, optional): If True, only count parameters with requires_grad=True.

    Returns:
        Tuple[int, int]: (param_bytes, buffer_bytes).
    """
    pbytes = 0
    for p in model.parameters():
        if trainable_only and not p.requires_grad:
            continue
        pbytes += tensor_nbytes(p)

    bbytes = 0
    for b in model.buffers():
        bbytes += tensor_nbytes(b)
    return pbytes, bbytes


def optimizer_state_bytes(optimizer: torch.optim.Optimizer) -> int:
    """Approximate bytes consumed by optimizer state tensors.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer instance.

    Returns:
        int: Sum of bytes of all tensor-like entries in optimizer.state.

    Notes:
        - This inspects `optimizer.state` (a dict keyed by Parameter), and sums
          bytes for all torch.Tensor values found recursively in the state dicts.
    """
    total = 0

    def _recurse(obj: Any) -> None:
        nonlocal total
        if torch.is_tensor(obj):
            total += tensor_nbytes(obj)
        elif isinstance(obj, dict):
            for v in obj.values():
                _recurse(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                _recurse(v)
        # ignore scalars/others

    for state in optimizer.state.values():
        _recurse(state)
    return total


# -----------------------------------------------------------------------------
# CPU / CUDA snapshot utilities
# -----------------------------------------------------------------------------


def _current_device_str(device: torch.device | None = None) -> str:
    """Return a canonical device string."""
    if device is None:
        if torch.cuda.is_available():
            return f"cuda:{torch.cuda.current_device()}"
        return "cpu"
    return f"{device.type}:{device.index}" if device.type == "cuda" else device.type


def capture_cuda_snapshot(device: torch.device | None = None) -> CudaMemorySnapshot:
    """Capture a point-in-time snapshot of CUDA memory metrics.

    Args:
        device (Optional[torch.device]): CUDA device to query (default: current).

    Returns:
        CudaMemorySnapshot: Snapshot; if CUDA is unavailable, all zeros on 'cpu'.
    """
    if not torch.cuda.is_available():
        return CudaMemorySnapshot(
            device="cpu",
            allocated_bytes=0,
            reserved_bytes=0,
            max_allocated_bytes=0,
            max_reserved_bytes=0,
            stats={},
        )
    dev = torch.cuda.current_device() if device is None else device.index
    allocated = int(torch.cuda.memory_allocated(dev))
    reserved = int(torch.cuda.memory_reserved(dev))
    max_alloc = int(torch.cuda.max_memory_allocated(dev))
    max_res = int(torch.cuda.max_memory_reserved(dev))
    stats = torch.cuda.memory_stats(dev)
    return CudaMemorySnapshot(
        device=f"cuda:{dev}",
        allocated_bytes=allocated,
        reserved_bytes=reserved,
        max_allocated_bytes=max_alloc,
        max_reserved_bytes=max_res,
        stats={k: int(v) for k, v in stats.items() if isinstance(v, (int,))},
    )


def reset_cuda_peak_memory(device: torch.device | None = None) -> None:
    """Reset CUDA peak memory statistics (allocated/reserved).

    Args:
        device (Optional[torch.device]): CUDA device (default: current).

    Returns:
        None
    """
    if torch.cuda.is_available():
        dev = torch.cuda.current_device() if device is None else device.index
        torch.cuda.reset_peak_memory_stats(dev)


def current_process_rss_bytes() -> int | None:
    """Return current process resident set size in bytes if `psutil` is available.

    Args:
        None

    Returns:
        Optional[int]: RSS bytes (or None if `psutil` is not available).
    """
    if psutil is None:
        return None
    try:
        return int(psutil.Process().memory_info().rss)
    except Exception:
        return None


# -----------------------------------------------------------------------------
# Peak CUDA memory measurement
# -----------------------------------------------------------------------------


def measure_peak_cuda_memory_during(
    fn: Callable[[], Any], *, device: torch.device | None = None, sync_cuda: bool = True
) -> PeakMemoryResult:
    """Run a callable and capture CUDA peak memory usage during its execution.

    Args:
        fn (Callable[[], Any]): Zero-arg callable to measure.
        device (Optional[torch.device]): CUDA device (default: current).
        sync_cuda (bool): If True, synchronize before/after timing.

    Returns:
        PeakMemoryResult: Start/end/peak allocated and reserved bytes and duration.

    Notes:
        - On CPU-only systems, returns zeros with device='cpu'.
        - Uses `torch.cuda.reset_peak_memory_stats` to isolate the measurement window.
    """
    dev_str = _current_device_str(device)
    rss0 = current_process_rss_bytes()

    if not torch.cuda.is_available():
        t0 = time.perf_counter()
        fn()
        dur = time.perf_counter() - t0
        rss1 = current_process_rss_bytes()
        return PeakMemoryResult(
            device=dev_str,
            duration_s=dur,
            start_allocated=0,
            end_allocated=0,
            peak_allocated=0,
            start_reserved=0,
            end_reserved=0,
            peak_reserved=0,
            cpu_rss_start=rss0,
            cpu_rss_end=rss1,
        )

    dev = torch.cuda.current_device() if device is None else device.index
    if sync_cuda:
        torch.cuda.synchronize(dev)
    reset_cuda_peak_memory(device)
    snap0 = capture_cuda_snapshot(device)
    t0 = time.perf_counter()
    fn()
    if sync_cuda:
        torch.cuda.synchronize(dev)
    dur = time.perf_counter() - t0
    snap1 = capture_cuda_snapshot(device)

    return PeakMemoryResult(
        device=f"cuda:{dev}",
        duration_s=dur,
        start_allocated=snap0.allocated_bytes,
        end_allocated=snap1.allocated_bytes,
        peak_allocated=snap1.max_allocated_bytes,
        start_reserved=snap0.reserved_bytes,
        end_reserved=snap1.reserved_bytes,
        peak_reserved=snap1.max_reserved_bytes,
        cpu_rss_start=rss0,
        cpu_rss_end=current_process_rss_bytes(),
    )


# -----------------------------------------------------------------------------
# Forward/backward convenience profilers
# -----------------------------------------------------------------------------


def measure_forward_peak_cuda_memory(
    model: torch.nn.Module,
    inputs: tuple[Any, ...],
    *,
    kwargs: dict[str, Any] | None = None,
    amp_dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> PeakMemoryResult:
    """Measure peak CUDA memory during a single forward pass.

    Args:
        model (torch.nn.Module): The model to run.
        inputs (Tuple[Any, ...]): Positional inputs for forward().
        kwargs (Optional[Dict[str, Any]]): Keyword inputs for forward().
        amp_dtype (Optional[torch.dtype]): If set and CUDA is available, run under autocast with this dtype.
        device (Optional[torch.device]): CUDA device (default: current).

    Returns:
        PeakMemoryResult: Peak memory during forward pass.
    """
    kwargs = kwargs or {}

    def _call() -> Any:
        model.train()  # training mode to resemble activation footprint
        if amp_dtype is not None and torch.cuda.is_available():
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                return model(*inputs, **kwargs)
        return model(*inputs, **kwargs)

    return measure_peak_cuda_memory_during(_call, device=device)


def measure_train_step_peak_cuda_memory(
    model: torch.nn.Module,
    inputs: tuple[Any, ...],
    loss_fn: Callable[[Any], torch.Tensor],
    *,
    kwargs: dict[str, Any] | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    amp_dtype: torch.dtype | None = None,
    use_grad_scaler: bool = True,
    zero_grad: bool = True,
    device: torch.device | None = None,
) -> PeakMemoryResult:
    """Measure peak CUDA memory during a single training step (fwd+loss+bwd+opt).

    Args:
        model (torch.nn.Module): Model to train.
        inputs (Tuple[Any, ...]): Positional inputs for forward().
        loss_fn (Callable[[Any], torch.Tensor]): Computes scalar loss from model output.
        kwargs (Optional[Dict[str, Any]]): Keyword inputs for forward().
        optimizer (Optional[torch.optim.Optimizer]): If provided, performs an optimizer step.
        amp_dtype (Optional[torch.dtype]): If set and CUDA is available, use autocast with this dtype.
        use_grad_scaler (bool): If True, use GradScaler when amp_dtype is set.
        zero_grad (bool): If True, zero model/optimizer grads before the step.
        device (Optional[torch.device]): CUDA device (default: current).

    Returns:
        PeakMemoryResult: Peak memory during the entire train step.

    Notes:
        - This function mutates model/optimizer state if `optimizer` is provided.
        - For a pure measurement without mutation, pass `optimizer=None` and avoid .step().
    """
    kwargs = kwargs or {}

    scaler = torch.cuda.amp.GradScaler(
        enabled=bool(amp_dtype is not None and use_grad_scaler and torch.cuda.is_available())
    )

    def _call() -> Any:
        if zero_grad:
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            else:
                model.zero_grad(set_to_none=True)
        model.train()
        if amp_dtype is not None and torch.cuda.is_available():
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                out = model(*inputs, **kwargs)
                loss = loss_fn(out)
            scaler.scale(loss).backward()
            if optimizer is not None:
                scaler.step(optimizer)
                scaler.update()
        else:
            out = model(*inputs, **kwargs)
            loss = loss_fn(out)
            loss.backward()
            if optimizer is not None:
                optimizer.step()
        return None

    return measure_peak_cuda_memory_during(_call, device=device)


# -----------------------------------------------------------------------------
# Activation size heuristic (forward-time)
# -----------------------------------------------------------------------------


def estimate_forward_activation_bytes(
    model: torch.nn.Module,
    inputs: tuple[Any, ...],
    *,
    kwargs: dict[str, Any] | None = None,
    include_non_float: bool = False,
) -> int:
    """Heuristically estimate activation bytes kept during forward.

    Args:
        model (torch.nn.Module): Model to inspect.
        inputs (Tuple[Any, ...]): Positional inputs for the forward pass.
        kwargs (Optional[Dict[str, Any]]): Keyword inputs for the forward pass.
        include_non_float (bool): If True, include integer/bool activations too.

    Returns:
        int: Estimated bytes from module outputs seen in forward hooks.

    Notes:
        - This is a heuristic: it sums the sizes of tensors emitted by modules'
          forward hooks. Actual training-time autograd saved tensors may differ.
        - Useful for relative comparisons across backends/layers.
    """
    kwargs = kwargs or {}
    total_bytes = 0

    def _accumulate(obj: Any) -> None:
        nonlocal total_bytes
        if torch.is_tensor(obj):
            if include_non_float or obj.is_floating_point():
                total_bytes += tensor_nbytes(obj)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                _accumulate(v)
        elif isinstance(obj, dict):
            for v in obj.values():
                _accumulate(v)

    handles: list[torch.utils.hooks.RemovableHandle] = []

    def _hook(_module: torch.nn.Module, _inp: tuple[Any, ...], out: Any) -> None:
        _accumulate(out)

    for m in model.modules():
        if m is model:
            continue
        handles.append(m.register_forward_hook(_hook))

    try:
        # Enable grad to mimic training graph (without backward).
        with torch.enable_grad():
            model.train()
            _ = model(*inputs, **kwargs)
    finally:
        for h in handles:
            h.remove()

    return int(total_bytes)


# -----------------------------------------------------------------------------
# High-level one-shot breakdown
# -----------------------------------------------------------------------------


def summarize_model_memory(
    model: torch.nn.Module,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    activation_estimate_bytes: int | None = None,
    trainable_only_params: bool = False,
) -> ModelMemoryBreakdown:
    """Summarize model memory components (params/buffers/optimizer/activations).

    Args:
        model (torch.nn.Module): Target model.
        optimizer (Optional[torch.optim.Optimizer]): Optimizer to account state.
        activation_estimate_bytes (Optional[int]): If provided (e.g., from
            `estimate_forward_activation_bytes`), include it in total.
        trainable_only_params (bool): If True, count only trainable params.

    Returns:
        ModelMemoryBreakdown: Structured breakdown including total_bytes.
    """
    pbytes, bbytes = model_param_and_buffer_bytes(model, trainable_only=trainable_only_params)
    obytes = optimizer_state_bytes(optimizer) if optimizer is not None else 0
    total = pbytes + bbytes + obytes + (activation_estimate_bytes or 0)
    return ModelMemoryBreakdown(
        param_bytes=pbytes,
        buffer_bytes=bbytes,
        optimizer_state_bytes=obytes,
        activation_bytes_estimate=activation_estimate_bytes,
        total_bytes=total,
    )
