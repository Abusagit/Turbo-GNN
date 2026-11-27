"""CPU mini-batch loader with pinned memory, async H2D overlap, and parallel worker processes.

Mirrors `PrefetchLoader` (`src/sampling/loader.py`) for CPU-resident graphs and CPU batching:

- **Single process (``num_workers=0``)**:
  - ``prefetch=False`` runs inline on the calling thread, pins the batch (when ``pin_memory=True``
    and targeting CUDA), and transfers it to ``device``.
  - ``prefetch=True`` prepares batches on a background thread, pins them, and copies them to GPU
    on a dedicated H2D CUDA stream so CPU sampling and PCIe transfers overlap with GPU training.
- **Parallel processes (``num_workers>=1``)**:
  - Worker processes (pure CPU, zero CUDA calls) sample neighbors in parallel without contending
    on the Python GIL. For `CPUBatchMaker` (via ``for_worker()`` / ``finalize_cpu_batch()``),
    workers ship only compact packed topology tensors (1 IPC FD per batch, keeping large
    ``x``/``y`` feature matrices out of ``/dev/shm``; forward CSRs only when targeting CUDA).
    Arbitrary custom ``produce`` callables are also supported directly.
  - A main-process transfer thread unpacks worker results, gathers ``x``/``y`` directly into a
    reusable page-locked staging ring (``_PinnedSlabRing``, avoiding per-batch ``cudaHostAlloc``
    syscalls), stages index tensors, and launches ``non_blocking=True`` H2D DMA + ``post_transfer``
    (e.g. GPU-side CSR transpose/bucketing or DGL block wrapping) on a dedicated CUDA stream,
    handing off to the compute stream via a CUDA event + ``record_stream`` (zero host syncs in
    the training loop).
  - Workers are kept alive across epochs by default (``persistent_workers=True``) so process
    startup is paid once.

Timing keys match `PrefetchLoader` and `resolve_gpu_ms` (all milliseconds):

- ``produce_host_ms``: CPU sampling + batching time (hidden by workers/prefetch thread);
- ``h2d_host_ms``: host time spent pinning and queuing the H2D transfer;
- ``produce_gpu_ms`` (via ``resolve_gpu_ms``): GPU time of the H2D transfer on the side stream;
- ``wait_host_ms``: exposed wall time the training loop waited for the next batch.
"""

from __future__ import annotations

import copy
import ctypes
import dataclasses
import os
import queue
import threading
import time
import traceback
import weakref
from collections.abc import Callable, Iterable, Iterator
from typing import Any

import torch
import torch.multiprocessing as mp

from src.sampling.loader import iter_tensors

_SENTINEL = object()


def _estimate_cpu_bytes(obj: Any) -> int:
    total = 64  # headroom for initial 64B alignment
    seen: set[int] = set()
    for t in iter_tensors(obj):
        tid = id(t)
        if tid in seen:
            continue
        seen.add(tid)
        if t.device.type == "cpu" and t.numel() > 0 and not t.is_pinned():
            total += (t.numel() * t.element_size() + 63) & ~63
    return total


class _PinnedSlab:
    """Reusable page-locked host memory slab with bump allocation and H2D completion tracking.

    Eliminates per-batch ``cudaHostAlloc`` / ``cudaFreeHost`` kernel syscalls and multi-core
    TLB shootdowns when mini-batch tensor shapes vary slightly from step to step.
    """

    def __init__(self) -> None:
        self.buf: torch.Tensor | None = None
        self.offset: int = 0
        self.ready: torch.cuda.Event | None = None

    def begin(self, required_bytes: int) -> None:
        if self.ready is not None:
            # Poll rather than calling event.synchronize() to stay compatible with
            # torch.cuda.set_sync_debug_mode("error") and release the GIL if still in flight.
            while not self.ready.query():
                time.sleep(50e-6)
            self.ready = None
        if self.buf is None or self.buf.numel() < required_bytes:
            # 1.25x headroom on initial alloc prevents growth reallocs on minor batch-size jitter.
            cap = max(int(required_bytes * 1.25), 4 * 1024 * 1024)
            if self.buf is not None:
                cap = max(cap, int(self.buf.numel() * 1.5))
            cap = (cap + 4095) & ~4095
            self.buf = torch.empty(cap, dtype=torch.uint8, pin_memory=True)
        self.offset = 0

    def alloc(self, shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        numel = 1
        for d in shape:
            numel *= d
        if numel == 0:
            return torch.empty(shape, dtype=dtype)
        elem_size = torch.empty((), dtype=dtype).element_size()
        nbytes = numel * elem_size
        start = (self.offset + 63) & ~63
        end = start + nbytes
        assert self.buf is not None and end <= self.buf.numel()
        self.offset = end
        return self.buf[start:end].view(dtype).reshape(shape)


class _PinnedSlabRing:
    """Round-robin ring of ``num_slots`` reusable pinned slabs (``depth + 2``)."""

    def __init__(self, num_slots: int) -> None:
        self._slabs = [_PinnedSlab() for _ in range(max(2, num_slots))]
        self._next = 0

    def acquire(self, required_bytes: int) -> _PinnedSlab:
        slab = self._slabs[self._next]
        self._next = (self._next + 1) % len(self._slabs)
        slab.begin(required_bytes)
        return slab


def _stage_in_slab(obj: Any, slab: _PinnedSlab, _memo: dict[int, Any] | None = None) -> Any:
    """Copy unpinned CPU tensors in ``obj`` into views of ``slab`` (zero ``cudaHostAlloc``)."""
    memo = _memo if _memo is not None else {}
    oid = id(obj)
    if oid in memo:
        return memo[oid]

    if isinstance(obj, torch.Tensor):
        if obj.device.type == "cpu" and obj.numel() > 0 and not obj.is_pinned():
            dst = slab.alloc(tuple(obj.shape), obj.dtype)
            if obj.is_contiguous():
                # Direct memcpy avoids waking PyTorch's multi-socket OpenMP pool for tiny index tensors.
                ctypes.memmove(dst.data_ptr(), obj.data_ptr(), obj.numel() * obj.element_size())
            else:
                dst.copy_(obj)
            res = dst
        else:
            res = obj
        memo[oid] = res
        return res
    if hasattr(obj, "forward_indptr") and hasattr(obj, "_edge_slice_cache"):
        obj.forward_indptr = _stage_in_slab(obj.forward_indptr, slab, memo)
        obj.forward_indices = _stage_in_slab(obj.forward_indices, slab, memo)
        if obj.is_directed:
            obj.backward_indptr = _stage_in_slab(obj.backward_indptr, slab, memo)
            obj.backward_indices = _stage_in_slab(obj.backward_indices, slab, memo)
        else:
            obj.backward_indptr = obj.forward_indptr
            obj.backward_indices = obj.forward_indices
        obj.forward_light_nodes = _stage_in_slab(obj.forward_light_nodes, slab, memo)
        obj.forward_heavy_nodes = _stage_in_slab(obj.forward_heavy_nodes, slab, memo)
        obj.backward_light_nodes = _stage_in_slab(obj.backward_light_nodes, slab, memo)
        obj.backward_heavy_nodes = _stage_in_slab(obj.backward_heavy_nodes, slab, memo)
        if obj.forward_edge_perm is not None:
            obj.forward_edge_perm = _stage_in_slab(obj.forward_edge_perm, slab, memo)
        if obj._edge_slice_cache:
            obj._edge_slice_cache = {k: _stage_in_slab(v, slab, memo) for k, v in obj._edge_slice_cache.items()}
        memo[oid] = obj
        return obj
    if isinstance(obj, dict):
        res = {k: _stage_in_slab(v, slab, memo) for k, v in obj.items()}
        memo[oid] = res
        return res
    if isinstance(obj, list):
        res = [_stage_in_slab(v, slab, memo) for v in obj]
        memo[oid] = res
        return res
    if isinstance(obj, tuple):
        items = [_stage_in_slab(v, slab, memo) for v in obj]
        res = type(obj)(*items) if hasattr(obj, "_fields") else tuple(items)
        memo[oid] = res
        return res
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        kwargs = {f.name: _stage_in_slab(getattr(obj, f.name), slab, memo) for f in dataclasses.fields(obj)}
        res = type(obj)(**kwargs)
        memo[oid] = res
        return res
    return obj


def pin_batch(obj: Any, _memo: dict[int, Any] | None = None) -> Any:
    """Recursively pin CPU tensors and graph containers in a mini-batch.

    Must be called in the main process (where the CUDA context lives), never in CPU worker
    processes. Already-pinned tensors are returned unchanged.
    """
    memo = _memo if _memo is not None else {}
    oid = id(obj)
    if oid in memo:
        return memo[oid]

    if isinstance(obj, torch.Tensor):
        res = obj.pin_memory() if (obj.device.type == "cpu" and obj.numel() > 0 and not obj.is_pinned()) else obj
        memo[oid] = res
        return res
    if hasattr(obj, "pin_memory") and callable(obj.pin_memory):
        res = obj.pin_memory()
        memo[oid] = res
        return res
    if isinstance(obj, dict):
        res = {k: pin_batch(v, memo) for k, v in obj.items()}
        memo[oid] = res
        return res
    if isinstance(obj, list):
        res = [pin_batch(v, memo) for v in obj]
        memo[oid] = res
        return res
    if isinstance(obj, tuple):
        items = [pin_batch(v, memo) for v in obj]
        res = type(obj)(*items) if hasattr(obj, "_fields") else tuple(items)
        memo[oid] = res
        return res
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        kwargs = {f.name: pin_batch(getattr(obj, f.name), memo) for f in dataclasses.fields(obj)}
        res = type(obj)(**kwargs)
        memo[oid] = res
        return res
    return obj


def transfer_batch_to_device(
    obj: Any, device: torch.device, *, non_blocking: bool = False, _memo: dict[int, Any] | None = None
) -> Any:
    """Recursively move tensors and graph containers in a mini-batch to ``device``."""
    memo = _memo if _memo is not None else {}
    oid = id(obj)
    if oid in memo:
        return memo[oid]

    if isinstance(obj, torch.Tensor):
        res = obj.to(device, non_blocking=non_blocking)
        memo[oid] = res
        return res
    if hasattr(obj, "to") and callable(obj.to):
        try:
            res = obj.to(device, non_blocking=non_blocking)
        except TypeError:
            res = obj.to(device)
        memo[oid] = res
        return res
    if isinstance(obj, dict):
        res = {k: transfer_batch_to_device(v, device, non_blocking=non_blocking, _memo=memo) for k, v in obj.items()}
        memo[oid] = res
        return res
    if isinstance(obj, list):
        res = [transfer_batch_to_device(v, device, non_blocking=non_blocking, _memo=memo) for v in obj]
        memo[oid] = res
        return res
    if isinstance(obj, tuple):
        items = [transfer_batch_to_device(v, device, non_blocking=non_blocking, _memo=memo) for v in obj]
        res = type(obj)(*items) if hasattr(obj, "_fields") else tuple(items)
        memo[oid] = res
        return res
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        kwargs = {
            f.name: transfer_batch_to_device(getattr(obj, f.name), device, non_blocking=non_blocking, _memo=memo)
            for f in dataclasses.fields(obj)
        }
        res = type(obj)(**kwargs)
        memo[oid] = res
        return res
    return obj


def _put_interruptible(q: queue.Queue, item: Any, stop: threading.Event, timeout: float = 0.05) -> bool:
    while not stop.is_set():
        try:
            q.put(item, timeout=timeout)
            return True
        except queue.Full:
            continue
    return False


def _default_worker_threads(num_workers: int) -> int:
    """Choose intra-worker CPU threads to saturate cores without barrier preemption jitter."""
    cpus = os.cpu_count() or 1
    if num_workers <= 0:
        return min(16, cpus)
    if num_workers == 1:
        return min(8, cpus)
    return max(1, min(4, min(16, cpus) // num_workers))


def _cpu_worker_loop(
    worker_id: int,
    produce: Callable[[torch.Tensor], Any],
    task_q: Any,
    result_q: Any,
    active_epoch: Any,
    worker_threads: int,
    base_seed: int | None,
) -> None:
    """Top-level worker entry point (picklable for ``spawn``). Runs purely on CPU."""
    torch.set_num_threads(worker_threads)
    if base_seed is not None:
        torch.manual_seed(int(base_seed) + 1_000_003 * (worker_id + 1))
    else:
        torch.seed()
    if hasattr(produce, "set_worker_seed"):
        produce.set_worker_seed(worker_id, base_seed)

    # Cancel queue join thread on exit so an aborted parent can never hang a worker on shutdown.
    result_q.cancel_join_thread()

    while True:
        task = task_q.get()
        if task is None:
            break
        epoch_id, batch_idx, seeds = task
        if epoch_id != active_epoch.value:
            continue
        try:
            t0 = time.perf_counter()
            batch = produce(seeds)
            produce_ms = (time.perf_counter() - t0) * 1e3
            if epoch_id != active_epoch.value:
                del batch
                continue
            result_q.put((epoch_id, batch_idx, batch, produce_ms, None))
            del batch
        except BaseException as exc:
            err = (type(exc).__name__, str(exc), traceback.format_exc())
            result_q.put((epoch_id, batch_idx, None, 0.0, err))


def _cleanup_workers(workers: list[Any], task_q: Any, result_q: Any, active_epoch: Any) -> None:
    """Shut down a worker pool without hanging even if results remain unread."""
    if active_epoch is not None:
        try:
            active_epoch.value = -1
        except Exception:
            pass
    if task_q is not None:
        for w in workers:
            if w.is_alive():
                try:
                    task_q.put_nowait(None)
                except Exception:
                    pass
    deadline = time.perf_counter() + 3.0
    for w in workers:
        if not w.is_alive():
            continue
        while w.is_alive() and time.perf_counter() < deadline:
            if result_q is not None:
                try:
                    result_q.get_nowait()
                except Exception:
                    pass
            w.join(timeout=0.05)
        if w.is_alive():
            w.terminate()
            w.join(timeout=1.0)
    for q in (task_q, result_q):
        if q is not None:
            try:
                q.cancel_join_thread()
                q.close()
            except Exception:
                pass


class CPUPrefetchLoader:
    """Iterate CPU-sampled mini-batches with optional pinned memory, async H2D, and worker processes.

    Can be used either one-shot (passing ``seeds`` to ``__init__`` and iterating ``for b, t in loader``)
    or reused across epochs via ``loader.epoch(seeds)`` so persistent worker processes are spawned once.

    Args:
        produce: CPU batch builder mapping a seed tensor to a CPU batch (e.g. `CPUBatchMaker`).
            Must be picklable when ``num_workers > 0``.
        seeds: optional default seed batches for ``__iter__`` / ``__len__``.
        num_workers: number of parallel CPU worker processes (``0`` = single-process mode).
        prefetch: in ``num_workers=0`` mode, run preparation on a background thread + side CUDA stream.
            Ignored when ``num_workers > 0`` (parallel processes always prefetch).
        depth: GPU/output prefetch queue depth (batches kept ready ahead of the training loop).
        pin_memory: pin CPU batch tensors in the main process before ``non_blocking=True`` H2D copy
            (effective when targeting a CUDA device).
        device: target device for yielded batches (``"cuda"`` / ``"cuda:i"`` / ``"cpu"``).
        persistent_workers: keep worker processes alive across calls to ``epoch()`` / ``__iter__()``.
        worker_threads: CPU thread count per worker process (sets both ``torch.set_num_threads``
            and ``sampler.num_threads`` for the native C++17 thread pool; also applies to
            ``num_workers=0`` when explicitly provided or when ``sampler.num_threads`` is ``None``;
            auto-chosen if ``None``).
        mp_context: multiprocessing start method (``"spawn"`` default: safe after CUDA init; also
            accepts ``"fork"`` or ``"forkserver"``).
        preserve_order: emit batches in input seed-batch order even when parallel workers finish
            out of order.
        post_transfer: optional hook run in the main process on the H2D stream right after moving
            a batch to ``device`` (defaults to ``produce.post_transfer`` if present).
        seed: optional base RNG seed for worker processes.
    """

    def __init__(
        self,
        produce: Callable[[torch.Tensor], Any],
        seeds: Iterable[torch.Tensor] | None = None,
        *,
        num_workers: int = 0,
        prefetch: bool = True,
        depth: int = 2,
        pin_memory: bool = True,
        device: torch.device | str = "cuda",
        persistent_workers: bool = True,
        worker_threads: int | None = None,
        mp_context: str | None = None,
        preserve_order: bool = True,
        post_transfer: Callable[[Any], Any] | None = None,
        seed: int | None = None,
    ) -> None:
        if num_workers < 0:
            raise ValueError(f"num_workers must be >= 0, got {num_workers}")
        dev = torch.device(device)

        self.produce = produce
        self.seeds = list(seeds) if seeds is not None else None
        self.num_workers = int(num_workers)
        self.prefetch = bool(prefetch)
        self.depth = max(1, int(depth))
        self.device = dev
        self._use_cuda = dev.type == "cuda"
        self.pin_memory = bool(pin_memory and self._use_cuda)
        self.persistent_workers = bool(persistent_workers)
        self.worker_threads = (
            int(worker_threads) if worker_threads is not None else _default_worker_threads(self.num_workers)
        )
        self.mp_context = mp_context or "spawn"
        self.preserve_order = bool(preserve_order)
        self.post_transfer = post_transfer if post_transfer is not None else getattr(produce, "post_transfer", None)
        self.seed = seed

        self._split_worker_gather = hasattr(produce, "for_worker") and hasattr(produce, "finalize_cpu_batch")

        # Keep CPUBatchMaker.pin_memory=False on producer copies: when self.pin_memory=True,
        # CPUPrefetchLoader stages tensors directly into its reusable _PinnedSlabRing instead of
        # triggering per-batch cudaHostAlloc syscalls.
        has_sampler = hasattr(produce, "sampler")
        override_single_threads = has_sampler and (
            worker_threads is not None or getattr(produce.sampler, "num_threads", None) is None
        )
        need_single_copy = self.num_workers == 0 and (
            getattr(produce, "pin_memory", False) or override_single_threads
        )
        if need_single_copy:
            self._single_produce = copy.copy(produce)
            if hasattr(self._single_produce, "pin_memory"):
                self._single_produce.pin_memory = False
            if override_single_threads:
                self._single_produce.sampler = copy.copy(self._single_produce.sampler)
                self._single_produce.sampler.num_threads = self.worker_threads
        else:
            self._single_produce = produce

        # In single-process mode targeting CUDA, use the topology worker + GPU-side block transpose
        # (and direct slab gather when pin_memory=True) to skip CPU CSR transpose and extra copies,
        # sharing self._single_produce.sampler so sampling and feature gather reuse one ThreadPool.
        if self.num_workers == 0 and self._use_cuda and self._split_worker_gather:
            self._single_topo_worker: Any = self._single_produce.for_worker(
                build_cuda_graphs=False, pack_ipc=False, share_sampler=True
            )
        else:
            self._single_topo_worker = None

        self._pinned_ring: _PinnedSlabRing | None = _PinnedSlabRing(self.depth + 2) if self.pin_memory else None
        self._workers: list[Any] = []
        self._task_q: Any = None
        self._result_q: Any = None
        self._active_epoch: Any = None
        self._epoch_id: int = 0
        self._finalizer: weakref.finalize | None = None
        self._active_threads: list[tuple[threading.Event, threading.Thread, queue.Queue]] = []

    def _resolve_device(self) -> torch.device:
        """Resolve bare ``cuda`` to ``cuda:<index>`` lazily (after workers are forked/spawned)."""
        if self._use_cuda and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        return self.device

    def __len__(self) -> int:
        if self.seeds is None:
            raise TypeError("CPUPrefetchLoader has no fixed seeds; pass seeds to epoch(seeds)")
        return len(self.seeds)

    def __enter__(self) -> CPUPrefetchLoader:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _stop_active_threads(self) -> None:
        active = list(self._active_threads)
        for stop, _, _ in active:
            stop.set()
        for _, thread, q in active:
            while thread.is_alive():
                try:
                    q.get(timeout=0.05)
                except queue.Empty:
                    pass
            thread.join()
        self._active_threads.clear()

    def close(self) -> None:
        """Stop active transfer threads, terminate persistent workers, reclaim C++ pools, and close IPC queues."""
        self._stop_active_threads()
        if self._finalizer is not None:
            self._finalizer.detach()
            self._finalizer = None
        if self._workers or self._task_q is not None or self._result_q is not None:
            _cleanup_workers(self._workers, self._task_q, self._result_q, self._active_epoch)
        self._workers = []
        self._task_q = None
        self._result_q = None
        self._active_epoch = None
        if hasattr(self._single_produce, "shutdown_pools"):
            self._single_produce.shutdown_pools()
        if self._split_worker_gather and self.produce is not self._single_produce and hasattr(self.produce, "shutdown_pools"):
            self.produce.shutdown_pools()

    # ------------------------------------------------------------ worker pool
    def _ensure_workers(self) -> None:
        if self._workers and all(w.is_alive() for w in self._workers):
            return
        if self._workers:
            self.close()

        if self.mp_context == "fork" and torch.cuda.is_initialized():
            raise RuntimeError(
                "CPUPrefetchLoader: mp_context='fork' cannot be used safely after CUDA has been initialized "
                "(forking a CUDA/multi-threaded process can deadlock); use mp_context='spawn' or 'forkserver'."
            )
        if self.mp_context == "fork" and hasattr(self.produce, "shutdown_pools"):
            # Shut down any live parent C++ thread pools before os.fork() so children never inherit
            # active mutexes or dead pthread handles across epochs / prior single-process calls.
            self.produce.shutdown_pools()

        self._split_worker_gather = hasattr(self.produce, "for_worker") and hasattr(self.produce, "finalize_cpu_batch")
        if self._split_worker_gather:
            # Always pack worker tensors into 1 contiguous buffer (1 IPC FD instead of ~26).
            # When targeting CUDA, ship forward CSRs only and transpose on the GPU H2D stream.
            worker_produce = self.produce.for_worker(
                build_cuda_graphs=not self._use_cuda,
                pack_ipc=True,
            )
        else:
            worker_produce = self.produce
        # Workers must never touch CUDA, so ensure pin_memory is off on the worker copy,
        # and enforce self.worker_threads on an isolated sampler copy so we never mutate
        # self.produce.sampler or share its parent C++ NativeSamplerState across fork().
        if getattr(worker_produce, "pin_memory", False):
            if worker_produce is self.produce:
                worker_produce = copy.copy(worker_produce)
            worker_produce.pin_memory = False
        if hasattr(worker_produce, "sampler"):
            if worker_produce is self.produce:
                worker_produce = copy.copy(worker_produce)
            if worker_produce.sampler is getattr(self.produce, "sampler", None):
                worker_produce.sampler = copy.copy(worker_produce.sampler)
            worker_produce.sampler.num_threads = self.worker_threads

        if self.mp_context != "fork" and hasattr(worker_produce, "share_memory_"):
            worker_produce.share_memory_()

        ctx = mp.get_context(self.mp_context)
        self._task_q = ctx.Queue()
        self._result_q = ctx.Queue()
        self._active_epoch = ctx.Value("q", self._epoch_id)
        self._workers = []
        try:
            for wid in range(self.num_workers):
                p = ctx.Process(
                    target=_cpu_worker_loop,
                    args=(
                        wid,
                        worker_produce,
                        self._task_q,
                        self._result_q,
                        self._active_epoch,
                        self.worker_threads,
                        self.seed,
                    ),
                    daemon=True,
                )
                p.start()
                self._workers.append(p)
        except BaseException:
            self.close()
            raise
        self._finalizer = weakref.finalize(
            self, _cleanup_workers, self._workers, self._task_q, self._result_q, self._active_epoch
        )

    # ----------------------------------------------------------- H2D transfer
    def _pin_and_transfer(
        self,
        batch_cpu: Any,
        stream: torch.cuda.Stream | None,
        *,
        needs_gather: bool = False,
        finalizer: Any | None = None,
    ) -> tuple[Any, torch.cuda.Event | None, tuple[torch.cuda.Event, torch.cuda.Event] | None]:
        fin = self.produce if finalizer is None else finalizer
        if not self._use_cuda:
            if needs_gather:
                batch_cpu = fin.finalize_cpu_batch(batch_cpu, pin_memory=False)
            batch_out = self.post_transfer(batch_cpu) if self.post_transfer is not None else batch_cpu
            return batch_out, None, None

        slab: _PinnedSlab | None = None
        if self.pin_memory and self._pinned_ring is not None:
            extra = fin.estimate_gather_bytes(batch_cpu) if (needs_gather and hasattr(fin, "estimate_gather_bytes")) else 0
            align_pad = 64 * len(batch_cpu.get("_packed_offsets", ())) if isinstance(batch_cpu, dict) else 0
            req = _estimate_cpu_bytes(batch_cpu) + extra + align_pad + 128
            slab = self._pinned_ring.acquire(req)
            if needs_gather:
                batch_cpu = fin.finalize_cpu_batch(batch_cpu, pin_memory=False, alloc=slab.alloc)
            batch_host = _stage_in_slab(batch_cpu, slab)
        else:
            if needs_gather:
                batch_cpu = fin.finalize_cpu_batch(batch_cpu, pin_memory=False)
            batch_host = batch_cpu

        assert stream is not None
        ready = torch.cuda.Event()
        try:
            with torch.cuda.stream(stream):
                start_ev = torch.cuda.Event(enable_timing=True)
                end_ev = torch.cuda.Event(enable_timing=True)
                start_ev.record(stream)
                batch_dev = transfer_batch_to_device(batch_host, self.device, non_blocking=self.pin_memory)
                if self.post_transfer is not None:
                    batch_dev = self.post_transfer(batch_dev)
                end_ev.record(stream)
        finally:
            # Always record completion on the stream so a partially queued DMA cannot be
            # overwritten if an exception occurs mid-transfer or in post_transfer.
            ready.record(stream)
            if slab is not None:
                slab.ready = ready
        del batch_host
        return batch_dev, ready, (start_ev, end_ev)

    # ----------------------------------------------------------------- inline
    def _iter_inline(self, seed_list: list[torch.Tensor]) -> Iterator[tuple[Any, dict[str, Any]]]:
        dev = self._resolve_device()
        stream = torch.cuda.current_stream(dev) if self._use_cuda else None
        use_topo = self._single_topo_worker is not None
        prod = self._single_topo_worker if use_topo else self._single_produce
        for s in seed_list:
            t0 = time.perf_counter()
            batch_cpu = prod(s)
            t1 = time.perf_counter()
            batch_dev, _, gpu_events = self._pin_and_transfer(
                batch_cpu, stream, needs_gather=use_topo, finalizer=self._single_produce
            )
            del batch_cpu
            t2 = time.perf_counter()
            produce_ms = (t1 - t0) * 1e3
            h2d_ms = (t2 - t1) * 1e3
            timing: dict[str, Any] = {
                "produce_host_ms": produce_ms,
                "h2d_host_ms": h2d_ms,
                "wait_host_ms": produce_ms + h2d_ms,
            }
            if gpu_events is not None:
                timing["_gpu_events"] = gpu_events
            yield batch_dev, timing

    # ------------------------------------------------------- threaded (0 wkr)
    def _iter_threaded(self, seed_list: list[torch.Tensor]) -> Iterator[tuple[Any, dict[str, Any]]]:
        if not seed_list:
            return
        dev = self._resolve_device()
        h2d_stream = torch.cuda.Stream(device=dev) if self._use_cuda else None
        compute = torch.cuda.current_stream(dev) if self._use_cuda else None
        out: queue.Queue = queue.Queue(maxsize=self.depth)
        stop = threading.Event()
        use_topo = self._single_topo_worker is not None
        prod = self._single_topo_worker if use_topo else self._single_produce

        def worker() -> None:
            try:
                if self._use_cuda:
                    torch.cuda.set_device(dev)
                for s in seed_list:
                    if stop.is_set():
                        break
                    t0 = time.perf_counter()
                    batch_cpu = prod(s)
                    t1 = time.perf_counter()
                    batch_dev, ready, gpu_events = self._pin_and_transfer(
                        batch_cpu, h2d_stream, needs_gather=use_topo, finalizer=self._single_produce
                    )
                    del batch_cpu
                    t2 = time.perf_counter()
                    timing: dict[str, Any] = {
                        "produce_host_ms": (t1 - t0) * 1e3,
                        "h2d_host_ms": (t2 - t1) * 1e3,
                    }
                    if gpu_events is not None:
                        timing["_gpu_events"] = gpu_events
                    if not _put_interruptible(out, (batch_dev, ready, timing), stop):
                        break
            except BaseException as exc:
                _put_interruptible(out, exc, stop)
            finally:
                _put_interruptible(out, _SENTINEL, stop)

        thread = threading.Thread(target=worker, daemon=True)
        entry = (stop, thread, out)
        self._active_threads.append(entry)
        thread.start()
        yielded = 0
        try:
            while True:
                t0 = time.perf_counter()
                item = out.get()
                wait_ms = (time.perf_counter() - t0) * 1e3
                if item is _SENTINEL:
                    break
                if isinstance(item, BaseException):
                    raise item
                batch_dev, ready, timing = item
                if ready is not None and compute is not None:
                    compute.wait_event(ready)
                    for t in iter_tensors(batch_dev):
                        if t.is_cuda:
                            t.record_stream(compute)
                timing["wait_host_ms"] = wait_ms
                yielded += 1
                yield batch_dev, timing
        finally:
            stop.set()
            while thread.is_alive():
                try:
                    out.get(timeout=0.05)
                except queue.Empty:
                    pass
            thread.join()
            if entry in self._active_threads:
                self._active_threads.remove(entry)
            if h2d_stream is not None and yielded < len(seed_list):
                h2d_stream.synchronize()

    # ---------------------------------------------------- multiprocess (>= 1)
    def _iter_multiprocess(self, seed_list: list[torch.Tensor]) -> Iterator[tuple[Any, dict[str, Any]]]:
        total = len(seed_list)
        if total == 0:
            return
        # Spawn/fork workers BEFORE resolving bare "cuda" (so fork works if caller hasn't initialized CUDA yet).
        self._ensure_workers()
        self._epoch_id += 1
        epoch_id = self._epoch_id
        self._active_epoch.value = epoch_id
        task_q = self._task_q
        result_q = self._result_q
        workers = self._workers

        dev = self._resolve_device()
        h2d_stream = torch.cuda.Stream(device=dev) if self._use_cuda else None
        compute = torch.cuda.current_stream(dev) if self._use_cuda else None
        gpu_out: queue.Queue = queue.Queue(maxsize=self.depth)
        stop = threading.Event()
        max_in_flight = max(self.num_workers * self.depth, self.num_workers + 1)

        def transfer_loop() -> None:
            try:
                if self._use_cuda:
                    torch.cuda.set_device(dev)
                submitted = 0
                received = 0
                emitted = 0
                next_to_emit = 0
                pending: dict[int, tuple[Any, float]] = {}

                def dispatch() -> None:
                    nonlocal submitted
                    # Bound total un-emitted batches (in workers + reorder buffer) so memory
                    # and shared-memory FDs stay bounded even if one worker straggles.
                    while submitted < total and (submitted - emitted) < max_in_flight and not stop.is_set():
                        # Clone seed slice so IPC carries only [B] ids, not the whole epoch split storage.
                        s = seed_list[submitted].detach().to(device="cpu").clone()
                        task_q.put((epoch_id, submitted, s))
                        submitted += 1

                def emit(batch_cpu: Any, produce_ms: float) -> bool:
                    nonlocal emitted
                    # Refill the worker window before gathering/pinning/copying or waiting on gpu_out so
                    # worker processes keep sampling while H2D transfer or training runs.
                    emitted += 1
                    dispatch()
                    t_xfer = time.perf_counter()
                    batch_dev, ready, gpu_events = self._pin_and_transfer(
                        batch_cpu, h2d_stream, needs_gather=self._split_worker_gather
                    )
                    del batch_cpu
                    timing: dict[str, Any] = {
                        "produce_host_ms": produce_ms,
                        "h2d_host_ms": (time.perf_counter() - t_xfer) * 1e3,
                    }
                    if gpu_events is not None:
                        timing["_gpu_events"] = gpu_events
                    return _put_interruptible(gpu_out, (batch_dev, ready, timing), stop)

                dispatch()
                while received < total and not stop.is_set():
                    try:
                        res_epoch, batch_idx, batch_cpu, produce_ms, err = result_q.get(timeout=0.2)
                    except queue.Empty:
                        dead = [w for w in workers if not w.is_alive() and w.exitcode not in (0, None)]
                        if dead:
                            raise RuntimeError(
                                f"CPU sampling worker (pid={dead[0].pid}) exited unexpectedly with code {dead[0].exitcode}"
                            )
                        continue

                    if res_epoch != epoch_id:
                        del batch_cpu
                        continue

                    received += 1
                    if err is not None:
                        exc_type, exc_msg, exc_tb = err
                        raise RuntimeError(f"CPU sampling worker failed with {exc_type}: {exc_msg}\n{exc_tb}")

                    if not self.preserve_order:
                        if not emit(batch_cpu, produce_ms):
                            break
                    else:
                        pending[batch_idx] = (batch_cpu, produce_ms)
                        while next_to_emit in pending and not stop.is_set():
                            b_cpu, p_ms = pending.pop(next_to_emit)
                            next_to_emit += 1
                            if not emit(b_cpu, p_ms):
                                break
            except BaseException as exc:
                _put_interruptible(gpu_out, exc, stop)
            finally:
                _put_interruptible(gpu_out, _SENTINEL, stop)

        transfer = threading.Thread(target=transfer_loop, daemon=True)
        entry = (stop, transfer, gpu_out)
        self._active_threads.append(entry)
        transfer.start()
        yielded = 0
        try:
            while True:
                t0 = time.perf_counter()
                item = gpu_out.get()
                wait_ms = (time.perf_counter() - t0) * 1e3
                if item is _SENTINEL:
                    break
                if isinstance(item, BaseException):
                    raise item
                batch_dev, ready, timing = item
                if ready is not None and compute is not None:
                    compute.wait_event(ready)
                    for t in iter_tensors(batch_dev):
                        if t.is_cuda:
                            t.record_stream(compute)
                timing["wait_host_ms"] = wait_ms
                yielded += 1
                yield batch_dev, timing
        finally:
            stop.set()
            if self._active_epoch is not None and yielded < total:
                self._epoch_id += 1
                self._active_epoch.value = self._epoch_id
            while transfer.is_alive():
                try:
                    gpu_out.get(timeout=0.05)
                except queue.Empty:
                    pass
            transfer.join()
            if entry in self._active_threads:
                self._active_threads.remove(entry)
            if h2d_stream is not None and yielded < total:
                h2d_stream.synchronize()
            if not self.persistent_workers:
                self.close()

    # -------------------------------------------------------------- public API
    def epoch(self, seeds: Iterable[torch.Tensor]) -> Iterator[tuple[Any, dict[str, Any]]]:
        """Iterate one epoch of ``seeds`` through this loader (reusing persistent workers)."""
        seed_list = list(seeds)
        if self.num_workers > 0:
            return self._iter_multiprocess(seed_list)
        if self.prefetch:
            return self._iter_threaded(seed_list)
        return self._iter_inline(seed_list)

    def __iter__(self) -> Iterator[tuple[Any, dict[str, Any]]]:
        if self.seeds is None:
            raise TypeError("CPUPrefetchLoader has no default seeds; call loader.epoch(seeds)")
        return self.epoch(self.seeds)
