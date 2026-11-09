"""Overlap batch preparation with training: producer thread + side CUDA stream.

Batch preparation (sampling, feature/label gather, building the kernels' graph objects) has
a few host syncs -- sizes of ``unique`` / masked selects. Run inline, every one of them stalls
the Python thread that launches the training kernels. With ``prefetch=True`` a producer
thread prepares batch i+1..i+depth on its own stream while batch i trains:

- syncs inside the producer wait for the *side* stream only (``.item()`` synchronises the
  current stream), so they block the producer thread, not the compute stream;
- the hand-off is a CUDA event recorded on the side stream after the batch is built; the
  consumer makes the compute stream wait on it -- no host sync;
- every tensor of the batch is ``record_stream``-ed on the compute stream, so the caching
  allocator does not hand its memory back to the producer while compute still reads it.

Timing per batch (all milliseconds):

- ``produce_host_ms``: wall time the producer spent building the batch (hidden when prefetching);
- ``produce_gpu_ms``: GPU time of the side-stream work (events around ``produce``);
- ``wait_host_ms``: time the training loop blocked waiting for the queue -- the **exposed**
  data-loading cost; with ``prefetch=False`` it is the whole inline preparation.
"""

from __future__ import annotations

import dataclasses
import queue
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from typing import Any

import torch

_SENTINEL = object()


def iter_tensors(obj: Any, _seen: set[int] | None = None) -> Iterator[torch.Tensor]:
    """Yield every tensor reachable through dataclasses, lists/tuples, dicts and plain objects."""
    seen = _seen if _seen is not None else set()
    if id(obj) in seen:
        return
    seen.add(id(obj))
    if isinstance(obj, torch.Tensor):
        yield obj
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            yield from iter_tensors(item, seen)
    elif isinstance(obj, dict):
        for item in obj.values():
            yield from iter_tensors(item, seen)
    elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        # vars() rather than fields(): also reaches attributes set in __post_init__ / caches.
        for item in vars(obj).values():
            yield from iter_tensors(item, seen)


class PrefetchLoader:
    """Iterate prepared batches, optionally prepared ahead on a side stream.

    Args:
        produce: builds one batch from a seed tensor, issuing its GPU work on the current stream.
        seeds: the epoch's seed batches.
        prefetch: run ``produce`` in a background thread on a dedicated stream.
        depth: how many batches may be prepared ahead.
        device: CUDA device.
    """

    def __init__(
        self,
        produce: Callable[[torch.Tensor], Any],
        seeds: Iterable[torch.Tensor],
        *,
        prefetch: bool = True,
        depth: int = 2,
        device: torch.device | str = "cuda",
    ) -> None:
        self.produce = produce
        self.seeds = list(seeds)
        self.prefetch = prefetch
        self.depth = max(1, depth)
        self.device = torch.device(device)

    def __len__(self) -> int:
        return len(self.seeds)

    # --------------------------------------------------------------- producing
    def _produced(self, stream: torch.cuda.Stream) -> Iterator[tuple[Any, dict[str, Any]]]:
        """Yield (batch, timing) in seed order, issuing GPU work on `stream`.

        A maker with ``launch``/``finalize`` (the sync-free sampler) is software-pipelined:
        batch i+1 is launched before batch i is finalized, so the sizes finalize waits for
        are normally already on the host when it polls for them.
        """
        two_stage = hasattr(self.produce, "launch") and hasattr(self.produce, "finalize")
        if not two_stage:
            for s in self.seeds:
                t0 = time.perf_counter()
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record(stream)
                batch = self.produce(s)
                end.record(stream)
                yield batch, {"produce_host_ms": (time.perf_counter() - t0) * 1e3, "_gpu_events": (start, end)}
            return

        def launch(s):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            t0 = time.perf_counter()
            start.record(stream)
            pending = self.produce.launch(s)
            end.record(stream)
            return pending, (start, end), (time.perf_counter() - t0) * 1e3

        prev = None
        for s in self.seeds:
            cur = launch(s)
            if prev is not None:
                yield self._finish(prev, stream)
            prev = cur
        if prev is not None:
            yield self._finish(prev, stream)

    def _finish(self, launched, stream) -> tuple[Any, dict[str, Any]]:
        pending, launch_events, launch_ms = launched
        t0 = time.perf_counter()
        end = torch.cuda.Event(enable_timing=True)
        batch = self.produce.finalize(pending)
        end.record(stream)
        host_ms = launch_ms + (time.perf_counter() - t0) * 1e3
        # GPU time: from the launch start to the end of the finalize work (gather, graph views).
        return batch, {"produce_host_ms": host_ms, "_gpu_events": (launch_events[0], end)}

    # ------------------------------------------------------------------ inline
    def _iter_inline(self) -> Iterator[tuple[Any, dict[str, float]]]:
        stream = torch.cuda.current_stream(self.device)
        for batch, timing in self._produced(stream):
            timing["wait_host_ms"] = timing["produce_host_ms"]
            yield batch, timing

    # ---------------------------------------------------------------- threaded
    def _producer(self, side: torch.cuda.Stream, out: queue.Queue, stop: threading.Event) -> None:
        try:
            torch.cuda.set_device(self.device)
            with torch.cuda.stream(side):
                for batch, timing in self._produced(side):
                    if stop.is_set():
                        break
                    ready = torch.cuda.Event()
                    ready.record(side)
                    out.put((batch, ready, timing))
        except BaseException as exc:  # surfaced in the consumer
            out.put(exc)
        finally:
            out.put(_SENTINEL)

    def _iter_prefetch(self) -> Iterator[tuple[Any, dict[str, float]]]:
        side = torch.cuda.Stream(device=self.device)
        compute = torch.cuda.current_stream(self.device)
        # The seed batches (randperm + index) were produced on the compute stream: the producer's
        # stream must not read them before that work has run.
        side.wait_stream(compute)
        out: queue.Queue = queue.Queue(maxsize=self.depth)
        stop = threading.Event()
        worker = threading.Thread(target=self._producer, args=(side, out, stop), daemon=True)
        worker.start()
        try:
            while True:
                t0 = time.perf_counter()
                item = out.get()
                wait_ms = (time.perf_counter() - t0) * 1e3
                if item is _SENTINEL:
                    break
                if isinstance(item, BaseException):
                    raise item
                batch, ready, timing = item
                compute.wait_event(ready)
                for t in iter_tensors(batch):
                    if t.is_cuda:
                        t.record_stream(compute)
                timing["wait_host_ms"] = wait_ms
                yield batch, timing
        finally:
            stop.set()
            # Drain so a producer blocked on a full queue can exit.
            while worker.is_alive():
                try:
                    out.get(timeout=0.1)
                except queue.Empty:
                    pass
            worker.join()

    def __iter__(self) -> Iterator[tuple[Any, dict[str, float]]]:
        return self._iter_prefetch() if self.prefetch else self._iter_inline()


def resolve_gpu_ms(timing: dict[str, Any]) -> dict[str, float]:
    """Turn the recorded event pair into ``produce_gpu_ms`` (call after a sync)."""
    out = {k: v for k, v in timing.items() if not k.startswith("_")}
    events = timing.get("_gpu_events")
    if events is not None:
        out["produce_gpu_ms"] = events[0].elapsed_time(events[1])
    return out
