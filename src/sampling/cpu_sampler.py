"""Uniform neighbor sampler and mini-batch builder on the CPU, in plain torch.

Mirrors `GPUNeighborSampler` (`src/sampling/neighbor_sampler.py`) and `OursBatchMaker`
(`scripts/train_sampled.py`) for graphs whose CSR, node features and labels reside in host
RAM rather than GPU VRAM:

- one block per layer, outermost (input) layer first;
- a block's destination nodes are the first ``num_dst`` of its source nodes, so
  ``x_dst = x_src[:num_dst]`` and layer ``i``'s destinations are layer ``i+1``'s sources;
- ``indptr [num_dst + 1]`` / ``indices [E]`` is the destination-row CSR whose column ids are
  local source ids in ``[0, num_src)``;
- ``fanouts[i]`` is the fanout of block ``i`` (outermost first), with ``-1`` taking all
  in-neighbors.

Key CPU-specific choices:

- **No O(N) scratch buffers**: frontier deduplication and relabeling run in
  ``O((n + E) log(n + E))`` using ``unique`` + ``searchsorted`` instead of scanning an
  ``N``-element mark array, so workers stay stateless and use zero graph-sized scratch RAM
  even on 100M-node graphs.
- **Hub-safe sampling without replacement**: over-full rows (``deg > fanout``) use vectorized
  Floyd sampling in ``O(H * fanout^2)`` rather than expanding and sorting all ``deg`` edges of
  high-degree hubs.
- **Zero-copy multiprocessing**: ``CPUBatchMaker`` is plain picklable Python/torch; calling
  ``share_memory_()`` puts the read-only graph and feature tensors in shared memory so
  parallel worker processes share a single copy without touching CUDA.
"""

from __future__ import annotations

import os
import time
from typing import Any

import torch

from src.sampling._cpu_sampler_cpp import NativeSamplerState, native_gather_rows
from src.sampling.neighbor_sampler import SampledBatch, SampledBlock


def _ensure_cpu(t: torch.Tensor, name: str) -> None:
    if t.device.type != "cpu":
        raise ValueError(f"{name} must be a CPU tensor, got {t.device}")


def _default_gather_threads() -> int:
    return min(16, max(1, os.cpu_count() or 1))


def _pack_tensors(tensors: list[torch.Tensor]) -> tuple[torch.Tensor, list[tuple[int, int, torch.dtype]]]:
    """Pack multiple 1-D CPU tensors into one contiguous uint8 buffer (1 IPC FD instead of N)."""
    offsets: list[tuple[int, int, torch.dtype]] = []
    pos = 0
    for t in tensors:
        nbytes = t.numel() * t.element_size()
        pos = (pos + 7) & ~7
        offsets.append((pos, nbytes, t.dtype))
        pos += nbytes
    buf = torch.empty(pos, dtype=torch.uint8)
    for t, (start, nbytes, _) in zip(tensors, offsets):
        if nbytes > 0:
            tc = t if t.is_contiguous() else t.contiguous()
            buf[start : start + nbytes].copy_(tc.view(torch.uint8))
    return buf, offsets


def _unpack_tensors(buf: torch.Tensor, offsets: list[tuple[int, int, torch.dtype]]) -> list[torch.Tensor]:
    out = []
    for start, nbytes, dt in offsets:
        if nbytes == 0:
            out.append(torch.empty(0, dtype=dt))
        else:
            out.append(buf[start : start + nbytes].view(dt))
    return out


class CPUNeighborSampler:
    """Samples multi-hop blocks from an in-neighbor CSR held on the CPU.

    Uses a C++17 + OpenMP native backend (`src/sampling/_cpu_sampler.cpp`) when a C++
    compiler is available, with an automatic pure-PyTorch fallback.

    Args:
        indptr: ``[N + 1]`` CPU in-CSR row pointers (row v lists the sources of edges u->v).
        indices: ``[E]`` CPU source ids (kept in its original compact dtype, e.g. int32).
        fanouts: per-layer fanouts, outermost layer first; ``-1`` = all neighbors.
        index_dtype: dtype of the emitted block CSR (int32 is fastest for our CUDA kernels).
        generator: optional CPU ``torch.Generator`` for reproducible sampling.
        num_threads: optional OpenMP thread count for the native CPU backend (auto if ``None``).
    """

    def __init__(
        self,
        indptr: torch.Tensor,
        indices: torch.Tensor,
        fanouts: list[int],
        index_dtype: torch.dtype = torch.int32,
        generator: torch.Generator | None = None,
        num_threads: int | None = None,
    ) -> None:
        _ensure_cpu(indptr, "indptr")
        _ensure_cpu(indices, "indices")
        if any(f == 0 or f < -1 for f in fanouts):
            raise ValueError(f"fanouts must be positive or -1 (all neighbors), got {fanouts}")
        if generator is not None and generator.device.type != "cpu":
            raise ValueError(f"generator must be on CPU, got {generator.device}")

        self.indptr = (indptr if indptr.dtype == torch.int64 else indptr.to(torch.int64)).contiguous()
        self.indices = indices.contiguous()
        self.fanouts = [int(f) for f in fanouts]
        self.index_dtype = index_dtype
        self.generator = generator
        self.num_threads = int(num_threads) if num_threads is not None else None
        self.num_nodes = indptr.numel() - 1
        self._native: NativeSamplerState | None = None
        self._native_checked: bool = False
        self._rng_counter: int = 0

    @property
    def device(self) -> torch.device:
        return self.indptr.device

    def __copy__(self) -> CPUNeighborSampler:
        # Shallow copies must never share a mutable C++ CpuSamplerState*.
        dup = object.__new__(type(self))
        dup.__dict__.update(self.__dict__)
        dup._native = None
        dup._native_checked = False
        return dup

    def __getstate__(self) -> dict[str, Any]:
        # Drop non-picklable C++ pointer and torch.Generator; workers recreate them lazily.
        state = self.__dict__.copy()
        state["generator"] = None
        state["_native"] = None
        state["_native_checked"] = False
        return state

    def _effective_threads(self) -> int:
        if self.num_threads is not None:
            return max(1, self.num_threads)
        return min(16, max(1, torch.get_num_threads()))

    def _get_native(self) -> NativeSamplerState | None:
        if not self._native_checked:
            self._native_checked = True
            if (
                self.indices.dtype in (torch.int32, torch.int64)
                and self.index_dtype in (torch.int32, torch.int64)
                and self.indptr.is_contiguous()
                and self.indices.is_contiguous()
            ):
                st = NativeSamplerState(self.num_nodes)
                if st.available:
                    self._native = st
        return self._native

    def _next_seed(self) -> int:
        if self.generator is not None:
            return int(torch.randint(0, 2**62, (1,), generator=self.generator).item())
        self._rng_counter += 1
        return (time.perf_counter_ns() ^ (os.getpid() * 0x9E3779B97F4A7C15) ^ (self._rng_counter * 0xBF58476D1CE4E5B9)) & 0xFFFFFFFFFFFFFFFF

    def share_memory_(self) -> CPUNeighborSampler:
        """Move read-only CSR arrays into shared memory for zero-copy worker processes."""
        if not self.indptr.is_contiguous():
            self.indptr = self.indptr.contiguous()
        if not self.indices.is_contiguous():
            self.indices = self.indices.contiguous()
        if not self.indptr.is_shared():
            self.indptr.share_memory_()
        if not self.indices.is_shared():
            self.indices.share_memory_()
        return self

    def _floyd(self, deg: torch.Tensor, f: int) -> torch.Tensor:
        """Sample ``f`` distinct positions in ``[0, deg[i])`` for each over-full row ``i``.

        Exact uniform sampling without replacement via Floyd's algorithm, vectorized over all
        over-full rows ``[H, f]`` in ``O(H * f^2)`` time independent of hub degrees.
        """
        h = deg.numel()
        picks = torch.empty((h, f), dtype=torch.int64)
        for it in range(f):
            j = deg - f + it  # j runs through [d - f, d)
            u = torch.rand(h, dtype=torch.float64, generator=self.generator)
            t = torch.minimum((u * (j + 1).to(torch.float64)).to(torch.int64), j)
            if it:
                taken = (picks[:, :it] == t[:, None]).any(dim=1)
                t = torch.where(taken, j, t)
            picks[:, it] = t
        return picks

    def _sample_edges(self, dst: torch.Tensor, fanout: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Pick in-edges of each node in ``dst``. Returns ``(counts [n], global src ids [E'])``."""
        n = dst.numel()
        if n == 0:
            return torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64)

        start = self.indptr[dst]
        deg = self.indptr[dst + 1] - start
        counts = deg if fanout < 0 else deg.clamp(max=fanout)
        total = int(counts.sum().item())
        if total == 0:
            return counts, torch.empty(0, dtype=torch.int64)

        row_start = torch.cumsum(counts, 0) - counts
        row = torch.repeat_interleave(torch.arange(n, dtype=torch.int64), counts, output_size=total)
        edge_pos = start[row] + (torch.arange(total, dtype=torch.int64) - row_start[row])

        if fanout > 0:
            big_idx = (deg > fanout).nonzero(as_tuple=False).view(-1)
            if big_idx.numel() > 0:
                picks = self._floyd(deg[big_idx], fanout)
                sampled_pos = start[big_idx][:, None] + picks
                slots = row_start[big_idx][:, None] + torch.arange(fanout, dtype=torch.int64)[None, :]
                edge_pos[slots.reshape(-1)] = sampled_pos.reshape(-1)

        src = self.indices[edge_pos]
        if src.dtype != torch.int64:
            src = src.to(torch.int64)
        return counts, src

    def _relabel_frontier(self, dst: torch.Tensor, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Assign local ids: ``dst`` first (``0..n-1``), then new sources in ascending id order.

        Stateless ``O((n + E) log(n + E))`` dedup with zero ``O(N)`` scratch memory.
        Returns ``(src_nodes [num_src] int64, col [E] index_dtype)``.
        """
        n = dst.numel()
        if src.numel() == 0:
            return dst, torch.empty(0, dtype=self.index_dtype)

        u_src, inv = torch.unique(src, sorted=True, return_inverse=True)
        if n == 0:
            return u_src, inv.to(self.index_dtype)

        dst_sorted, dst_order = torch.sort(dst)
        pos = torch.searchsorted(dst_sorted, u_src)
        in_dst = (pos < n) & (dst_sorted[pos.clamp(max=n - 1)] == u_src)
        new = u_src[~in_dst]

        u_local = torch.empty(u_src.numel(), dtype=self.index_dtype)
        u_local[in_dst] = dst_order[pos[in_dst]].to(self.index_dtype)
        u_local[~in_dst] = torch.arange(n, n + new.numel(), dtype=self.index_dtype)
        col = u_local[inv]
        src_nodes = torch.cat([dst, new])
        return src_nodes, col

    def sample(self, seeds: torch.Tensor) -> SampledBatch:
        """Sample one multi-hop mini-batch for ``seeds`` on the CPU."""
        seeds_cpu = seeds.to(device="cpu", dtype=torch.int64).contiguous()
        native = self._get_native()
        if native is not None:
            dst = seeds_cpu
            threads = self._effective_threads()
            blocks: list[SampledBlock] = []
            for fanout in reversed(self.fanouts):
                indptr, col, src_nodes = native.sample_layer(
                    self.indptr,
                    self.indices,
                    dst,
                    fanout,
                    self.index_dtype,
                    self._next_seed(),
                    threads,
                )
                blocks.append(
                    SampledBlock(
                        indptr=indptr,
                        indices=col,
                        num_src=src_nodes.numel(),
                        fanout=fanout,
                    )
                )
                dst = src_nodes
            blocks.reverse()
            return SampledBatch(seeds=seeds_cpu, input_nodes=dst, blocks=blocks)

        dst = seeds_cpu
        blocks = []
        for fanout in reversed(self.fanouts):
            n = dst.numel()
            counts, src = self._sample_edges(dst, fanout)
            src_nodes, col = self._relabel_frontier(dst, src)

            indptr = torch.zeros(n + 1, dtype=self.index_dtype)
            if n > 0:
                indptr[1:] = torch.cumsum(counts, 0).to(self.index_dtype)

            blocks.append(
                SampledBlock(
                    indptr=indptr,
                    indices=col,
                    num_src=src_nodes.numel(),
                    fanout=fanout,
                )
            )
            dst = src_nodes

        blocks.reverse()
        return SampledBatch(seeds=seeds_cpu, input_nodes=dst, blocks=blocks)

    def build_cuda_graph(
        self,
        blk: SampledBlock,
        *,
        fwd_heavy_threshold: int | None = None,
        bwd_heavy_threshold: int | None = None,
        quantile: float | None = None,
        fwd_order: str = "natural",
        bwd_order: str = "natural",
    ):
        """Build a CPU `AdjacencyForwardBackwardWithNodeBuckets` for a sampled block."""
        from turbo_gnn.graph import AdjacencyForwardBackwardWithNodeBuckets as G

        for name, mode in (("fwd_order", fwd_order), ("bwd_order", bwd_order)):
            if mode not in ("natural", "degree"):
                raise ValueError(f"{name} must be 'natural' or 'degree', got {mode!r}")

        native = self._get_native() if quantile is None else None
        if (
            native is not None
            and blk.indptr.device.type == "cpu"
            and blk.indices.device.type == "cpu"
            and blk.indptr.dtype == blk.indices.dtype
            and blk.indptr.dtype in (torch.int32, torch.int64)
        ):
            fip = blk.indptr if blk.indptr.is_contiguous() else blk.indptr.contiguous()
            fix = blk.indices if blk.indices.is_contiguous() else blk.indices.contiguous()
            bwd_indptr, bwd_indices, fwd_light, fwd_heavy, bwd_light, bwd_heavy = native.transpose_and_bucket(
                fip,
                fix,
                blk.num_src,
                fwd_threshold=fwd_heavy_threshold,
                bwd_threshold=bwd_heavy_threshold,
                fwd_order=fwd_order,
                bwd_order=bwd_order,
                num_threads=self._effective_threads(),
            )
            return G(
                forward_indptr=blk.indptr,
                forward_indices=blk.indices,
                backward_indptr=bwd_indptr,
                backward_indices=bwd_indices,
                forward_light_nodes=fwd_light,
                forward_heavy_nodes=fwd_heavy,
                backward_light_nodes=bwd_light,
                backward_heavy_nodes=bwd_heavy,
                max_degree=blk.fanout if blk.fanout > 0 else -1,
                backward_max_degree=blk.num_dst,
                is_directed=True,
            )

        return G.from_block_csr(
            blk.indptr,
            blk.indices,
            blk.num_src,
            max_in_degree=blk.fanout if blk.fanout > 0 else None,
            forward_heavy_threshold=fwd_heavy_threshold,
            backward_heavy_threshold=bwd_heavy_threshold,
            quantile=quantile,
            forward_order=fwd_order,
            backward_order=bwd_order,
        )


class _CPUTopologyWorker:
    """Lightweight worker callable for `CPUBatchMaker`: holds CSR topology only, never ``x``/``y``.

    Parallel worker processes sample neighbors and build per-layer block CSRs + buckets on CPU,
    returning compact index tensors (a few MB) instead of shipping large gathered feature
    matrices through ``/dev/shm``. The main process then gathers ``x[input_nodes]`` and
    ``y[seeds]`` directly into pinned host memory in one pass before async H2D DMA.
    """

    def __init__(
        self,
        sampler: CPUNeighborSampler,
        *,
        backend: str,
        fwd_heavy_threshold: int | None,
        bwd_heavy_threshold: int | None,
        quantile: float | None,
        fwd_order: str,
        bwd_order: str,
        heavy_edge_slices: dict[str, int],
        seed: int | None,
        build_cuda_graphs: bool = True,
        pack_ipc: bool = False,
    ) -> None:
        self.sampler = sampler
        self.backend = backend
        self.fwd_heavy_threshold = fwd_heavy_threshold
        self.bwd_heavy_threshold = bwd_heavy_threshold
        self.quantile = quantile
        self.fwd_order = fwd_order
        self.bwd_order = bwd_order
        self.heavy_edge_slices = heavy_edge_slices
        self.seed = seed
        self.build_cuda_graphs = build_cuda_graphs
        self.pack_ipc = pack_ipc

    def share_memory_(self) -> _CPUTopologyWorker:
        self.sampler.share_memory_()
        return self

    def set_worker_seed(self, worker_id: int, base_seed: int | None = None) -> None:
        gen = torch.Generator(device="cpu")
        if base_seed is not None:
            gen.manual_seed(int(base_seed) + 1_000_003 * (worker_id + 1))
        elif self.seed is not None:
            gen.manual_seed(int(self.seed) + 1_000_003 * (worker_id + 1))
        else:
            gen.seed()
        self.sampler.generator = gen

    def _cuda_graph_for(self, blk: SampledBlock):
        g = self.sampler.build_cuda_graph(
            blk,
            fwd_heavy_threshold=self.fwd_heavy_threshold,
            bwd_heavy_threshold=self.bwd_heavy_threshold,
            quantile=self.quantile,
            fwd_order=self.fwd_order,
            bwd_order=self.bwd_order,
        )
        for direction, size in self.heavy_edge_slices.items():
            g.heavy_edge_slices(direction, size)
        return g

    def __call__(self, seeds: torch.Tensor) -> dict[str, Any]:
        batch = self.sampler.sample(seeds)
        stats = [(b.num_dst, b.num_src, b.num_edges) for b in batch.blocks]
        if self.backend == "cuda" and self.build_cuda_graphs:
            graphs = [self._cuda_graph_for(b) for b in batch.blocks]
            return {
                "_input_nodes": batch.input_nodes,
                "_seeds": batch.seeds,
                "graphs": graphs,
                "stats": stats,
            }
        # Raw forward-CSR blocks (transposed on GPU on the H2D stream, or wrapped into DGL blocks).
        if self.pack_ipc:
            tensors = [batch.input_nodes, batch.seeds]
            block_meta = []
            for b in batch.blocks:
                tensors.append(b.indptr)
                tensors.append(b.indices)
                block_meta.append((b.num_src, b.fanout))
            packed, offsets = _pack_tensors(tensors)
            return {
                "_packed_buf": packed,
                "_packed_offsets": offsets,
                "_packed_block_meta": block_meta,
                "graphs": None,
                "stats": stats,
            }
        return {
            "_input_nodes": batch.input_nodes,
            "_seeds": batch.seeds,
            "graphs": None,
            "stats": stats,
            "_raw_blocks": batch.blocks,
        }


class CPUBatchMaker:
    """CPU mini-batch producer: samples neighbors, gathers features/labels, builds block graphs.

    Matches `OursBatchMaker`'s output dict schema (``x``, ``y``, ``graphs``, ``stats``) while
    executing entirely on CPU so it can run either inline/threaded in the main process or
    inside pure-CPU ``torch.multiprocessing`` worker processes (no CUDA calls in workers).

    When ``pin_memory=True`` (single-process mode only; never enable inside worker processes),
    features, labels and block tensors are placed directly in page-locked host memory to avoid
    an extra host copy before async H2D transfer.
    """

    def __init__(
        self,
        indptr: torch.Tensor,
        indices: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        fanouts: list[int],
        *,
        backend: str = "cuda",
        index_dtype: torch.dtype = torch.int32,
        x_dtype: torch.dtype | None = None,
        fwd_heavy_threshold: int | None = None,
        bwd_heavy_threshold: int | None = None,
        quantile: float | None = None,
        fwd_order: str = "natural",
        bwd_order: str = "natural",
        heavy_edge_slices: dict[str, int] | None = None,
        pin_memory: bool = False,
        seed: int | None = None,
    ) -> None:
        if backend not in ("cuda", "dgl"):
            raise ValueError(f"backend must be 'cuda' or 'dgl', got {backend!r}")
        for name, mode in (("fwd_order", fwd_order), ("bwd_order", bwd_order)):
            if mode not in ("natural", "degree"):
                raise ValueError(f"{name} must be 'natural' or 'degree', got {mode!r}")
        _ensure_cpu(x, "x")
        _ensure_cpu(y, "y")

        gen = None
        if seed is not None:
            gen = torch.Generator(device="cpu")
            gen.manual_seed(int(seed))

        self.sampler = CPUNeighborSampler(indptr, indices, fanouts, index_dtype=index_dtype, generator=gen)
        self.x = x.contiguous()
        self.y = y.contiguous()
        self.backend = backend
        self.x_dtype = x_dtype
        self.fwd_heavy_threshold = fwd_heavy_threshold
        self.bwd_heavy_threshold = bwd_heavy_threshold
        self.quantile = quantile
        self.fwd_order = fwd_order
        self.bwd_order = bwd_order
        self.heavy_edge_slices = {k: int(v) for k, v in (heavy_edge_slices or {}).items() if int(v) > 0}
        self.pin_memory = pin_memory
        self.seed = seed

    def share_memory_(self) -> CPUBatchMaker:
        """Share the read-only graph, features and labels across spawned worker processes."""
        self.sampler.share_memory_()
        if not self.x.is_contiguous():
            self.x = self.x.contiguous()
        if not self.y.is_contiguous():
            self.y = self.y.contiguous()
        if not self.x.is_shared():
            self.x.share_memory_()
        if not self.y.is_shared():
            self.y.share_memory_()
        return self

    def set_worker_seed(self, worker_id: int, base_seed: int | None = None) -> None:
        """Give a worker process its own CPU generator so parallel workers never collide."""
        gen = torch.Generator(device="cpu")
        if base_seed is not None:
            gen.manual_seed(int(base_seed) + 1_000_003 * (worker_id + 1))
        elif self.seed is not None:
            gen.manual_seed(int(self.seed) + 1_000_003 * (worker_id + 1))
        else:
            gen.seed()
        self.sampler.generator = gen

    def for_worker(self, *, build_cuda_graphs: bool = True, pack_ipc: bool = False) -> _CPUTopologyWorker:
        """Return a topology-only worker callable (omits ``x``/``y`` from worker IPC)."""
        return _CPUTopologyWorker(
            self.sampler,
            backend=self.backend,
            fwd_heavy_threshold=self.fwd_heavy_threshold,
            bwd_heavy_threshold=self.bwd_heavy_threshold,
            quantile=self.quantile,
            fwd_order=self.fwd_order,
            bwd_order=self.bwd_order,
            heavy_edge_slices=self.heavy_edge_slices,
            seed=self.seed,
            build_cuda_graphs=build_cuda_graphs,
            pack_ipc=pack_ipc,
        )

    def _gather(
        self,
        tensor: torch.Tensor,
        idx: torch.Tensor,
        out_dtype: torch.dtype | None = None,
        *,
        pin_memory: bool | None = None,
        alloc: Any | None = None,
    ) -> torch.Tensor:
        pin = self.pin_memory if pin_memory is None else pin_memory
        if out_dtype is None or tensor.dtype == out_dtype:
            shape = (idx.numel(), *tensor.shape[1:])
            if alloc is not None:
                out = alloc(shape, tensor.dtype)
            else:
                out = torch.empty(shape, dtype=tensor.dtype, pin_memory=pin)
            if idx.numel() > 0:
                if not native_gather_rows(tensor, idx, out, _default_gather_threads()):
                    torch.index_select(tensor, 0, idx, out=out)
            return out
        out = torch.index_select(tensor, 0, idx)
        if out.dtype != out_dtype:
            out = out.to(out_dtype)
        if alloc is not None:
            dst = alloc(tuple(out.shape), out.dtype)
            dst.copy_(out)
            return dst
        if pin:
            out = out.pin_memory()
        return out

    def estimate_gather_bytes(self, raw: dict[str, Any]) -> int:
        """Bytes needed in a pinned staging slab for ``x`` and ``y`` of a topology-worker batch."""
        if "_packed_offsets" in raw:
            offsets = raw["_packed_offsets"]
            n_in = offsets[0][1] // 8
            n_seeds = offsets[1][1] // 8
        else:
            n_in = int(raw["_input_nodes"].numel())
            n_seeds = int(raw["_seeds"].numel())
        x_dt = self.x_dtype if self.x_dtype is not None else self.x.dtype
        x_row = (self.x.numel() // self.x.shape[0]) if self.x.shape[0] > 0 else 0
        y_row = (self.y.numel() // self.y.shape[0]) if self.y.shape[0] > 0 else 0
        x_bytes = n_in * x_row * torch.empty((), dtype=x_dt).element_size()
        y_bytes = n_seeds * y_row * self.y.element_size()
        return ((x_bytes + 63) & ~63) + ((y_bytes + 63) & ~63)

    def finalize_cpu_batch(
        self, raw: dict[str, Any], *, pin_memory: bool = False, alloc: Any | None = None
    ) -> dict[str, Any]:
        """Unpack IPC buffer if needed and gather ``x``/``y`` in the main process."""
        if "_packed_buf" in raw:
            buf = raw.pop("_packed_buf")
            offsets = raw.pop("_packed_offsets")
            block_meta = raw.pop("_packed_block_meta")
            tensors = _unpack_tensors(buf, offsets)
            input_nodes, seeds = tensors[0], tensors[1]
            blocks = []
            for idx, (num_src, fanout) in enumerate(block_meta):
                ip = tensors[2 + 2 * idx]
                ix = tensors[3 + 2 * idx]
                blocks.append(SampledBlock(indptr=ip, indices=ix, num_src=num_src, fanout=fanout))
            raw["_raw_blocks"] = blocks
        else:
            input_nodes = raw.pop("_input_nodes")
            seeds = raw.pop("_seeds")
        raw["x"] = self._gather(self.x, input_nodes, out_dtype=self.x_dtype, pin_memory=pin_memory, alloc=alloc)
        raw["y"] = self._gather(self.y, seeds, pin_memory=pin_memory, alloc=alloc)
        return raw

    def _cuda_graph_for(self, blk: SampledBlock):
        g = self.sampler.build_cuda_graph(
            blk,
            fwd_heavy_threshold=self.fwd_heavy_threshold,
            bwd_heavy_threshold=self.bwd_heavy_threshold,
            quantile=self.quantile,
            fwd_order=self.fwd_order,
            bwd_order=self.bwd_order,
        )
        for direction, size in self.heavy_edge_slices.items():
            g.heavy_edge_slices(direction, size)
        if self.pin_memory:
            g.pin_memory()
        return g

    def __call__(self, seeds: torch.Tensor) -> dict[str, Any]:
        batch = self.sampler.sample(seeds)
        x_batch = self._gather(self.x, batch.input_nodes, out_dtype=self.x_dtype)
        y_batch = self._gather(self.y, batch.seeds)
        stats = [(b.num_dst, b.num_src, b.num_edges) for b in batch.blocks]

        if self.backend == "cuda":
            graphs = [self._cuda_graph_for(b) for b in batch.blocks]
            return {
                "x": x_batch,
                "y": y_batch,
                "graphs": graphs,
                "stats": stats,
            }

        # DGL backend: keep raw CSR blocks on CPU (so worker processes never touch DGL/CUDA).
        # `post_transfer` materializes `dgl.create_block` on the target device after H2D copy.
        raw_blocks = batch.blocks
        if self.pin_memory:
            raw_blocks = [
                SampledBlock(
                    indptr=b.indptr.pin_memory(),
                    indices=b.indices.pin_memory(),
                    num_src=b.num_src,
                    fanout=b.fanout,
                )
                for b in raw_blocks
            ]
        return {
            "x": x_batch,
            "y": y_batch,
            "graphs": None,
            "stats": stats,
            "_dgl_blocks": raw_blocks,
        }

    def post_transfer(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Finalize a transferred batch on the target device (GPU transpose/buckets or DGL blocks)."""
        raw_blocks = batch.pop("_raw_blocks", None)
        if raw_blocks is None:
            raw_blocks = batch.pop("_dgl_blocks", None)
        if raw_blocks is None:
            return batch
        if self.backend == "cuda":
            from turbo_gnn.graph import AdjacencyForwardBackwardWithNodeBuckets as G

            graphs = []
            for b in raw_blocks:
                g = G.from_block_csr(
                    b.indptr,
                    b.indices,
                    b.num_src,
                    max_in_degree=b.fanout if b.fanout > 0 else None,
                    forward_heavy_threshold=self.fwd_heavy_threshold,
                    backward_heavy_threshold=self.bwd_heavy_threshold,
                    quantile=self.quantile,
                    forward_order=self.fwd_order,
                    backward_order=self.bwd_order,
                )
                for direction, size in self.heavy_edge_slices.items():
                    g.heavy_edge_slices(direction, size)
                graphs.append(g)
            batch["graphs"] = graphs
            return batch
        import dgl

        graphs = []
        csr_refs = []
        for b in raw_blocks:
            eids = torch.arange(b.num_edges, dtype=b.indptr.dtype, device=b.indptr.device)
            graphs.append(
                dgl.create_block(
                    ("csc", (b.indptr, b.indices, eids)),
                    num_src_nodes=b.num_src,
                    num_dst_nodes=b.num_dst,
                    idtype=b.indptr.dtype,
                    device=b.indptr.device,
                )
            )
            # Keep every tensor passed into DGL in the batch so record_stream() covers them all.
            csr_refs.append((b.indptr, b.indices, eids))
        batch["graphs"] = graphs
        batch["_csr"] = csr_refs
        return batch
