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

from typing import Any

import torch

from src.sampling.neighbor_sampler import SampledBatch, SampledBlock


def _ensure_cpu(t: torch.Tensor, name: str) -> None:
    if t.device.type != "cpu":
        raise ValueError(f"{name} must be a CPU tensor, got {t.device}")


class CPUNeighborSampler:
    """Samples multi-hop blocks from an in-neighbor CSR held on the CPU.

    Args:
        indptr: ``[N + 1]`` CPU in-CSR row pointers (row v lists the sources of edges u->v).
        indices: ``[E]`` CPU source ids (kept in its original compact dtype, e.g. int32).
        fanouts: per-layer fanouts, outermost layer first; ``-1`` = all neighbors.
        index_dtype: dtype of the emitted block CSR (int32 is fastest for our CUDA kernels).
        generator: optional CPU ``torch.Generator`` for reproducible sampling.
    """

    def __init__(
        self,
        indptr: torch.Tensor,
        indices: torch.Tensor,
        fanouts: list[int],
        index_dtype: torch.dtype = torch.int32,
        generator: torch.Generator | None = None,
    ) -> None:
        _ensure_cpu(indptr, "indptr")
        _ensure_cpu(indices, "indices")
        if any(f == 0 or f < -1 for f in fanouts):
            raise ValueError(f"fanouts must be positive or -1 (all neighbors), got {fanouts}")
        if generator is not None and generator.device.type != "cpu":
            raise ValueError(f"generator must be on CPU, got {generator.device}")

        self.indptr = indptr if indptr.dtype == torch.int64 else indptr.to(torch.int64)
        self.indices = indices
        self.fanouts = [int(f) for f in fanouts]
        self.index_dtype = index_dtype
        self.generator = generator
        self.num_nodes = indptr.numel() - 1

    @property
    def device(self) -> torch.device:
        return self.indptr.device

    def __getstate__(self) -> dict[str, Any]:
        # torch.Generator is not safely picklable across spawn processes; workers seed
        # their own independent generators in CPUBatchMaker.set_worker_seed.
        state = self.__dict__.copy()
        state["generator"] = None
        return state

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
        seeds_cpu = seeds.to(device="cpu", dtype=torch.int64)
        dst = seeds_cpu
        blocks: list[SampledBlock] = []
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
        from turbo_gnn.graph import AdjacencyForwardBackwardWithNodeBuckets as G

        g = G.from_block_csr(
            blk.indptr,
            blk.indices,
            blk.num_src,
            max_in_degree=blk.fanout if blk.fanout > 0 else None,
            forward_heavy_threshold=self.fwd_heavy_threshold,
            backward_heavy_threshold=self.bwd_heavy_threshold,
            quantile=self.quantile,
            forward_order=self.fwd_order,
            backward_order=self.bwd_order,
        )
        for direction, size in self.heavy_edge_slices.items():
            g.heavy_edge_slices(direction, size)
        return g

    def __call__(self, seeds: torch.Tensor) -> dict[str, Any]:
        batch = self.sampler.sample(seeds)
        stats = [(b.num_dst, b.num_src, b.num_edges) for b in batch.blocks]
        if self.backend == "cuda":
            graphs = [self._cuda_graph_for(b) for b in batch.blocks]
            return {
                "_input_nodes": batch.input_nodes,
                "_seeds": batch.seeds,
                "graphs": graphs,
                "stats": stats,
            }
        return {
            "_input_nodes": batch.input_nodes,
            "_seeds": batch.seeds,
            "graphs": None,
            "stats": stats,
            "_dgl_blocks": batch.blocks,
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
        self.x = x
        self.y = y
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

    def for_worker(self) -> _CPUTopologyWorker:
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
        )

    def _gather(
        self,
        tensor: torch.Tensor,
        idx: torch.Tensor,
        out_dtype: torch.dtype | None = None,
        *,
        pin_memory: bool | None = None,
    ) -> torch.Tensor:
        pin = self.pin_memory if pin_memory is None else pin_memory
        if pin and (out_dtype is None or tensor.dtype == out_dtype):
            out = torch.empty((idx.numel(), *tensor.shape[1:]), dtype=tensor.dtype, pin_memory=True)
            if idx.numel() > 0:
                torch.index_select(tensor, 0, idx, out=out)
            return out
        out = torch.index_select(tensor, 0, idx)
        if out_dtype is not None and out.dtype != out_dtype:
            out = out.to(out_dtype)
        if pin:
            out = out.pin_memory()
        return out

    def finalize_cpu_batch(self, raw: dict[str, Any], *, pin_memory: bool = False) -> dict[str, Any]:
        """Gather ``x``/``y`` in the main process (directly into pinned memory when enabled)."""
        input_nodes = raw.pop("_input_nodes")
        seeds = raw.pop("_seeds")
        raw["x"] = self._gather(self.x, input_nodes, out_dtype=self.x_dtype, pin_memory=pin_memory)
        raw["y"] = self._gather(self.y, seeds, pin_memory=pin_memory)
        return raw

    def _cuda_graph_for(self, blk: SampledBlock):
        from turbo_gnn.graph import AdjacencyForwardBackwardWithNodeBuckets as G

        g = G.from_block_csr(
            blk.indptr,
            blk.indices,
            blk.num_src,
            max_in_degree=blk.fanout if blk.fanout > 0 else None,
            forward_heavy_threshold=self.fwd_heavy_threshold,
            backward_heavy_threshold=self.bwd_heavy_threshold,
            quantile=self.quantile,
            forward_order=self.fwd_order,
            backward_order=self.bwd_order,
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
        """Finalize a transferred batch on the target device (builds DGL blocks when needed)."""
        if self.backend != "dgl":
            return batch
        raw_blocks = batch.pop("_dgl_blocks", None)
        if raw_blocks is None:
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
