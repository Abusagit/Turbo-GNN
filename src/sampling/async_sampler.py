"""Neighbor sampler with no host synchronisation: every GPU op has a host-static shape.

`GPUNeighborSampler` needs a few data-dependent sizes per layer (edges sampled, new frontier
nodes), and reading each one on the host is a stream sync. This sampler never reads one while
sampling. It works on padded capacities known before launch, keeps the real counts on the
device, and ships them to pinned host memory with one non-blocking copy at the end:

- **launch** (stage 1) issues all GPU work for a batch and returns immediately: per layer,
  ``n_cap`` destination rows (``n`` real), ``E_cap`` edge slots -- a static bound from the
  degree distribution (the sum of the ``n_cap`` largest ``min(deg, fanout)``), and
  ``min(n_cap + E_cap, N)`` source rows. Sampling without replacement uses Floyd's
  algorithm vectorised over ``[n_cap, fanout]`` (uniform over subsets, shape independent of
  the degrees); the frontier dedup uses a node-indexed mark table + cumsum + scatter (no
  ``unique``/``nonzero``); the forward CSR, the transposed CSR and the light/heavy node order
  are all built padded. The real sizes ``[n, E, num_src, heavy_fwd, heavy_bwd]`` per layer go
  to pinned memory via ``copy_(non_blocking=True)``, followed by a CUDA event.
- **finalize** (stage 2) runs once that event has completed -- checked with ``event.query()``,
  never a blocking sync -- and slices exact views out of the padded buffers. Callers pipeline
  the two: launch batch i+1, then finalize batch i (see `PrefetchLoader`).

Block layout matches `GPUNeighborSampler`: outermost block first, destinations are the first
``num_dst`` sources, CSR rows are destinations with local source ids. A fanout of -1 takes the
full neighborhood: it is resolved to the graph's maximum in-degree, so no row is subsampled and
the edge capacity is the sum of the ``n_cap`` largest degrees (still static, still capturable).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch


@dataclass
class BlockParts:
    """One block as exact views into the padded buffers (all index tensors in ``index_dtype``)."""

    indptr: torch.Tensor  # [num_dst + 1] forward CSR, rows = destinations
    indices: torch.Tensor  # [E] local source ids
    bwd_indptr: torch.Tensor  # [num_src + 1] transposed CSR, rows = sources
    bwd_indices: torch.Tensor  # [E] local destination ids, (src, dst) order
    fwd_light: torch.Tensor
    fwd_heavy: torch.Tensor
    bwd_light: torch.Tensor
    bwd_heavy: torch.Tensor
    num_src: int
    fanout: int

    @property
    def num_dst(self) -> int:
        return self.indptr.numel() - 1

    @property
    def num_edges(self) -> int:
        return self.indices.numel()


@dataclass
class FinalBatch:
    seeds: torch.Tensor
    input_nodes: torch.Tensor  # [num_src of block 0] global ids, seeds first
    blocks: list[BlockParts]  # outermost first


@dataclass
class _PaddedLayer:
    fanout: int
    n_cap: int
    src_cap: int
    indptr: torch.Tensor  # [n_cap + 1]
    indices: torch.Tensor  # [E_cap], valid prefix of length E
    bwd_indptr: torch.Tensor  # [src_cap + 1]
    bwd_indices: torch.Tensor  # [E_cap], valid prefix of length E
    fwd_order: torch.Tensor  # [n_cap]: heavy rows, then light rows, then padding
    bwd_order: torch.Tensor  # [src_cap]


@dataclass
class PendingBatch:
    seeds: torch.Tensor
    layers: list[_PaddedLayer]  # innermost (seed) layer first, as sampled
    src_nodes: torch.Tensor  # [src_cap] of the outermost layer, padded with N
    sizes_host: torch.Tensor  # pinned [L, 5] int64, valid once `ready` has completed
    ready: torch.cuda.Event


class AsyncNeighborSampler:
    """Sync-free uniform neighbor sampler (see module docstring).

    Args:
        indptr, indices: in-neighbor CSR of the full graph on the GPU.
        fanouts: per-layer fanouts, outermost first; all must be positive.
        index_dtype: dtype of the emitted CSR / bucket lists.
        fwd_heavy_threshold / bwd_heavy_threshold: rows with in-degree (forward) or source
            out-degree (backward) >= threshold go to the heavy bucket; None = all light.
        fwd_order / bwd_order: ``"natural"`` keeps rows in id order inside each bucket;
            ``"degree"`` sorts them by sampled degree, largest first, so the warps packed into
            one thread block get similar work (the paper's degree ordering, applied per block).
            Sampling caps forward in-degrees at the fanout, so the skew that is left is on the
            backward (transposed) side, where a hub sampled by many destinations becomes one
            long row -- ``bwd_order="degree"`` targets exactly that.
        generator: optional CUDA ``torch.Generator``.
    """

    def __init__(
        self,
        indptr: torch.Tensor,
        indices: torch.Tensor,
        fanouts: list[int],
        index_dtype: torch.dtype = torch.int32,
        fwd_heavy_threshold: int | None = None,
        bwd_heavy_threshold: int | None = None,
        generator: torch.Generator | None = None,
        fwd_order: str = "natural",
        bwd_order: str = "natural",
    ) -> None:
        for name, v in (("fwd_order", fwd_order), ("bwd_order", bwd_order)):
            if v not in ("natural", "degree"):
                raise ValueError(f"{name} must be 'natural' or 'degree', got {v!r}")
        if any(f == 0 or f < -1 for f in fanouts):
            raise ValueError(f"fanouts must be positive or -1 (all neighbors), got {fanouts}")
        self.indptr = indptr.to(torch.int64)
        self.indices = indices.to(torch.int64)
        # -1 (full neighborhood) = fanout max in-degree: every row then has deg <= fanout, so it
        # keeps all its in-edges, Floyd never runs (no row is over-full), and the static edge
        # bound is the sum of the n_cap largest degrees. One host read, here, before training.
        max_deg = max(1, int((self.indptr[1:] - self.indptr[:-1]).max())) if indptr.numel() > 1 else 1
        self.fanouts = [max_deg if f == -1 else int(f) for f in fanouts]
        self.index_dtype = index_dtype
        self.fwd_t = fwd_heavy_threshold
        self.bwd_t = bwd_heavy_threshold
        self.fwd_order_mode = fwd_order
        self.bwd_order_mode = bwd_order
        self.generator = generator
        self.num_nodes = indptr.numel() - 1
        self.num_edges = indices.numel()
        dev = indptr.device
        n = self.num_nodes
        # Node-indexed scratch with one extra dummy slot (id N) that padding entries point at,
        # followed by private write slots for the frontier scatter (grown in stage 1 to E_cap).
        self._e_slots = torch.arange(1, device=dev)
        self._local = torch.full((n + 2,), -1, dtype=torch.int64, device=dev)
        self._mark = torch.zeros(n + 1, dtype=torch.bool, device=dev)
        # Static capacity bounds per fanout, computed here (before training: these few host
        # reads are the only syncs the sampler ever makes). For any n destination rows the
        # sampled edges are at most the sum of the n largest min(deg, f) over the whole graph,
        # and at most min(n, #nodes with deg > f) rows need Floyd sampling.
        deg = self.indptr[1:] - self.indptr[:-1]
        self._edge_bound: dict[int, torch.Tensor] = {}  # f -> CPU prefix sums, [N + 1]
        self._big_bound: dict[int, int] = {}
        for f in set(self.fanouts):
            top = torch.sort(deg.clamp(max=f), descending=True).values
            self._edge_bound[f] = torch.cat([top.new_zeros(1), torch.cumsum(top, 0)]).cpu()
            self._big_bound[f] = int((deg > f).sum())

    def padded_edge_capacity(self, batch_size: int) -> int:
        """Edge slots one batch of `batch_size` seeds is padded to, summed over layers (host-side)."""
        total, n_cap = 0, batch_size
        for f in reversed(self.fanouts):
            e = self.edge_cap(f, n_cap)
            total += e
            n_cap = min(n_cap + e, self.num_nodes)
        return total

    def edge_cap(self, f: int, n_cap: int) -> int:
        """Static bound on the edges n_cap destination rows can sample at fanout f (host-side)."""
        return max(1, int(self._edge_bound[f][min(n_cap, self.num_nodes)]))

    @property
    def device(self) -> torch.device:
        return self.indptr.device

    # ------------------------------------------------------------------ stage 1
    def _floyd(self, deg: torch.Tensor, f: int) -> torch.Tensor:
        """[n_cap, f] distinct in-row positions: Floyd's algorithm where deg > f, 0..f-1 otherwise."""
        n_cap = deg.numel()
        k = torch.arange(f, device=self.device)
        picks = torch.empty(n_cap, f, dtype=torch.int64, device=self.device)
        for it in range(f):
            j = deg - f + it  # Floyd: for j in [d-f, d): t ~ U[0, j]; take t unless taken, else j
            u = torch.rand(n_cap, dtype=torch.float64, device=self.device, generator=self.generator)
            t = torch.minimum((u * (j + 1).to(torch.float64)).to(torch.int64), j)
            if it:
                taken = (picks[:, :it] == t[:, None]).any(1)
                t = torch.where(taken, j, t)
            picks[:, it] = t
        return torch.where((deg > f)[:, None], picks, k[None, :].expand(n_cap, f))

    def _bucket_order(self, counts: torch.Tensor, valid: torch.Tensor, threshold: int | None, mode: str = "natural"):
        """Row order heavy | light | padding, and the heavy count (both on the device).

        ``mode="degree"`` additionally sorts rows by degree, descending, inside each bucket.
        One device sort either way: no host sync, so it stays CUDA-graph capturable."""
        zero = torch.zeros((), dtype=torch.int64, device=self.device)
        if threshold is None and mode == "natural":
            # All light; valid rows are already a prefix, so the identity order works.
            return torch.arange(counts.numel(), device=self.device), zero
        heavy = valid & (counts >= threshold) if threshold is not None else torch.zeros_like(valid)
        bucket = torch.where(heavy, 0, torch.where(valid, 1, 2))
        if mode == "degree":
            # bucket in the high bits, (2^40 - 1 - degree) in the low bits: ascending sort =
            # bucket order, then degree descending; ties keep id order (stable sort).
            big = 1 << 40
            key = bucket * big + (big - 1 - counts.clamp(min=0, max=big - 1))
        else:
            key = bucket
        return torch.sort(key, stable=True).indices, (heavy.sum() if threshold is not None else zero)

    def launch(self, seeds: torch.Tensor) -> PendingBatch:
        """Issue every GPU op for one batch on the current stream; no host sync."""
        dst = seeds.to(device=self.device, dtype=torch.int64)
        n = torch.full((), dst.numel(), dtype=torch.int64, device=self.device)
        sizes_host = torch.empty((len(self.fanouts), 5), dtype=torch.int64, pin_memory=True)
        layers, src_nodes = self._launch_core(dst, n, sizes_host)
        ready = torch.cuda.Event()
        ready.record()
        return PendingBatch(seeds=seeds, layers=layers, src_nodes=src_nodes, sizes_host=sizes_host, ready=ready)

    def _launch_core(self, dst: torch.Tensor, n: torch.Tensor, sizes_host: torch.Tensor):
        """The whole stage 1 for destinations `dst` (padded with id N past the real count `n`, a
        0-dim device tensor). Static shapes only, so it can be captured in a CUDA graph."""
        dev, N, idt = self.device, self.num_nodes, self.index_dtype
        n_cap = dst.numel()
        local, mark = self._local, self._mark
        layers: list[_PaddedLayer] = []
        sizes = []
        for f in reversed(self.fanouts):
            rows = torch.arange(n_cap, device=dev)
            valid_row = rows < n
            dst_c = dst.clamp(max=N - 1)  # padding (id N) reads row N-1; masked out below
            start = self.indptr[dst_c]
            deg = torch.where(valid_row, self.indptr[dst_c + 1] - start, 0)
            counts = deg.clamp(max=f)

            # ---- sample into a flat, row-major edge layout of static size E_cap (the degree
            # bound, far below n_cap * f when most rows have deg < f). Slot e belongs to the row
            # whose cumulative count first exceeds e; slots past the real total are padding.
            E_cap = self.edge_cap(f, n_cap)
            if self._e_slots.numel() < E_cap:  # host-side ints: runs in the eager warm-up, never in a capture
                self._e_slots = torch.arange(E_cap, device=dev)
                self._local = torch.full((N + 1 + E_cap,), -1, dtype=torch.int64, device=dev)
                local = self._local
            row_end = torch.cumsum(counts, 0)
            row_start = row_end - counts
            num_e = row_end[-1]
            e = torch.arange(E_cap, device=dev)
            evalid = e < num_e
            erow = torch.searchsorted(row_end, e, right=True).clamp(max=n_cap - 1)
            rank = e - row_start[erow]  # 0..counts-1 within the row (garbage on padding)
            inrow = rank
            # Rows with deg > f: Floyd over a compacted [H_cap, f] array of just those rows.
            H_cap = min(n_cap, self._big_bound[f])
            if H_cap > 0:
                big = deg > f
                bidx = torch.cumsum(big, 0) - 1
                big_rows = torch.zeros(H_cap + n_cap, dtype=torch.int64, device=dev)
                big_rows.scatter_(0, torch.where(big, bidx, H_cap + rows), rows)  # others -> private slots
                big_rows = big_rows[:H_cap]
                bvalid = torch.arange(H_cap, device=dev) < big.sum()
                bdeg = torch.where(bvalid, deg[big_rows], f)  # padding rows: deg == f -> identity picks
                picks = self._floyd(bdeg, f)
                ebig = big[erow] & evalid
                pick = picks[bidx[erow].clamp(0, H_cap - 1), rank.clamp(0, f - 1)]
                inrow = torch.where(ebig, pick, rank)
            edge_pos = (start[erow] + inrow).clamp(0, self.num_edges - 1)
            src = torch.where(evalid, self.indices[edge_pos], N)  # padding -> N

            # ---- frontier: destinations first, then new sources in id order. Sort-based, so the
            # work is O(E_cap log E_cap) in the sampled edges, not O(N) in the graph: on a 111M-node
            # graph the dense node-mask/cumsum version cost ~47 ms of GPU time per batch.
            mark.index_fill_(0, dst, True)  # dst ids are not new
            mark[N:].fill_(True)  # nor is the padding dummy
            s_sorted, _ = torch.sort(src, stable=True)  # padding (id N) sorts last; stable = no sync
            first = torch.ones_like(s_sorted, dtype=torch.bool)
            first[1:] = s_sorted[1:] != s_sorted[:-1]
            isnew = first & ~mark[s_sorted]  # mark[N] is set, so padding is never new
            nrank = torch.cumsum(isnew, 0) - 1  # duplicates of a new node share its rank
            num_new = nrank[-1] + 1
            ingroup = ~mark[s_sorted]  # entries whose value is a new node (first or duplicate)
            mark.index_fill_(0, dst, False)
            mark[N:].fill_(False)
            local.index_copy_(0, dst, rows)  # padding all lands on the dummy slot
            # New nodes get n + rank; every other entry writes to a private slot past N + 1 (one
            # each), never to one shared dump address: many writes to one address serialise.
            local.scatter_(0, torch.where(ingroup, s_sorted, N + 1 + self._e_slots[:E_cap]), n + nrank)
            col = local[src]  # padding slots read the dummy; masked below
            src_cap = min(n_cap + E_cap, N)
            src_nodes = torch.full((src_cap + E_cap,), N, dtype=torch.int64, device=dev)
            src_nodes[:n_cap] = dst
            src_nodes.scatter_(0, torch.where(isnew, n + nrank, src_cap + self._e_slots[:E_cap]), s_sorted)
            src_nodes = src_nodes[:src_cap]
            num_src = n + num_new
            local.index_fill_(0, src_nodes, -1)
            local[N : N + 1].fill_(-1)

            # ---- forward CSR (rows = destinations): slots are already row-major, so the column
            # ids need no compaction; the valid prefix has length num_e.
            indptr = torch.cat([row_end.new_zeros(1), row_end])
            indices = col

            # ---- transposed CSR (rows = sources); padding keys sort last
            key = torch.where(evalid, col, src_cap)
            key_sorted, perm = torch.sort(key, stable=True)
            bwd_indices = erow[perm]
            # Row pointers straight from the sorted keys (padding keys = src_cap sort last), with no
            # atomics: row r starts at the first key >= r.
            bwd_indptr = torch.searchsorted(key_sorted, torch.arange(src_cap + 1, device=dev))
            bwd_counts = bwd_indptr[1:] - bwd_indptr[:-1]

            # ---- light/heavy order per direction
            fwd_order, h_f = self._bucket_order(counts, valid_row, self.fwd_t, self.fwd_order_mode)
            src_valid = torch.arange(src_cap, device=dev) < num_src
            bwd_order, h_b = self._bucket_order(bwd_counts, src_valid, self.bwd_t, self.bwd_order_mode)

            layers.append(
                _PaddedLayer(
                    fanout=f,
                    n_cap=n_cap,
                    src_cap=src_cap,
                    indptr=indptr.to(idt),
                    indices=indices.to(idt),
                    bwd_indptr=bwd_indptr.to(idt),
                    bwd_indices=bwd_indices.to(idt),
                    fwd_order=fwd_order.to(idt),
                    bwd_order=bwd_order.to(idt),
                )
            )
            sizes.append(torch.stack([n, num_e, num_src, h_f, h_b]))
            dst, n, n_cap = src_nodes, num_src, src_cap

        sizes_host.copy_(torch.stack(sizes), non_blocking=True)
        return layers, dst

    # ------------------------------------------------------------------ stage 2
    @staticmethod
    def wait_ready(pending: PendingBatch, poll_s: float = 50e-6) -> None:
        """Wait for the size copy by polling the event (no blocking CUDA sync)."""
        while not pending.ready.query():
            time.sleep(poll_s)

    def finalize(self, pending: PendingBatch, clone: bool = False) -> FinalBatch:
        """Exact views of a launched batch. Polls (never syncs) until its sizes have landed.

        ``clone=True`` copies the exact-size slices out (used by `GraphedNeighborSampler`, whose
        buffers a later replay overwrites)."""
        self.wait_ready(pending)
        sizes = pending.sizes_host.tolist()  # host memory: no device access
        blocks = []
        own = (lambda t: t.clone()) if clone else (lambda t: t)
        for layer, (n, e, num_src, h_f, h_b) in zip(pending.layers, sizes):
            blocks.append(
                BlockParts(
                    indptr=own(layer.indptr[: n + 1]),
                    indices=own(layer.indices[:e]),
                    bwd_indptr=own(layer.bwd_indptr[: num_src + 1]),
                    bwd_indices=own(layer.bwd_indices[:e]),
                    fwd_light=own(layer.fwd_order[h_f:n]),
                    fwd_heavy=own(layer.fwd_order[:h_f]),
                    bwd_light=own(layer.bwd_order[h_b:num_src]),
                    bwd_heavy=own(layer.bwd_order[:h_b]),
                    num_src=num_src,
                    fanout=layer.fanout,
                )
            )
        blocks.reverse()
        input_nodes = own(pending.src_nodes[: sizes[-1][2]])
        return FinalBatch(seeds=pending.seeds, input_nodes=input_nodes, blocks=blocks)


class GraphedNeighborSampler:
    """`AsyncNeighborSampler` with stage 1 captured in CUDA graphs: one replay per batch.

    Stage 1 issues a few hundred small kernels per batch from Python; on a dedicated GPU that
    CPU-side launch cost (and the GIL it takes from the training thread) is what shows, not the
    GPU work. All its shapes are static for a given batch size, so it is captured once and
    replayed. Seeds go into a fixed buffer of ``batch_size`` padded with the dummy id N; the real
    count is a device scalar, so one graph serves the short last batch too.

    ``instances`` graphs are used round-robin so a replay never overwrites a batch that has not
    been finalized yet (the loader launches batch i+1 before finalizing batch i); `finalize`
    clones the exact-size slices out, so finalized batches outlive later replays.

    Captured lazily on the first `launch`, on whatever stream is current (the producer's).
    """

    def __init__(self, sampler: AsyncNeighborSampler, batch_size: int, instances: int = 3) -> None:
        self.s = sampler
        self.batch_size = batch_size
        self.instances = instances
        self._graphs: list[tuple] = []
        self._next = 0

    def _capture(self) -> None:
        s, B, dev = self.s, self.batch_size, self.s.device
        # A private generator, registered with every captured graph. Captured on the default
        # CUDA generator, the replays' Philox ranges overlapped the eager draws issued after
        # them (dropout, the epoch shuffle): dropout masks then correlated with the neighbour
        # picks, training loss dropped and ogbn-products val accuracy fell from 0.88 to 0.69.
        if s.generator is None:
            s.generator = torch.Generator(device=dev)
            s.generator.manual_seed(int(torch.randint(0, 2**62, (1,), device="cpu")))
        for _ in range(self.instances):
            seeds_buf = torch.full((B,), s.num_nodes, dtype=torch.int64, device=dev)
            n_buf = torch.full((), B, dtype=torch.int64, device=dev)
            sizes_host = torch.empty((len(s.fanouts), 5), dtype=torch.int64, pin_memory=True)
            # Warm-up on a side stream (allocator pools, cub temp storage), as capture requires.
            warm = torch.cuda.Stream(device=dev)
            warm.wait_stream(torch.cuda.current_stream(dev))
            with torch.cuda.stream(warm):
                for _ in range(2):
                    s._launch_core(seeds_buf, n_buf, sizes_host)
            torch.cuda.current_stream(dev).wait_stream(warm)
            graph = torch.cuda.CUDAGraph()
            graph.register_generator_state(s.generator)
            with torch.cuda.graph(graph):
                layers, src_nodes = s._launch_core(seeds_buf, n_buf, sizes_host)
            self._graphs.append((graph, seeds_buf, n_buf, sizes_host, layers, src_nodes))

    def launch(self, seeds: torch.Tensor) -> PendingBatch:
        if not self._graphs:
            self._capture()
        graph, seeds_buf, n_buf, sizes_host, layers, src_nodes = self._graphs[self._next % self.instances]
        self._next += 1
        m = seeds.numel()
        assert m <= self.batch_size, f"batch of {m} seeds exceeds the captured size {self.batch_size}"
        seeds_buf[:m].copy_(seeds.to(device=seeds_buf.device, dtype=torch.int64), non_blocking=True)
        seeds_buf[m:].fill_(self.s.num_nodes)
        n_buf.fill_(m)
        graph.replay()
        ready = torch.cuda.Event()
        ready.record()
        return PendingBatch(seeds=seeds, layers=layers, src_nodes=src_nodes, sizes_host=sizes_host, ready=ready)

    def finalize(self, pending: PendingBatch) -> FinalBatch:
        return self.s.finalize(pending, clone=True)
