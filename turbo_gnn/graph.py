"""Graph representation with forward/backward CSR and node bucketing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class EdgeSliceTable:
    """One thread block per slice of one heavy node's edge list. See `heavy_edge_slices`."""

    chunk_node: torch.Tensor  # [num_slices] int32, heavy-bucket slot (compacted, not node id)
    chunk_start: torch.Tensor  # [num_slices] int32, edge offset within that node's CSR row
    node_chunk_offset: torch.Tensor  # [num_heavy + 1] int32, prefix sum of slices per node
    num_slices: int


def _bucket_nodes_by_degree(
    degree_counts: torch.Tensor,
    quantile: float,
    index_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Partition nodes into light/heavy buckets based on degree quantile.

    Args:
        degree_counts: Per-node degree counts.
        quantile: Quantile threshold (-1 disables, putting all nodes in light).
        index_dtype: If provided, cast output indices to this dtype.

    Returns:
        (light_node_indices, heavy_node_indices)
    """
    if quantile != -1:
        thresh = torch.quantile(degree_counts.float(), quantile).item()
    else:
        thresh = degree_counts.max().item() + 1

    light = (degree_counts < thresh).nonzero(as_tuple=False).view(-1)
    heavy = (degree_counts >= thresh).nonzero(as_tuple=False).view(-1)
    if index_dtype is not None:
        light = light.to(index_dtype)
        heavy = heavy.to(index_dtype)
    return light, heavy


def build_csr_as_is(
    edge_index: torch.Tensor,
    edge_weight: Optional[torch.Tensor],
    num_nodes: int,
    do_transpose: bool = False,
):
    """Build CSR from COO edge_index.

    Returns:
        row_ptr, cols, w, counts
    """
    if do_transpose:
        rows = edge_index[1]
        cols = edge_index[0]
    else:
        rows = edge_index[0]
        cols = edge_index[1]

    N = num_nodes
    perm = (rows * N + cols).argsort()
    rows = rows[perm]
    cols = cols[perm]
    w = edge_weight[perm] if edge_weight is not None else None

    counts = torch.bincount(rows, minlength=N)
    row_ptr = torch.zeros(N + 1, dtype=torch.long, device=rows.device)
    row_ptr[1:] = counts.cumsum(0)

    return row_ptr, cols, w, counts


@dataclass
class AdjacencyForwardBackwardWithNodeBuckets:
    """Dual-CSR graph representation with degree-based node bucketing.

    Stores a graph as two CSR matrices (forward and backward/transposed) plus
    per-direction node partitions into "light" (low-degree) and "heavy"
    (high-degree) buckets.  This layout is consumed by all turbo_gnn CUDA
    kernels:

    - **Forward CSR** (``forward_indptr``, ``forward_indices``): rows are
      destination nodes, columns are source nodes.  Used in the forward pass
      to gather neighbor features into each destination.
    - **Backward CSR** (``backward_indptr``, ``backward_indices``): the
      transpose -- rows are source nodes, columns are destination nodes.
      Used in the backward pass to scatter gradients back to sources.
    - **Node buckets**: light/heavy partitions let kernels choose different
      execution strategies per node (e.g. atomic writes for light nodes vs.
      tiled reductions for heavy nodes).

    All index tensors must share the same dtype (int32 or int64).  Unsigned
    dtypes (uint32) are supported and converted internally via
    ``_to_signed_view`` for arithmetic.

    Construct via :meth:`from_edge_list`, :meth:`from_csr`, or
    :meth:`from_dgl`.
    """

    forward_indptr: torch.Tensor
    forward_indices: torch.Tensor
    backward_indptr: torch.Tensor
    backward_indices: torch.Tensor
    forward_light_nodes: torch.Tensor
    forward_heavy_nodes: torch.Tensor
    backward_light_nodes: torch.Tensor
    backward_heavy_nodes: torch.Tensor

    max_degree: int = -1
    _device: torch.device = torch.device("cpu")
    is_directed: bool | None = None  # None = auto-detect, True/False = explicit

    def __post_init__(self):
        self._device = self.forward_indptr.device
        idx_dtype = self.forward_indptr.dtype
        for name, t in [
            ("forward_indices", self.forward_indices),
            ("backward_indptr", self.backward_indptr),
            ("backward_indices", self.backward_indices),
            ("forward_light_nodes", self.forward_light_nodes),
            ("forward_heavy_nodes", self.forward_heavy_nodes),
            ("backward_light_nodes", self.backward_light_nodes),
            ("backward_heavy_nodes", self.backward_heavy_nodes),
        ]:
            assert t.dtype == idx_dtype, f"{name} dtype {t.dtype} doesn't match forward_indptr dtype {idx_dtype}"
        self.index_dtype = idx_dtype
        self._edge_slice_cache: dict[tuple[str, int], EdgeSliceTable] = {}
        self._min_heavy_degree_cache: dict[str, int] = {}
        self._heavy_edge_count_cache: dict[str, int] = {}

        indptr = self._to_signed_view(self.forward_indptr)
        degrees = indptr[1:] - indptr[:-1]
        self.max_degree = degrees.max().item()
        assert self.max_degree != -1

        # Auto-detect directedness if not explicitly set
        if self.is_directed is None:
            self.is_directed = not (
                torch.equal(self.forward_indptr, self.backward_indptr)
                and torch.equal(self.forward_indices, self.backward_indices)
            )

        # Alias backward CSR to forward CSR for undirected graphs (saves memory)
        if not self.is_directed:
            self.backward_indptr = self.forward_indptr
            self.backward_indices = self.forward_indices

    @staticmethod
    def _to_signed_view(t: torch.Tensor) -> torch.Tensor:
        """View unsigned index tensor as its signed counterpart for arithmetic."""
        if t.dtype == torch.uint32:
            return t.view(torch.int32)
        elif t.dtype == torch.uint64:
            return t.view(torch.int64)
        return t

    @property
    def light_nodes(self) -> torch.Tensor:
        return self.forward_light_nodes

    @property
    def heavy_nodes(self) -> torch.Tensor:
        return self.forward_heavy_nodes

    @property
    def device(self) -> torch.device:
        return self._device

    def min_heavy_degree(self, direction: str) -> int:
        """Degree of the *smallest* node still in the heavy bucket, i.e. the bucketing threshold.

        This is the natural scale for sizing an edge slice. A fixed slice of 256 edges means very
        different things on two graphs: on ogbn-arxiv the smallest heavy node has 88 neighbours, so
        256 covers it whole and the split does nothing for most of the bucket, while on
        hm-categories the threshold is 6,330 and 256 cuts every heavy row into dozens of pieces.
        Expressing the slice as a fraction of this threshold makes the parameter mean the same
        thing across graphs, which a raw edge count cannot.

        The threshold is a good scale only when the heavy bucket is reasonably tight. On
        web-fraud it spans 45 to 228,991, so a slice sized from the minimum is far too small for
        the hubs that dominate that bucket's runtime -- reaching a useful slice there needs a
        divisor well below 1. Graphs whose heavy degrees cluster (ogbn-proteins, hm-categories)
        do not have this problem.

        Returns 0 when the heavy bucket is empty.
        """
        cached = self._min_heavy_degree_cache.get(direction)
        if cached is not None:
            return cached
        heavy = getattr(self, f"{direction}_heavy_nodes")
        if heavy.numel() == 0:
            self._min_heavy_degree_cache[direction] = 0
            return 0
        indptr = self._to_signed_view(getattr(self, f"{direction}_indptr"))
        deg = (indptr[1:] - indptr[:-1]).index_select(0, heavy.to(torch.int64))
        val = int(deg.min())
        self._min_heavy_degree_cache[direction] = val
        return val

    def heavy_edge_count(self, direction: str) -> int:
        """Total edges owned by the heavy bucket. Cached; this is the scale that matters."""
        cached = self._heavy_edge_count_cache.get(direction)
        if cached is not None:
            return cached
        heavy = getattr(self, f"{direction}_heavy_nodes")
        if heavy.numel() == 0:
            self._heavy_edge_count_cache[direction] = 0
            return 0
        indptr = self._to_signed_view(getattr(self, f"{direction}_indptr"))
        deg = (indptr[1:] - indptr[:-1]).index_select(0, heavy.to(torch.int64))
        val = int(deg.sum())
        self._heavy_edge_count_cache[direction] = val
        return val

    def heavy_slice_for_blocks_per_sm(self, direction: str, blocks_per_sm: float) -> int:
        """Slice size that fills the device with `blocks_per_sm` blocks per SM.

        Sizing the slice from a degree statistic does not work. Measured against the true optimum
        on eight graphs, the implied constant for `min_heavy_degree` spans 1125x and for the
        edge-weighted median 222x -- hm-categories (heavy degrees ~6,330) and web-fraud (~45) want
        similar slices because they have similar *edge counts*, not similar degrees.

        Block count predicts it: over the graphs where the choice matters, the optimum sits at
        1,300-9,300 blocks, a 7x spread. So size the slice to hit a target block count, which is
        also device-relative rather than tied to one GPU's SM count.

        The curve is broad -- the plateau runs from roughly 64 to 2,048 edges on most graphs -- so
        this does not need to be exact. It does need to avoid the ends: a slice of 32 edges is
        *slower* than not slicing at all on twitch-views (0.60x) and ogbn-proteins (0.34x), where
        the merge and launch overhead swamp the gain.

        `blocks_per_sm <= 0` disables slicing and returns 0.
        """
        if blocks_per_sm <= 0:
            return 0
        e = self.heavy_edge_count(direction)
        if e <= 0:
            return 0
        dev = getattr(self, f"{direction}_indptr").device
        sm = torch.cuda.get_device_properties(dev).multi_processor_count if dev.type == "cuda" else 1
        return max(1, int(round(e / (blocks_per_sm * sm))))

    def heavy_edge_slices(self, direction: str, slice_size: int) -> EdgeSliceTable:
        """Map each thread block to one slice of one heavy node's edge list.

        The heavy bucket is assigned one block per node today, so a graph with 1,715 heavy nodes
        launches 1,715 blocks whose runtimes differ by the degree spread -- tiny grid, extreme
        imbalance. Splitting the bucket's edges into fixed-size slices instead gives a block per
        slice: balanced, and as many blocks as the work actually warrants.

        A slice never spans two nodes, so a node whose degree is not a multiple of `slice_size`
        simply ends with one short slice, and the per-slice partial results stay independently
        mergeable. Nodes with no edges keep exactly one (empty) slice so the merge still sees
        them and writes their identity result.

        Returns `chunk_node` (heavy-bucket slot per slice, i.e. the compacted position used to
        index partial buffers -- not the node id), `chunk_start` (edge offset within that node's
        CSR row) and `node_chunk_offset` (prefix sum over slices per node, so the merge kernel
        can find one node's slices).

        Cached per (direction, slice_size): building it costs a device->host sync for the total
        slice count, which must not be paid on every launch.
        """
        key = (direction, int(slice_size))
        cached = self._edge_slice_cache.get(key)
        if cached is not None:
            return cached
        if slice_size <= 0:
            raise ValueError(f"slice_size must be positive, got {slice_size}")

        indptr = self._to_signed_view(getattr(self, f"{direction}_indptr"))
        heavy = getattr(self, f"{direction}_heavy_nodes")
        dev = indptr.device
        num_heavy = int(heavy.numel())

        if num_heavy == 0:
            empty = torch.empty(0, dtype=torch.int32, device=dev)
            table = EdgeSliceTable(empty, empty, torch.zeros(1, dtype=torch.int32, device=dev), 0)
            self._edge_slice_cache[key] = table
            return table

        deg = (indptr[1:] - indptr[:-1]).index_select(0, heavy.to(torch.int64))
        # clamp so a degree-0 heavy node still gets one empty slice rather than vanishing
        counts = torch.div(deg + slice_size - 1, slice_size, rounding_mode="floor").clamp(min=1)

        offset = torch.zeros(num_heavy + 1, dtype=torch.int64, device=dev)
        torch.cumsum(counts, 0, out=offset[1:])
        num_slices = int(offset[-1])  # the one sync; cached with the table

        chunk_node = torch.repeat_interleave(torch.arange(num_heavy, dtype=torch.int64, device=dev), counts)
        chunk_start = (torch.arange(num_slices, dtype=torch.int64, device=dev) - offset[chunk_node]) * slice_size

        table = EdgeSliceTable(
            chunk_node.to(torch.int32).contiguous(),
            chunk_start.to(torch.int32).contiguous(),
            offset.to(torch.int32).contiguous(),
            num_slices,
        )
        self._edge_slice_cache[key] = table
        return table

    def sorted_by_degree(self) -> AdjacencyForwardBackwardWithNodeBuckets:
        """New instance whose light/heavy buckets are ordered by descending degree.

        This is longest-processing-time-first (LPT) scheduling: per-node cost is
        proportional to degree, so handing the expensive nodes out first and letting the
        cheap ones fill the tail is the standard makespan-minimising order. The persistent
        schedulers walk the bucket arrays in order, so sorting them here is all it takes.

        CSR tensors are shared with the original -- only the bucket index arrays are new,
        and the result is worth caching on the graph since it is reused across many calls.

        Note this trades locality for balance: ascending node order streams ``indptr`` and
        ``indices`` sequentially, while degree order scatters those reads. Whether it wins
        depends on the graph, which is why it is opt-in rather than the default.
        """
        fwd_indptr = self._to_signed_view(self.forward_indptr)
        fwd_deg = fwd_indptr[1:] - fwd_indptr[:-1]
        bwd_indptr = self._to_signed_view(self.backward_indptr)
        bwd_deg = bwd_indptr[1:] - bwd_indptr[:-1]

        def _sort(bucket: torch.Tensor, degrees: torch.Tensor) -> torch.Tensor:
            if bucket.numel() == 0:
                return bucket
            order = torch.argsort(degrees[bucket.long()], descending=True)
            return bucket[order].contiguous()

        return AdjacencyForwardBackwardWithNodeBuckets(
            forward_indptr=self.forward_indptr,
            forward_indices=self.forward_indices,
            backward_indptr=self.backward_indptr,
            backward_indices=self.backward_indices,
            forward_light_nodes=_sort(self.forward_light_nodes, fwd_deg),
            forward_heavy_nodes=_sort(self.forward_heavy_nodes, fwd_deg),
            backward_light_nodes=_sort(self.backward_light_nodes, bwd_deg),
            backward_heavy_nodes=_sort(self.backward_heavy_nodes, bwd_deg),
            is_directed=self.is_directed,
        )

    def sorted_by_locality(self) -> AdjacencyForwardBackwardWithNodeBuckets:
        """New instance whose buckets are ordered so nearby nodes share neighbours.

        These kernels are frequently limited by cache reuse rather than bandwidth: adjacent
        output nodes read overlapping neighbour features, so if they are processed close
        together the second one's reads hit in L2. ``scripts/scheduler_headroom.py`` shows
        ogbn-proteins moving more feature traffic than HBM could have delivered in the measured
        time, which is only possible because the caches carried it.

        Visit order is therefore a real performance parameter, and a large one -- shuffling
        ogbn-proteins' bucket randomly costs 30%. This orders each bucket by reverse
        Cuthill-McKee, which is the standard bandwidth-reducing permutation of a sparse matrix
        and clusters connected nodes together.

        **It does not always win, so it is opt-in.** Graphs already stored in a locality-
        friendly order have nothing to gain and can lose: ogbn-proteins measures 0.90x against
        its natural order, while ogbn-arxiv measures 1.14x and twitch-views 1.09x. Try it, and
        compare against :meth:`sorted_by_degree`, which balances instead and wins elsewhere
        (twitch-views 1.16x).

        Like :meth:`sorted_by_degree` this only changes *which node a block visits*, never
        where its result is written, so results are unchanged. CSR tensors are shared; only the
        bucket index arrays are new, so it is worth caching on the graph.

        Requires scipy, which is an optional dependency of this package.
        """
        try:
            from scipy.sparse import csr_matrix
            from scipy.sparse.csgraph import reverse_cuthill_mckee
        except ImportError as exc:  # pragma: no cover - depends on the install
            raise ImportError("sorted_by_locality() needs scipy, an optional dependency: pip install scipy") from exc

        import numpy as np

        indptr = self._to_signed_view(self.forward_indptr)
        n = indptr.numel() - 1
        adj = csr_matrix(
            (
                np.ones(self.forward_indices.numel(), dtype=np.int8),
                self.forward_indices.cpu().numpy().astype(np.int64),
                indptr.cpu().numpy().astype(np.int64),
            ),
            shape=(n, n),
        )
        # symmetric_mode=True treats the graph as undirected, which is what we want: reuse
        # comes from sharing neighbours in either direction.
        permutation = reverse_cuthill_mckee(adj, symmetric_mode=True)
        rank_np = np.empty(n, dtype=np.int64)
        rank_np[permutation] = np.arange(n)
        rank = torch.from_numpy(rank_np).to(self.forward_indptr.device)

        def _order(bucket: torch.Tensor) -> torch.Tensor:
            if bucket.numel() == 0:
                return bucket
            return bucket[torch.argsort(rank[bucket.long()])].contiguous()

        return AdjacencyForwardBackwardWithNodeBuckets(
            forward_indptr=self.forward_indptr,
            forward_indices=self.forward_indices,
            backward_indptr=self.backward_indptr,
            backward_indices=self.backward_indices,
            forward_light_nodes=_order(self.forward_light_nodes),
            forward_heavy_nodes=_order(self.forward_heavy_nodes),
            backward_light_nodes=_order(self.backward_light_nodes),
            backward_heavy_nodes=_order(self.backward_heavy_nodes),
            is_directed=self.is_directed,
        )

    def repartition(self, **kwargs) -> AdjacencyForwardBackwardWithNodeBuckets:
        """New instance with same CSR but re-bucketed nodes. CSR tensors are shared."""
        fwd_q = kwargs.get("forward_huge_degree_threshold_quantile")
        bwd_q = kwargs.get("backward_huge_degree_threshold_quantile")

        fwd_light, fwd_heavy = self.forward_light_nodes, self.forward_heavy_nodes
        bwd_light, bwd_heavy = self.backward_light_nodes, self.backward_heavy_nodes

        if fwd_q is not None:
            fwd_indptr = self._to_signed_view(self.forward_indptr)
            fwd_degrees = fwd_indptr[1:] - fwd_indptr[:-1]
            fwd_light, fwd_heavy = _bucket_nodes_by_degree(fwd_degrees, fwd_q, index_dtype=self.index_dtype)
        if bwd_q is not None:
            bwd_indptr = self._to_signed_view(self.backward_indptr)
            bwd_degrees = bwd_indptr[1:] - bwd_indptr[:-1]
            bwd_light, bwd_heavy = _bucket_nodes_by_degree(bwd_degrees, bwd_q, index_dtype=self.index_dtype)

        return AdjacencyForwardBackwardWithNodeBuckets(
            forward_indptr=self.forward_indptr,
            forward_indices=self.forward_indices,
            backward_indptr=self.backward_indptr,
            backward_indices=self.backward_indices,
            forward_light_nodes=fwd_light,
            forward_heavy_nodes=fwd_heavy,
            backward_light_nodes=bwd_light,
            backward_heavy_nodes=bwd_heavy,
            is_directed=self.is_directed,
        )

    def to(self, device) -> AdjacencyForwardBackwardWithNodeBuckets:
        self._edge_slice_cache = {}
        self._min_heavy_degree_cache = {}
        self._heavy_edge_count_cache = {}
        self.forward_indptr = self.forward_indptr.to(device)
        self.forward_indices = self.forward_indices.to(device)
        if self.is_directed:
            self.backward_indptr = self.backward_indptr.to(device)
            self.backward_indices = self.backward_indices.to(device)
        else:
            # Preserve aliasing: backward CSR shares tensors with forward CSR
            self.backward_indptr = self.forward_indptr
            self.backward_indices = self.forward_indices
        self.forward_light_nodes = self.forward_light_nodes.to(device)
        self.forward_heavy_nodes = self.forward_heavy_nodes.to(device)
        self.backward_light_nodes = self.backward_light_nodes.to(device)
        self.backward_heavy_nodes = self.backward_heavy_nodes.to(device)
        # `_device` backs the `device` property; without this it keeps reporting the device the
        # graph was built on, however many times the tensors have been moved since.
        self._device = self.forward_indptr.device
        torch.cuda.empty_cache()
        return self

    @classmethod
    def from_edge_list(
        cls,
        edge_index: torch.Tensor,
        num_nodes: int,
        quantile: float = 0.99,
        index_dtype: torch.dtype | None = None,
        is_directed: bool | None = None,
    ) -> AdjacencyForwardBackwardWithNodeBuckets:
        """Build from COO edge_index [2, E]. Constructs forward + backward CSR.

        Args:
            is_directed: None = auto-detect, True = always directed,
                         False = skip backward CSR (alias to forward).
        """
        fwd_indptr, fwd_indices, _, fwd_counts = build_csr_as_is(edge_index, None, num_nodes, do_transpose=True)

        if is_directed is not False:
            bwd_indptr, bwd_indices, _, bwd_counts = build_csr_as_is(edge_index, None, num_nodes, do_transpose=False)
        else:
            bwd_counts = fwd_counts

        if index_dtype is not None:
            fwd_indptr = fwd_indptr.to(index_dtype)
            fwd_indices = fwd_indices.to(index_dtype)
            if is_directed is not False:
                bwd_indptr = bwd_indptr.to(index_dtype)
                bwd_indices = bwd_indices.to(index_dtype)

        # For explicitly undirected, alias after dtype conversion
        if is_directed is False:
            bwd_indptr = fwd_indptr
            bwd_indices = fwd_indices

        idx_dt = index_dtype or fwd_indptr.dtype
        fwd_light, fwd_heavy = _bucket_nodes_by_degree(fwd_counts, quantile, index_dtype=idx_dt)
        if is_directed is False:
            bwd_light, bwd_heavy = fwd_light, fwd_heavy
        else:
            bwd_light, bwd_heavy = _bucket_nodes_by_degree(bwd_counts, quantile, index_dtype=idx_dt)

        return cls(
            forward_indptr=fwd_indptr,
            forward_indices=fwd_indices,
            backward_indptr=bwd_indptr,
            backward_indices=bwd_indices,
            forward_light_nodes=fwd_light,
            forward_heavy_nodes=fwd_heavy,
            backward_light_nodes=bwd_light,
            backward_heavy_nodes=bwd_heavy,
            is_directed=is_directed,
        )

    @classmethod
    def from_csr(
        cls,
        fwd_indptr: torch.Tensor,
        fwd_indices: torch.Tensor,
        bwd_indptr: torch.Tensor,
        bwd_indices: torch.Tensor,
        quantile: float = 0.99,
        index_dtype: torch.dtype | None = None,
        is_directed: bool | None = None,
    ) -> AdjacencyForwardBackwardWithNodeBuckets:
        """Build from pre-computed forward and backward CSR arrays.

        Args:
            is_directed: None = auto-detect, True = always directed,
                         False = alias backward CSR to forward.
        """
        if index_dtype is not None:
            fwd_indptr = fwd_indptr.to(index_dtype)
            fwd_indices = fwd_indices.to(index_dtype)
            if is_directed is not False:
                bwd_indptr = bwd_indptr.to(index_dtype)
                bwd_indices = bwd_indices.to(index_dtype)

        if is_directed is False:
            bwd_indptr = fwd_indptr
            bwd_indices = fwd_indices

        idx_dt = index_dtype or fwd_indptr.dtype

        signed_fwd = cls._to_signed_view(fwd_indptr)
        fwd_counts = signed_fwd[1:] - signed_fwd[:-1]

        fwd_light, fwd_heavy = _bucket_nodes_by_degree(fwd_counts, quantile, index_dtype=idx_dt)

        if is_directed is False:
            bwd_light, bwd_heavy = fwd_light, fwd_heavy
        else:
            signed_bwd = cls._to_signed_view(bwd_indptr)
            bwd_counts = signed_bwd[1:] - signed_bwd[:-1]
            bwd_light, bwd_heavy = _bucket_nodes_by_degree(bwd_counts, quantile, index_dtype=idx_dt)

        return cls(
            forward_indptr=fwd_indptr,
            forward_indices=fwd_indices,
            backward_indptr=bwd_indptr,
            backward_indices=bwd_indices,
            forward_light_nodes=fwd_light,
            forward_heavy_nodes=fwd_heavy,
            backward_light_nodes=bwd_light,
            backward_heavy_nodes=bwd_heavy,
            is_directed=is_directed,
        )

    @classmethod
    def from_dgl(
        cls,
        dgl_graph,
        quantile: float = 0.99,
        index_dtype: torch.dtype | None = None,
        is_directed: bool | None = None,
    ) -> AdjacencyForwardBackwardWithNodeBuckets:
        """Build from DGL graph (optional dep). Delegates to from_edge_list."""
        src, dst = dgl_graph.edges()
        edge_index = torch.stack([src, dst], dim=0)
        num_nodes = dgl_graph.num_nodes()
        return cls.from_edge_list(
            edge_index, num_nodes, quantile=quantile, index_dtype=index_dtype, is_directed=is_directed
        )
