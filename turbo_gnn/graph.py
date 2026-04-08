"""Graph representation with forward/backward CSR and node bucketing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


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
