"""Uniform neighbor sampler on the GPU, in plain torch.

Plain torch on purpose: the same code runs in both benchmark environments (ours, and the
torch-2.4 + DGL baseline), so the "DGL consumes our sampler" control differs from our runs
only in the conv layers.

Output convention (matches DGL's ``to_block``):

- one block per layer, outermost (input) layer first;
- a block's destination nodes are the first ``num_dst`` of its source nodes, so
  ``x_dst = x_src[:num_dst]`` and a layer's destinations are the next layer's sources;
- ``indptr [num_dst + 1]`` / ``indices [E]`` is the destination-row CSR whose column ids are
  *local* source ids in ``[0, num_src)`` -- exactly the forward CSR our kernels read.

``fanouts[i]`` is the fanout of block ``i`` (outermost first), as in
``dgl.dataloading.NeighborSampler``; -1 takes every in-neighbor. Sampling is without
replacement: a node with ``deg <= fanout`` keeps all its in-edges, otherwise ``fanout``
distinct ones chosen uniformly (random keys, segmented sort).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SampledBlock:
    indptr: torch.Tensor  # [num_dst + 1]
    indices: torch.Tensor  # [E] local source ids
    num_src: int
    fanout: int

    @property
    def num_dst(self) -> int:
        return self.indptr.numel() - 1

    @property
    def num_edges(self) -> int:
        return self.indices.numel()


@dataclass
class SampledBatch:
    seeds: torch.Tensor  # [B] global ids of the output nodes
    input_nodes: torch.Tensor  # [num_src of block 0] global ids, seeds first
    blocks: list[SampledBlock]  # outermost first


class GPUNeighborSampler:
    """Samples blocks from an in-neighbor CSR held on the GPU.

    Args:
        indptr: ``[N + 1]`` in-CSR row pointers (row v lists the sources of edges u->v).
        indices: ``[E]`` source ids.
        fanouts: per-layer fanouts, outermost layer first; -1 = all neighbors.
        index_dtype: dtype of the emitted block CSR (int32 is what our kernels run fastest on).
        generator: optional ``torch.Generator`` on the graph's device, for reproducible sampling.
    """

    def __init__(
        self,
        indptr: torch.Tensor,
        indices: torch.Tensor,
        fanouts: list[int],
        index_dtype: torch.dtype = torch.int32,
        generator: torch.Generator | None = None,
    ) -> None:
        self.indptr = indptr.to(torch.int64)
        self.indices = indices
        self.fanouts = list(fanouts)
        self.index_dtype = index_dtype
        self.generator = generator
        self.num_nodes = indptr.numel() - 1
        # Global id -> local id of the block being built; -1 = not in the block. Reset after
        # every layer by touching only the entries that were set, so it costs O(block), not O(N).
        self._local = torch.full((self.num_nodes,), -1, dtype=torch.int64, device=indptr.device)
        # Membership marks for the frontier dedup; all False between layers.
        self._mark = torch.zeros(self.num_nodes, dtype=torch.bool, device=indptr.device)

    @property
    def device(self) -> torch.device:
        return self.indptr.device

    def _sample_edges(self, dst: torch.Tensor, fanout: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Pick in-edges of each node in ``dst``. Returns (counts [n], global src ids [E'], row-major).

        One host sync (the two sizes, read together); everything else has static shapes, so
        no boolean-mask indexing (each of those is a hidden ``nonzero`` sync).
        """
        n = dst.numel()
        start = self.indptr[dst]
        deg = self.indptr[dst + 1] - start
        counts = deg if fanout < 0 else deg.clamp(max=fanout)
        row_start = torch.cumsum(counts, 0) - counts  # first output slot of each row

        if fanout < 0:
            total = int(counts.sum())
            if total == 0:
                return counts, self.indices.new_empty(0)
            row = torch.repeat_interleave(torch.arange(n, device=self.device), counts, output_size=total)
            edge_pos = start[row] + (torch.arange(total, device=self.device) - row_start[row])
            return counts, self.indices[edge_pos]

        # Rows with deg > fanout need a uniform subset; rows with deg <= fanout keep everything.
        # Only the over-full rows go through the random-key sort, but to avoid selecting them with
        # a mask we lay out *their* edges via repeat_interleave over a degree vector that is zero
        # for every other row.
        big_deg = torch.where(deg > fanout, deg, torch.zeros_like(deg))
        sizes = torch.stack([counts.sum(), big_deg.sum()]).tolist()  # the one sync
        total, big_total = int(sizes[0]), int(sizes[1])
        if total == 0:
            return counts, self.indices.new_empty(0)

        # Default: slot k of row r takes the row's k-th in-edge (exact for deg <= fanout).
        row = torch.repeat_interleave(torch.arange(n, device=self.device), counts, output_size=total)
        edge_pos = start[row] + (torch.arange(total, device=self.device) - row_start[row])

        if big_total:
            # Every in-edge of every over-full row, row-major; rank = position within its row.
            brow = torch.repeat_interleave(torch.arange(n, device=self.device), big_deg, output_size=big_total)
            bstart = torch.cumsum(big_deg, 0) - big_deg
            brank = torch.arange(big_total, device=self.device) - bstart[brow]
            # Sort by (row, random): an int64 key, row in the high bits, 31 random bits low.
            rnd = torch.randint(0, 2**31 - 1, (big_total,), device=self.device, generator=self.generator)
            order = torch.argsort((brow << 31) | rnd)
            # Sorting keeps rows contiguous, so sorted position j has in-row rank brank[j]: edge
            # order[j] is the brank[j]-th pick of its row. The first `fanout` picks are the sample.
            pick_rank = torch.empty_like(brank)
            pick_rank[order] = brank
            chosen = pick_rank < fanout
            # Scatter chosen edges into their row's output slots; the rest go to a dump slot.
            out_slot = torch.where(chosen, row_start[brow] + pick_rank, torch.full_like(brank, total))
            buf = torch.empty(total + 1, dtype=edge_pos.dtype, device=self.device)
            buf[:total] = edge_pos
            buf.scatter_(0, out_slot, start[brow] + brank)
            edge_pos = buf[:total]

        return counts, self.indices[edge_pos]

    def sample(self, seeds: torch.Tensor) -> SampledBatch:
        dst = seeds.to(device=self.device, dtype=torch.int64)
        blocks: list[SampledBlock] = []
        local = self._local
        for fanout in reversed(self.fanouts):
            n = dst.numel()
            counts, src = self._sample_edges(dst, fanout)
            src = src.to(torch.int64)

            # Destinations first (local ids 0..n-1), then sources not already present, numbered
            # n, n+1, ... in id order. Mark sampled sources in a global table, unmark the
            # destinations, and one nonzero (the layer's only other sync) yields the new nodes
            # sorted and deduplicated -- cheaper than torch.unique (sort + a size read + a count).
            mark = self._mark
            # index_fill_, not `mark[idx] = True`: assigning a Python scalar copies it host->device
            # first, and that copy blocks the host.
            mark.index_fill_(0, src, True)
            mark.index_fill_(0, dst, False)
            new = mark.nonzero(as_tuple=False).view(-1)
            mark.index_fill_(0, new, False)  # reset only what we touched
            local[dst] = torch.arange(n, device=self.device)
            local[new] = torch.arange(n, n + new.numel(), device=self.device)
            col = local[src]
            src_nodes = torch.cat([dst, new])
            local.index_fill_(0, src_nodes, -1)

            indptr = torch.zeros(n + 1, dtype=torch.int64, device=self.device)
            indptr[1:] = torch.cumsum(counts, 0)
            blocks.append(
                SampledBlock(
                    indptr=indptr.to(self.index_dtype),
                    indices=col.to(self.index_dtype),
                    num_src=src_nodes.numel(),
                    fanout=fanout,
                )
            )
            dst = src_nodes
        blocks.reverse()
        return SampledBatch(seeds=seeds, input_nodes=dst, blocks=blocks)


def seed_batches(
    train_idx: torch.Tensor, batch_size: int, shuffle: bool = True, generator: torch.Generator | None = None
) -> list[torch.Tensor]:
    """Split training node ids into mini-batches (last one may be short), shuffled per epoch."""
    if shuffle:
        perm = torch.randperm(train_idx.numel(), device=train_idx.device, generator=generator)
        train_idx = train_idx[perm]
    return list(torch.split(train_idx, batch_size))
