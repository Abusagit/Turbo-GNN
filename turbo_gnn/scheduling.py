from __future__ import annotations

import heapq

import numpy as np
import torch


def sort_by_degree_desc(
    node_indices: torch.Tensor,
    indptr: torch.Tensor,
) -> torch.Tensor:
    if indptr.dtype == torch.uint32:
        signed = indptr.view(torch.int32)
    elif indptr.dtype == torch.uint64:
        signed = indptr.view(torch.int64)
    else:
        signed = indptr

    degrees_full = signed[1:] - signed[:-1]
    degrees_bucket = degrees_full[node_indices.long()]
    order = torch.argsort(degrees_bucket, descending=True, stable=True)
    return node_indices[order].contiguous()


def edge_balanced_partition(
    node_indices: torch.Tensor,
    indptr: torch.Tensor,
    num_blocks: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    n = node_indices.shape[0]
    device = node_indices.device

    offsets = torch.zeros(num_blocks + 1, dtype=torch.int32, device=device)
    offsets[-1] = n

    if n == 0 or num_blocks <= 1:
        return node_indices, offsets

    signed_indptr = indptr
    if indptr.dtype == torch.uint32:
        signed_indptr = indptr.view(torch.int32)
    elif indptr.dtype == torch.uint64:
        signed_indptr = indptr.view(torch.int64)

    degrees = (signed_indptr[1:] - signed_indptr[:-1])[node_indices.long()]
    degrees_np = degrees.cpu().numpy().astype(np.int64)

    order = np.argsort(-degrees_np, kind="stable")

    heap = [(np.int64(0), b) for b in range(num_blocks)]
    heapq.heapify(heap)
    bin_lists: list[list[int]] = [[] for _ in range(num_blocks)]
    for pos in order:
        cur_sum, b = heapq.heappop(heap)
        bin_lists[b].append(int(pos))
        heapq.heappush(heap, (cur_sum + degrees_np[pos], b))

    reordered_positions = np.empty(n, dtype=np.int64)
    off = 0
    off_list = [0]
    for b in range(num_blocks):
        blen = len(bin_lists[b])
        reordered_positions[off : off + blen] = bin_lists[b]
        off += blen
        off_list.append(off)

    reordered = node_indices[torch.from_numpy(reordered_positions).to(device)]
    return reordered.contiguous(), torch.tensor(off_list, dtype=torch.int32, device=device)
