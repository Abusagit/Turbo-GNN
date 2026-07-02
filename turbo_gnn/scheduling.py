from __future__ import annotations

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
