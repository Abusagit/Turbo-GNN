"""Degree buckets for the DRDo experiment; no changes to the CUDA graph ABI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from turbo_gnn.simulation import BlockSpec, CostModel, KernelConfig


@dataclass(frozen=True)
class DegreeBucket:
    node_ids: np.ndarray
    heavy: bool


def partition_degree_buckets(
    degrees: Sequence[int],
    count: int,
    strategy: Literal["quantile", "log", "tail"] = "quantile",
    heavy_quantile: float = 0.99,
    degree_ordering: bool = False,
) -> list[DegreeBucket]:
    """Partition degrees while preserving the resource class of every node.

    quantile/log split only light nodes. tail splits both sides of the heavy threshold
    using geometric tail probabilities (for count=4, q=.99: .9, .99, .999).

    Equal degrees always stay together; empty ranges are omitted. Natural node order
    is preserved inside every range unless degree_ordering is explicitly enabled.
    Buckets are returned from highest to lowest degree, as CUDA launch order.
    """
    values = np.asarray(degrees, dtype=np.int64)
    if values.ndim != 1 or np.any(values < 0):
        raise ValueError("degrees must be a one-dimensional non-negative sequence")
    if count < 2:
        raise ValueError("count must be at least 2")
    if not 0 < heavy_quantile < 1:
        raise ValueError("heavy_quantile must be in (0, 1)")
    if strategy not in ("quantile", "log", "tail"):
        raise ValueError(f"unknown strategy: {strategy}")
    if strategy == "tail" and count % 2:
        raise ValueError("tail requires an even count to preserve the heavy threshold")
    if not values.size:
        return []
    # Use the same degree threshold convention as graph.py (ties go to heavy).
    threshold = float(np.quantile(values.astype(np.float32), heavy_quantile))
    heavy = np.flatnonzero(values >= threshold)
    light = np.flatnonzero(values < threshold)
    groups = [DegreeBucket(heavy, True)] if heavy.size else []
    if strategy == "tail":
        quantiles = 1 - (1 - heavy_quantile) ** (2 * np.arange(1, count) / count)
        quantiles[count // 2 - 1] = heavy_quantile
        cuts = np.quantile(values.astype(np.float32), quantiles)
        labels = np.searchsorted(cuts, values, side="right")
        groups = []
        for label in range(count - 1, -1, -1):
            nodes = np.flatnonzero(labels == label)
            if nodes.size:
                groups.append(DegreeBucket(nodes, bool(values[nodes].min() >= threshold)))
    elif light.size:
        light_values = values[light]
        if strategy == "quantile":
            cuts = np.quantile(light_values, np.arange(1, count - 1) / (count - 1))
        else:
            # log1p handles isolated nodes without special buckets or log(0).
            cuts = np.expm1(np.linspace(np.log1p(light_values.min()), np.log1p(light_values.max()), count)[1:-1])
        labels = np.searchsorted(cuts, light_values, side="right")
        for label in range(count - 2, -1, -1):
            nodes = light[labels == label]
            if nodes.size:
                groups.append(DegreeBucket(nodes, False))
    if degree_ordering:
        groups = [
            DegreeBucket(bucket.node_ids[np.argsort(-values[bucket.node_ids], kind="stable")], bucket.heavy)
            for bucket in groups
        ]
    return groups


def build_bucket_workloads(
    degrees: Sequence[int],
    buckets: list[DegreeBucket],
    cost_model: CostModel,
    num_heads: int = 1,
    max_blocks_light: int = 8,
    max_blocks_heavy: int = 4,
    occupancy: float = 1.0,
) -> tuple[dict[str, list[BlockSpec]], dict[str, KernelConfig]]:
    """One block per node/head, with original node IDs and fixed resource classes."""
    if num_heads <= 0:
        raise ValueError("num_heads must be positive")
    values = np.asarray(degrees, dtype=np.int64)
    workloads, kernels = {}, {}
    for index, bucket in enumerate(buckets):
        name = f"bucket_{index}"
        workloads[name] = [
            BlockSpec(head * len(bucket.node_ids) + offset, name, (int(node),), (int(values[node]),), cost_model)
            for head in range(num_heads)
            for offset, node in enumerate(bucket.node_ids)
        ]
        kernels[name] = KernelConfig(name, max_blocks_heavy if bucket.heavy else max_blocks_light, occupancy)
    return workloads, kernels
