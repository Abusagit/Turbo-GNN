from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass, field
from math import ceil, floor
from typing import Literal, Sequence

import numpy as np


Assignment = Literal["contiguous", "grid_strided", "lpt"]
LaunchMode = Literal["single", "sequential", "concurrent"]


@dataclass(frozen=True)
class KernelConfig:
    name: str
    max_blocks_per_sm: int
    occupancy: float = 1.0

    @property
    def resident_blocks_per_sm(self) -> int:
        if self.max_blocks_per_sm <= 0:
            raise ValueError("max_blocks_per_sm must be positive")
        if not 0 < self.occupancy <= 1:
            raise ValueError("occupancy must be in (0, 1]")
        return max(1, floor(self.max_blocks_per_sm * self.occupancy + 1e-12))


@dataclass(frozen=True)
class BlockSpec:
    block_id: int
    kernel: str
    node_ids: tuple[int, ...]
    degrees: tuple[int, ...]

    @property
    def cost(self) -> int:
        return sum(d + 2 for d in self.degrees)


@dataclass
class SimulationConfig:
    num_sms: int = 132
    bandwidth_cap: int | None = None
    seed: int = 42
    launch_mode: LaunchMode = "single"
    light_launch_latency: int = 0


@dataclass
class SimulationResult:
    makespan: int
    perfect_packing_time: float
    imbalance_ratio: float
    drain_tail: int
    retired_95_time: int
    total_work: int
    bandwidth_cap: int
    sm_utilisation: np.ndarray
    bandwidth_utilisation: np.ndarray
    retired_work: np.ndarray
    active_blocks: dict[str, np.ndarray]
    metadata: dict[str, object] = field(default_factory=dict)

    def summary(self) -> dict[str, object]:
        result = {
            "makespan": self.makespan,
            "perfect_packing_time": self.perfect_packing_time,
            "imbalance_ratio": self.imbalance_ratio,
            "drain_tail": self.drain_tail,
            "retired_95_time": self.retired_95_time,
            "total_work": self.total_work,
            "bandwidth_cap": self.bandwidth_cap,
            "mean_slot_utilisation": float(self.sm_utilisation.mean()) if self.sm_utilisation.size else 0.0,
            "mean_bandwidth_utilisation": (
                float(self.bandwidth_utilisation.mean())
                if self.bandwidth_utilisation.size
                else 0.0
            ),
        }
        result.update(self.metadata)
        return result


def bandwidth_cap_from_hardware(
    bandwidth_gbps: float,
    feature_dim: int,
    dtype_bytes: int,
    timestep_duration_ns: float,
) -> int:
    """Calculate how many feature rows HBM can transfer in one timestep."""
    if min(bandwidth_gbps, feature_dim, dtype_bytes, timestep_duration_ns) <= 0:
        raise ValueError("bandwidth inputs must be positive")
    bytes_per_timestep = bandwidth_gbps * timestep_duration_ns  # GB/s == bytes/ns
    bytes_per_row = feature_dim * dtype_bytes
    return max(1, floor(bytes_per_timestep / bytes_per_row))


def _validate_degrees(degrees: Sequence[int]) -> np.ndarray:
    degree_array = np.asarray(degrees, dtype=np.int64)
    if degree_array.ndim != 1 or np.any(degree_array < 0):
        raise ValueError("degrees must be a one-dimensional non-negative sequence")
    return degree_array


def _lpt_bins(degrees: np.ndarray, num_blocks: int) -> list[list[int]]:
    bins: list[list[int]] = [[] for _ in range(num_blocks)]
    loads = [(0, block_id) for block_id in range(num_blocks)]
    heapq.heapify(loads)
    for node in np.argsort(-degrees, kind="stable"):
        load, block_id = heapq.heappop(loads)
        bins[block_id].append(int(node))
        heapq.heappush(loads, (load + int(degrees[node]), block_id))
    return bins


def build_blocks(
    degrees: Sequence[int],
    kernel: str,
    assignment: Assignment = "contiguous",
    vertices_per_block: int = 1,
    num_blocks: int | None = None,
    num_heads: int = 1,
) -> list[BlockSpec]:
    degree_array = _validate_degrees(degrees)
    if num_heads <= 0:
        raise ValueError("num_heads must be positive")
    num_nodes = len(degree_array)
    if vertices_per_block <= 0:
        raise ValueError("vertices_per_block must be positive")
    if num_blocks is None:
        num_blocks = max(1, ceil(num_nodes / vertices_per_block)) if num_nodes else 0
    if num_blocks < 0:
        raise ValueError("num_blocks must be non-negative")
    if num_nodes and num_blocks == 0:
        raise ValueError("num_blocks must be positive for a non-empty workload")

    if assignment == "contiguous":
        groups = [
            list(range(start, min(start + vertices_per_block, num_nodes)))
            for start in range(0, num_nodes, vertices_per_block)
        ]
    elif assignment == "grid_strided":
        groups = [list(range(block_id, num_nodes, num_blocks)) for block_id in range(num_blocks)]
    elif assignment == "lpt":
        groups = _lpt_bins(degree_array, num_blocks)
    else:
        raise ValueError(f"unknown assignment: {assignment}")

    groups = [g for g in groups if g]
    return [
        BlockSpec(
            head * len(groups) + block_id,
            kernel,
            tuple(g),
            tuple(int(degree_array[node]) for node in g),
        )
        for head in range(num_heads)
        for block_id, g in enumerate(groups)
    ]


def build_heavy_slices(
    degrees: Sequence[int], slice_size: int, kernel: str = "heavy", num_heads: int = 1
) -> list[BlockSpec]:
    degree_array = _validate_degrees(degrees)
    if slice_size <= 0:
        raise ValueError("slice_size must be positive")
    if num_heads <= 0:
        raise ValueError("num_heads must be positive")
    blocks: list[BlockSpec] = []
    for _ in range(num_heads):
        for node, degree in enumerate(degree_array):
            if degree == 0:
                sizes = [0]
            else:
                sizes = [
                    min(slice_size, int(degree) - start)
                    for start in range(0, int(degree), slice_size)
                ]
            for size in sizes:
                blocks.append(BlockSpec(len(blocks), kernel, (node,), (size,)))
    return blocks


@dataclass
class _Running:
    spec: BlockSpec
    remaining: int
    sm: int


def simulate(
    workloads: dict[str, Sequence[BlockSpec]],
    kernels: dict[str, KernelConfig],
    config: SimulationConfig,
) -> SimulationResult:
    if config.num_sms <= 0:
        raise ValueError("num_sms must be positive")
    unknown = set(workloads) - set(kernels)
    if unknown:
        raise ValueError(f"missing kernel configs: {sorted(unknown)}")
    if config.launch_mode not in {"single", "sequential", "concurrent"}:
        raise ValueError(f"unknown launch_mode: {config.launch_mode}")

    rng = np.random.default_rng(config.seed)
    queues = {
        name: deque(sorted(blocks, key=lambda block: block.block_id))
        for name, blocks in workloads.items()
    }

    total_work = sum(block.cost for blocks in workloads.values() for block in blocks)
    bandwidth_cap = config.bandwidth_cap
    if bandwidth_cap is None:
        bandwidth_cap = max(1, total_work)
    if bandwidth_cap <= 0:
        raise ValueError("bandwidth_cap must be positive")

    kernel_names = list(workloads)
    if "heavy" in kernel_names:
        kernel_names.remove("heavy")
        kernel_names.insert(0, "heavy")
    running: list[_Running] = []
    sm_load = np.zeros(config.num_sms, dtype=np.float64)
    sm_history: list[np.ndarray] = []
    bw_history: list[float] = []
    retired_history: list[int] = []
    active_history = {name: [] for name in kernel_names}
    processed_work = 0
    retired_work = 0
    time = 0
    next_kernel_index = 0

    def has_pending(name: str) -> bool:
        return bool(queues[name] or any(block.spec.kernel == name for block in running))

    def released(name: str) -> bool:
        if config.launch_mode == "concurrent" and name != "heavy":
            return time >= config.light_launch_latency
        if config.launch_mode == "sequential" and name != "heavy" and "heavy" in workloads:
            return not has_pending("heavy")
        return True

    def admit_one(name: str) -> bool:
        if not queues[name] or not released(name):
            return False
        weight = 1.0 / kernels[name].resident_blocks_per_sm
        eligible = np.flatnonzero(sm_load + weight <= 1.0 + 1e-12)
        if not len(eligible):
            return False
        sm = int(rng.choice(eligible))
        spec = queues[name].popleft()
        running.append(_Running(spec, spec.cost, sm))
        sm_load[sm] += weight
        return True

    while processed_work < total_work:
        made_progress = True
        while made_progress:
            made_progress = False
            for offset in range(len(kernel_names)):
                name = kernel_names[(next_kernel_index + offset) % len(kernel_names)]
                if admit_one(name):
                    made_progress = True
                    next_kernel_index = (kernel_names.index(name) + 1) % len(kernel_names)
                    break

        if not running:
            future = []
            light_is_queued = any(queues[name] for name in kernel_names if name != "heavy")
            if config.launch_mode == "concurrent" and light_is_queued:
                future.append(config.light_launch_latency)
            if not future:
                raise RuntimeError("simulation deadlocked")
            target = max(time + 1, min(future))
            while time < target:
                sm_history.append(np.zeros(config.num_sms, dtype=np.float64))
                bw_history.append(0.0)
                retired_history.append(retired_work)
                for name in kernel_names:
                    active_history[name].append(0)
                time += 1
            continue

        request_order = rng.permutation(len(running))
        granted = request_order[: min(bandwidth_cap, len(running))]
        finished_indices: list[int] = []
        for running_index in granted:
            item = running[int(running_index)]
            item.remaining -= 1
            processed_work += 1
            if item.remaining == 0:
                finished_indices.append(int(running_index))

        retired_work += sum(running[index].spec.cost for index in finished_indices)
        sm_history.append(sm_load.copy())
        bw_history.append(len(granted) / bandwidth_cap)
        retired_history.append(retired_work)
        for name in kernel_names:
            active_history[name].append(sum(block.spec.kernel == name for block in running))

        for running_index in sorted(finished_indices, reverse=True):
            item = running.pop(running_index)
            name = item.spec.kernel
            sm_load[item.sm] -= 1.0 / kernels[name].resident_blocks_per_sm
        time += 1

    makespan = len(sm_history)
    retired = np.asarray(retired_history, dtype=np.int64)
    threshold = 0.95 * total_work
    retired_95 = int(np.searchsorted(retired, threshold, side="left") + 1) if total_work else 0
    slot_lower_bound = sum(
        block.cost / kernels[name].resident_blocks_per_sm
        for name, blocks in workloads.items() for block in blocks
    ) / config.num_sms
    bandwidth_lower_bound = total_work / bandwidth_cap
    perfect_packing_time = max(slot_lower_bound, bandwidth_lower_bound)
    return SimulationResult(
        makespan=makespan,
        perfect_packing_time=perfect_packing_time,
        imbalance_ratio=makespan / perfect_packing_time if perfect_packing_time else 1.0,
        drain_tail=makespan - retired_95,
        retired_95_time=retired_95,
        total_work=total_work,
        bandwidth_cap=bandwidth_cap,
        sm_utilisation=np.stack(sm_history) if sm_history else np.empty((0, config.num_sms)),
        bandwidth_utilisation=np.asarray(bw_history),
        retired_work=retired,
        active_blocks={
            name: np.asarray(values, dtype=np.int64)
            for name, values in active_history.items()
        },
        metadata={
            "num_sms": config.num_sms,
            "seed": config.seed,
            "launch_mode": config.launch_mode,
            "light_launch_latency": config.light_launch_latency,
            "resident_blocks_per_sm": {name: kernel.resident_blocks_per_sm for name, kernel in kernels.items()},
        },
    )
