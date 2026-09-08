from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass, field
from math import ceil, floor
from typing import Literal, Mapping, Sequence

import numpy as np

Assignment = Literal["contiguous", "grid_strided", "lpt"]
LaunchMode = Literal["single", "sequential", "concurrent"]


@dataclass(frozen=True)
class CostModel:
    """Thread-block lifetime as ``alpha + beta * degree`` ticks per node.

    One tick is the time to process one neighbour, so ``beta`` is 1 by construction for a
    calibrated model and ``alpha`` carries the prologue and epilogue: loading the node's own
    feature row, and writing its output back to HBM.  ``alpha = 2, beta = 1`` reproduces the
    original ``D + 2`` model exactly.

    Real convolutions have a much heavier prologue than min-aggregation -- GT and GATv2 read
    Q/K/V rows and attention parameters and write a logsumexp -- so ``alpha`` is fitted per
    (conv, pass, head dim) by :mod:`turbo_gnn.calibration` rather than assumed.
    """

    alpha: float = 2.0
    beta: float = 1.0

    def __post_init__(self) -> None:
        if self.alpha < 0:
            raise ValueError("alpha must be non-negative")
        if self.beta <= 0:
            raise ValueError("beta must be positive")

    def node_cost(self, degree: int) -> float:
        return self.alpha + self.beta * degree

    def block_cost(self, degrees: Sequence[int]) -> int:
        """Ticks a block spends on ``degrees``, rounded to a whole tick and at least one."""
        return max(1, round(sum(self.node_cost(int(degree)) for degree in degrees)))


DEFAULT_COST_MODEL = CostModel()


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
    cost_model: CostModel = DEFAULT_COST_MODEL

    @property
    def cost(self) -> int:
        return self.cost_model.block_cost(self.degrees)


@dataclass
class SimulationConfig:
    num_sms: int = 132
    bandwidth_cap: int | None = None
    seed: int = 42
    launch_mode: LaunchMode = "single"
    light_launch_latency: int = 0
    record_history: bool = True
    """Keep the per-SM occupancy matrix.

    It is what the heatmap draws, and it is [makespan x num_sms] -- 242 MB on a run of a quarter
    million ticks. A sweep that only wants the scalars should turn it off; everything else in
    the result is one-dimensional and stays."""

    depends_on: dict[str, str] = field(default_factory=dict)
    """Kernels that cannot start until another kernel has fully drained.

    A separate launch is a global barrier, so split-K's merge pass cannot begin before the last
    slice retires however much of the device is idle.  That wait is the price the split pays for
    the balance it buys, and leaving it out would make slicing look free."""

    ns_per_tick: float | None = None
    """Wall-clock duration of one tick, from :func:`turbo_gnn.calibration.fit_cost_model`.

    Set it and the makespan becomes a predicted time in nanoseconds, directly comparable to a
    measured kernel time.  Left unset, the simulation is only comparable to itself."""


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
    mean_slot_occupancy: float = 0.0
    slot_bound: float = 0.0
    bandwidth_bound: float = 0.0
    critical_path: int = 0
    ns_per_tick: float | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def binding_bound(self) -> str:
        """Which of the three lower bounds decides ``perfect_packing_time``.

        Reading ``imbalance_ratio`` without this is a trap.  When the critical path binds, the
        ratio is measuring one enormous node that no scheduling policy can split, not packing.
        """
        bounds = {"slot": self.slot_bound, "bandwidth": self.bandwidth_bound, "critical_path": self.critical_path}
        return max(bounds, key=lambda name: bounds[name])

    @property
    def predicted_ms(self) -> float | None:
        """Makespan in milliseconds, or None when the timebase was never calibrated."""
        if self.ns_per_tick is None:
            return None
        return self.makespan * self.ns_per_tick / 1e6

    def summary(self) -> dict[str, object]:
        result: dict[str, object] = {
            "makespan": self.makespan,
            "perfect_packing_time": self.perfect_packing_time,
            "imbalance_ratio": self.imbalance_ratio,
            "drain_tail": self.drain_tail,
            "retired_95_time": self.retired_95_time,
            "total_work": self.total_work,
            "bandwidth_cap": self.bandwidth_cap,
            "slot_bound": self.slot_bound,
            "bandwidth_bound": self.bandwidth_bound,
            "critical_path": self.critical_path,
            "binding_bound": self.binding_bound,
            "predicted_ms": self.predicted_ms,
            "mean_slot_utilisation": self.mean_slot_occupancy,
            "mean_bandwidth_utilisation": (
                float(self.bandwidth_utilisation.mean()) if self.bandwidth_utilisation.size else 0.0
            ),
        }
        result.update(self.metadata)
        return result


def bandwidth_cap_from_hardware(
    bandwidth_gbps: float,
    feature_dim: int,
    dtype_bytes: int,
    memory_latency_ns: float,
) -> int:
    """How many feature-row fetches the memory system keeps in flight at once.

    By Little's law, a system delivering ``bandwidth`` at ``latency`` has
    ``bandwidth * latency`` bytes outstanding at any moment; divided by the row size, that is
    the number of blocks that can be making progress simultaneously -- which is exactly what
    the simulator's per-tick cap means.

    This deliberately does not involve the tick duration.  Sizing the cap as "rows per tick"
    and then deriving the tick from a measured per-edge time counts the machine's parallelism
    twice: the measurement is an aggregate rate over thousands of concurrent blocks, so the
    concurrency is already inside it.
    """
    if min(bandwidth_gbps, feature_dim, dtype_bytes, memory_latency_ns) <= 0:
        raise ValueError("bandwidth inputs must be positive")
    bytes_in_flight = bandwidth_gbps * memory_latency_ns  # GB/s == bytes/ns
    bytes_per_row = feature_dim * dtype_bytes
    return max(1, floor(bytes_in_flight / bytes_per_row))


def _validate_degrees(degrees: Sequence[int]) -> np.ndarray:
    degree_array = np.asarray(degrees, dtype=np.int64)
    if degree_array.ndim != 1 or np.any(degree_array < 0):
        raise ValueError("degrees must be a one-dimensional non-negative sequence")
    return degree_array


def _lpt_bins(degrees: np.ndarray, num_blocks: int, cost_model: CostModel) -> list[list[int]]:
    costs = cost_model.alpha + cost_model.beta * degrees.astype(np.float64)
    bins: list[list[int]] = [[] for _ in range(num_blocks)]
    loads = [(0.0, block_id) for block_id in range(num_blocks)]
    heapq.heapify(loads)
    for node in np.argsort(-costs, kind="stable"):
        load, block_id = heapq.heappop(loads)
        bins[block_id].append(int(node))
        heapq.heappush(loads, (load + float(costs[node]), block_id))
    return bins


def build_blocks(
    degrees: Sequence[int],
    kernel: str,
    assignment: Assignment = "contiguous",
    vertices_per_block: int = 1,
    num_blocks: int | None = None,
    num_heads: int = 1,
    cost_model: CostModel = DEFAULT_COST_MODEL,
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
        groups = _lpt_bins(degree_array, num_blocks, cost_model)
    else:
        raise ValueError(f"unknown assignment: {assignment}")

    groups = [g for g in groups if g]
    return [
        BlockSpec(
            head * len(groups) + block_id,
            kernel,
            tuple(g),
            tuple(int(degree_array[node]) for node in g),
            cost_model,
        )
        for head in range(num_heads)
        for block_id, g in enumerate(groups)
    ]


def build_heavy_slices(
    degrees: Sequence[int],
    slice_size: int,
    kernel: str = "heavy",
    num_heads: int = 1,
    cost_model: CostModel = DEFAULT_COST_MODEL,
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
                sizes = [min(slice_size, int(degree) - start) for start in range(0, int(degree), slice_size)]
            for size in sizes:
                blocks.append(BlockSpec(len(blocks), kernel, (node,), (size,), cost_model))
    return blocks


def slices_per_node(degree: int, slice_size: int) -> int:
    """Blocks :func:`build_heavy_slices` makes for one node.  An isolated node still gets one."""
    if slice_size <= 0:
        raise ValueError("slice_size must be positive")
    return max(1, ceil(int(degree) / slice_size))


def slice_size_for_blocks_per_sm(degrees: Sequence[int], blocks_per_sm: float, num_sms: int) -> int:
    """Slice size that fills the device with ``blocks_per_sm`` blocks per SM.

    Ported from the measured result in the split-K work: sizing a slice from a degree statistic
    does not transfer between graphs, because graphs wanting similar slices have similar heavy
    *edge counts* rather than similar degrees.  Targeting a block count does transfer, and is
    device-relative rather than tied to one card's SM count.

    Returns 0, meaning "do not slice", for a non-positive target or an empty bucket.
    """
    edges = int(np.asarray(degrees, dtype=np.int64).sum()) if len(degrees) else 0
    if blocks_per_sm <= 0 or edges <= 0 or num_sms <= 0:
        return 0
    return max(1, round(edges / (blocks_per_sm * num_sms)))


def build_slice_merge(
    degrees: Sequence[int],
    slice_size: int,
    kernel: str = "merge",
    num_heads: int = 1,
    cost_model: CostModel = DEFAULT_COST_MODEL,
) -> list[BlockSpec]:
    """The second launch that combines a sliced node's partial results.

    Split-K's slice kernel stops before normalising and writes per-slice partial state; a merge
    kernel then runs the same n-way reduction across a node's slices.  Its grid is one block per
    heavy node, and each block reads one partial per slice, so a partial costs what a neighbour
    costs and the node's merge is ``alpha + beta * slice_count``.

    This is a separate launch, which is why :attr:`SimulationConfig.depends_on` exists: no merge
    block may start until the last slice has retired.
    """
    degree_array = _validate_degrees(degrees)
    if num_heads <= 0:
        raise ValueError("num_heads must be positive")
    counts = [slices_per_node(int(degree), slice_size) for degree in degree_array]
    return [
        BlockSpec(head * len(counts) + node, kernel, (node,), (count,), cost_model)
        for head in range(num_heads)
        for node, count in enumerate(counts)
    ]


@dataclass
class _Running:
    spec: BlockSpec
    remaining: int
    sm: int


def simulate(
    workloads: Mapping[str, Sequence[BlockSpec]],
    kernels: Mapping[str, KernelConfig],
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
    queues = {name: deque(sorted(blocks, key=lambda block: block.block_id)) for name, blocks in workloads.items()}

    total_work = sum(block.cost for blocks in workloads.values() for block in blocks)
    bandwidth_cap = config.bandwidth_cap
    if bandwidth_cap is None:
        bandwidth_cap = max(1, total_work)
    if bandwidth_cap <= 0:
        raise ValueError("bandwidth_cap must be positive")

    unknown_dependency = {k: v for k, v in config.depends_on.items() if k not in workloads or v not in workloads}
    if unknown_dependency:
        raise ValueError(f"depends_on names a kernel that has no workload: {unknown_dependency}")

    kernel_names = list(workloads)
    for preferred in ("merge", "heavy"):
        if preferred in kernel_names:
            kernel_names.remove(preferred)
            kernel_names.insert(0, preferred)
    running: list[_Running] = []
    sm_load = np.zeros(config.num_sms, dtype=np.float64)
    sm_history: list[np.ndarray] = []
    occupancy_total = 0.0
    bw_history: list[float] = []
    retired_history: list[int] = []
    active_history: dict[str, list[int]] = {name: [] for name in kernel_names}
    processed_work = 0
    retired_work = 0
    steps = 0
    time = 0
    next_kernel_index = 0

    def has_pending(name: str) -> bool:
        return bool(queues[name] or any(block.spec.kernel == name for block in running))

    def released(name: str) -> bool:
        blocker = config.depends_on.get(name)
        if blocker is not None and blocker in queues and has_pending(blocker):
            return False
        if config.launch_mode == "concurrent" and name not in ("heavy", "merge"):
            return time >= config.light_launch_latency
        if config.launch_mode == "sequential" and name not in ("heavy", "merge") and "heavy" in workloads:
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
            light_is_queued = any(queues[name] for name in kernel_names if name not in ("heavy", "merge"))
            if config.launch_mode == "concurrent" and light_is_queued:
                future.append(config.light_launch_latency)
            if not future:
                raise RuntimeError("simulation deadlocked")
            target = max(time + 1, min(future))
            while time < target:
                if config.record_history:
                    sm_history.append(np.zeros(config.num_sms, dtype=np.float64))
                steps += 1
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
        if config.record_history:
            sm_history.append(sm_load.copy())
        occupancy_total += float(sm_load.sum())
        steps += 1
        bw_history.append(len(granted) / bandwidth_cap)
        retired_history.append(retired_work)
        for name in kernel_names:
            active_history[name].append(sum(block.spec.kernel == name for block in running))

        for running_index in sorted(finished_indices, reverse=True):
            item = running.pop(running_index)
            name = item.spec.kernel
            sm_load[item.sm] -= 1.0 / kernels[name].resident_blocks_per_sm
        time += 1

    makespan = steps
    retired = np.asarray(retired_history, dtype=np.int64)
    threshold = 0.95 * total_work
    retired_95 = int(np.searchsorted(retired, threshold, side="left") + 1) if total_work else 0
    slot_lower_bound = (
        sum(block.cost / kernels[name].resident_blocks_per_sm for name, blocks in workloads.items() for block in blocks)
        / config.num_sms
    )
    bandwidth_lower_bound = total_work / bandwidth_cap
    critical_path = max((block.cost for blocks in workloads.values() for block in blocks), default=0)
    perfect_packing_time = max(slot_lower_bound, bandwidth_lower_bound, critical_path)
    return SimulationResult(
        makespan=makespan,
        perfect_packing_time=perfect_packing_time,
        slot_bound=slot_lower_bound,
        bandwidth_bound=bandwidth_lower_bound,
        critical_path=critical_path,
        imbalance_ratio=makespan / perfect_packing_time if perfect_packing_time else 1.0,
        drain_tail=makespan - retired_95,
        retired_95_time=retired_95,
        total_work=total_work,
        bandwidth_cap=bandwidth_cap,
        sm_utilisation=np.stack(sm_history) if sm_history else np.empty((0, config.num_sms)),
        mean_slot_occupancy=occupancy_total / (steps * config.num_sms) if steps else 0.0,
        bandwidth_utilisation=np.asarray(bw_history),
        retired_work=retired,
        active_blocks={name: np.asarray(values, dtype=np.int64) for name, values in active_history.items()},
        ns_per_tick=config.ns_per_tick,
        metadata={
            "num_sms": config.num_sms,
            "seed": config.seed,
            "launch_mode": config.launch_mode,
            "light_launch_latency": config.light_launch_latency,
            "resident_blocks_per_sm": {name: kernel.resident_blocks_per_sm for name, kernel in kernels.items()},
            "cost_model": {
                name: {"alpha": block.cost_model.alpha, "beta": block.cost_model.beta}
                for name, blocks in workloads.items()
                for block in blocks[:1]
            },
        },
    )
