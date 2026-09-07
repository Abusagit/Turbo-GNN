"""Fit the simulator's cost model to measured kernel times.

The load-imbalance simulator charges a node ``alpha + beta * degree`` ticks (see
:class:`turbo_gnn.simulation.CostModel`).  Assuming ``alpha = 2, beta = 1`` -- one tick for the
neighbour, one for the node's own feature load, one for the output store -- is defensible for
min-aggregation and wrong for everything else: GT and GATv2 read Q/K/V rows and attention
parameters in the prologue and write a logsumexp in the epilogue, and the backward pass walks
the neighbourhood twice.  Until those constants come from measurement, every simulated number
is comparable only to other simulated numbers.

The fit
-------
A kernel launch over a whole bucket costs a fixed amount to launch, plus a fixed amount per
node, plus a fixed amount per edge::

    t(N, E) = c + a * N + b * E      [nanoseconds]

so regressing measured launch time on the graph's node and edge counts identifies all three.
``c`` is a nuisance parameter -- launch and teardown overhead, which the simulator does not
model -- but it has to be in the regression rather than absorbed: without it, the smallest
graphs are pure overhead and drag ``a`` up by an order of magnitude.  Two useful things fall
out of one regression:

* ``alpha = a / b`` and ``beta = 1``, because the simulator's tick *is* one neighbour;
* ``ns_per_tick = b``, which is the ``timestep_duration_ns`` the bandwidth cap needs and which
  was previously a hand-typed 1.0.

Fitting is weighted by ``1 / t`` by default.  Measured times in this project span four orders
of magnitude (0.03 ms on cora to 385 ms on ogbn-products); an unweighted fit is a fit to
ogbn-products alone.  Relative weighting asks instead that the model be within a fixed
*percentage* on every graph, which is what makes it usable as a predictor.

What the residuals mean
-----------------------
``a`` and ``b`` are constants, so the model has no notion of cache locality.  Anything the
visit order buys through L2 reuse lands in the residual, and graphs whose working set dwarfs L2
are exactly where the fit will be worst.  :attr:`FitResult.worst_graphs` reports them, and that
list is a direct measure of how much of the real behaviour the simulator cannot see.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Literal, Sequence

import numpy as np

from turbo_gnn.simulation import CostModel

Weighting = Literal["relative", "uniform"]

SCHEMA = "turbo-gnn-cost-model/1"

MIN_EDGE_TIME_SHARE = 0.2
"""Below this, the kernel's time does not scale with degree and the tick loses its meaning.

min-aggregation's backward pass is the case in point: it propagates one gradient per output
node, to the argmin neighbour, so its cost is O(N) and not O(E).  Its measured time per node is
flat across four orders of magnitude of edge count, and no ``alpha + beta * degree`` model can
describe it."""

MAX_USABLE_REL_ERROR = 0.35
"""Median relative error above which a fit should not be used to predict wall clock."""


@dataclass(frozen=True)
class Measurement:
    """One timed kernel launch over a whole graph."""

    conv: str
    pass_name: str
    head_dim: int
    graph: str
    num_nodes: int
    num_edges: int
    time_ms: float

    @property
    def key(self) -> str:
        return f"{self.conv}/{self.pass_name}/{self.head_dim}"


@dataclass(frozen=True)
class FitResult:
    """Cost model fitted to one (conv, pass, head dim) cell."""

    alpha: float
    beta: float
    ns_per_tick: float
    ns_per_node: float
    ns_per_edge: float
    launch_overhead_ns: float
    r2: float
    median_rel_error: float
    p90_rel_error: float
    max_rel_error: float
    num_samples: int
    edge_time_share: float
    weighting: Weighting = "relative"
    note: str = ""
    worst_graphs: list[tuple[str, float]] = field(default_factory=list)

    @property
    def trustworthy(self) -> bool:
        """Whether this cell can be used to predict wall clock.  See :attr:`note` for why not."""
        return not self.note

    @property
    def cost_model(self) -> CostModel:
        return CostModel(alpha=self.alpha, beta=self.beta)

    def predict_ms(self, num_nodes: int, num_edges: int) -> float:
        """Measured wall clock this fit expects, launch overhead included."""
        return (self.launch_overhead_ns + self.ns_per_node * num_nodes + self.ns_per_edge * num_edges) / 1e6


def _nnls(design: np.ndarray, target: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Least squares over a handful of non-negative coefficients.

    With three variables the active set can simply be enumerated -- all eight subsets of
    coefficients pinned to zero -- which is exact and needs no solver.  Negative coefficients
    are not merely ugly here: a negative per-node cost would mean that adding an isolated node
    makes the kernel finish sooner.
    """
    num_coefficients = design.shape[1]
    weighted_design = design * weights[:, None]
    weighted_target = target * weights
    best: np.ndarray | None = None
    best_sse = np.inf

    for mask in range(1 << num_coefficients):
        free = [i for i in range(num_coefficients) if mask >> i & 1]
        solution = np.zeros(num_coefficients)
        if free:
            partial, *_ = np.linalg.lstsq(weighted_design[:, free], weighted_target, rcond=None)
            if np.any(partial < 0):
                continue
            solution[free] = partial
        sse = float(np.sum((weighted_design @ solution - weighted_target) ** 2))
        if sse < best_sse:
            best, best_sse = solution, sse

    assert best is not None  # the all-zero solution is always feasible
    return best


def fit_cost_model(
    measurements: Sequence[Measurement],
    weighting: Weighting = "relative",
    fit_launch_overhead: bool = True,
) -> FitResult:
    """Regress measured launch time on node and edge counts.

    Args:
        measurements: Timings for one (conv, pass, head dim) cell, on at least three graphs.
        weighting: ``"relative"`` minimises percentage error (default, see module docstring);
            ``"uniform"`` minimises absolute error and is dominated by the largest graph.
        fit_launch_overhead: Fit the constant term.  Turning it off attributes launch overhead
            to ``alpha``, which is what makes an uncorrected fit on small graphs go wrong.

    Returns:
        FitResult: Fitted ``alpha``, the tick duration, and how well the three constants explain
        the measurements.

    Raises:
        ValueError: Too few measurements, a non-positive time, or a degenerate fit whose
            per-edge cost came out as zero (which leaves the tick undefined).
    """
    num_coefficients = 3 if fit_launch_overhead else 2
    if len(measurements) < num_coefficients:
        raise ValueError(
            f"need at least {num_coefficients} measurements to fit {num_coefficients} coefficients, "
            f"got {len(measurements)}"
        )
    if any(m.time_ms <= 0 for m in measurements):
        raise ValueError("measured times must be positive")

    counts = np.array([[m.num_nodes, m.num_edges] for m in measurements], dtype=np.float64)
    design = np.column_stack([np.ones(len(measurements)), counts]) if fit_launch_overhead else counts
    target = np.array([m.time_ms * 1e6 for m in measurements], dtype=np.float64)
    weights = 1.0 / target if weighting == "relative" else np.ones_like(target)

    solution = _nnls(design, target, weights)
    launch_overhead, ns_per_node, ns_per_edge = solution if fit_launch_overhead else (0.0, *solution)
    if ns_per_edge <= 0:
        raise ValueError(
            "fit produced a non-positive per-edge cost, so one tick has no duration; "
            "the measurements are probably all on graphs of near-identical edge count"
        )

    predicted = design @ solution
    relative_error = np.abs(predicted - target) / target
    weighted_mean = float(np.sum(weights**2 * target) / np.sum(weights**2))
    residual = float(np.sum((weights * (predicted - target)) ** 2))
    total = float(np.sum((weights * (target - weighted_mean)) ** 2))

    edge_time_share = float(np.mean(ns_per_edge * counts[:, 1] / predicted))
    median_rel_error = float(np.median(relative_error))
    notes = []
    if ns_per_node == 0:
        notes.append("per-node cost pinned to zero by the non-negativity constraint")
    if edge_time_share < MIN_EDGE_TIME_SHARE:
        notes.append(
            f"node-dominated: edges explain only {edge_time_share:.0%} of the predicted time, so a tick "
            "is not a neighbour here and alpha is not meaningful"
        )
    if median_rel_error > MAX_USABLE_REL_ERROR:
        notes.append(f"median error {median_rel_error:.0%} is too large to predict wall clock")

    worst = sorted(zip((m.graph for m in measurements), relative_error), key=lambda pair: -pair[1])
    return FitResult(
        alpha=float(ns_per_node / ns_per_edge),
        beta=1.0,
        ns_per_tick=float(ns_per_edge),
        ns_per_node=float(ns_per_node),
        ns_per_edge=float(ns_per_edge),
        launch_overhead_ns=float(launch_overhead),
        r2=1.0 - residual / total if total > 0 else 1.0,
        median_rel_error=median_rel_error,
        p90_rel_error=float(np.quantile(relative_error, 0.9)),
        max_rel_error=float(relative_error.max()),
        num_samples=len(measurements),
        edge_time_share=edge_time_share,
        weighting=weighting,
        note="; ".join(notes),
        worst_graphs=[(graph, float(error)) for graph, error in worst[:3]],
    )


def fit_all(
    measurements: Iterable[Measurement],
    weighting: Weighting = "relative",
    min_samples: int = 3,
    fit_launch_overhead: bool = True,
) -> tuple[dict[str, FitResult], dict[str, str]]:
    """Fit one cost model per (conv, pass, head dim).

    Returns:
        tuple: The fits, and the cells that could not be fitted mapped to the reason why.
    """
    cells: dict[str, list[Measurement]] = {}
    for measurement in measurements:
        cells.setdefault(measurement.key, []).append(measurement)

    fits: dict[str, FitResult] = {}
    skipped: dict[str, str] = {}
    for key, cell in sorted(cells.items()):
        if len(cell) < min_samples:
            skipped[key] = f"only {len(cell)} measurement(s), need {min_samples}"
            continue
        try:
            fits[key] = fit_cost_model(cell, weighting, fit_launch_overhead)
        except ValueError as error:
            skipped[key] = str(error)
    return fits, skipped


def save_cost_models(path: Path, fits: dict[str, FitResult], source: str, weighting: Weighting) -> None:
    payload = {
        "schema": SCHEMA,
        "source": source,
        "weighting": weighting,
        "models": {key: asdict(fit) for key, fit in sorted(fits.items())},
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def load_cost_models(path: Path) -> dict[str, FitResult]:
    payload = json.loads(Path(path).read_text())
    if payload.get("schema") != SCHEMA:
        raise ValueError(f"{path}: expected schema {SCHEMA!r}, got {payload.get('schema')!r}")
    return {
        key: FitResult(**(model | {"worst_graphs": [tuple(pair) for pair in model.get("worst_graphs", [])]}))
        for key, model in payload["models"].items()
    }


def lookup(fits: dict[str, FitResult], conv: str, pass_name: str, head_dim: int) -> FitResult:
    key = f"{conv}/{pass_name}/{head_dim}"
    if key not in fits:
        raise KeyError(f"no cost model for {key!r}; calibrated cells are {sorted(fits)}")
    return fits[key]


# --------------------------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------------------------

_GRAPH_HEADER = re.compile(r"^###\s+(\S+)")
_GRAPH_STATS = re.compile(r"^\s+N=([\d,]+)\s+E=([\d,]+)")
_CELL_ROW = re.compile(
    r"^\s+(?P<conv>\S+)\s+(?P<dim>\d+)\s+(?P<pass>forward|backward)\s+(?P<baseline>[\d.]+)\s+[\d.]+\s"
)


def measurements_from_kernel_benchmark_summary(path: Path) -> list[Measurement]:
    """Parse ``reports/kernel-benchmarks/summary.txt`` as written by ``benchmark_kernels.py``.

    The baseline column is the one to fit: it is ``one_per_block`` in natural node order, which
    is exactly the configuration the simulator reproduces as ``contiguous`` with one vertex per
    block.  The ``best`` column is a different schedule on every row and would fit nothing.
    """
    measurements: list[Measurement] = []
    graph: str | None = None
    num_nodes = num_edges = 0
    for line in Path(path).read_text().splitlines():
        header = _GRAPH_HEADER.match(line)
        if header:
            graph, num_nodes, num_edges = header.group(1), 0, 0
            continue
        stats = _GRAPH_STATS.match(line)
        if stats:
            num_nodes = int(stats.group(1).replace(",", ""))
            num_edges = int(stats.group(2).replace(",", ""))
            continue
        row = _CELL_ROW.match(line)
        if row and graph and num_edges:
            measurements.append(
                Measurement(
                    conv=row.group("conv"),
                    pass_name=row.group("pass"),
                    head_dim=int(row.group("dim")),
                    graph=graph,
                    num_nodes=num_nodes,
                    num_edges=num_edges,
                    time_ms=float(row.group("baseline")),
                )
            )
    return measurements


def measurements_from_benchmark_json(paths: Iterable[Path]) -> list[Measurement]:
    """Read records written by ``scripts/benchmark_kernels.py --json-out``.

    Accepts one record per file, a JSON array of records, or a ``.jsonl`` file.
    """
    measurements: list[Measurement] = []
    for path in paths:
        text = Path(path).read_text()
        if path.suffix == ".jsonl":
            records = [json.loads(line) for line in text.splitlines() if line.strip()]
        else:
            parsed = json.loads(text)
            records = parsed if isinstance(parsed, list) else [parsed]
        for record in records:
            stats = record.get("graph") or {}
            missing = [f for f in ("conv", "mode", "head_dim", "ms_per_iter") if record.get(f) is None]
            if missing or not stats.get("num_edges"):
                raise ValueError(f"{path}: record is missing {missing or ['graph.num_edges']}")
            measurements.append(
                Measurement(
                    conv=str(record["conv"]),
                    pass_name=str(record["mode"]),
                    head_dim=int(record["head_dim"]),
                    graph=str(record.get("dataset", path.stem)),
                    num_nodes=int(stats["num_nodes"]),
                    num_edges=int(stats["num_edges"]),
                    time_ms=float(record["ms_per_iter"]),
                )
            )
    return measurements


def measurements_from_csv(path: Path) -> list[Measurement]:
    """Read a CSV with columns conv, pass, head_dim, graph, num_nodes, num_edges, time_ms."""
    with Path(path).open(newline="") as stream:
        return [
            Measurement(
                conv=row["conv"],
                pass_name=row["pass"],
                head_dim=int(row["head_dim"]),
                graph=row.get("graph", "?"),
                num_nodes=int(row["num_nodes"]),
                num_edges=int(row["num_edges"]),
                time_ms=float(row["time_ms"]),
            )
            for row in csv.DictReader(stream)
        ]
