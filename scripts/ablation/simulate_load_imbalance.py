#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from itertools import product
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from turbo_gnn.calibration import load_cost_models, lookup  # noqa: E402
from turbo_gnn.graph import AdjacencyForwardBackwardWithNodeBuckets  # noqa: E402
from turbo_gnn.simulation import (  # noqa: E402
    CostModel,
    KernelConfig,
    SimulationConfig,
    bandwidth_cap_from_hardware,
    build_blocks,
    build_heavy_slices,
    simulate,
)

REAL_SOURCE = {
    "ogbn-arxiv": "ogbn",
    "ogbn-products": "ogbn",
    "web-traffic": "pyg",
    "hm-categories": "pyg",
    "city-roads-L": "pyg",
}


def make_powerlaw_degrees(n: int, avg_degree: int, exponent: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.zipf(exponent, n).astype(np.float64)
    raw *= (n * avg_degree) / max(raw.sum(), 1)
    degrees = np.floor(raw).astype(np.int64)
    remainder = n * avg_degree - int(degrees.sum())
    if remainder > 0:
        degrees[rng.choice(n, size=min(remainder, n), replace=False)] += 1
    return degrees


def load_real_degrees(
    name: str, root: str, quantile: float, direction: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Degrees and light/heavy buckets for one pass.

    The backward pass walks the transposed CSR -- rows are source nodes -- so its degrees and
    its bucketing are a different partition of the same graph, and only coincide when the graph
    is undirected. Simulating a backward launch on forward degrees would silently model the
    wrong workload.
    """
    from src.data.datasets import DatasetConfig, load_single_graph

    cfg = DatasetConfig(source=REAL_SOURCE[name], name=name, root=root, conv_backend="cuda")
    graph_data = load_single_graph(cfg)
    edge_index = graph_data.edge_index
    if not isinstance(edge_index, torch.Tensor):
        edge_index = torch.as_tensor(edge_index)
    graph = AdjacencyForwardBackwardWithNodeBuckets.from_edge_list(
        edge_index, int(graph_data.num_nodes), quantile=quantile, index_dtype=torch.int64
    )
    return bucket_degrees(graph, direction)


def bucket_degrees(graph, direction: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split one direction's CSR into all / light / heavy degrees."""
    if direction not in ("forward", "backward"):
        raise ValueError(f"unknown direction: {direction!r}")
    indptr = getattr(graph, f"{direction}_indptr")
    degrees = (indptr[1:] - indptr[:-1]).cpu().numpy()
    light = getattr(graph, f"{direction}_light_nodes").long().cpu().numpy()
    heavy = getattr(graph, f"{direction}_heavy_nodes").long().cpu().numpy()
    return degrees, degrees[light], degrees[heavy]


def load_degrees(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if args.dataset.startswith("synth-N"):
        n = int(args.dataset.removeprefix("synth-N"))
        all_degrees = make_powerlaw_degrees(n, args.avg_degree, args.exponent, args.seed)
        threshold = np.quantile(all_degrees, args.quantile) if args.quantile != -1 else np.inf
        return all_degrees, all_degrees[all_degrees < threshold], all_degrees[all_degrees >= threshold]
    if args.dataset in REAL_SOURCE:
        return load_real_degrees(args.dataset, args.data_root, args.quantile, args.pass_name)
    raise ValueError(f"unknown dataset {args.dataset!r}; use synth-N<size> or a known real graph")


def safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", value)


def save_trace(path: Path, result) -> None:
    payload = {
        "sm_utilisation": result.sm_utilisation.astype(np.float32),
        "bandwidth_utilisation": result.bandwidth_utilisation.astype(np.float32),
        "retired_work": result.retired_work,
    }
    payload.update({f"active_{name}": values for name, values in result.active_blocks.items()})
    np.savez_compressed(path, **payload)


def plot_result(path: Path, title: str, result) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(path.parent.parent / ".matplotlib"))
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to generate simulation plots") from exc

    time = np.arange(1, result.makespan + 1)
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), constrained_layout=True)
    axes[0].plot(time, result.sm_utilisation.mean(axis=1), linewidth=1)
    axes[0].set(ylabel="slot utilisation", ylim=(0, 1.02), title=title)
    axes[1].plot(time, result.bandwidth_utilisation, linewidth=1, color="tab:orange")
    axes[1].set(ylabel="HBM utilisation", ylim=(0, 1.02))
    image = axes[2].imshow(
        result.sm_utilisation.T, aspect="auto", interpolation="nearest", origin="lower", vmin=0, vmax=1
    )
    axes[2].set(xlabel="time unit", ylabel="SM", title="Per-SM occupancy")
    fig.colorbar(image, ax=axes[2], label="occupied resource fraction")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_comparison(path: Path, rows: list[dict[str, object]]) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(path.parent / ".matplotlib"))
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to generate simulation plots") from exc

    colours = {"contiguous": "tab:blue", "grid_strided": "tab:orange", "lpt": "tab:green"}
    fig, axes = plt.subplots(2, 1, figsize=(max(12, len(rows) * 0.16), 7), sharex=True, constrained_layout=True)
    for assignment, colour in colours.items():
        indices = [i for i, row in enumerate(rows) if row["assignment"] == assignment]
        axes[0].scatter(indices, [rows[i]["imbalance_ratio"] for i in indices], s=14, label=assignment, color=colour)
        axes[1].scatter(indices, [rows[i]["drain_tail"] for i in indices], s=14, color=colour)
    axes[0].set(ylabel="T / T*", title="Load-imbalance sweep")
    axes[0].legend(ncol=4)
    axes[1].set(xlabel="experiment id", ylabel="drain tail")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", default="synth-N65536")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--quantile", type=float, default=0.99)
    parser.add_argument("--avg-degree", type=int, default=8)
    parser.add_argument("--exponent", type=float, default=2.3)

    parser.add_argument(
        "--assignments",
        nargs="+",
        choices=["contiguous", "grid_strided", "lpt"],
        default=["contiguous", "grid_strided", "lpt"],
    )
    parser.add_argument("--vertices-per-block", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--num-blocks", type=int, default=1056, help="Number of grid-strided or LPT blocks")
    parser.add_argument(
        "--heavy-slice-sizes",
        type=int,
        nargs="+",
        default=[0, 256],
        help="0 means one block per heavy node",
    )
    parser.add_argument(
        "--launch-modes",
        nargs="+",
        choices=["single", "sequential", "concurrent"],
        default=["sequential", "concurrent"],
    )
    parser.add_argument("--launch-latency", type=int, default=5)

    parser.add_argument(
        "--cost-model",
        type=Path,
        help="cost_models.json from calibrate_cost_model.py; sets alpha, beta and the tick duration",
    )
    parser.add_argument("--conv", default="gt", help="Cost model cell to use from --cost-model")
    parser.add_argument(
        "--pass",
        dest="pass_name",
        default="forward",
        choices=["forward", "backward"],
        help="Selects both the cost model cell and which CSR the degrees come from",
    )
    parser.add_argument("--head-dim", type=int, default=128, help="Cost model cell to use from --cost-model")
    parser.add_argument("--alpha", type=float, help="Override the fitted per-node cost, in ticks")
    parser.add_argument("--beta", type=float, help="Override the fitted per-neighbour cost, in ticks")
    parser.add_argument(
        "--allow-untrustworthy-cost-model",
        action="store_true",
        help="Simulate a cell the calibration flagged as degenerate; expect a very slow, meaningless run",
    )

    parser.add_argument("--sms", type=int, default=132)
    parser.add_argument("--max-blocks-light", type=int, default=8)
    parser.add_argument("--max-blocks-heavy", type=int, default=4)
    parser.add_argument("--occupancies", type=float, nargs="+", default=[0.5, 0.75, 1.0])
    parser.add_argument("--memory-bandwidth-gbps", type=float, default=3350.0)
    parser.add_argument("--feature-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=1)
    parser.add_argument("--dtype-bytes", type=int, default=4)
    parser.add_argument(
        "--timestep-duration-ns",
        type=float,
        help="Duration of one simulator iteration in nanoseconds; default is the calibrated tick, or 1.0",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("simulation_results"))
    return parser.parse_args()


def resolve_cost_model(args: argparse.Namespace) -> tuple[CostModel, float, float | None, dict[str, object]]:
    """Pick the cost model and the tick duration, and describe where they came from.

    A calibrated cell supplies both: ``alpha`` from the regression, and the tick duration the
    bandwidth cap needs.  Without one the simulator falls back to ``D + 2`` on a nominal 1 ns
    tick -- self-consistent, but not a wall clock, so no predicted time is reported.

    Returns:
        tuple: The cost model, the tick duration to size the bandwidth cap with, the tick
        duration to report predicted times with (None when uncalibrated), and provenance for
        ``config.json``.
    """
    provenance: dict[str, object] = {"cost_model_source": "default D+2"}
    calibrated = False
    alpha, beta, tick = 2.0, 1.0, 1.0
    if args.cost_model:
        fit = lookup(load_cost_models(args.cost_model), args.conv, args.pass_name, args.head_dim)
        alpha, beta, tick = fit.alpha, fit.beta, fit.ns_per_tick
        calibrated = True
        provenance = {
            "cost_model_source": f"{args.cost_model}:{args.conv}/{args.pass_name}/{args.head_dim}",
            "cost_model_r2": fit.r2,
            "cost_model_median_rel_error": fit.median_rel_error,
            "cost_model_note": fit.note,
        }
        if fit.note and not args.allow_untrustworthy_cost_model:
            # Refuse rather than warn: a degenerate fit is not merely inaccurate. A cell whose
            # time does not scale with degree fits an enormous alpha, and every node then costs
            # thousands of ticks, so the sweep runs for hours to produce a meaningless number.
            raise SystemExit(
                f"{args.conv}/{args.pass_name}/{args.head_dim}: {fit.note}.\n"
                "Pass --allow-untrustworthy-cost-model to simulate it anyway."
            )
    if args.alpha is not None:
        alpha, provenance["cost_model_source"] = args.alpha, "command line"
    if args.beta is not None:
        beta, provenance["cost_model_source"] = args.beta, "command line"
    if args.timestep_duration_ns is not None:
        tick, calibrated = args.timestep_duration_ns, True
    if not calibrated:
        print(
            "warning: no --cost-model and no --timestep-duration-ns, so the tick has no duration; "
            "reporting ticks only. Run scripts/ablation/calibrate_cost_model.py to fix this.",
            file=sys.stderr,
        )
    provenance |= {"alpha": alpha, "beta": beta, "timestep_duration_ns": tick, "calibrated": calibrated}
    return CostModel(alpha=alpha, beta=beta), tick, tick if calibrated else None, provenance


def main() -> int:
    args = parse_args()
    cost_model, timestep_duration_ns, ns_per_tick, provenance = resolve_cost_model(args)
    all_degrees, light_degrees, heavy_degrees = load_degrees(args)
    args.out.mkdir(parents=True, exist_ok=True)
    trace_dir = args.out / "traces"
    plot_dir = args.out / "plots"
    trace_dir.mkdir(exist_ok=True)
    plot_dir.mkdir(exist_ok=True)

    bandwidth_cap = bandwidth_cap_from_hardware(
        args.memory_bandwidth_gbps,
        args.feature_dim,
        args.dtype_bytes,
        timestep_duration_ns,
    )
    rows: list[dict[str, object]] = []
    experiment = 0
    block_layouts: list[tuple[str, int | None, int | None]] = []
    for assignment in args.assignments:
        if assignment == "contiguous":
            block_layouts.extend((assignment, value, None) for value in args.vertices_per_block)
        else:
            block_layouts.append((assignment, None, args.num_blocks))

    for block_layout, heavy_slice, launch_mode, occupancy in product(
        block_layouts,
        args.heavy_slice_sizes,
        args.launch_modes,
        args.occupancies,
    ):
        assignment, vertices_per_block, block_count = block_layout
        if launch_mode == "single" and len(heavy_degrees):
            light_for_run = all_degrees
            heavy_for_run = np.empty(0, dtype=np.int64)
        else:
            light_for_run, heavy_for_run = light_degrees, heavy_degrees

        blocks = build_blocks(
            light_for_run,
            "light",
            assignment,
            1 if vertices_per_block is None else vertices_per_block,
            block_count,
            args.num_heads,
            cost_model,
        )
        workloads = {"light": blocks}
        if len(heavy_for_run):
            workloads["heavy"] = (
                build_heavy_slices(heavy_for_run, heavy_slice, num_heads=args.num_heads, cost_model=cost_model)
                if heavy_slice > 0
                else build_blocks(
                    heavy_for_run, "heavy", "contiguous", 1, num_heads=args.num_heads, cost_model=cost_model
                )
            )
        kernels = {
            "light": KernelConfig("light", args.max_blocks_light, occupancy),
        }
        if "heavy" in workloads:
            kernels["heavy"] = KernelConfig("heavy", args.max_blocks_heavy, occupancy)
        result = simulate(
            workloads,
            kernels,
            SimulationConfig(
                num_sms=args.sms,
                bandwidth_cap=bandwidth_cap,
                seed=args.seed,
                launch_mode=launch_mode,
                light_launch_latency=args.launch_latency,
                ns_per_tick=ns_per_tick,
            ),
        )
        layout_tag = f"v{vertices_per_block}" if assignment == "contiguous" else f"k{block_count}"
        tag = safe_name(f"{experiment:04d}_{assignment}_{layout_tag}_slice{heavy_slice}_{launch_mode}_occ{occupancy:g}")
        row = result.summary()
        row.update(
            experiment=experiment,
            tag=tag,
            assignment=assignment,
            heavy_slice_size=heavy_slice,
            occupancy=occupancy,
            light_nodes=len(light_for_run),
            heavy_nodes=len(heavy_for_run),
        )
        if assignment == "contiguous":
            row["vertices_per_block"] = vertices_per_block
        else:
            row["num_blocks"] = block_count
        rows.append(row)
        save_trace(trace_dir / f"{tag}.npz", result)
        plot_result(plot_dir / f"{tag}.png", tag, result)
        predicted = "" if result.predicted_ms is None else f" predicted={result.predicted_ms:.3f}ms"
        print(
            f"{tag}: T={result.makespan} T*={result.perfect_packing_time:.2f} "
            f"imbalance={result.imbalance_ratio:.3f} tail={result.drain_tail}{predicted}"
        )
        experiment += 1

    with (args.out / "summary.csv").open("w", newline="") as stream:
        fields = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {key: json.dumps(value) if isinstance(value, dict) else value for key, value in row.items()}
            )
    plot_comparison(args.out / "comparison.png", rows)
    metadata = vars(args).copy()
    metadata.update(
        dataset=str(args.dataset),
        light_nodes=len(light_degrees),
        heavy_nodes=len(heavy_degrees),
        bandwidth_cap=bandwidth_cap,
        **provenance,
    )
    metadata = {key: str(value) if isinstance(value, Path) else value for key, value in metadata.items()}
    (args.out / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
