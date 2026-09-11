#!/usr/bin/env python3
"""Every graph, every calibrated kernel, every launch mode -- as one table.

The launch-mode figure answers one cell at a time, and the full grid is 14 graphs by 12
(conv, pass, head dim) cells by two quantiles by five modes.  As images that is unreadable; as
rows it is a table you can sort, and it is what says whether a conclusion drawn from ogbn-arxiv
holds anywhere else.

    python scripts/ablation/sweep_all.py --cost-model cost_models.json --out reports/sweep.csv

Costly and resumable, in that order.  A run with no slicing lasts about as long as the graph's
largest node, and web-fraud's is 228,991 ticks; the whole grid is hours.  Rows already in the
output are skipped on a restart, so the job can be stopped and continued, and every row is
flushed as it is produced.

Cells the calibration flagged as degenerate are skipped: their alpha is not meaningful and a
run would be both very slow and misleading.  --include-untrustworthy overrides that.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
sys.path.append(str(Path(__file__).resolve().parent))

from simulate_load_imbalance import (  # noqa: E402
    BASELINE_ASSIGNMENT,
    BASELINE_LAUNCH_MODE,
    BASELINE_VERTICES_PER_BLOCK,
    add_common_arguments,
    build_workloads,
    dependencies,
    load_degrees,
    resolve_heavy_slice,
)

from turbo_gnn.calibration import anchor_tick_ns, load_cost_models  # noqa: E402
from turbo_gnn.simulation import (  # noqa: E402
    CostModel,
    SimulationConfig,
    bandwidth_cap_from_hardware,
    simulate,
)

DEFAULT_GRAPHS = [
    "cora",
    "citeseer",
    "pubmed",
    "city-roads-M",
    "artnet-exp",
    "tolokers-2",
    "city-roads-L",
    "ogbn-arxiv",
    "city-reviews",
    "twitch-views",
    "hm-categories",
    "avazu-ctr",
    "pokec-regions",
    "web-fraud",
]
FIELDS = [
    "graph",
    "conv",
    "pass",
    "head_dim",
    "quantile",
    "launch_mode",
    "split_k",
    "num_nodes",
    "num_edges",
    "max_degree",
    "skew",
    "heavy_nodes",
    "slice_size",
    "makespan",
    "perfect_packing_time",
    "binding_bound",
    "imbalance_ratio",
    "slot_bound",
    "bandwidth_bound",
    "critical_path",
    "drain_tail",
    "total_work",
    "mean_slot_occupancy",
    "ns_per_tick",
    "predicted_ms",
    "launch_overhead_ms",
    "alpha",
    "beta",
    "cost_model_median_rel_error",
    "seconds",
]


def run_one(args, cost_model, degrees, bandwidth_cap, mode, blocks_per_sm, tick_ns):
    slice_size = resolve_heavy_slice(0, blocks_per_sm, degrees[2], args.sms)
    workloads, kernels, _, _ = build_workloads(
        args,
        cost_model,
        degrees,
        BASELINE_ASSIGNMENT,
        BASELINE_VERTICES_PER_BLOCK,
        None,
        slice_size,
        mode,
        max(args.occupancies),
    )
    result = simulate(
        workloads,
        kernels,
        SimulationConfig(
            num_sms=args.sms,
            bandwidth_cap=bandwidth_cap,
            seed=args.seed,
            launch_mode=mode,
            light_launch_latency=args.launch_latency,
            depends_on=dependencies(workloads, mode),
            ns_per_tick=tick_ns,
            record_history=False,
        ),
    )
    return result, slice_size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_arguments(parser)
    parser.add_argument("--graphs", nargs="+", default=DEFAULT_GRAPHS)
    parser.add_argument("--quantiles", type=float, nargs="+", default=[0.99, 0.95])
    parser.add_argument(
        "--launch-modes",
        nargs="+",
        choices=["single", "sequential", "concurrent"],
        default=["single", "sequential", "concurrent"],
    )
    parser.add_argument("--slice-blocks-per-sm", type=float, nargs="+", default=[0, 8])
    parser.add_argument("--cells", nargs="*", help="conv/pass/head_dim keys; default is every calibrated cell")
    parser.add_argument("--include-untrustworthy", action="store_true", help="Also run cells the fit flagged")
    parser.add_argument("--out", type=Path, default=Path("reports/sweep.csv"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.cost_model:
        print("--cost-model is required: without it the cells and the timebase are undefined", file=sys.stderr)
        return 1
    fits = load_cost_models(args.cost_model)
    cells = args.cells or sorted(fits)
    skipped = [c for c in cells if fits[c].note and not args.include_untrustworthy]
    cells = [c for c in cells if c not in skipped]
    if skipped:
        print(f"skipping {len(skipped)} flagged cell(s): {', '.join(skipped)}\n", file=sys.stderr)

    done: set[tuple[str, ...]] = set()
    if args.out.exists():
        with args.out.open(newline="") as stream:
            done = {tuple(r[k] for k in FIELDS[:7]) for r in csv.DictReader(stream)}
        print(f"resuming: {len(done)} row(s) already in {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    stream = args.out.open("a", newline="")
    writer = csv.DictWriter(stream, fieldnames=FIELDS)
    if not done:
        writer.writeheader()

    bandwidth_cap = bandwidth_cap_from_hardware(
        args.memory_bandwidth_gbps, args.feature_dim, args.dtype_bytes, args.memory_latency_ns
    )
    degree_cache: dict[tuple[str, str, float], tuple[np.ndarray, ...] | None] = {}

    for graph, cell, quantile in product(args.graphs, cells, args.quantiles):
        conv, pass_name, head_dim = cell.split("/")
        fit = fits[cell]
        args.dataset, args.quantile, args.pass_name = graph, quantile, pass_name
        args.conv, args.head_dim = conv, int(head_dim)
        cost_model = CostModel(alpha=fit.alpha, beta=fit.beta)

        key = (graph, pass_name, quantile)
        if key not in degree_cache:
            try:
                degree_cache[key] = load_degrees(args)
            except Exception as error:  # noqa: BLE001 - a graph that will not load must not end the sweep
                print(f"{graph}: unavailable -- {type(error).__name__}: {str(error)[:90]}", file=sys.stderr)
                degree_cache[key] = None
        degrees = degree_cache[key]
        if degrees is None:
            continue
        all_degrees = degrees[0]

        # The tick is anchored on the measured baseline, so it is one run per (graph, cell,
        # quantile) -- and that run is also one of the rows, so it is not wasted.
        tick_ns = None
        for mode, blocks_per_sm in product(args.launch_modes, args.slice_blocks_per_sm):
            if mode == "single" and blocks_per_sm > 0:
                continue
            identity = (graph, conv, pass_name, head_dim, str(quantile), mode, str(blocks_per_sm > 0))
            if identity in done:
                continue
            if tick_ns is None:
                started = time.monotonic()
                baseline, _ = run_one(args, cost_model, degrees, bandwidth_cap, BASELINE_LAUNCH_MODE, 0, None)
                tick_ns = anchor_tick_ns(fit, len(all_degrees), int(all_degrees.sum()), baseline.makespan)
                print(f"  {graph}/{cell}/q{quantile}: anchored in {time.monotonic() - started:.0f}s")

            started = time.monotonic()
            result, slice_size = run_one(args, cost_model, degrees, bandwidth_cap, mode, blocks_per_sm, tick_ns)
            elapsed = time.monotonic() - started
            row = {
                "graph": graph,
                "conv": conv,
                "pass": pass_name,
                "head_dim": head_dim,
                "quantile": quantile,
                "launch_mode": mode,
                "split_k": blocks_per_sm > 0,
                "num_nodes": len(all_degrees),
                "num_edges": int(all_degrees.sum()),
                "max_degree": int(all_degrees.max()),
                "skew": round(float(all_degrees.max() / all_degrees.mean()), 1),
                "heavy_nodes": len(degrees[2]),
                "slice_size": slice_size,
                "makespan": result.makespan,
                "perfect_packing_time": round(result.perfect_packing_time, 2),
                "binding_bound": result.binding_bound,
                "imbalance_ratio": round(result.imbalance_ratio, 4),
                "slot_bound": round(result.slot_bound, 2),
                "bandwidth_bound": round(result.bandwidth_bound, 2),
                "critical_path": result.critical_path,
                "drain_tail": result.drain_tail,
                "total_work": result.total_work,
                "mean_slot_occupancy": round(result.mean_slot_occupancy, 4),
                "ns_per_tick": round(tick_ns, 4),
                "predicted_ms": round((result.predicted_ms or 0.0) + fit.launch_overhead_ns / 1e6, 6),
                "launch_overhead_ms": round(fit.launch_overhead_ns / 1e6, 6),
                "alpha": round(fit.alpha, 4),
                "beta": fit.beta,
                "cost_model_median_rel_error": round(fit.median_rel_error, 4),
                "seconds": round(elapsed, 1),
            }
            writer.writerow(row)
            stream.flush()
            print(
                f"  {graph:<14} {cell:<22} q{quantile} {mode:<11} split-K={str(blocks_per_sm > 0):<5} "
                f"T={result.makespan:>8,} ({result.binding_bound}) ratio={result.imbalance_ratio:.3f} "
                f"[{elapsed:.0f}s]"
            )
    stream.close()
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
