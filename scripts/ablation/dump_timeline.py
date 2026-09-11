#!/usr/bin/env python3
"""Write the per-tick numbers behind one panel of the launch-mode figure.

A heatmap is 1600 pixels wide, so on a long run each pixel averages a hundred ticks or more and
anything shorter than that disappears.  On web-traffic the light bucket's drain is 46 ticks
against a 244,231-tick run -- three tenths of one pixel -- and the figure shows an abrupt end
where the simulation actually has a normal, gradual drain.  This prints and saves the ticks so
the claim can be checked rather than believed.

    python scripts/ablation/dump_timeline.py --dataset web-traffic --launch-mode sequential \
        --cost-model cost_models.json --conv gt --pass forward --head-dim 128 \
        --out reports/launch-modes/web-traffic-sequential.csv

The CSV has one row per tick: occupied slot fraction averaged over the SMs, how many blocks of
each kernel are resident, and how much work has retired.  ``--tail`` selects how many of the
final ticks to print; the CSV always holds the whole run.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
sys.path.append(str(Path(__file__).resolve().parent))

from simulate_load_imbalance import (  # noqa: E402
    BASELINE_ASSIGNMENT,
    BASELINE_HEAVY_SLICE,
    BASELINE_VERTICES_PER_BLOCK,
    add_common_arguments,
    anchor_tick,
    build_workloads,
    load_degrees,
    resolve_cost_model,
)

from turbo_gnn.simulation import (  # noqa: E402
    SimulationConfig,
    SimulationResult,
    bandwidth_cap_from_hardware,
    simulate,
)


def write_csv(path: Path, result: SimulationResult, tick_ns: float | None) -> None:
    occupancy = result.sm_utilisation.mean(axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        names = sorted(result.active_blocks)
        writer = csv.writer(stream)
        writer.writerow(["tick", "time_us", "slot_occupancy", *(f"active_{n}" for n in names), "retired_work"])
        for step in range(result.makespan):
            writer.writerow(
                [
                    step + 1,
                    f"{(step + 1) * tick_ns / 1e3:.4f}" if tick_ns else "",
                    f"{occupancy[step]:.6f}",
                    *(int(result.active_blocks[n][step]) for n in names),
                    int(result.retired_work[step]),
                ]
            )


def print_tail(result: SimulationResult, tick_ns: float | None, tail: int, every: int, exact: int = 20) -> None:
    """Print the last ticks, thinned by ``every`` except for the final ``exact`` of them.

    The drain can be five ticks long, so any thinning steps over it and reports a run that
    ended at full occupancy.  The last stretch always prints tick by tick.
    """
    occupancy = result.sm_utilisation.mean(axis=1)
    names = sorted(result.active_blocks)
    start = max(0, result.makespan - tail)
    exact_from = max(start, result.makespan - exact)
    steps = [*range(start, exact_from, every), *range(exact_from, result.makespan)]

    header = f"{'tick':>9}{'time, us':>11}{'occupancy':>11}" + "".join(f"{'blocks ' + n:>15}" for n in names)
    print(f"\nlast {result.makespan - start} ticks, every {every}, then every tick:\n{header}")
    for step in steps:
        time = f"{(step + 1) * tick_ns / 1e3:9.3f}" if tick_ns else f"{'':>9}"
        active = "".join(f"{int(result.active_blocks[n][step]):>15,}" for n in names)
        print(f"{step + 1:>9,}{time:>11}{occupancy[step]:>10.2%}{active}")


def print_phases(result: SimulationResult, tick_ns: float | None) -> None:
    """Where the machine fills, where it empties, and how wide those edges are in pixels."""
    occupancy = result.sm_utilisation.mean(axis=1)
    busy = occupancy > 0.5
    edges = np.flatnonzero(np.diff(busy.astype(np.int8)))
    pixel = result.makespan / 1600
    print(f"\nmakespan {result.makespan:,} ticks", end="")
    if tick_ns:
        print(f" = {result.makespan * tick_ns / 1e3:.1f} us", end="")
    print(f";  one pixel of a 1600-wide figure = {pixel:,.0f} ticks")
    print("transitions across half occupancy (tick, and how many pixels from the end):")
    for edge in edges:
        direction = "fills" if busy[edge + 1] else "empties"
        print(f"    {edge + 1:>9,}  {direction:<8} {(result.makespan - edge - 1) / pixel:8.2f} px from the end")
    if not len(edges):
        print("    none: occupancy never crosses half")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_arguments(parser)
    parser.add_argument("--launch-mode", choices=["single", "sequential", "concurrent"], default="sequential")
    parser.add_argument("--tail", type=int, default=300, help="How many of the final ticks to print")
    parser.add_argument("--every", type=int, default=10, help="Print every Nth tick of the tail")
    parser.add_argument("--out", type=Path, help="Write the whole run as CSV")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.occupancies = [max(args.occupancies)]
    cost_model, fit, _ = resolve_cost_model(args)
    degrees = load_degrees(args)
    all_degrees, _, heavy = degrees
    bandwidth_cap = bandwidth_cap_from_hardware(
        args.memory_bandwidth_gbps, args.feature_dim, args.dtype_bytes, args.memory_latency_ns
    )
    tick_ns, _ = anchor_tick(args, cost_model, fit, degrees, bandwidth_cap)

    workloads, kernels, light_run, heavy_run = build_workloads(
        args,
        cost_model,
        degrees,
        BASELINE_ASSIGNMENT,
        BASELINE_VERTICES_PER_BLOCK,
        None,
        BASELINE_HEAVY_SLICE,
        args.launch_mode,
        args.occupancies[0],
    )
    print(f"{args.dataset} {args.conv}/{args.pass_name}/d{args.head_dim} {args.launch_mode}")
    for name, blocks in sorted(workloads.items()):
        costs = np.array([block.cost for block in blocks])
        print(
            f"  {name:<6} {len(blocks):>10,} blocks  work {costs.sum():>13,} ticks  "
            f"longest block {costs.max():>9,}  shortest {costs.min():>5,}"
        )
    result = simulate(
        workloads,
        kernels,
        SimulationConfig(
            num_sms=args.sms,
            bandwidth_cap=bandwidth_cap,
            seed=args.seed,
            launch_mode=args.launch_mode,
            light_launch_latency=args.launch_latency,
            ns_per_tick=tick_ns,
        ),
    )
    print_phases(result, tick_ns)
    print_tail(result, tick_ns, args.tail, args.every)
    if args.out:
        write_csv(args.out, result, tick_ns)
        print(f"\nwrote {result.makespan:,} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
