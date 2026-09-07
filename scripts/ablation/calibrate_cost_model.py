#!/usr/bin/env python3
"""Fit the simulator's cost model to measured kernel times.

Produces the ``cost_models.json`` that ``simulate_load_imbalance.py --cost-model`` consumes,
turning simulated makespans into predicted milliseconds.  See :mod:`turbo_gnn.calibration` for
what is being fitted and why.

    # from the checked-in benchmark report
    python scripts/ablation/calibrate_cost_model.py \
        --from-summary reports/kernel-benchmarks/summary.txt --out cost_models.json

    # from fresh runs of benchmark_kernels.py --json-out
    python scripts/ablation/calibrate_cost_model.py --from-json runs/*.json --out cost_models.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from turbo_gnn.calibration import (  # noqa: E402
    Measurement,
    fit_all,
    measurements_from_benchmark_json,
    measurements_from_csv,
    measurements_from_kernel_benchmark_summary,
    save_cost_models,
)


def collect(args: argparse.Namespace) -> tuple[list[Measurement], str]:
    measurements: list[Measurement] = []
    sources: list[str] = []
    for path in args.from_summary:
        measurements += measurements_from_kernel_benchmark_summary(path)
        sources.append(str(path))
    if args.from_json:
        paths = [p for entry in args.from_json for p in (sorted(entry.glob("*.json")) if entry.is_dir() else [entry])]
        measurements += measurements_from_benchmark_json(paths)
        sources.append(f"{len(paths)} benchmark json record(s)")
    for path in args.from_csv:
        measurements += measurements_from_csv(path)
        sources.append(str(path))
    return measurements, ", ".join(sources)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--from-summary", type=Path, nargs="+", default=[], help="reports/kernel-benchmarks/summary.txt"
    )
    parser.add_argument(
        "--from-json", type=Path, nargs="+", default=[], help="benchmark_kernels.py --json-out files or dirs"
    )
    parser.add_argument("--from-csv", type=Path, nargs="+", default=[], help="conv,pass,head_dim,... CSV")
    parser.add_argument("--weighting", choices=["relative", "uniform"], default="relative")
    parser.add_argument("--min-samples", type=int, default=3)
    parser.add_argument("--out", type=Path, default=Path("cost_models.json"))
    args = parser.parse_args()
    if not (args.from_summary or args.from_json or args.from_csv):
        parser.error("give at least one of --from-summary / --from-json / --from-csv")
    return args


def main() -> int:
    args = parse_args()
    measurements, source = collect(args)
    if not measurements:
        print("no measurements found", file=sys.stderr)
        return 1

    fits, skipped = fit_all(measurements, args.weighting, args.min_samples)
    if not fits:
        print(f"no cell could be fitted from {len(measurements)} measurement(s): {skipped}", file=sys.stderr)
        return 1

    print(f"{len(measurements)} measurements from {source}, {len(fits)} cell(s) fitted\n")
    header = f"{'cell':<24}{'alpha':>9}{'ns/tick':>9}{'launch us':>11}{'R2':>7}{'med err':>9}{'edge%':>7}"
    print(f"{header}   worst graph")
    for key, fit in fits.items():
        worst = f"{fit.worst_graphs[0][0]} {fit.worst_graphs[0][1]:.0%}" if fit.worst_graphs else "-"
        print(
            f"{key:<24}{fit.alpha:>9.1f}{fit.ns_per_tick:>9.3f}{fit.launch_overhead_ns / 1e3:>11.1f}"
            f"{fit.r2:>7.2f}{fit.median_rel_error:>8.0%}{fit.edge_time_share:>7.0%}   {worst}"
        )
    unusable = {key: fit.note for key, fit in fits.items() if fit.note}
    if unusable:
        print("\ndo not predict wall clock from these cells:")
        for key, note in unusable.items():
            print(f"  {key}: {note}")
    for key, reason in sorted(skipped.items()):
        print(f"skipped {key}: {reason}")

    save_cost_models(args.out, fits, source, args.weighting)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
