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
import json
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
        paths = [
            p
            for entry in args.from_json
            for p in (sorted([*entry.glob("*.json"), *entry.glob("*.jsonl")]) if entry.is_dir() else [entry])
        ]
        measurements += measurements_from_benchmark_json(paths)
        sources.append(f"{len(paths)} benchmark json record(s)")
    for path in args.from_csv:
        measurements += measurements_from_csv(path)
        sources.append(str(path))
    return measurements, ", ".join(sources)


def measurement_metadata(paths: list[Path]) -> dict[str, object]:
    """Keep device/config provenance and refuse to fit incompatible measurements together."""
    records: list[dict[str, object]] = []
    for entry in paths:
        for path in sorted([*entry.glob("*.json"), *entry.glob("*.jsonl")]) if entry.is_dir() else [entry]:
            if path.suffix == ".jsonl":
                records.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
            else:
                data = json.loads(path.read_text())
                records.extend(data if isinstance(data, list) else [data])
    metadata = {}
    for field in ("gpu", "sms", "heads", "dtype", "quantile"):
        values = {record[field] for record in records if field in record}
        if len(values) > 1:
            raise ValueError(f"mixed {field} in measurements: {values}; calibrate each configuration separately")
        if values and all(field in record for record in records):
            metadata[field] = next(iter(values))
    return metadata


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
    metadata = measurement_metadata(args.from_json)
    measurements, source = collect(args)
    if args.from_summary or args.from_csv:
        metadata = {}  # These legacy formats do not carry device/config provenance.
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

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_cost_models(args.out, fits, source, args.weighting)
    payload = json.loads(args.out.read_text())
    payload["measurement_metadata"] = metadata
    payload["measurement_files"] = [str(path) for path in args.from_json]
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
