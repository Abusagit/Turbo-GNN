#!/usr/bin/env python3
"""Compare multiple degree buckets with ordering disabled (plus ordered controls)."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.append(str(Path(__file__).resolve().parent))
from simulate_load_imbalance import load_real_degrees, make_powerlaw_degrees  # noqa: E402

from turbo_gnn.bucket_simulation import build_bucket_workloads, partition_degree_buckets
from turbo_gnn.calibration import load_cost_models, lookup
from turbo_gnn.simulation import SimulationConfig, bandwidth_cap_from_hardware, simulate

# Main-text Tables 1--3, pp. 7--8 of 10897_Faster_Generalized_Messa.pdf.
PAPER_GRAPHS = [
    "artnet-exp",
    "avazu-ctr",
    "citeseer",
    "city-reviews",
    "city-roads-L",
    "city-roads-M",
    "cora",
    "hm-categories",
    "ogbn-arxiv",
    "ogbn-products",
    "ogbn-proteins",
    "pokec-regions",
    "pubmed",
    "tolokers-2",
    "twitch-views",
    "web-fraud",
]
# SXM 80GB profiles; PCIe / 40GB cards must supply explicit overrides.
HARDWARE = {"h100": (132, 3350.0), "a100": (108, 2039.0)}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graphs", nargs="+", default=["tolokers-2"], help="Names, or 'paper' for all 16")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--cost-model", type=Path, required=True)
    parser.add_argument("--hardware", choices=HARDWARE, required=True)
    parser.add_argument("--sms", type=int)
    parser.add_argument("--memory-bandwidth-gbps", type=float)
    parser.add_argument("--conv", choices=["gt", "gat_v2", "min_aggr"], default="gat_v2")
    parser.add_argument("--head-dim", type=int, default=128)
    # Existing measurement script times forward+backward under the 'backward' label.
    # Do not present that calibration as the cost of a backward-only launch.
    parser.add_argument("--buckets", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument(
        "--strategies", choices=["quantile", "log", "tail"], nargs="+", default=["quantile", "log", "tail"]
    )
    parser.add_argument(
        "--launch-modes", choices=["sequential", "concurrent"], nargs="+", default=["sequential", "concurrent"]
    )
    parser.add_argument("--orderings", choices=["natural", "descending"], nargs="+", default=["natural"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--quantile", type=float, default=0.99)
    parser.add_argument("--max-blocks-light", type=int, default=8)
    parser.add_argument("--max-blocks-heavy", type=int, default=4)
    parser.add_argument("--occupancy", type=float, default=1.0)
    parser.add_argument("--memory-latency-ns", type=float, default=400.0)
    parser.add_argument(
        "--extra-launch-us",
        type=float,
        nargs="+",
        default=[0, 2, 5],
        help="Sensitivity: serialized surcharge per nonempty bucket beyond the two-bucket baseline",
    )
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if min(args.buckets) < 2 or min(args.extra_launch_us) < 0:
        parser.error("bucket counts must be >=2 and launch costs >=0")
    if "tail" in args.strategies and any(count % 2 for count in args.buckets):
        parser.error("tail strategy requires even bucket counts")
    if "paper" in args.graphs:
        args.graphs = list(
            dict.fromkeys(name for item in args.graphs for name in (PAPER_GRAPHS if item == "paper" else [item]))
        )
    return args


def validate_model_metadata(payload, args):
    meta = payload.get("measurement_metadata", {})
    gpu = str(meta.get("gpu", ""))
    if args.hardware.upper() not in gpu.upper():
        raise ValueError(
            f"cost model GPU {gpu!r} does not identify {args.hardware}; recalibrate with the current scripts"
        )
    if meta.get("sms") is not None and meta["sms"] != args.sms:
        raise ValueError(f"cost model has {meta['sms']} SMs, simulation requests {args.sms}")
    if meta.get("quantile") is not None and not np.isclose(meta["quantile"], args.quantile):
        raise ValueError("calibration and simulation must use the same heavy quantile")
    if "dtype" not in meta or "heads" not in meta:
        raise ValueError(
            "model lacks dtype/heads provenance; rerun measure_kernel_times.py and calibrate_cost_model.py"
        )
    return meta


def plot_rows(rows, out):
    os.environ.setdefault("MPLCONFIGDIR", str(out / ".matplotlib"))
    import matplotlib.pyplot as plt

    for graph in dict.fromkeys(row["graph"] for row in rows):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        for ax, mode in zip(axes, ("sequential", "concurrent")):
            for strategy in dict.fromkeys(row["strategy"] for row in rows):
                for overhead in sorted({row["extra_launch_us"] for row in rows}):
                    selected = [
                        row
                        for row in rows
                        if row["graph"] == graph
                        and row["launch_mode"] == mode
                        and row["strategy"] == strategy
                        and row["ordering"] == "natural"
                        and row["extra_launch_us"] == overhead
                    ]
                    if not selected:
                        continue
                    counts = sorted({row["requested_buckets"] for row in selected})
                    means = [
                        np.mean([row["speedup_vs_2_natural"] for row in selected if row["requested_buckets"] == count])
                        for count in counts
                    ]
                    ax.plot(counts, means, marker="o", label=f"{strategy}, +{overhead:g} us/launch")
            ax.axhline(1, color="gray", linestyle="--")
            ax.set(title=mode, xlabel="Requested buckets", ylabel="Predicted speedup vs 2 natural")
            ax.legend(fontsize=7)
        fig.suptitle(f"{graph}: Bucket simulation, degree ordering OFF")
        fig.savefig(out / f"{graph}-speedup.png", dpi=160)
        plt.close(fig)


def main():
    args = parse_args()
    default_sms, default_bandwidth = HARDWARE[args.hardware]
    args.sms = args.sms or default_sms
    args.memory_bandwidth_gbps = args.memory_bandwidth_gbps or default_bandwidth
    payload = json.loads(args.cost_model.read_text())
    meta = validate_model_metadata(payload, args)
    fit = lookup(load_cost_models(args.cost_model), args.conv, "forward", args.head_dim)
    if not fit.trustworthy:
        raise ValueError(f"untrustworthy cost-model cell: {fit.note}")
    heads = int(meta["heads"])
    dtype_bytes = {"float32": 4, "float16": 2, "bfloat16": 2}[meta["dtype"]]
    cap = bandwidth_cap_from_hardware(args.memory_bandwidth_gbps, args.head_dim, dtype_bytes, args.memory_latency_ns)
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "measurement_metadata": meta,
        "model_sha256": hashlib.sha256(args.cost_model.read_bytes()).hexdigest(),
        "paper_graphs": PAPER_GRAPHS,
        "scope": "CPU scheduling simulation; fp32 forward kernel calibration; no measured multi-bucket CUDA speedup",
        "assumptions": [
            "one block per node/head; fixed light/heavy residency; no slicing",
            "highest-degree bucket launched first; natural order inside each bucket unless labelled descending",
            "concurrent launches available at tick zero; extra launch cost is an explicit sensitivity surcharge",
            "preprocessing, cache locality and per-bucket kernel tuning are not modelled",
        ],
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    rows = []
    for graph in args.graphs:
        degrees = (
            make_powerlaw_degrees(int(graph.removeprefix("synth-N")), 8, 2.3, 42)
            if graph.startswith("synth-N")
            else load_real_degrees(graph, args.data_root, args.quantile, "forward")[0]
        )
        print(f"{graph}: N={len(degrees):,}, E={degrees.sum():,}, max degree={degrees.max()}", flush=True)
        # One timebase for every variant and every scheduler seed, anchored to the
        # two-bucket sequential natural baseline, exactly as in the existing pipeline.
        base_buckets = partition_degree_buckets(degrees, 2, heavy_quantile=args.quantile)
        base_workloads, base_kernels = build_bucket_workloads(
            degrees, base_buckets, fit.cost_model, heads, args.max_blocks_light, args.max_blocks_heavy, args.occupancy
        )
        baseline = simulate(
            base_workloads,
            base_kernels,
            SimulationConfig(
                num_sms=args.sms, bandwidth_cap=cap, seed=args.seeds[0], launch_mode="sequential", record_history=False
            ),
        )
        work_ns = fit.ns_per_node * len(degrees) + fit.ns_per_edge * int(degrees.sum())
        tick = work_ns / baseline.makespan
        for seed in args.seeds:
            # Always calculate the two paired controls, even when only larger bucket counts are requested.
            cache = {}
            variants = list(
                dict.fromkeys(
                    [(2, "quantile", "natural"), (2, "quantile", "descending")]
                    + [
                        (count, strategy if count > 2 else "quantile", ordering)
                        for count in args.buckets
                        for strategy in args.strategies
                        for ordering in args.orderings
                    ]
                )
            )
            for count, strategy, ordering in variants:
                buckets = partition_degree_buckets(degrees, count, strategy, args.quantile, ordering == "descending")
                workloads, kernels = build_bucket_workloads(
                    degrees,
                    buckets,
                    fit.cost_model,
                    heads,
                    args.max_blocks_light,
                    args.max_blocks_heavy,
                    args.occupancy,
                )
                details = [
                    {
                        "nodes": len(b.node_ids),
                        "edges": int(degrees[b.node_ids].sum()),
                        "min_degree": int(degrees[b.node_ids].min()),
                        "max_degree": int(degrees[b.node_ids].max()),
                        "heavy": b.heavy,
                    }
                    for b in buckets
                ]
                for mode in args.launch_modes:
                    result = simulate(
                        workloads,
                        kernels,
                        SimulationConfig(
                            num_sms=args.sms,
                            bandwidth_cap=cap,
                            seed=seed,
                            launch_mode=mode,
                            light_launch_latency=0,
                            record_history=False,
                            ns_per_tick=tick,
                        ),
                    )
                    cache[count, strategy, ordering, mode] = result
                    # launch_overhead_ns is the WHOLE fitted baseline intercept, not a per-launch measurement.
                    modeled_ms = result.makespan * tick / 1e6 + fit.launch_overhead_ns / 1e6
                    for overhead in args.extra_launch_us:
                        predicted_ms = modeled_ms + max(0, len(buckets) - len(base_buckets)) * overhead / 1000
                        natural = cache[2, "quantile", "natural", mode]
                        ordered = cache.get((2, "quantile", "descending", mode))
                        row = result.summary() | {
                            "graph": graph,
                            "num_nodes": len(degrees),
                            "num_edges": int(degrees.sum()),
                            "hardware": args.hardware,
                            "conv": args.conv,
                            "head_dim": args.head_dim,
                            "dtype": meta["dtype"],
                            "heads": heads,
                            "requested_buckets": count,
                            "effective_buckets": len(buckets),
                            "strategy": strategy,
                            "ordering": ordering,
                            "extra_launch_us": overhead,
                            "predicted_ms": predicted_ms,
                            "predicted_work_ms": result.predicted_ms,
                            "speedup_vs_2_natural": (natural.makespan * tick / 1e6 + fit.launch_overhead_ns / 1e6)
                            / predicted_ms,
                            "speedup_vs_2_ordered": (
                                (ordered.makespan * tick / 1e6 + fit.launch_overhead_ns / 1e6) / predicted_ms
                                if ordered
                                else None
                            ),
                            "tick_ns": tick,
                            "buckets": details,
                        }
                        rows.append(row)
                    print(
                        f"  seed={seed} {strategy:8} B={count}/{len(buckets)} {ordering:10} {mode:10} "
                        f"T={result.makespan:6} model={modeled_ms:.4f} ms",
                        flush=True,
                    )
            # Fill ordered references for the natural control that ran first.
            for row in rows:
                if row["graph"] == graph and row["seed"] == seed:
                    ordered = cache[2, "quantile", "descending", row["launch_mode"]]
                    row_ms = row["predicted_ms"]
                    assert isinstance(row_ms, float)
                    row["speedup_vs_2_ordered"] = (
                        ordered.makespan * tick / 1e6 + fit.launch_overhead_ns / 1e6
                    ) / row_ms
            # Persist complete seeds; a later graph failure leaves earlier results reviewable.
            with (args.out / "results.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(
                    {key: json.dumps(value) if isinstance(value, (dict, list)) else value for key, value in row.items()}
                    for row in rows
                )
    if args.plot:
        plot_rows(rows, args.out)
    print(f"Wrote {len(rows)} rows to {args.out / 'results.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
