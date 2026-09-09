#!/usr/bin/env python3
"""Resident blocks over time, every configuration on one axis.

The per-SM heatmap spends 132 pixel rows on an axis that carries almost no information: a
block goes to a random free SM, so the SMs are interchangeable and any stripe is noise rather
than a straggler's identity.  Collapsing that axis to a single number -- how many blocks are
resident -- frees the vertical for a log scale, and the drain that the heatmap renders as a
uniform dark field becomes a readable line falling from a thousand blocks to one.

One panel then holds every configuration instead of one each, so the comparison is a
side-by-side of lines rather than an eyeball across stacked images.

    python scripts/ablation/plot_occupancy_profile.py --dataset ogbn-arxiv \
        --cost-model cost_models.json --conv gt --pass forward --head-dim 128 \
        --out reports/profiles/ogbn-arxiv-q99.png
"""

from __future__ import annotations

import argparse
import os
import sys
from itertools import product
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
sys.path.append(str(Path(__file__).resolve().parent))

from simulate_load_imbalance import (  # noqa: E402
    BASELINE_ASSIGNMENT,
    BASELINE_VERTICES_PER_BLOCK,
    add_common_arguments,
    anchor_tick,
    build_workloads,
    dependencies,
    load_degrees,
    resolve_cost_model,
    resolve_heavy_slice,
)

from turbo_gnn.simulation import (  # noqa: E402
    SimulationConfig,
    SimulationResult,
    bandwidth_cap_from_hardware,
    simulate,
)

# Categorical slots 1-5 of the reference palette, assigned in fixed order and never cycled.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d8d7d2"


def resident(result: SimulationResult) -> np.ndarray:
    """Blocks resident at each tick, summed over the kernels."""
    return sum(result.active_blocks.values())


def thin(values: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Reduce to at most ``width`` points, keeping the maximum in each bin.

    The drain matters more than the average here, and averaging a bin that straddles the end of
    a phase would round the cliff off.  Maxima keep the shape and never invent occupancy.
    """
    if len(values) <= width:
        return np.arange(len(values)), values
    edges = np.linspace(0, len(values), width + 1).astype(int)
    keep = [(a, values[a:b].max()) for a, b in zip(edges[:-1], edges[1:]) if b > a]
    return np.array([k[0] for k in keep]), np.array([k[1] for k in keep])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_arguments(parser)
    parser.add_argument(
        "--launch-modes",
        nargs="+",
        choices=["single", "sequential", "concurrent"],
        default=["single", "sequential", "concurrent"],
    )
    parser.add_argument("--slice-blocks-per-sm", type=float, nargs="+", default=[0, 8])
    parser.add_argument("--max-points", type=int, default=2000)
    parser.add_argument("--out", type=Path, default=Path("occupancy_profile.png"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.occupancies = [max(args.occupancies)]
    cost_model, fit, provenance = resolve_cost_model(args)
    degrees = load_degrees(args)
    bandwidth_cap = bandwidth_cap_from_hardware(
        args.memory_bandwidth_gbps, args.feature_dim, args.dtype_bytes, args.memory_latency_ns
    )
    tick_ns, _ = anchor_tick(args, cost_model, fit, degrees, bandwidth_cap)
    scale = (tick_ns or 1.0) / 1e3  # ticks -> microseconds

    runs: list[tuple[str, SimulationResult]] = []
    for mode, blocks_per_sm in product(args.launch_modes, args.slice_blocks_per_sm):
        if mode == "single" and blocks_per_sm > 0:
            continue
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
            args.occupancies[0],
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
        label = mode if slice_size <= 0 else f"{mode} + split-K"
        runs.append((label, result))
        print(f"  {label:<22} T={result.makespan:>8,} ({result.binding_bound}) ratio={result.imbalance_ratio:.3f}")

    os.environ.setdefault("MPLCONFIGDIR", str(args.out.parent / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(13, 6), constrained_layout=True)
    # A full machine is not one number: a block's footprint depends on its kernel, so the light
    # kernel fills at 132x8 while merge, one warp per block, fits four times as many.
    capacities = {"light": args.max_blocks_light, "heavy": args.max_blocks_heavy, "merge": args.max_blocks_merge}
    overhead_us = fit.launch_overhead_ns / 1e3 if fit else 0.0
    span = max(r.makespan for _, r in runs) * scale
    for name, per_sm in capacities.items():
        full = args.sms * per_sm
        axis.axhline(full, color=GRID, lw=1.2, zorder=1)
        # Right-aligned: the left of the plot is where every curve is steepest.
        axis.text(span * 1.16, full, f"{full:,} = full of {name}", color=MUTED, fontsize=8, va="center", ha="right")
    ends = sorted((r.makespan * scale, label, c) for (label, r), c in zip(runs, SERIES))
    for (label, result), colour in zip(runs, SERIES):
        steps, values = thin(resident(result), args.max_points)
        axis.plot(steps * scale, np.maximum(values, 0.5), lw=2, color=colour, label=label, zorder=3)
        # Direct-label each line where it ends. Runs that finish together get stacked offsets so
        # the makespans -- the numbers being compared -- do not overprint each other.
        end = result.makespan * scale
        rank = [e[1] for e in ends].index(label)
        axis.annotate(
            f"{end + overhead_us:.0f} us  {label}",
            (end, 0.5),
            textcoords="offset points",
            xytext=(6, 4 + 13 * (rank % 3)),
            color=colour,
            fontsize=9,
            fontweight="bold",
            zorder=5,
            arrowprops={"arrowstyle": "-", "color": colour, "lw": 1, "shrinkA": 0, "shrinkB": 2},
        )

    axis.set_yscale("log")
    axis.set_ylim(0.4, args.sms * max(capacities.values()) * 2.4)
    axis.set_xlim(0, span * 1.17)
    axis.set_ylabel("resident thread blocks", color=INK)
    axis.set_xlabel(f"time, {'us' if tick_ns else 'ticks'}", color=INK)
    axis.grid(True, which="major", color=GRID, lw=0.6, zorder=0)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    axis.legend(frameon=False, ncol=len(runs), loc="upper center", fontsize=9)

    all_degrees = degrees[0]
    axis.set_title(
        f"{args.dataset}  N={len(all_degrees):,} E={int(all_degrees.sum()):,} "
        f"max deg {all_degrees.max():,} (skew {all_degrees.max() / all_degrees.mean():.0f}x)   "
        f"{args.conv}/{args.pass_name}/d{args.head_dim}  q{args.quantile}  alpha={provenance['alpha']:.2f}",
        color=INK,
        fontsize=11,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=160, facecolor="#fcfcfb")
    plt.close(figure)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
