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

# Colour carries the launch mode and dash carries split-K, so the two factors read
# independently instead of competing for five arbitrary hues. Slots 1-3 of the reference
# palette, in fixed order.
MODE_COLOUR = {"single": "#2a78d6", "sequential": "#eb6834", "concurrent": "#1baf7a"}
INK, MUTED, GRID, RULE = "#0b0b0b", "#52514e", "#d8d7d2", "#8a8983"


def occupancy(result: SimulationResult, kernels: dict, num_sms: int) -> np.ndarray:
    """Fraction of the machine's block slots occupied at each tick.

    A block of kernel k takes ``1 / resident_blocks_per_sm[k]`` of one SM, so this is the same
    quantity the per-SM heatmap coloured -- reconstructed from the per-kernel counts, which are
    one-dimensional, rather than from the [ticks x SMs] matrix.

    Plotting this instead of a raw block count matters for reading the figure: a merge block is
    one warp and 2,112 of them fit, so on a block count the merge phase spikes above the light
    kernel's ceiling and looks like a glitch. As a fraction, full is full.
    """
    total = np.zeros(result.makespan, dtype=np.float64)
    for name, counts in result.active_blocks.items():
        total += counts / (kernels[name].resident_blocks_per_sm * num_sms)
    return total


def utilisation(series: np.ndarray) -> float:
    """Normalised area under the occupancy curve: what fraction of the machine was used.

    A perfect run is a rectangle -- every slot busy for the whole makespan -- and scores 1. The
    area itself is fixed, being the total work, so this equals ``slot_bound / makespan``: the
    reciprocal of the imbalance ratio whenever the slot bound is what binds.

    It is the better of the two to quote. The imbalance ratio measures distance to a floor that
    may itself be one undividable node, and then reports 1.000 for a run that left the machine
    88% idle; this number says 0.12 and means it.
    """
    return float(series.mean()) if len(series) else 0.0


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

    runs: list[tuple[str, str, bool, SimulationResult, dict]] = []
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
        runs.append((label, mode, slice_size > 0, result, kernels))
        print(f"  {label:<22} T={result.makespan:>8,} ({result.binding_bound}) ratio={result.imbalance_ratio:.3f}")

    os.environ.setdefault("MPLCONFIGDIR", str(args.out.parent / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(13, 6.4), constrained_layout=True)
    overhead_us = fit.launch_overhead_ns / 1e3 if fit else 0.0
    span = max(r.makespan for *_, r, _ in runs) * scale
    floor = 0.5 / (args.sms * max(args.max_blocks_light, args.max_blocks_heavy, args.max_blocks_merge))

    axis.axhline(1.0, color=RULE, lw=1.4, zorder=1)
    axis.text(span * 1.15, 1.0, "machine full", color=MUTED, fontsize=9, va="bottom", ha="right")

    ends = sorted(r.makespan for *_, r, _ in runs)
    for label, mode, split, result, kernels in runs:
        series = occupancy(result, kernels, args.sms)
        steps, values = thin(series, args.max_points)
        colour, style = MODE_COLOUR[mode], ((2, 2) if split else ())
        end = result.makespan * scale
        # Carry the line down to the floor at the end. Left to stop mid-air it is ambiguous
        # whether the run finished there or the series simply ran out of data.
        axis.plot(
            np.append(steps * scale, [end, end]),
            np.append(np.maximum(values, floor), [values[-1] if len(values) else floor, floor]),
            lw=2,
            color=colour,
            label=label,
            zorder=3,
            dashes=style,
        )
        rank = ends.index(result.makespan)
        axis.annotate(
            f"{end + overhead_us:.0f} us   util {utilisation(series):.0%}",
            (end, floor),
            textcoords="offset points",
            xytext=(6, 3 + 14 * (rank % 3)),
            color=colour,
            fontsize=9,
            fontweight="bold",
            zorder=5,
            arrowprops={"arrowstyle": "-", "color": colour, "lw": 1, "shrinkA": 0, "shrinkB": 2},
        )

    axis.set_yscale("log")
    axis.set_ylim(floor * 0.75, 1.6)
    axis.set_xlim(0, span * 1.16)
    axis.set_ylabel("fraction of the machine's block slots occupied", color=INK)
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
        f"{args.conv}/{args.pass_name}/d{args.head_dim}  q{args.quantile}  alpha={provenance['alpha']:.2f}\n"
        "util = area under the curve / a full rectangle = the fraction of the machine actually used",
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
