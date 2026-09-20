#!/usr/bin/env python3
"""Two per-SM occupancy heatmaps, stacked: the bucketed baseline and concurrent + split-K.

The detail goes in the paper's caption, so each panel carries only a short "(a) ..." tag in
the style of the schematic figure; the axes say what they measure and nothing more.  Time is
in abstract units -- the simulator's ticks -- because the wall clock depends on the card the
cost model was fitted on.  Panel geometry matches the five-panel figure, so the two can sit
in the same column width without rescaling.

    python scripts/ablation/plot_heatmap_pair.py --dataset ogbn-arxiv \
        --cost-model cost_models.json --conv gt --pass forward --head-dim 128 \
        --sms 108 --memory-bandwidth-gbps 2039 --out pair.png
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
sys.path.append(str(Path(__file__).resolve().parent))

from plot_launch_modes import bin_columns  # noqa: E402
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

# Panel 1 is the bucketed baseline: heavy and light kernels, one after the other, no slicing.
# Panel 2 overlaps them and cuts the heavy bucket into edge slices.
PANELS = [("sequential", 0.0), ("concurrent", 8.0)]
DEFAULT_LABELS = ["(a)", "(b)"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_arguments(parser)
    parser.add_argument("--max-columns", type=int, default=1600, help="Time steps are averaged down to this width")
    parser.add_argument(
        "--independent-x",
        action="store_true",
        help="Give each panel its own time range instead of a shared one; use when the makespans "
        "differ by more than about 10x and the shorter panel would otherwise be a sliver",
    )
    parser.add_argument("--labels", nargs=2, default=DEFAULT_LABELS, help="Tag to the left of each panel")
    parser.add_argument("--xlabel", default="Simulated cycles")
    parser.add_argument("--ylabel", default="SM index")
    parser.add_argument("--colorbar-label", default="Utilization metric")
    parser.add_argument("--colorbar-side", choices=["right", "bottom"], default="right")
    parser.add_argument(
        "--cmap",
        nargs="+",
        default=["viridis"],
        help="One or more matplotlib colormaps; add _r to reverse one. Several are rendered from a "
        "single simulation, with the name folded into each output file",
    )
    parser.add_argument(
        "--hatch",
        default="///",
        help="Hatch drawn past a panel's makespan, so a run that finishes early reads as finished "
        "rather than as an idle machine; empty string turns it off",
    )
    parser.add_argument(
        "--hatch-linewidth",
        type=float,
        default=1.0,
        help="Thickness of the hatch strokes in points; matplotlib's default is 1.0",
    )
    parser.add_argument(
        "--end-line-width",
        type=float,
        default=1.8,
        help="Rule drawn where the kernel finishes, in points; 0 turns it off",
    )
    parser.add_argument("--end-line-color", default="black")
    parser.add_argument("--label-size", type=float, default=16.0)
    parser.add_argument("--width", type=float, default=10.0, help="Figure width in inches")
    parser.add_argument("--panel-height", type=float, default=1.7, help="Height of one heatmap in inches")
    parser.add_argument("--no-colorbar", dest="colorbar", action="store_false")
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--out", type=Path, default=Path("heatmap_pair.png"))
    return parser.parse_args()


def run_panel(args: argparse.Namespace, cost_model, degrees, bandwidth_cap, tick_ns, mode, blocks_per_sm):
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
    return simulate(
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
        ),
    )


def plot(path: Path, results: list[SimulationResult], num_sms: int, args: argparse.Namespace, cmap: str) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(path.parent / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Hatch stroke width is a global, not a patch property, so it has to be set before drawing.
    matplotlib.rcParams["hatch.linewidth"] = args.hatch_linewidth

    figure, axes = plt.subplots(
        len(results),
        1,
        # A colour bar on the right costs width, not height, so the strip the x label and the
        # bar would need at the bottom is only reserved when the bar actually goes there.
        figsize=(args.width, (0.9 if args.colorbar_side == "bottom" else 0.6) + args.panel_height * len(results)),
        sharex=not args.independent_x,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    span = max(r.makespan for r in results)

    # Reserve a strip at the left for the panel tags: they are placed after the layout is
    # resolved, so the engine cannot make room for them on its own.
    tag_strip = (2.4 * args.label_size / 72) / args.width
    figure.get_layout_engine().set(rect=(tag_strip, 0, 1 - tag_strip, 1))

    # Past a panel's makespan nothing is running, which is zero occupancy -- paint it the
    # colour zero has inside the panel rather than leaving bare white that reads as data.
    idle = matplotlib.colormaps[cmap](0.0)

    image = None
    for index, (axis, result) in enumerate(zip(axes, results), start=1):
        axis.set_facecolor(idle)
        image = axis.imshow(
            bin_columns(result.sm_utilisation, args.max_columns).T,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            cmap=cmap,
            vmin=0,
            vmax=1,
            extent=(0.0, float(result.makespan), 0.0, num_sms),
        )
        axis.set_xlim(0.0, float(result.makespan if args.independent_x else span))
        # Past its own makespan a panel has no machine to show, only the shared axis running
        # on. Black on white, the same in every colormap, so it never reads as a data value.
        if args.hatch and not args.independent_x and result.makespan < span:
            axis.add_patch(
                matplotlib.patches.Rectangle(
                    (result.makespan, 0),
                    span - result.makespan,
                    num_sms,
                    facecolor="white",
                    edgecolor="black",
                    hatch=args.hatch,
                    linewidth=0.0,
                )
            )
        # The moment the kernel finishes is the one number the figure is making a claim about,
        # so it gets a rule of its own rather than being left to the colour change.
        if args.end_line_width > 0:
            axis.axvline(result.makespan, color=args.end_line_color, linewidth=args.end_line_width, zorder=5)
        axis.set_ylabel(args.ylabel, fontsize=args.label_size)
        axis.tick_params(labelsize=args.label_size - 3)
    axes[-1].set_xlabel(args.xlabel, fontsize=args.label_size)

    if args.colorbar:
        fraction = 0.05 if args.colorbar_side == "bottom" else 0.015
        bar = figure.colorbar(image, ax=axes.tolist(), location=args.colorbar_side, fraction=fraction, pad=0.015)
        bar.set_label(args.colorbar_label, fontsize=args.label_size)
        bar.ax.tick_params(labelsize=args.label_size - 3)

    # Place the tags once the layout is settled: each sits a hair in from the left edge,
    # vertically centred on its own panel, so the spacing does not drift when the font size
    # or the y label changes.
    figure.canvas.draw()
    for axis, label in zip(axes, args.labels):
        box = axis.get_position()
        figure.text(
            0.12 * tag_strip,
            (box.y0 + box.y1) / 2,
            label,
            fontsize=args.label_size,
            fontweight="bold",
            va="center",
            ha="left",
        )
    figure.savefig(path, dpi=args.dpi)
    plt.close(figure)


def main() -> int:
    args = parse_args()
    args.occupancies = [max(args.occupancies)]
    cost_model, fit, _ = resolve_cost_model(args)
    degrees = load_degrees(args)
    bandwidth_cap = bandwidth_cap_from_hardware(
        args.memory_bandwidth_gbps, args.feature_dim, args.dtype_bytes, args.memory_latency_ns
    )
    tick_ns, _ = anchor_tick(args, cost_model, fit, degrees, bandwidth_cap)

    results = []
    for index, (mode, blocks_per_sm) in enumerate(PANELS, start=1):
        result = run_panel(args, cost_model, degrees, bandwidth_cap, tick_ns, mode, blocks_per_sm)
        results.append(result)
        print(
            f"panel {index}: {mode}{' + split-K' if blocks_per_sm else '':<10} "
            f"T={result.makespan:>9,} T*={result.perfect_packing_time:>10,.1f} "
            f"T/T*={result.imbalance_ratio:.3f} ({result.binding_bound}) "
            f"occupancy={result.mean_slot_occupancy:.3f} "
            f"drain tail={result.drain_tail / result.makespan:.0%}"
        )
    print(f"compression {results[0].makespan / results[1].makespan:.1f}x")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    for cmap in args.cmap:
        # One simulation, many palettes: the name only enters the filename when there is more
        # than one, so a single-colormap run still writes exactly what --out asked for.
        path = args.out if len(args.cmap) == 1 else args.out.with_name(f"{args.out.stem}-{cmap}{args.out.suffix}")
        plot(path, results, args.sms, args, cmap)
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
