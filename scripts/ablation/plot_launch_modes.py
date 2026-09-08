#!/usr/bin/env python3
"""Per-SM occupancy over time, one panel per launch mode.

The figure the load-imbalance story rests on: 132 SMs down the y axis, time along x, colour
being the fraction of each SM's block slots that are occupied.  Filling, steady state and drain
are all visible at once, and stacking the launch modes on a shared time axis shows directly what
overlapping the light and heavy buckets buys.

Everything but the launch mode is held fixed at the measured baseline layout -- one block per
node in natural order -- so the panels differ in exactly one variable.

    python scripts/ablation/plot_launch_modes.py --dataset synth-N65536 \
        --cost-model cost_models.json --conv gt --pass forward --head-dim 128 \
        --out launch_modes.png

Read the panels with the reported binding bound in mind.  When the critical path binds, the long
thin tail is one node too large to split, and no launch mode can do anything about it.
"""

from __future__ import annotations

import argparse
import os
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

MODE_BLURB = {
    "single": "one kernel over every node, no bucketing",
    "sequential": "heavy bucket drains before light starts",
    "concurrent": "light starts a launch latency after heavy",
}


def bin_columns(utilisation: np.ndarray, max_columns: int) -> np.ndarray:
    """Average time steps into at most ``max_columns`` columns.

    A long run is tens of thousands of ticks wide and would be resampled by the renderer
    anyway, dropping whole stalls between pixels.  Averaging keeps them visible.
    """
    num_steps = utilisation.shape[0]
    if num_steps <= max_columns:
        return utilisation
    edges = np.linspace(0, num_steps, max_columns + 1).astype(int)
    return np.stack([utilisation[start:stop].mean(axis=0) for start, stop in zip(edges[:-1], edges[1:])])


def panel_title(mode: str, result: SimulationResult, launch_overhead_ms: float) -> str:
    predicted = "" if result.predicted_ms is None else f"   {(result.predicted_ms + launch_overhead_ms) * 1e3:.1f} us"
    return (
        f"{mode} -- {MODE_BLURB[mode]}   |   T/T* = {result.imbalance_ratio:.3f} "
        f"({result.binding_bound})   drain tail {result.drain_tail / result.makespan:.0%}{predicted}"
    )


def plot(
    path: Path,
    results: dict[str, SimulationResult],
    num_sms: int,
    tick_ns: float | None,
    overhead_ms: float,
    max_columns: int,
    title: str,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(path.parent / ".matplotlib"))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to generate the launch-mode figure") from exc

    unit, scale = ("us", 1e-3) if tick_ns else ("ticks", 1.0)
    span = max(r.makespan * (tick_ns * scale if tick_ns else 1.0) for r in results.values())

    fig, axes = plt.subplots(
        len(results), 1, figsize=(13, 1.1 + 2.4 * len(results)), sharex=True, constrained_layout=True
    )
    axes = np.atleast_1d(axes)
    image = None
    for axis, (mode, result) in zip(axes, results.items()):
        width = result.makespan * (tick_ns * scale if tick_ns else 1.0)
        image = axis.imshow(
            bin_columns(result.sm_utilisation, max_columns).T,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            vmin=0,
            vmax=1,
            extent=(0.0, width, 0.0, num_sms),
        )
        axis.set(ylabel="SM", title=panel_title(mode, result, overhead_ms))
        axis.set_xlim(0.0, span)
    axes[-1].set_xlabel(f"time, {unit}")
    fig.colorbar(image, ax=axes.tolist(), label="occupied slot fraction", fraction=0.02)
    fig.suptitle(title)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_arguments(parser)
    parser.add_argument(
        "--launch-modes",
        nargs="+",
        choices=["single", "sequential", "concurrent"],
        default=["single", "sequential", "concurrent"],
    )
    parser.add_argument("--max-columns", type=int, default=1600, help="Time steps are averaged down to this width")
    parser.add_argument("--out", type=Path, default=Path("launch_modes.png"))
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
    overhead_ms = fit.launch_overhead_ns / 1e6 if fit else 0.0
    occupancy = args.occupancies[0]

    results: dict[str, SimulationResult] = {}
    for mode in args.launch_modes:
        workloads, kernels, light, heavy = build_workloads(
            args,
            cost_model,
            degrees,
            BASELINE_ASSIGNMENT,
            BASELINE_VERTICES_PER_BLOCK,
            None,
            BASELINE_HEAVY_SLICE,
            mode,
            occupancy,
        )
        results[mode] = simulate(
            workloads,
            kernels,
            SimulationConfig(
                num_sms=args.sms,
                bandwidth_cap=bandwidth_cap,
                seed=args.seed,
                launch_mode=mode,
                light_launch_latency=args.launch_latency,
                ns_per_tick=tick_ns,
            ),
        )
        result = results[mode]
        print(
            f"{mode:<11} T={result.makespan:>7} T*={result.perfect_packing_time:>10.1f} ({result.binding_bound}) "
            f"imbalance={result.imbalance_ratio:.3f} tail={result.drain_tail:>6} "
            f"light={len(light)} heavy={len(heavy)}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    all_degrees = degrees[0]
    # Skew is the reason any of this matters, so it belongs on the figure: max/avg degree says
    # at a glance whether a long drain tail was ever avoidable.
    title = (
        f"{args.dataset}  N={len(all_degrees):,} E={int(all_degrees.sum()):,} "
        f"avg deg {all_degrees.mean():.1f}  max deg {all_degrees.max():,} "
        f"(skew {all_degrees.max() / all_degrees.mean():.0f}x)  heavy {len(degrees[2]):,}\n"
        f"{args.conv}/{args.pass_name}/d{args.head_dim}  "
        f"alpha={provenance['alpha']:.2f} beta={provenance['beta']:.2f}  "
        f"{args.sms} SMs x {args.max_blocks_light} slots  occ={occupancy:g}"
    )
    plot(args.out, results, args.sms, tick_ns, overhead_ms, args.max_columns, title)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
