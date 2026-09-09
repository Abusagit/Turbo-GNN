#!/usr/bin/env python3
"""One launch-mode figure per graph, kernel and quantile.

The whole grid: 14 graphs by every calibrated (conv, pass, head dim) cell by both bucketing
quantiles.  Figures land in ``<out>/<conv>-<pass>-d<dim>/<graph>-q<NN>.png``, so a directory
name says which kernel it is rather than leaving it implied.

    python scripts/ablation/plot_all.py --cost-model cost_models.json --out reports/launch-modes

Hours, and resumable: a figure that already exists is skipped, so the job can be stopped and
restarted.  Graph degrees are loaded once per (graph, pass, quantile) and reused across the
cells that share them -- the largest graphs take longer to read than to simulate.

Cells the calibration flagged are skipped, as in the sweep: their alpha is not meaningful.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from itertools import product
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
sys.path.append(str(Path(__file__).resolve().parent))

from plot_launch_modes import render  # noqa: E402
from simulate_load_imbalance import add_common_arguments, load_degrees  # noqa: E402
from sweep_all import DEFAULT_GRAPHS  # noqa: E402

from turbo_gnn.calibration import load_cost_models  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_arguments(parser)
    parser.add_argument("--graphs", nargs="+", default=DEFAULT_GRAPHS)
    parser.add_argument("--quantiles", type=float, nargs="+", default=[0.99, 0.95])
    parser.add_argument("--cells", nargs="*", help="conv/pass/head_dim keys; default is every calibrated cell")
    parser.add_argument("--include-untrustworthy", action="store_true", help="Also plot cells the fit flagged")
    parser.add_argument(
        "--launch-modes",
        nargs="+",
        choices=["single", "sequential", "concurrent"],
        default=["single", "sequential", "concurrent"],
    )
    parser.add_argument("--slice-blocks-per-sm", type=float, nargs="+", default=[0, 8])
    parser.add_argument("--max-columns", type=int, default=1600)
    parser.add_argument("--out", type=Path, default=Path("reports/launch-modes"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.cost_model:
        print("--cost-model is required", file=sys.stderr)
        return 1
    fits = load_cost_models(args.cost_model)
    cells = args.cells or sorted(fits)
    flagged = [c for c in cells if fits[c].note and not args.include_untrustworthy]
    cells = [c for c in cells if c not in flagged]
    if flagged:
        print(f"skipping {len(flagged)} flagged cell(s): {', '.join(flagged)}\n", file=sys.stderr)

    planned = len(args.graphs) * len(cells) * len(args.quantiles)
    print(f"{len(args.graphs)} graphs x {len(cells)} cells x {len(args.quantiles)} quantiles = {planned} figures\n")

    # Loading pokec-regions or web-fraud costs more than simulating a small graph, so the
    # degrees are cached and the loop runs graph-outermost to keep each one hot.
    done = failed = skipped = 0
    for graph, quantile in product(args.graphs, args.quantiles):
        degrees: dict[str, tuple[np.ndarray, ...]] = {}
        for cell in cells:
            conv, pass_name, head_dim = cell.split("/")
            out = args.out / f"{conv}-{pass_name}-d{head_dim}" / f"{graph}-q{str(quantile)[2:]}.png"
            if out.exists():
                skipped += 1
                continue
            args.dataset, args.quantile, args.pass_name = graph, quantile, pass_name
            args.conv, args.head_dim, args.out_file = conv, int(head_dim), out
            started = time.monotonic()
            try:
                if pass_name not in degrees:
                    degrees[pass_name] = load_degrees(args)
                figure_args = argparse.Namespace(**{**vars(args), "out": out})
                render(figure_args, degrees[pass_name])
                done += 1
            except Exception:  # noqa: BLE001 - one bad figure must not end a multi-hour job
                print(f"{graph}/{cell}/q{quantile} FAILED:\n{traceback.format_exc()}", file=sys.stderr)
                failed += 1
                continue
            print(f"  [{done + skipped + failed}/{planned}] {out}  ({time.monotonic() - started:.0f}s)")

    print(f"\n{done} written, {skipped} already present, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
