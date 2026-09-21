"""Aggregate `benchmark_scheduler_suite.py` JSON into a per-graph verdict table.

The suite emits one row per (graph, conv, head dim, direction, policy). What matters when
choosing a default is narrower: for each (graph, conv, dim, direction), which policy was
fastest and by how much -- and, separately, how each *fixed* policy would do if it had to be
the default for everything, since a default cannot be picked per graph.

    python scripts/summarize_scheduler_bench.py results/*.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict


def geomean(xs: list[float]) -> float:
    return statistics.geometric_mean(xs) if xs else float("nan")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="+")
    p.add_argument("--dim", type=int, nargs="+", default=None, help="restrict to these head dims")
    args = p.parse_args()

    rows: list[dict] = []
    for f in args.files:
        rows += json.load(open(f))
    if args.dim:
        rows = [r for r in rows if r["head_dim"] in args.dim]
    if not rows:
        raise SystemExit("no rows")

    # --- best policy per cell
    cells: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        cells[(r["graph"], r["conv"], r["head_dim"], r["dir"])].append(r)

    print(f"{'graph':<15}{'conv':<9}{'dim':>5}{'dir':>5}{'base ms':>10}{'best ms':>10}{'x':>7}  best policy")
    per_graph: dict[str, list[float]] = defaultdict(list)
    for key in sorted(cells):
        rs = cells[key]
        best = min(rs, key=lambda r: r["ms"])
        g, c, d, dr = key
        per_graph[g].append(best["speedup"])
        flag = " " if best["speedup"] >= 1.0 else "*"
        print(
            f"{g:<15}{c:<9}{d:>5}{dr:>5}{best['baseline_ms']:>10.3f}{best['ms']:>10.3f}"
            f"{best['speedup']:>7.2f}{flag} {best['policy']}"
        )

    print("\nper-graph geomean of the best policy (oracle: policy chosen per cell)")
    for g, xs in sorted(per_graph.items(), key=lambda kv: -geomean(kv[1])):
        print(f"  {g:<16}{geomean(xs):>6.2f}x   over {len(xs)} cells")

    # --- how each fixed policy would do as a single default
    by_policy: dict[str, list[float]] = defaultdict(list)
    for key, rs in cells.items():
        for r in rs:
            by_policy[r["policy"]].append(r["speedup"])
    n_cells = len(cells)
    print(f"\nif one policy had to be the default, across all {n_cells} cells")
    print(f"  {'policy':<26}{'geomean':>9}{'worst':>8}{'best':>8}{'>=1.0':>8}")
    for pol, xs in sorted(by_policy.items(), key=lambda kv: -geomean(kv[1])):
        if len(xs) < n_cells * 0.8:  # policies that failed on some cells are not candidates
            continue
        wins = sum(x >= 1.0 for x in xs)
        print(f"  {pol:<26}{geomean(xs):>9.2f}{min(xs):>8.2f}{max(xs):>8.2f}{wins:>5}/{len(xs)}")

    all_best = [min(rs, key=lambda r: r["ms"])["speedup"] for rs in cells.values()]
    print(
        f"\noverall: oracle geomean {geomean(all_best):.2f}x, "
        f"{sum(x >= 1.0 for x in all_best)}/{len(all_best)} cells at or above baseline"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
