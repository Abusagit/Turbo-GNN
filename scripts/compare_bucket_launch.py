"""Compare light/heavy bucket launch strategies, each at its own best configuration.

`run_kernel_benchmark_matrix.py --bucket-launch ...` writes one JSON per
(graph, conv, head dim, pass, schedule, bucket_launch), each carrying a `sweep` entry per node
order. This reassembles that into, for every cell, the best time *each strategy can reach* when
it is allowed to pick its own schedule and node order -- which is the only fair comparison:
a strategy that happens to suit a different schedule should not be judged on the schedule that
suits another.

    python scripts/compare_bucket_launch.py reports/kernel-benchmarks-streams
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

MODES = ["sequential", "heavy_first", "concurrent"]
REFERENCE = "sequential"


def load(dir_path: Path):
    """{(graph,conv,dim,pass): {mode: {(schedule,order): ms}}} plus per-graph facts."""
    cells: dict[tuple, dict[str, dict[tuple, float]]] = defaultdict(lambda: defaultdict(dict))
    facts: dict[str, dict] = {}
    for f in sorted(dir_path.glob("*.json")):
        if f.name in {"grid.json"}:
            continue
        d = json.loads(f.read_text())
        graph = f.name.split("__")[0]
        mode = d["kernel_params"]["bucket_launch"]
        key = (graph, d["conv"], d["feature_dim"], d["mode"])
        sched = d["kernel_params"]["schedule"]
        for pt in d.get("sweep") or [
            {"graph_config": {"node_order": d["node_order"]}, "ms_per_iter": d["ms_per_iter"]}
        ]:
            cells[key][mode][(sched, pt["graph_config"].get("node_order", "natural"))] = pt["ms_per_iter"]
        facts.setdefault(graph, d["graph"])
    return cells, facts


def gm(xs) -> float:
    """Geometric mean, skipping cells a strategy has no measurement for."""
    return statistics.geometric_mean([x for x in xs if x])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dir", nargs="?", default="reports/kernel-benchmarks-streams")
    p.add_argument("--json-out", default=None)
    args = p.parse_args()

    cells, facts = load(Path(args.dir))
    rows = []
    for key, per_mode in sorted(cells.items()):
        if REFERENCE not in per_mode:
            continue
        best = {m: min(g.values()) for m, g in per_mode.items() if g}
        cfg = {m: min(g.items(), key=lambda kv: kv[1])[0] for m, g in per_mode.items() if g}
        ref = best[REFERENCE]
        winner = min(best, key=lambda m: best[m])
        rows.append(
            {
                "graph": key[0],
                "conv": key[1],
                "head_dim": key[2],
                "mode": key[3],
                **{f"{m}_ms": best.get(m) for m in MODES},
                **{f"{m}_cfg": "+".join(cfg[m]) for m in MODES if m in cfg},
                **{f"{m}_x": (ref / best[m]) if m in best else None for m in MODES},
                "winner": winner,
                "win_margin": ref / best[winner],
            }
        )
    if not rows:
        raise SystemExit(f"no complete cells in {args.dir}")

    print(f"{len(rows)} cells from {len(list(Path(args.dir).glob('*.json')))} runs in {args.dir}")
    print(
        f"Each strategy is given its own best (schedule x node order); speedups are vs '{REFERENCE}' at *its* best.\n"
    )

    print(f"  {'strategy':<14}{'geomean':>9}{'worst':>8}{'best':>8}{'cells won':>11}{'faster than ref':>17}")
    for m in MODES:
        xs = [r[f"{m}_x"] for r in rows if r.get(f"{m}_x")]
        won = sum(r["winner"] == m for r in rows)
        faster = sum(1 for r in rows if r.get(f"{m}_x") and r[f"{m}_x"] > 1.005)
        print(f"  {m:<14}{gm(xs):>9.4f}{min(xs):>8.3f}{max(xs):>8.3f}{won:>8}/{len(rows):<3}{faster:>13}/{len(rows)}")

    print("\n  by pass and head dim (geomean vs sequential)")
    print(f"    {'dim':>4} {'pass':<9}" + "".join(f"{m:>14}" for m in MODES[1:]))
    for dim in sorted({r["head_dim"] for r in rows}):
        for mode in ("forward", "backward"):
            sub = [r for r in rows if r["head_dim"] == dim and r["mode"] == mode]
            if sub:
                print(
                    f"    {dim:>4} {mode:<9}" + "".join(f"{gm([r[f'{m}_x'] for r in sub]):>14.4f}" for m in MODES[1:])
                )

    print("\n  per graph (geomean vs sequential)")
    by = defaultdict(list)
    for r in rows:
        by[r["graph"]].append(r)
    print(f"    {'graph':<15}{'avg deg':>9}" + "".join(f"{m:>14}" for m in MODES[1:]) + "   best cell for concurrent")

    def graph_key(g):
        xs = [r["concurrent_x"] for r in by[g] if r["concurrent_x"]]
        return -gm(xs) if xs else 0.0

    for g in sorted(by, key=graph_key):
        rs = by[g]
        b = max(rs, key=lambda r: r["concurrent_x"] or 0)
        if not b["concurrent_x"]:
            continue
        print(
            f"    {g:<15}{facts[g]['avg_degree']:>9.1f}"
            + "".join(f"{gm([r[f'{m}_x'] for r in rs]):>14.4f}" for m in MODES[1:])
            + f"   {b['concurrent_x']:.2f}x {b['conv']}/{b['head_dim']}/{b['mode'][:3]}"
        )

    # mid-run, a cell may have its reference but not yet its concurrent measurement
    top = sorted([r for r in rows if r["concurrent_x"]], key=lambda r: -r["concurrent_x"])
    print("\n  largest concurrent wins")
    for r in top[:8]:
        print(
            f"    {r['graph']:<15}{r['conv']:<9}{r['head_dim']:>4} {r['mode']:<9}"
            f"{r['sequential_ms']:>10.4f} -> {r['concurrent_ms']:>9.4f} ms  "
            f"x{r['concurrent_x']:.3f}  [{r['concurrent_cfg']}]"
        )
    print("  largest concurrent losses")
    for r in top[-5:]:
        print(
            f"    {r['graph']:<15}{r['conv']:<9}{r['head_dim']:>4} {r['mode']:<9}"
            f"{r['sequential_ms']:>10.4f} -> {r['concurrent_ms']:>9.4f} ms  "
            f"x{r['concurrent_x']:.3f}  [{r['concurrent_cfg']}]"
        )

    if args.json_out:
        Path(args.json_out).write_text(json.dumps({"cells": rows, "graphs": facts}, indent=1))
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
