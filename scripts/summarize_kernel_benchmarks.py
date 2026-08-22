"""Turn a `run_kernel_benchmark_matrix.py` output directory into a report.

Each JSON file covers one (graph, conv, head dim, pass, schedule) and carries a `sweep` entry
per node order, so the full policy x order grid for a cell is reassembled here rather than
re-measured. The baseline every speedup is quoted against is `one_per_block` in natural node
order -- the launch these kernels used before the scheduler existed.

    python scripts/summarize_kernel_benchmarks.py reports/kernel-benchmarks
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

BASELINE = ("one_per_block", "natural")


def load(dir_path: Path) -> tuple[dict, dict]:
    """Read every run into {(graph,conv,dim,mode): {(schedule,order): ms}} plus graph facts."""
    cells: dict[tuple, dict[tuple, float]] = defaultdict(dict)
    facts: dict[str, dict] = {}
    for f in sorted(dir_path.glob("*.json")):
        d = json.loads(f.read_text())
        graph = f.name.split("__")[0]
        key = (graph, d["conv"], d["feature_dim"], d["mode"])
        sched = d["kernel_params"]["schedule"]
        for point in d.get("sweep") or [
            {"graph_config": {"node_order": d["node_order"]}, "ms_per_iter": d["ms_per_iter"]}
        ]:
            order = point["graph_config"].get("node_order", "natural")
            cells[key][(sched, order)] = point["ms_per_iter"]
        facts.setdefault(graph, d["graph"] | {"self_loops": d["self_loops"]})
    return cells, facts


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dir", nargs="?", default="reports/kernel-benchmarks")
    p.add_argument("--json-out", default=None, help="also write the reassembled grid as one JSON")
    args = p.parse_args()

    root = Path(args.dir)
    cells, facts = load(root)
    if not cells:
        raise SystemExit(f"no JSON files in {root}")

    rows = []
    incomplete = []
    for key, grid in sorted(cells.items()):
        base = grid.get(BASELINE)
        if base is None:
            incomplete.append(key)
            continue
        best_cfg, best_ms = min(grid.items(), key=lambda kv: kv[1])
        rows.append(
            {
                "graph": key[0],
                "conv": key[1],
                "head_dim": key[2],
                "mode": key[3],
                "baseline_ms": base,
                "best_ms": best_ms,
                "best_schedule": best_cfg[0],
                "best_order": best_cfg[1],
                "speedup": base / best_ms,
                "configs": len(grid),
            }
        )

    by_graph = defaultdict(list)
    for r in rows:
        by_graph[r["graph"]].append(r)

    print(f"{len(rows)} cells from {len(list(root.glob('*.json')))} runs in {root}")
    print(f"baseline = {BASELINE[0]} in {BASELINE[1]} node order; self-loops added by the loader\n")

    for graph in sorted(by_graph, key=lambda g: -statistics.geometric_mean([r["speedup"] for r in by_graph[g]])):
        rs = by_graph[graph]
        gm = statistics.geometric_mean([r["speedup"] for r in rs])
        f = facts[graph]
        print(f"### {graph}   geomean {gm:.3f}x   {sum(r['speedup'] >= 1 for r in rs)}/{len(rs)} at or above")
        print(
            f"    N={f['num_nodes']:,} E={f['num_edges']:,} avg_deg={f['avg_degree']:.1f} "
            f"max_deg={f['max_degree']:,} heavy_fwd={f['forward_heavy_nodes']:,}"
        )
        print(f"    {'conv':<9}{'dim':>4}{'pass':>10}{'baseline':>11}{'best':>11}{'x':>7}  winning config")
        for r in sorted(rs, key=lambda r: (r["conv"], r["head_dim"], r["mode"])):
            print(
                f"    {r['conv']:<9}{r['head_dim']:>4}{r['mode']:>10}{r['baseline_ms']:>11.4f}"
                f"{r['best_ms']:>11.4f}{r['speedup']:>7.2f}  {r['best_schedule']}+{r['best_order']}"
            )
        print()

    print("=" * 72)
    for dim in sorted({r["head_dim"] for r in rows}):
        for mode in ("forward", "backward"):
            xs = [r["speedup"] for r in rows if r["head_dim"] == dim and r["mode"] == mode]
            if xs:
                print(
                    f"  d={dim:<4} {mode:<9} geomean {statistics.geometric_mean(xs):.3f}x   "
                    f"{sum(x >= 1 for x in xs)}/{len(xs)} at or above baseline"
                )
    allx = [r["speedup"] for r in rows]
    print(
        f"\n  overall   geomean {statistics.geometric_mean(allx):.3f}x   "
        f"{sum(x >= 1 for x in allx)}/{len(allx)} at or above baseline"
    )

    # How each fixed (schedule, order) would do if it had to be the single default.
    per_cfg = defaultdict(list)
    for key, grid in cells.items():
        base = grid.get(BASELINE)
        if base is None:
            continue
        for cfg, ms in grid.items():
            per_cfg[cfg].append(base / ms)
    n = len(rows)
    print(f"\n  if one configuration had to be the default, across all {n} cells")
    print(f"    {'config':<32}{'geomean':>9}{'worst':>8}{'best':>8}{'>=1.0':>9}")
    for cfg, xs in sorted(per_cfg.items(), key=lambda kv: -statistics.geometric_mean(kv[1])):
        if len(xs) < n * 0.9:
            continue
        print(
            f"    {cfg[0] + '+' + cfg[1]:<32}{statistics.geometric_mean(xs):>9.3f}"
            f"{min(xs):>8.2f}{max(xs):>8.2f}{sum(x >= 1 for x in xs):>6}/{len(xs)}"
        )

    if incomplete:
        print(
            f"\n  {len(incomplete)} cell(s) missing their baseline run: "
            + ", ".join("/".join(map(str, k)) for k in incomplete[:6])
        )
    if args.json_out:
        Path(args.json_out).write_text(json.dumps({"cells": rows, "graphs": facts}, indent=1))
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
