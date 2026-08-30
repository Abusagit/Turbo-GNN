"""Two-stage ablation: what the scheduler adds, and what concurrent streams add on top.

Reads `run_scheduler_ablation.py` output, where every kernel parameter is pinned to the value
autotuning already chose for that cell and only the two new knobs vary. Because all three
stages come from one sweep over one loaded graph, they cannot drift relative to each other the
way separately-run arms can.

    A  baseline    one_per_block, natural order, sequential buckets
    B  scheduling  best (schedule x node order), still sequential
    C  + streams   the same grid with concurrent buckets allowed

    python scripts/summarize_scheduler_ablation.py reports/scheduler-ablation
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def gm(xs) -> float:
    return statistics.geometric_mean([x for x in xs if x])


def load(d: Path):
    rows: list[dict] = []
    facts: dict[str, dict] = {}
    for f in sorted(d.glob("*.json")):
        if f.name == "grid.json":
            continue
        j = json.loads(f.read_text())
        graph = f.name.split("__")[0]
        pts = {}
        for s in j["sweep"]:
            c = s["graph_config"]
            key = (
                c.get("kernel:schedule"),
                c.get("kernel:forward_bucket_launch") or c.get("kernel:backward_bucket_launch"),
                c.get("node_order", "natural"),
            )
            pts[key] = s["ms_per_iter"]
        a = pts.get(("one_per_block", "sequential", "natural"))
        seq = [v for k, v in pts.items() if k[1] == "sequential"]
        con = [v for k, v in pts.items() if k[1] == "concurrent"]
        if a is None or not seq or not con:
            continue
        b = min(seq)
        c_ = min(b, min(con))
        best = min(pts.items(), key=lambda kv: kv[1])[0]
        rows.append(
            {
                "graph": graph,
                "conv": j["conv"],
                "head_dim": j["feature_dim"],
                "mode": j["mode"],
                "A_ms": a,
                "B_ms": b,
                "C_ms": c_,
                "B_over_A": a / b,
                "C_over_B": b / c_,
                "C_over_A": a / c_,
                "best_schedule": best[0],
                "best_bucket": best[1],
                "best_order": best[2],
                "streams_helped": min(con) < b * 0.995,
            }
        )
        facts.setdefault(graph, j["graph"])
    return rows, facts


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dir", nargs="?", default="reports/scheduler-ablation")
    p.add_argument("--json-out", default=None)
    args = p.parse_args()
    rows, facts = load(Path(args.dir))
    if not rows:
        raise SystemExit(f"no complete cells in {args.dir}")

    print(f"{len(rows)} cells from {args.dir}")
    print(
        "Every other kernel parameter pinned to the autotuned argmax; only schedule, node "
        "order and\nbucket launch vary. A/B/C all measured in one sweep per cell.\n"
    )
    print(f"  {'':<22}{'B/A scheduling':>16}{'C/B streams':>14}{'C/A both':>11}")
    for lbl, sel in (
        ("all cells", lambda r: True),
        ("forward", lambda r: r["mode"] == "forward"),
        ("backward", lambda r: r["mode"] == "backward"),
        ("head dim 128", lambda r: r["head_dim"] == 128),
        ("head dim 256", lambda r: r["head_dim"] == 256),
        ("min_aggr", lambda r: r["conv"] == "min_aggr"),
        ("gat_v2", lambda r: r["conv"] == "gat_v2"),
        ("gt", lambda r: r["conv"] == "gt"),
    ):
        s = [r for r in rows if sel(r)]
        if s:
            print(
                f"  {lbl:<22}{gm([r['B_over_A'] for r in s]):>16.4f}"
                f"{gm([r['C_over_B'] for r in s]):>14.4f}{gm([r['C_over_A'] for r in s]):>11.4f}"
            )

    print(
        f"\n  streams improved on the best sequential configuration in "
        f"{sum(r['streams_helped'] for r in rows)}/{len(rows)} cells"
    )
    for label, key in (("schedule", "best_schedule"), ("bucket launch", "best_bucket"), ("node order", "best_order")):
        print(f"  winning {label:<14}{dict(Counter(r[key] for r in rows).most_common())}")

    print("\n  per graph")
    by = defaultdict(list)
    for r in rows:
        by[r["graph"]].append(r)
    print(f"    {'graph':<15}{'avg deg':>9}{'B/A':>8}{'C/B':>8}{'C/A':>8}   best cell")
    for g in sorted(by, key=lambda g: -gm([r["C_over_A"] for r in by[g]])):
        rs = by[g]
        b = max(rs, key=lambda r: r["C_over_A"])
        print(
            f"    {g:<15}{facts[g]['avg_degree']:>9.1f}{gm([r['B_over_A'] for r in rs]):>8.3f}"
            f"{gm([r['C_over_B'] for r in rs]):>8.3f}{gm([r['C_over_A'] for r in rs]):>8.3f}"
            f"   {b['C_over_A']:.2f}x {b['conv']}/{b['head_dim']}/{b['mode'][:3]}"
        )

    print("\n  largest gains from streams alone (C/B)")
    for r in sorted(rows, key=lambda r: -r["C_over_B"])[:8]:
        print(
            f"    {r['graph']:<15}{r['conv']:<9}{r['head_dim']:>4} {r['mode']:<9}"
            f"{r['B_ms']:>9.4f} -> {r['C_ms']:>9.4f} ms  x{r['C_over_B']:.3f}"
        )

    if args.json_out:
        Path(args.json_out).write_text(json.dumps({"cells": rows, "graphs": facts}, indent=1))
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
