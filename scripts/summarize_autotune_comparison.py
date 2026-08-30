"""Autotuned kernels versus the pre-scheduler baseline.

Reassembles `run_autotune_comparison.py` output into one row per
(graph, conv, head dim, pass), with the configuration the search chose. Baseline is
`schedule=one_per_block` with both bucket launches sequential and autotuning off -- the launch
these kernels used before any of the scheduler, visit-order or stream work existed.

    python scripts/summarize_autotune_comparison.py reports/autotune-comparison
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def gm(xs) -> float:
    return statistics.geometric_mean([x for x in xs if x])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dir", nargs="?", default="reports/autotune-comparison")
    p.add_argument("--json-out", default=None)
    args = p.parse_args()

    cells: dict[tuple, dict] = defaultdict(dict)
    facts: dict[str, dict] = {}
    for f in sorted(Path(args.dir).glob("*.json")):
        if f.name == "grid.json":
            continue
        d = json.loads(f.read_text())
        parts = f.stem.split("__")
        cells[(parts[0], d["conv"], d["feature_dim"], d["mode"])][parts[-1]] = d
        facts.setdefault(parts[0], d["graph"])

    rows = []
    for k, v in sorted(cells.items()):
        if "baseline" not in v or "autotuned" not in v:
            continue
        b, a = v["baseline"], v["autotuned"]
        sel = a.get("autotune_selected") or {}
        kc, gc = sel.get("kernel_config") or {}, sel.get("graph_config") or {}
        rows.append(
            {
                "graph": k[0],
                "conv": k[1],
                "head_dim": k[2],
                "mode": k[3],
                "baseline_ms": b["ms_per_iter"],
                "autotuned_ms": a["ms_per_iter"],
                "speedup": b["ms_per_iter"] / a["ms_per_iter"],
                "bucket_launch": kc.get("forward_bucket_launch") or kc.get("backward_bucket_launch"),
                "quantile": gc.get(
                    "forward_huge_degree_threshold_quantile", gc.get("backward_huge_degree_threshold_quantile")
                ),
                "kernel_config": kc,
                "peak_mb_baseline": b["memory"]["peak_mb"],
                "peak_mb_autotuned": a["memory"]["peak_mb"],
            }
        )
    if not rows:
        raise SystemExit(f"no complete cells in {args.dir}")

    xs = [r["speedup"] for r in rows]
    print(f"{len(rows)} cells from {len(list(Path(args.dir).glob('*.json')))} runs in {args.dir}")
    print("baseline = one_per_block, both bucket launches sequential, no autotuning\n")
    print(
        f"  overall   geomean {gm(xs):.4f}x   {sum(x >= 1 for x in xs)}/{len(xs)} at or above baseline   "
        f"worst {min(xs):.3f}  best {max(xs):.3f}"
    )
    for dim in sorted({r["head_dim"] for r in rows}):
        for mode in ("forward", "backward"):
            sub = [r["speedup"] for r in rows if r["head_dim"] == dim and r["mode"] == mode]
            if sub:
                print(f"  d={dim:<4} {mode:<9} geomean {gm(sub):.4f}x   {sum(x >= 1 for x in sub)}/{len(sub)}")
    for conv in sorted({r["conv"] for r in rows}):
        sub = [r["speedup"] for r in rows if r["conv"] == conv]
        print(f"  {conv:<9} geomean {gm(sub):.4f}x   {sum(x >= 1 for x in sub)}/{len(sub)}")

    print("\n  what the search chose for bucket_launch")
    for mode in ("forward", "backward"):
        c = Counter(r["bucket_launch"] for r in rows if r["mode"] == mode and r["bucket_launch"])
        tot = sum(c.values())
        if tot:
            conc = c.get("concurrent", 0)
            print(f"    {mode:<9} concurrent {conc}/{tot} ({conc / tot * 100:.0f}%)")
    # is concurrency associated with a better outcome?
    for mode in ("forward", "backward"):
        for bl in ("concurrent", "sequential"):
            sub = [r["speedup"] for r in rows if r["mode"] == mode and r["bucket_launch"] == bl]
            if sub:
                print(f"    {mode:<9} chose {bl:<11} -> geomean {gm(sub):.3f}x over {len(sub)} cells")

    print("\n  per graph")
    by = defaultdict(list)
    for r in rows:
        by[r["graph"]].append(r)
    print(f"    {'graph':<15}{'nodes':>11}{'avg deg':>9}{'overall':>9}{'forward':>9}{'backward':>10}   best cell")
    for g in sorted(by, key=lambda g: -gm([r["speedup"] for r in by[g]])):
        rs = by[g]
        fw = [r["speedup"] for r in rs if r["mode"] == "forward"]
        bw = [r["speedup"] for r in rs if r["mode"] == "backward"]
        b = max(rs, key=lambda r: r["speedup"])
        print(
            f"    {g:<15}{facts[g]['num_nodes']:>11,}{facts[g]['avg_degree']:>9.1f}"
            f"{gm([r['speedup'] for r in rs]):>9.3f}{gm(fw):>9.3f}{gm(bw):>10.3f}"
            f"   {b['speedup']:.2f}x {b['conv']}/{b['head_dim']}/{b['mode'][:3]}"
        )

    top = sorted(rows, key=lambda r: -r["speedup"])
    print("\n  largest gains")
    for r in top[:8]:
        print(
            f"    {r['graph']:<15}{r['conv']:<9}{r['head_dim']:>4} {r['mode']:<9}"
            f"{r['baseline_ms']:>10.4f} -> {r['autotuned_ms']:>9.4f} ms  x{r['speedup']:.2f}  bl={r['bucket_launch']}"
        )
    print("  largest losses")
    for r in top[-6:]:
        print(
            f"    {r['graph']:<15}{r['conv']:<9}{r['head_dim']:>4} {r['mode']:<9}"
            f"{r['baseline_ms']:>10.4f} -> {r['autotuned_ms']:>9.4f} ms  x{r['speedup']:.2f}  bl={r['bucket_launch']}"
        )

    if args.json_out:
        Path(args.json_out).write_text(json.dumps({"cells": rows, "graphs": facts}, indent=1))
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
