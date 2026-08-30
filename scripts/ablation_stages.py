"""Two-stage ablation: how much does the scheduler add, and how much do streams add on top.

Answers a different question from `summarize_autotune_comparison.py`, which compares the
autotuner as a whole against an untuned baseline and so mixes the new work in with parameters
that already existed. Here each stage adds exactly one thing:

    A  baseline          one_per_block, natural node order, sequential buckets
    B  + scheduling      best of (schedule x node order), still sequential buckets
    C  + streams         the same sweep, with concurrent buckets

B is the scheduler and visit-order work at its best; C - B is what concurrent streams add on
top of it. Both stages take the best configuration per cell, so neither is judged on a
configuration that suits the other.

Reads `reports/kernel-benchmarks-streams/`, where all three were measured under one build with
every other kernel parameter held fixed.

    python scripts/ablation_stages.py
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

BASELINE_CFG = ("one_per_block", "natural")


def gm(xs) -> float:
    return statistics.geometric_mean([x for x in xs if x])


def load(dir_path: Path):
    """{(graph,conv,dim,pass): {bucket_launch: {(schedule,order): ms}}} plus graph facts."""
    cells: dict[tuple, dict[str, dict[tuple, float]]] = defaultdict(lambda: defaultdict(dict))
    facts: dict[str, dict] = {}
    for f in sorted(dir_path.glob("*.json")):
        if f.name == "grid.json":
            continue
        d = json.loads(f.read_text())
        graph = f.name.split("__")[0]
        bl = d["kernel_params"].get("bucket_launch") or d["kernel_params"].get("forward_bucket_launch")
        key = (graph, d["conv"], d["feature_dim"], d["mode"])
        sched = d["kernel_params"]["schedule"]
        for pt in d.get("sweep") or [
            {"graph_config": {"node_order": d["node_order"]}, "ms_per_iter": d["ms_per_iter"]}
        ]:
            cells[key][bl][(sched, pt["graph_config"].get("node_order", "natural"))] = pt["ms_per_iter"]
        facts.setdefault(graph, d["graph"])
    return cells, facts


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dir", nargs="?", default="reports/kernel-benchmarks-streams")
    p.add_argument("--json-out", default=None)
    args = p.parse_args()

    cells, facts = load(Path(args.dir))
    rows = []
    for key, per_bl in sorted(cells.items()):
        seq, con = per_bl.get("sequential", {}), per_bl.get("concurrent", {})
        a = seq.get(BASELINE_CFG)
        if a is None or not seq or not con:
            continue
        b = min(seq.values())
        c = min(con.values())
        rows.append(
            {
                "graph": key[0],
                "conv": key[1],
                "head_dim": key[2],
                "mode": key[3],
                "A_baseline_ms": a,
                "B_sched_ms": b,
                "C_streams_ms": min(b, c),
                "B_over_A": a / b,
                "C_over_B": b / min(b, c),
                "C_over_A": a / min(b, c),
                "B_cfg": "+".join(min(seq.items(), key=lambda kv: kv[1])[0]),
                "C_cfg": "+".join(min(con.items(), key=lambda kv: kv[1])[0]),
                "streams_helped": c < b * 0.995,
            }
        )
    if not rows:
        raise SystemExit(f"no complete cells in {args.dir}")

    print(f"{len(rows)} cells from {args.dir}\n")
    print("Stage A = one_per_block + natural order + sequential buckets (the pre-work launch).")
    print("Stage B = best (schedule x node order), sequential buckets.")
    print("Stage C = B, plus concurrent buckets where they help.\n")
    print(f"  {'':<22}{'B/A (scheduling)':>19}{'C/B (streams)':>16}{'C/A (both)':>13}")
    for label, sel in (
        ("all cells", lambda r: True),
        ("forward", lambda r: r["mode"] == "forward"),
        ("backward", lambda r: r["mode"] == "backward"),
        ("head dim 128", lambda r: r["head_dim"] == 128),
        ("head dim 256", lambda r: r["head_dim"] == 256),
    ):
        sub = [r for r in rows if sel(r)]
        print(
            f"  {label:<22}{gm([r['B_over_A'] for r in sub]):>19.4f}"
            f"{gm([r['C_over_B'] for r in sub]):>16.4f}{gm([r['C_over_A'] for r in sub]):>13.4f}"
        )
    helped = sum(r["streams_helped"] for r in rows)
    print(f"\n  streams improved on the best sequential configuration in {helped}/{len(rows)} cells")
    fw = [r for r in rows if r["mode"] == "forward"]
    bw = [r for r in rows if r["mode"] == "backward"]
    print(
        f"    forward  {sum(r['streams_helped'] for r in fw)}/{len(fw)}"
        f"    backward {sum(r['streams_helped'] for r in bw)}/{len(bw)}"
    )

    print("\n  per graph (geomean)")
    by = defaultdict(list)
    for r in rows:
        by[r["graph"]].append(r)
    print(f"    {'graph':<15}{'avg deg':>9}{'B/A':>8}{'C/B':>8}{'C/A':>8}   best cell overall")
    for g in sorted(by, key=lambda g: -gm([r["C_over_A"] for r in by[g]])):
        rs = by[g]
        b = max(rs, key=lambda r: r["C_over_A"])
        print(
            f"    {g:<15}{facts[g]['avg_degree']:>9.1f}{gm([r['B_over_A'] for r in rs]):>8.3f}"
            f"{gm([r['C_over_B'] for r in rs]):>8.3f}{gm([r['C_over_A'] for r in rs]):>8.3f}"
            f"   {b['C_over_A']:.2f}x {b['conv']}/{b['head_dim']}/{b['mode'][:3]}"
        )

    print("\n  where streams add the most on top of scheduling")
    for r in sorted(rows, key=lambda r: -r["C_over_B"])[:8]:
        print(
            f"    {r['graph']:<15}{r['conv']:<9}{r['head_dim']:>4} {r['mode']:<9}"
            f"B {r['B_sched_ms']:>9.4f} -> C {r['C_streams_ms']:>9.4f} ms  x{r['C_over_B']:.3f}  [{r['C_cfg']}]"
        )

    if args.json_out:
        Path(args.json_out).write_text(json.dumps({"cells": rows, "graphs": facts}, indent=1))
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
