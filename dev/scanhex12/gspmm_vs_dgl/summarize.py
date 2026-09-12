"""One row per graph: shape, and the geomean speedup over DGL per kind.

    python summarize.py --tag float32_bwd [--dim 64] [--markdown]

At sixteen graphs a per-cell chart is 183 bars, so this is the readable view:
what each graph is, and whether turbo_gnn wins on it.  Read the "клеток"
column before comparing two rows -- a graph past run.sh's edge limit measured
copy_u alone, the rest measured the whole 6x3 table, so their geomeans average
different work.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st

KINDS = ("fwd", "bwd", "fb")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag", default="float32_bwd", help="e.g. float32_bwd, float16_bwd_st2")
    p.add_argument("--results", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--markdown", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    files = sorted(glob.glob(os.path.join(args.results, f"results_*_{args.tag}.json")))
    if not files:
        raise SystemExit(f"no results for tag {args.tag} in {args.results}")

    rows = []
    for path in files:
        blob = json.load(open(path))
        meta = blob["meta"]
        per_kind = {k: [] for k in KINDS}
        for c in blob["cells"]:
            if c["d"] == args.dim and c["kind"] in per_kind:
                per_kind[c["kind"]].append(c["dgl_ms"] / c["turbo_ms"])
        if not any(per_kind.values()):
            continue
        mean_deg = meta["E"] / meta["N"]
        rows.append({
            "graph": meta["graph"],
            "N": meta["N"],
            "E": meta["E"],
            "skew": meta["max_degree"] / mean_deg,
            "max_degree": meta["max_degree"],
            "cells": len(per_kind["fwd"]),
            "copy_u_only": len(meta.get("ops", [])) == 1,
            **{k: (st.geometric_mean(v) if v else None) for k, v in per_kind.items()},
        })

    rows.sort(key=lambda r: -(r["bwd"] or 0))

    if args.markdown:
        print(f"| граф | N | E | макс. вх. | перекос | клеток | forward | backward | fwd+bwd |")
        print("|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            note = " ¹" if r["copy_u_only"] else ""
            print(f"| `{r['graph']}`{note} | {r['N']:,} | {r['E']:,} | {r['max_degree']:,} | "
                  f"{r['skew']:.0f}× | {r['cells']} | "
                  + " | ".join(f"{r[k]:.2f}x" if r[k] else "—" for k in KINDS) + " |")
        print("\n¹ только `copy_u`: операнд `[E, d]` при таком числе рёбер не влезает в память.")
    else:
        print(f"{'граф':16s} {'N':>10s} {'E':>12s} {'макс.вх':>9s} {'перекос':>8s} {'кл.':>4s} "
              f"{'fwd':>7s} {'bwd':>7s} {'fb':>7s}")
        for r in rows:
            print(f"{r['graph'] + ('*' if r['copy_u_only'] else ''):16s} {r['N']:10,d} {r['E']:12,d} "
                  f"{r['max_degree']:9,d} {r['skew']:7.0f}× {r['cells']:4d} "
                  + " ".join(f"{r[k]:6.2f}x" if r[k] else "     —" for k in KINDS))
        print("\n* только copy_u (операнд [E, d] не влезает)")

    for kind in KINDS:
        vals = [r[kind] for r in rows if r[kind]]
        if vals:
            print(f"по всем графам, {kind}: geomean {st.geometric_mean(vals):.2f}x, "
                  f"худший {min(vals):.2f}x ({min(rows, key=lambda r: r[kind] or 9)['graph']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
