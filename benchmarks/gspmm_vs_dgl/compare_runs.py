"""Per-graph diff of two result sets: what changed between two code states.

    python compare_runs.py --old results_prev_0909am --new results --tag float32_bwd

Compares skewgnn_ms directly rather than the dgl/skewgnn ratio.  The ratio moves
when either side moves, and DGL did not change between the runs -- so skewgnn
against skewgnn isolates the code change, and the DGL column is the control: if
it drifted much, the two runs were not measured under the same conditions and
the skewgnn column cannot carry the weight either.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st

KINDS = ("fwd", "bwd")


def parse_args() -> argparse.Namespace:
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--old", default=os.path.join(here, "results_prev_0909am"))
    p.add_argument("--new", default=os.path.join(here, "results"))
    p.add_argument("--tag", default="float32_bwd")
    p.add_argument("--dim", type=int, default=64)
    return p.parse_args()


def cells(directory: str, tag: str, dim: int) -> dict[tuple[str, str, str, str], dict]:
    out = {}
    for path in glob.glob(os.path.join(directory, f"results_*_{tag}.json")):
        blob = json.load(open(path))
        graph = blob["meta"]["graph"]
        for c in blob["cells"]:
            if c["d"] == dim:
                out[(graph, c["kind"], c["op"], c["reduce"])] = c
    return out


def main() -> int:
    args = parse_args()
    old, new = cells(args.old, args.tag, args.dim), cells(args.new, args.tag, args.dim)
    shared = sorted(set(old) & set(new))
    if not shared:
        raise SystemExit(f"no cells in common for tag {args.tag}")

    per_graph: dict[str, dict[str, list[tuple[float, float]]]] = {}
    for key in shared:
        graph, kind = key[0], key[1]
        if kind not in KINDS:
            continue
        per_graph.setdefault(graph, {}).setdefault(kind, []).append(
            (old[key]["skewgnn_ms"] / new[key]["skewgnn_ms"], old[key]["dgl_ms"] / new[key]["dgl_ms"])
        )

    print(f"tag={args.tag}, d={args.dim}: skewgnn old/new (>1: the new code is faster), "
          f"DGL as the control\n")
    print(f"{'graph':16s} " + " ".join(f"{k + ':skewgnn':>12s} {k + ':dgl':>10s}" for k in KINDS))
    rows = []
    for graph in sorted(per_graph):
        line = f"{graph:16s} "
        summary = {}
        for kind in KINDS:
            pairs = per_graph[graph].get(kind, [])
            if not pairs:
                line += f"{'—':>12s} {'—':>10s} "
                continue
            t = st.geometric_mean([p[0] for p in pairs])
            d = st.geometric_mean([p[1] for p in pairs])
            summary[kind] = t
            line += f"{t:11.2f}x {d:9.2f}x "
        rows.append((graph, summary))
        print(line)

    print()
    for kind in KINDS:
        vals = [s[kind] for _, s in rows if kind in s]
        if not vals:
            continue
        best = max(rows, key=lambda r: r[1].get(kind, 0))
        worst = min(rows, key=lambda r: r[1].get(kind, 9))
        print(f"{kind}: geomean {st.geometric_mean(vals):.2f}x | "
              f"best {best[1][kind]:.2f}x ({best[0]}) | worst {worst[1][kind]:.2f}x ({worst[0]})")
    print(f"\n{len(shared)} shared cells; cells missing from either set are skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
