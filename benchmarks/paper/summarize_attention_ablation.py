"""Turn the outputs of run_attention_ablation.sh and run_attention_baselines.sh into the rows
of Table 1 (--heads 2 --width 128) or one block of Appendix J.

Every point of a cell's sweep is first reduced to its median over the repeats (r1, r2, ...).
Each column then takes the fastest point that uses only the techniques enabled so far:

    prior        quantile -1 (one bucket), natural order, sequential, no slicing, no staging
    +bucketing   any split quantile
    +ordering    ... and degree order
    +concurrent  ... and concurrent bucket launch
    +slicing     ... and heavy-node edge slicing
    +async       ... and asynchronous staging

so a technique that does not help a graph is simply not selected, as the autotuner would do.

    python benchmarks/paper/summarize_attention_ablation.py --heads 2 --width 128
"""

import argparse
import collections
import glob
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from graphs import ATTENTION_GRAPHS  # noqa: E402

RUNGS = [
    ("prior", lambda k: k[0] == -1 and k[1] == "natural" and k[2] == "sequential" and k[3] == 0 and k[4] == 0 and k[5] == 0),
    ("+bucketing", lambda k: k[1] == "natural" and k[2] == "sequential" and k[3] == 0 and k[4] == 0 and k[5] == 0),
    ("+ordering", lambda k: k[2] == "sequential" and k[3] == 0 and k[4] == 0 and k[5] == 0),
    ("+concurrent", lambda k: k[3] == 0 and k[4] == 0 and k[5] == 0),
    ("+slicing", lambda k: k[4] == 0 and k[5] == 0),
    ("+async", lambda k: True),
]


def ladder(root, heads, width, stem):
    points = collections.defaultdict(list)
    for path in glob.glob(f"{root}/h{heads}-d{width}-*/{stem}__forward.json"):
        j = json.load(open(path))
        for s in j["sweep"]:
            gc, kc = s["graph_config"], s["kernel_config"]
            key = (gc["quantile"], gc["node_order"], kc["forward_bucket_launch"],
                   kc["forward_heavy_edge_slice"], kc["pipeline_stages"], kc["heavy_pipeline_stages"])
            points[key].append(s["ms_per_iter"])
    return {k: statistics.median(v) for k, v in points.items()}


def baseline(root, backend, heads, width, stem):
    p = f"{root}/{backend}/gat_v2-h{heads}-d{width}-forward__{stem}.json"
    return json.load(open(p))["ms_per_iter"] if os.path.exists(p) else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--width", type=int, default=128, help="per-head width (--feature-dim)")
    ap.add_argument("--ablation", default="results/attention_ablation")
    ap.add_argument("--baselines", default="results/attention_baselines")
    a = ap.parse_args()

    cols = ["DGL", "PyG"] + [n for n, _ in RUNGS]
    print(f"GATv2 forward, fp16, {a.heads} heads x {a.width}, milliseconds (OOM: baseline did not fit)")
    print(f"{'graph':15s}" + "".join(f"{c:>12s}" for c in cols))
    over = collections.defaultdict(list)
    for name, cfg in ATTENTION_GRAPHS:
        stem = os.path.basename(cfg)[: -len(".yaml")]
        pts = ladder(a.ablation, a.heads, a.width, stem)
        if not pts:
            continue
        rungs = [min(v for k, v in pts.items() if f(k)) for _, f in RUNGS]
        dgl, pyg = (baseline(a.baselines, b, a.heads, a.width, stem) for b in ("dgl", "pyg"))
        cells = [dgl, pyg] + rungs
        print(f"{name:15s}" + "".join(f"{c:12.3f}" if c else f"{'OOM':>12s}" for c in cells))
        over["prior"].append(rungs[0] / rungs[-1])
        if dgl:
            over["DGL"].append(dgl / rungs[-1])
        if pyg:
            over["PyG"].append(pyg / rungs[-1])
    for k, v in over.items():
        print(f"median speedup of +async over {k}: {statistics.median(v):.2f}x ({len(v)} graphs)")


if __name__ == "__main__":
    main()
