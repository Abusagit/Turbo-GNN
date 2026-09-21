"""How much of the work does the heavy bucket actually hold?

The edge-parallel decomposition only helps the heavy bucket, so its payoff is Amdahl-bounded by
that bucket's share of runtime. Two numbers decide whether the work is worth doing:

  edge share  E_heavy / E     -- how much of the arithmetic lives there
  time share  heavy kernel time / total conv time

A bucket that is 5% of edges but 40% of time is exactly the case this optimisation exists for. A
bucket that is 5% of edges and 6% of time caps the win at 6%, and the effort belongs elsewhere.

Edge share is computed directly from the CSR. Time share needs per-kernel durations, which come
from `ncu`; light and heavy launches of the same kernel are told apart by their warps-per-block
template argument (light is dispatched over {1,2,4}, heavy over {8,16,32}), except in the
reduction path where the two have different kernel names outright.

    python scripts/heavy_share.py --gpu 1
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import subprocess
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from run_kernel_benchmark_matrix import GRAPHS  # noqa: E402

NCU = "/usr/local/cuda/bin/ncu"
HEAVY_WARPS = {8, 16, 32}


def edge_share(bg) -> dict:
    """Fraction of edges owned by the heavy bucket, per direction."""
    out = {}
    for d in ("forward", "backward"):
        indptr = getattr(bg.repr, f"{d}_indptr")
        heavy = getattr(bg.repr, f"{d}_heavy_nodes")
        ip = indptr.to(torch.int64)
        deg = ip[1:] - ip[:-1]
        e_total = int(deg.sum())
        e_heavy = int(deg.index_select(0, heavy.to(torch.int64)).sum()) if heavy.numel() else 0
        out[d] = {
            "edges": e_total,
            "heavy_nodes": int(heavy.numel()),
            "heavy_edges": e_heavy,
            "edge_share": e_heavy / e_total if e_total else 0.0,
            "mean_heavy_degree": e_heavy / heavy.numel() if heavy.numel() else 0.0,
        }
    return out


def classify(kernel: str) -> str | None:
    """light / heavy / other, from the kernel name and its warps template argument."""
    if "reduction_aggr_forward_light" in kernel:
        return "light"
    if "reduction_aggr_forward_heavy" in kernel or "unpack_results" in kernel:
        return "heavy"
    # Attention kernels carry <SK, WARPS, D, ...>; the second argument separates the buckets.
    m = re.search(
        r"(?:GraphAttentionForward_CSR_MH_v2_D|GATv2Forward_Kernel|"
        r"graph_attn_backward_csrT_kernel_D|GATv2Backward_AL|GATv2Backward_R)<\s*\d+\s*,\s*(\d+)",
        kernel,
    )
    if m:
        return "heavy" if int(m.group(1)) in HEAVY_WARPS else "light"
    return None


def time_share(gpu: int, graph: str, dim: int, mode: str, timeout: int = 900) -> dict:
    cmd = [
        "sudo",
        "-n",
        "env",
        f"CUDA_VISIBLE_DEVICES={gpu}",
        f"PYTHONPATH={REPO}",
        NCU,
        "--profile-from-start",
        "off",
        "--metrics",
        "gpu__time_duration.sum",
        "--csv",
        str(REPO / ".venv/bin/python3"),
        str(REPO / "scripts/roofline_counters.py"),
        "--graph",
        graph,
        "--head-dim",
        str(dim),
        "--mode",
        mode,
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {}
    start = r.stdout.find('"ID","Process ID"')
    if start < 0:
        return {}
    acc: dict[str, dict[str, float]] = {}
    conv = None
    for row in csv.DictReader(io.StringIO(r.stdout[start:])):
        k = row.get("Kernel Name", "")
        for c, ps in (
            ("min_aggr", ("reduction_aggr_", "unpack_results_")),
            ("gat_v2", ("GATv2", "ReduceGradA")),
            ("gt", ("GraphAttention", "graph_attn_", "compute_D_mh")),
        ):
            if any(p in k for p in ps):
                conv = c
        if conv is None:
            continue
        try:
            ns = float(row["Metric Value"].replace(",", ""))
        except (KeyError, ValueError):
            continue
        bucket = classify(k) or "other"
        a = acc.setdefault(conv, {"light": 0.0, "heavy": 0.0, "other": 0.0})
        a[bucket] += ns
    return acc


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--graphs", nargs="+", default=[g for g, _ in GRAPHS])
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--modes", nargs="+", default=["forward", "backward"])
    p.add_argument("--out", default="reports/heavy-share")
    args = p.parse_args()

    from benchmark_kernels import load_graph

    torch.cuda.set_device(args.gpu)
    dev = torch.device("cuda")
    by_graph = dict(GRAPHS)
    rows = []
    print(f"{'graph':<16}{'E_heavy/E':>11}{'heavy nodes':>13}{'mean deg':>10}   time share (heavy) by conv", flush=True)
    for g in args.graphs:
        if g not in by_graph:
            continue
        ns = argparse.Namespace(
            dataset=by_graph[g],
            num_nodes=0,
            avg_degree=10,
            quantile=0.99,
            index_dtype="int32",
            self_loops=True,
            node_order="natural",
        )
        try:
            bg = load_graph(ns, dev, "cuda")
        except Exception as exc:
            print(f"  {g}: SKIPPED ({type(exc).__name__})", file=sys.stderr)
            continue
        es = edge_share(bg)
        del bg
        torch.cuda.empty_cache()

        ts = {m: time_share(args.gpu, g, args.head_dim, m) for m in args.modes}
        rows.append({"graph": g, "edge_share": es, "time_share": ts})

        f = es["forward"]
        parts = []
        for m in args.modes:
            for conv, a in sorted(ts.get(m, {}).items()):
                tot = a["light"] + a["heavy"] + a["other"]
                if tot:
                    parts.append(f"{conv[:3]}/{m[:3]} {a['heavy'] / tot * 100:.0f}%")
        print(
            f"{g:<16}{f['edge_share'] * 100:>10.1f}%{f['heavy_nodes']:>13,}{f['mean_heavy_degree']:>10.0f}   "
            + "  ".join(parts),
            flush=True,
        )

    out = REPO / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "heavy_share.json").write_text(json.dumps(rows, indent=1))
    print(f"\n-> {(out / 'heavy_share.json').relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
