"""Counter-level comparison of the edge-sliced heavy path against the node-per-block one.

For each cell, profiles the same convolution twice -- `heavy_edge_slice=0` and the slice the
autotuner selected -- with that cell's argmax parameters applied to both, and attributes every
kernel launch to the light bucket, the heavy bucket, or the merge.

Attribution matters because light and heavy launches of the attention kernels share a kernel
name and differ only in their warps template argument (light is dispatched over {1,2,4}, heavy
over {8,16,32}). Without splitting on that, the heavy bucket's cost hides inside the total and
the whole point of the measurement is lost.

    sudo-capable: python scripts/run_edge_slice_ncu.py --gpu 0 --out reports/edge-slice
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

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(1, str(REPO / "scripts"))

NCU = "/usr/local/cuda/bin/ncu"
METRICS = ",".join(
    [
        "gpu__time_duration.sum",
        "dram__bytes_read.sum",
        "dram__bytes_write.sum",
        "sm__warps_active.avg.pct_of_peak_sustained_active",
        "launch__grid_size",
        "launch__registers_per_thread",
    ]
)
PEAK = 1.747e12  # measured on this A100, not the datasheet figure
HEAVY_WARPS = {8, 16, 32}
SWEEP = REPO / "reports/autotune-edge-slice"


def bucket_of(kernel: str) -> str:
    # "Merge" must be tested before "Slice": GraphAttentionMergeSlices_D contains both, and
    # folding the merge into the heavy bucket would hide exactly the overhead this measures.
    if "Merge" in kernel or "merge_slices" in kernel or "unpack_results" in kernel:
        return "merge"
    if "Slice" in kernel or "slice_kernel" in kernel:
        return "heavy"
    if "reduction_aggr_forward_light" in kernel:
        return "light"
    if "reduction_aggr_forward_heavy" in kernel:
        return "heavy"
    m = re.search(r"<\s*\d+\s*,\s*(\d+)", kernel)
    if m:
        return "heavy" if int(m.group(1)) in HEAVY_WARPS else "light"
    return "other"


def profile(
    gpu: int, graph: str, conv: str, dim: int, mode: str, sl: int, params: Path | None, timeout: int = 1800
) -> dict:
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
        METRICS,
        "--csv",
        str(REPO / ".venv/bin/python3"),
        str(REPO / "scripts/profile_edge_slice.py"),
        "--graph",
        graph,
        "--conv",
        conv,
        "--head-dim",
        str(dim),
        "--mode",
        mode,
        "--slice",
        str(sl),
    ]
    if params:
        cmd += ["--params-json", str(params)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {}
    i = r.stdout.find('"ID","Process ID"')
    if i < 0:
        return {}
    acc: dict[str, dict[str, float]] = {}
    for row in csv.DictReader(io.StringIO(r.stdout[i:])):
        b = bucket_of(row.get("Kernel Name", ""))
        try:
            val = float(row["Metric Value"].replace(",", ""))
        except (KeyError, ValueError):
            continue
        m = row["Metric Name"]
        d = acc.setdefault(b, {})
        # occupancy / registers / grid describe a launch, so take the max rather than a sum
        d[m] = (
            max(d.get(m, 0.0), val) if ("pct_of_peak" in m or "registers" in m or "grid" in m) else d.get(m, 0.0) + val
        )
    out = {}
    for b, d in acc.items():
        us = d.get("gpu__time_duration.sum", 0) / 1e3
        by = d.get("dram__bytes_read.sum", 0) + d.get("dram__bytes_write.sum", 0)
        out[b] = {
            "us": us,
            "gb": by / 1e9,
            "pct_peak": (by / (us * 1e-6) / PEAK * 100) if us else 0.0,
            "occ": d.get("sm__warps_active.avg.pct_of_peak_sustained_active", 0.0),
            "grid": d.get("launch__grid_size", 0.0),
            "reg": d.get("launch__registers_per_thread", 0.0),
        }
    return out


def selected_slice(graph: str, conv: str, dim: int, mode: str) -> tuple[int, Path | None]:
    f = SWEEP / f"{graph}__{conv}__d{dim}__{mode}__autotuned-concurrent.json"
    if not f.exists():
        return 0, None
    rec = json.loads(f.read_text())
    kc = rec.get("autotune_selected", {}).get("kernel_config", {})
    key = "forward_heavy_edge_slice" if mode == "forward" else "backward_heavy_edge_slice"
    return int(kc.get(key) or 0), f


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--graphs", nargs="+", default=["ogbn-arxiv", "web-fraud", "twitch-views", "ogbn-proteins"])
    p.add_argument("--conv", nargs="+", default=["gt", "gat_v2", "min_aggr"])
    p.add_argument("--head-dims", type=int, nargs="+", default=[128])
    p.add_argument("--modes", nargs="+", default=["forward", "backward"])
    p.add_argument("--out", default="reports/edge-slice")
    p.add_argument(
        "--force-slice",
        type=int,
        default=0,
        help="profile this slice even where the autotuner chose 0; shows why it declined",
    )
    args = p.parse_args()

    rows = []
    print(
        f"{'cell':<40}{'bucket':<8}{'old us':>10}{'new us':>10}{'gain':>7}"
        f"{'old %pk':>9}{'new %pk':>9}{'old occ':>9}{'new occ':>9}",
        flush=True,
    )
    for graph in args.graphs:
        for conv in args.conv:
            for dim in args.head_dims:
                for mode in args.modes:
                    sl, params = selected_slice(graph, conv, dim, mode)
                    if sl == 0 and args.force_slice:
                        sl = args.force_slice  # measure the path the autotuner rejected
                    if sl == 0:
                        print(f"{graph}/{conv}/d{dim}/{mode}: autotuner chose slice=0, skipping", file=sys.stderr)
                        continue
                    old = profile(args.gpu, graph, conv, dim, mode, 0, params)
                    new = profile(args.gpu, graph, conv, dim, mode, sl, params)
                    if not old or not new:
                        print(f"{graph}/{conv}/d{dim}/{mode}: NO DATA", file=sys.stderr)
                        continue
                    tag = f"{graph}/{conv}/d{dim}/{mode[:3]} sl={sl}"
                    rec = {
                        "graph": graph,
                        "conv": conv,
                        "head_dim": dim,
                        "mode": mode,
                        "slice": sl,
                        "old": old,
                        "new": new,
                    }
                    rows.append(rec)
                    for b in ("heavy", "light", "merge"):
                        o, nw = old.get(b), new.get(b)
                        if not (o or nw):
                            continue
                        ou, nu = (o or {}).get("us", 0), (nw or {}).get("us", 0)
                        gain = f"{ou / nu:.2f}x" if ou and nu else "-"
                        print(
                            f"{tag[:39]:<40}{b:<8}{ou:>10.1f}{nu:>10.1f}{gain:>7}"
                            f"{(o or {}).get('pct_peak', 0):>8.1f}%{(nw or {}).get('pct_peak', 0):>8.1f}%"
                            f"{(o or {}).get('occ', 0):>8.1f}%{(nw or {}).get('occ', 0):>8.1f}%",
                            flush=True,
                        )
                    to = sum(v["us"] for v in old.values())
                    tn = sum(v["us"] for v in new.values())
                    print(f"{tag[:39]:<40}{'TOTAL':<8}{to:>10.1f}{tn:>10.1f}{to / tn if tn else 0:>6.2f}x", flush=True)
                    rec["total_old_us"], rec["total_new_us"] = to, tn

    out = REPO / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "ncu_comparison.json").write_text(json.dumps(rows, indent=1))
    print(f"\n{len(rows)} cells -> {(out / 'ncu_comparison.json').relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
