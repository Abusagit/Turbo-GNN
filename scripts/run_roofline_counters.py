"""Run `roofline_counters.py` under ncu across every cell and collect measured DRAM traffic.

One ncu invocation per (graph, head_dim, mode); all three convolutions share it so each graph
is built once per invocation rather than once per cell. Counter access needs root on this
driver (see ERR_NVGPUCTRPERM), so each invocation goes through `sudo env` -- plain `sudo` would
drop CUDA_VISIBLE_DEVICES and PYTHONPATH and silently profile the wrong device.

    python scripts/run_roofline_counters.py --gpu 4 --out reports/roofline
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from run_kernel_benchmark_matrix import GRAPHS  # noqa: E402

NCU = "/usr/local/cuda/bin/ncu"
METRICS = "dram__bytes_read.sum,dram__bytes_write.sum"

# Kernel-name prefixes are unambiguous across the three families. Framework kernels (at::...)
# carry no family of their own, so they are attributed to whichever convolution launched most
# recently -- the convolutions run strictly in sequence, so that is well defined.
FAMILY = [
    ("min_aggr", ("reduction_aggr_", "unpack_results_")),
    ("gat_v2", ("GATv2", "ReduceGradA")),
    ("gt", ("GraphAttention", "graph_attn_", "compute_D_mh")),
]


def classify(kernel: str) -> str | None:
    for conv, prefixes in FAMILY:
        if any(p in kernel for p in prefixes):
            return conv
    return None


def parse(out: str) -> dict[str, dict]:
    """Sum read+write bytes per convolution, in launch order, carrying framework kernels."""
    start = out.find('"ID","Process ID"')
    if start < 0:
        return {}
    rows = list(csv.DictReader(io.StringIO(out[start:])))
    acc: dict[str, dict] = {}
    current = None
    for r in rows:
        conv = classify(r.get("Kernel Name", ""))
        if conv:
            current = conv
        if current is None:
            continue  # kernels before the first convolution launched
        try:
            val = float(r["Metric Value"].replace(",", ""))
        except (KeyError, ValueError):
            continue
        a = acc.setdefault(current, {"read": 0.0, "write": 0.0, "launches": 0, "own_launches": 0})
        if r["Metric Name"] == "dram__bytes_read.sum":
            a["read"] += val
            a["launches"] += 1
            if conv:
                a["own_launches"] += 1
        elif r["Metric Name"] == "dram__bytes_write.sum":
            a["write"] += val
    return acc


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--out", default="reports/roofline")
    p.add_argument("--graphs", nargs="+", default=[g for g, _ in GRAPHS])
    p.add_argument("--head-dims", type=int, nargs="+", default=[128, 256])
    p.add_argument("--modes", nargs="+", default=["forward", "backward"])
    p.add_argument("--timeout", type=int, default=1800)
    args = p.parse_args()

    out_dir = REPO / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "counters.json"
    cells = json.loads(dest.read_text()) if dest.exists() else []
    done = {(c["graph"], c["head_dim"], c["mode"]) for c in cells}

    todo = [(g, d, m) for g in args.graphs for d in args.head_dims for m in args.modes if (g, d, m) not in done]
    print(f"{len(todo)} invocations to run ({len(done)} already present)", flush=True)

    for i, (graph, dim, mode) in enumerate(todo, 1):
        cmd = [
            "sudo",
            "-n",
            "env",
            f"CUDA_VISIBLE_DEVICES={args.gpu}",
            f"PYTHONPATH={REPO}",
            NCU,
            "--profile-from-start",
            "off",
            "--metrics",
            METRICS,
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
        t0 = time.time()
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
            acc = parse(r.stdout)
        except subprocess.TimeoutExpired:
            print(f"  [{i}/{len(todo)}] {graph} d{dim} {mode}: TIMEOUT", flush=True)
            continue
        if not acc:
            tail = (r.stderr or "").strip().splitlines()[-1:] or ["no output"]
            print(f"  [{i}/{len(todo)}] {graph} d{dim} {mode}: NO DATA -- {tail[0][:90]}", flush=True)
            continue
        for conv, a in acc.items():
            cells.append(
                {
                    "graph": graph,
                    "head_dim": dim,
                    "mode": mode,
                    "conv": conv,
                    "dram_read": a["read"],
                    "dram_write": a["write"],
                    "dram_total": a["read"] + a["write"],
                    "launches": a["launches"],
                    "kernel_launches": a["own_launches"],
                }
            )
        dest.write_text(json.dumps(cells, indent=1))
        gb = sum(a["read"] + a["write"] for a in acc.values()) / 1e9
        print(
            f"  [{i}/{len(todo)}] {graph} d{dim} {mode}: {len(acc)} convs, {gb:.2f} GB total, {time.time() - t0:.0f}s",
            flush=True,
        )

    print(f"\n{len(cells)} conv-cells -> {dest.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
