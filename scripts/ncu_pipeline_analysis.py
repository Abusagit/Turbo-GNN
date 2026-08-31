#!/usr/bin/env python3
"""Counter-level audit of the cp.async pipeline: is it enabled, is it used, and does it overlap?

Three questions, each answered by a metric rather than by a timing:

  1. ENABLED  -- does the launched instantiation carry PIPELINE_STAGES > 0?
                 Read back from the kernel's dynamic shared memory, which grows with depth.
  2. USED     -- are cp.async instructions actually issued?
                 `smsp__inst_executed_op_ldgsts` counts LDGSTS (the sm_80 cp.async op). Zero
                 means the pipelined branch never ran, whatever the flag said.
  3. OVERLAPS -- does it hide latency?
                 `long_scoreboard` is the stall reason for waiting on a global load. A working
                 prefetch moves stall weight off long_scoreboard; a depth-1 pipeline does not,
                 because it waits on the copy it just issued.

Run under sudo: counter collection is root-only on this driver (CVE-2018-6260 mitigation).
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from run_kernel_benchmark_matrix import GRAPHS  # noqa: E402

# Kept deliberately small: ncu serialises every kernel launch, so each extra metric costs a
# full replay pass. These are the ones that separate "not enabled" from "enabled but useless".
METRICS = [
    # (2) is the pipeline actually issuing async copies?
    "smsp__inst_executed_op_ldgsts.sum",
    # (3) where do warps stall? long_scoreboard == waiting on a global load.
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct",
    # memory-level parallelism and whether it turned into bandwidth
    "dram__bytes.sum",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    # what the depth cost us
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "launch__registers_per_thread",
    "launch__shared_mem_per_block_dynamic",
    "launch__grid_size",
    "gpu__time_duration.sum",
]


def run_one(graph: str, cfg: str, conv: str, dim: int, mode: str, stages: int, gpu: int, out_dir: Path) -> list[dict]:
    """Profile one configuration and return one row per kernel launch."""
    bench = [
        sys.executable,
        str(REPO / "scripts/benchmark_kernels.py"),
        "--backend",
        "cuda",
        "--conv",
        conv,
        "--dataset",
        cfg,
        "--feature-dim",
        str(dim),
        "--heads",
        "1",
        "--mode",
        mode,
        # ncu replays every kernel; keep the iteration count at the floor.
        "--iters",
        "1",
        "--warmup",
        "0",
        "-K",
        "schedule=one_per_block",
        "-K",
        "forward_bucket_launch=concurrent",
        "-K",
        "backward_bucket_launch=concurrent",
        "-K",
        f"pipeline_stages={stages}",
        "-K",
        f"backward_pipeline_stages={stages}",
        "--quantile",
        "0.99",
    ]
    cmd = [
        "ncu",
        "--target-processes",
        "all",
        "--csv",
        "--page",
        "raw",
        "--metrics",
        ",".join(METRICS),
        # Only the aggregation kernels matter; skip PyTorch's own launches.
        "--kernel-name-base",
        "demangled",
        *bench,
    ]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(REPO))
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=3600)
    (out_dir / f"{graph}__{conv}__d{dim}__{mode}__s{stages}.log").write_text(proc.stdout + "\n" + proc.stderr)

    rows: list[dict] = []
    body = proc.stdout[proc.stdout.index('"ID"') :] if '"ID"' in proc.stdout else ""
    if not body:
        return rows
    for r in csv.DictReader(io.StringIO(body)):
        name = r.get("Kernel Name", "")
        # our kernels only -- attention, aggregation, slice and merge
        if not any(t in name for t in ("GATv2", "GraphAttention", "graph_attn", "reduction_aggr")):
            continue
        rows.append(
            {
                "graph": graph,
                "conv": conv,
                "dim": dim,
                "mode": mode,
                "stages": stages,
                "kernel": name.split("(")[0][:70],
                "metric": r.get("Metric Name", ""),
                "value": r.get("Metric Value", ""),
            }
        )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", nargs="+", default=["ogbn-arxiv", "ogbn-products", "web-fraud"])
    ap.add_argument("--conv", nargs="+", default=["gt", "gat_v2"])
    ap.add_argument("--dims", type=int, nargs="+", default=[128])
    ap.add_argument("--modes", nargs="+", default=["forward", "backward"])
    ap.add_argument("--stages", type=int, nargs="+", default=[0, 2, 3])
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--out", default="reports/ncu-pipeline")
    args = ap.parse_args()

    out_dir = REPO / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    by_graph = dict(GRAPHS)

    all_rows: list[dict] = []
    todo = [
        (g, c, d, m, s)
        for g in args.graphs
        for c in args.conv
        for d in args.dims
        for m in args.modes
        for s in args.stages
    ]
    for i, (g, c, d, m, s) in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {g} {c} d{d} {m} stages={s}", flush=True)
        try:
            rows = run_one(g, by_graph[g], c, d, m, s, args.gpu, out_dir)
        except subprocess.TimeoutExpired:
            print("    timed out", flush=True)
            continue
        print(f"    {len(rows)} metric rows", flush=True)
        all_rows.extend(rows)

    (out_dir / "metrics.json").write_text(json.dumps(all_rows, indent=1))
    print(f"\n{len(all_rows)} rows -> {out_dir / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
