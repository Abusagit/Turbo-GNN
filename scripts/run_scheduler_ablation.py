"""Isolate what the scheduler and concurrent streams add, on top of already-tuned kernels.

Every other kernel parameter is pinned to the value autotuning already selected for that cell,
so nothing in the comparison is confounded by tuning that predates this work. Only two things
vary: the node->block schedule (with its visit order) and whether the light/heavy buckets launch
concurrently. The whole grid is measured inside one process against one loaded graph, which is
what makes this affordable -- re-autotuning each arm cost hours per cell on the large graphs.

Each cell yields three numbers:

    A  baseline    one_per_block, natural order, sequential buckets
    B  scheduling  best (schedule x node order), still sequential
    C  + streams   the same grid with concurrent buckets

B/A is what the scheduler adds; C/B is what concurrency adds on top of it.

    python scripts/run_scheduler_ablation.py --out reports/scheduler-ablation
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from benchmark_kernels import CUDA_CONV_PARAMS  # noqa: E402
from free_gpus import free_indices, is_free, wait_until_free  # noqa: E402
from run_kernel_benchmark_matrix import GRAPHS  # noqa: E402

SCHEDULES = "one_per_block,grid_stride,precomputed,dynamic"
ORDERS = "natural,degree,locality"
#: Parameters this ablation varies; anything else is replayed from the autotuned configuration.
VARIED = {"schedule", "blocks_per_sm", "sched_chunk", "forward_bucket_launch", "backward_bucket_launch"}


def replay_args(source: Path, graph: str, conv: str, dim: int, mode: str) -> list[str] | None:
    """`-K` flags pinning this cell's kernel parameters to the autotuned argmax."""
    if not source.exists():
        return None
    cells = json.loads(source.read_text())["cells"]
    hit = next(
        (c for c in cells if c["graph"] == graph and c["conv"] == conv and c["head_dim"] == dim and c["mode"] == mode),
        None,
    )
    if hit is None:
        return None
    valid = {p.name for p in CUDA_CONV_PARAMS[conv]}
    out: list[str] = []
    for k, v in (hit.get("kernel_config") or {}).items():
        if k in VARIED:
            continue
        name = k if k in valid else k.removeprefix("forward_").removeprefix("backward_")
        if name in valid:
            out += ["-K", f"{name}={v}"]
    if hit.get("quantile") is not None:
        out += ["--quantile", str(hit["quantile"])]
    return out


def run_cell(gpu: int, graph: str, cfg: str, conv: str, dim: int, mode: str, out_dir: Path, args):
    tag = f"{graph}__{conv}__d{dim}__{mode}"
    dest = out_dir / f"{tag}.json"
    if dest.exists() and not args.force:
        return tag, True, "cached"
    replay = replay_args(REPO / args.replay_source, graph, conv, dim, mode)
    if replay is None:
        return tag, False, "no autotuned configuration to replay"

    # Only the bucket parameter for the pass being timed; the other has no effect here and
    # would double the grid for nothing.
    bucket = "forward_bucket_launch" if mode == "forward" else "backward_bucket_launch"
    cmd = [
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
        "--iters",
        str(args.iters),
        "--warmup",
        str(args.warmup),
        "--sweep",
        f"node_order={ORDERS}",
        "--sweep-kernel",
        f"schedule={args.schedules}",
        "--sweep-kernel",
        f"{bucket}=sequential,concurrent",
        "--json-out",
        str(dest),
        *replay,
    ]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(REPO))
    if not wait_until_free(gpu, timeout_s=args.wait_free_s):
        return tag, False, f"gpu {gpu} busy"
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO, env=env, timeout=args.timeout)
    if proc.returncode != 0 or not dest.exists():
        err = (proc.stderr or "").strip().splitlines()
        return tag, False, (err[-1][:160] if err else f"exit {proc.returncode}")
    if not is_free(gpu):
        dest.unlink(missing_ok=True)  # another tenant appeared; the timing is not usable
        return tag, False, "gpu was shared during the run"
    return tag, True, f"{time.time() - t0:.0f}s"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="reports/scheduler-ablation")
    p.add_argument("--replay-source", default="reports/autotune-comparison/grid.json")
    p.add_argument("--graphs", nargs="+", default=[g for g, _ in GRAPHS])
    p.add_argument("--conv", nargs="+", default=["min_aggr", "gat_v2", "gt"])
    p.add_argument("--head-dims", type=int, nargs="+", default=[128, 256])
    p.add_argument("--modes", nargs="+", default=["forward", "backward"])
    p.add_argument("--schedules", default=SCHEDULES)
    p.add_argument("--gpu", type=int, default=None, help="pin to this GPU (default: first free)")
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--wait-free-s", type=float, default=3600.0)
    p.add_argument("--timeout", type=int, default=7200)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    free = free_indices(64)
    gpu = args.gpu if args.gpu is not None else (free[0] if free else None)
    if gpu is None:
        raise SystemExit("no free GPU")
    out_dir = REPO / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    by_graph = dict(GRAPHS)

    jobs = [
        (g, by_graph[g], c, d, m)
        for g in args.graphs
        if g in by_graph
        for c in args.conv
        for d in args.head_dims
        for m in args.modes
    ]
    n_sched = len(args.schedules.split(","))
    print(f"GPU {gpu} -> {out_dir.relative_to(REPO)}", flush=True)
    print(
        f"{len(jobs)} cells, {n_sched * len(ORDERS.split(',')) * 2} configurations each, all in one process per cell",
        flush=True,
    )

    failed = 0
    for i, (g, cfg, c, d, m) in enumerate(jobs, 1):
        try:
            tag, ok, note = run_cell(gpu, g, cfg, c, d, m, out_dir, args)
        except subprocess.TimeoutExpired:
            tag, ok, note = f"{g}__{c}__d{d}__{m}", False, "timeout"
        if not ok:
            failed += 1
            print(f"  [{i}/{len(jobs)}] FAIL {tag}: {note}", file=sys.stderr, flush=True)
        elif note != "cached":
            print(f"  [{i}/{len(jobs)}] {tag}  {note}", flush=True)
    print(f"\n{len(list(out_dir.glob('*.json')))} cells in {out_dir.relative_to(REPO)}; {failed} failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
