"""Drive `benchmark_kernels.py` over every graph, conv, head dim and pass.

`benchmark_kernels.py` measures one configuration per process, which is the right shape for a
single question but not for a matrix: loading ogbn-products costs more than the measurement
does. This runner amortises that by sweeping the node orders inside each process
(`--sweep node_order=...`), so a graph is loaded once per schedule rather than once per cell,
and by spreading graphs across the GPUs that are actually free.

Every configuration is written out as its own JSON file, so a partial run is still usable and
re-running skips what is already on disk.

    python scripts/run_kernel_benchmark_matrix.py --out reports/kernel-benchmarks
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

#: The benchmark set: everything in configs/datasets/main/, plus ogbn-proteins, which lives in
#: secondary/ because it has no node features and so is not a node-classification benchmark.
GRAPHS = [
    ("ogbn-arxiv", "configs/datasets/main/ogbn_arxiv.yaml"),
    ("ogbn-products", "configs/datasets/main/ogbn_products.yaml"),
    ("ogbn-proteins", "configs/datasets/secondary/ogbn_proteins.yaml"),
    ("avazu-ctr", "configs/datasets/main/avazu_ctr.yaml"),
    ("city-reviews", "configs/datasets/main/city_reviews.yaml"),
    ("city-roads-M", "configs/datasets/main/city_roads_m.yaml"),
    ("hm-categories", "configs/datasets/main/hm_categories.yaml"),
    ("tolokers-2", "configs/datasets/main/tolokers_2.yaml"),
    ("twitch-views", "configs/datasets/main/twitch_views.yaml"),
    # The rest of the GraphLand set.
    ("artnet-exp", "configs/datasets/graphland_remaining/artnet_exp.yaml"),
    ("city-roads-L", "configs/datasets/graphland_remaining/city_roads_l.yaml"),
    ("pokec-regions", "configs/datasets/graphland_remaining/pokec_regions.yaml"),
    ("web-fraud", "configs/datasets/graphland_remaining/web_fraud.yaml"),
    # The small citation networks, which sit two orders of magnitude below everything above
    # and are where per-launch overhead should dominate if it dominates anywhere.
    ("cora", "configs/datasets/secondary/cora.yaml"),
    ("citeseer", "configs/datasets/secondary/citeseer.yaml"),
    ("pubmed", "configs/datasets/secondary/pubmed.yaml"),
]

CONVS = ["min_aggr", "gat_v2", "gt"]
SCHEDULES = ["one_per_block", "grid_stride", "precomputed", "dynamic"]
ORDERS = "natural,degree,locality"


def free_gpus(count: int) -> list[int]:
    out = subprocess.run(
        [sys.executable, str(REPO / "scripts/free_gpus.py"), "--count", str(count)],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    if out.returncode != 0:
        raise SystemExit(f"not enough free GPUs: {out.stderr.strip()}")
    return [int(x) for x in out.stdout.strip().split(",")]


def run_one(
    gpu: int, graph: str, cfg: str, conv: str, dim: int, mode: str, sched: str, out_dir: Path, args
) -> tuple[str, bool, str]:
    tag = f"{graph}__{conv}__d{dim}__{mode}__{sched}"
    dest = out_dir / f"{tag}.json"
    if dest.exists() and not args.force:
        return tag, True, "cached"

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
        "--sweep",
        f"node_order={ORDERS}",
        "-K",
        f"schedule={sched}",
        "--iters",
        str(args.iters),
        "--warmup",
        str(args.warmup),
        "--json-out",
        str(dest),
    ]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(REPO))
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO, env=env, timeout=args.timeout)
    if proc.returncode != 0 or not dest.exists():
        err = (proc.stderr or "").strip().splitlines()
        return tag, False, (err[-1][:160] if err else f"exit {proc.returncode}")
    return tag, True, f"{time.time() - t0:.0f}s"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="reports/kernel-benchmarks", help="directory for the per-run JSON files")
    p.add_argument("--graphs", nargs="+", default=[g for g, _ in GRAPHS])
    p.add_argument("--conv", nargs="+", default=CONVS)
    p.add_argument("--head-dims", type=int, nargs="+", default=[128, 256])
    p.add_argument("--modes", nargs="+", default=["forward", "backward"])
    p.add_argument("--gpus", type=int, default=3, help="how many free GPUs to spread across")
    p.add_argument("--iters", type=int, default=100, help="timed budget in ms (benchmark_kernels default)")
    p.add_argument("--warmup", type=int, default=20, help="warmup budget in ms")
    p.add_argument("--timeout", type=int, default=3600, help="per-run timeout in seconds")
    p.add_argument("--force", action="store_true", help="re-run configurations already on disk")
    args = p.parse_args()

    out_dir = REPO / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    gpus = free_gpus(args.gpus)
    print(f"free GPUs: {gpus}   ->  {out_dir.relative_to(REPO)}", flush=True)

    by_graph = dict(GRAPHS)
    jobs: dict[int, list[tuple]] = {g: [] for g in gpus}
    # One graph is pinned to one GPU so its dataset cache and CSR build are reused across that
    # graph's whole column of the matrix, and two workers never load the same big graph at once.
    for i, graph in enumerate([g for g in args.graphs if g in by_graph]):
        gpu = gpus[i % len(gpus)]
        for conv in args.conv:
            for dim in args.head_dims:
                for mode in args.modes:
                    for sched in SCHEDULES:
                        jobs[gpu].append((graph, by_graph[graph], conv, dim, mode, sched))

    total = sum(len(v) for v in jobs.values())
    print(f"{total} runs ({len(ORDERS.split(','))} node orders swept inside each)", flush=True)
    done = failed = 0

    def worker(gpu: int) -> None:
        nonlocal done, failed
        for graph, cfg, conv, dim, mode, sched in jobs[gpu]:
            try:
                tag, ok, note = run_one(gpu, graph, cfg, conv, dim, mode, sched, out_dir, args)
            except subprocess.TimeoutExpired:
                tag, ok, note = f"{graph}__{conv}__d{dim}__{mode}__{sched}", False, "timeout"
            done += 1
            if not ok:
                failed += 1
                print(f"  [{done}/{total}] FAIL {tag}: {note}", file=sys.stderr, flush=True)
            elif note != "cached":
                print(f"  [{done}/{total}] {tag}  {note}", flush=True)

    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        list(pool.map(worker, gpus))

    files = sorted(out_dir.glob("*.json"))
    print(f"\n{len(files)} JSON files in {out_dir.relative_to(REPO)}; {failed} run(s) failed")
    return 1 if failed and not files else 0


if __name__ == "__main__":
    raise SystemExit(main())
