"""Head-to-head CPU neighbor-sampling & DataLoader benchmark: Ours (C++17, no OpenMP) vs DGL.

Supports BOTH:
- ``--device cpu``: pure CPU timing (uses ZERO GPU; safe with ``CUDA_VISIBLE_DEVICES=""``).
- ``--device cuda``: full CPU-sampling -> pinned H2D -> GPU-block pipeline (ready for GPU runs).

Each side (DGL vs Ours) runs in an isolated subprocess so thread pools and host allocators
never interfere across runs. Every row records ``avg_src`` and ``avg_edges`` to verify
identical sampled workload parity.

Examples:
    # CPU-only benchmark on real graphs (zero GPU usage):
    CUDA_VISIBLE_DEVICES="" TURBO_GNN_ALLOW_MISSING_EXTENSION=1 \\
        /home/fvelikon/envs/turbo-dgl/bin/python benchmarks/bench_cpu_sampler_vs_dgl.py \\
        --datasets configs/datasets/main/ogbn_arxiv.yaml,configs/datasets/main/ogbn_products.yaml,configs/datasets/secondary/reddit.yaml,configs/datasets/graphland_remaining/pokec_regions.yaml \\
        --device cpu --threads 1,8,16 --workers 0,4,8

    # Future GPU end-to-end benchmark (CPU graph -> pinned H2D -> GPU):
    CUDA_VISIBLE_DEVICES=6 TURBO_GNN_ALLOW_MISSING_EXTENSION=1 \\
        /home/fvelikon/envs/turbo-dgl/bin/python benchmarks/bench_cpu_sampler_vs_dgl.py \\
        --datasets configs/datasets/main/ogbn_arxiv.yaml,configs/datasets/main/ogbn_products.yaml,configs/datasets/secondary/reddit.yaml \\
        --device cuda --threads 1,16 --workers 0,4,8 --pin-memory on
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("OUTDATED_IGNORE", "1")
sys.modules.setdefault("outdated", None)

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
DEFAULT_DATASETS = [
    "configs/datasets/main/ogbn_arxiv.yaml",
    "configs/datasets/main/ogbn_products.yaml",
    "configs/datasets/secondary/reddit.yaml",
    "configs/datasets/graphland_remaining/pokec_regions.yaml",
]


def _in_csr(edge_index: torch.Tensor, num_nodes: int) -> tuple[torch.Tensor, torch.Tensor]:
    src, dst = edge_index[0], edge_index[1]
    order = torch.argsort(dst, stable=True)
    counts = torch.bincount(dst, minlength=num_nodes)
    indptr = torch.zeros(num_nodes + 1, dtype=torch.int64)
    indptr[1:] = torch.cumsum(counts, 0)
    return indptr, src[order].to(torch.int32).contiguous()


def _load_dataset_cpu(dataset_yaml: str, dtype: torch.dtype):
    from src.data.datasets import DatasetConfig, load_single_graph

    with open(dataset_yaml, encoding="utf-8") as f:
        ds = yaml.safe_load(f)["dataset"]
    sample = load_single_graph(
        DatasetConfig(
            source=ds["source"],
            name=ds["name"],
            root=os.environ.get("TURBO_GNN_DATA_ROOT") or ds.get("root", "data"),
            conv_backend="pyg",
            allow_random_split=ds.get("allow_random_split", False),
            native_targets=True,
        )
    )
    N = sample.x.shape[0]
    x = sample.x.to(device="cpu", dtype=dtype).contiguous()
    y_raw = sample.y.squeeze(-1) if (sample.y.ndim == 2 and sample.y.shape[1] == 1) else sample.y
    y = (y_raw if y_raw.is_floating_point() else y_raw.long()).to(device="cpu").contiguous()
    train_idx = sample.train_mask.nonzero(as_tuple=False).view(-1).contiguous()
    return ds["name"], sample.edge_index, N, x, y, train_idx


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _run_worker(args: argparse.Namespace) -> dict[str, Any]:
    from src.sampling import CPUBatchMaker, CPUNeighborSampler, CPUPrefetchLoader, seed_batches

    device = torch.device("cuda", torch.cuda.current_device()) if args.device == "cuda" else torch.device("cpu")
    dtype = DTYPES[args.dtype]
    fanouts = [int(f) for f in args.fanouts.split(",")]
    threads_list = [int(t) for t in args.threads.split(",") if t.strip()]
    workers_list = [int(w) for w in args.workers.split(",") if w.strip()]

    name, edge_index, N, x, y, train_idx = _load_dataset_cpu(args.dataset, dtype)
    g_rng = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(train_idx.numel(), generator=g_rng)
    num_batches = min(args.batches, max(1, train_idx.numel() // args.batch_size))
    sub_idx = train_idx[perm[: num_batches * args.batch_size]].contiguous()
    seeds_list = seed_batches(sub_idx, args.batch_size, shuffle=False)
    sample_batches = seeds_list[: min(args.sample_batches, len(seeds_list))]

    records: list[dict[str, Any]] = []

    if args.side == "dgl":
        import dgl

        # DGL 2.4's fused=True + idtype=int32 fails to reset its internal mapping across batches,
        # silently dropping ~90% of nodes after batch 0. Use idtype=int64 for valid fused=True.
        dgl_idt = torch.int64 if args.dgl_fused == "on" else torch.int32
        dgl_fused = args.dgl_fused == "on"
        g = dgl.graph((edge_index[0].contiguous(), edge_index[1].contiguous()), num_nodes=N, idtype=dgl_idt)
        g.ndata["feat"] = x
        g.ndata["label"] = y
        g.create_formats_()

        for th in threads_list:
            torch.set_num_threads(th)
            dgl_s = dgl.dataloading.NeighborSampler(fanouts, fused=dgl_fused)
            for s in sample_batches[:2]:
                _ = dgl_s.sample(g, s.to(dgl_idt))
            times_ms = []
            tot_n, tot_e = 0, 0
            for _rep in range(args.repeats):
                t0 = time.perf_counter()
                rep_n, rep_e = 0, 0
                for s in sample_batches:
                    inp, _, blks = dgl_s.sample(g, s.to(dgl_idt))
                    rep_n += int(inp.numel())
                    rep_e += sum(int(b.num_edges()) for b in blks)
                times_ms.append((time.perf_counter() - t0) / len(sample_batches) * 1e3)
                tot_n, tot_e = rep_n, rep_e
            records.append(
                {
                    "kind": f"pure_sample[{th}t]",
                    "ms_per_batch": statistics.median(times_ms),
                    "avg_src": tot_n // len(sample_batches),
                    "avg_edges": tot_e // len(sample_batches),
                }
            )

        torch.set_num_threads(args.loader_threads)
        for nw in workers_list:
            dgl_s = dgl.dataloading.NeighborSampler(
                fanouts, fused=dgl_fused, prefetch_node_feats=["feat"], prefetch_labels=["label"]
            )
            dgl_dl_kw: dict[str, Any] = dict(
                batch_size=args.batch_size,
                shuffle=False,
                drop_last=False,
                device=device,
                num_workers=nw,
                persistent_workers=nw > 0,
            )
            if device.type == "cuda":
                dgl_dl_kw["pin_prefetcher"] = args.pin_memory == "on"
            dl = dgl.dataloading.DataLoader(g, sub_idx.to(dgl_idt), dgl_s, **dgl_dl_kw)
            for _, _, blks in dl:
                _ = blks[0].srcdata["feat"].shape
            _sync_if_cuda(device)
            times_ms = []
            tot_n, tot_e = 0, 0
            for _rep in range(args.repeats):
                t0 = time.perf_counter()
                rep_n, rep_e = 0, 0
                for inp, _, blks in dl:
                    rep_n += int(inp.numel())
                    rep_e += sum(int(b.num_edges()) for b in blks)
                    if device.type == "cuda":
                        _ = blks[0].srcdata["feat"].sum()
                _sync_if_cuda(device)
                times_ms.append((time.perf_counter() - t0) / len(seeds_list) * 1e3)
                tot_n, tot_e = rep_n, rep_e
            del dl
            records.append(
                {
                    "kind": f"e2e_loader[nw={nw}]",
                    "ms_per_batch": statistics.median(times_ms),
                    "avg_src": tot_n // len(seeds_list),
                    "avg_edges": tot_e // len(seeds_list),
                }
            )
    else:
        indptr, indices = _in_csr(edge_index, N)
        for th in threads_list:
            s = CPUNeighborSampler(indptr, indices, fanouts, index_dtype=torch.int32, num_threads=th)
            for sd in sample_batches[:2]:
                _ = s.sample(sd)
            times_ms = []
            tot_n, tot_e = 0, 0
            for _rep in range(args.repeats):
                t0 = time.perf_counter()
                rep_n, rep_e = 0, 0
                for sd in sample_batches:
                    b = s.sample(sd)
                    rep_n += int(b.input_nodes.numel())
                    rep_e += sum(blk.num_edges for blk in b.blocks)
                times_ms.append((time.perf_counter() - t0) / len(sample_batches) * 1e3)
                tot_n, tot_e = rep_n, rep_e
            records.append(
                {
                    "kind": f"pure_sample[{th}t]",
                    "ms_per_batch": statistics.median(times_ms),
                    "avg_src": tot_n // len(sample_batches),
                    "avg_edges": tot_e // len(sample_batches),
                }
            )

        for nw in workers_list:
            maker = CPUBatchMaker(indptr, indices, x, y, fanouts, backend="cuda", index_dtype=torch.int32, seed=args.seed)
            w_threads = args.loader_threads if nw == 0 else None
            with CPUPrefetchLoader(
                maker,
                num_workers=nw,
                prefetch=True,
                depth=args.prefetch_depth,
                pin_memory=args.pin_memory == "on",
                device=device,
                persistent_workers=nw > 0,
                worker_threads=w_threads,
                seed=args.seed,
            ) as loader:
                for b, _ in loader.epoch(seeds_list):
                    _ = b["x"].shape
                _sync_if_cuda(device)
                times_ms = []
                tot_n, tot_e = 0, 0
                for _rep in range(args.repeats):
                    t0 = time.perf_counter()
                    rep_n, rep_e = 0, 0
                    for b, _ in loader.epoch(seeds_list):
                        rep_n += int(b["x"].shape[0])
                        rep_e += sum(st[2] for st in b["stats"])
                        if device.type == "cuda":
                            _ = b["x"].sum()
                    _sync_if_cuda(device)
                    times_ms.append((time.perf_counter() - t0) / len(seeds_list) * 1e3)
                    tot_n, tot_e = rep_n, rep_e
            records.append(
                {
                    "kind": f"e2e_loader[nw={nw}]",
                    "ms_per_batch": statistics.median(times_ms),
                    "avg_src": tot_n // len(seeds_list),
                    "avg_edges": tot_e // len(seeds_list),
                }
            )

    return {
        "dataset": name,
        "num_nodes": N,
        "num_edges": int(edge_index.shape[1]),
        "feat_dim": int(x.shape[1]),
        "side": args.side,
        "device": args.device,
        "records": records,
    }


def _invoke_subprocess(args: argparse.Namespace, dataset_yaml: str, side: str) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--_worker-run",
        "--side",
        side,
        "--dataset",
        dataset_yaml,
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--fanouts",
        args.fanouts,
        "--batch-size",
        str(args.batch_size),
        "--batches",
        str(args.batches),
        "--sample-batches",
        str(args.sample_batches),
        "--threads",
        args.threads,
        "--workers",
        args.workers,
        "--loader-threads",
        str(args.loader_threads),
        "--prefetch-depth",
        str(args.prefetch_depth),
        "--pin-memory",
        args.pin_memory,
        "--dgl-fused",
        args.dgl_fused,
        "--repeats",
        str(args.repeats),
        "--seed",
        str(args.seed),
    ]
    env = os.environ.copy()
    env.setdefault("TURBO_GNN_ALLOW_MISSING_EXTENSION", "1")
    if args.device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    res = subprocess.run(cmd, capture_output=True, text=True, env=env, check=False)
    if res.returncode != 0:
        raise RuntimeError(f"Subprocess failed ({side} on {dataset_yaml}):\n{res.stderr}\n{res.stdout}")
    for line in reversed(res.stdout.splitlines()):
        if line.startswith("__RESULT_JSON__"):
            return json.loads(line[len("__RESULT_JSON__") :])
    raise RuntimeError(f"No JSON result from {side} on {dataset_yaml}:\n{res.stdout}\n{res.stderr}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", default=",".join(DEFAULT_DATASETS), help="comma-separated dataset YAML paths")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="target batch device (cpu = zero GPU usage)")
    p.add_argument("--dtype", choices=list(DTYPES), default="bf16")
    p.add_argument("--fanouts", default="15,10,5")
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--batches", type=int, default=30, help="batches per DataLoader epoch")
    p.add_argument("--sample-batches", type=int, default=20, help="batches per pure sample() run")
    p.add_argument("--threads", default="1,8,16", help="thread counts for pure sample()")
    p.add_argument("--workers", default="0,4,8", help="worker process counts for E2E DataLoader")
    p.add_argument("--loader-threads", type=int, default=16, help="threads for nw=0 DataLoader")
    p.add_argument("--prefetch-depth", type=int, default=2)
    p.add_argument("--pin-memory", choices=["on", "off"], default="on", help="used when --device cuda")
    p.add_argument("--dgl-fused", choices=["on", "off"], default="on", help="DGL NeighborSampler(fused=...)")
    p.add_argument("--mode", choices=["both", "dgl", "ours"], default="both")
    p.add_argument("--repeats", type=int, default=2, help="timed repeats per configuration (median reported)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="", help="optional JSON output path")
    # Internal subprocess flags
    p.add_argument("--_worker-run", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--side", choices=["dgl", "ours"], default="ours", help=argparse.SUPPRESS)
    p.add_argument("--dataset", default="", help=argparse.SUPPRESS)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args._worker_run:
        res = _run_worker(args)
        print("__RESULT_JSON__" + json.dumps(res), flush=True)
        return

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    sides = ["dgl", "ours"] if args.mode == "both" else [args.mode]
    all_results: list[dict[str, Any]] = []

    print(
        f"CPU Sampler Benchmark (device={args.device}, fanouts=[{args.fanouts}], "
        f"batch_size={args.batch_size}, dtype={args.dtype}, dgl_fused={args.dgl_fused})",
        flush=True,
    )
    for ds_yaml in datasets:
        per_side: dict[str, dict[str, Any]] = {}
        for side in sides:
            per_side[side] = _invoke_subprocess(args, ds_yaml, side)
        ref = per_side[sides[0]]
        print(
            f"\n=== {ref['dataset']} (N={ref['num_nodes']:,}, E={ref['num_edges']:,}, F={ref['feat_dim']}) ===",
            flush=True,
        )
        if args.mode == "both":
            dgl_map = {r["kind"]: r for r in per_side["dgl"]["records"]}
            our_map = {r["kind"]: r for r in per_side["ours"]["records"]}
            print(f"  {'Benchmark':22s} | {'DGL (ms)':>10s} | {'Ours (ms)':>10s} | {'Speedup':>8s} | {'avg_src (DGL/Ours)':>20s}")
            print(f"  {'-'*22}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}-+-{'-'*20}")
            for r in per_side["ours"]["records"]:
                k = r["kind"]
                dm = dgl_map[k]["ms_per_batch"]
                om = r["ms_per_batch"]
                sp = dm / om if om > 0 else float("inf")
                print(
                    f"  {k:22s} | {dm:10.2f} | {om:10.2f} | {sp:7.2f}x | "
                    f"{dgl_map[k]['avg_src']:>9d}/{r['avg_src']:<9d}",
                    flush=True,
                )
        else:
            for r in ref["records"]:
                print(
                    f"  {r['kind']:22s} | {r['ms_per_batch']:10.2f} ms/batch | avg_src={r['avg_src']:,}",
                    flush=True,
                )
        all_results.append({"dataset_yaml": ds_yaml, "results": per_side})

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"args": vars(args), "benchmarks": all_results}, indent=2))
        print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
