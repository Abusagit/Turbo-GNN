"""Benchmark GATv2 / MinAggr forward with 4 scheduling strategies on real graphs.

Datasets: ogbn-arxiv, web-traffic, hm-categories.

Strategies compared (light-path only):
  legacy           — original: gridSize = num_nodes, blockIdx.x = node
  gsl              — grid-strided loop with gridSize = SM_count * C
  balanced         — edge-balanced partition (block_offsets)
  balanced_atomic  — balanced + runtime bid via atomicAdd (heaviest-first)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

# PyTorch 2.6+ defaults torch.load to weights_only=True, which breaks legacy
# pickled dataset caches. Restore the old default for trusted local caches.
_orig_torch_load = torch.load
def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _torch_load_compat

sys.path.append(str(Path(__file__).resolve().parent.parent))

import turbo_gnn._C as _C  # noqa: E402
from src.benchmarking.microbench import get_gpu_info, time_callable  # noqa: E402
from src.data.datasets import DatasetConfig, load_single_graph  # noqa: E402
from turbo_gnn.graph import AdjacencyForwardBackwardWithNodeBuckets  # noqa: E402
from turbo_gnn.scheduling import edge_balanced_partition  # noqa: E402


DATASET_SOURCES = {
    "ogbn-arxiv":   "ogbn",
    "web-traffic":  "pyg",
    "hm-categories": "pyg",
}


def load_real_edge_index(name: str, root: str = "data"):
    if name not in DATASET_SOURCES:
        raise ValueError(f"unknown dataset {name!r}; supported: {list(DATASET_SOURCES)}")
    cfg = DatasetConfig(source=DATASET_SOURCES[name], name=name, root=root, conv_backend="cuda")
    graph = load_single_graph(cfg)
    ei = graph.edge_index
    if not isinstance(ei, torch.Tensor):
        ei = torch.as_tensor(ei)
    return ei.to("cuda"), int(graph.num_nodes)


def build_bucketed(edge_index, N, quantile):
    return AdjacencyForwardBackwardWithNodeBuckets.from_edge_list(
        edge_index, N, quantile=quantile, index_dtype=torch.int32,
    )


def max_degree_of(indptr):
    signed = indptr
    if indptr.dtype == torch.uint32:
        signed = indptr.view(torch.int32)
    elif indptr.dtype == torch.uint64:
        signed = indptr.view(torch.int64)
    return int((signed[1:] - signed[:-1]).max().item())


def degree_summary(indptr):
    signed = indptr
    if indptr.dtype == torch.uint32:
        signed = indptr.view(torch.int32)
    elif indptr.dtype == torch.uint64:
        signed = indptr.view(torch.int64)
    deg = (signed[1:] - signed[:-1]).float()
    return {
        "N": int(signed.numel()) - 1,
        "mean": deg.mean().item(),
        "p50": deg.median().item(),
        "p99": torch.quantile(deg, 0.99).item(),
        "max": deg.max().item(),
    }


def resolve_grid_sizes(multipliers, sm_count):
    return [max(1, int(round(m * sm_count))) for m in multipliers]


def _empty_offsets():
    return torch.empty(0, dtype=torch.int32, device="cuda")


def bench_gatv2(graph, H, D, dtype, schedule, grid_size, num_blocks, warmup, iters):
    N = int(graph.forward_indptr.numel()) - 1
    xl = torch.randn(N, H, D, device="cuda", dtype=dtype)
    xr = torch.randn(N, H, D, device="cuda", dtype=dtype)
    aw = torch.randn(H, D, device="cuda", dtype=dtype)

    block_offsets = _empty_offsets()
    light_nodes = graph.forward_light_nodes
    gs_override = 0
    use_dynamic = False

    if schedule == "legacy":
        pass
    elif schedule == "gsl":
        gs_override = grid_size
    elif schedule in ("balanced", "balanced_atomic"):
        light_nodes, block_offsets = edge_balanced_partition(
            graph.forward_light_nodes, graph.forward_indptr, num_blocks,
        )
        use_dynamic = (schedule == "balanced_atomic")
    else:
        raise ValueError(schedule)

    def _fn():
        return _C.gatv2_forward(
            xl, xr,
            graph.forward_indptr, graph.forward_indices,
            aw, 0.2,
            light_nodes, graph.forward_heavy_nodes,
            1, 8,
            gs_override,
            block_offsets,
            use_dynamic,
        )

    return time_callable(_fn, warmup=warmup, iters=iters, do_memory_profile=False)


def bench_minaggr(graph, D, dtype, schedule, grid_size, num_blocks, warmup, iters, reduce="min"):
    N = int(graph.forward_indptr.numel()) - 1
    x = torch.randn(N, D, device="cuda", dtype=dtype)
    max_deg = max_degree_of(graph.forward_indptr)

    block_offsets = _empty_offsets()
    light_nodes = graph.forward_light_nodes
    gs_override = 0
    use_dynamic = False

    if schedule == "legacy":
        pass
    elif schedule == "gsl":
        gs_override = grid_size
    elif schedule in ("balanced", "balanced_atomic"):
        light_nodes, block_offsets = edge_balanced_partition(
            graph.forward_light_nodes, graph.forward_indptr, num_blocks,
        )
        use_dynamic = (schedule == "balanced_atomic")
    else:
        raise ValueError(schedule)

    def _fn():
        return _C.reduction_aggr_forward_partitioned(
            graph.forward_indptr,
            graph.forward_indices,
            x,
            light_nodes,
            graph.forward_heavy_nodes,
            max_deg,
            8, 128, False, 32, 8,
            reduce,
            gs_override,
            block_offsets,
            use_dynamic,
        )

    return time_callable(_fn, warmup=warmup, iters=iters, do_memory_profile=False)


def parse_args():
    p = argparse.ArgumentParser(description="Real-graph scheduling benchmark.")
    p.add_argument("--datasets", nargs="+", default=list(DATASET_SOURCES),
                   choices=list(DATASET_SOURCES))
    p.add_argument("--op", choices=["gatv2", "minaggr", "both"], default="both")
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--minaggr-dim", type=int, default=64)
    p.add_argument("--dtype", choices=["fp32", "fp16"], default="fp32")
    p.add_argument("--quantile", type=float, default=-1.0,
                   help="light/heavy split quantile; -1 means all-light.")
    p.add_argument("--sm-multipliers", type=float, nargs="+", default=[1, 2, 4, 8, 16])
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--data-root", type=str, default="data")
    p.add_argument("--json-out", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        print("CUDA required.")
        return 1

    device = torch.device("cuda", 0)
    torch.set_default_device(device)
    dtype = torch.float32 if args.dtype == "fp32" else torch.float16

    gpu_info = get_gpu_info(device)
    sm_count = gpu_info["sm_count"]
    grid_sizes = resolve_grid_sizes(args.sm_multipliers, sm_count)

    ops = ["gatv2", "minaggr"] if args.op == "both" else [args.op]
    rows = []
    stats_by_ds = {}

    for ds_name in args.datasets:
        print(f"\n=== Dataset: {ds_name} ===")
        try:
            edge_index, N = load_real_edge_index(ds_name, args.data_root)
        except Exception as e:
            print(f"  skip: could not load {ds_name}: {e}")
            continue

        graph = build_bucketed(edge_index, N, quantile=args.quantile)
        stats = degree_summary(graph.forward_indptr)
        stats["light_nodes"] = int(graph.forward_light_nodes.numel())
        stats["heavy_nodes"] = int(graph.forward_heavy_nodes.numel())
        stats_by_ds[ds_name] = stats
        print(f"  N={stats['N']} mean_deg={stats['mean']:.1f} p99={stats['p99']:.0f} "
              f"max={stats['max']:.0f} light={stats['light_nodes']} heavy={stats['heavy_nodes']}")

        for op in ops:
            if op == "gatv2":
                bench_fn = lambda sched, gs, nb: bench_gatv2(  # noqa: E731
                    graph, args.heads, args.head_dim, dtype, sched, gs, nb,
                    args.warmup, args.iters,
                )
            else:
                bench_fn = lambda sched, gs, nb: bench_minaggr(  # noqa: E731
                    graph, args.minaggr_dim, dtype, sched, gs, nb,
                    args.warmup, args.iters,
                )

            def _record(sched, gs, nb):
                res = bench_fn(sched, gs, nb)
                rows.append({
                    "dataset": ds_name, "op": op, "schedule": sched,
                    "grid_size": gs, "num_blocks": nb,
                    "ms_per_iter": res.ms_per_iter,
                })

            # legacy — one run
            _record("legacy", 0, 0)
            # gsl / balanced / balanced_atomic across grid_sizes / num_blocks
            for gs in grid_sizes:
                _record("gsl", gs, 0)
                _record("balanced", 0, gs)
                _record("balanced_atomic", 0, gs)

    print()
    print(f"GPU: {gpu_info.get('device_name', '?')} (SMs={sm_count})")
    print(f"dtype={args.dtype} heads={args.heads} head_dim={args.head_dim} "
          f"minaggr_dim={args.minaggr_dim} quantile={args.quantile}")
    print()

    header = (f"{'dataset':>13} | {'op':>7} | {'sched':>16} | {'grid':>6} | "
              f"{'ms/iter':>10} | {'vs_legacy':>9}")
    print(header)
    print("-" * len(header))

    key = lambda r: (r["dataset"], r["op"])  # noqa: E731
    grouped = {}
    for r in rows:
        grouped.setdefault(key(r), []).append(r)

    for (ds, op), items in grouped.items():
        legacy_ms = next(r["ms_per_iter"] for r in items if r["schedule"] == "legacy")
        for r in items:
            gs_show = r["grid_size"] if r["schedule"] == "gsl" else r["num_blocks"]
            gs_s = "-" if r["schedule"] == "legacy" else str(gs_show)
            speed = legacy_ms / r["ms_per_iter"]
            print(f"{ds:>13} | {op:>7} | {r['schedule']:>16} | {gs_s:>6} | "
                  f"{r['ms_per_iter']:>10.4f} | {speed:>8.2f}x")
        print("-" * len(header))

    if args.json_out:
        Path(args.json_out).write_text(json.dumps({
            "gpu_info": gpu_info,
            "degree_stats": stats_by_ds,
            "rows": rows,
            "config": vars(args),
        }, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
