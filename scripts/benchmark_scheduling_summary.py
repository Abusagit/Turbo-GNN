"""Controlled combined benchmark for scheduling strategies.

Runs GATv2 and MinAggr forward on synthetic power-law graphs at multiple
quantiles and multiple grid multipliers. All timings for one (op, N, quantile)
setting are collected back-to-back in a single Python session on the same
pre-built graph, with long warmup/rep. Reports the median of 3 repeats plus
min/max to expose noise.

Output is a markdown-friendly summary table for easy sharing.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import triton.testing

sys.path.append(str(Path(__file__).resolve().parent.parent))

import turbo_gnn._C as _C  # noqa: E402
from turbo_gnn.graph import AdjacencyForwardBackwardWithNodeBuckets  # noqa: E402
from turbo_gnn.scheduling import edge_balanced_partition  # noqa: E402


SCHEDULES = ["legacy", "gsl", "balanced", "balanced_atomic"]


def _finalize(src, dst, N, device):
    src_all = torch.cat([src, dst, torch.arange(N, device=device)])
    dst_all = torch.cat([dst, src, torch.arange(N, device=device)])
    flat = src_all.long() * N + dst_all.long()
    flat = torch.unique(flat)
    return torch.stack([flat // N, flat % N])


def make_powerlaw(N, avg_degree, seed=42, exponent=2.3, device="cuda"):
    gen = torch.Generator(device=device).manual_seed(seed)
    E = N * avg_degree
    ranks = torch.arange(1, N + 1, device=device, dtype=torch.float)
    weights = ranks.pow(-1.0 / (exponent - 1.0))
    weights = weights[torch.randperm(N, device=device, generator=gen)]
    probs = weights / weights.sum()
    src = torch.multinomial(probs, E, replacement=True, generator=gen)
    dst = torch.multinomial(probs, E, replacement=True, generator=gen)
    return _finalize(src, dst, N, device)


def build_bucketed(edge_index, N, quantile):
    return AdjacencyForwardBackwardWithNodeBuckets.from_edge_list(
        edge_index, N, quantile=quantile, index_dtype=torch.int32,
    )


def signed_indptr(indptr):
    if indptr.dtype == torch.uint32:
        return indptr.view(torch.int32)
    if indptr.dtype == torch.uint64:
        return indptr.view(torch.int64)
    return indptr


def graph_stats(graph):
    deg = signed_indptr(graph.forward_indptr).diff().float()
    return {
        "N": int(deg.numel()),
        "mean": float(deg.mean()),
        "p99": float(torch.quantile(deg, 0.99)),
        "max": float(deg.max()),
        "light": int(graph.forward_light_nodes.numel()),
        "heavy": int(graph.forward_heavy_nodes.numel()),
    }


def bench(fn, warmup_ms: float, rep_ms: float, repeats: int = 3):
    times = []
    for _ in range(repeats):
        t = triton.testing.do_bench(fn, warmup=warmup_ms, rep=rep_ms)
        times.append(float(t))
    return {
        "median": statistics.median(times),
        "min": min(times),
        "max": max(times),
        "spread_pct": 100.0 * (max(times) - min(times)) / statistics.median(times),
    }


@dataclass
class Row:
    op: str
    N: int
    quantile: float
    schedule: str
    grid_x: int          # -1 = legacy
    ms_median: float
    ms_min: float
    ms_max: float
    spread_pct: float


def bench_gatv2(graph, H, D, dtype, warmup_ms, rep_ms, repeats,
                schedule, grid_size):
    empty_offsets = torch.empty(0, dtype=torch.int32, device="cuda")
    light_nodes = graph.forward_light_nodes
    gs_override = 0
    block_offsets = empty_offsets
    use_dynamic = False
    if schedule == "legacy":
        pass
    elif schedule == "gsl":
        gs_override = grid_size
    elif schedule in ("balanced", "balanced_atomic"):
        light_nodes, block_offsets = edge_balanced_partition(
            graph.forward_light_nodes, graph.forward_indptr, grid_size,
        )
        use_dynamic = (schedule == "balanced_atomic")
    N = int(signed_indptr(graph.forward_indptr).numel()) - 1
    xl = torch.randn(N, H, D, device="cuda", dtype=dtype)
    xr = torch.randn(N, H, D, device="cuda", dtype=dtype)
    aw = torch.randn(H, D, device="cuda", dtype=dtype)

    def _fn():
        return _C.gatv2_forward(
            xl, xr,
            graph.forward_indptr, graph.forward_indices,
            aw, 0.2,
            light_nodes, graph.forward_heavy_nodes,
            1, 8, gs_override, block_offsets, use_dynamic,
        )
    return bench(_fn, warmup_ms, rep_ms, repeats)


def bench_minaggr(graph, D, dtype, warmup_ms, rep_ms, repeats,
                  schedule, grid_size, reduce="min"):
    empty_offsets = torch.empty(0, dtype=torch.int32, device="cuda")
    light_nodes = graph.forward_light_nodes
    gs_override = 0
    block_offsets = empty_offsets
    use_dynamic = False
    if schedule == "legacy":
        pass
    elif schedule == "gsl":
        gs_override = grid_size
    elif schedule in ("balanced", "balanced_atomic"):
        light_nodes, block_offsets = edge_balanced_partition(
            graph.forward_light_nodes, graph.forward_indptr, grid_size,
        )
        use_dynamic = (schedule == "balanced_atomic")
    N = int(signed_indptr(graph.forward_indptr).numel()) - 1
    x = torch.randn(N, D, device="cuda", dtype=dtype)
    max_deg = int(signed_indptr(graph.forward_indptr).diff().max().item())

    def _fn():
        return _C.reduction_aggr_forward_partitioned(
            graph.forward_indptr, graph.forward_indices, x,
            light_nodes, graph.forward_heavy_nodes, max_deg,
            8, 128, False, 32, 8, reduce,
            gs_override, block_offsets, use_dynamic,
        )
    return bench(_fn, warmup_ms, rep_ms, repeats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[65536, 262144])
    ap.add_argument("--avg-degree", type=int, default=8)
    ap.add_argument("--exponent", type=float, default=2.3)
    ap.add_argument("--quantiles", type=float, nargs="+", default=[0.9, 0.99, 0.999, -1.0])
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--minaggr-dim", type=int, default=64)
    ap.add_argument("--sm-multipliers", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--ops", nargs="+", default=["gatv2", "minaggr"], choices=["gatv2", "minaggr"])
    ap.add_argument("--warmup-ms", type=float, default=200.0)
    ap.add_argument("--rep-ms", type=float, default=1000.0)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required.")
        return 1

    torch.set_default_device("cuda")
    props = torch.cuda.get_device_properties(0)
    sm_count = props.multi_processor_count
    grid_sizes = [m * sm_count for m in args.sm_multipliers]

    print(f"GPU: {props.name}  (SMs={sm_count})")
    print(f"warmup={args.warmup_ms}ms  rep={args.rep_ms}ms  repeats={args.repeats}")
    print(f"grid_sizes = {grid_sizes} (multipliers {args.sm_multipliers} x SM)")
    print()

    rows: list[Row] = []
    all_stats = {}

    for N in args.sizes:
        ei = make_powerlaw(N, args.avg_degree, seed=42, exponent=args.exponent)
        for q in args.quantiles:
            graph = build_bucketed(ei, N, quantile=q)
            stats = graph_stats(graph)
            all_stats[(N, q)] = stats
            print(f"[graph] N={N} q={q}  N={stats['N']} mean={stats['mean']:.1f} p99={stats['p99']:.0f} "
                  f"max={stats['max']:.0f}  light={stats['light']} heavy={stats['heavy']}")

            for op in args.ops:
                bench_fn = bench_gatv2 if op == "gatv2" else bench_minaggr
                if op == "gatv2":
                    call_kwargs = dict(H=args.heads, D=args.head_dim, dtype=torch.float32,
                                       warmup_ms=args.warmup_ms, rep_ms=args.rep_ms, repeats=args.repeats)
                else:
                    call_kwargs = dict(D=args.minaggr_dim, dtype=torch.float32,
                                       warmup_ms=args.warmup_ms, rep_ms=args.rep_ms, repeats=args.repeats)

                # legacy first
                r = bench_fn(graph, schedule="legacy", grid_size=0, **call_kwargs)
                rows.append(Row(op, N, q, "legacy", -1, r["median"], r["min"], r["max"], r["spread_pct"]))

                for gs in grid_sizes:
                    for sched in ("gsl", "balanced", "balanced_atomic"):
                        r = bench_fn(graph, schedule=sched, grid_size=gs, **call_kwargs)
                        rows.append(Row(op, N, q, sched, gs, r["median"], r["min"], r["max"], r["spread_pct"]))
                print(f"  [{op}] done N={N} q={q}")
        del ei
    print()

    # Summary table 1: raw numbers per (op, N, quantile, gs, schedule)
    md = []
    md.append("# Scheduling benchmark summary\n")
    md.append(f"GPU: **{props.name}** ({sm_count} SM). ")
    md.append(f"powerlaw exp=**{args.exponent}**, avg_deg=**{args.avg_degree}**, ")
    md.append(f"GATv2: H={args.heads} D={args.head_dim}, MinAggr: D={args.minaggr_dim}, fp32. ")
    md.append(f"warmup={args.warmup_ms}ms, rep={args.rep_ms}ms, repeats={args.repeats} (median reported).\n")

    md.append("\n## Graph stats\n")
    md.append("| N | quantile | mean_deg | p99 | max | light | heavy |")
    md.append("|---:|---:|---:|---:|---:|---:|---:|")
    for (N, q), s in all_stats.items():
        md.append(f"| {N} | {q} | {s['mean']:.1f} | {s['p99']:.0f} | {s['max']:.0f} | {s['light']} | {s['heavy']} |")

    for op in args.ops:
        md.append(f"\n## {op} forward — median ms/iter (speedup vs legacy)\n")
        for N in args.sizes:
            for q in args.quantiles:
                key = (op, N, q)
                sub = [r for r in rows if (r.op, r.N, r.quantile) == key]
                if not sub:
                    continue
                legacy_ms = next(r.ms_median for r in sub if r.schedule == "legacy")
                md.append(f"\n**N = {N}, quantile = {q}** — legacy = **{legacy_ms:.3f} ms**\n")
                md.append("| grid_x (x SM) | gsl | balanced | balanced_atomic |")
                md.append("|:---|---:|---:|---:|")
                for gs in grid_sizes:
                    mult = gs // sm_count
                    line = [f"{gs} ({mult}x)"]
                    for sched in ("gsl", "balanced", "balanced_atomic"):
                        r = next((r for r in sub if r.schedule == sched and r.grid_x == gs), None)
                        if r is None:
                            line.append("-")
                            continue
                        speed = legacy_ms / r.ms_median
                        line.append(f"{r.ms_median:.3f} ms ({speed:.2f}x)")
                    md.append("| " + " | ".join(line) + " |")

        md.append(f"\n### {op} — measurement stability (spread max/min-1 across 3 repeats)\n")
        md.append("| N | q | schedule | grid_x | median (ms) | min (ms) | max (ms) | spread |")
        md.append("|---:|---:|:---|---:|---:|---:|---:|---:|")
        for N in args.sizes:
            for q in args.quantiles:
                for r in [r for r in rows if r.op == op and r.N == N and r.quantile == q]:
                    md.append(f"| {N} | {q} | {r.schedule} | {r.grid_x if r.grid_x > 0 else '-'} | "
                              f"{r.ms_median:.3f} | {r.ms_min:.3f} | {r.ms_max:.3f} | {r.spread_pct:.1f}% |")

    md_txt = "\n".join(md)
    if args.out:
        Path(args.out).write_text(md_txt)
        print(f"[written] {args.out}")
    print(md_txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
