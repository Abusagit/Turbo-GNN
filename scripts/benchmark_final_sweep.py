"""Final scheduling sweep: legacy vs balanced vs balanced_atomic.

Axes:
  * op:      gatv2, minaggr forward
  * dim:     64, 128, 256  (GATv2 head_dim, MinAggr feature dim)
  * dataset: synth-powerlaw / real graphs via load_single_graph
  * schedule: legacy | balanced | balanced_atomic
  * grid_size (for balanced): {1, 2, 4, 8} x SM_count

Methodology:
  warmup=200ms, rep=1000ms, 3 repeats (median reported, min/max shown).
  Same pre-built graph across all schedules within one (dataset, N, quantile) block.

Output: single markdown log file (path via --out).
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import triton.testing

# torch.load(weights_only=False) compat for legacy pickled dataset caches
_orig_torch_load = torch.load
def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _torch_load_compat

sys.path.append(str(Path(__file__).resolve().parent.parent))

import turbo_gnn._C as _C  # noqa: E402
from turbo_gnn.graph import AdjacencyForwardBackwardWithNodeBuckets  # noqa: E402
from turbo_gnn.scheduling import edge_balanced_partition  # noqa: E402


SCHEDULES = ["legacy", "balanced", "balanced_atomic"]

# Sources for real graphs (load_single_graph)
REAL_SOURCE = {
    "ogbn-arxiv":    "ogbn",
    "ogbn-products": "ogbn",
    "web-traffic":   "pyg",
    "hm-categories": "pyg",
    "city-roads-L":  "pyg",
}


# ---------- Graph builders ----------

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


def load_real_edge_index(name: str, root: str):
    # deferred import to keep synth-only mode fast
    from src.data.datasets import DatasetConfig, load_single_graph
    if name not in REAL_SOURCE:
        raise ValueError(f"unknown real dataset {name!r}")
    cfg = DatasetConfig(source=REAL_SOURCE[name], name=name, root=root, conv_backend="cuda")
    graph = load_single_graph(cfg)
    ei = graph.edge_index
    if not isinstance(ei, torch.Tensor):
        ei = torch.as_tensor(ei)
    return ei.to("cuda"), int(graph.num_nodes)


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


# ---------- Bench ----------

def bench(fn, warmup_ms: float, rep_ms: float, repeats: int = 3):
    times = []
    for _ in range(repeats):
        t = triton.testing.do_bench(fn, warmup=warmup_ms, rep=rep_ms)
        times.append(float(t))
    med = statistics.median(times)
    return {
        "median": med,
        "min": min(times),
        "max": max(times),
        "spread_pct": 100.0 * (max(times) - min(times)) / med if med > 0 else 0.0,
    }


def bench_gatv2(graph, H, D, dtype, warmup_ms, rep_ms, repeats, schedule, grid_size):
    empty_offsets = torch.empty(0, dtype=torch.int32, device="cuda")
    light_nodes = graph.forward_light_nodes
    gs_override = 0
    block_offsets = empty_offsets
    use_dynamic = False
    if schedule == "legacy":
        pass
    elif schedule in ("balanced", "balanced_atomic"):
        light_nodes, block_offsets = edge_balanced_partition(
            graph.forward_light_nodes, graph.forward_indptr, grid_size,
        )
        use_dynamic = (schedule == "balanced_atomic")
    else:
        raise ValueError(schedule)
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


def bench_minaggr(graph, D, dtype, warmup_ms, rep_ms, repeats, schedule, grid_size, reduce="min"):
    empty_offsets = torch.empty(0, dtype=torch.int32, device="cuda")
    light_nodes = graph.forward_light_nodes
    gs_override = 0
    block_offsets = empty_offsets
    use_dynamic = False
    if schedule == "legacy":
        pass
    elif schedule in ("balanced", "balanced_atomic"):
        light_nodes, block_offsets = edge_balanced_partition(
            graph.forward_light_nodes, graph.forward_indptr, grid_size,
        )
        use_dynamic = (schedule == "balanced_atomic")
    else:
        raise ValueError(schedule)
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


# ---------- Sweep ----------

@dataclass
class Row:
    dataset: str
    N: int
    quantile: float
    op: str
    dim: int
    schedule: str
    grid_x: int
    ms_median: float
    ms_min: float
    ms_max: float
    spread_pct: float


@dataclass
class DatasetSpec:
    tag: str          # short label for report
    edge_index: torch.Tensor
    num_nodes: int
    kind: str = "synth"  # "synth" or "real"
    stats_note: str = ""


def collect_datasets(args) -> list[DatasetSpec]:
    specs: list[DatasetSpec] = []
    if "synth-powerlaw" in args.datasets:
        for N in args.sizes:
            ei = make_powerlaw(N, args.avg_degree, seed=42, exponent=args.exponent)
            specs.append(DatasetSpec(
                tag=f"synth-powerlaw-N{N}-exp{args.exponent}",
                edge_index=ei, num_nodes=N, kind="synth",
                stats_note=f"powerlaw exp={args.exponent} avg={args.avg_degree}",
            ))
    for ds_name in args.datasets:
        if ds_name == "synth-powerlaw":
            continue
        try:
            ei, n = load_real_edge_index(ds_name, args.data_root)
            specs.append(DatasetSpec(
                tag=ds_name, edge_index=ei, num_nodes=n, kind="real",
                stats_note=f"real dataset from {args.data_root}",
            ))
        except Exception as e:
            print(f"[warn] failed to load {ds_name!r}: {e}")
    return specs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["synth-powerlaw"],
                    help="synth-powerlaw and/or names of real datasets")
    ap.add_argument("--data-root", default="/home/sinfillo/Turbo-GNN/data")
    ap.add_argument("--sizes", type=int, nargs="+", default=[65536, 262144],
                    help="applies to synth-powerlaw only")
    ap.add_argument("--avg-degree", type=int, default=8)
    ap.add_argument("--exponent", type=float, default=2.3)
    ap.add_argument("--quantile", type=float, default=0.99,
                    help="single quantile (production setting)")
    ap.add_argument("--dims", type=int, nargs="+", default=[64, 128, 256])
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--sm-multipliers", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--ops", nargs="+", default=["gatv2", "minaggr"], choices=["gatv2", "minaggr"])
    ap.add_argument("--warmup-ms", type=float, default=200.0)
    ap.add_argument("--rep-ms", type=float, default=1000.0)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", type=str, required=True, help="markdown log path (outside repo)")
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
    print(f"grid_sizes = {grid_sizes} ({args.sm_multipliers} x SM)")
    print(f"dims       = {args.dims}")
    print(f"quantile   = {args.quantile}")
    print(f"datasets   = {args.datasets}")

    specs = collect_datasets(args)
    if not specs:
        print("no datasets to bench, exit.")
        return 1

    rows: list[Row] = []
    all_stats: dict[str, dict] = {}
    started_at = time.time()

    for spec in specs:
        graph = build_bucketed(spec.edge_index, spec.num_nodes, quantile=args.quantile)
        stats = graph_stats(graph)
        stats["stats_note"] = spec.stats_note
        all_stats[spec.tag] = stats
        print(f"\n[graph] {spec.tag}: N={stats['N']} mean={stats['mean']:.1f} "
              f"p99={stats['p99']:.0f} max={stats['max']:.0f} "
              f"light={stats['light']} heavy={stats['heavy']}")

        for op in args.ops:
            for dim in args.dims:
                bench_fn = bench_gatv2 if op == "gatv2" else bench_minaggr
                if op == "gatv2":
                    call_kwargs = dict(H=args.heads, D=dim, dtype=torch.float32,
                                       warmup_ms=args.warmup_ms, rep_ms=args.rep_ms,
                                       repeats=args.repeats)
                else:
                    call_kwargs = dict(D=dim, dtype=torch.float32,
                                       warmup_ms=args.warmup_ms, rep_ms=args.rep_ms,
                                       repeats=args.repeats)
                # legacy first
                r = bench_fn(graph, schedule="legacy", grid_size=0, **call_kwargs)
                rows.append(Row(spec.tag, spec.num_nodes, args.quantile, op, dim,
                                "legacy", -1, r["median"], r["min"], r["max"], r["spread_pct"]))
                for gs in grid_sizes:
                    for sched in ("balanced", "balanced_atomic"):
                        r = bench_fn(graph, schedule=sched, grid_size=gs, **call_kwargs)
                        rows.append(Row(spec.tag, spec.num_nodes, args.quantile, op, dim,
                                        sched, gs, r["median"], r["min"], r["max"], r["spread_pct"]))
                elapsed = time.time() - started_at
                print(f"  [{op} dim={dim}] done  (elapsed {elapsed/60:.1f}m)")

    # ---------- Report ----------
    md: list[str] = []
    md.append("# Final scheduling sweep — report\n")
    md.append(f"GPU: **{props.name}** ({sm_count} SM). ")
    md.append(f"quantile = **{args.quantile}**, GATv2 heads = **{args.heads}**, fp32.  ")
    md.append(f"warmup = {args.warmup_ms} ms, rep = {args.rep_ms} ms, repeats = {args.repeats} "
              f"(**median** reported). Schedules: `legacy`, `balanced`, `balanced_atomic`.\n")
    md.append(f"Total wallclock: {(time.time() - started_at)/60:.1f} minutes.\n")

    md.append("\n## Graph stats\n")
    md.append("| dataset | N | mean_deg | p99 | max | light | heavy |")
    md.append("|:---|---:|---:|---:|---:|---:|---:|")
    for tag, s in all_stats.items():
        md.append(f"| {tag} | {s['N']} | {s['mean']:.1f} | {s['p99']:.0f} | "
                  f"{s['max']:.0f} | {s['light']} | {s['heavy']} |")

    # Best-schedule summary
    md.append("\n## Best schedule per (dataset, op, dim)\n")
    md.append("Best speedup (`ms_legacy / ms_best`) over all grid multipliers `{1,2,4,8} x SM`.\n")
    md.append("| dataset | op | dim | legacy ms | best sched | grid | best ms | speedup |")
    md.append("|:---|:---|---:|---:|:---|---:|---:|---:|")
    for tag in all_stats.keys():
        for op in args.ops:
            for dim in args.dims:
                sub = [r for r in rows if r.dataset == tag and r.op == op and r.dim == dim]
                if not sub:
                    continue
                legacy = next(r for r in sub if r.schedule == "legacy")
                non_leg = [r for r in sub if r.schedule != "legacy"]
                best = min(non_leg, key=lambda r: r.ms_median)
                speed = legacy.ms_median / best.ms_median
                md.append(f"| {tag} | {op} | {dim} | {legacy.ms_median:.3f} | "
                          f"{best.schedule} | {best.grid_x} | {best.ms_median:.3f} | "
                          f"**{speed:.2f}x** |")

    md.append("\n## Full detail per (dataset, op, dim)\n")
    for tag in all_stats.keys():
        for op in args.ops:
            for dim in args.dims:
                sub = [r for r in rows if r.dataset == tag and r.op == op and r.dim == dim]
                if not sub:
                    continue
                legacy_ms = next(r.ms_median for r in sub if r.schedule == "legacy")
                md.append(f"\n**{tag} / {op} / dim={dim}** — legacy = **{legacy_ms:.3f} ms**\n")
                md.append("| grid_x (x SM) | balanced ms (speedup) | balanced_atomic ms (speedup) |")
                md.append("|:---|---:|---:|")
                for gs in grid_sizes:
                    mult = gs // sm_count
                    line = [f"{gs} ({mult}x)"]
                    for sched in ("balanced", "balanced_atomic"):
                        r = next((r for r in sub if r.schedule == sched and r.grid_x == gs), None)
                        if r is None:
                            line.append("—")
                            continue
                        line.append(f"{r.ms_median:.3f} ({legacy_ms / r.ms_median:.2f}x)")
                    md.append("| " + " | ".join(line) + " |")

    md.append("\n## Measurement stability (spread max-min / median, per row)\n")
    md.append("Rows with spread > 3% are potentially noisy.\n")
    md.append("| dataset | op | dim | schedule | grid_x | median (ms) | spread |")
    md.append("|:---|:---|---:|:---|---:|---:|---:|")
    for r in rows:
        if r.spread_pct > 3.0:
            md.append(f"| {r.dataset} | {r.op} | {r.dim} | {r.schedule} | "
                      f"{r.grid_x if r.grid_x > 0 else '-'} | {r.ms_median:.3f} | {r.spread_pct:.1f}% |")

    md_txt = "\n".join(md)
    Path(args.out).write_text(md_txt)
    print(f"\n[written] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
