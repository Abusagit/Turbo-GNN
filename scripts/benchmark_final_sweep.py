import argparse
import statistics
import sys
import time
from pathlib import Path

import torch
import triton.testing

_orig_torch_load = torch.load
def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _torch_load_compat

sys.path.append(str(Path(__file__).resolve().parent.parent))

import turbo_gnn._C as _C  # noqa: E402
from turbo_gnn.graph import AdjacencyForwardBackwardWithNodeBuckets  # noqa: E402
from turbo_gnn.scheduling import edge_balanced_partition  # noqa: E402


REAL_SOURCE = {
    "ogbn-arxiv":    "ogbn",
    "ogbn-products": "ogbn",
    "web-traffic":   "pyg",
    "hm-categories": "pyg",
    "city-roads-L":  "pyg",
}


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


def load_real_edge_index(name, root):
    from src.data.datasets import DatasetConfig, load_single_graph
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


def _signed_indptr(indptr):
    if indptr.dtype == torch.uint32:
        return indptr.view(torch.int32)
    if indptr.dtype == torch.uint64:
        return indptr.view(torch.int64)
    return indptr


def graph_stats(graph):
    deg = _signed_indptr(graph.forward_indptr).diff().float()
    return {
        "N": int(deg.numel()),
        "mean": float(deg.mean()),
        "p99": float(torch.quantile(deg, 0.99)),
        "max": float(deg.max()),
        "light": int(graph.forward_light_nodes.numel()),
        "heavy": int(graph.forward_heavy_nodes.numel()),
    }


def bench(fn, warmup_ms, rep_ms, repeats=3):
    times = [float(triton.testing.do_bench(fn, warmup=warmup_ms, rep=rep_ms)) for _ in range(repeats)]
    med = statistics.median(times)
    return {
        "median": med,
        "min": min(times),
        "max": max(times),
        "spread_pct": 100.0 * (max(times) - min(times)) / med if med > 0 else 0.0,
    }


def _prepare(graph, schedule, grid_size):
    empty = torch.empty(0, dtype=torch.int32, device="cuda")
    if schedule == "legacy":
        return graph.forward_light_nodes, 0, empty, False
    if schedule == "gsl":
        return graph.forward_light_nodes, grid_size, empty, False
    if schedule in ("balanced", "balanced_atomic"):
        light, offsets = edge_balanced_partition(
            graph.forward_light_nodes, graph.forward_indptr, grid_size,
        )
        return light, 0, offsets, (schedule == "balanced_atomic")
    raise ValueError(schedule)


def bench_gatv2(graph, H, D, dtype, warmup_ms, rep_ms, repeats, schedule, grid_size):
    light, gs_override, block_offsets, use_dynamic = _prepare(graph, schedule, grid_size)
    N = int(_signed_indptr(graph.forward_indptr).numel()) - 1
    xl = torch.randn(N, H, D, device="cuda", dtype=dtype)
    xr = torch.randn(N, H, D, device="cuda", dtype=dtype)
    aw = torch.randn(H, D, device="cuda", dtype=dtype)

    def _fn():
        return _C.gatv2_forward(
            xl, xr,
            graph.forward_indptr, graph.forward_indices,
            aw, 0.2,
            light, graph.forward_heavy_nodes,
            1, 8, gs_override, block_offsets, use_dynamic,
        )
    return bench(_fn, warmup_ms, rep_ms, repeats)


def bench_minaggr(graph, D, dtype, warmup_ms, rep_ms, repeats, schedule, grid_size, reduce="min"):
    light, gs_override, block_offsets, use_dynamic = _prepare(graph, schedule, grid_size)
    N = int(_signed_indptr(graph.forward_indptr).numel()) - 1
    x = torch.randn(N, D, device="cuda", dtype=dtype)
    max_deg = int(_signed_indptr(graph.forward_indptr).diff().max().item())

    def _fn():
        return _C.reduction_aggr_forward_partitioned(
            graph.forward_indptr, graph.forward_indices, x,
            light, graph.forward_heavy_nodes, max_deg,
            8, 128, False, 32, 8, reduce,
            gs_override, block_offsets, use_dynamic,
        )
    return bench(_fn, warmup_ms, rep_ms, repeats)


def collect_specs(args):
    specs = []
    if "synth-powerlaw" in args.datasets:
        for N in args.sizes:
            ei = make_powerlaw(N, args.avg_degree, seed=42, exponent=args.exponent)
            specs.append({
                "tag": f"synth-powerlaw-N{N}-exp{args.exponent}",
                "edge_index": ei,
                "num_nodes": N,
            })
    for name in args.datasets:
        if name == "synth-powerlaw":
            continue
        try:
            ei, n = load_real_edge_index(name, args.data_root)
            specs.append({"tag": name, "edge_index": ei, "num_nodes": n})
        except Exception as e:
            print(f"skip {name}: {e}")
    return specs


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["synth-powerlaw"])
    p.add_argument("--data-root", default="data")
    p.add_argument("--sizes", type=int, nargs="+", default=[65536, 262144])
    p.add_argument("--avg-degree", type=int, default=8)
    p.add_argument("--exponent", type=float, default=2.3)
    p.add_argument("--quantile", type=float, default=0.99)
    p.add_argument("--dims", type=int, nargs="+", default=[64, 128, 256])
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--sm-multipliers", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--ops", nargs="+", default=["gatv2", "minaggr"], choices=["gatv2", "minaggr"])
    p.add_argument("--warmup-ms", type=float, default=200.0)
    p.add_argument("--rep-ms", type=float, default=1000.0)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--out", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        print("CUDA required.")
        return 1

    torch.set_default_device("cuda")
    props = torch.cuda.get_device_properties(0)
    sm = props.multi_processor_count
    grid_sizes = [m * sm for m in args.sm_multipliers]

    print(f"GPU: {props.name} (SMs={sm})  grid_sizes={grid_sizes}  dims={args.dims}  q={args.quantile}")
    specs = collect_specs(args)
    if not specs:
        return 1

    rows = []
    stats = {}
    started = time.time()

    for spec in specs:
        graph = build_bucketed(spec["edge_index"], spec["num_nodes"], quantile=args.quantile)
        s = graph_stats(graph)
        stats[spec["tag"]] = s
        print(f"[{spec['tag']}] N={s['N']} mean={s['mean']:.1f} p99={s['p99']:.0f} "
              f"max={s['max']:.0f} light={s['light']} heavy={s['heavy']}")

        for op in args.ops:
            bench_fn = bench_gatv2 if op == "gatv2" else bench_minaggr
            for dim in args.dims:
                if op == "gatv2":
                    kw = dict(H=args.heads, D=dim, dtype=torch.float32,
                              warmup_ms=args.warmup_ms, rep_ms=args.rep_ms, repeats=args.repeats)
                else:
                    kw = dict(D=dim, dtype=torch.float32,
                              warmup_ms=args.warmup_ms, rep_ms=args.rep_ms, repeats=args.repeats)
                r = bench_fn(graph, schedule="legacy", grid_size=0, **kw)
                rows.append((spec["tag"], op, dim, "legacy", -1, r))
                for gs in grid_sizes:
                    for sched in ("gsl", "balanced", "balanced_atomic"):
                        r = bench_fn(graph, schedule=sched, grid_size=gs, **kw)
                        rows.append((spec["tag"], op, dim, sched, gs, r))
                elapsed = time.time() - started
                print(f"  [{op} dim={dim}] done (elapsed {elapsed/60:.1f}m)")

    write_report(args, props, sm, grid_sizes, stats, rows, time.time() - started)
    print(f"[written] {args.out}")
    return 0


def write_report(args, props, sm, grid_sizes, stats, rows, wall_s):
    md = []
    md.append("# Scheduling sweep\n")
    md.append(f"GPU: **{props.name}** ({sm} SM). quantile = **{args.quantile}**, "
              f"GATv2 heads = **{args.heads}**, fp32.  ")
    md.append(f"warmup = {args.warmup_ms} ms, rep = {args.rep_ms} ms, repeats = {args.repeats}.\n")
    md.append(f"Total: {wall_s/60:.1f} min.\n")

    md.append("\n## Graph stats\n")
    md.append("| dataset | N | mean_deg | p99 | max | light | heavy |")
    md.append("|:---|---:|---:|---:|---:|---:|---:|")
    for tag, s in stats.items():
        md.append(f"| {tag} | {s['N']} | {s['mean']:.1f} | {s['p99']:.0f} | "
                  f"{s['max']:.0f} | {s['light']} | {s['heavy']} |")

    md.append("\n## Best schedule per (dataset, op, dim)\n")
    md.append("| dataset | op | dim | legacy ms | best sched | grid | best ms | speedup |")
    md.append("|:---|:---|---:|---:|:---|---:|---:|---:|")
    for tag in stats:
        for op in args.ops:
            for dim in args.dims:
                sub = [r for r in rows if r[0] == tag and r[1] == op and r[2] == dim]
                if not sub:
                    continue
                legacy = next(r for r in sub if r[3] == "legacy")
                non_leg = [r for r in sub if r[3] != "legacy"]
                best = min(non_leg, key=lambda r: r[5]["median"])
                speed = legacy[5]["median"] / best[5]["median"]
                md.append(f"| {tag} | {op} | {dim} | {legacy[5]['median']:.3f} | "
                          f"{best[3]} | {best[4]} | {best[5]['median']:.3f} | **{speed:.2f}x** |")

    md.append("\n## Full detail\n")
    for tag in stats:
        for op in args.ops:
            for dim in args.dims:
                sub = [r for r in rows if r[0] == tag and r[1] == op and r[2] == dim]
                if not sub:
                    continue
                legacy_ms = next(r[5]["median"] for r in sub if r[3] == "legacy")
                md.append(f"\n**{tag} / {op} / dim={dim}** — legacy = **{legacy_ms:.3f} ms**\n")
                md.append("| grid_x (x SM) | gsl ms (speedup) | balanced ms (speedup) | balanced_atomic ms (speedup) |")
                md.append("|:---|---:|---:|---:|")
                for gs in grid_sizes:
                    mult = gs // sm
                    line = [f"{gs} ({mult}x)"]
                    for sched in ("gsl", "balanced", "balanced_atomic"):
                        r = next((r for r in sub if r[3] == sched and r[4] == gs), None)
                        if r is None:
                            line.append("—")
                            continue
                        ms = r[5]["median"]
                        line.append(f"{ms:.3f} ({legacy_ms / ms:.2f}x)")
                    md.append("| " + " | ".join(line) + " |")

    md.append("\n## Measurement stability (spread > 3%)\n")
    md.append("| dataset | op | dim | schedule | grid_x | median (ms) | spread |")
    md.append("|:---|:---|---:|:---|---:|---:|---:|")
    for tag, op, dim, sched, gx, r in rows:
        if r["spread_pct"] > 3.0:
            md.append(f"| {tag} | {op} | {dim} | {sched} | "
                      f"{gx if gx > 0 else '-'} | {r['median']:.3f} | {r['spread_pct']:.1f}% |")

    Path(args.out).write_text("\n".join(md))


if __name__ == "__main__":
    raise SystemExit(main())
