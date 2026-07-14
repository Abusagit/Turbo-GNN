import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))

import turbo_gnn._C as _C  # noqa: E402
from src.benchmarking.microbench import get_gpu_info, time_callable  # noqa: E402
from turbo_gnn.graph import AdjacencyForwardBackwardWithNodeBuckets  # noqa: E402
from turbo_gnn.scheduling import sort_by_degree_desc  # noqa: E402


def _finalize(src, dst, N, device):
    src_all = torch.cat([src, dst, torch.arange(N, device=device)])
    dst_all = torch.cat([dst, src, torch.arange(N, device=device)])
    flat = src_all.long() * N + dst_all.long()
    flat_unique = torch.unique(flat)
    return torch.stack([flat_unique // N, flat_unique % N])


def make_graph_er(N, avg_degree, seed, device="cuda"):
    gen = torch.Generator(device=device).manual_seed(seed)
    E = N * avg_degree
    src = torch.randint(0, N, (E,), device=device, generator=gen)
    dst = torch.randint(0, N, (E,), device=device, generator=gen)
    return _finalize(src, dst, N, device)


def make_graph_powerlaw(N, avg_degree, seed, exponent=2.5, device="cuda"):
    gen = torch.Generator(device=device).manual_seed(seed)
    E = N * avg_degree
    ranks = torch.arange(1, N + 1, device=device, dtype=torch.float)
    weights = ranks.pow(-1.0 / (exponent - 1.0))
    weights = weights[torch.randperm(N, device=device, generator=gen)]
    probs = weights / weights.sum()
    src = torch.multinomial(probs, E, replacement=True, generator=gen)
    dst = torch.multinomial(probs, E, replacement=True, generator=gen)
    return _finalize(src, dst, N, device)


def make_graph(N, avg_degree, seed, graph_type="powerlaw", exponent=2.5, device="cuda"):
    if graph_type == "er":
        return make_graph_er(N, avg_degree, seed, device)
    if graph_type == "powerlaw":
        return make_graph_powerlaw(N, avg_degree, seed, exponent, device)
    raise ValueError(f"unknown graph_type={graph_type}")


def build_bucketed(edge_index, N, quantile):
    return AdjacencyForwardBackwardWithNodeBuckets.from_edge_list(
        edge_index, N, quantile=quantile, index_dtype=torch.int32
    )


def degree_stats(graph):
    indptr = graph.forward_indptr
    if indptr.dtype == torch.uint32:
        indptr = indptr.view(torch.int32)
    elif indptr.dtype == torch.uint64:
        indptr = indptr.view(torch.int64)
    deg = (indptr[1:] - indptr[:-1]).float()
    return {
        "mean": deg.mean().item(),
        "p50": deg.median().item(),
        "p99": torch.quantile(deg, 0.99).item(),
        "max": deg.max().item(),
        "light_nodes": int(graph.forward_light_nodes.numel()),
        "heavy_nodes": int(graph.forward_heavy_nodes.numel()),
    }


def bench_one(N, avg_degree, H, D, dtype, grid_size_override, warmup, iters,
              seed=42, schedule="none", graph_type="powerlaw", exponent=2.5,
              quantile=-1.0):
    edge_index = make_graph(N, avg_degree, seed, graph_type, exponent)
    graph = build_bucketed(edge_index, N, quantile=quantile)
    if schedule == "sort":
        graph.forward_light_nodes = sort_by_degree_desc(
            graph.forward_light_nodes, graph.forward_indptr,
        )
        if graph.forward_heavy_nodes.numel() > 0:
            graph.forward_heavy_nodes = sort_by_degree_desc(
                graph.forward_heavy_nodes, graph.forward_indptr,
            )
    xl = torch.randn(N, H, D, device="cuda", dtype=dtype)
    xr = torch.randn(N, H, D, device="cuda", dtype=dtype)
    aw = torch.randn(H, D, device="cuda", dtype=dtype)

    def _fn():
        return _C.gatv2_forward(
            xl,
            xr,
            graph.forward_indptr,
            graph.forward_indices,
            aw,
            0.2,
            graph.forward_light_nodes,
            graph.forward_heavy_nodes,
            1,
            8,
            grid_size_override,
        )

    return time_callable(_fn, warmup=warmup, iters=iters, do_memory_profile=False)


def parse_args():
    p = argparse.ArgumentParser(description="GATv2 forward: legacy vs grid-strided vs sorted.")
    p.add_argument("--sizes", type=int, nargs="+", default=[8192, 65536, 262144])
    p.add_argument("--avg-degree", type=int, default=8)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--dtype", choices=["fp32", "fp16"], default="fp32")
    p.add_argument("--graph", choices=["er", "powerlaw"], default="powerlaw")
    p.add_argument("--exponent", type=float, default=2.5)
    p.add_argument("--quantile", type=float, default=-1.0)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
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

    configs = [
        (0, "none", "legacy"),
        (sm_count, "none", f"sm×1 ({sm_count})"),
        (sm_count * 2, "none", f"sm×2 ({sm_count * 2})"),
        (sm_count * 2, "sort", f"sm×2+sort"),
        (sm_count * 4, "sort", f"sm×4+sort"),
    ]

    rows = []
    stats_by_N = {}
    for N in args.sizes:
        ref = build_bucketed(
            make_graph(N, args.avg_degree, 42, args.graph, args.exponent),
            N, quantile=args.quantile,
        )
        stats_by_N[N] = degree_stats(ref)
        del ref
        for gs, sched, label in configs:
            res = bench_one(
                N=N,
                avg_degree=args.avg_degree,
                H=args.heads,
                D=args.head_dim,
                dtype=dtype,
                grid_size_override=gs,
                warmup=args.warmup,
                iters=args.iters,
                schedule=sched,
                graph_type=args.graph,
                exponent=args.exponent,
                quantile=args.quantile,
            )
            rows.append({
                "N": N,
                "avg_degree": args.avg_degree,
                "heads": args.heads,
                "head_dim": args.head_dim,
                "dtype": args.dtype,
                "graph": args.graph,
                "exponent": args.exponent,
                "quantile": args.quantile,
                "grid_size_override": gs,
                "schedule": sched,
                "label": label,
                "ms_per_iter": res.ms_per_iter,
            })

    print()
    print(f"GPU: {gpu_info.get('device_name', '?')}  (SMs={sm_count})")
    print(f"graph={args.graph} exponent={args.exponent} quantile={args.quantile} "
          f"heads={args.heads} head_dim={args.head_dim} dtype={args.dtype} "
          f"avg_degree={args.avg_degree}")
    print()
    print("degree distribution (per N):")
    for N in args.sizes:
        s = stats_by_N[N]
        print(f"  N={N:>8}: mean={s['mean']:.1f}  p50={s['p50']:.0f}  "
              f"p99={s['p99']:.0f}  max={s['max']:.0f}  "
              f"light={s['light_nodes']}  heavy={s['heavy_nodes']}")
    print()

    labels = [c[2] for c in configs]
    header = f"{'N':>10} | " + " | ".join(f"{lab:>18}" for lab in labels) + " | " + f"{'best/legacy':>12}"
    print(header)
    print("-" * len(header))

    by_N = {}
    for r in rows:
        by_N.setdefault(r["N"], {})[r["label"]] = r["ms_per_iter"]
    for N, m in by_N.items():
        baseline = m["legacy"]
        cells = " | ".join(f"{m[lab]:>18.4f}" for lab in labels)
        best = min(m[lab] for lab in labels if lab != "legacy")
        speedup = baseline / best
        print(f"{N:>10} | {cells} | {speedup:>11.2f}x")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(
            {"gpu_info": gpu_info, "degree_stats": stats_by_N, "rows": rows}, indent=2
        ))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
