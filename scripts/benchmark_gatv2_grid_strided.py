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


def make_graph(N, avg_degree, seed, device="cuda"):
    gen = torch.Generator(device=device).manual_seed(seed)
    E = N * avg_degree
    src = torch.randint(0, N, (E,), device=device, generator=gen)
    dst = torch.randint(0, N, (E,), device=device, generator=gen)
    src_all = torch.cat([src, dst, torch.arange(N, device=device)])
    dst_all = torch.cat([dst, src, torch.arange(N, device=device)])
    edge_index = torch.stack([src_all, dst_all])
    flat = edge_index[0] * N + edge_index[1]
    flat_unique = torch.unique(flat)
    return torch.stack([flat_unique // N, flat_unique % N])


def build_bucketed(edge_index, N, quantile):
    return AdjacencyForwardBackwardWithNodeBuckets.from_edge_list(
        edge_index, N, quantile=quantile, index_dtype=torch.int32
    )


def bench_one(N, avg_degree, H, D, dtype, grid_size_override, warmup, iters,
              seed=42, schedule="none"):
    edge_index = make_graph(N, avg_degree, seed)
    graph = build_bucketed(edge_index, N, quantile=-1)
    if schedule == "sort":
        graph.forward_light_nodes = sort_by_degree_desc(
            graph.forward_light_nodes, graph.forward_indptr,
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
    for N in args.sizes:
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
            )
            rows.append({
                "N": N,
                "avg_degree": args.avg_degree,
                "heads": args.heads,
                "head_dim": args.head_dim,
                "dtype": args.dtype,
                "grid_size_override": gs,
                "schedule": sched,
                "label": label,
                "ms_per_iter": res.ms_per_iter,
            })

    print()
    print(f"GPU: {gpu_info.get('device_name', '?')}  (SMs={sm_count})")
    print(f"heads={args.heads} head_dim={args.head_dim} dtype={args.dtype} avg_degree={args.avg_degree}")
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
            {"gpu_info": gpu_info, "rows": rows}, indent=2
        ))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
