"""turbo_gnn.gspmm vs dgl.ops over the 6 x 3 operator table.

    python -m tests.performance.gspmm_benchmarks
    python -m tests.performance.gspmm_benchmarks --ops mul --reducers sum --dtype float16
    python -m tests.performance.gspmm_benchmarks --graph ogbn-arxiv

Correctness is checked before timing; the CSR permutation turbo_gnn needs for
edge data is reported as its own column, since whether it belongs in the
comparison depends on whether the edge weights are static or model-produced.
"""

from __future__ import annotations

import argparse
import statistics
import sys

import torch

try:
    import dgl
except ImportError:  # pragma: no cover
    sys.exit("this benchmark needs dgl: pip install dgl (see Makefile install-bench)")

from turbo_gnn import gspmm
from turbo_gnn.graph import AdjacencyForwardBackwardWithNodeBuckets

OPS = ["copy_u", "copy_e", "add", "sub", "mul", "div"]
REDUCERS = ["sum", "min", "max"]


def dgl_op_name(op: str, reduce: str) -> str:
    """ "mul", "sum" -> "u_mul_e_sum"; copy_* keep their own naming."""
    if op in ("copy_u", "copy_e"):
        return f"{op}_{reduce}"
    return f"u_{op}_e_{reduce}"


def time_ms(fn, iters: int, repeats: int, warmup: int = 10) -> float:
    """Median over `repeats` of the mean per-call time of `iters` calls, in ms."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(repeats):
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / iters)
    return statistics.median(samples)


def random_edge_index(num_nodes: int, avg_degree: int, device: str, seed: int = 0) -> torch.Tensor:
    """Uniform random edges plus a self-loop on every node."""
    g = torch.Generator(device=device).manual_seed(seed)
    num_edges = num_nodes * avg_degree
    src = torch.randint(0, num_nodes, (num_edges,), device=device, generator=g)
    dst = torch.randint(0, num_nodes, (num_edges,), device=device, generator=g)
    loops = torch.arange(num_nodes, device=device)
    return torch.stack([torch.cat([src, loops]), torch.cat([dst, loops])])


def load_edge_index(name: str, device: str) -> tuple[torch.Tensor, int]:
    """A named OGB/PyG graph, self-loops added."""
    from ogb.nodeproppred import PygNodePropPredDataset

    data = PygNodePropPredDataset(name=name)[0]
    ei = data.edge_index.to(device)
    n = int(data.num_nodes)
    loops = torch.arange(n, device=device)
    ei = torch.stack([torch.cat([ei[0], loops]), torch.cat([ei[1], loops])])
    return ei, n


def make_operands(op: str, num_nodes: int, num_edges: int, d: int, dtype, device: str, seed: int = 0):
    """Node/edge operands in COO (DGL) order; None for the operand `op` ignores."""
    g = torch.Generator(device=device).manual_seed(seed)
    x = None
    if op != "copy_e":
        x = torch.randn(num_nodes, d, device=device, dtype=dtype, generator=g)
    e = None
    if op != "copy_u":
        e = torch.rand(num_edges, d, device=device, dtype=dtype, generator=g) + 0.5
    return x, e


def run_case(
    *,
    op: str,
    reduce: str,
    g_dgl,
    turbo,
    num_nodes: int,
    num_edges: int,
    d: int,
    dtype,
    device: str,
    args,
) -> dict:
    x, e = make_operands(op, num_nodes, num_edges, d, dtype, device)
    ref_fn = getattr(dgl.ops, dgl_op_name(op, reduce))
    kernel_kw = {
        "op": op,
        "reduce": reduce,
        "warps_per_block": args.warps,
        "features_per_block": args.fpb,
        "tiles_y": args.tiles_y,
    }
    if args.autotune:
        kernel_kw["autotune"] = True

    e_csr = turbo.to_csr_edge_order(e) if e is not None else None

    with torch.no_grad():
        want = ref_fn(g_dgl, *[a for a in (x, e) if a is not None])
        got = gspmm(turbo, x, e_csr, **kernel_kw)
        want = want.reshape(got.shape)
        tol = 1e-2 if dtype is not torch.float32 else 1e-4
        ok = torch.allclose(got, want, rtol=tol, atol=tol)

    row = {"op": op, "reduce": reduce, "ok": ok}
    if not ok:
        row["max_abs_err"] = (got.float() - want.float()).abs().max().item()
        return row

    row["dgl_fwd"] = time_ms(lambda: ref_fn(g_dgl, *[a for a in (x, e) if a is not None]), args.iters, args.repeats)
    row["turbo_fwd"] = time_ms(lambda: gspmm(turbo, x, e_csr, **kernel_kw), args.iters, args.repeats)

    row["perm"] = time_ms(lambda: turbo.to_csr_edge_order(e), args.iters, args.repeats) if e is not None else 0.0

    if args.backward:
        x_g = x.detach().clone().requires_grad_(True) if x is not None else None
        e_g = e.detach().clone().requires_grad_(True) if e is not None else None
        x_r = x.detach().clone().requires_grad_(True) if x is not None else None
        e_r = e.detach().clone().requires_grad_(True) if e is not None else None
        seed_grad = torch.randn(num_nodes, d, device=device, dtype=dtype)

        def turbo_step():
            for t in (x_g, e_g):
                if t is not None:
                    t.grad = None
            rhs = turbo.to_csr_edge_order(e_g) if e_g is not None else None
            gspmm(turbo, x_g, rhs, **kernel_kw).reshape(num_nodes, d).backward(seed_grad)

        def dgl_step():
            for t in (x_r, e_r):
                if t is not None:
                    t.grad = None
            ref_fn(g_dgl, *[a for a in (x_r, e_r) if a is not None]).reshape(num_nodes, d).backward(seed_grad)

        row["dgl_fb"] = time_ms(dgl_step, args.iters, args.repeats)
        row["turbo_fb"] = time_ms(turbo_step, args.iters, args.repeats)

    return row


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--graph", default="random", help="'random' or an OGB name, e.g. ogbn-arxiv")
    p.add_argument("--nodes", type=int, default=200_000)
    p.add_argument("--avg-degree", type=int, default=15)
    p.add_argument("--feat-dims", default="32,64,128", help="comma-separated feature widths")
    p.add_argument("--ops", default=",".join(OPS))
    p.add_argument("--reducers", default=",".join(REDUCERS))
    p.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--quantile", type=float, default=0.95, help="degree quantile for the light/heavy split")
    p.add_argument("--index-dtype", default="int32", choices=["int32", "int64"])
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--repeats", type=int, default=7)
    p.add_argument("--warps", type=int, default=8)
    p.add_argument("--fpb", type=int, default=32)
    p.add_argument("--tiles-y", type=int, default=8)
    p.add_argument("--backward", action="store_true", help="also time forward+backward")
    p.add_argument(
        "--autotune", action="store_true", help="measure the autotuned config instead of --warps/--fpb/--tiles-y"
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        sys.exit("no CUDA device")
    device = "cuda"
    dtype = getattr(torch, args.dtype)
    index_dtype = getattr(torch, args.index_dtype)

    if args.graph == "random":
        edge_index = random_edge_index(args.nodes, args.avg_degree, device)
        num_nodes = args.nodes
    else:
        edge_index, num_nodes = load_edge_index(args.graph, device)
    num_edges = edge_index.size(1)

    g_dgl = dgl.graph((edge_index[0], edge_index[1]), num_nodes=num_nodes)
    turbo = AdjacencyForwardBackwardWithNodeBuckets.from_edge_list(
        edge_index, num_nodes=num_nodes, quantile=args.quantile, index_dtype=index_dtype
    ).to(device)

    n_heavy = turbo.forward_heavy_nodes.numel()
    print(f"device      {torch.cuda.get_device_name(0)}")
    print(f"graph       {args.graph}: N={num_nodes} E={num_edges} avg_deg={num_edges / num_nodes:.1f}")
    print(f"buckets     light={turbo.forward_light_nodes.numel()} heavy={n_heavy} (quantile={args.quantile})")
    print(f"dtype       {args.dtype} / index {args.index_dtype}")
    print(f"launch      warps={args.warps} fpb={args.fpb} tiles_y={args.tiles_y} autotune={args.autotune}")
    print(f"timing      {args.iters} iters x {args.repeats} repeats, median\n")

    hdr = f"{'op':8} {'red':5} {'d':>5} {'dgl fwd':>10} {'turbo fwd':>10} {'x':>7} {'perm':>8} {'x w/perm':>9}"
    if args.backward:
        hdr += f" {'dgl f+b':>10} {'turbo f+b':>10} {'x':>7}"
    print(hdr)
    print("-" * len(hdr))

    speedups, speedups_perm = [], []
    for d in [int(s) for s in args.feat_dims.split(",")]:
        for op in args.ops.split(","):
            for reduce in args.reducers.split(","):
                row = run_case(
                    op=op,
                    reduce=reduce,
                    g_dgl=g_dgl,
                    turbo=turbo,
                    num_nodes=num_nodes,
                    num_edges=num_edges,
                    d=d,
                    dtype=dtype,
                    device=device,
                    args=args,
                )
                if not row["ok"]:
                    print(f"{op:8} {reduce:5} {d:5}   MISMATCH vs DGL (max abs err {row['max_abs_err']:.3g})")
                    continue

                sp = row["dgl_fwd"] / row["turbo_fwd"]
                sp_perm = row["dgl_fwd"] / (row["turbo_fwd"] + row["perm"])
                speedups.append(sp)
                speedups_perm.append(sp_perm)
                line = (
                    f"{op:8} {reduce:5} {d:5} {row['dgl_fwd']:10.3f} {row['turbo_fwd']:10.3f} "
                    f"{sp:6.2f}x {row['perm']:8.3f} {sp_perm:8.2f}x"
                )
                if args.backward:
                    line += f" {row['dgl_fb']:10.3f} {row['turbo_fb']:10.3f} {row['dgl_fb'] / row['turbo_fb']:6.2f}x"
                print(line)

    if speedups:
        print(
            f"\nforward speedup vs DGL: geomean {statistics.geometric_mean(speedups):.2f}x "
            f"(min {min(speedups):.2f}x, max {max(speedups):.2f}x)"
        )
        print(f"  including the CSR permutation: geomean {statistics.geometric_mean(speedups_perm):.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
