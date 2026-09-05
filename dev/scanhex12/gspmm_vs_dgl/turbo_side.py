"""Time turbo_gnn.gspmm on the data exported by dgl_side.py.

    python turbo_side.py <dir> <d> [--check] [--ops copy_u]

Prints RESULT lines on stderr for plot.py. The exported edge operand is already
in CSR order, so no to_csr_edge_order gather is timed here.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys

import numpy as np
import torch

from turbo_gnn import gspmm
from turbo_gnn.graph import AdjacencyForwardBackwardWithNodeBuckets

ALL_OPS = ["copy_u", "copy_e", "add", "sub", "mul", "div"]
REDUCERS = ["sum", "min", "max"]


def load(path: str, dtype, count: int) -> torch.Tensor:
    arr = np.fromfile(path, dtype=dtype, count=count)
    if arr.size != count:
        sys.exit(f"{path}: expected {count} elements, got {arr.size}")
    return torch.from_numpy(arr)


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dir", help="export directory written by dgl_side.py")
    p.add_argument("d", type=int, help="feature width to run")
    p.add_argument("--ops", default=None, help="default: whatever meta.json recorded")
    p.add_argument("--check", action="store_true", help="verify against DGL's exported output")
    p.add_argument("--quantile", type=float, default=None, help="default: meta.json")
    p.add_argument("--warps", type=int, default=8)
    p.add_argument("--fpb", type=int, default=32)
    p.add_argument("--tiles-y", type=int, default=8)
    p.add_argument("--iters", type=int, default=None)
    p.add_argument("--repeats", type=int, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        sys.exit("no CUDA device")
    import json

    with open(os.path.join(args.dir, "meta.json")) as fh:
        meta = json.load(fh)
    n, e = meta["N"], meta["E"]
    d = args.d
    ops = (args.ops.split(",") if args.ops else meta["ops"])
    iters = args.iters or meta.get("iters", 30)
    repeats = args.repeats or meta.get("repeats", 7)
    quantile = args.quantile if args.quantile is not None else meta.get("quantile", 0.95)
    dev = "cuda"

    ptr = load(os.path.join(args.dir, "ptr.i32"), np.int32, n + 1).to(dev)
    idx = load(os.path.join(args.dir, "idx.i32"), np.int32, e).to(dev)

    graph = AdjacencyForwardBackwardWithNodeBuckets.from_csr(
        ptr, idx, ptr, idx, quantile=quantile, index_dtype=torch.int32, is_directed=False
    ).to(dev)

    got_heavy = graph.forward_heavy_nodes.numel()
    if meta.get("num_heavy") is not None and got_heavy != meta["num_heavy"]:
        print(f"warning: heavy bucket is {got_heavy}, dgl_side.py recorded {meta['num_heavy']}", file=sys.stderr)

    x = load(os.path.join(args.dir, f"x_{d}.f32"), np.float32, n * d).view(n, d).to(dev)
    needs_edge = any(o != "copy_u" for o in ops)
    e_full = e_bcast = None
    if needs_edge:
        e_full = load(os.path.join(args.dir, f"e_{d}.f32"), np.float32, e * d).view(e, d).to(dev)
        e_bcast = load(os.path.join(args.dir, f"eb_{d}.f32"), np.float32, e).to(dev)

    kw = dict(warps_per_block=args.warps, features_per_block=args.fpb, tiles_y=args.tiles_y)

    print(f"{'op':8} {'red':5} {'d':>5} {'turbo(ms)':>11}")
    print("-" * 32)
    failures = 0
    for op in ops:
        for reduce in REDUCERS:
            lhs = None if op == "copy_e" else x
            rhs = None if op == "copy_u" else e_full

            if args.check:
                ref_path = os.path.join(args.dir, f"ref_{op}_{reduce}_{d}.f32")
                if os.path.exists(ref_path):
                    want = load(ref_path, np.float32, n * d).view(n, d).to(dev)
                    with torch.no_grad():
                        got = gspmm(graph, lhs, rhs, op=op, reduce=reduce, **kw)
                    if not torch.allclose(got, want, rtol=2e-3, atol=2e-3):
                        err = (got - want).abs().max().item()
                        print(f"  MISMATCH {op}/{reduce}: max abs err {err:.3g}")
                        failures += 1

            t = time_ms(lambda: gspmm(graph, lhs, rhs, op=op, reduce=reduce, **kw), iters, repeats)
            print(f"{op:8} {reduce:5} {d:5} {t:11.3f}")
            print(f"RESULT\t{op}\t{reduce}\t{d}\t{t:.6f}", file=sys.stderr)

    if failures:
        print(f"\n{failures} MISMATCHES -- timings are not comparable", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
