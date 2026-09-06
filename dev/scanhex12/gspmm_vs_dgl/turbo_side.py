"""Time turbo_gnn.gspmm on the data exported by dgl_side.py.

    python turbo_side.py <dir> <d> [--check] [--backward] [--pipeline-stages 1]

Prints RESULT lines on stderr for plot.py. The exported edge operand is already
in CSR order, so no to_csr_edge_order gather is timed here.

The graph is rebuilt with the real transposed CSR, not an alias of the forward
one: the node gradient of reduce="sum" is this same g-SpMM walked on the
transpose, so aliasing would silently produce a wrong gradient.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

import numpy as np
import torch

from turbo_gnn import gspmm
from turbo_gnn.graph import AdjacencyForwardBackwardWithNodeBuckets

ALL_OPS = ["copy_u", "copy_e", "add", "sub", "mul", "div"]
REDUCERS = ["sum", "min", "max"]
EXT = {"float32": ("f32", np.float32), "float16": ("f16", np.float16)}


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
    p.add_argument("--dtype", default=None, choices=["float32", "float16"], help="default: meta.json")
    p.add_argument("--backward", action="store_true", help="also time forward+backward")
    p.add_argument("--check", action="store_true", help="verify against DGL's exported output and gradients")
    p.add_argument("--quantile", type=float, default=None, help="default: meta.json")
    p.add_argument("--warps", type=int, default=8)
    p.add_argument("--fpb", type=int, default=32)
    p.add_argument("--tiles-y", type=int, default=8)
    p.add_argument("--pipeline-stages", type=int, default=0)
    p.add_argument("--iters", type=int, default=None, help="default: meta.json")
    p.add_argument("--repeats", type=int, default=None, help="default: meta.json")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        sys.exit("no CUDA device")
    dev = "cuda"

    with open(os.path.join(args.dir, "meta.json")) as fh:
        meta = json.load(fh)
    n, e = meta["N"], meta["E"]
    d = args.d
    ops = args.ops.split(",") if args.ops else meta["ops"]
    iters = args.iters or meta.get("iters", 30)
    repeats = args.repeats or meta.get("repeats", 7)
    quantile = args.quantile if args.quantile is not None else meta.get("quantile", 0.95)
    dtype_name = args.dtype or meta.get("dtype", "float32")
    ext, np_dtype = EXT[dtype_name]
    tol = 1e-2 if dtype_name != "float32" else 2e-3
    # fp16 cannot represent a sum over a 13k-degree node to a fixed 1e-2: the
    # output quantum at |value| ~ 256 is already 0.25, so both libraries land
    # half an ulp apart. Scale atol by the reference magnitude for fp16 only;
    # fp32 stays strict, which is what caught the CSR-ordering bug above.
    def tolerances(want):
        if dtype_name == "float32":
            return tol, tol
        return tol, tol * max(1.0, want.abs().max().item())

    if args.backward and not meta.get("backward"):
        sys.exit(f"{args.dir} was exported without --backward: no reference gradients to check against")

    ptr = load(os.path.join(args.dir, "ptr.i32"), np.int32, n + 1).to(dev)
    idx = load(os.path.join(args.dir, "idx.i32"), np.int32, e).to(dev)
    bwd_path = os.path.join(args.dir, "bwd_ptr.i32")
    if os.path.exists(bwd_path):
        bwd_ptr = load(bwd_path, np.int32, n + 1).to(dev)
        bwd_idx = load(os.path.join(args.dir, "bwd_idx.i32"), np.int32, e).to(dev)
        directed = True
    elif args.backward:
        sys.exit(f"{args.dir} has no bwd_ptr.i32; re-export with the current dgl_side.py")
    else:
        bwd_ptr, bwd_idx, directed = ptr, idx, False

    graph = AdjacencyForwardBackwardWithNodeBuckets.from_csr(
        ptr, idx, bwd_ptr, bwd_idx, quantile=quantile, index_dtype=torch.int32, is_directed=directed
    ).to(dev)

    got_heavy = graph.forward_heavy_nodes.numel()
    if meta.get("num_heavy") is not None and got_heavy != meta["num_heavy"]:
        print(f"warning: heavy bucket is {got_heavy}, dgl_side.py recorded {meta['num_heavy']}", file=sys.stderr)

    x_base = load(os.path.join(args.dir, f"x_{d}.{ext}"), np_dtype, n * d).view(n, d).to(dev)
    needs_edge = any(o != "copy_u" for o in ops)
    e_base = None
    if needs_edge:
        e_base = load(os.path.join(args.dir, f"e_{d}.{ext}"), np_dtype, e * d).view(e, d).to(dev)
    gseed = None
    if args.backward:
        gseed = load(os.path.join(args.dir, f"gseed_{d}.{ext}"), np_dtype, n * d).view(n, d).to(dev)

    kw = dict(warps_per_block=args.warps, features_per_block=args.fpb, tiles_y=args.tiles_y,
              pipeline_stages=args.pipeline_stages)

    # references are stored as values at these fixed sampled flat indices
    si_node = si_edge = None
    if os.path.exists(os.path.join(args.dir, f"samp_node_{d}.i64")):
        si_node = load(os.path.join(args.dir, f"samp_node_{d}.i64"), np.int64, min(65536, n * d)).to(dev)
        si_edge = load(os.path.join(args.dir, f"samp_edge_{d}.i64"), np.int64, min(65536, e * d)).to(dev)

    def ref(prefix, op, reduce, count):
        path = os.path.join(args.dir, f"{prefix}_{op}_{reduce}_{d}.{ext}")
        if not os.path.exists(path) or si_node is None:
            return None
        return load(path, np_dtype, count).to(dev)

    print(f"{'kind':5} {'op':8} {'red':5} {'d':>5} {'turbo(ms)':>11}")
    print("-" * 40)
    failures = 0
    for op in ops:
        for reduce in REDUCERS:
            lhs = None if op == "copy_e" else x_base
            rhs = None if op == "copy_u" else e_base

            if args.check:
                want = ref("ref", op, reduce, si_node.numel() if si_node is not None else 0)
                if want is not None:
                    with torch.no_grad():
                        got = gspmm(graph, lhs, rhs, op=op, reduce=reduce, **kw).reshape(-1)[si_node]
                    rt, at = tolerances(want.float())
                    if not torch.allclose(got.float(), want.float(), rtol=rt, atol=at):
                        print(f"  MISMATCH out {op}/{reduce}: max abs err "
                              f"{(got.float() - want.float()).abs().max().item():.3g}")
                        failures += 1

            t = time_ms(lambda: gspmm(graph, lhs, rhs, op=op, reduce=reduce, **kw), iters, repeats)
            print(f"{'fwd':5} {op:8} {reduce:5} {d:5} {t:11.3f}")
            print(f"RESULT\tfwd\t{op}\t{reduce}\t{d}\t{t:.6f}\t{args.pipeline_stages}", file=sys.stderr)

            if not args.backward:
                continue

            xg = x_base.detach().clone().requires_grad_(True) if op != "copy_e" else None
            eg = e_base.detach().clone().requires_grad_(True) if op != "copy_u" else None
            leaves = [t_ for t_ in (xg, eg) if t_ is not None]

            def step():
                for t_ in leaves:
                    t_.grad = None
                gspmm(graph, xg, eg, op=op, reduce=reduce, **kw).reshape(n, d).backward(gseed)

            if args.check:
                step()
                for name, tensor, prefix, si in (("grad_lhs", xg, "refgx", si_node),
                                                 ("grad_rhs", eg, "refge", si_edge)):
                    if tensor is None or si is None:
                        continue
                    want = ref(prefix, op, reduce, si.numel())
                    if want is None:
                        continue
                    got_g = tensor.grad.reshape(-1)[si]
                    rt, at = tolerances(want.float())
                    if not torch.allclose(got_g.float(), want.float(), rtol=rt, atol=10 * at):
                        print(f"  MISMATCH {name} {op}/{reduce}: max abs err "
                              f"{(got_g.float() - want.float()).abs().max().item():.3g}")
                        failures += 1

            t = time_ms(step, iters, repeats)
            print(f"{'fb':5} {op:8} {reduce:5} {d:5} {t:11.3f}")
            print(f"RESULT\tfb\t{op}\t{reduce}\t{d}\t{t:.6f}\t{args.pipeline_stages}", file=sys.stderr)

            # backward alone: one forward, then replay the backward with
            # retain_graph so only the backward kernels are timed
            kept = gspmm(graph, xg, eg, op=op, reduce=reduce, **kw).reshape(n, d)

            def step_bwd():
                for t_ in leaves:
                    t_.grad = None
                kept.backward(gseed, retain_graph=True)

            t = time_ms(step_bwd, iters, repeats)
            del kept
            print(f"{'bwd':5} {op:8} {reduce:5} {d:5} {t:11.3f}")
            print(f"RESULT\tbwd\t{op}\t{reduce}\t{d}\t{t:.6f}\t{args.pipeline_stages}", file=sys.stderr)

            del xg, eg, leaves, step, step_bwd
            torch.cuda.empty_cache()

    if failures:
        print(f"\n{failures} MISMATCHES -- timings are not comparable", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
