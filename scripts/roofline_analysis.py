"""Are these kernels bandwidth-bound, and is per-edge compute slower than per-edge loading?

Two questions, one measurement pass.

**1. Saturation.** Hardware counters are unavailable on this machine (`ncu` reports
`ERR_NVGPUCTRPERM`), so actual DRAM traffic cannot be read. What *can* be done rigorously is to
model the **compulsory** traffic -- the bytes a kernel must move if nothing is reused -- and
divide by measured time to get an *apparent* bandwidth. Comparing that against the device's
empirically measured peak partitions every configuration into two regimes:

    apparent < peak   the kernel is not bandwidth-bound. Something else limits it: latency,
                      occupancy, or arithmetic. There is headroom a scheduler can address.
    apparent > peak   the kernel moved less than the compulsory model says, so the caches
                      supplied the difference. apparent/peak is then a *lower bound* on the
                      reuse factor, and the kernel is reuse-bound rather than bandwidth-bound.

The compulsory model is an upper bound on DRAM traffic and a lower bound on reuse, so neither
conclusion depends on knowing the true traffic. That is what makes the comparison safe without
counters.

**2. Compute versus loading, per edge.** For each edge the kernel must load the neighbour's
feature vector; at peak bandwidth that load takes `bytes_per_edge / peak` seconds. Measured
time per edge divided by that number answers the question directly: above 1.0, the kernel spends
longer on an edge than merely streaming its features would take, so message passing is *not*
free relative to the load it rides on.

Peak bandwidth is measured here rather than taken from the datasheet, because the achievable
figure on an A100 is roughly 80% of the 2.0 TB/s spec and using the spec would overstate
saturation by that margin.

    python scripts/roofline_analysis.py --out reports/roofline
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from benchmark_kernels import load_graph  # noqa: E402
from run_kernel_benchmark_matrix import GRAPHS  # noqa: E402

from turbo_gnn import gatv2_aggr, graph_transformer_aggr, reduction_aggr  # noqa: E402

F32 = 4  # every measurement here is fp32
IDX = 4  # int32 CSR indices


def measure_peak_bandwidth(device: torch.device, mib: int = 512, iters: int = 30) -> float:
    """Achievable HBM bandwidth in bytes/s, from a large read+write stream.

    A datasheet figure would flatter every ratio below. This is the number the kernels are
    actually competing against.
    """
    n = mib * 1024 * 1024 // F32
    a = torch.empty(n, device=device, dtype=torch.float32).uniform_()
    b = torch.empty_like(a)
    for _ in range(5):
        b.copy_(a)
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        b.copy_(a)
    end.record()
    torch.cuda.synchronize()
    seconds = start.elapsed_time(end) / 1e3 / iters
    return float((2 * n * F32) / seconds)  # one read + one write per element


def timed(fn, warmup: int = 10, iters: int = 30) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)) / iters / 1e3  # seconds


def traffic_model(conv: str, mode: str, n: int, e: int, d: int, h: int) -> tuple[int, int, int]:
    """Compulsory bytes, bytes attributable to one edge, and FLOPs, for one kernel launch.

    "Compulsory" means: every distinct value the kernel must read at least once and every value
    it writes, counted once, assuming no reuse across edges. Real traffic is at most this, and
    less wherever a neighbour's features are read by several nodes -- which is exactly the reuse
    the second regime above detects.

    Each term is written out rather than folded together so the model can be checked against
    the kernels.
    """
    w = h * d * F32  # one node's feature vector for all heads

    if conv == "min_aggr":
        if mode == "forward":
            # per edge: the column index, then the neighbour's features
            # per node: the two CSR row bounds, the output row, the argmin row (int32 per feature)
            per_edge = IDX + d * F32
            total = e * per_edge + n * (2 * IDX + d * F32 + d * IDX)
            flops = e * d  # one compare per feature per edge
        else:
            # The backward is a scatter over (node, feature) driven by argmin indices; it never
            # walks the edge list, which is why it does not scale with E.
            per_edge = 0
            total = n * d * (F32 + IDX + F32)  # grad_out, arg_idx, atomic write into grad_x
            flops = n * d
    elif conv == "gat_v2":
        if mode == "forward":
            per_edge = IDX + w  # column index + the neighbour's right-hand features
            total = e * per_edge + n * (2 * IDX + w + w + h * F32)  # + own features, output, logsumexp
            flops = e * (4 * h * d)  # add, leaky-relu, dot with the attention vector, weighted accumulate
        else:
            # Both CSR directions are walked: the AL kernel over the forward graph and the R
            # kernel over the transpose, each loading a neighbour row.
            per_edge = 2 * (IDX + w)
            total = e * per_edge + n * (4 * IDX + 3 * w + h * F32)
            flops = e * (8 * h * d)
    else:  # gt
        if mode == "forward":
            per_edge = IDX + 2 * w  # column index + the neighbour's K and V
            total = e * per_edge + n * (2 * IDX + w + w + h * F32)  # + own Q, output, logsumexp
            flops = e * (4 * h * d)  # QK dot, then the V accumulate
        else:
            per_edge = IDX + 3 * w  # K, V and the incoming gradient contribution per edge
            total = e * per_edge + n * (4 * IDX + 5 * w + h * F32)
            flops = e * (8 * h * d)
    return total, per_edge, flops


def build_call(conv: str, g, n: int, dim: int, heads: int, mode: str, params: dict):
    dev = torch.device("cuda")
    gen = torch.Generator(dev).manual_seed(0)
    need_grad = mode == "backward"

    def rnd(*shape):
        return torch.randn(*shape, device=dev, generator=gen, requires_grad=need_grad)

    if conv == "min_aggr":
        x = rnd(n, dim)
        fwd = lambda: reduction_aggr(g, x, **params)  # noqa: E731
        leaves = [x]
    elif conv == "gat_v2":
        xl, xr = rnd(n, heads, dim), rnd(n, heads, dim)
        a = torch.randn(heads, dim, device=dev, generator=gen)
        fwd = lambda: gatv2_aggr(g, xl, xr, a, 0.2, **params)  # noqa: E731
        leaves = [xl, xr]
    else:
        q, k, v = (rnd(n, heads, dim) for _ in range(3))
        fwd = lambda: graph_transformer_aggr(g, q, q, k, v, None, **params)  # noqa: E731
        leaves = [q, k, v]

    if mode == "forward":
        with torch.no_grad():
            return lambda: fwd()
    out = fwd()
    grad = torch.ones_like(out)
    return lambda: out.backward(grad, retain_graph=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="reports/roofline")
    p.add_argument("--graphs", nargs="+", default=[g for g, _ in GRAPHS])
    p.add_argument("--conv", nargs="+", default=["min_aggr", "gat_v2", "gt"])
    p.add_argument("--head-dims", type=int, nargs="+", default=[128, 256])
    p.add_argument("--modes", nargs="+", default=["forward", "backward"])
    p.add_argument("--heads", type=int, default=1)
    p.add_argument("--quantile", type=float, default=0.99)
    p.add_argument("--root", default="data")
    p.add_argument("--iters", type=int, default=30)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("needs CUDA")
    dev = torch.device("cuda")
    peak = measure_peak_bandwidth(dev)
    name = torch.cuda.get_device_properties(0).name
    print(f"{name}: measured peak {peak / 1e12:.3f} TB/s (read+write stream)\n", flush=True)

    by_graph = dict(GRAPHS)
    rows = []
    for graph in args.graphs:
        if graph not in by_graph:
            continue
        import argparse as _ap

        ns = _ap.Namespace(
            dataset=by_graph[graph],
            num_nodes=0,
            avg_degree=10,
            quantile=args.quantile,
            index_dtype="int32",
            self_loops=True,
            node_order="natural",
        )
        try:
            bg = load_graph(ns, dev, "cuda")
        except Exception as exc:
            print(f"  {graph}: SKIPPED ({type(exc).__name__})", file=sys.stderr)
            continue
        g = bg.repr
        n = bg.num_nodes
        e = bg.stats["num_edges"]
        for conv in args.conv:
            for dim in args.head_dims:
                for mode in args.modes:
                    try:
                        call = build_call(conv, g, n, dim, args.heads, mode, {})
                        secs = timed(call, iters=args.iters)
                    except Exception as exc:
                        print(f"  {graph}/{conv}/{dim}/{mode}: {type(exc).__name__}: {str(exc)[:70]}", file=sys.stderr)
                        continue
                    total, per_edge, flops = traffic_model(conv, mode, n, e, dim, args.heads)
                    apparent = total / secs
                    t_edge = secs / e if e else float("nan")
                    t_load_edge = per_edge / peak if per_edge else float("nan")
                    rows.append(
                        {
                            "graph": graph,
                            "conv": conv,
                            "head_dim": dim,
                            "mode": mode,
                            "nodes": n,
                            "edges": e,
                            "ms": secs * 1e3,
                            "model_bytes": total,
                            "bytes_per_edge": per_edge,
                            "flops": flops,
                            "apparent_GBs": apparent / 1e9,
                            "pct_of_peak": apparent / peak * 100,
                            "reuse_lower_bound": max(1.0, apparent / peak),
                            "ns_per_edge": t_edge * 1e9 if e else None,
                            "ns_per_edge_at_peak": t_load_edge * 1e9 if per_edge else None,
                            "compute_vs_load": (t_edge / t_load_edge) if per_edge else None,
                            "arithmetic_intensity": flops / total,
                            "achieved_GFLOPs": flops / secs / 1e9,
                        }
                    )
                    r = rows[-1]
                    cv = f"{r['compute_vs_load']:.2f}" if r["compute_vs_load"] else "  -"
                    print(
                        f"  {graph:<15}{conv:<9}{dim:>4} {mode:<9}{r['ms']:>9.3f} ms"
                        f"{r['apparent_GBs']:>9.0f} GB/s{r['pct_of_peak']:>8.0f}%  edge x{cv}",
                        flush=True,
                    )
        del bg, g
        torch.cuda.empty_cache()

    out = REPO / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "roofline.json").write_text(json.dumps({"peak_bytes_per_s": peak, "device": name, "cells": rows}, indent=1))
    print(f"\n{len(rows)} cells -> {(out / 'roofline.json').relative_to(REPO)}")
    if rows:
        sat = [r for r in rows if r["pct_of_peak"] >= 100]
        print(f"  apparent bandwidth at or above measured peak (cache-assisted): {len(sat)}/{len(rows)}")
        cb = [r for r in rows if r["compute_vs_load"] and r["compute_vs_load"] > 1.0]
        have = sum(1 for r in rows if r["compute_vs_load"])
        print(f"  per-edge time exceeding the per-edge load time: {len(cb)}/{have}")
        print(f"  median % of peak: {statistics.median(r['pct_of_peak'] for r in rows):.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
