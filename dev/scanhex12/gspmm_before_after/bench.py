"""turbo_gnn.gspmm against itself: time one tree, change the kernels, time again.

    python bench.py results/before_uniform.json --graph random
    python bench.py results/after_skewed.json  --graph skewed

Needs no DGL, which is the point -- the turbo-vs-DGL numbers in ../RESULTS.md
answer "are we faster than the thing we replace", and this one answers "did
this commit help", where DGL is a constant that only adds noise.

Forward and backward are timed separately, and the backward over one retained
graph rather than as (fwd+bwd) - fwd: that subtraction puts the noise of two
measurements into the number being compared.

Read the caveats in ../RESULTS.md before comparing two files: within one build
these numbers repeat to under 1%, but a cool card and a hot one differ by up to
8%, so a pair has to be measured back to back.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from turbo_gnn.graph import AdjacencyForwardBackwardWithNodeBuckets
from turbo_gnn.ops import gspmm

# One cell per shape of work rather than the whole 6x3 table: copy_u has no
# edge gradient, copy_e no node gradient, add none that reads the operands,
# mul/div both, and min/max take the arg-scatter backward instead of the
# transposed one.
CELLS = [
    ("copy_u", "sum"),
    ("copy_e", "sum"),
    ("add", "sum"),
    ("mul", "sum"),
    ("div", "sum"),
    ("mul", "min"),
    ("div", "min"),
    ("copy_u", "max"),
]


def make_graph(num_nodes: int, avg_degree: int, kind: str, device="cuda", seed=0):
    """`skewed` is dst ~ U^4, matching ../gspmm_vs_dgl/dgl_side.py.

    In-degree then concentrates on a handful of nodes, which is the only shape
    where the heavy-node path, its slicing and the backward's load balance are
    visible at all -- at a uniform degree of 8 none of them matter.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    num_edges = num_nodes * avg_degree
    src = torch.randint(0, num_nodes, (num_edges,), device=device, generator=g)
    if kind == "skewed":
        dst = (torch.rand(num_edges, device=device, generator=g) ** 4 * num_nodes).long().clamp_(0, num_nodes - 1)
    else:
        dst = torch.randint(0, num_nodes, (num_edges,), device=device, generator=g)

    # Self-loops keep every in-degree non-zero, so no cell is measuring the
    # isolated-node path.
    loops = torch.arange(num_nodes, device=device)
    edge_index = torch.stack([torch.cat([src, loops]), torch.cat([dst, loops])])
    return AdjacencyForwardBackwardWithNodeBuckets.from_edge_list(
        edge_index, num_nodes=num_nodes, quantile=0.95, index_dtype=torch.int32
    ).to(device)


def warm_up_device(fn, seconds=1.5) -> None:
    """Burn the clock ramp before anything is timed.

    The per-measurement warmup is not enough on its own: the *first*
    measurement a process takes came out 11-18% slower than the next one, and
    varied run to run while later ones repeated to 0.5%.  Whichever cell
    happened to be first therefore carried a penalty that looked exactly like a
    regression -- so the ramp is paid once, up front, and thrown away.
    """
    from time import perf_counter

    deadline = perf_counter() + seconds
    while perf_counter() < deadline:
        for _ in range(20):
            fn()
        torch.cuda.synchronize()


def time_ms(fn, iters=30, repeats=5, warmup=10) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) / iters)
    return statistics.median(samples)


def measure_cell(graph, num_nodes, num_edges, op, reduce, d, dtype, stages) -> dict:
    """Time one (op, reduce, d) both ways, or record why it could not run.

    An OOM is caught and reported rather than raised: this sweep is meant to be
    started and left alone, and a wide operand at float32 running out of memory
    should cost that one cell, not the night.
    """
    row = {"op": op, "reduce": reduce, "d": d}
    x = e = y = grad_out = None
    try:
        if op != "copy_e":
            x = torch.randn(num_nodes, d, device="cuda", dtype=dtype, requires_grad=True)
        if op != "copy_u":
            # offset away from zero so that div stays well conditioned
            e = (torch.rand(num_edges, d, device="cuda", dtype=dtype) + 0.5).requires_grad_(True)

        kw = {"op": op, "reduce": reduce, "pipeline_stages": stages}
        row["fwd_ms"] = time_ms(lambda: gspmm(graph, x, e, **kw))

        y = gspmm(graph, x, e, **kw)
        grad_out = torch.randn_like(y)

        def backward():
            for t in (x, e):
                if t is not None:
                    t.grad = None
            y.backward(grad_out, retain_graph=True)

        row["bwd_ms"] = time_ms(backward)
    except torch.OutOfMemoryError as exc:
        row["error"] = f"out of memory: {exc}".split("\n")[0]
    except RuntimeError as exc:
        row["error"] = str(exc).split("\n")[0]
    finally:
        del x, e, y, grad_out
        torch.cuda.empty_cache()
    return row


def run(args) -> dict:
    dtype = getattr(torch, args.dtype)
    graph = make_graph(args.nodes, args.degree, args.graph)
    num_edges = graph.forward_indices.numel()
    indptr = graph.forward_indptr.long()
    degrees = indptr[1:] - indptr[:-1]

    warm = torch.randn(args.nodes, args.dims[0], device="cuda", dtype=dtype)
    warm_up_device(lambda: gspmm(graph, warm, None, op="copy_u", reduce="sum"))
    del warm
    torch.cuda.empty_cache()

    cells = []
    for d in args.dims:
        for op, reduce in CELLS:
            cells.append(measure_cell(graph, args.nodes, num_edges, op, reduce, d, dtype, args.stages))

    return {
        "meta": {
            "graph": args.graph,
            "N": args.nodes,
            "E": num_edges,
            "dims": args.dims,
            "dtype": args.dtype,
            "stages": args.stages,
            "max_degree": int(degrees.max().item()),
            "heavy_nodes": int(graph.forward_heavy_nodes.numel()),
            "gpu": torch.cuda.get_device_name(0),
            "label": args.label,
        },
        "cells": cells,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out", help="where to write the results JSON")
    p.add_argument("--graph", default="random", choices=["random", "skewed"])
    p.add_argument("--nodes", type=int, default=169343)
    p.add_argument("--degree", type=int, default=8)
    p.add_argument("--dims", default="64", help="comma-separated feature widths, e.g. 32,64,128")
    p.add_argument("--dtype", default="float16")
    p.add_argument("--stages", type=int, default=0)
    p.add_argument("--label", default=None, help="shown on the chart, e.g. a commit subject")
    args = p.parse_args()
    args.dims = [int(v) for v in args.dims.split(",")]
    return args


def main() -> int:
    args = parse_args()
    blob = run(args)
    meta = blob["meta"]

    print(f"{meta['dtype']}, {meta['graph']}, N={meta['N']:,}, E={meta['E']:,}, "
          f"d={','.join(map(str, meta['dims']))}, stages={meta['stages']}")
    print(f"max in-degree {meta['max_degree']:,}, heavy nodes {meta['heavy_nodes']:,}, {meta['gpu']}")
    print(f"{'cell':14s} {'d':>4s} {'fwd':>8s} {'bwd':>8s}")
    for c in blob["cells"]:
        name = f"{c['op']}/{c['reduce']}"
        if "error" in c:
            print(f"{name:14s} {c['d']:4d}   SKIPPED  {c['error'][:60]}")
        else:
            print(f"{name:14s} {c['d']:4d} {c['fwd_ms']:8.3f} {c['bwd_ms']:8.3f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(blob, indent=1))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
