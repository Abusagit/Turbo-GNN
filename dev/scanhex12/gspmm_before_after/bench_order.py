"""Node visit order: natural bucket order against descending degree.

    python bench_order.py results/order_ogbn-arxiv_float16.json --graph ogbn-arxiv

Both configurations are the same build and the same graph -- only the order of
the light/heavy bucket arrays differs, which is what the kernels walk.  The
forward is bit-exact either way (the array picks which node a block visits,
never where the result is written), so this measures scheduling alone.

Reuses bench.py's graph loading and timing, so a number here is comparable to
one there.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

# bench.py sits next to this file, and the wrapper that runs this one through
# runpy does not put that directory on the path itself.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench import CELLS, make_graph, time_ms, warm_up_device
from turbo_gnn.ops import gspmm


def measure(graph, num_nodes, num_edges, op, reduce, d, dtype, stages) -> dict:
    row = {"op": op, "reduce": reduce, "d": d}
    x = e = y = grad_out = None
    try:
        if op != "copy_e":
            x = torch.randn(num_nodes, d, device="cuda", dtype=dtype, requires_grad=True)
        if op != "copy_u":
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


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out")
    p.add_argument("--graph", default="random")
    p.add_argument("--nodes", type=int, default=200_000)
    p.add_argument("--degree", type=int, default=16)
    p.add_argument("--dims", default="64")
    p.add_argument("--dtype", default="float16", choices=["float32", "float16"])
    p.add_argument("--stages", type=int, default=0)
    p.add_argument("--graph-cache", default="data/graph_cache")
    args = p.parse_args()
    dims = [int(v) for v in args.dims.split(",")]
    dtype = getattr(torch, args.dtype)

    natural, num_nodes = make_graph(args.graph, args.nodes, args.degree, args.graph_cache)
    num_edges = natural.forward_indices.numel()
    indptr = natural.forward_indptr.long()
    degrees = indptr[1:] - indptr[:-1]
    by_degree = natural.sorted_by_degree()

    warm = torch.randn(num_nodes, dims[0], device="cuda", dtype=dtype)
    warm_up_device(lambda: gspmm(natural, warm, None, op="copy_u", reduce="sum"))
    del warm
    torch.cuda.empty_cache()

    cells = []
    for d in dims:
        for op, reduce in CELLS:
            row = {"op": op, "reduce": reduce, "d": d}
            for name, g in (("natural", natural), ("degree", by_degree)):
                got = measure(g, num_nodes, num_edges, op, reduce, d, dtype, args.stages)
                for key in ("fwd_ms", "bwd_ms", "error"):
                    if key in got:
                        row[f"{name}_{key}"] = got[key]
            cells.append(row)
            fwd = row.get("natural_fwd_ms")
            deg = row.get("degree_fwd_ms")
            if fwd and deg:
                print(f"  {op:8} {reduce:4} d={d:<4} fwd {fwd:8.3f} -> {deg:8.3f}  x{fwd / deg:.3f}"
                      f"   bwd {row['natural_bwd_ms']:8.3f} -> {row['degree_bwd_ms']:8.3f}"
                      f"  x{row['natural_bwd_ms'] / row['degree_bwd_ms']:.3f}")

    ratios = {
        kind: [c[f"natural_{kind}_ms"] / c[f"degree_{kind}_ms"] for c in cells if f"degree_{kind}_ms" in c]
        for kind in ("fwd", "bwd")
    }
    blob = {
        "meta": {
            "graph": args.graph,
            "N": num_nodes,
            "E": num_edges,
            "dims": dims,
            "dtype": args.dtype,
            "stages": args.stages,
            "max_degree": int(degrees.max().item()),
            "mean_degree": float(degrees.float().mean().item()),
            "skew": float(degrees.max().item() / max(degrees.float().mean().item(), 1e-9)),
            "heavy_nodes": int(natural.forward_heavy_nodes.numel()),
            "gpu": torch.cuda.get_device_name(0),
        },
        "cells": cells,
    }
    for kind, values in ratios.items():
        if values:
            blob["meta"][f"{kind}_geomean"] = float(statistics.geometric_mean(values))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(blob, open(args.out, "w"), indent=1)
    summary = " ".join(f"{k} geomean x{blob['meta'][f'{k}_geomean']:.3f}" for k in ratios if f"{k}_geomean" in blob["meta"])
    print(f"{args.graph} {args.dtype} skew={blob['meta']['skew']:.0f}x -> {summary}  ({args.out})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
