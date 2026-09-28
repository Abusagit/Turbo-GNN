"""PyG and DGL, timed in the SAME process, for the generalized primitives in the paper's tables.

The paper's ours-vs-DGL ratios come from two existing harnesses; neither has a PyG arm. Timing PyG
next to DGL in one process gives a paired PyG/DGL ratio that is free of machine drift, which then
chains onto the published ours/DGL ratio.

  g-SpMM   follows dev/scanhex12/gspmm_vs_dgl/dgl_side.py: graph from the npz cache plus one
           self-loop per node, operands in the target dtype, 10 warmup, 30 iters x 3 repeats,
           median of the repeat means; fwd, and bwd replayed with retain_graph.
  g-SDDMM  follows gsddmm_all.py: two independent node
           operands (u, v), 5 warmup + 20 timed calls, forward only.

PyG is used the way MessagePassing uses it internally: index_select the endpoint rows, apply the
message op, and aggregate with torch_geometric.utils.scatter.

Run with the baselines interpreter (see README) (torch 2.4 + DGL 2.4 + PyG 2.8).
"""
from pathlib import Path
import argparse, json, os, statistics, sys, time

import numpy as np
import torch
import dgl
import torch_geometric
from torch_geometric.utils import scatter

CACHE = os.environ.get("GRAPH_CACHE", str(Path(__file__).resolve().parents[2] / "data" / "graph_cache"))


def sync_time(fn, warmup, iters, repeats):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(repeats):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(iters):
            fn()
        b.record(); b.synchronize()
        out.append(a.elapsed_time(b) / iters)
    return statistics.median(out)


def graph(name, dev):
    blob = np.load(f"{CACHE}/{name}.npz")
    n = int(blob["num_nodes"])
    ei = torch.from_numpy(blob["edge_index"]).long().to(dev)
    loops = torch.arange(n, device=dev)
    src, dst = torch.cat([ei[0], loops]), torch.cat([ei[1], loops])
    return src, dst, n


def spmm(name, dims, ops, dt, rec):
    dev = "cuda"
    src, dst, n = graph(name, dev)
    g = dgl.graph((src, dst), num_nodes=n)
    E = src.numel()
    for d in dims:
        for op, red in ops:
            if op != "copy_u" and E > 10_000_000:
                continue                     # same rule as the DGL harness: [E, d] does not fit
            x = torch.randn(n, d, device=dev, dtype=dt)
            e = torch.rand(E, d, device=dev, dtype=dt) + 0.5 if op != "copy_u" else None
            gseed = torch.randn(n, d, device=dev, dtype=dt)
            dgl_fn = getattr(dgl.ops, f"copy_u_{red}" if op == "copy_u" else f"u_{op}_e_{red}")
            arms = {
                "dgl": lambda x, e: dgl_fn(g, x) if e is None else dgl_fn(g, x, e),
                "pyg": lambda x, e: scatter(x.index_select(0, src) if e is None
                                            else x.index_select(0, src) * e,
                                            dst, dim=0, dim_size=n, reduce=red),
            }
            cell = {"family": "gspmm", "graph": name, "d": d, "op": op, "reduce": red, "E": E, "N": n}
            for arm, f in arms.items():
                try:
                    cell[f"{arm}_fwd"] = sync_time(lambda: f(x, e), 10, 30, 3)
                    xs = x.clone().requires_grad_(True)
                    es = e.clone().requires_grad_(True) if e is not None else None
                    kept = f(xs, es)
                    leaves = [t for t in (xs, es) if t is not None]
                    def bwd():
                        for t in leaves: t.grad = None
                        kept.backward(gseed, retain_graph=True)
                    cell[f"{arm}_bwd"] = sync_time(bwd, 10, 30, 3)
                    del kept, xs, es, leaves
                except torch.cuda.OutOfMemoryError:
                    cell[f"{arm}_fwd"] = cell.get(f"{arm}_fwd") or "OOM"
                    cell[f"{arm}_bwd"] = "OOM"
                torch.cuda.empty_cache()
            rec(cell)
            del x, e, gseed
            torch.cuda.empty_cache()


def sddmm(name, dims, ops, dt, rec):
    dev = "cuda"
    src, dst, n = graph(name, dev)
    g = dgl.graph((src, dst), num_nodes=n)
    E = src.numel()
    pyg_op = {"dot": lambda a, b: (a * b).sum(-1, keepdim=True), "add": torch.add,
              "sub": torch.sub, "mul": torch.mul, "div": torch.div}
    for d in dims:
        for op in ops:
            u = torch.randn(n, d, device=dev, dtype=dt)
            v = torch.randn(n, d, device=dev, dtype=dt)
            dgl_fn = getattr(dgl.ops, f"u_{op}_v")
            arms = {"dgl": lambda: dgl_fn(g, u, v),
                    "pyg": lambda: pyg_op[op](u.index_select(0, src), v.index_select(0, dst))}
            cell = {"family": "gsddmm", "graph": name, "d": d, "op": f"u_{op}_v", "E": E, "N": n}
            for arm, f in arms.items():
                try:
                    cell[f"{arm}_fwd"] = sync_time(f, 5, 20, 1)
                except torch.cuda.OutOfMemoryError:
                    cell[f"{arm}_fwd"] = "OOM"
                torch.cuda.empty_cache()
            rec(cell)
            del u, v
            torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["gspmm", "gsddmm"], required=True)
    ap.add_argument("--graphs", nargs="+", required=True)
    ap.add_argument("--dims", type=int, nargs="+", default=[128])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dt = torch.float16
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            c = json.loads(l); done.add((c["graph"], c["d"], c["op"], c.get("reduce")))
    fh = open(a.out, "a")
    def rec(c):
        fh.write(json.dumps(c) + "\n"); fh.flush()
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in c.items()}, flush=True)
    print(f"{torch.cuda.get_device_name(0)} | torch {torch.__version__} | dgl {dgl.__version__} "
          f"| pyg {torch_geometric.__version__}", flush=True)
    for name in a.graphs:
        t0 = time.time()
        if a.family == "gspmm":
            ops = [("copy_u", "sum"), ("copy_u", "max"), ("mul", "sum")]
            ops = [o for o in ops if (name, a.dims[0], o[0], o[1]) not in done]
            if ops: spmm(name, a.dims, ops, dt, rec)
        else:
            ops = [o for o in ["dot", "add", "mul"] if (name, a.dims[0], f"u_{o}_v", None) not in done]
            if ops: sddmm(name, a.dims, ops, dt, rec)
        print(f"== {name} done in {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
