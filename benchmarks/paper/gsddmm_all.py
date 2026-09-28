"""All 32 g-SDDMM operators x widths, DGL or ours, with one lean direct-call protocol.

Replaces, for every ratio the paper states, the original sweep that went through the repo's
benchmark wrapper (a ~0.15 ms per-call floor that inflated small-graph speedups). Both backends:
  npz edge list + one self-loop per node; operands fp16, lhs ~ randn, rhs ~ rand + 0.5, node
  operands [N, D] and edge operands [E, D]; 5 warmup + 20 timed calls between CUDA events.

  --backend dgl   baselines interpreter (see README) (DGL 2.4): dgl.ops.<lhs>_<op>_<rhs>, copy_u, copy_v
  --backend ours  Turbo-GNN .venv: ops.gsddmm(variant="node") and (variant="edge"), defaults
"""
import os
from pathlib import Path
import argparse, json, statistics, sys
import numpy as np
import torch

CACHE = os.environ.get("GRAPH_CACHE", str(Path(__file__).resolve().parents[2] / "data" / "graph_cache"))
SIDE = {"u": "src", "v": "dst", "e": "edge"}
OPS = ["copy_u", "copy_v"] + [f"{l}_{o}_{r}" for l in "uve" for o in ("add", "sub", "mul", "div", "dot")
                              for r in "uve" if l != r]


def sync_time(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters):
        fn()
    b.record(); b.synchronize()
    return a.elapsed_time(b) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["dgl", "ours"], required=True)
    ap.add_argument("--graphs", nargs="+", required=True)
    ap.add_argument("--dims", type=int, nargs="+", default=[32, 64, 128, 256])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    assert len(OPS) == 32
    dev = "cuda"
    if a.backend == "dgl":
        import dgl
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from turbo_gnn.graph import AdjacencyForwardBackwardWithNodeBuckets as Graph
        from turbo_gnn import ops
    done = set()
    try:
        for l in open(a.out):
            c = json.loads(l); done.add((c["graph"], c["d"], c["op"]))
    except FileNotFoundError:
        pass
    fh = open(a.out, "a")
    for name in a.graphs:
        blob = np.load(f"{CACHE}/{name}.npz")
        n = int(blob["num_nodes"])
        ei = torch.from_numpy(blob["edge_index"]).long().to(dev)
        loops = torch.arange(n, device=dev)
        src, dst = torch.cat([ei[0], loops]), torch.cat([ei[1], loops])
        del ei
        if a.backend == "dgl":
            g = dgl.graph((src, dst), num_nodes=n)
        else:
            g = Graph.from_edge_list(torch.stack([src, dst]), n, index_dtype=torch.int32).to(dev)
        E = src.numel()
        del src, dst
        torch.cuda.empty_cache()
        for d in a.dims:
            for opname in OPS:
                if (name, d, opname) in done:
                    continue
                rec = {"graph": name, "d": d, "op": opname, "E": E, "N": n, "backend": a.backend}
                try:
                    if opname.startswith("copy"):
                        lhs_k, op, rhs_k = opname[-1], "copy", None
                    else:
                        lhs_k, op, rhs_k = opname.split("_")
                    mk = lambda k, f: f((E if k == "e" else n), d, device=dev, dtype=torch.float16)
                    lhs = mk(lhs_k, torch.randn)
                    rhs = None if rhs_k is None else mk(rhs_k, torch.rand) + 0.5
                    if a.backend == "dgl":
                        f = getattr(dgl.ops, opname)
                        fns = {"dgl": (lambda: f(g, lhs)) if rhs is None else (lambda: f(g, lhs, rhs))}
                    else:
                        kw = dict(op=op, lhs_target=SIDE[lhs_k], rhs_target=SIDE[rhs_k] if rhs_k else "dst")
                        fns = {v: (lambda v=v: ops.gsddmm(g, lhs, rhs, variant=v, **kw)) for v in ("node", "edge")}
                    for arm, fn in fns.items():
                        try:
                            rec[arm] = sync_time(fn)
                        except torch.cuda.OutOfMemoryError:
                            rec[arm] = "OOM"
                        torch.cuda.empty_cache()
                    del lhs, rhs
                except torch.cuda.OutOfMemoryError:
                    rec["error"] = "OOM allocating operands"
                torch.cuda.empty_cache()
                fh.write(json.dumps(rec) + "\n"); fh.flush()
            print(f"{name} d={d} done", flush=True)
        del g
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
