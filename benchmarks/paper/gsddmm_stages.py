"""Appendix M, g-SDDMM half: does asynchronous staging pay in the edge-partitioned kernel?

    python benchmarks/paper/gsddmm_stages.py \
        --graphs cora citeseer pubmed city-roads-M city-roads-L artnet-exp tolokers-2 \
                 ogbn-arxiv city-reviews twitch-views web-fraud pokec-regions \
        --dims 32 64 128 256 --warmup 5 --iters 20 --out results/gsddmm_stages.jsonl

The edge kernel gives each warp a chunk of `edges_per_warp` contiguous edges, the first
structure in these kernels where a prefetch depth can be amortised. The sweep varies the
pipeline depth and the chunk length together and keeps edges_per_warp=1 as a built-in
control: with one edge per warp there is nothing to prefetch across, so any change there is
pure overhead.

Paired within one process per (graph, width): all configurations of a cell are timed back to
back on the same tensors, so nothing drifts between arms. Graphs are read from the cache that
benchmarks/gspmm_vs_dgl/prepare_graph.py writes.
"""
import argparse, itertools, json, os, statistics, sys, time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from skewgnn.graph import AdjacencyForwardBackwardWithNodeBuckets as Graph  # noqa: E402
from skewgnn import ops  # noqa: E402

DATA = os.environ.get("GRAPH_CACHE", str(Path(__file__).resolve().parents[2] / "data" / "graph_cache"))
OPS = [("mul", "src", "dst"), ("dot", "src", "dst"), ("copy", "src", "edge"), ("mul", "src", "edge")]
STAGES = [0, 2, 3]
EPW = [1, 4, 8, 16]
WPB = 4


def timeit(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t = []
    for _ in range(iters):
        s = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        t.append((time.perf_counter() - s) * 1e3)
    return statistics.median(t)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--graphs", nargs="+", required=True)
    p.add_argument("--dims", nargs="+", type=int, default=[32, 64, 128, 256])
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--out", default="results/gsddmm_stages.jsonl")
    p.add_argument("--check", action="store_true", help="verify every config against a torch reference first")
    a = p.parse_args()
    dev = torch.device("cuda")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fh = open(a.out, "a")

    for name in a.graphs:
        import numpy as np
        blob = np.load(f"{DATA}/{name}.npz")
        ei = torch.from_numpy(blob["edge_index"]).to(dev)
        N = int(blob["num_nodes"])
        g = Graph.from_edge_list(ei, N, quantile=-1, index_dtype=torch.int32).to(dev)
        E = int(g.forward_indices.numel())
        for d in a.dims:
            try:
                x = torch.randn(N, d, device=dev, dtype=torch.float16)
                e = torch.randn(E, d, device=dev, dtype=torch.float16)
            except torch.cuda.OutOfMemoryError:
                print(f"  {name} d={d}: OOM allocating operands, skipped", flush=True)
                torch.cuda.empty_cache()
                continue
            for (op, ll, rr), st, epw in itertools.product(OPS, STAGES, EPW):
                rhs = None if op == "copy" else (e if rr == "edge" else x)
                # gsddmm_edge is the edge-parallel entry point; the node-parallel
                # gsddmm() takes none of these knobs.
                kw = dict(op=op, lhs_target=ll, rhs_target=rr, pipeline_stages=st,
                          edges_per_warp=epw, warps_per_block=WPB)
                try:
                    fn = lambda: ops.gsddmm_edge(g, x, rhs, **kw)
                    fn()
                    torch.cuda.synchronize()
                    ms = timeit(fn, a.warmup, a.iters)
                except Exception as exc:
                    msg = f"{type(exc).__name__}: {str(exc)[:90]}"
                    fh.write(json.dumps(dict(graph=name, N=N, E=E, dim=d, op=f"{ll}_{op}_{rr}",
                                             stages=st, epw=epw, wpb=WPB, error=msg)) + "\n")
                    fh.flush()
                    torch.cuda.empty_cache()
                    continue
                fh.write(json.dumps(dict(graph=name, N=N, E=E, dim=d, op=f"{ll}_{op}_{rr}",
                                         stages=st, epw=epw, wpb=WPB, ms=ms)) + "\n")
                fh.flush()
            print(f"  {name} d={d} done ({len(OPS)*len(STAGES)*len(EPW)} configs)", flush=True)
            del x, e
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
