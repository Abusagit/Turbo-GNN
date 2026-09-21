"""Appendix A: the degree statistics of every graph in the graph cache.

    python benchmarks/paper/graph_meta.py            # reads GRAPH_CACHE, writes results/graph_meta.json

Compute graph metadata from the exported edge lists, rather than hard-coding it.

The report previously carried a hand-written table of (nodes, edges, max degree)
for 14 graphs. This derives every figure from the actual edge list for all of
them, and records BOTH edge counts that matter:

  E_raw  edges in the exported edge_index
  E_csr  edges the benchmark harness actually walks, after
         AdjacencyForwardBackwardWithNodeBuckets.from_edge_list() is done with it

Those two differ -- cora is 13,264 raw but 15,972 in the harness tables -- and
quoting the wrong one misstates every per-edge figure derived from it.
"""
import glob, json, os, sys
from pathlib import Path
import numpy as np
import torch

CACHE = os.environ.get("GRAPH_CACHE", str(Path(__file__).resolve().parents[2] / "data" / "graph_cache"))
OUT = sys.argv[1] if len(sys.argv) > 1 else "results/graph_meta.json"
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def stats(name):
    blob = np.load(f"{CACHE}/{name}.npz")
    ei = torch.from_numpy(blob["edge_index"]).long()
    N = int(blob["num_nodes"])
    E = int(ei.shape[1])
    src, dst = ei[0], ei[1]
    ind = torch.bincount(dst, minlength=N).float()
    outd = torch.bincount(src, minlength=N).float()
    q = torch.tensor([0.5, 0.95, 0.99])
    # quantile() caps at 16M elements; sample deterministically above that
    sample = ind if ind.numel() <= 16_000_000 else ind[torch.linspace(0, ind.numel() - 1, 16_000_000).long()]
    med, p95, p99 = [float(v) for v in torch.quantile(sample, q)]
    mx = float(ind.max())
    top1 = max(1, N // 100)
    conc = float(ind.sort(descending=True).values[:top1].sum() / max(E, 1))
    selfl = int((src == dst).sum())
    # symmetric? compare the sorted (min,max) pair multiset against the edge multiset
    a = (src.min(dst) * N + src.max(dst))
    sym = bool(torch.equal(*[x.sort().values for x in (a, a)])) and E % 2 == 0
    return dict(name=name, N=N, E_raw=E, avg_deg=E / N, max_in=mx, max_out=float(outd.max()),
                med_in=med, p95_in=p95, p99_in=p99, skew=mx / (E / N) if E else 0,
                isolated=int((ind == 0).sum()), self_loops=selfl,
                top1pct_edge_share=conc)


def csr_edges(name):
    """Edges the harness walks, which is what its own tables report."""
    from skewgnn.graph import AdjacencyForwardBackwardWithNodeBuckets as G
    blob = np.load(f"{CACHE}/{name}.npz")
    ei = torch.from_numpy(blob["edge_index"]).long()
    g = G.from_edge_list(ei, int(blob["num_nodes"]), quantile=-1, index_dtype=torch.int32)
    return int(g.forward_indices.numel())


out = {}
for p in sorted(glob.glob(f"{CACHE}/*.npz")):
    n = os.path.basename(p)[:-4]
    try:
        s = stats(n)
        s["E_csr"] = csr_edges(n)
        out[n] = s
        print(f"  {n:<18} N={s['N']:>10,}  E_raw={s['E_raw']:>12,}  E_csr={s['E_csr']:>12,}  "
              f"max_in={int(s['max_in']):>8,}  skew={s['skew']:>8.0f}x", flush=True)
    except Exception as exc:
        print(f"  {n:<18} FAILED {type(exc).__name__}: {exc}", flush=True)
os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
json.dump(out, open(OUT, "w"), indent=1)
print(f"\nwrote {OUT}  ({len(out)} graphs)")
