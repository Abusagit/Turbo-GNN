"""How much of each kernel's time is feature traffic? That is the ceiling on scheduling wins.

Run this before trying to make a scheduling change pay off, because it tells you whether there
is anything to win and what kind of win is available.

A scheduler can remove block-launch overhead and load imbalance. It cannot remove a byte of the
neighbour-feature reads, which are fixed by the graph and the head dim. Comparing measured time
against the time those reads alone would take at achievable bandwidth bounds what any
scheduling change could possibly do.

    CUDA_VISIBLE_DEVICES=$(python scripts/free_gpus.py --count 1) python scripts/scheduler_headroom.py
"""

from __future__ import annotations

import argparse

import torch

from turbo_gnn import AdjacencyForwardBackwardWithNodeBuckets, reduction_aggr

# Achievable (not peak) HBM bandwidth for an A100-SXM4-80GB. Override with --bandwidth on
# other hardware; the ratios below are only as meaningful as this number.
DEFAULT_BW = 1.55e12


def load(name: str, root: str, quantile: float) -> AdjacencyForwardBackwardWithNodeBuckets:
    """Edge list straight from OGB or GraphLand, without the self-loops the training path adds."""
    orig = torch.load

    def _load(*a, **kw):
        kw["weights_only"] = False
        return orig(*a, **kw)

    torch.load = _load
    try:
        if name.startswith("ogbn-"):
            from ogb.nodeproppred import NodePropPredDataset

            graph, _ = NodePropPredDataset(name=name, root=root)[0]
            ei, n = torch.from_numpy(graph["edge_index"]), int(graph["num_nodes"])
        else:
            from src.data.graphland_datasets import GraphLandDataset

            data = GraphLandDataset(root=root, name=name, split="RL")[0]
            ei, n = data.edge_index, int(data.num_nodes)
    finally:
        torch.load = orig
    return AdjacencyForwardBackwardWithNodeBuckets.from_edge_list(
        ei.to("cuda", torch.long), n, quantile=quantile, index_dtype=torch.int32
    ).to("cuda")


def timed(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)) / iters


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--graphs", nargs="+", default=["ogbn-arxiv", "tolokers-2", "ogbn-proteins"])
    p.add_argument("--head-dims", type=int, nargs="+", default=[128, 256])
    p.add_argument("--root", default="data")
    p.add_argument("--quantile", type=float, default=0.99)
    p.add_argument("--bandwidth", type=float, default=DEFAULT_BW, help="achievable HBM bytes/s")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("needs CUDA")

    print(f"  {'graph':<16}{'d':>5}{'edges':>14}{'measured':>12}{'traffic floor':>15}{'floor/meas':>12}")
    for name in args.graphs:
        g = load(name, args.root, args.quantile)
        n = g.forward_indptr.numel() - 1
        indptr = g._to_signed_view(g.forward_indptr)
        e = int((indptr[1:] - indptr[:-1]).sum())
        for d in args.head_dims:
            x = torch.randn(n, d, device="cuda")
            ms = timed(lambda xx=x, gr=g: reduction_aggr(gr, xx, schedule="one_per_block"), 10, 20)
            # One feature row read per edge, one written per node, plus the CSR itself.
            floor_ms = (e * d * 4 + n * d * 4 + (e + n) * 4) / args.bandwidth * 1e3
            print(f"  {name:<16}{d:>5}{e:>14,}{ms:>10.3f} ms{floor_ms:>13.3f} ms{floor_ms / ms * 100:>11.0f}%")
            del x
            torch.cuda.empty_cache()
        del g
        torch.cuda.empty_cache()

    print(
        "\n"
        "  below 100% -- DRAM-bound with little neighbour reuse (ogbn-arxiv measures 68-75%).\n"
        "                The rest is block-launch overhead, imbalance and compute, and that rest\n"
        "                is all a scheduler can ever address. Wins of 1.05-1.10x are a fair share\n"
        "                of a ~30% budget; there is no 2x hiding in it.\n\n"
        "  above 100% -- the graph could not have moved its own traffic across HBM in the time it\n"
        "                took, so the caches absorbed most of it (ogbn-proteins measures 263%).\n"
        "                These kernels are L2-reuse-bound: adjacent nodes share neighbours, so the\n"
        "                same feature rows are re-read from cache. Reuse depends entirely on the\n"
        "                order nodes are visited in -- exactly what a scheduler changes. That is\n"
        "                why ogbn-proteins swings from 0.57x to 1.10x across policies while\n"
        "                ogbn-arxiv barely moves: on a cache-bound graph the scheduler's first job\n"
        "                is to not destroy locality, and only its second job is to balance."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
