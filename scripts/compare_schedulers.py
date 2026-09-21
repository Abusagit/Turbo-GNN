"""Compare scheduler policies on one graph, timing with CUDA events.

Loads the graph once and sweeps policies (and `blocks_per_sm` for the persistent ones) in a
single process, which is far cheaper than one `ncu` launch per configuration. Use `ncu` via
``scripts/profile_scheduler.py`` afterwards for the occupancy detail on whichever
configuration looks interesting.

    python scripts/compare_schedulers.py --ogbn ogbn-arxiv --conv min_aggr
    python scripts/compare_schedulers.py --ogbn ogbn-proteins --conv gt --heads 8
"""

from __future__ import annotations

import argparse
import sys

import torch

from turbo_gnn import AdjacencyForwardBackwardWithNodeBuckets, gatv2_aggr, graph_transformer_aggr, reduction_aggr

SCHEDULES = ["one_per_block", "grid_stride", "precomputed", "dynamic"]
PERSISTENT = SCHEDULES[1:]


def load_ogbn(name: str, quantile: float) -> AdjacencyForwardBackwardWithNodeBuckets:
    from ogb.nodeproppred import NodePropPredDataset

    # ogb calls torch.load() on its own preprocessed cache, which PyTorch >= 2.6 refuses by
    # default (weights_only=True). The file is one we just downloaded from OGB, so opt out
    # for the duration of the load only.
    _orig_load = torch.load

    def _load(*a, **kw):
        kw["weights_only"] = False
        return _orig_load(*a, **kw)

    torch.load = _load
    try:
        graph, _ = NodePropPredDataset(name=name, root="data")[0]
    finally:
        torch.load = _orig_load
    ei = torch.from_numpy(graph["edge_index"]).to("cuda", torch.long)
    return AdjacencyForwardBackwardWithNodeBuckets.from_edge_list(
        ei, int(graph["num_nodes"]), quantile=quantile, index_dtype=torch.int32
    ).to("cuda")


def describe(g: AdjacencyForwardBackwardWithNodeBuckets) -> None:
    """Print the in-degree distribution: the skew is what a balancing scheduler reacts to."""
    indptr = g._to_signed_view(g.forward_indptr)
    deg = (indptr[1:] - indptr[:-1]).float()
    q = torch.quantile(deg, torch.tensor([0.5, 0.9, 0.99, 0.999], device=deg.device))
    print(
        f"  N={deg.numel():,}  E={int(deg.sum()):,}  mean={deg.mean():.1f}  "
        f"p50={q[0]:.0f}  p90={q[1]:.0f}  p99={q[2]:.0f}  p99.9={q[3]:.0f}  max={deg.max():.0f}\n"
        f"  buckets @ quantile: light={g.forward_light_nodes.numel():,}  heavy={g.forward_heavy_nodes.numel():,}"
    )


def make_fn(conv: str, g, n: int, feature_dim: int, heads: int, kw: dict):
    dev = torch.device("cuda")
    if conv == "min_aggr":
        x = torch.randn(n, feature_dim, device=dev)
        return lambda: reduction_aggr(g, x, **kw)
    if conv == "gat_v2":
        shape = (n, heads, feature_dim)
        xl, xr = torch.randn(*shape, device=dev), torch.randn(*shape, device=dev)
        a = torch.randn(heads, feature_dim, device=dev)
        return lambda: gatv2_aggr(g, xl, xr, a, 0.2, **kw)
    hd = feature_dim // heads
    q, k, v = (torch.randn(n, heads, hd, device=dev) for _ in range(3))
    return lambda: graph_transformer_aggr(g, q, q, k, v, None, **kw)


def timed(fn, warmup: int, iters: int) -> float:
    """Milliseconds per call, measured with CUDA events."""
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
    p.add_argument("--ogbn", default=None, help="real OGB graph, e.g. ogbn-arxiv / ogbn-proteins")
    p.add_argument("--num-nodes", type=int, default=200000, help="random-graph fallback")
    p.add_argument("--avg-degree", type=int, default=15)
    p.add_argument("--conv", default="min_aggr", choices=["min_aggr", "gat_v2", "gt"])
    p.add_argument("--quantile", type=float, default=0.99)
    p.add_argument("--feature-dim", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--blocks-per-sm", type=int, nargs="+", default=[8, 16, 32, 64])
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("needs CUDA")
    torch.manual_seed(0)

    if args.ogbn:
        g = load_ogbn(args.ogbn, args.quantile)
    else:
        dev = torch.device("cuda")
        src = torch.arange(args.num_nodes, device=dev).repeat_interleave(args.avg_degree)
        dst = torch.randint(0, args.num_nodes, (args.num_nodes * args.avg_degree,), device=dev)
        g = AdjacencyForwardBackwardWithNodeBuckets.from_edge_list(
            torch.stack([src, dst]), args.num_nodes, quantile=args.quantile, index_dtype=torch.int32
        ).to(dev)

    n = g.forward_indptr.numel() - 1
    print(f"{args.ogbn or 'random'} / conv={args.conv} / feature_dim={args.feature_dim} / heads={args.heads}")
    describe(g)

    g_lpt = g.sorted_by_degree()
    sm = torch.cuda.get_device_properties(0).multi_processor_count
    print(f"  SM count={sm}\n")

    base = timed(
        make_fn(args.conv, g, n, args.feature_dim, args.heads, {"schedule": "one_per_block"}), args.warmup, args.iters
    )
    print(f"  {'one_per_block (baseline)':<34} {base:8.3f} ms")

    rows = []
    for sched in PERSISTENT:
        for bps in args.blocks_per_sm:
            for lpt, graph in (("", g), (" +LPT", g_lpt)):
                kw = {"schedule": sched, "blocks_per_sm": bps}
                fn = make_fn(args.conv, graph, n, args.feature_dim, args.heads, kw)
                try:
                    ms = timed(fn, args.warmup, args.iters)
                except Exception as exc:  # a policy may legitimately not apply
                    print(f"  {sched}{lpt} bps={bps}: {type(exc).__name__}: {str(exc)[:60]}", file=sys.stderr)
                    continue
                rows.append((ms, f"{sched}{lpt}", bps))

    rows.sort()
    print(f"\n  {'policy':<26}{'bps':>5}{'ms':>10}{'vs baseline':>14}")
    for ms, name, bps in rows:
        delta = (ms / base - 1) * 100
        print(f"  {name:<26}{bps:>5}{ms:>10.3f}{delta:>13.1f}%")

    best_ms, best_name, best_bps = rows[0]
    verdict = "FASTER" if best_ms < base else "slower"
    print(
        f"\n  best persistent: {best_name} @ bps={best_bps} -> {best_ms:.3f} ms "
        f"({verdict} than one_per_block by {abs(best_ms / base - 1) * 100:.1f}%)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
