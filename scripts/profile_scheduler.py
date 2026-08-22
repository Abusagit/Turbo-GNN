"""Drive one turbo_gnn kernel under a fixed schedule, for profiling.

Kept deliberately small so that `ncu` sees a handful of launches and nothing else. Grid shape
is printed to stderr so the persistent block count can be checked against `blocks_per_sm *
SM_count` without parsing the profiler's output.

    ncu --kernel-name regex:reduction_aggr_forward_light \
        --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed \
        .venv/bin/python3 scripts/profile_scheduler.py --schedule dynamic --iters 3
"""

from __future__ import annotations

import argparse
import sys

import torch

from turbo_gnn import AdjacencyForwardBackwardWithNodeBuckets, gatv2_aggr, graph_transformer_aggr, reduction_aggr

CONVS = ("min_aggr", "gat_v2", "gt")


def load_ogbn(name: str, quantile: float) -> AdjacencyForwardBackwardWithNodeBuckets:
    """Load a real OGB node-property graph straight from its edge list.

    Deliberately bypasses src/data/datasets.py: that path adds self-loops and pulls in PyG,
    and here we want the graph's own degree distribution, which is the thing the scheduler
    is supposed to react to.
    """
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


def degree_stats(g: AdjacencyForwardBackwardWithNodeBuckets) -> str:
    """Summarise the in-degree distribution -- the skew is what makes balancing matter."""
    indptr = g._to_signed_view(g.forward_indptr)
    deg = (indptr[1:] - indptr[:-1]).float()
    q = torch.quantile(deg, torch.tensor([0.5, 0.9, 0.99, 0.999], device=deg.device))
    return (
        f"N={deg.numel():,} E={int(deg.sum()):,} mean={deg.mean():.1f} "
        f"p50={q[0]:.0f} p90={q[1]:.0f} p99={q[2]:.0f} p99.9={q[3]:.0f} max={deg.max():.0f}"
    )


def build_graph(
    num_nodes: int, avg_degree: int, quantile: float, *, skew: bool
) -> AdjacencyForwardBackwardWithNodeBuckets:
    """Random graph, optionally with a heavy tail so load imbalance actually exists.

    A uniform random graph has almost no degree spread, so every block gets the same amount
    of work and no scheduler can beat any other. `skew` gives a small set of nodes a very
    large in-degree, which is the regime the dynamic queue exists for.
    """
    dev = torch.device("cuda")
    src = torch.arange(num_nodes, device=dev).repeat_interleave(avg_degree)
    dst = torch.randint(0, num_nodes, (num_nodes * avg_degree,), device=dev)
    if skew:
        hubs = max(1, num_nodes // 200)
        extra = num_nodes * avg_degree
        hub_src = torch.randint(0, num_nodes, (extra,), device=dev)
        hub_dst = torch.randint(0, hubs, (extra,), device=dev)
        src = torch.cat([src, hub_src])
        dst = torch.cat([dst, hub_dst])
    return AdjacencyForwardBackwardWithNodeBuckets.from_edge_list(
        torch.stack([src, dst]), num_nodes, quantile=quantile, index_dtype=torch.int32
    ).to(dev)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--conv", default="min_aggr", choices=CONVS)
    p.add_argument("--ogbn", default=None, help="load a real OGB graph (e.g. ogbn-arxiv) instead of a random one")
    p.add_argument("--schedule", default="dynamic", choices=["one_per_block", "grid_stride", "precomputed", "dynamic"])
    p.add_argument("--blocks-per-sm", type=int, default=8)
    p.add_argument("--num-nodes", type=int, default=200000)
    p.add_argument("--avg-degree", type=int, default=15)
    p.add_argument("--quantile", type=float, default=0.99)
    p.add_argument("--feature-dim", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--iters", type=int, default=3, help="timed launches; keep small under ncu")
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--skew", action="store_true", help="add hub nodes so degrees are heavy-tailed")
    p.add_argument("--lpt", action="store_true", help="order buckets by descending degree")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("needs CUDA")
    dev = torch.device("cuda")
    torch.manual_seed(0)

    if args.ogbn:
        g = load_ogbn(args.ogbn, args.quantile)
        args.num_nodes = g.forward_indptr.numel() - 1
    else:
        g = build_graph(args.num_nodes, args.avg_degree, args.quantile, skew=args.skew)
    print(f"  graph: {args.ogbn or 'random'} {degree_stats(g)}", file=sys.stderr)
    if args.lpt:
        g = g.sorted_by_degree()

    sm = torch.cuda.get_device_properties(0).multi_processor_count
    kw = {"schedule": args.schedule, "blocks_per_sm": args.blocks_per_sm}

    if args.conv == "min_aggr":
        x = torch.randn(args.num_nodes, args.feature_dim, device=dev)
        fn = lambda: reduction_aggr(g, x, **kw)  # noqa: E731
        heads = 1
    elif args.conv == "gat_v2":
        shape = (args.num_nodes, args.heads, args.feature_dim)
        xl, xr = torch.randn(*shape, device=dev), torch.randn(*shape, device=dev)
        a = torch.randn(args.heads, args.feature_dim, device=dev)
        fn = lambda: gatv2_aggr(g, xl, xr, a, 0.2, **kw)  # noqa: E731
        heads = args.heads
    else:
        hd = args.feature_dim // args.heads
        q, k, v = (torch.randn(args.num_nodes, args.heads, hd, device=dev) for _ in range(3))
        fn = lambda: graph_transformer_aggr(g, q, q, k, v, None, **kw)  # noqa: E731
        heads = args.heads

    light, heavy = g.forward_light_nodes.numel(), g.forward_heavy_nodes.numel()
    if args.schedule == "one_per_block":
        gx_light, gx_heavy = light, heavy
    else:
        gx_light = min(max(1, -(-(sm * args.blocks_per_sm) // max(1, heads))), max(1, light))
        gx_heavy = min(max(1, -(-(sm * args.blocks_per_sm) // max(1, heads))), max(1, heavy))
    print(
        f"conv={args.conv} schedule={args.schedule} blocks_per_sm={args.blocks_per_sm} SM={sm} heads={heads}\n"
        f"  light bucket: {light} nodes -> grid.x={gx_light}\n"
        f"  heavy bucket: {heavy} nodes -> grid.x={gx_heavy}\n"
        f"  max_degree={g.max_degree}",
        file=sys.stderr,
    )

    for _ in range(args.warmup):
        fn()
    torch.cuda.synchronize()
    for _ in range(args.iters):
        fn()
    torch.cuda.synchronize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
