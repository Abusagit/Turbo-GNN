"""Forward and backward timings for every scheduler policy, across the benchmark graphs.

``scripts/compare_schedulers.py`` answers "which policy wins on this one graph"; this script
answers the question the policies were built for: **does the persistent path beat the
one-block-per-node launch across the graphs we actually care about, at the head dimensions we
actually run?** It therefore differs in three ways:

* it walks the graphs named in ``configs/datasets/main/`` plus ogbn-proteins, loading each
  one once and reusing it for every configuration;
* it times the **backward** pass as well as the forward one, separately, because they use
  different kernels with different degrees of load imbalance (backward walks the transposed
  CSR, so a graph that is balanced forward can be skewed backward);
* it reports each policy against the ``one_per_block`` baseline *per graph and per direction*,
  so a win on one graph cannot hide a regression on another.

Timing is CUDA-event based with the graph resident, so the numbers are kernel time plus
launch overhead and nothing else -- no dataset loading, no allocation of the input tensors.

    python scripts/benchmark_scheduler_suite.py --conv gt --head-dims 128 256
    python scripts/benchmark_scheduler_suite.py --graphs ogbn-arxiv ogbn-proteins --backward
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback

import torch

from turbo_gnn import AdjacencyForwardBackwardWithNodeBuckets, gatv2_aggr, graph_transformer_aggr, reduction_aggr

BASELINE = "one_per_block"
PERSISTENT = ["grid_stride", "precomputed", "dynamic"]

# The graphs named by configs/datasets/main/, plus ogbn-proteins. Split by loader because the
# GraphLand ones come through a PyG InMemoryDataset and the OGB ones through NodePropPredDataset.
OGB_GRAPHS = ["ogbn-arxiv", "ogbn-products", "ogbn-proteins"]
GRAPHLAND_GRAPHS = ["avazu-ctr", "city-reviews", "city-roads-M", "hm-categories", "tolokers-2", "twitch-views"]
ALL_GRAPHS = OGB_GRAPHS + GRAPHLAND_GRAPHS


def _no_weights_only(fn):
    """OGB and older PyG caches are pickled objects, which torch>=2.6 refuses by default."""

    def wrapped(*a, **kw):
        kw["weights_only"] = False
        return fn(*a, **kw)

    return wrapped


def load_edge_index(name: str, root: str = "data") -> tuple[torch.Tensor, int]:
    """Return ``(edge_index, num_nodes)`` on the GPU, without adding self-loops.

    Self-loops are what ``src/data/datasets.py`` adds for training; here they would flatten
    the degree distribution slightly, and the distribution is the thing under test.
    """
    orig, torch.load = torch.load, _no_weights_only(torch.load)
    try:
        if name in OGB_GRAPHS:
            from ogb.nodeproppred import NodePropPredDataset

            graph, _ = NodePropPredDataset(name=name, root=root)[0]
            ei = torch.from_numpy(graph["edge_index"])
            return ei.to("cuda", torch.long), int(graph["num_nodes"])

        from src.data.graphland_datasets import GraphLandDataset

        data = GraphLandDataset(root=root, name=name, split="RL")[0]
        return data.edge_index.to("cuda", torch.long), int(data.num_nodes)
    finally:
        torch.load = orig


def build(name: str, quantile: float, root: str) -> AdjacencyForwardBackwardWithNodeBuckets:
    ei, n = load_edge_index(name, root)
    return AdjacencyForwardBackwardWithNodeBuckets.from_edge_list(ei, n, quantile=quantile, index_dtype=torch.int32).to(
        "cuda"
    )


def degree_summary(g: AdjacencyForwardBackwardWithNodeBuckets) -> dict:
    indptr = g._to_signed_view(g.forward_indptr)
    deg = (indptr[1:] - indptr[:-1]).float()
    q = torch.quantile(deg, torch.tensor([0.5, 0.99], device=deg.device))
    return {
        "N": int(deg.numel()),
        "E": int(deg.sum()),
        "mean_deg": round(float(deg.mean()), 1),
        "p50": int(q[0]),
        "p99": int(q[1]),
        "max": int(deg.max()),
        "light": int(g.forward_light_nodes.numel()),
        "heavy": int(g.forward_heavy_nodes.numel()),
    }


def make_inputs(conv: str, n: int, head_dim: int, heads: int, *, grad: bool):
    """Allocate once per (conv, shape); reused across every policy so allocation is not timed."""
    dev = torch.device("cuda")
    gen = torch.Generator(dev).manual_seed(0)
    r = lambda *s: torch.randn(*s, device=dev, generator=gen, requires_grad=grad)  # noqa: E731
    if conv == "min_aggr":
        return (r(n, head_dim),)
    if conv == "gat_v2":
        return r(n, heads, head_dim), r(n, heads, head_dim), torch.randn(heads, head_dim, device=dev, generator=gen)
    return r(n, heads, head_dim), r(n, heads, head_dim), r(n, heads, head_dim)


def call(conv: str, g, inputs, kw):
    if conv == "min_aggr":
        return reduction_aggr(g, inputs[0], **kw)
    if conv == "gat_v2":
        return gatv2_aggr(g, inputs[0], inputs[1], inputs[2], 0.2, **kw)
    q, k, v = inputs
    return graph_transformer_aggr(g, q, q, k, v, None, **kw)


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


def time_forward(conv, g, inputs, kw, warmup, iters) -> float:
    with torch.no_grad():
        return timed(lambda: call(conv, g, inputs, kw), warmup, iters)


def time_backward(conv, g, inputs, kw, warmup, iters) -> float:
    """Backward only: the forward is replayed but its cost is subtracted out.

    ``autograd.grad`` needs a fresh graph each call, so the forward cannot be hoisted. Timing
    ``forward+backward`` and subtracting a separately measured forward is the standard way to
    isolate the backward kernels, and it is what the numbers below report.
    """
    leaves = [t for t in inputs if t.requires_grad]

    def fwd_bwd():
        out = call(conv, g, inputs, kw)
        torch.autograd.grad(out, leaves, torch.ones_like(out), retain_graph=False)

    total = timed(fwd_bwd, warmup, iters)
    fwd = timed(lambda: call(conv, g, inputs, kw), warmup, iters)
    return max(total - fwd, 0.0)


def measure_cell(name, conv, hd, direction, g, orders, info, args) -> list[dict]:
    """Time every policy for one (graph, conv, head dim, direction) and return its rows.

    A function rather than an inner loop so the input tensors are released at return: at head
    dim 256 ogbn-products needs most of the device, and the next cell cannot start until the
    previous one's activations are gone.
    """
    need_grad = direction == "bwd"
    try:
        inputs = make_inputs(conv, info["N"], hd, args.heads, grad=need_grad)
    except torch.OutOfMemoryError:
        print(f"  {conv} d={hd} {direction}: OOM allocating inputs", file=sys.stderr)
        torch.cuda.empty_cache()
        return []

    runner = time_backward if need_grad else time_forward
    try:
        base = runner(conv, g, inputs, {"schedule": BASELINE}, args.warmup, args.iters)
    except Exception as exc:
        print(f"  {conv} d={hd} {direction}: baseline failed: {exc}"[:200], file=sys.stderr)
        torch.cuda.empty_cache()
        return []

    out: list[dict] = []
    best: tuple[float, str] | None = None
    print(f"  {conv:<8} d={hd:<4} {direction}   baseline {base:8.3f} ms")
    for sched in PERSISTENT:
        for bps in args.blocks_per_sm:
            for chunk in args.sched_chunk if sched == "dynamic" else [1]:
                for tag, gr in orders:
                    kw = {"schedule": sched, "blocks_per_sm": bps, "sched_chunk": chunk}
                    try:
                        ms = runner(conv, gr, inputs, kw, args.warmup, args.iters)
                    except Exception:
                        traceback.print_exc(limit=1, file=sys.stderr)
                        continue
                    label = f"{sched}{tag}/bps{bps}" + (f"/c{chunk}" if sched == "dynamic" else "")
                    out.append(
                        dict(
                            graph=name,
                            conv=conv,
                            head_dim=hd,
                            dir=direction,
                            policy=label,
                            ms=ms,
                            baseline_ms=base,
                            speedup=base / ms,
                            **info,
                        )
                    )
                    if best is None or ms < best[0]:
                        best = (ms, label)
    if best:
        mark = "WIN " if best[0] < base else "LOSS"
        print(f"           {mark} best {best[1]:<28} {best[0]:8.3f} ms  x{base / best[0]:.2f}")
    torch.cuda.empty_cache()
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--graphs", nargs="+", default=ALL_GRAPHS)
    p.add_argument("--conv", nargs="+", default=["min_aggr", "gat_v2", "gt"])
    p.add_argument("--head-dims", type=int, nargs="+", default=[128, 256])
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--quantile", type=float, default=0.99)
    p.add_argument("--root", default="data")
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--blocks-per-sm", type=int, nargs="+", default=[256])
    p.add_argument("--sched-chunk", type=int, nargs="+", default=[4])
    p.add_argument("--lpt", action="store_true", help="also try each policy on a descending-degree order")
    p.add_argument("--no-backward", action="store_true")
    p.add_argument("--json", default=None, help="write raw rows here")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("needs CUDA")

    rows: list[dict] = []
    for name in args.graphs:
        t0 = time.time()
        try:
            g = build(name, args.quantile, args.root)
        except Exception as exc:
            print(f"\n### {name}: SKIPPED -- {type(exc).__name__}: {str(exc)[:160]}", file=sys.stderr)
            continue
        info = degree_summary(g)
        orders = [("", g)] + ([("+LPT", g.sorted_by_degree())] if args.lpt else [])
        print(f"\n### {name}  ({time.time() - t0:.0f}s to load)")
        print(
            f"    N={info['N']:,} E={info['E']:,} mean={info['mean_deg']} p50={info['p50']} "
            f"p99={info['p99']} max={info['max']} light={info['light']:,} heavy={info['heavy']:,}"
        )
        for conv in args.conv:
            for hd in args.head_dims:
                for direction in ["fwd"] + ([] if args.no_backward else ["bwd"]):
                    rows += measure_cell(name, conv, hd, direction, g, orders, info, args)
        torch.cuda.empty_cache()

    if args.json and rows:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=1)
        print(f"\nwrote {len(rows)} rows -> {args.json}")

    if rows:
        wins = sum(r["speedup"] > 1.0 for r in rows)
        print(f"\n{wins}/{len(rows)} configurations faster than {BASELINE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
