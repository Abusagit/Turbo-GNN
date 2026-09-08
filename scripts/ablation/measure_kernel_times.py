#!/usr/bin/env python3
"""Time the turbo_gnn kernels on this GPU, in the form the cost-model fit consumes.

:mod:`turbo_gnn.calibration` needs one number per (graph, conv, pass, head dim): the wall clock
of a launch over a whole graph, alongside that graph's node and edge counts.  This produces
exactly that, as JSON records interchangeable with ``benchmark_kernels.py --json-out``.

Measure and calibrate on the *same* GPU.  ``alpha`` is a ratio of two per-kernel constants and
does not travel between architectures, and neither does the tick: fitting on one card and
simulating another silently mixes two machines.

    python scripts/ablation/measure_kernel_times.py --out measurements.jsonl
    python scripts/ablation/calibrate_cost_model.py --from-json measurements.jsonl \
        --out cost_models.json

Timing uses CUDA events around a fixed number of launches, after a warmup, with the graph and
inputs resident.  It measures the kernel as the simulator models it -- one launch over the whole
node set -- so it excludes the host-side work a training step would also pay.
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

sys.path.append(str(Path(__file__).resolve().parent))

from simulate_load_imbalance import make_powerlaw_degrees  # noqa: E402

from src.data.datasets import DatasetConfig, load_single_graph  # noqa: E402
from turbo_gnn.graph import AdjacencyForwardBackwardWithNodeBuckets  # noqa: E402
from turbo_gnn.ops import gatv2_aggr, graph_transformer_aggr, reduction_aggr  # noqa: E402

DEFAULT_GRAPHS = ["Cora", "tolokers-2", "city-roads-L", "ogbn-arxiv", "twitch-views"]
CONVS = ["gt", "gat_v2", "min_aggr"]

# Off by default, and the reason is worth knowing before turning it on. Generated graphs were
# meant to fill out the regression's design space, since the real graphs to hand cluster around
# N ~ 150k. Measured, they cost 2-6 ns per edge against 0.36-1.55 for real graphs of every size
# -- and no locality window fixes it, nor does it track the L2 working set. A real graph's edge
# order is the product of how it was collected; a generated one has no such order, and the cost
# model has no term for the difference, so mixing the two put 60%+ median error into cells that
# fit to 15% on real graphs alone. Use these to explore, never to calibrate.
DEFAULT_SYNTHETIC: list[str] = []


def synthetic_edge_index(
    num_nodes: int, avg_degree: int, exponent: float, seed: int, locality_window: int
) -> torch.Tensor:
    """A power-law graph with the requested node count and average degree.

    In-degrees follow the power law -- they are what the forward CSR rows are. Sources are drawn
    from a window around the destination rather than uniformly, and that is not cosmetic.
    With uniform sources every neighbour fetch misses L2, and these graphs measured 3 to 13
    times more nanoseconds per edge than real graphs of the same size -- a difference the cost
    model cannot express, so it lands wholly in the residual and wrecks the fit. A window
    reproduces the locality a real graph gets from being stored in a sensible order.
    """
    degrees = torch.as_tensor(make_powerlaw_degrees(num_nodes, avg_degree, exponent, seed))
    destinations = torch.repeat_interleave(torch.arange(num_nodes), degrees)
    generator = torch.Generator().manual_seed(seed)
    window = min(locality_window, num_nodes)
    offsets = torch.randint(-(window // 2), window // 2 + 1, (int(degrees.sum()),), generator=generator)
    sources = (destinations + offsets) % num_nodes
    return torch.stack([sources, destinations])


def build_graph(
    name: str, root: str, quantile: float, device: torch.device, exponent: float, seed: int, locality_window: int
):
    if ":" in name:
        num_nodes, avg_degree = (int(part) for part in name.split(":", 1))
        edge_index = synthetic_edge_index(num_nodes, avg_degree, exponent, seed, locality_window)
    else:
        sample = load_single_graph(DatasetConfig(source="auto", name=name, root=root, conv_backend="cuda"))
        num_nodes = int(sample.num_nodes)
        edge_index = sample.edge_index
        if not isinstance(edge_index, torch.Tensor):
            edge_index = torch.as_tensor(edge_index)
    graph = AdjacencyForwardBackwardWithNodeBuckets.from_edge_list(
        edge_index.to(device), num_nodes, quantile=quantile, index_dtype=torch.int32
    )
    return graph, num_nodes, int(graph.forward_indices.numel())


def make_call(conv: str, graph, num_nodes: int, heads: int, head_dim: int, device: torch.device, requires_grad: bool):
    """A zero-argument callable running one launch of ``conv``, plus the tensors it needs alive."""
    feature_dim = heads * head_dim
    kwargs = {"device": device, "dtype": torch.float32, "requires_grad": requires_grad}
    if conv == "gt":
        x = torch.randn(num_nodes, feature_dim, **kwargs)
        q, k, v = (torch.randn(num_nodes, heads, head_dim, **kwargs) for _ in range(3))
        return lambda: graph_transformer_aggr(graph, x, q, k, v, head_dim**-0.5), [q, k, v]
    if conv == "gat_v2":
        x = torch.randn(num_nodes, heads, head_dim, **kwargs)
        neighbours = torch.randn(num_nodes, heads, head_dim, **kwargs)
        attention = torch.randn(heads, head_dim, **kwargs)
        return lambda: gatv2_aggr(graph, x, neighbours, attention), [x, neighbours, attention]
    if conv == "min_aggr":
        x = torch.randn(num_nodes, feature_dim, **kwargs)
        return lambda: reduction_aggr(graph, x, reduce="min"), [x]
    raise ValueError(f"unknown conv: {conv}")


def time_ms(call, backward: bool, warmup: int, iters: int) -> float:
    """Milliseconds per launch, measured with CUDA events.

    The seed gradient is materialised rather than taken from ``out.sum().backward()``: that
    route hands the kernel an expanded view with a zero stride, which the GATv2 backward
    rejects outright.
    """
    seed = torch.ones_like(call()) if backward else None

    def once() -> None:
        out = call()
        if backward:
            out.backward(seed)

    for _ in range(warmup):
        once()
    torch.cuda.synchronize()

    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        once()
    stop.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(stop)) / iters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--graphs", nargs="*", default=DEFAULT_GRAPHS)
    parser.add_argument(
        "--synthetic",
        nargs="*",
        default=DEFAULT_SYNTHETIC,
        metavar="NODES:AVG_DEGREE",
        help="Power-law graphs generated to spread the regression's design points",
    )
    parser.add_argument("--exponent", type=float, default=2.3, help="Power-law exponent for --synthetic")
    parser.add_argument(
        "--locality-window",
        type=int,
        default=4096,
        help="Synthetic edges connect nodes within this index distance; approximates a well-ordered graph",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--convs", nargs="+", choices=CONVS, default=CONVS)
    parser.add_argument("--head-dims", type=int, nargs="+", default=[128, 256])
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--passes", nargs="+", choices=["forward", "backward"], default=["forward", "backward"])
    parser.add_argument("--quantile", type=float, default=0.99)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("measurements.jsonl"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        print("no CUDA device; this script measures real kernels and cannot run on CPU", file=sys.stderr)
        return 1
    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    gpu = torch.cuda.get_device_name(device)
    print(f"measuring on {gpu}\n")

    records = []
    unavailable: list[str] = []
    for graph_name in [*args.graphs, *args.synthetic]:
        try:
            graph, num_nodes, num_edges = build_graph(
                graph_name, args.data_root, args.quantile, device, args.exponent, args.seed, args.locality_window
            )
        except Exception as error:  # noqa: BLE001 - a graph that will not load must not end the run
            # One unreachable download should not cost the whole sweep. The loader's "auto"
            # source falls back to DGL when PyG fails, so the exception that surfaces is often
            # a missing dgl module rather than the download error underneath it; report the
            # chain so the real cause is visible.
            causes: list[str] = []
            current: BaseException | None = error
            while current is not None and len(causes) < 3:
                causes.append(f"{type(current).__name__}: {str(current)[:110]}")
                current = current.__context__ if current.__cause__ is None else current.__cause__
            print(f"{graph_name}: unavailable -- {' <- '.join(causes)}\n", file=sys.stderr)
            unavailable.append(graph_name)
            continue
        print(f"{graph_name}: N={num_nodes:,} E={num_edges:,} heavy={graph.forward_heavy_nodes.numel():,}")
        for conv, head_dim, pass_name in product(args.convs, args.head_dims, args.passes):
            backward = pass_name == "backward"
            try:
                call, _tensors = make_call(conv, graph, num_nodes, args.heads, head_dim, device, backward)
                elapsed = time_ms(call, backward, args.warmup, args.iters)
            except (RuntimeError, torch.cuda.OutOfMemoryError) as error:
                print(f"    {conv:<9} d{head_dim:<4} {pass_name:<9} skipped: {str(error)[:70]}")
                torch.cuda.empty_cache()
                continue
            records.append(
                {
                    "conv": conv,
                    "mode": pass_name,
                    "head_dim": head_dim,
                    "heads": args.heads,
                    "dataset": graph_name,
                    "graph": {"num_nodes": num_nodes, "num_edges": num_edges},
                    "ms_per_iter": elapsed,
                    "gpu": gpu,
                }
            )
            print(f"    {conv:<9} d{head_dim:<4} {pass_name:<9} {elapsed:10.4f} ms")
            torch.cuda.empty_cache()
        del graph
        torch.cuda.empty_cache()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(record) + "\n" for record in records))
    print(f"\nwrote {len(records)} measurements to {args.out}")
    if unavailable:
        print(f"could not load: {', '.join(unavailable)}", file=sys.stderr)
    return 0 if records else 1


if __name__ == "__main__":
    raise SystemExit(main())
