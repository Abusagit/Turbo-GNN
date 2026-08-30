"""Profile one (graph, head_dim, mode) cell under ncu, one launch per convolution.

Companion to `roofline_analysis.py`. That script models compulsory traffic because hardware
counters were unavailable; this one measures the traffic directly, so the model becomes a
claim to check rather than the foundation of the analysis.

All three convolutions run in one process so the graph is built once. Warmup runs outside the
profiled region -- `--profile-from-start off` plus `cudaProfilerStart` means ncu sees exactly
one steady-state iteration per convolution, with caches already warm. Attribution back to a
convolution is by kernel-name prefix, which is unambiguous: the three families are
`reduction_aggr_*`, `GATv2*`/`ReduceGradA*`, and `GraphAttention*`/`graph_attn_*`/`compute_D_mh*`.

Invoked by `run_roofline_counters.py`; not usually run by hand.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from roofline_analysis import build_call  # noqa: E402
from run_kernel_benchmark_matrix import GRAPHS  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--graph", required=True)
    p.add_argument("--head-dim", type=int, required=True)
    p.add_argument("--mode", required=True)
    p.add_argument("--heads", type=int, default=1)
    p.add_argument("--quantile", type=float, default=0.99)
    p.add_argument("--warmup", type=int, default=5)
    args = p.parse_args()

    from benchmark_kernels import load_graph

    dev = torch.device("cuda")
    cfg = dict(GRAPHS)[args.graph]
    ns = argparse.Namespace(
        dataset=cfg,
        num_nodes=0,
        avg_degree=10,
        quantile=args.quantile,
        index_dtype="int32",
        self_loops=True,
        node_order="natural",
    )
    bg = load_graph(ns, dev, "cuda")
    print(f"# graph={args.graph} nodes={bg.num_nodes} edges={bg.stats['num_edges']}", file=sys.stderr)

    for conv in ("min_aggr", "gat_v2", "gt"):
        try:
            call = build_call(conv, bg.repr, bg.num_nodes, args.head_dim, args.heads, args.mode, {})
        except Exception as exc:
            print(f"# {conv}: build failed {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        for _ in range(args.warmup):  # warm the caches before ncu starts counting
            call()
        torch.cuda.synchronize()
        torch.cuda.profiler.start()
        call()
        torch.cuda.synchronize()
        torch.cuda.profiler.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
