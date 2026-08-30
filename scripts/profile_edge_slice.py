"""Profile one cell under ncu with the heavy bucket decomposed either way.

Runs a single steady-state iteration with `heavy_edge_slice` off and on, so counters attribute
to the kernels each decomposition actually launches. Warmup sits outside the profiled region via
cudaProfilerStart, so caches are warm and no warmup launch is counted.

Kernel parameters come from a completed autotune cell (`--params-json`), so the comparison is
argmax-vs-argmax rather than default-vs-default -- otherwise the old path is measured in a
configuration nobody would ship.

Invoked by `run_edge_slice_ncu.py`; counter access needs root on this driver.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(1, str(REPO / "scripts"))

from run_kernel_benchmark_matrix import GRAPHS  # noqa: E402

from turbo_gnn import gatv2_aggr, graph_transformer_aggr, reduction_aggr  # noqa: E402

SLICE_KEYS = ("forward_heavy_edge_slice", "backward_heavy_edge_slice")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--graph", required=True)
    p.add_argument("--conv", default="gt")
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--mode", default="forward", choices=["forward", "backward"])
    p.add_argument("--slice", type=int, required=True, help="0 = node-per-block, >0 = edge-sliced")
    p.add_argument("--params-json", default=None, help="a completed autotune cell to take argmax params from")
    p.add_argument("--warmup", type=int, default=5)
    args = p.parse_args()

    from benchmark_kernels import load_graph

    dev = torch.device("cuda")
    ns = argparse.Namespace(
        dataset=dict(GRAPHS)[args.graph],
        num_nodes=0,
        avg_degree=10,
        quantile=0.99,
        index_dtype="int32",
        self_loops=True,
        node_order="natural",
    )
    bg = load_graph(ns, dev, "cuda")
    g, n = bg.repr, bg.num_nodes

    fn = {"gt": graph_transformer_aggr, "gat_v2": gatv2_aggr, "min_aggr": reduction_aggr}[args.conv]
    accepted = set(inspect.signature(fn).parameters)
    kw: dict = {}
    if args.params_json and Path(args.params_json).exists():
        rec = json.loads(Path(args.params_json).read_text())
        # autotune_selected holds what the search actually picked; kernel_params is the CLI echo.
        sel = rec.get("autotune_selected", {}).get("kernel_config", {})
        for k, v in {**rec.get("kernel_params", {}), **sel}.items():
            if k in accepted and k not in {"scale", "negative_slope"} and k not in SLICE_KEYS:
                kw[k] = v
    # The slice under test overrides whatever the cell recorded.
    key = "forward_heavy_edge_slice" if args.mode == "forward" else "backward_heavy_edge_slice"
    if key in accepted:
        kw[key] = args.slice

    print(
        f"# {args.graph} {args.conv} d{args.head_dim} {args.mode} slice={args.slice} "
        f"heavy={g.forward_heavy_nodes.numel()} edges={bg.stats['num_edges']} params={kw}",
        file=sys.stderr,
    )

    gen = torch.Generator(dev).manual_seed(0)
    need_grad = args.mode == "backward"

    def rnd(*shape):
        return torch.randn(*shape, device=dev, generator=gen, requires_grad=need_grad)

    if args.conv == "min_aggr":
        x = rnd(n, args.head_dim)
        call = lambda: fn(g, x, **kw)  # noqa: E731
    elif args.conv == "gat_v2":
        q, k = rnd(n, 1, args.head_dim), rnd(n, 1, args.head_dim)
        a = torch.ones(1, args.head_dim, device=dev) * 0.05
        call = lambda: fn(g, q, k, a, 0.2, **kw)  # noqa: E731
    else:
        q, k, v = rnd(n, 1, args.head_dim), rnd(n, 1, args.head_dim), rnd(n, 1, args.head_dim)
        call = lambda: fn(g, q, q, k, v, None, **kw)  # noqa: E731

    if args.mode == "forward":
        with torch.no_grad():
            for _ in range(args.warmup):
                call()
            torch.cuda.synchronize()
            torch.cuda.profiler.start()
            call()
            torch.cuda.synchronize()
            torch.cuda.profiler.stop()
    else:
        out = call()
        grad = torch.ones_like(out)
        for _ in range(args.warmup):
            out.backward(grad, retain_graph=True)
        torch.cuda.synchronize()
        torch.cuda.profiler.start()
        out.backward(grad, retain_graph=True)
        torch.cuda.synchronize()
        torch.cuda.profiler.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
