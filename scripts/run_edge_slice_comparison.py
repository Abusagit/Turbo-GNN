"""Three-arm comparison of the edge-sliced heavy path against baseline and previous best.

The heavy bucket used to get one thread block per node, which on a skewed graph means a grid of
a few thousand blocks whose runtimes differ by the whole degree spread. This measures what
replacing that with one block per fixed-size run of edges is worth, against two references:

    baseline       stock defaults -- one_per_block scheduling, sequential buckets
    previous best  the winning configuration from the autotune-concurrent sweep, replayed
    edge slice     that same configuration plus the new decomposition, slice size swept

Reporting the middle arm matters: without it a speedup over the untuned baseline would silently
bundle in everything the earlier tuning work already found.

    python scripts/run_edge_slice_comparison.py --gpu 1 --out reports/edge-slice
"""

from __future__ import annotations

import argparse
import glob
import inspect
import json
import statistics
import sys
from collections.abc import Callable
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
# The repo root must come first. A `pip install` of this package leaves a full copy (including
# _C.so) in site-packages, and a script launched as `python scripts/foo.py` gets sys.path[0] =
# scripts/, never the cwd -- so `import turbo_gnn` silently resolves to that stale copy and
# measures old kernels. Putting REPO first pins the import to the working tree.
sys.path.insert(0, str(REPO))
sys.path.insert(1, str(REPO / "scripts"))

from run_kernel_benchmark_matrix import GRAPHS  # noqa: E402

from turbo_gnn import gatv2_aggr, graph_transformer_aggr, reduction_aggr  # noqa: E402

BEST_DIR = REPO / "reports/autotune-concurrent"


def _assert_local_import() -> None:
    """Refuse to run against a stale installed copy -- the numbers would be meaningless."""
    import turbo_gnn

    got = Path(turbo_gnn.__file__).resolve()
    if REPO not in got.parents:
        raise SystemExit(f"turbo_gnn resolved to {got}, not the working tree at {REPO}")


def best_params(graph: str, conv: str, dim: int) -> dict:
    """Winning kernel_params from the autotune-concurrent sweep, if that cell was measured."""
    f = BEST_DIR / f"{graph}__{conv}__d{dim}__forward__autotuned-concurrent.json"
    if not f.exists():
        return {}
    params: dict = json.loads(f.read_text()).get("kernel_params", {})
    return params


def call(conv, g, n, dim, params):
    """Build a no-grad closure for one convolution, passing only parameters it accepts."""
    dev = torch.device("cuda")
    gen = torch.Generator(dev).manual_seed(0)
    fn = {"gt": graph_transformer_aggr, "gat_v2": gatv2_aggr, "min_aggr": reduction_aggr}[conv]
    accepted = set(inspect.signature(fn).parameters)
    # scale / negative_slope are supplied positionally below, so drop them from the recorded
    # parameters or the call gets two values for the same argument.
    kw = {k: v for k, v in params.items() if k in accepted and k not in {"scale", "negative_slope"}}

    if conv == "min_aggr":
        x = torch.randn(n, dim, device=dev, generator=gen)
        return lambda: fn(g, x, **kw)
    q = torch.randn(n, 1, dim, device=dev, generator=gen)
    k = torch.randn(n, 1, dim, device=dev, generator=gen)
    if conv == "gat_v2":
        a = torch.ones(1, dim, device=dev) * 0.05
        return lambda: fn(g, q, k, a, 0.2, **kw)
    v = torch.randn(n, 1, dim, device=dev, generator=gen)
    return lambda: fn(g, q, q, k, v, None, **kw)


def timed(fn: Callable[[], object], warm: int = 10, iters: int = 50) -> float:
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return float(s.elapsed_time(e)) / iters


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--graphs", nargs="+", default=[g for g, _ in GRAPHS])
    p.add_argument("--conv", nargs="+", default=["gt", "gat_v2", "min_aggr"])
    p.add_argument("--head-dims", type=int, nargs="+", default=[128, 256])
    p.add_argument("--slices", type=int, nargs="+", default=[128, 256, 512, 1024])
    p.add_argument("--out", default="reports/edge-slice")
    args = p.parse_args()

    _assert_local_import()
    torch.cuda.set_device(args.gpu)
    from benchmark_kernels import load_graph

    by_graph = dict(GRAPHS)
    rows = []
    print(
        f"{'graph':<15}{'conv':<9}{'dim':>4}{'baseline':>10}{'best':>9}{'sliced':>9}"
        f"{'vs base':>9}{'vs best':>9}  slice",
        flush=True,
    )

    for graph in args.graphs:
        if graph not in by_graph:
            continue
        ns = argparse.Namespace(
            dataset=by_graph[graph],
            num_nodes=0,
            avg_degree=10,
            quantile=0.99,
            index_dtype="int32",
            self_loops=True,
            node_order="natural",
        )
        try:
            bg = load_graph(ns, torch.device("cuda"), "cuda")
        except Exception as exc:
            print(f"  {graph}: SKIPPED ({type(exc).__name__})", file=sys.stderr)
            continue
        g, n = bg.repr, bg.num_nodes

        for conv in args.conv:
            for dim in args.head_dims:
                try:
                    with torch.no_grad():
                        t_base = timed(call(conv, g, n, dim, {}))
                        bp = best_params(graph, conv, dim)
                        t_best = timed(call(conv, g, n, dim, bp)) if bp else t_base
                        sliced = {}
                        for sl in args.slices:
                            sliced[sl] = timed(call(conv, g, n, dim, {**bp, "forward_heavy_edge_slice": sl}))
                except Exception as exc:
                    print(f"  {graph}/{conv}/{dim}: {type(exc).__name__}: {str(exc)[:70]}", file=sys.stderr)
                    continue
                bl, t_sl = min(sliced.items(), key=lambda kv: kv[1])
                rows.append(
                    {
                        "graph": graph,
                        "conv": conv,
                        "head_dim": dim,
                        "baseline_ms": t_base,
                        "best_ms": t_best,
                        "sliced_ms": t_sl,
                        "best_slice": bl,
                        "vs_baseline": t_base / t_sl,
                        "vs_best": t_best / t_sl,
                        "had_best_params": bool(bp),
                        "per_slice_ms": sliced,
                    }
                )
                print(
                    f"{graph:<15}{conv:<9}{dim:>4}{t_base:>9.3f}{t_best:>9.3f}{t_sl:>9.3f}"
                    f"{t_base / t_sl:>8.2f}x{t_best / t_sl:>8.2f}x  {bl}",
                    flush=True,
                )
        del bg, g
        torch.cuda.empty_cache()

    out = REPO / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "comparison.json").write_text(json.dumps(rows, indent=1))

    print(f"\n{len(rows)} cells -> {(out / 'comparison.json').relative_to(REPO)}")
    for conv in args.conv:
        sub = [r for r in rows if r["conv"] == conv]
        if sub:
            print(
                f"  {conv:<9} median vs best {statistics.median(r['vs_best'] for r in sub):.2f}x   "
                f"max {max(r['vs_best'] for r in sub):.2f}x   "
                f"regressions (<0.98x): {sum(r['vs_best'] < 0.98 for r in sub)}/{len(sub)}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
