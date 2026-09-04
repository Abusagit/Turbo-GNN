"""Regression check: does routing GATv2 forward through self.kernel (instead of
calling the free gatv2_aggr() function directly) cost anything, and does the
offline autotuner (conv.autotune()) actually find a config that is no worse
than the untuned default -- now that it's correctly wired to pipeline_stages?

Three things are measured:

  A) gatv2_aggr() called directly, on pre-projected inputs (the old,
     pre-fix call path)
  B) conv.kernel(...) called directly on the same pre-projected inputs,
     default params (the new delegation path) -- A vs B isolates the cost
     of the TunableKernel.__call__ indirection itself.
  C) conv(x, graph) before vs after conv.autotune(x, graph_sample) (grid
     search incl. pipeline_stages in {0,1,2,4}) -- shows whether autotuning
     finds an equal-or-better config for the full module.

Usage:
    python dev/sinfillo/bench_autotune_regression.py --num-nodes 50000 --avg-degree 16 \
        --feature-dim 128 --heads 4 --tune-backward
"""

import argparse
import sys

import torch

sys.path.append("./")

from src.backends.registry import BackendRegistry
from src.benchmarking.microbench import time_callable
from src.data.datasets import MODEL_BACKEND_TO_GRAPH_REPR, GraphSample
from turbo_gnn._autotune import AutotuneConfig
from turbo_gnn.ops import gatv2_aggr


def _make_random_graph(num_nodes: int, avg_degree: int, device: torch.device):
    E = max(1, num_nodes * max(1, avg_degree))
    src = torch.randint(0, num_nodes, (E,), device=device, dtype=torch.long)
    dst = torch.randint(0, num_nodes, (E,), device=device, dtype=torch.long)
    return torch.stack([src, dst], dim=0), None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GATv2 autotuner regression check")
    p.add_argument("--num-nodes", type=int, default=50_000)
    p.add_argument("--avg-degree", type=int, default=16)
    p.add_argument("--feature-dim", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--tune-warmup", type=int, default=5)
    p.add_argument("--tune-iters", type=int, default=15)
    p.add_argument("--tune-backward", action="store_true", help="Also grid-search + time the backward pass.")
    p.add_argument(
        "--regression-tol",
        type=float,
        default=1.05,
        help="A run is flagged as a regression if slower than its baseline by more "
        "than this factor (guards against timing noise).",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        print("CUDA is not available; this check requires a GPU.")
        return 1

    device = torch.device("cuda", 0)
    torch.set_default_device(device)
    torch.manual_seed(args.seed)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"N={args.num_nodes} avg_degree={args.avg_degree} feature_dim={args.feature_dim} "
        f"heads={args.heads}\n"
    )

    edge_index, edge_weight = _make_random_graph(args.num_nodes, args.avg_degree, device=device)
    x = torch.randn(args.num_nodes, args.feature_dim, device=device, requires_grad=True)

    graph_sample = GraphSample(
        backend=MODEL_BACKEND_TO_GRAPH_REPR["cuda"],
        x=x,
        y=torch.zeros(args.num_nodes, device=device),
        edge_index=edge_index,
        edge_weight=edge_weight,
    )

    backend = BackendRegistry.get_backend("cuda")
    conv = backend.create_conv("gat_v2", feature_dim=args.feature_dim, heads=args.heads, bias=False).to(device)

    def _bench(fn, warmup, iters):
        return time_callable(fn, warmup=warmup, iters=iters, do_memory_profile=False)

    verdict_bits: list[bool] = []

    # =========================================================================
    # Part 1: untuned default, full conv(x, graph) -- the baseline everyone cares about.
    # Captured *before* autotune() touches anything (graph_repr still pristine).
    # =========================================================================
    graph = graph_sample.graph_repr
    default_kwargs = {p.name: p.default for p in conv.get_tunable_forward_kernel_params()}

    out_default = conv(x, graph)
    res_default = _bench(lambda: conv(x, graph), args.warmup, args.iters)

    # =========================================================================
    # Part 2 (A vs B): does routing through self.kernel cost anything, isolated
    # from the surrounding nn.Linear projections?
    # =========================================================================
    with torch.no_grad():
        x_left, x_right = conv.left_right_projection(x).split(conv.heads * conv.head_dim, -1)
        x_left = x_left.view(-1, conv.heads, conv.head_dim)
        x_right = x_right.view(-1, conv.heads, conv.head_dim)

    def _call_free():
        return gatv2_aggr(graph, x_left, x_right, conv.attn_weights.data, conv.negative_slope)

    def _call_kernel():
        return conv.kernel(
            graph,
            x_left,
            x_neighbors=x_right,
            attention_weights=conv.attn_weights.data,
            negative_slope=conv.negative_slope,
        )

    out_a = _call_free()
    res_a = _bench(_call_free, args.warmup, args.iters)
    out_b = _call_kernel()
    res_b = _bench(_call_kernel, args.warmup, args.iters)

    max_diff_ab = (out_a.float() - out_b.float()).abs().max().item()
    ab_ratio = res_b.ms_per_iter / res_a.ms_per_iter
    ab_ok = ab_ratio <= args.regression_tol
    verdict_bits.append(ab_ok)

    print("=== A vs B: does routing through self.kernel cost anything? ===")
    print(f"A) gatv2_aggr() direct:      {res_a.ms_per_iter:.4f} ms  (default params)")
    print(f"B) conv.kernel(...) direct:  {res_b.ms_per_iter:.4f} ms  (default params, same call otherwise)")
    print(f"   max output diff: {max_diff_ab:.2e}  (should be ~0, both use identical default params)")
    print(f"   B/A ratio: {ab_ratio:.3f}x  ({'OK' if ab_ok else 'REGRESSION'})\n")

    # =========================================================================
    # Part 3: run the offline autotuner (may repartition graph_sample and will
    # apply the best kernel config to `conv` as a side effect).
    # =========================================================================
    print("=== Running conv.autotune() (grid search incl. pipeline_stages in {0,1,2,4}) ===")
    tune_config = AutotuneConfig(
        warmup=args.tune_warmup,
        iters=args.tune_iters,
        tune_backward=args.tune_backward,
        cache_dir=None,
    )
    best_config = conv.autotune(x, graph_sample, config=tune_config)
    print(f"Best config found: {best_config}\n")

    # graph_repr may have been rebuilt if a graph param (e.g. the light/heavy
    # partition threshold) was retuned -- always re-fetch after autotune().
    graph = graph_sample.graph_repr

    # =========================================================================
    # Part 4: default vs tuned, full conv(x, graph) forward.
    # `conv` is already left in the tuned state by autotune(), so just measure it.
    # =========================================================================
    out_tuned = conv(x, graph)
    res_tuned = _bench(lambda: conv(x, graph), args.warmup, args.iters)

    max_diff_fwd = (out_default.float() - out_tuned.float()).abs().max().item()
    fwd_speedup = res_default.ms_per_iter / res_tuned.ms_per_iter
    fwd_ok = res_tuned.ms_per_iter <= res_default.ms_per_iter * args.regression_tol
    verdict_bits.append(fwd_ok)

    print("=== Untuned-default vs tuned, full conv(x, graph) forward ===")
    print(f"Default config:  {default_kwargs}")
    print(f"Tuned config:    {best_config}")
    print(f"Default:  {res_default.ms_per_iter:.4f} ms")
    print(f"Tuned:    {res_tuned.ms_per_iter:.4f} ms")
    print(f"Speedup:  {fwd_speedup:.3f}x  ({'OK, no regression' if fwd_ok else 'REGRESSION'})")
    print(f"Max output diff (default vs tuned): {max_diff_fwd:.2e}")

    # =========================================================================
    # Part 5 (optional): default vs tuned, forward+backward.
    # =========================================================================
    if args.tune_backward:
        print("\n=== Untuned-default vs tuned, forward+backward ===")

        def _fwd_bwd(cfg):
            conv.configure(**{k: v for k, v in cfg.items() if k in default_kwargs})
            x_local = x.detach().clone().requires_grad_(True)

            def _bench_fn():
                out_i = conv(x_local, graph)
                out_i.backward(torch.ones_like(out_i))

            return _bench(_bench_fn, args.warmup, args.iters)

        res_default_bwd = _fwd_bwd(default_kwargs)
        res_tuned_bwd = _fwd_bwd(best_config)
        bwd_speedup = res_default_bwd.ms_per_iter / res_tuned_bwd.ms_per_iter
        bwd_ok = res_tuned_bwd.ms_per_iter <= res_default_bwd.ms_per_iter * args.regression_tol
        verdict_bits.append(bwd_ok)

        print(f"Default fwd+bwd: {res_default_bwd.ms_per_iter:.4f} ms")
        print(f"Tuned fwd+bwd:   {res_tuned_bwd.ms_per_iter:.4f} ms")
        print(f"Speedup:  {bwd_speedup:.3f}x  ({'OK, no regression' if bwd_ok else 'REGRESSION'})")

        # leave conv in the tuned state, matching the rest of the report
        conv.configure(**{k: v for k, v in best_config.items() if k in default_kwargs})

    print("\n" + "=" * 60)
    if all(verdict_bits):
        print("PASS: no regression detected.")
        return 0
    print("FAIL: at least one comparison regressed beyond tolerance -- see above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
