import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Some dataset loaders (e.g. ogb's NodePropPredDataset) call torch.load() on
# their own locally-cached preprocessed files without weights_only=False;
# PyTorch >=2.6 defaults weights_only=True, which breaks loading those trusted local caches.
_orig_torch_load = torch.load


def _torch_load_weights_only_false(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)


torch.load = _torch_load_weights_only_false

from src.backends.registry import BackendRegistry
from src.benchmarking.microbench import MicrobenchResult, get_gpu_info, time_callable
from src.data.datasets import MODEL_BACKEND_TO_GRAPH_REPR, DatasetConfig, GraphSample, load_single_graph

doc = """
Layer microbenchmark launcher.

Creates a random graph and features, instantiates a backend convolution, and
times forward/backward kernel using CUDA events (or wall-clock on CPU).
"""


def _make_random_graph(
    num_nodes: int, avg_degree: int, *, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Generate an Erdos-Renyi-like random edge_index with approx avg_degree.

    Args:
        num_nodes (int): Number of nodes.
        avg_degree (int): Approximate average out-degree.
        device (torch.device): Torch device.

    Returns:
        Tuple[torch.Tensor, Optional[torch.Tensor]]: (edge_index [2,E], edge_weight or None)
    """
    E = max(1, num_nodes * max(1, avg_degree))
    src = torch.randint(0, num_nodes, (E,), device=device, dtype=torch.long)
    dst = torch.randint(0, num_nodes, (E,), device=device, dtype=torch.long)
    edge_index = torch.stack([src, dst], dim=0)
    return edge_index, None


def _collect_kernel_params(conv) -> dict:
    kernel = getattr(conv, "kernel", conv)

    params = {}
    for getter in (
        "get_tunable_forward_kernel_params",
        "get_tunable_backward_kernel_params",
        "get_tunable_forward_graph_params",
        "get_tunable_backward_graph_params",
    ):
        fn = getattr(kernel, getter, None)
        if fn is None:
            continue
        for p in fn():
            params[p.name] = getattr(kernel, p.name, None)
    return params


def parse_args() -> argparse.Namespace:
    """Parse CLI args.

    Returns:
        argparse.Namespace: Parsed args.
    """
    p = argparse.ArgumentParser(description="Microbenchmark graph conv layers.")
    p.add_argument(
        "--layer",
        type=str,
        required=True,
        help="Layer/op name; may be a comma-separated list to benchmark several ops in one process "
        "(e.g. 'u_add_v,e_dot_v,copy_v' with --aggr --backend dgl).",
    )
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--backend", type=str, required=True, help="Backend name (pyg|dgl|...).")
    p.add_argument(
        "--aggr",
        action="store_true",
        help="Use backend.create_aggr (aggregation-only, no projections) instead of create_conv. "
        "For the dgl backend any raw dgl.ops gspmm/gsddmm op name (u_add_v, e_dot_v, copy_v, "
        "u_mul_e_sum, ...) is launched directly; node/edge operands are generated automatically.",
    )
    p.add_argument(
        "--dataset",
        type=str,
        help="Path to dataset YAML. If not presented, graph with `--num-nodes` and `--avg-degree` will be "
        "generated for the benchmark",
    )
    p.add_argument("--num-nodes", type=int, default=20000)
    p.add_argument("--avg-degree", type=int, default=10)
    p.add_argument("--feature_dim", type=int, default=128)
    p.add_argument("--heads", type=int, default=1)
    p.add_argument("--mode", type=str, default="forward", choices=["forward", "backward"])
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--amp", type=str, default="none", choices=["none", "bf16", "fp16"])
    p.add_argument(
        "--exact-iters",
        action="store_true",
        help="Issue exactly --warmup + --iters calls instead of using triton.testing.do_bench "
        "(whose warmup/rep are milliseconds and which adds 6 calibration calls). Use this under "
        "a kernel profiler such as ncu to keep the launch count small and deterministic.",
    )
    p.add_argument("--json-out", type=str, default=None, help="Optional path to write JSON result.")
    p.add_argument(
        "--csv-out",
        type=str,
        default=None,
        help="Optional path to append results to as CSV (header written on first use; "
        "one row per benchmarked layer/op).",
    )
    p.add_argument(
        "--pipeline-stages",
        type=int,
        default=0,
        help="Forward async-copy pipeline stage count for CUDA convs (0 disables the pipeline). "
        "Ignored when --autotune is set.",
    )
    p.add_argument(
        "--backward-pipeline-stages",
        type=int,
        default=0,
        help="Backward async-copy pipeline stage count for CUDA convs (0 disables the pipeline). "
        "Ignored when --autotune is set.",
    )
    p.add_argument(
        "--autotune",
        action="store_true",
        help="Autotune kernel/graph params (incl. pipeline_stages) via grid search "
        "instead of using --pipeline-stages manually.",
    )
    p.add_argument("--autotune-warmup", type=int, default=5, help="Warmup iters for each autotune grid-search trial.")
    p.add_argument("--autotune-iters", type=int, default=15, help="Timed iters for each autotune grid-search trial.")
    return p.parse_args()


def _build_aggr_operands(
    aggr: torch.nn.Module,
    num_nodes: int,
    num_edges: int,
    feature_dim: int,
    device: torch.device,
) -> list[torch.Tensor]:
    """Create input operands for an aggregation based on its ``operand_kinds``.

    Kind "u"/"v" produces node features [num_nodes, feature_dim], kind "e"
    produces edge features [num_edges, feature_dim]. Aggrs without an
    ``operand_kinds`` attribute default to a single node-feature operand.
    All operands require grad so backward-mode timing works.

    Args:
        aggr (torch.nn.Module): Aggregation module (may define operand_kinds).
        num_nodes (int): Number of graph nodes.
        num_edges (int): Number of graph edges (of the actual graph object).
        feature_dim (int): Feature width.
        device (torch.device): Torch device.

    Returns:
        list[torch.Tensor]: Operand tensors in the order the aggr expects them.
    """
    operands = []
    for kind in getattr(aggr, "operand_kinds", ("u",)):
        n = num_edges if kind == "e" else num_nodes
        operands.append(torch.randn(n, feature_dim, device=device, requires_grad=True))
    return operands


def _append_csv_rows(csv_path: str, rows: list[dict]) -> None:
    """Append result rows to a CSV file, writing the header on first use.

    Nested dict/list values are JSON-encoded so every row stays flat.

    Args:
        csv_path (str): Destination CSV path (created with parents if needed).
        rows (list[dict]): Rows to append; keys form the column set.
    """
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    flat_rows = [{k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in row.items()} for row in rows]
    fieldnames: list[str] = []
    for row in flat_rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(flat_rows)


def main() -> int:
    """Entry: run the microbenchmark.

    Returns:
        int: Exit code.
    """
    args = parse_args()
    device = torch.device("cuda", args.device) if torch.cuda.is_available() else torch.device("cpu")
    torch.set_default_device(device)

    # graph + features
    if args.dataset is None:
        edge_index, edge_weight = _make_random_graph(
            args.num_nodes,
            args.avg_degree,
            device=device,
        )

        x = torch.randn(
            args.num_nodes,
            args.feature_dim,
            device=device,
            requires_grad=True,
        )

        sample = GraphSample(
            backend=MODEL_BACKEND_TO_GRAPH_REPR[args.backend],
            x=x,
            y=torch.zeros(args.num_nodes, device=device),
            edge_index=edge_index,
            edge_weight=edge_weight,
        )

        dataset_name = "random"

    else:
        with open(args.dataset, encoding="utf-8") as f:
            dataset_cfg_top_level = yaml.safe_load(f)

        dataset_cfg = dataset_cfg_top_level["dataset"]

        sample = load_single_graph(
            DatasetConfig(
                source=dataset_cfg["source"],
                name=dataset_cfg["name"],
                root=dataset_cfg.get("root", "data"),
                conv_backend=args.backend,
                allow_random_split=dataset_cfg.get("allow_random_split", False),
                kernel_related_kwargs=dataset_cfg.get(
                    "kernel_related_kwargs",
                    {},
                ),
            )
        )

        x = torch.randn(
            sample.num_nodes,
            args.feature_dim,
            device=device,
            requires_grad=True,
        )

        dataset_name = dataset_cfg["name"]

    graph = sample.graph_repr
    num_nodes = sample.num_nodes
    # Edge operands must match the edge count of the actual graph object:
    # e.g. DGL graphs get self-loops added on top of the raw edge_index.
    num_edges = int(graph.num_edges()) if hasattr(graph, "num_edges") else sample.num_edges
    head_dim = args.feature_dim

    layers = [s.strip() for s in args.layer.split(",") if s.strip()]
    backend = BackendRegistry.get_backend(args.backend)

    # measure function
    amp_dtype = None
    if args.amp == "bf16":
        amp_dtype = torch.bfloat16
    elif args.amp == "fp16":
        amp_dtype = torch.float16

    rows: list[dict] = []
    for layer_idx, layer in enumerate(layers, start=1):
        # conv / aggr
        if args.aggr:
            conv = backend.create_aggr(layer, feature_dim=args.feature_dim, heads=args.heads)
        elif layer not in {"gat_v2", "gt", "gat_v1"}:
            conv = backend.create_conv(layer, feature_dim=args.feature_dim)
        else:
            conv = backend.create_conv(layer, feature_dim=args.feature_dim, heads=args.heads)

        conv = conv.to(device)
        autotuned_config: dict = {}
        if args.backend == "cuda" and not args.aggr:
            if args.autotune:
                from turbo_gnn._autotune import AutotuneConfig

                tune_cfg = AutotuneConfig(
                    warmup=args.autotune_warmup,
                    iters=args.autotune_iters,
                    tune_backward=(args.mode == "backward"),
                    cache_dir=None,
                )
                autotuned_config = conv.autotune(x, sample, config=tune_cfg)
                graph = sample.graph_repr
            else:
                conv.configure(
                    forward_pipeline_stages=args.pipeline_stages,
                    backward_pipeline_stages=args.backward_pipeline_stages,
                )

        operands = _build_aggr_operands(conv, num_nodes, num_edges, args.feature_dim, device) if args.aggr else None

        def _fn_forward() -> torch.Tensor:
            if amp_dtype is not None and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    out = conv(*operands, graph) if operands is not None else conv(x, graph)
            else:
                out = conv(*operands, graph) if operands is not None else conv(x, graph)
            return out

        Y = None
        grad_output = None
        if args.mode == "backward" or not args.exact_iters:
            Y = _fn_forward().requires_grad_(True)
            # the aggr output may be edge-shaped ([E, d]) or reduced ([E, 1] for
            # dot ops), so the grad seed must follow the output, not x
            grad_output = torch.randn_like(Y)

        def _fn_backward() -> None:
            nonlocal grad_output, Y
            Y.backward(grad_output, retain_graph=True)  # type: ignore[union-attr]

        fn = _fn_forward if args.mode == "forward" else _fn_backward
        res: MicrobenchResult = time_callable(
            fn,
            warmup=args.warmup,
            iters=args.iters,
            do_memory_profile=False,
            exact_iters=args.exact_iters,
        )

        base_dict = {
            "backend": args.backend,
            "conv_type": layer,
            "dataset": dataset_name,
            "dataset_config": args.dataset,
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "feature_dim": args.feature_dim,
            "heads": args.heads,
            "head_dim": head_dim,
            "amp": args.amp,
            "aggr": args.aggr,
            "mode": args.mode,
            "autotuned": args.autotune,
            "autotuned_config": autotuned_config,
            "forward_pipeline_stages": autotuned_config.get("forward_pipeline_stages", args.pipeline_stages),
            "backward_pipeline_stages": autotuned_config.get("backward_pipeline_stages", args.backward_pipeline_stages),
            # The stage count actually swept for this run's --mode (what plot_pipeline_results.py
            # groups by): forward stages in forward mode, backward stages in backward mode.
            "pipeline_stages": (
                autotuned_config.get("forward_pipeline_stages", args.pipeline_stages)
                if args.mode == "forward"
                else autotuned_config.get("backward_pipeline_stages", args.backward_pipeline_stages)
            ),
            "kernel_params": _collect_kernel_params(conv),
            "iters": res.iters,
            "ms_per_iter": res.ms_per_iter,
            "device": res.device,
            "memory": res.memory_allocated,
        } | get_gpu_info(device)

        rows.append(base_dict)
        print(f"[{layer_idx}/{len(layers)}] {layer}: {res.ms_per_iter:.4f} ms/iter", flush=True)

        del conv, operands, Y, grad_output
        torch.cuda.empty_cache()

    payload = rows[0] if len(rows) == 1 else rows
    print(json.dumps(payload, indent=4))

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(payload, indent=4))

    if args.csv_out:
        _append_csv_rows(args.csv_out, rows)
        print(f"appended {len(rows)} row(s) to {args.csv_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
