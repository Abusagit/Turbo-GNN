import argparse
import json
import sys
from pathlib import Path
from typing import Any

import optuna
import torch
import yaml

sys.path.append("./")

from src.backends.registry import BackendRegistry
from src.benchmarking.microbench import MicrobenchResult, time_callable
from src.data.datasets import DatasetConfig, load_single_graph

doc = """
Kernel tuning microbenchmark launcher.

Loads a graph dataset and performs hyperparameter optimization for backend-specific
convolution kernels using Optuna. For each trial, creates a convolution layer with
suggested parameters, times the forward pass using CUDA events (or wall-clock on CPU),
and returns the configuration with the lowest execution time.

The parameter space is defined in a YAML config file where each parameter specifies:
- type: categorical, int, or float
- Type-specific fields (choices for categorical, low/high for int/float, etc.)

Uses Optuna's TPE (Tree-structured Parzen Estimator) sampler for Bayesian optimization
to efficiently explore the parameter space and find optimal kernel configurations.
"""


def parse_args() -> argparse.Namespace:
    """Parse CLI args.

    Returns:
        argparse.Namespace: Parsed args.
    """
    p = argparse.ArgumentParser(description="Microbenchmark graph conv layers.")
    p.add_argument("--conv_type", type=str, required=True, help="Convolution name (mean_aggr|sum_aggr|...).")
    p.add_argument("--backend", type=str, required=True, help="Backend name (cusparse|...).")
    p.add_argument("--dataset", type=str, required=True, help="Path to dataset YAML.")
    p.add_argument("--in-ch", type=int, default=128)
    p.add_argument("--amp", type=str, default="none", choices=["none", "bf16", "fp16"])
    p.add_argument(
        "--optuna-config",
        type=str,
        required=True,
        help="YAML file with parameter space config",
    )
    p.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Number of trials for optimization. Default: 100.",
    )
    p.add_argument("--json-out", type=str, default=None, help="Optional path to write JSON result.")
    return p.parse_args()


def main() -> int:
    """Entry: run the tuning.

    Returns:
        int: Exit code.
    """
    print("Starting kernel tuning...")
    sys.stdout.flush()
    args = parse_args()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    print(f"Using device: {device}")
    sys.stdout.flush()

    torch.set_default_device(device)

    with open(args.dataset, encoding="utf-8") as f:
        dataset_cfg_top_level = yaml.safe_load(f)
        dataset_cfg = dataset_cfg_top_level["dataset"]

    with open(args.optuna_config, encoding="utf-8") as f:
        param_config: dict[str, dict[str, Any]] = yaml.safe_load(f)
        kernel_parameters = param_config["parameters"]
        dataset_parameters = param_config.get("dataset_related_parameters", None)

    graph = load_single_graph(
        DatasetConfig(
            source=dataset_cfg["source"],
            name=dataset_cfg["name"],
            root=dataset_cfg["root"],
            conv_backend=args.backend,
        )
    )
    x = torch.randn(graph.num_nodes, args.in_ch, device=device)

    backend = BackendRegistry.get_backend(args.backend)

    amp_dtype = None
    if args.amp == "bf16":
        amp_dtype = torch.bfloat16
    elif args.amp == "fp16":
        amp_dtype = torch.float16

    def objective(trial: optuna.Trial) -> float:
        nonlocal graph
        """Optuna objective function that creates a conv with suggested params and times it."""
        cfg: dict[str, Any] = {}
        for param_name, param_spec in kernel_parameters.items():
            cfg[param_name] = getattr(trial, f"suggest_{param_spec['type']}")(
                param_name, **{k: v for k, v in param_spec.items() if k != "type"}
            )

        if dataset_parameters is not None:
            dataset_rebuild_cfg = {}
            for param_name, param_spec in dataset_parameters.items():
                dataset_rebuild_cfg[param_name] = getattr(trial, f"suggest_{param_spec['type']}")(
                    param_name, **{k: v for k, v in param_spec.items() if k != "type"}
                )
            graph = graph.update_graph_repr_with_new_hyperparameters(new_kernel_related_kwargs=dataset_rebuild_cfg)

        conv = backend.create_conv(args.conv_type, feature_dim=args.in_ch, **cfg)
        conv = conv.to(device)

        def _fn_forward() -> None:
            if amp_dtype is not None and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    _ = conv(x, graph.graph_repr)
            else:
                _ = conv(x, graph.graph_repr)

        torch.cuda.empty_cache()
        res: MicrobenchResult = time_callable(_fn_forward, warmup=1, iters=5)

        return res.ms_per_iter

    sampler = optuna.samplers.TPESampler(multivariate=True, group=True)
    study = optuna.create_study(
        direction="minimize",  # Minimize ms_per_iter
        sampler=sampler,
    )

    study.optimize(objective, n_trials=args.n_trials)

    output = {
        "best_config": study.best_params,
        "best_ms_per_iter": study.best_value,
    }

    print(json.dumps(output, indent=2))

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(output, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
