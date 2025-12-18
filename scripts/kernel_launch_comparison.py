import argparse
import concurrent.futures
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from json import dumps
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import torch
import yaml
from dotenv import load_dotenv

try:
    import comet_ml
except ImportError:
    comet_ml = None

sys.path.append("./")

from src.backends.registry import BackendRegistry
from src.benchmarking.microbench import MicrobenchResult, get_gpu_info, time_callable
from src.data.datasets import DatasetConfig, load_single_graph

doc = """
Kernel large comparison script launching several microbenchmarks for chosen datasetsand backend.
Uses specified convolution with the combinations of hidden dims

Logs results to Comet ML
"""


DEVICE = None
COMET_WORKSPACE = "None"
COMET_PROJECT_NAME = "None"
COMET_EXP_NAME = ""


def parse_args() -> argparse.Namespace:
    """Parse CLI args.

    Returns:
        argparse.Namespace: Parsed args.
    """
    global DEVICE, COMET_WORKSPACE, COMET_EXP_NAME, COMET_PROJECT_NAME
    load_dotenv()

    p = argparse.ArgumentParser(description="Multi-backend convolution measurements")

    p.add_argument(
        "--device",
        type=int,
        default=0,
        help="Device index",
    )

    p.add_argument(
        "--conv_params_grid",
        type=str,
        help="Path to configs for convolution parameters",
    )

    p.add_argument(
        "--conv_type",
        type=str,
        required=True,
        help="Convolution name (mean_aggr|sum_aggr|...).",
    )

    p.add_argument(
        "--backends",
        type=str,
        required=True,
        nargs="+",
        help="Backends names for the convolution",
    )

    p.add_argument(
        "--target_backend",
        type=str,
        required=True,
        default="cuda",
        help="Target backend against which to compare other backends",
    )

    p.add_argument(
        "--datasets",
        type=str,
        required=True,
        nargs="+",
        help="Paths to dataset YAML.",
    )

    p.add_argument(
        "--amp",
        type=str,
        default="none",
        choices=["none", "bf16", "fp16"],
    )

    p.add_argument(
        "--use_comet",
        action="store_true",
    )

    p.add_argument(
        "--comet_project_name",
        type=str,
        default="kernel_results",  # TODO
    )

    p.add_argument(
        "--comet_workspace",
        type=str,
        default="accelerating-gnns-2",  # TODO
    )

    p.add_argument(
        "--comet_experiment_name_prefix",
        type=str,
        default="",
    )

    p.add_argument("--out", type=Path, default=None, help="Optional path to write table")

    args = p.parse_args()

    if comet_ml is None and args.use_comet:
        raise ImportError(
            "CometML is not installed, however `--use_comet` is true."
            "Either install the package or disable comet logging"
        )

    DEVICE = (
        torch.device("cuda", args.device)
        if args.device is not None and torch.cuda.is_available()
        else torch.device("cpu")
    )

    torch.set_default_device(DEVICE)

    COMET_WORKSPACE = args.comet_workspace
    COMET_EXP_NAME = args.comet_experiment_name_prefix
    COMET_PROJECT_NAME = args.comet_project_name

    print(f"GLOBAL DEVICE IS SET: {DEVICE=}")

    return args


def measure_kernel_performance(
    X: torch.Tensor,
    graph: Any,
    conv: Callable[..., Any],
) -> dict[str, Any]:
    def forward_function():
        nonlocal X, graph, conv
        return conv(X, graph)

    grad_output = torch.randn_like(X)

    Y = forward_function()

    def backward_function():
        Y.backward(grad_output, retain_graph=True)

    try:
        forward_function_measurements: MicrobenchResult = time_callable(forward_function, warmup=3, iters=10)
    except Exception as e:
        print(f"Couldn't measure forward performance for convolution {conv}. Exception: {e}")
        forward_function_measurements = MicrobenchResult(
            iters=10,
            ms_per_iter=float("nan"),
            device="cuda",
            memory_allocated=None,
        )

    forward_results = {
        "forward_ms": forward_function_measurements.ms_per_iter,
        "forward_memory_mb": forward_function_measurements.memory_allocated,
    }

    try:
        backward_function_measurements: MicrobenchResult = time_callable(backward_function, warmup=3, iters=10)
    except Exception as e:
        print(f"Couldn't measure backward performance for convolution {conv}. Exception: {e}")
        backward_function_measurements = MicrobenchResult(
            iters=10,
            ms_per_iter=float("nan"),
            device="cuda",
            memory_allocated=None,
        )

    backward_results = {
        "backward_ms": bwd_ms if (bwd_ms := backward_function_measurements.ms_per_iter) != float("nan") else None,
        "backward_memory_mb": backward_function_measurements.memory_allocated,
    }

    overall_dict = forward_results | backward_results
    return overall_dict


def generate_experiment_name(
    prefix: str,
    conv_type: str,
    backend: str,
    dataset_name: str,
    other_params: dict[str, Any],
):
    experiment_name = f"{prefix}_{conv_type}_{backend}_{dataset_name}_".lstrip("_")
    experiment_name += "_".join(f"{key}_{val}" for key, val in other_params.items()).strip("_")

    return experiment_name.strip("_")


def load_results_to_comet(results_dict):
    exp_config = comet_ml.ExperimentConfig(name=results_dict["experiment_name"])

    experiment = comet_ml.start(
        api_key=os.getenv("COMET_TOKEN"),
        project_name=COMET_PROJECT_NAME,
        workspace=COMET_WORKSPACE,
        experiment_config=exp_config,
        mode="create",
    )

    experiment.log_metrics(results_dict, step=0)
    experiment.end()


def main():
    args = parse_args()

    CONV_TYPE = args.conv_type

    datasets_configs_to_load: list[dict[str, str]] = []

    for dataset_cfg_path in args.datasets:
        with open(dataset_cfg_path, encoding="utf-8") as f:
            datasets_configs_to_load.append(yaml.safe_load(f)["dataset"])

    with open(args.conv_params_grid, encoding="utf-8") as f:
        kernels_parameters_dict = yaml.safe_load(f)

        keys = kernels_parameters_dict.keys()
        values = kernels_parameters_dict.values()

        convolution_parameters_grid = [dict(zip(keys, combo)) for combo in product(*values)]
        del keys, values

    results_for_table = []

    print(f"Backends are: {args.backends}")
    for backend in args.backends:
        try:
            backend_module = BackendRegistry.get_backend(backend)
        except Exception as e:
            print(f"Couldn't load backend={backend} for conv={args.conv_type}. Exception: {e}")
            continue

        for dataset_config in datasets_configs_to_load:
            dataset_name = dataset_config["name"]

            graph = load_single_graph(
                DatasetConfig(
                    source=dataset_config["source"],
                    name=dataset_config["name"],
                    root=dataset_config["root"],
                    conv_backend=backend,
                )
            )

            num_nodes = graph.num_nodes
            graph_repr = graph.graph_repr

            del graph
            torch.cuda.empty_cache()

            for layer_parameters_dict_instance in convolution_parameters_grid:
                feature_dim = layer_parameters_dict_instance["feature_dim"]
                x = torch.randn(num_nodes, feature_dim, device=DEVICE, requires_grad=True)

                try:
                    conv = backend_module.create_conv(args.conv_type, **layer_parameters_dict_instance)
                    conv = conv.to(DEVICE)
                except Exception as e:
                    print(f"Couldnt create conv={args.conv_type} for {backend=}. Exception: {e}")
                    continue

                measurements_dict = measure_kernel_performance(X=x, graph=graph_repr, conv=conv)

                common_dict = {
                    "conv_type": args.conv_type,
                    "dataset": dataset_name,
                    "backend": backend,
                }

                experiment_name = generate_experiment_name(
                    prefix=COMET_EXP_NAME,
                    conv_type=args.conv_type,
                    backend=backend,
                    dataset_name=dataset_name,
                    other_params=layer_parameters_dict_instance,
                )

                overall_dict = common_dict | layer_parameters_dict_instance | measurements_dict | get_gpu_info()
                overall_dict["experiment_name"] = experiment_name

                results_for_table.append(overall_dict)

                print(dumps(overall_dict, indent=4))
            else:
                del x

            del graph_repr
            torch.cuda.empty_cache()

    df_for_dump = (
        pd.DataFrame(results_for_table).sort_values(by=["dataset", "backend", "feature_dim"]).reset_index(drop=True)
    )

    if args.out is not None:
        args.out.parent.mkdir(exist_ok=True, parents=True)
        df_for_dump.to_csv(args.out)

    print("=============== DONE ===============\nRESULTS:")
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)

    df_without_constant_columns = df_for_dump.loc[:, (df_for_dump != df_for_dump.iloc[0]).any()].drop(
        "experiment_name",
        axis="columns",
    )

    value_cols = [
        col for col in df_without_constant_columns.columns if col not in ["feature_dim", "dataset", "backend"]
    ]

    pivoted = df_without_constant_columns.pivot_table(
        index=["dataset", "feature_dim"], columns="backend", values=value_cols
    )

    pivoted.columns = [f"{backend}_{col}" for col, backend in pivoted.columns]
    pivoted = pivoted.reset_index()
    print(pivoted.to_markdown())

    if args.use_comet:
        with ThreadPoolExecutor(max_workers=len(results_for_table)) as executor:
            futures = {executor.submit(load_results_to_comet, d): d for d in results_for_table}
            for future in concurrent.futures.as_completed(futures):
                result_dict = futures[future]
                try:
                    data = future.result()
                except Exception as exc:
                    print("%r generated an exception: %s" % (result_dict["experiment_name"], exc))
                else:
                    print(f"Future for {result_dict['experiment_name']} is DONE")


if __name__ == "__main__":
    main()
