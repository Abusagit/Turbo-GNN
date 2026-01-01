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
import torch.multiprocessing as mp
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

mp.set_start_method("spawn", force=True)
queue = mp.Queue()

DEVICE = None
COMET_WORKSPACE = "None"
COMET_PROJECT_NAME = "None"
COMET_EXP_NAME = ""


BACKENDS_PRONE_TO_ERROR = {"f3s"}


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

    try:
        forward_function_measurements: MicrobenchResult = time_callable(forward_function, warmup=3, iters=5)
    except (Exception, torch.OutOfMemoryError) as e:
        print(f"Couldn't measure forward performance for convolution {conv}. Exception: {e}")
        forward_function_measurements = MicrobenchResult(
            iters=10,
            ms_per_iter=float("nan"),
            device="cuda",
            memory_allocated=None,
        )
        torch.cuda.empty_cache()

    forward_results = {
        "forward_ms": forward_function_measurements.ms_per_iter,
        "forward_memory_mb": forward_function_measurements.memory_allocated,
    }

    try:
        Y = forward_function()
    except (Exception, torch.OutOfMemoryError) as e:
        print(f"Couldn't measure forward performance for convolution {conv}. Exception: {e}")

        forward_results = {
            "forward_ms": None,
            "forward_memory_mb": None,
        }

        backward_results = {
            "backward_ms": None,
            "backward_memory_mb": None,
        }
        torch.cuda.empty_cache()
        return forward_results | backward_results

    try:
        grad_output = torch.randn_like(X)

        def backward_function():
            Y.backward(grad_output, retain_graph=True)

        backward_function_measurements: MicrobenchResult = time_callable(backward_function, warmup=3, iters=10)
    except (Exception, torch.OutOfMemoryError) as e:
        print(f"Couldn't measure backward performance for convolution {conv}. Exception: {e}")
        backward_function_measurements = MicrobenchResult(
            iters=10,
            ms_per_iter=float("nan"),
            device="cuda",
            memory_allocated=None,
        )
        torch.cuda.empty_cache()

    backward_results = {
        "backward_ms": bwd_ms if (bwd_ms := backward_function_measurements.ms_per_iter) != float("nan") else None,  # type: ignore
        "backward_memory_mb": backward_function_measurements.memory_allocated,  # type: ignore
    }

    overall_dict = forward_results | backward_results
    return overall_dict


def _run_measurement_in_subprocess(X, graph, conv, queue):
    """Run measurement in isolated process"""
    try:
        result = measure_kernel_performance(X, graph, conv)
        queue.put(("success", result))
    except Exception as e:
        queue.put(("error", str(e)))


def measure_kernel_performance_safe(X, graph, conv, timeout=60):
    """Wrapper that runs measurement in subprocess to catch hard crashes"""
    process = mp.Process(target=_run_measurement_in_subprocess, args=(X, graph, conv, queue))

    process.start()
    process.join(timeout=timeout)

    if process.is_alive():
        process.terminate()
        process.join()
        return {
            "forward_ms": None,
            "forward_memory_mb": None,
            "backward_ms": None,
            "backward_memory_mb": None,
            "error": "Timeout or hung",
        }

    if process.exitcode != 0:
        return {
            "forward_ms": None,
            "forward_memory_mb": None,
            "backward_ms": None,
            "backward_memory_mb": None,
            "error": f"Process crashed with exit code {process.exitcode}",
        }

    if not queue.empty():
        status, result = queue.get()
        if status == "success":
            return result
        else:
            return {
                "forward_ms": None,
                "forward_memory_mb": None,
                "backward_ms": None,
                "backward_memory_mb": None,
                "error": result,
            }

    return {
        "forward_ms": None,
        "forward_memory_mb": None,
        "backward_ms": None,
        "backward_memory_mb": None,
        "error": "Unknown error",
    }


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


def get_parameters_grid_from_config(parameters_dict: dict[str, list[Any]]):
    keys = parameters_dict.keys()
    values = parameters_dict.values()

    parameters_grid = [dict(zip(keys, combo)) for combo in product(*values)]
    return parameters_grid


def main():
    args = parse_args()

    CONV_TYPE = args.conv_type

    PARAMETERS_USED_IN_SWEEP: set[str] = set()  # collect parameters used for kernel comparison

    datasets_configs_to_load: list[dict[str, str]] = []

    with open(args.conv_params_grid, encoding="utf-8") as f:
        top_level_config = yaml.safe_load(f)

        conv_parameters_dict = top_level_config.get("params_grid")
        kernel_specific_parameters_dict = top_level_config.get("kernel_related_kwargs", {})

        datasets_config = top_level_config["datasets"]
        base_dir = Path(datasets_config["base_path"])
        for dir_name, dir_params in datasets_config["dirs"].items():
            load_all_configs = dir_params.get("all", True)
            configs_dir = base_dir / dir_name
            if load_all_configs:
                all_files_in_current_dir: list[str] = list(map(str, configs_dir.glob("*.yaml")))
            else:
                all_files_in_current_dir = [
                    configs_dir / f"{dataset_name}.yaml" for dataset_name in dir_params.get("choices", [])
                ]
            for cfg_path in all_files_in_current_dir:
                with open(cfg_path, encoding="utf-8") as f_read:
                    datasets_configs_to_load.append(yaml.safe_load(f_read)["dataset"])

    results_for_table = []

    print(f"Backends are: {args.backends}")
    for backend in args.backends:
        try:
            backend_module = BackendRegistry.get_backend(backend)
        except Exception as e:
            print(f"Couldn't load backend={backend} for conv={CONV_TYPE}. Exception: {e}")
            continue

        convolution_parameters_grid = get_parameters_grid_from_config(
            conv_parameters_dict.get("all", {}) | conv_parameters_dict.get(backend, {})
        )

        kernel_param_grid_for_backend = kernel_specific_parameters_dict.get(backend, {})

        kernel_specific_parameters_grid_for_datasets = get_parameters_grid_from_config(
            kernel_specific_parameters_dict.get("all", {"graph_reordering_partition_size": [-1]})
            | kernel_param_grid_for_backend
        )

        for dataset_config in datasets_configs_to_load:
            dataset_name = dataset_config["name"]

            try:
                graph = load_single_graph(
                    DatasetConfig(
                        source=dataset_config["source"],
                        name=dataset_config["name"],
                        root=dataset_config["root"],
                        conv_backend=backend,
                    )
                )
            except Exception as e:
                print(f"Couldn't load graph {dataset_name}, exception: {e}")
                break

            for kernel_specific_dataset_config in kernel_specific_parameters_grid_for_datasets:
                graph = graph.update_graph_repr_with_new_hyperparameters(
                    new_kernel_related_kwargs=kernel_specific_dataset_config,
                )
                PARAMETERS_USED_IN_SWEEP |= set(kernel_specific_dataset_config.keys())

                num_nodes = graph.num_nodes
                graph_repr = graph.graph_repr
                for layer_parameters_dict_instance in convolution_parameters_grid:
                    try:
                        feature_dim = layer_parameters_dict_instance["feature_dim"]
                        x = torch.randn(num_nodes, feature_dim, device=DEVICE, requires_grad=True)
                        PARAMETERS_USED_IN_SWEEP |= set(layer_parameters_dict_instance.keys())
                        conv = backend_module.create_conv(CONV_TYPE, **layer_parameters_dict_instance)
                        conv = conv.to(DEVICE)
                    except Exception as e:
                        print(f"Couldnt create conv={CONV_TYPE} for {backend=}. Exception: {e}")
                        torch.cuda.empty_cache()
                        continue

                    if backend in BACKENDS_PRONE_TO_ERROR:
                        measurements_dict = measure_kernel_performance_safe(X=x, graph=graph_repr, conv=conv)
                    else:
                        measurements_dict = measure_kernel_performance(X=x, graph=graph_repr, conv=conv)

                    common_dict = {
                        "conv_type": CONV_TYPE,
                        "dataset": dataset_name,
                        "backend": backend,
                        "num_nodes": graph.num_nodes,
                        "num_edges": graph.edge_index.shape[1],
                        "avg_node_degree": graph.num_nodes / graph.edge_index.shape[1],
                    }

                    experiment_name = generate_experiment_name(
                        prefix=COMET_EXP_NAME,
                        conv_type=CONV_TYPE,
                        backend=backend,
                        dataset_name=dataset_name,
                        other_params=layer_parameters_dict_instance,
                    )

                    overall_dict = (
                        common_dict
                        | layer_parameters_dict_instance
                        | measurements_dict
                        | kernel_specific_dataset_config
                        | get_gpu_info()
                    )
                    overall_dict["experiment_name"] = experiment_name

                    results_for_table.append(overall_dict)

                    print(dumps(overall_dict, indent=4))
                    del x
                    torch.cuda.empty_cache()

                del graph_repr
                torch.cuda.empty_cache()

            del graph
            torch.cuda.empty_cache()

    values_to_groupby = ["dataset", "backend", "feature_dim"] + sorted(PARAMETERS_USED_IN_SWEEP)

    df_for_dump = pd.DataFrame(results_for_table).sort_values(by=values_to_groupby).reset_index(drop=True)

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
    # add placeholders in canse of a single backend/dataset/feature_dim parameter
    if "backend" not in df_without_constant_columns.columns:
        df_without_constant_columns["backend"] = df_for_dump.loc[0, "backend"]
    if "feature_dim" not in df_without_constant_columns.columns:
        df_without_constant_columns["feature_dim"] = df_for_dump.loc[0, "feature_dim"]
    if "dataset" not in df_without_constant_columns.columns:
        df_without_constant_columns["dataset"] = df_for_dump.loc[0, "dataset"]

    value_cols = [
        col for col in df_without_constant_columns.columns if col not in ["feature_dim", "dataset", "backend"]
    ]

    index = (
        ["dataset", "feature_dim"] if "heads" not in PARAMETERS_USED_IN_SWEEP else ["dataset", "feature_dim", "heads"]
    )
    pivoted = df_without_constant_columns.pivot_table(index=index, columns="backend", values=value_cols)

    pivoted.columns = [f"{backend}_{col}" for col, backend in pivoted.columns]
    pivoted = pivoted.reset_index()
    print(pivoted.to_markdown())

    if args.use_comet:
        with ThreadPoolExecutor(max_workers=len(results_for_table)) as executor:
            futures = {executor.submit(load_results_to_comet, d): d for d in results_for_table}
            for future in concurrent.futures.as_completed(futures):
                result_dict = futures[future]
                try:
                    _ = future.result()
                except Exception as exc:
                    print("%r generated an exception: %s" % (result_dict["experiment_name"], exc))
                else:
                    print(f"Future for {result_dict['experiment_name']} is DONE")


if __name__ == "__main__":
    main()
