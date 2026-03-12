"""
Autotuning engine for CUDA backend kernel and graph parameters.

Performs grid search grouped by graph params (outer) then kernel params (inner)
to minimize expensive graph rebuilds. Results are cached to JSON on disk.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
import shutil
from pathlib import Path
from typing import Any

import torch

import src.benchmarking.microbench as _microbench
from src.backends.base import AutotuneConfig, BaseConvolution, TunableKernel, TunableParam
from src.data.datasets import GraphSample

logger = logging.getLogger(__name__)


def _get_gpu_name(device) -> str:
    """Return the GPU device name, or ``"cpu"`` when not on a CUDA device."""
    if torch.cuda.is_available() and getattr(device, "type", None) == "cuda":
        return torch.cuda.get_device_properties(device).name  # type: ignore
    return "cpu"


class AutotuneCache:
    """Per-trial JSON cache for individual autotuning timing results.

    Each trial (a single parameter configuration) is stored as its own
    JSON file under ``{cache_dir}/{ConvClassName}/{trial_key}.json``.
    """

    @staticmethod
    def compute_trial_key(
        conv_class: str,
        feature_dim: int,
        num_nodes: int,
        num_edges: int,
        gpu_name: str,
        trial_config: dict[str, Any],
    ) -> str:
        """Compute a SHA256 cache key for a single trial configuration."""
        key_data = {
            "conv_class": conv_class,
            "feature_dim": feature_dim,
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "gpu_name": gpu_name,
            "config": trial_config,
        }
        raw = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()

    @staticmethod
    def _trial_path(cache_dir: str, conv_class_name: str, trial_key: str) -> Path:
        return Path(cache_dir) / conv_class_name / f"{trial_key}.json"

    @staticmethod
    def load_trial(cache_dir: str, conv_class_name: str, trial_key: str) -> float | None:
        """Load a cached trial's ``ms_per_iter``, or ``None`` if not found."""
        path = AutotuneCache._trial_path(cache_dir, conv_class_name, trial_key)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            return float(data["ms_per_iter"])
        except (json.JSONDecodeError, OSError, KeyError, TypeError, ValueError):
            return None

    @staticmethod
    def save_trial(cache_dir: str, conv_class_name: str, trial_key: str, ms_per_iter: float) -> None:
        """Persist a single trial timing result."""
        path = AutotuneCache._trial_path(cache_dir, conv_class_name, trial_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"ms_per_iter": ms_per_iter}))

    @staticmethod
    def clear_cache(cache_dir: str, conv_class_name: str | None = None) -> int:
        """Delete cached trial files. Returns count of files deleted."""
        cache_path = Path(cache_dir)
        if not cache_path.exists():
            return 0

        if conv_class_name is not None:
            subdir = cache_path / conv_class_name
            if not subdir.is_dir():
                return 0
            count = sum(1 for _ in subdir.glob("*.json"))
            shutil.rmtree(subdir)
            return count

        count = 0
        for subdir in cache_path.iterdir():
            if subdir.is_dir():
                count += sum(1 for _ in subdir.glob("*.json"))
                shutil.rmtree(subdir)
        return count


def _build_combinations(params: list[TunableParam]) -> list[dict[str, Any]]:
    """Build all combinations from a list of TunableParam."""
    if not params:
        return [{}]
    names = [p.name for p in params]
    value_lists = [p.values for p in params]
    return [dict(zip(names, combo)) for combo in itertools.product(*value_lists)]


def _apply_best_config(
    target,
    graph_sample: GraphSample,
    best_config: dict[str, Any],
    graph_params: list[TunableParam],
) -> None:
    """Apply the best configuration, separating graph from kernel params.

    ``target`` can be a :class:`BaseConvolution` or :class:`TunableKernel` —
    anything with a ``configure(**kwargs)`` method.
    """
    graph_param_names = {p.name for p in graph_params}

    graph_cfg = {k: v for k, v in best_config.items() if k in graph_param_names}
    kernel_cfg = {k: v for k, v in best_config.items() if k not in graph_param_names}

    if graph_cfg:
        current_kwargs = dict(graph_sample.kernel_related_kwargs)
        current_kwargs.update(graph_cfg)
        graph_sample.update_graph_repr_with_new_hyperparameters(current_kwargs)

    if kernel_cfg:
        target.configure(**kernel_cfg)


def _grid_search(
    target,
    x: torch.Tensor,
    graph_sample: GraphSample,
    config: AutotuneConfig,
    kernel_params: list[TunableParam],
    graph_params: list[TunableParam],
    make_bench_fn,
    *,
    target_name: str,
    feature_dim: int,
    gpu_name: str,
) -> tuple[dict[str, Any], float]:
    """Run grid search: outer loop over graph combos, inner over kernel combos.

    Individual trial results are cached per-trial when ``config.cache_dir``
    is set, so previously-timed configurations are reused even if the
    parameter space changes.

    Args:
        target: The convolution or kernel callable to tune (anything with
            ``configure(**kwargs)``).
        x: Input features for benchmarking.
        graph_sample: GraphSample instance for graph param tuning.
        config: Autotuning configuration.
        kernel_params: Kernel parameters to search over.
        graph_params: Graph parameters to search over.
        make_bench_fn: Callable(target, x, graph_repr) -> zero-arg callable for timing.
        target_name: Name of *target* (for cache keys / logging).
        feature_dim: Input feature dimensionality (for cache keys).
        gpu_name: GPU device name string (for cache keys).

    Returns:
        (best_config_dict, best_ms)
    """
    time_callable = _microbench.time_callable

    graph_combos = _build_combinations(graph_params)
    kernel_combos = _build_combinations(kernel_params)

    total_trials = len(graph_combos) * len(kernel_combos)
    logger.info(
        "Grid search %s: %d graph combos x %d kernel combos = %d total trials",
        target_name,
        len(graph_combos),
        len(kernel_combos),
        total_trials,
    )

    use_cache = config.cache_dir is not None and config.use_cache
    save_cache = config.cache_dir is not None

    best_ms = float("inf")
    best_config: dict[str, Any] = {}
    trial = 0

    for graph_cfg in graph_combos:
        # apply graph params (expensive: rebuilds CSR, partitions nodes)
        if graph_cfg:
            current_kwargs = dict(graph_sample.kernel_related_kwargs)
            current_kwargs.update(graph_cfg)
            graph_sample.update_graph_repr_with_new_hyperparameters(current_kwargs)

        graph_repr = graph_sample.graph_repr

        for kernel_cfg in kernel_combos:
            trial += 1
            combined_cfg = {**graph_cfg, **kernel_cfg}

            # apply kernel params
            if kernel_cfg:
                target.configure(**kernel_cfg)

            # per-trial cache lookup
            trial_key = None
            ms = None
            if use_cache:
                trial_key = AutotuneCache.compute_trial_key(
                    target_name,
                    feature_dim,
                    graph_sample.num_nodes,
                    graph_sample.num_edges,
                    gpu_name,
                    combined_cfg,
                )
                ms = AutotuneCache.load_trial(config.cache_dir, target_name, trial_key)  # type: ignore
                if ms is not None:
                    logger.debug("Trial %d/%d: %s -> %.3f ms (cached)", trial, total_trials, combined_cfg, ms)

            if ms is None:
                bench_fn = make_bench_fn(target, x, graph_repr)
                result = time_callable(bench_fn, warmup=config.warmup, iters=config.iters, do_memory_profile=False)
                ms = result.ms_per_iter
                logger.debug("Trial %d/%d: %s -> %.3f ms", trial, total_trials, combined_cfg, ms)

                # save this trial
                if save_cache:
                    if trial_key is None:
                        trial_key = AutotuneCache.compute_trial_key(
                            target_name,
                            feature_dim,
                            graph_sample.num_nodes,
                            graph_sample.num_edges,
                            gpu_name,
                            combined_cfg,
                        )
                    AutotuneCache.save_trial(config.cache_dir, target_name, trial_key, ms)  # type: ignore

            if ms < best_ms:
                best_ms = ms
                best_config = combined_cfg

    return best_config, best_ms


def run_autotune(
    conv: BaseConvolution,
    x: torch.Tensor,
    graph_sample: GraphSample,
    config: AutotuneConfig,
) -> dict:
    """Core autotuning search with separate forward/backward parameter spaces.

    Runs independent grid searches for forward and backward passes, then
    merges results. Forward uses get_tunable_forward_kernel_params() and
    get_tunable_forward_graph_params(); backward uses get_tunable_backward_kernel_params()
    and get_tunable_backward_graph_params().

    Args:
        conv: The convolution module to tune.
        x: Input features for benchmarking.
        graph_sample: GraphSample instance for graph param tuning.
        config: Autotuning configuration.

    Returns:
        Dict of best parameter name -> value mappings.
    """
    fwd_kernel_params = conv.get_tunable_forward_kernel_params()
    fwd_graph_params = conv.get_tunable_forward_graph_params()
    bwd_kernel_params = conv.get_tunable_backward_kernel_params() if config.tune_backward else []
    bwd_graph_params = conv.get_tunable_backward_graph_params() if config.tune_backward else []

    all_params = fwd_kernel_params + fwd_graph_params + bwd_kernel_params + bwd_graph_params

    if not all_params:
        logger.info("No tunable parameters declared. Skipping autotuning.")
        return {}

    target_name = type(conv).__name__
    feature_dim = x.shape[1] if x.ndim > 1 else 1
    gpu_name = _get_gpu_name(x.device)

    best_fwd: dict[str, Any] = {}
    best_bwd: dict[str, Any] = {}

    # --- fwd grid search ---
    fwd_params = fwd_kernel_params + fwd_graph_params
    if fwd_params:

        def _make_fwd_bench(c, xi, g):
            def _bench():
                c.forward(xi, g)

            return _bench

        logger.info("Autotuning %s forward pass:", target_name)
        best_fwd, fwd_ms = _grid_search(
            conv,
            x,
            graph_sample,
            config,
            fwd_kernel_params,
            fwd_graph_params,
            _make_fwd_bench,
            target_name=target_name,
            feature_dim=feature_dim,
            gpu_name=gpu_name,
        )
        logger.info("Forward best: %s (%.3f ms)", best_fwd, fwd_ms)

    # --- bwd grid search ---
    bwd_params = bwd_kernel_params + bwd_graph_params
    if config.tune_backward and bwd_params:

        def _make_bwd_bench(c, xi, g):
            grad_output = torch.randn_like(xi)
            out = c.forward(xi, g)

            def _bench():
                out.backward(grad_output, retain_graph=True)

            return _bench

        logger.info("Autotuning %s backward pass:", target_name)
        best_bwd, bwd_ms = _grid_search(
            conv,
            x,
            graph_sample,
            config,
            bwd_kernel_params,
            bwd_graph_params,
            _make_bwd_bench,
            target_name=target_name,
            feature_dim=feature_dim,
            gpu_name=gpu_name,
        )
        logger.info("Backward best: %s (%.3f ms)", best_bwd, bwd_ms)

    # merge (no overlap by design)
    best_config = {**best_fwd, **best_bwd}
    all_graph_params = fwd_graph_params + bwd_graph_params

    logger.info("Autotuning %s complete. Best config: %s", target_name, best_config)

    # apply best config
    _apply_best_config(conv, graph_sample, best_config, all_graph_params)

    return best_config


def run_autotune_kernel(
    kernel: TunableKernel,
    x: torch.Tensor,
    graph_sample: GraphSample,
    config: AutotuneConfig,
) -> dict:
    """Autotuning search for a :class:`TunableKernel` callable.

    Uses ``kernel.make_forward_bench_fn`` / ``kernel.make_backward_bench_fn``
    to construct timing callables instead of ``conv.forward()``.

    Args:
        kernel: The kernel callable to tune.
        x: Input features for benchmarking.
        graph_sample: GraphSample instance for graph param tuning.
        config: Autotuning configuration.

    Returns:
        Dict of best parameter name -> value mappings.
    """
    fwd_kernel_params = kernel.get_tunable_forward_kernel_params()
    fwd_graph_params = kernel.get_tunable_forward_graph_params()
    bwd_kernel_params = kernel.get_tunable_backward_kernel_params() if config.tune_backward else []
    bwd_graph_params = kernel.get_tunable_backward_graph_params() if config.tune_backward else []

    all_params = fwd_kernel_params + fwd_graph_params + bwd_kernel_params + bwd_graph_params

    if not all_params:
        logger.info("No tunable parameters declared on %s. Skipping autotuning.", kernel.name)
        return {}

    target_name = kernel.name
    feature_dim = x.shape[1] if x.ndim > 1 else 1
    gpu_name = _get_gpu_name(x.device)

    best_fwd: dict[str, Any] = {}
    best_bwd: dict[str, Any] = {}

    # --- fwd grid search ---
    fwd_params = fwd_kernel_params + fwd_graph_params
    if fwd_params:

        def _make_fwd_bench(k, xi, g):
            return k.make_forward_bench_fn(xi, g)

        logger.info("Autotuning %s forward pass:", target_name)
        best_fwd, fwd_ms = _grid_search(
            kernel,
            x,
            graph_sample,
            config,
            fwd_kernel_params,
            fwd_graph_params,
            _make_fwd_bench,
            target_name=target_name,
            feature_dim=feature_dim,
            gpu_name=gpu_name,
        )
        logger.info("Forward best: %s (%.3f ms)", best_fwd, fwd_ms)

    # --- bwd grid search ---
    bwd_params = bwd_kernel_params + bwd_graph_params
    if config.tune_backward and bwd_params:

        def _make_bwd_bench(k, xi, g):
            return k.make_backward_bench_fn(xi, g)

        logger.info("Autotuning %s backward pass:", target_name)
        best_bwd, bwd_ms = _grid_search(
            kernel,
            x,
            graph_sample,
            config,
            bwd_kernel_params,
            bwd_graph_params,
            _make_bwd_bench,
            target_name=target_name,
            feature_dim=feature_dim,
            gpu_name=gpu_name,
        )
        logger.info("Backward best: %s (%.3f ms)", best_bwd, bwd_ms)

    # merge (no overlap by design)
    best_config = {**best_fwd, **best_bwd}
    all_graph_params = fwd_graph_params + bwd_graph_params

    logger.info("Autotuning %s complete. Best config: %s", target_name, best_config)

    # apply best config
    _apply_best_config(kernel, graph_sample, best_config, all_graph_params)

    return best_config
