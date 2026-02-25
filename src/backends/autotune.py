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
from pathlib import Path
from typing import Any

import torch

import src.benchmarking.microbench as _microbench
from src.backends.base import AutotuneConfig, BaseConvolution, TunableParam
from src.data.datasets import GraphSample

logger = logging.getLogger(__name__)


class AutotuneCache:
    """JSON-backed cache for autotuning results."""

    @staticmethod
    def compute_cache_key(
        conv_class: str,
        feature_dim: int,
        num_nodes: int,
        num_edges: int,
        device: str,
        param_space: list[TunableParam],
    ) -> str:
        """Compute a SHA256-based cache key from the tuning context."""
        key_data = {
            "conv_class": conv_class,
            "feature_dim": feature_dim,
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "device": device,
            "param_space": [{"name": p.name, "values": p.values} for p in param_space],
        }
        raw = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()

    @staticmethod
    def _cache_path(cache_dir: str, conv_class_name: str, key: str) -> Path:
        return Path(cache_dir) / f"{conv_class_name}_{key}_autotune_cache.json"

    @staticmethod
    def load(cache_dir: str, conv_class_name: str, key: str) -> dict | None:
        """Load cached result for the given key, or None if not found."""
        path = AutotuneCache._cache_path(cache_dir, conv_class_name, key)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())  # type: ignore
        except (json.JSONDecodeError, OSError):
            return None

    @staticmethod
    def save(cache_dir: str, conv_class_name: str, key: str, result: dict) -> None:
        """Save a result to the cache file."""
        path = AutotuneCache._cache_path(cache_dir, conv_class_name, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2))

    @staticmethod
    def clear_cache(cache_dir: str, conv_class_name: str | None = None) -> int:
        """Delete cache files. Returns count of files deleted."""
        cache_path = Path(cache_dir)
        if not cache_path.exists():
            return 0

        if conv_class_name is not None:
            pattern = f"{conv_class_name}_*_autotune_cache.json"
        else:
            pattern = "*_autotune_cache.json"

        count = 0
        for path in cache_path.glob(pattern):
            path.unlink()
            count += 1
        return count


def _build_combinations(params: list[TunableParam]) -> list[dict[str, Any]]:
    """Build all combinations from a list of TunableParam."""
    if not params:
        return [{}]
    names = [p.name for p in params]
    value_lists = [p.values for p in params]
    return [dict(zip(names, combo)) for combo in itertools.product(*value_lists)]


def _apply_best_config(
    conv: BaseConvolution,
    graph_sample: GraphSample,
    best_config: dict[str, Any],
    graph_params: list[TunableParam],
) -> None:
    """Apply the best configuration, separating graph from kernel params."""
    graph_param_names = {p.name for p in graph_params}

    graph_cfg = {k: v for k, v in best_config.items() if k in graph_param_names}
    kernel_cfg = {k: v for k, v in best_config.items() if k not in graph_param_names}

    if graph_cfg:
        current_kwargs = dict(graph_sample.kernel_related_kwargs)
        current_kwargs.update(graph_cfg)
        graph_sample.update_graph_repr_with_new_hyperparameters(current_kwargs)

    if kernel_cfg:
        conv.configure(**kernel_cfg)


def _grid_search(
    conv: BaseConvolution,
    x: torch.Tensor,
    graph_sample: GraphSample,
    config: AutotuneConfig,
    kernel_params: list[TunableParam],
    graph_params: list[TunableParam],
    make_bench_fn,
) -> tuple[dict[str, Any], float]:
    """Run grid search: outer loop over graph combos, inner over kernel combos.

    Args:
        conv: The convolution module to tune.
        x: Input features for benchmarking.
        graph_sample: GraphSample instance for graph param tuning.
        config: Autotuning configuration.
        kernel_params: Kernel parameters to search over.
        graph_params: Graph parameters to search over.
        make_bench_fn: Callable(conv, x, graph_repr) -> zero-arg callable for timing.

    Returns:
        (best_config_dict, best_ms)
    """
    time_callable = _microbench.time_callable

    graph_combos = _build_combinations(graph_params)
    kernel_combos = _build_combinations(kernel_params)

    total_trials = len(graph_combos) * len(kernel_combos)
    conv_class_name = type(conv).__name__
    logger.info(
        "Grid search %s: %d graph combos x %d kernel combos = %d total trials",
        conv_class_name,
        len(graph_combos),
        len(kernel_combos),
        total_trials,
    )

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
            # apply kernel params
            if kernel_cfg:
                conv.configure(**kernel_cfg)

            bench_fn = make_bench_fn(conv, x, graph_repr)

            result = time_callable(bench_fn, warmup=config.warmup, iters=config.iters, do_memory_profile=False)
            ms = result.ms_per_iter

            combined_cfg = {**graph_cfg, **kernel_cfg}
            logger.debug("Trial %d/%d: %s -> %.3f ms", trial, total_trials, combined_cfg, ms)

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

    conv_class_name = type(conv).__name__
    feature_dim = x.shape[1] if x.ndim > 1 else 1

    # check cache (key includes all four param lists)
    if config.cache_dir is not None and config.use_cache:
        cache_key = AutotuneCache.compute_cache_key(
            conv_class=conv_class_name,
            feature_dim=feature_dim,
            num_nodes=graph_sample.num_nodes,
            num_edges=graph_sample.num_edges,
            device=str(x.device),
            param_space=all_params,
        )
        cached = AutotuneCache.load(config.cache_dir, conv_class_name, cache_key)
        if cached is not None:
            all_graph_params = fwd_graph_params + bwd_graph_params
            logger.info("Autotuning cache hit for %s. Applying cached config: %s", conv_class_name, cached)
            _apply_best_config(conv, graph_sample, cached, all_graph_params)
            return cached

    best_fwd: dict[str, Any] = {}
    best_bwd: dict[str, Any] = {}

    # --- fwd grid search ---
    fwd_params = fwd_kernel_params + fwd_graph_params
    if fwd_params:

        def _make_fwd_bench(c, xi, g):
            def _bench():
                c.forward(xi, g)

            return _bench

        logger.info("Autotuning %s forward pass:", conv_class_name)
        best_fwd, fwd_ms = _grid_search(
            conv,
            x,
            graph_sample,
            config,
            fwd_kernel_params,
            fwd_graph_params,
            _make_fwd_bench,
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

        logger.info("Autotuning %s backward pass:", conv_class_name)
        best_bwd, bwd_ms = _grid_search(
            conv,
            x,
            graph_sample,
            config,
            bwd_kernel_params,
            bwd_graph_params,
            _make_bwd_bench,
        )
        logger.info("Backward best: %s (%.3f ms)", best_bwd, bwd_ms)

    # merge (no overlap by design)
    best_config = {**best_fwd, **best_bwd}
    all_graph_params = fwd_graph_params + bwd_graph_params

    logger.info("Autotuning %s complete. Best config: %s", conv_class_name, best_config)

    # apply best config
    _apply_best_config(conv, graph_sample, best_config, all_graph_params)

    # save to cache
    if config.cache_dir is not None:
        cache_key = AutotuneCache.compute_cache_key(
            conv_class=conv_class_name,
            feature_dim=feature_dim,
            num_nodes=graph_sample.num_nodes,
            num_edges=graph_sample.num_edges,
            device=str(x.device),
            param_space=all_params,
        )
        AutotuneCache.save(config.cache_dir, conv_class_name, cache_key, best_config)
        logger.info("Saved autotuning result to cache at %s", config.cache_dir)

    return best_config
