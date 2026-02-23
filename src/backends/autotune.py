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


def run_autotune(
    conv: BaseConvolution,
    x: torch.Tensor,
    graph_sample: GraphSample,
    config: AutotuneConfig,
) -> dict:
    """Core autotuning search.

    Grid search grouped by graph params (outer) then kernel params (inner)
    to minimize expensive graph rebuilds.

    Args:
        conv: The convolution module to tune.
        x: Input features for benchmarking.
        graph_sample: GraphSample instance for graph param tuning.
        config: Autotuning configuration.

    Returns:
        Dict of best parameter name -> value mappings.
    """
    time_callable = _microbench.time_callable

    kernel_params = conv.get_tunable_kernel_params()
    graph_params = conv.get_tunable_graph_params()
    all_params = kernel_params + graph_params

    if not all_params:
        logger.info("No tunable parameters declared. Skipping autotuning.")
        return {}

    conv_class_name = type(conv).__name__
    feature_dim = x.shape[1] if x.ndim > 1 else 1

    # check cache
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
            logger.info("Autotuning cache hit for %s. Applying cached config: %s", conv_class_name, cached)
            _apply_best_config(conv, graph_sample, cached, graph_params)
            return cached

    graph_combos = _build_combinations(graph_params)
    kernel_combos = _build_combinations(kernel_params)

    total_trials = len(graph_combos) * len(kernel_combos)
    logger.info(
        "Autotuning %s: %d graph combos x %d kernel combos = %d total trials",
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

            # build benchmark callable that calls forward directly (bypasses hooks)
            if config.tune_backward:

                def _bench_fn(c=conv, xi=x, g=graph_repr):
                    grad_output = torch.randn_like(x)
                    out = c.forward(xi, g)
                    out.backward(grad_output, retain_graph=True)
            else:

                def _bench_fn(c=conv, xi=x, g=graph_repr):
                    c.forward(xi, g)

            result = time_callable(_bench_fn, warmup=config.warmup, iters=config.iters, do_memory_profile=False)
            ms = result.ms_per_iter

            combined_cfg = {**graph_cfg, **kernel_cfg}
            logger.debug("Trial %d/%d: %s -> %.3f ms", trial, total_trials, combined_cfg, ms)

            if ms < best_ms:
                best_ms = ms
                best_config = combined_cfg

    logger.info("Autotuning %s complete. Best config: %s (%.3f ms)", conv_class_name, best_config, best_ms)

    # apply best config
    _apply_best_config(conv, graph_sample, best_config, graph_params)

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
