"""Autotuning infrastructure for tunable CUDA kernels."""

from __future__ import annotations

import functools
import inspect
import itertools
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, ClassVar

import torch

logger = logging.getLogger(__name__)


@dataclass
class TunableParam:
    """A single tunable parameter for autotuning grid search.

    Attributes:
        name: Prefixed param name (e.g. 'forward_warps_per_block').
        values: Candidate values for grid search.
        default: Default value when not tuning.
    """

    name: str
    values: list
    default: Any


@dataclass
class AutotuneConfig:
    """Configuration for the autotuning engine.

    Attributes:
        warmup: Number of warmup iterations before timing.
        iters: Number of timed iterations.
        tune_backward: Whether to include backward pass in timing.
        cache_dir: Directory for JSON cache files. None disables caching.
        use_cache: Whether to load from cache if available.
    """

    warmup: int = 10
    iters: int = 50
    tune_backward: bool = False
    cache_dir: str | None = None
    use_cache: bool = True


def _build_combinations(params: list[TunableParam]) -> list[dict[str, Any]]:
    """Build all combinations from a list of TunableParam."""
    if not params:
        return [{}]
    names = [p.name for p in params]
    value_lists = [p.values for p in params]
    return [dict(zip(names, combo)) for combo in itertools.product(*value_lists)]


class _InlineAutotuneCache:
    """Tiered in-memory cache for inline autotuning results.

    Tiers: id(graph) -> CSR pointer hash -> (num_nodes, num_edges, feat_dim).
    Cached value: {"kernel_config": dict, "graph_repr": AdjacencyForwardBackwardWithNodeBuckets}
    """

    def __init__(self):
        self._cache: dict[int, dict[int, dict[tuple, dict]]] = {}

    @staticmethod
    def _csr_hash(graph_repr) -> int:
        return hash((graph_repr.forward_indptr.data_ptr(), graph_repr.backward_indptr.data_ptr()))

    @staticmethod
    def _shape_key(graph_repr, feat_dim: int) -> tuple:
        num_nodes = graph_repr.forward_indptr.numel() - 1
        num_edges = graph_repr.forward_indices.numel()
        return (num_nodes, num_edges, feat_dim)

    def lookup(self, graph_repr, feat_dim: int) -> dict | None:
        gid = id(graph_repr)
        tier1 = self._cache.get(gid)
        if tier1 is not None:
            csr_h = self._csr_hash(graph_repr)
            tier2 = tier1.get(csr_h)
            if tier2 is not None:
                key = self._shape_key(graph_repr, feat_dim)
                return tier2.get(key)
        return None

    def store(self, graph_repr, feat_dim: int, result: dict) -> None:
        gid = id(graph_repr)
        if gid not in self._cache:
            self._cache[gid] = {}
        csr_h = self._csr_hash(graph_repr)
        if csr_h not in self._cache[gid]:
            self._cache[gid][csr_h] = {}
        key = self._shape_key(graph_repr, feat_dim)
        self._cache[gid][csr_h][key] = result


class TunableKernel(ABC):
    """Base class for kernel callables that support autotuning.

    Subclasses implement ``_execute`` (the raw kernel invocation) and declare
    tunable parameters via ``get_tunable_*`` methods.
    """

    _shared_instances: ClassVar[dict[tuple, TunableKernel]] = {}

    def __init__(self) -> None:
        self._autotune_enabled: bool = False
        self._is_tuned: bool = False
        self._is_autotuning: bool = False
        self._autotune_config: AutotuneConfig = AutotuneConfig()
        self._inline_cache: _InlineAutotuneCache = _InlineAutotuneCache()

    @abstractmethod
    def _execute(self, graph, x, **kwargs):
        """Raw kernel invocation. graph = AdjacencyForwardBackwardWithNodeBuckets."""
        ...

    def __call__(self, *args, autotune=False, autotune_config=None, **kwargs):
        if autotune and not self._is_autotuning:
            graph = args[0]
            x = args[1]
            extra_args = args[2:]

            feat_dim = x.shape[-1] if x.ndim > 1 else 1
            cached = self._inline_cache.lookup(graph, feat_dim)
            if cached is not None:
                if cached["kernel_config"]:
                    self.configure(**cached["kernel_config"])
                return self._execute(cached["graph_repr"], x, *extra_args, **kwargs)

            config = autotune_config or self._autotune_config
            result = self._inline_autotune(x, graph, config, **kwargs)
            self._inline_cache.store(graph, feat_dim, result)
            return self._execute(result["graph_repr"], x, *extra_args, **kwargs)

        return self._execute(*args, **kwargs)

    # ------ tunable param declarations ------

    def get_tunable_forward_kernel_params(self) -> list[TunableParam]:
        return []

    def get_tunable_forward_graph_params(self) -> list[TunableParam]:
        return []

    def get_tunable_backward_kernel_params(self) -> list[TunableParam]:
        return []

    def get_tunable_backward_graph_params(self) -> list[TunableParam]:
        return []

    # ------ configuration ------

    def configure(self, **kwargs: Any) -> None:
        """Apply tunable parameter values via setattr."""
        for k, v in kwargs.items():
            setattr(self, k, v)

    # ------ benchmarking helpers ------

    def make_forward_bench_fn(self, x: torch.Tensor, graph_repr, **kwargs) -> Callable:
        def _bench():
            return self._execute(graph_repr, x, **kwargs)

        return _bench

    def make_backward_bench_fn(self, x: torch.Tensor, graph_repr, **kwargs) -> Callable:
        fwd_fn = self.make_forward_bench_fn(x, graph_repr, **kwargs)
        out = fwd_fn()
        if out is None or not isinstance(out, torch.Tensor):
            raise RuntimeError(
                f"{type(self).__name__}.make_forward_bench_fn must return a tensor "
                "for default make_backward_bench_fn to work"
            )
        grad = torch.randn_like(out)

        def _bench():
            result = fwd_fn()
            result.backward(grad, retain_graph=True)

        return _bench

    # ------ inline autotuning ------

    def _inline_autotune(self, x, graph_repr, config=None, **kwargs):
        """Full grid search over graph + kernel params. Returns result dict."""
        from turbo_gnn._timer import time_callable

        config = config or self._autotune_config
        kernel_params = self.get_tunable_forward_kernel_params()
        graph_params = self.get_tunable_forward_graph_params()
        if not kernel_params and not graph_params:
            return {"kernel_config": {}, "graph_repr": graph_repr}

        graph_combos = _build_combinations(graph_params)
        kernel_combos = _build_combinations(kernel_params)
        best_ms, best_result = float("inf"), {"kernel_config": {}, "graph_repr": graph_repr}
        self._is_autotuning = True
        try:
            for graph_cfg in graph_combos:
                current_graph = graph_repr.repartition(**graph_cfg) if graph_cfg else graph_repr
                for kernel_cfg in kernel_combos:
                    if kernel_cfg:
                        self.configure(**kernel_cfg)
                    try:
                        bench_fn = self.make_forward_bench_fn(x, current_graph, **kwargs)
                        ms = time_callable(
                            bench_fn,
                            warmup=config.warmup,
                            iters=config.iters,
                        ).ms_per_iter
                    except RuntimeError:
                        logger.debug("Skipping invalid config: graph=%s kernel=%s", graph_cfg, kernel_cfg)
                        continue
                    if ms < best_ms:
                        best_ms = ms
                        best_result = {"kernel_config": kernel_cfg, "graph_repr": current_graph}
        finally:
            self._is_autotuning = False
        if best_result["kernel_config"]:
            self.configure(**best_result["kernel_config"])
        return best_result

    # ------ singleton factory ------

    @classmethod
    def _get_or_create(cls, **init_kwargs) -> TunableKernel:
        key = (cls.__name__, tuple(sorted(init_kwargs.items())))
        if key not in TunableKernel._shared_instances:
            TunableKernel._shared_instances[key] = cls(**init_kwargs)
        return TunableKernel._shared_instances[key]

    @property
    def name(self) -> str:
        return type(self).__name__


def with_autotune(kernel_class, *, init_params=()):
    """Decorator that adds autotune=True support to a kernel function.

    When autotune=False (default): calls the original function as-is.
    When autotune=True: uses kernel_class singleton for autotuning + cached execution.

    Args:
        kernel_class: TunableKernel subclass.
        init_params: Function kwarg names forwarded to kernel __init__ (e.g. ("reduce",)).
    """

    def decorator(fn):
        sig = inspect.signature(fn)
        param_names = list(sig.parameters.keys())

        @functools.wraps(fn)
        def wrapper(*args, autotune=False, autotune_config=None, **kwargs):
            if not autotune:
                return fn(*args, **kwargs)

            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            all_kw = dict(bound.arguments)

            graph = all_kw.pop(param_names[0])
            x = all_kw.pop(param_names[1])

            init_kw = {p: all_kw[p] for p in init_params if p in all_kw}
            exec_kw = {k: v for k, v in all_kw.items() if k not in init_params}

            kernel = kernel_class._get_or_create(**init_kw)

            feat_dim = x.shape[-1] if x.ndim > 1 else 1
            cached = kernel._inline_cache.lookup(graph, feat_dim)
            if cached is not None:
                if cached["kernel_config"]:
                    kernel.configure(**cached["kernel_config"])
                return kernel._execute(cached["graph_repr"], x, **exec_kw)

            config = autotune_config or kernel._autotune_config
            result = kernel._inline_autotune(x, graph, config, **exec_kw)
            kernel._inline_cache.store(graph, feat_dim, result)
            return kernel._execute(result["graph_repr"], x, **exec_kw)

        return wrapper

    return decorator
