"""
Base classes for backend implementations and graph convolution layers.

This module provides abstract base classes that define the interface for all backend
implementations and convolution layers in the benchmarking framework.
"""

from __future__ import annotations

import functools
import inspect
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, ClassVar

import torch
import torch.nn as nn

from src.data.datasets import GraphSample

logger = logging.getLogger(__name__)

__doc__ = """
Base module for backend implementations.

This module defines the core abstractions for graph neural network backends including:
- GraphFormat: Enum for supported graph representations
- BaseBackend: Abstract base class for backend implementations
- BaseConvolution: Abstract base class for graph convolution layers

The module ensures consistent interfaces across different backend implementations
(DGL, PyG, CUDA, etc.) and provides common functionality like caching and profiling.
"""


class GraphFormat(Enum):
    """Enumeration of supported graph format representations.

    Attributes:
        EDGE_INDEX: PyTorch Geometric style edge list format
        ADJ_MATRIX: Dense adjacency matrix format
        DGL_GRAPH: DGL graph object format
        CSR: Compressed Sparse Row format
        COO: Coordinate format for sparse matrices

    NOTE can be expanded if we find something interesting and useful

    """

    EDGE_INDEX = "edge_index"
    ADJ_MATRIX = "adj_matrix"
    DGL_GRAPH = "dgl_graph"
    CSR = "csr"
    COO = "coo"


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


def _autotune_forward_pre_hook(module: BaseConvolution, args):
    """Lazy autotuning: triggers on first forward when autotune is enabled."""
    if not (module._autotune_enabled and not module._is_tuned and not module._is_autotuning):
        return None

    x = args[0]
    graph = args[1] if len(args) > 1 else None
    graph_sample = module._graph_sample_ref

    # also accept GraphSample passed directly as graph arg.
    if graph_sample is None and graph is not None and isinstance(graph, GraphSample):
        graph_sample = graph

    if graph_sample is None:
        logger.warning("autotune=True but no GraphSample available. Skipping autotuning.")
        module._is_tuned = True
        return None

    module.autotune(x, graph_sample)
    return None


class _InlineAutotuneCache:
    """Tiered in-memory cache for inline autotuning results.

    Tiers: id(graph) → CSR pointer hash → (num_nodes, num_edges, feat_dim).
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
    tunable parameters via ``get_tunable_*`` methods.  This is *not* an
    ``nn.Module`` — it is a lightweight callable that can be used
    standalone or registered on a :class:`BaseConvolution`.
    """

    _shared_instances: ClassVar[dict[tuple, TunableKernel]] = {}

    def __init__(self) -> None:
        # autotuning state (mirrors BaseConvolution)
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
        """Autotune-aware call (for direct kernel usage, e.g. from conv modules).

        When autotune=True and not currently autotuning, performs inline
        autotuning with caching. Otherwise delegates to _execute.
        """
        if autotune and not self._is_autotuning:
            # Expect args = (graph, x, ...)
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

        # Non-autotune path: pass all args through to _execute
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
        """Return a zero-arg callable that runs the forward kernel once.

        Default implementation calls _execute. Subclasses can override.
        """

        def _bench():
            return self._execute(graph_repr, x, **kwargs)

        return _bench

    def make_backward_bench_fn(self, x: torch.Tensor, graph_repr, **kwargs) -> Callable:
        """Return a zero-arg callable that runs forward + backward once.

        Default implementation calls :meth:`make_forward_bench_fn`, runs it
        once to get an output, then returns a closure that calls ``.backward()``.
        """
        fwd_fn = self.make_forward_bench_fn(x, graph_repr, **kwargs)
        # Run forward once to capture the output for backward
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
        import src.benchmarking.microbench as _microbench
        from src.backends.autotune import _build_combinations

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
                        ms = _microbench.time_callable(
                            bench_fn,
                            warmup=config.warmup,
                            iters=config.iters,
                            do_memory_profile=False,
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

    # ------ autotuning (GraphSample-based, for conv module usage) ------

    def autotune(self, x: torch.Tensor, graph_sample: GraphSample, config: AutotuneConfig | None = None) -> dict:
        """Run autotuning on this kernel callable."""
        from src.backends.autotune import run_autotune_kernel

        if config is not None:
            self._autotune_config = config

        self._is_autotuning = True
        try:
            best = run_autotune_kernel(self, x, graph_sample, self._autotune_config)
        finally:
            self._is_autotuning = False

        self._is_tuned = True
        return best

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

            # Bind all args to parameter names
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            all_kw = dict(bound.arguments)

            graph = all_kw.pop(param_names[0])  # graph repr
            x = all_kw.pop(param_names[1])  # input features

            # Separate init kwargs from execution kwargs
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


class BaseBackend(ABC):
    """Abstract base class for all graph neural network backends.

    This class defines the interface that all backend implementations must follow,
    ensuring consistency across different implementations (DGL, PyG, CUDA, etc.).

    Attributes:
        device: The torch device to use for computations
        dtype: The data type for tensors
    """

    def __init__(self, device: str = "cuda", dtype: torch.dtype = torch.float32) -> None:
        """Initialize the backend.

        Args:
            device: Device to use for computations ('cuda' or 'cpu')
            dtype: Data type for tensors (default: torch.float32)
        """
        self.device = torch.device(device)
        self.dtype = dtype

    @abstractmethod
    def create_conv(
        self,
        conv_type: str,
        **kwargs: Any,
    ) -> BaseConvolution:
        """Factory for convolution layers.

        Args:
            conv_type (str): 'gcn' currently. (Extend with GATv2/GIN/SAGE as needed.)
            **kwargs (Any): Extra arguments for some layers.

        Returns:
            BaseConvolution: An instance of the requested conv.
        """
        pass


class BaseConvolution(nn.Module):
    """Abstract base class for graph convolution layers.

    This class provides the interface and common functionality for all graph
    convolution implementations across different backends.

    Attributes:
        use_bias: Whether to use bias parameters
        cached: Whether to cache normalized adjacency matrices
        normalize: Whether to apply normalization
        weight: Learnable weight matrix
        bias: Learnable bias vector (optional)
    """

    def __init__(self, bias: bool = True, dropout: float = 0.0, **kwargs: Any) -> None:
        """Initialize the graph convolution layer.

        Args:
            bias: Whether to add a learnable bias term
            dropout: Dropout probability (default: 0.0)
            **kwargs: Additional backend-specific arguments
        """
        super().__init__()
        self.use_bias = bias
        self.dropout = dropout

        # kernel callable delegation
        self._kernel_callables: list[TunableKernel] = []

        # autotuning state
        self._autotune_enabled: bool = False
        self._is_tuned: bool = False
        self._is_autotuning: bool = False
        self._autotune_config: AutotuneConfig = AutotuneConfig()
        self._graph_sample_ref: GraphSample | None = None

    @abstractmethod
    def forward(self, x: torch.Tensor, graph: Any, **kwargs: Any) -> torch.Tensor:
        """Perform forward pass of the graph convolution.

        Args:
            x: Input node features of shape [num_nodes, feature_dim]
            graph: Graph structure in backend-specific format
            **kwargs: Additional backend-specific arguments

        Returns:
            Output node features of shape [num_nodes, feature_dim]
        """
        pass

    def register_kernel(self, kernel: TunableKernel) -> None:
        """Register a kernel callable for delegation of tunable params."""
        self._kernel_callables.append(kernel)

    def get_tunable_forward_kernel_params(self) -> list[TunableParam]:
        """Return kernel parameter search space for autotuning.

        Aggregates from all registered kernel callables by default.
        Override in subclasses to declare tunable kernel parameters directly.
        """
        params: list[TunableParam] = []
        for k in self._kernel_callables:
            params.extend(k.get_tunable_forward_kernel_params())
        return params

    def get_tunable_forward_graph_params(self) -> list[TunableParam]:
        """Return graph parameter search space for autotuning.

        Aggregates from all registered kernel callables by default.
        Override in subclasses to declare tunable graph-level parameters directly.
        """
        params: list[TunableParam] = []
        for k in self._kernel_callables:
            params.extend(k.get_tunable_forward_graph_params())
        return params

    def get_tunable_backward_kernel_params(self) -> list[TunableParam]:
        """Backward-pass kernel parameter search space."""
        params: list[TunableParam] = []
        for k in self._kernel_callables:
            params.extend(k.get_tunable_backward_kernel_params())
        return params

    def get_tunable_backward_graph_params(self) -> list[TunableParam]:
        """Backward-pass graph parameter search space."""
        params: list[TunableParam] = []
        for k in self._kernel_callables:
            params.extend(k.get_tunable_backward_graph_params())
        return params

    def configure(self, **kwargs: Any) -> None:
        """Apply tunable parameter values.

        Routes params to the kernel callable that owns them.
        Remaining params are set on self via setattr.
        """
        if not self._kernel_callables:
            for k, v in kwargs.items():
                setattr(self, k, v)
            return

        # build mapping: param_name -> kernel
        kernel_param_names: dict[str, TunableKernel] = {}
        for kernel in self._kernel_callables:
            for p in (
                kernel.get_tunable_forward_kernel_params()
                + kernel.get_tunable_forward_graph_params()
                + kernel.get_tunable_backward_kernel_params()
                + kernel.get_tunable_backward_graph_params()
            ):
                kernel_param_names[p.name] = kernel

        # group kwargs by owner
        kernel_kwargs: dict[int, dict[str, Any]] = {}
        for k, v in kwargs.items():
            owner = kernel_param_names.get(k)
            if owner is not None:
                kid = id(owner)
                if kid not in kernel_kwargs:
                    kernel_kwargs[kid] = {}
                kernel_kwargs[kid][k] = v
            else:
                setattr(self, k, v)

        # apply grouped kwargs to each kernel
        for kernel in self._kernel_callables:
            kid = id(kernel)
            if kid in kernel_kwargs:
                kernel.configure(**kernel_kwargs[kid])

    def autotune(self, x: torch.Tensor, graph_sample: GraphSample, config: AutotuneConfig | None = None) -> dict:
        """Explicitly run autotuning to find optimal parameters.

        Args:
            x: Input node features for benchmarking.
            graph_sample: GraphSample instance (needed for graph param tuning).
            config: Optional config override.

        Returns:
            Dict of best parameter name -> value mappings.
        """
        from src.backends.autotune import run_autotune

        if config is not None:
            self._autotune_config = config

        self._is_autotuning = True
        try:
            best = run_autotune(self, x, graph_sample, self._autotune_config)
        finally:
            self._is_autotuning = False

        self._is_tuned = True
        return best

    def enable_autotune(self, config: AutotuneConfig | None = None, graph_sample: GraphSample | None = None) -> None:
        """Enable lazy autotuning that triggers on first forward() call.

        Args:
            config: Optional autotuning configuration.
            graph_sample: Optional GraphSample for graph param tuning.
        """
        self._autotune_enabled = True
        if config is not None:
            self._autotune_config = config
        if graph_sample is not None:
            self._graph_sample_ref = graph_sample
        self.register_forward_pre_hook(_autotune_forward_pre_hook)
