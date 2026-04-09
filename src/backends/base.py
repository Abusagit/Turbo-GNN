"""
Base classes for backend implementations and graph convolution layers.

This module provides abstract base classes that define the interface for all backend
implementations and convolution layers in the benchmarking framework.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

import torch
import torch.nn as nn

from src.data.datasets import GraphSample

# Import core autotune infrastructure from turbo_gnn
from turbo_gnn._autotune import (
    AutotuneConfig,
    TunableKernel,
    TunableParam,
    _InlineAutotuneCache,
    with_autotune,
)

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
    """Enumeration of supported graph format representations."""

    EDGE_INDEX = "edge_index"
    ADJ_MATRIX = "adj_matrix"
    DGL_GRAPH = "dgl_graph"
    CSR = "csr"
    COO = "coo"


# Monkey-patch TunableKernel.autotune for GraphSample-dependent full autotune
# (not included in turbo_gnn because it depends on GraphSample/research code)
def _tunable_kernel_autotune(
    self, x: torch.Tensor, graph_sample: GraphSample, config: AutotuneConfig | None = None
) -> dict:
    """Run autotuning on this kernel callable (GraphSample-dependent)."""
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


TunableKernel.autotune = _tunable_kernel_autotune  # type: ignore


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


class BaseBackend(ABC):
    """Abstract base class for all graph neural network backends."""

    def __init__(self, device: str = "cuda", dtype: torch.dtype = torch.float32) -> None:
        self.device = torch.device(device)
        self.dtype = dtype

    @abstractmethod
    def create_conv(
        self,
        conv_type: str,
        **kwargs: Any,
    ) -> BaseConvolution:
        pass

    def create_aggr(
        self,
        conv_type: str,
        **kwargs: Any,
    ) -> BaseAggr:
        """Create an aggregation-only callable (no linear projections).

        Override in subclasses that support aggregation-only benchmarking.
        """
        raise NotImplementedError(f"create_aggr not implemented for {type(self).__name__}")


class BaseAggr(nn.Module):
    """Aggregation-only callable (no linear projections).

    Subclasses implement ``forward()`` which takes pre-projected tensors
    and a graph, returning aggregated features.  The exact signature
    depends on the conv type (simple aggr vs attention-based).

    """

    def __init__(self, conv_type: str, **kwargs: Any) -> None:
        super().__init__()
        self.conv_type = conv_type

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        pass


class ConvAsAggr(BaseAggr):
    """Wrap a projection-free BaseConvolution as a BaseAggr.

    Use this for backends where the conv module already performs
    pure aggregation without any linear projections.
    """

    def __init__(self, conv: nn.Module) -> None:
        super().__init__(conv_type=conv.__class__.__name__)
        self._conv = conv

    def forward(self, x: torch.Tensor, graph: Any, **kwargs: Any) -> torch.Tensor:
        return self._conv(x, graph, **kwargs)


class BaseConvolution(nn.Module):
    """Abstract base class for graph convolution layers."""

    def __init__(self, bias: bool = True, dropout: float = 0.0, **kwargs: Any) -> None:
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
        pass

    def register_kernel(self, kernel: TunableKernel) -> None:
        """Register a kernel callable for delegation of tunable params."""
        self._kernel_callables.append(kernel)

    def get_tunable_forward_kernel_params(self) -> list[TunableParam]:
        params: list[TunableParam] = []
        for k in self._kernel_callables:
            params.extend(k.get_tunable_forward_kernel_params())
        return params

    def get_tunable_forward_graph_params(self) -> list[TunableParam]:
        params: list[TunableParam] = []
        for k in self._kernel_callables:
            params.extend(k.get_tunable_forward_graph_params())
        return params

    def get_tunable_backward_kernel_params(self) -> list[TunableParam]:
        params: list[TunableParam] = []
        for k in self._kernel_callables:
            params.extend(k.get_tunable_backward_kernel_params())
        return params

    def get_tunable_backward_graph_params(self) -> list[TunableParam]:
        params: list[TunableParam] = []
        for k in self._kernel_callables:
            params.extend(k.get_tunable_backward_graph_params())
        return params

    def configure(self, **kwargs: Any) -> None:
        """Apply tunable parameter values."""
        if not self._kernel_callables:
            for k, v in kwargs.items():
                setattr(self, k, v)
            return

        kernel_param_names: dict[str, TunableKernel] = {}
        for kernel in self._kernel_callables:
            for p in (
                kernel.get_tunable_forward_kernel_params()
                + kernel.get_tunable_forward_graph_params()
                + kernel.get_tunable_backward_kernel_params()
                + kernel.get_tunable_backward_graph_params()
            ):
                kernel_param_names[p.name] = kernel

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

        for kernel in self._kernel_callables:
            kid = id(kernel)
            if kid in kernel_kwargs:
                kernel.configure(**kernel_kwargs[kid])

    def autotune(self, x: torch.Tensor, graph_sample: GraphSample, config: AutotuneConfig | None = None) -> dict:
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
        self._autotune_enabled = True
        if config is not None:
            self._autotune_config = config
        if graph_sample is not None:
            self._graph_sample_ref = graph_sample
        self.register_forward_pre_hook(_autotune_forward_pre_hook)
