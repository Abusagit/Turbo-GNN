"""
Base classes for backend implementations and graph convolution layers.

This module provides abstract base classes that define the interface for all backend
implementations and convolution layers in the benchmarking framework.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

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

    def get_tunable_forward_kernel_params(self) -> list[TunableParam]:
        """Return kernel parameter search space for autotuning.

        Override in subclasses to declare tunable kernel parameters.
        """
        return []

    def get_tunable_forward_graph_params(self) -> list[TunableParam]:
        """Return graph parameter search space for autotuning.

        Override in subclasses to declare tunable graph-level parameters.
        """
        return []

    def get_tunable_backward_kernel_params(self) -> list[TunableParam]:
        """Backward-pass kernel parameter search space."""
        return []

    def get_tunable_backward_graph_params(self) -> list[TunableParam]:
        """Backward-pass graph parameter search space."""
        return []

    def configure(self, **kwargs: Any) -> None:
        """Apply tunable parameter values.

        Sets attributes using the full prefixed names from TunableParam.
        Subclasses can override to delegate to inner convolutions or
        apply custom logic.
        """
        for k, v in kwargs.items():
            setattr(self, k, v)

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
