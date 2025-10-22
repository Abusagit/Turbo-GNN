"""
Base classes for backend implementations and graph convolution layers.

This module provides abstract base classes that define the interface for all backend
implementations and convolution layers in the benchmarking framework.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

import torch
import torch.nn as nn

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
            conv_type (str): 'gcn' currently. (Extend with GAT/GIN/SAGE as needed.)
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
