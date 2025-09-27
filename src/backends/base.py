"""
Base classes for backend implementations and graph convolution layers.

This module provides abstract base classes that define the interface for all backend
implementations and convolution layers in the benchmarking framework.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
import torch
import torch.nn as nn
from enum import Enum

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
        _cache: Internal cache for storing intermediate results
        _profiling_enabled: Flag to enable/disable profiling
    """
    
    def __init__(
        self, 
        device: str = 'cuda', 
        dtype: torch.dtype = torch.float32
    ) -> None:
        """Initialize the backend.

        Args:
            device: Device to use for computations ('cuda' or 'cpu')
            dtype: Data type for tensors (default: torch.float32)
        """
        self.device = torch.device(device)
        self.dtype = dtype
        self._cache: Dict[str, Any] = {}
        self._profiling_enabled: bool = False

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the name of the backend.

        Returns:
            Name identifier for the backend
        """
        pass

    @property
    @abstractmethod
    def supported_formats(self) -> List[GraphFormat]:
        """Get list of supported graph formats.

        Returns:
            List of GraphFormat enums that this backend supports
        """
        pass

    @abstractmethod
    def check_availability(self) -> bool:
        """Check if the backend is available on the current system.

        Returns:
            True if backend is available, False otherwise
        """
        pass

    def enable_profiling(self) -> None:
        """Enable profiling hooks for performance monitoring."""
        self._profiling_enabled = True

    def disable_profiling(self) -> None:
        """Disable profiling hooks."""
        self._profiling_enabled = False

    def clear_cache(self) -> None:
        """Clear all internal caches to free memory."""
        self._cache.clear()

    def get_cache_size(self) -> int:
        """Get the current size of the internal cache.

        Returns:
            Number of items in the cache
        """
        return len(self._cache)


class BaseConvolution(nn.Module):
    """Abstract base class for graph convolution layers.

    This class provides the interface and common functionality for all graph
    convolution implementations across different backends.

    Attributes:
        in_channels: Number of input features
        out_channels: Number of output features
        use_bias: Whether to use bias parameters
        cached: Whether to cache normalized adjacency matrices
        normalize: Whether to apply normalization
        weight: Learnable weight matrix
        bias: Learnable bias vector (optional)
        _cached_result: Cached computation results
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        cached: bool = False,
        normalize: bool = True,
        dropout: float = 0.0,
        **kwargs: Any
    ) -> None:
        """Initialize the graph convolution layer.

        Args:
            in_channels: Number of input feature channels
            out_channels: Number of output feature channels
            bias: Whether to add a learnable bias term
            cached: Whether to cache normalized graph structure
            normalize: Whether to apply symmetric normalization
            dropout: Dropout probability (default: 0.0)
            **kwargs: Additional backend-specific arguments
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_bias = bias
        self.cached = cached
        self.normalize = normalize
        self.dropout = dropout
        
        # Initialize parameters
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
        self._cached_result: Optional[Any] = None
    
    def reset_parameters(self) -> None:
        """Initialize layer parameters using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    @abstractmethod
    def forward(
        self, 
        x: torch.Tensor, 
        graph: Any, 
        **kwargs: Any
    ) -> torch.Tensor:
        """Perform forward pass of the graph convolution.

        Args:
            x: Input node features of shape [num_nodes, in_channels]
            graph: Graph structure in backend-specific format
            **kwargs: Additional backend-specific arguments

        Returns:
            Output node features of shape [num_nodes, out_channels]
        """
        pass
    
    def reset_cache(self) -> None:
        """Reset cached computations to free memory."""
        self._cached_result = None
    
    def extra_repr(self) -> str:
        """Get extra representation string for printing.

        Returns:
            String representation of layer configuration
        """
        return (f'in_channels={self.in_channels}, '
                f'out_channels={self.out_channels}, '
                f'bias={self.use_bias}, '
                f'cached={self.cached}, '
                f'normalize={self.normalize}')
