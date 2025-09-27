"""
Registry system for backend and convolution layer implementations.

This module provides a centralized registry for managing different backend
implementations and their associated convolution layers.
"""

from typing import Dict, Type, Any, List
import logging
from .base import BaseBackend, BaseConvolution

__doc__ = """
Backend and convolution registry module.

This module implements a registry pattern for managing backend implementations
and their associated graph convolution layers. It provides decorators for
registration and factory methods for instantiation.

The registry supports:
- Multiple backend implementations (DGL, PyG, CUDA, etc.)
- Multiple convolution types per backend (GCN, GAT, GraphSAGE, etc.)
- Runtime backend availability checking
- Factory methods for creating instances
"""

logger = logging.getLogger(__name__)


class BackendRegistry:
    """Registry for managing backend implementations and convolution layers.
    
    This class provides a centralized registry for all backend implementations
    and their associated convolution layers. It uses a decorator pattern for
    registration and factory pattern for instantiation.
    
    Attributes:
        _backends: Dictionary mapping backend names to backend classes
        _convolutions: Nested dictionary mapping backends to convolution types
    """
    
    _backends: Dict[str, Type[BaseBackend]] = {}
    _convolutions: Dict[str, Dict[str, Type[BaseConvolution]]] = {}
    
    @classmethod
    def register_backend(cls, name: str) -> Any:
        """Decorator for registering a new backend implementation.

        Args:
            name: Unique name identifier for the backend

        Returns:
            Decorator function that registers the backend class

        Raises:
            Warning: If a backend with the same name already exists
        """
        def decorator(backend_class: Type[BaseBackend]) -> Type[BaseBackend]:
            if name in cls._backends:
                logger.warning(f"Overwriting existing backend: {name}")
            cls._backends[name] = backend_class
            cls._convolutions[name] = {}
            logger.info(f"Registered backend: {name}")
            return backend_class
        return decorator
    
    @classmethod
    def register_convolution(cls, backend: str, conv_type: str) -> Any:
        """Decorator for registering a convolution layer for a specific backend.

        Args:
            backend: Name of the backend this convolution belongs to
            conv_type: Type of convolution (e.g., 'gcn', 'gat', 'sage')

        Returns:
            Decorator function that registers the convolution class

        Raises:
            Warning: If a convolution of the same type already exists for the backend
        """
        def decorator(conv_class: Type[BaseConvolution]) -> Type[BaseConvolution]:
            if backend not in cls._convolutions:
                cls._convolutions[backend] = {}
            if conv_type in cls._convolutions[backend]:
                logger.warning(f"Overwriting {conv_type} for backend {backend}")
            cls._convolutions[backend][conv_type] = conv_class
            logger.info(f"Registered {conv_type} convolution for {backend}")
            return conv_class
        return decorator
    
    @classmethod
    def get_backend(cls, name: str, **kwargs: Any) -> BaseBackend:
        """Factory method to create a backend instance.

        Args:
            name: Name of the backend to instantiate
            **kwargs: Additional arguments to pass to backend constructor

        Returns:
            Instantiated backend object

        Raises:
            ValueError: If backend name is not registered
            RuntimeError: If backend is not available on the system
        """
        if name not in cls._backends:
            available = ", ".join(cls._backends.keys())
            raise ValueError(f"Backend '{name}' not registered. Available: {available}")
        
        backend_class = cls._backends[name]
        backend = backend_class(**kwargs)

        return backend
    
    @classmethod
    def get_convolution(
        cls, 
        backend: str, 
        conv_type: str,
        **kwargs: Any
    ) -> BaseConvolution:
        """Factory method to create a convolution layer for a specific backend.

        Args:
            backend: Name of the backend
            conv_type: Type of convolution layer
            **kwargs: Additional arguments for convolution constructor

        Returns:
            Instantiated convolution layer

        Raises:
            ValueError: If backend or convolution type is not registered
        """
        if backend not in cls._convolutions:
            raise ValueError(f"Backend '{backend}' not registered")
        if conv_type not in cls._convolutions[backend]:
            available = ", ".join(cls._convolutions[backend].keys())
            raise ValueError(f"Convolution '{conv_type}' not available for {backend}. "
                             f"Available: {available}")

        conv_class = cls._convolutions[backend][conv_type]
        return conv_class(**kwargs)
    
    @classmethod
    def list_backends(cls) -> List[str]:
        """Get list of all registered backend names.

        Returns:
            List of registered backend names
        """
        return list(cls._backends.keys())
    
    @classmethod
    def list_convolutions(cls, backend: str) -> List[str]:
        """Get list of convolution types available for a specific backend.

        Args:
            backend: Name of the backend

        Returns:
            List of available convolution types for the backend
        """
        if backend not in cls._convolutions:
            return []
        return list(cls._convolutions[backend].keys())
    
    @classmethod
    def get_backend_info(cls, name: str) -> Dict[str, Any]:
        """Get detailed information about a backend.

        Args:
            name: Name of the backend

        Returns:
            Dictionary containing backend information

        Raises:
            ValueError: If backend is not registered
        """
        if name not in cls._backends:
            raise ValueError(f"Backend '{name}' not registered")
        
        backend_class = cls._backends[name]
        return {
            'name': name,
            'class': backend_class.__name__,
            'module': backend_class.__module__,
            'convolutions': cls.list_convolutions(name)
        }


# convenience functions for module-level access

def register_backend(name: str) -> Any:
    """Register a backend implementation.

    Args:
        name: Unique name for the backend

    Returns:
        Decorator function for registration
    """
    return BackendRegistry.register_backend(name)


def register_convolution(backend: str, conv_type: str) -> Any:
    """Register a convolution layer for a backend.

    Args:
        backend: Name of the backend
        conv_type: Type of convolution

    Returns:
        Decorator function for registration
    """
    return BackendRegistry.register_convolution(backend, conv_type)


def get_backend(name: str, **kwargs: Any) -> BaseBackend:
    """Get a backend instance.

    Args:
        name: Name of the backend
        **kwargs: Additional backend arguments

    Returns:
        Backend instance
    """
    return BackendRegistry.get_backend(name, **kwargs)


def get_convolution(
    backend: str, 
    conv_type: str, 
    **kwargs: Any
) -> BaseConvolution:
    """Get a convolution layer instance.

    Args:
        backend: Name of the backend
        conv_type: Type of convolution
        **kwargs: Additional convolution arguments

    Returns:
        Convolution layer instance
    """
    return BackendRegistry.get_convolution(backend, conv_type, **kwargs)
