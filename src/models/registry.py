from collections.abc import Callable
from typing import Any, Dict

import torch.nn as nn

doc = """
Simple registry for model architectures.

- @register("name"): decorator to register an nn.Module builder/class
- build("name", **kwargs): instantiate a registered model
"""


class ModelRegistry:
    """Registry mapping a string name to a model factory or nn.Module subclass."""

    _items: dict[str, Callable[..., nn.Module]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[Callable[..., nn.Module]], Callable[..., nn.Module]]:
        """Register a model builder/class under a name.

        Args:
            name (str): Unique key for the model (e.g., 'node_classifier').

        Returns:
            Callable[..., nn.Module]: The same callable, after registration.
        """

        def deco(fn: Callable[..., nn.Module]) -> Callable[..., nn.Module]:
            cls._items[name] = fn
            return fn

        return deco

    @classmethod
    def build(cls, name: str, **kwargs: Any) -> nn.Module:
        """Instantiate a registered model.

        Args:
            name (str): Model name to build.
            **kwargs (Any): Keyword args forwarded to the model factory/constructor.

        Returns:
            nn.Module: The constructed model.

        Raises:
            KeyError: If the name is not registered.
        """
        if name not in cls._items:
            raise KeyError(f"Unknown model '{name}'. Registered: {list(cls._items)}")
        return cls._items[name](**kwargs)


# Convenience free functions
def register(name: str) -> Callable[[Callable[..., nn.Module]], Callable[..., nn.Module]]:
    """Decorator to register a model in the global ModelRegistry.

    Args:
        name (str): Unique model name.

    Returns:
        Callable[..., nn.Module]: Decorator that registers the target callable.
    """
    return ModelRegistry.register(name)


def build(name: str, **kwargs: Any) -> nn.Module:
    """Instantiate a model from the global registry.

    Args:
        name (str): Model name.
        **kwargs (Any): Constructor kwargs.

    Returns:
        nn.Module: Instantiated model.
    """
    return ModelRegistry.build(name, **kwargs)
