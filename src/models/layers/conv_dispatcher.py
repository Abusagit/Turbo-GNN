from typing import Any

from ...backends.registry import BackendRegistry

doc = """
Dispatcher that instantiates a convolution for (conv_type, backend) via BackendRegistry.
"""


def create_conv_layer(
    conv_type: str,
    backend: str,
    in_channels: int,
    out_channels: int,
    **kwargs: Any,
):
    """
    Create a convolution layer via the backend registry.

    Args:
        conv_type (str): Convolution type ("gcn", "gat", "sage", "gin", ...).
        backend (str): Backend name ("pyg", "dgl", "torch_native", ...).
        in_channels (int): Input feature size.
        out_channels (int): Output feature size.
        **kwargs (Any): Additional layer params (heads, bias, aggr, etc).

    Returns:
        torch.nn.Module: A backend-specific convolution layer instance.
    """

    if conv_type.lower() != "gat":
        kwargs.pop("heads", None)
        kwargs.pop("concat", None)

    backend_inst = BackendRegistry.get_backend(backend)
    return backend_inst.create_conv(conv_type, in_channels, out_channels, **kwargs)
