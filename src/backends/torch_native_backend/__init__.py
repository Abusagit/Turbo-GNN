from .conv import (
    TorchNativeAdjMatBackend,
    TorchNativeBackend,
    TorchNativeGCNBackend,
    TorchNativeMeanAggrBackend,
    TorchNativeSumAggrBackend,
)

doc = """
Torch-native backend (edge-index + torch.sparse CSR/COO baselines).
"""

__all__ = [
    "TorchNativeBackend",
    "TorchNativeGCNBackend",
    "TorchNativeMeanAggrBackend",
    "TorchNativeSumAggrBackend",
    "TorchNativeAdjMatBackend",
]
