from .conv import TorchNativeGCNBackend, TorchNativeMeanAggrBackend, TorchNativeSumAggrBackend

doc = """
Torch-native backend (edge-index + torch.sparse CSR/COO baselines).
"""

__all__ = [
    "TorchNativeGCNBackend",
    "TorchNativeMeanAggrBackend",
    "TorchNativeSumAggrBackend",
]
