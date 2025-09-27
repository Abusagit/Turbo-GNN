from .conv import TorchNativeBackend

doc = """
Torch-native backend (edge-index + torch.sparse CSR/COO baselines).
"""

__all__ = ["TorchNativeBackend"]
