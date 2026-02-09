import typing as tp
from typing import Any, Literal

import torch
from torch import nn

from src.data.converters import AdjacencyForwardBackwardWithNodeBuckets

from ..base import BaseBackend
from ..registry import BackendRegistry
from .dot_aggr.utils import dot_aggr
from .mean_aggr.utils import mean_aggr

doc = """
CUDA backend: wraps cuda-written kernels .
"""


class _Cuda_test_conv(nn.Module):
    """
    Min-aggregation convolution using custom CUDA extension.

    Expects:
      - x: [N, F] float32 cuda
      - graph: (edge_ptr, edge_idx) where
            edge_ptr: [N+1] int32 cuda
            edge_idx: [E]   int32 cuda
      - light/heavy node partitions are stored as buffers inside MinAggr module
    """

    def __init__(
        self,
        /,
        conv_type: tp.Literal["mean", "dot"],
        **kwargs,
    ) -> None:
        super().__init__()

        self.kernel_kind = kwargs.get("kernel_kind", 1)
        self.use_second_access = kwargs.get("use_second_access", False)
        self.use_vectorized_loads = kwargs.get("use_vectorized_loads", False)

        if conv_type == "dot":
            self.conv = dot_aggr
        elif conv_type == "mean":
            self.conv = mean_aggr

    def forward(
        self,
        x: torch.Tensor,
        graph: AdjacencyForwardBackwardWithNodeBuckets,
        *,
        edge_weight: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        edge_ptr = graph.forward_indptr
        edge_idx = graph.forward_indices
        return self.conv(
            edge_ptr,
            edge_idx,
            x,
            self.kernel_kind,
            self.use_second_access,
            self.use_vectorized_loads,
        )


@BackendRegistry.register_backend("cuda_test")
class CUDABTestackend(BaseBackend):
    """Backend that instantiates CUDA-based convolutions."""

    def create_conv(
        self,
        conv_type: str,
        **kwargs: Any,
    ):
        """Factory for CUDA convolution layers.

        Args:
            conv_type (str): 'gat_v2' or 'min_aggr' currently.
            feature_dim (int): Input (and output) feature size.
            **kwargs (Any): Extra arguments for CUDA layers.

        Returns:
            BaseConvolution: An instance of the requested CUDA conv.
        """
        feature_dim = kwargs.pop("feature_dim")
        heads = kwargs.pop("heads", 1)

        ct = conv_type.lower()
        match ct:
            case "mean_aggr":
                return _Cuda_test_conv(
                    conv_type="mean",
                    **kwargs,
                )
            case "dot_aggr":
                return _Cuda_test_conv(
                    conv_type="dot",
                    **kwargs,
                )

        raise KeyError(f"Unsupported conv_type for DGL backend: {conv_type}")
