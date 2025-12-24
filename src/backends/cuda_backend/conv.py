from typing import Any, Literal

import torch

from ..base import BaseBackend, BaseConvolution
from ..registry import BackendRegistry
from .min_aggr.utils import min_aggr

doc = """
CUDA Backend: custom CUDA extensions (min_aggr + backward).
Expects CSR graph representation: (edge_ptr, edge_idx).
"""


class _CudaMinAggrConv(BaseConvolution):
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
        *,
        bias: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(bias=bias, **kwargs)
        if bias:
            raise NotImplementedError("bias=True is not supported for pure min aggregation in this backend.")
        self.op = None

    def forward(
        self,
        x: torch.Tensor,
        graph: Any,
        *,
        edge_weight: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if len(graph) >= 4:
            edge_ptr, edge_idx, light, heavy = graph[0], graph[1], graph[2], graph[3]
        elif len(graph) == 3:
            edge_ptr, edge_idx, _w = graph
            raise ValueError("cuda_backend needs (edge_ptr, edge_idx, light, heavy) in graph.")
        else:
            edge_ptr, edge_idx = graph
            raise ValueError("cuda_backend needs (edge_ptr, edge_idx, light, heavy) in graph.")

        return min_aggr(edge_ptr, edge_idx, x, light, heavy)


class _CudaSimpleAggrConv(BaseConvolution):
    def __init__(
        self,
        aggr_type: Literal["min"] = "min",
        *,
        bias: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(bias=bias, **kwargs)
        if aggr_type != "min":
            raise NotImplementedError(f"Only aggr_type='min' is implemented, got {aggr_type}")
        self.conv = _CudaMinAggrConv(bias=bias)

    def forward(
        self,
        x: torch.Tensor,
        graph: Any,
        *,
        edge_weight: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        return self.conv(x, graph, edge_weight=edge_weight, **kwargs)


@BackendRegistry.register_backend("cuda")
class CudaBackend(BaseBackend):
    """Backend instantiating custom CUDA-powered convolutions."""

    def create_conv(self, conv_type: str, **kwargs: Any):
        """
        Factory for CUDA backend convs.

        Supported:
          - "min_aggr"
        """
        _ = kwargs.pop("feature_dim", None)

        if conv_type == "min_aggr":
            return _CudaSimpleAggrConv(
                aggr_type="min",
                bias=False,
            )

        raise KeyError(f"Unsupported conv_type for cuda backend: {conv_type}")
