from typing import Any, Optional

import torch
from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch.conv import GATv2Conv as _GAT

from ..base import BaseBackend, BaseConvolution
from ..registry import BackendRegistry
from .utils import csr_SPMM_normalized

doc = """
CuSparse backend: wraps CuSparse matmul-based convolutions behind the BaseBackend interface.
"""


class _СuSparseMatMulConvFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, graph, norm_type: str, cu_sparse_algorithm_id: int = -1):
        ctx.save_for_backward(*graph)
        ctx.norm_type = norm_type
        ctx.cu_sparse_algorithm_id = cu_sparse_algorithm_id

        row_pointers, column_indices, edge_weight = graph
        return csr_SPMM_normalized(
            indptr=row_pointers,
            indices=column_indices,
            features=x,
            edge_weights=edge_weight,
            norm=norm_type,
            algorithm=cu_sparse_algorithm_id,
            do_transpose_a=False,
        )

    @staticmethod
    def backward(ctx, grad_out):
        row_pointers, column_indices, edge_weight = ctx.saved_tensors

        grad_x = csr_SPMM_normalized(
            indptr=row_pointers,
            indices=column_indices,
            features=grad_out,
            edge_weights=edge_weight,
            norm=ctx.norm_type,
            algorithm=ctx.cu_sparse_algorithm_id,
            do_transpose_a=True,
        )
        return grad_x, None, None, None


class _СuSparseMatMulConv(BaseConvolution):
    """CuSparse-backend MatMulConv wrapper."""

    def __init__(self, norm_type: str, cu_sparse_algorithm_id: int):
        super().__init__(0, 0, False, 0)

        assert norm_type in ("none", "right", "left", "both")
        assert cu_sparse_algorithm_id in (-1, 0, 1, 2, 3)

        self.norm_type = norm_type
        self.cu_sparse_algorithm_id = cu_sparse_algorithm_id

    def forward(
        self,
        x: torch.Tensor,
        graph: tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
        **kwargs: Any,
    ) -> torch.Tensor:
        """Apply GraphConv.

        Args:
            x (torch.Tensor): Node features [N, Fin].
            graph (tuple[torch.Tensor, torch.Tensor], Optional[torch.Tensor]):
                Adj matrix in CSR format. (row pointers, column indices, edge weights).
            **kwargs (Any): Extra kwargs (ignored).

        Returns:
            torch.Tensor: Output features [N, Fout].
        """

        return _СuSparseMatMulConvFn.apply(x, graph, self.norm_type, self.cu_sparse_algorithm_id)


@BackendRegistry.register_backend("cusparse")
class СuSparseBackend(BaseBackend):
    """Backend that instantiates cusparse-based convolutions. Only matmul-based convolutions are supported."""

    def create_conv(
        self,
        conv_type: str,
        in_channels: int,
        out_channels: int,
        cu_sparse_algorithm_id: int = -1,
        **kwargs: Any,
    ) -> BaseConvolution:
        """Factory for cusparse convolution layers.

        Args:
            conv_type (str): 'sum', 'mean', 'random_walk', 'gcn'
            in_channels (int): Input feature size.
            out_channels (int): Output feature size.
            cu_sparse_algorithm_id (int): algorithm for CuSparse to use: -1 (default), 0, 1, 2, 3.
            **kwargs (Any): ignored.

        Returns:
            BaseConvolution: An instance of the requested CuSparse conv.
        """

        conv_type = conv_type.lower()

        if conv_type == "sum_aggr":
            return _СuSparseMatMulConv(norm_type="none", cu_sparse_algorithm_id=cu_sparse_algorithm_id)
        if conv_type == "mean_aggr":
            return _СuSparseMatMulConv(norm_type="right", cu_sparse_algorithm_id=cu_sparse_algorithm_id)
        if conv_type == "random_walk":
            return _СuSparseMatMulConv(norm_type="left", cu_sparse_algorithm_id=cu_sparse_algorithm_id)
        if conv_type == "gcn":
            return _СuSparseMatMulConv(norm_type="both", cu_sparse_algorithm_id=cu_sparse_algorithm_id)
        raise KeyError(f"Unsupported conv_type for DGL backend: {conv_type}")
