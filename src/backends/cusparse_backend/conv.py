from typing import Any, Optional

import torch

from ..base import BaseAggr, BaseBackend, BaseConvolution, ConvAsAggr
from ..registry import BackendRegistry
from .utils import csr_SPMM_normalized

doc = """
CuSparse backend: wraps CuSparse matmul-based convolutions behind the BaseBackend interface.
"""


class _СuSparseMatMulConvFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, graph, norm_type: str, cu_sparse_algorithm_id: int, block_dim: int):
        ctx.save_for_backward(*graph)
        ctx.norm_type = norm_type
        ctx.cu_sparse_algorithm_id = cu_sparse_algorithm_id
        ctx.block_dim = block_dim

        row_pointers, column_indices, edge_weight = graph
        return csr_SPMM_normalized(
            indptr=row_pointers,
            indices=column_indices,
            features=x,
            edge_weights=edge_weight,
            norm=norm_type,
            algorithm=cu_sparse_algorithm_id,
            do_transpose_a=False,
            block_dim=block_dim,
        )

    @staticmethod
    def backward(ctx, *grad_outputs):
        row_pointers, column_indices, edge_weight = ctx.saved_tensors

        grad_x = csr_SPMM_normalized(
            indptr=row_pointers,
            indices=column_indices,
            features=grad_outputs[0],
            edge_weights=edge_weight,
            norm=ctx.norm_type,
            algorithm=ctx.cu_sparse_algorithm_id,
            do_transpose_a=True,
            block_dim=ctx.block_dim,
        )
        return grad_x, None, None, None, None


class _СuSparseMatMulConvPrecomputedBWDMatrixFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, graph, norm_type: str, cu_sparse_algorithm_id: int, block_dim: int):
        fwd_row_pointers, fwd_column_indices, fwd_edge_weight, bwd_row_pointers, bwd_column_indices, bwd_edge_weight = (
            graph
        )

        ctx.save_for_backward(bwd_row_pointers, bwd_column_indices, bwd_edge_weight)
        ctx.norm_type = norm_type
        ctx.cu_sparse_algorithm_id = cu_sparse_algorithm_id
        ctx.block_dim = block_dim

        return csr_SPMM_normalized(
            indptr=fwd_row_pointers,
            indices=fwd_column_indices,
            features=x,
            edge_weights=fwd_edge_weight,
            norm=norm_type,
            algorithm=cu_sparse_algorithm_id,
            do_transpose_a=False,
            block_dim=block_dim,
        )

    @staticmethod
    def backward(ctx, *grad_outputs):
        bwd_row_pointers, bwd_column_indices, bwd_edge_weight = ctx.saved_tensors

        grad_x = csr_SPMM_normalized(
            indptr=bwd_row_pointers,
            indices=bwd_column_indices,
            features=grad_outputs[0],
            edge_weights=bwd_edge_weight,
            norm=ctx.norm_type,
            algorithm=ctx.cu_sparse_algorithm_id,
            do_transpose_a=False,
            block_dim=ctx.block_dim,
        )

        return grad_x, None, None, None, None


class _СuSparseMatMulConv(BaseConvolution):
    """CuSparse-backend MatMulConv wrapper."""

    def __init__(self, norm_type: str, cu_sparse_algorithm_id: int, block_dim: int):
        super().__init__(bias=False, dropout=0.0)

        assert norm_type in ("none", "right", "left", "both")
        assert cu_sparse_algorithm_id in (-1, 0, 1, 2, 3)

        self.norm_type = norm_type
        self.cu_sparse_algorithm_id = cu_sparse_algorithm_id
        self.block_dim = block_dim

    def forward(
        self,
        x: torch.Tensor,
        graph: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Apply GraphConv.

        Args:
            x (torch.Tensor): Node features [N, Fin].
            graph (tuple[torch.Tensor, torch.Tensor], Optional[torch.Tensor]):
                Adj matrix in CSR format. (row pointers, column indices, edge weights)
                OR two matrices for forward and backward.
            **kwargs (Any): Extra kwargs (ignored).

        Returns:
            torch.Tensor: Output features [N, Fout].
        """

        return _СuSparseMatMulConvFn.apply(x, graph, self.norm_type, self.cu_sparse_algorithm_id, self.block_dim)


class _СuSparseMatMulPrecomputedBWDConv(_СuSparseMatMulConv):
    """CuSparse-backend MatMulConv wrapper."""

    def forward(
        self,
        x: torch.Tensor,
        graph: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Apply GraphConv.

        Args:
            x (torch.Tensor): Node features [N, Fin].
            graph (tuple[torch.Tensor, torch.Tensor], Optional[torch.Tensor]):
                Adj matrix in CSR format. (row pointers, column indices, edge weights)
                OR two matrices for forward and backward.
            **kwargs (Any): Extra kwargs (ignored).

        Returns:
            torch.Tensor: Output features [N, Fout].
        """

        return _СuSparseMatMulConvPrecomputedBWDMatrixFn.apply(
            x, graph, self.norm_type, self.cu_sparse_algorithm_id, self.block_dim
        )


@BackendRegistry.register_backend("cusparse")
class СuSparseBackend(BaseBackend):
    """Backend that instantiates cusparse-based convolutions. Only matmul-based convolutions are supported."""

    def create_conv(
        self,
        conv_type: str,
        cu_sparse_algorithm_id: int = -1,
        block_dim: int = 256,
        **kwargs: Any,
    ) -> BaseConvolution:
        """Factory for cusparse convolution layers.

        Args:
            conv_type (str): 'sum', 'mean', 'random_walk', 'gcn'
            cu_sparse_algorithm_id (int): algorithm for CuSparse to use: -1 (default), 0, 1, 2, 3.
            **kwargs (Any): ignored.

        Returns:
            BaseConvolution: An instance of the requested CuSparse conv.
        """

        conv_type = conv_type.lower()

        if conv_type == "sum_aggr":
            return _СuSparseMatMulConv(
                norm_type="none", cu_sparse_algorithm_id=cu_sparse_algorithm_id, block_dim=block_dim
            )
        if conv_type == "mean_aggr":
            return _СuSparseMatMulConv(
                norm_type="right", cu_sparse_algorithm_id=cu_sparse_algorithm_id, block_dim=block_dim
            )
        if conv_type == "random_walk":
            return _СuSparseMatMulConv(
                norm_type="left", cu_sparse_algorithm_id=cu_sparse_algorithm_id, block_dim=block_dim
            )
        if conv_type == "gcn":
            return _СuSparseMatMulConv(
                norm_type="both", cu_sparse_algorithm_id=cu_sparse_algorithm_id, block_dim=block_dim
            )
        raise KeyError(f"Unsupported conv_type for CuSparse backend: {conv_type}")

    def create_aggr(self, conv_type: str, **kwargs: Any) -> BaseAggr:
        # CuSparse convs are already pure SpMM — no projections to strip.
        return ConvAsAggr(self.create_conv(conv_type, **kwargs))


@BackendRegistry.register_backend("cusparse_precomputed_bwd")
class СuSparsePrecomputeBWDBackend(BaseBackend):
    """Backend that instantiates cusparse-based convolutions. Only matmul-based convolutions are supported."""

    def create_conv(
        self,
        conv_type: str,
        cu_sparse_algorithm_id: int = -1,
        block_dim: int = 256,
        **kwargs: Any,
    ) -> BaseConvolution:
        """Factory for cusparse convolution layers.

        Args:
            conv_type (str): 'sum', 'mean', 'random_walk', 'gcn'
            cu_sparse_algorithm_id (int): algorithm for CuSparse to use: -1 (default), 0, 1, 2, 3.
            **kwargs (Any): ignored.

        Returns:
            BaseConvolution: An instance of the requested CuSparse conv.
        """

        conv_type = conv_type.lower()

        if conv_type == "sum_aggr":
            return _СuSparseMatMulPrecomputedBWDConv(
                norm_type="none", cu_sparse_algorithm_id=cu_sparse_algorithm_id, block_dim=block_dim
            )
        if conv_type == "mean_aggr":
            return _СuSparseMatMulPrecomputedBWDConv(
                norm_type="right", cu_sparse_algorithm_id=cu_sparse_algorithm_id, block_dim=block_dim
            )
        if conv_type == "random_walk":
            return _СuSparseMatMulPrecomputedBWDConv(
                norm_type="left", cu_sparse_algorithm_id=cu_sparse_algorithm_id, block_dim=block_dim
            )
        if conv_type == "gcn":
            return _СuSparseMatMulPrecomputedBWDConv(
                norm_type="both", cu_sparse_algorithm_id=cu_sparse_algorithm_id, block_dim=block_dim
            )
        raise KeyError(f"Unsupported conv_type for CuSparse precomputed_bwd backend: {conv_type}")

    def create_aggr(self, conv_type: str, **kwargs: Any) -> BaseAggr:
        return ConvAsAggr(self.create_conv(conv_type, **kwargs))
