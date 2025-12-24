import math
from typing import Any

try:
    import TCGNN
except ImportError:
    print("TCGNN is not found!")
    TCGNN = None

import torch

from src.backends.base import BaseBackend, BaseConvolution
from src.backends.registry import BackendRegistry


class TCGNNFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, weights, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow):
        ctx.save_for_backward(X, weights, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow)

        # X_prime = torch.mm(X, weights)
        X_prime = TCGNN.forward(X, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow)[0]

        return X_prime

    @staticmethod
    def backward(ctx, d_output):
        X, weights, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow = ctx.saved_tensors

        # SPMM backward propaAGNNion.

        d_input = TCGNN.forward(d_output, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow)[0]

        # TODO fix for directed graphs!!!!!!!

        # # GEMM backward propaAGNNion.
        # d_input = torch.mm(d_input_prime, weights.transpose(0, 1))
        # d_weights = torch.mm(X.transpose(0, 1), d_input_prime)
        return d_input, None, None, None, None, None, None, None


class TCGNNFunction_GIN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, weights, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow):
        # SpMM: Neighbor AggreAGNNion.
        X_prime = TCGNN.forward(X, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow)[0]

        ctx.save_for_backward(X_prime, weights, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow)

        # GEMM node update
        X_prime = torch.mm(X_prime, weights)

        return X_prime

    @staticmethod
    def backward(ctx, d_output):
        X_prime, weights, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow = ctx.saved_tensors

        # GEMM backward propaAGNNion.
        # d_X_prime = torch.mm(d_output, weights.transpose(0, 1))
        # d_weights = torch.mm(X_prime.transpose(0, 1), d_output)

        # SPMM backward propaAGNNion.
        d_input = TCGNN.forward(
            d_output.transpose(-1, -2), row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow
        )[0]

        return d_input, None, None, None, None, None, None, None


class TCGNNFunction_AGNN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, weights, attention_w, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow):
        # SpMM: Neighbor AggreAGNNion.
        X_prime = TCGNN.forward(X, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow)[0]

        ctx.save_for_backward(
            X_prime, weights, attention_w, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow
        )

        return X_prime

    @staticmethod
    def backward(ctx, d_output):
        X_prime, weights, attention_w, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow = (
            ctx.saved_tensors
        )

        # GEMM backward propaAGNNion.
        d_X_prime = torch.mm(d_output, weights.transpose(0, 1))
        d_weights = torch.mm(X_prime.transpose(0, 1), d_output)
        d_attention_w = torch.mm(X_prime.transpose(0, 1), d_output)

        # SPMM backward propaAGNNion.
        d_input = TCGNN.forward(d_X_prime, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow)[0]

        return d_input, d_weights, d_attention_w, None, None, None, None, None, None


class _GINConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(_GINConv, self).__init__()
        self.weights = torch.nn.Parameter(torch.randn(input_dim, output_dim))

    def forward(self, X, graph: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]):
        """
        @param:
        X:  the input tensor of the graph node embedding, shape: [n_nodes, n_dim].
        A:  the CSR node pointer of the graph, shape: [node, 1].
        edges: the CSR edge list of the graph, shape: [edge, 1].
        partitioin: for the graph with the part-based optimziation.
        """
        row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow = graph

        return TCGNNFunction_GIN.apply(
            X, self.weights, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow
        )


class _GCNConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(_GCNConv, self).__init__()

        self.weights = torch.nn.Parameter(torch.randn(input_dim, output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weights.size(1))
        self.weights.data.uniform_(-stdv, stdv)

    def forward(self, X, graph: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]):
        """
        @param:
        X:  the input tensor of the graph node embedding, shape: [n_nodes, n_dim].
        A:  the CSR node pointer of the graph, shape: [node, 1].
        edges: the CSR edge list of the graph, shape: [edge, 1].
        partitioin: for the graph with the part-based optimziation.
        """
        row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow = graph
        return TCGNNFunction.apply(X, self.weights, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow)


class _AGNNConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim, n_heads=4):
        super(_AGNNConv, self).__init__()
        self.weights = torch.nn.Parameter(torch.randn(input_dim, output_dim))
        self.attention_w = torch.nn.Parameter(torch.randn(1, n_heads))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weights.size(1))
        self.weights.data.uniform_(-stdv, stdv)

    def forward(self, X, graph: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]):
        """
        @param:
        X:  the input tensor of the graph node embedding, shape: [n_nodes, n_dim].
        A:  the CSR node pointer of the graph, shape: [node, 1].
        edges: the CSR edge list of the graph, shape: [edge, 1].
        partitioin: for the graph with the part-based optimziation.
        """
        row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow = graph
        return TCGNNFunction_AGNN.apply(
            X, self.weights, self.attention_w, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow
        )


@BackendRegistry.register_backend("tcgnn")
class TcgnnBackend(BaseBackend):
    """
    Backend that instantiates TCGNN-based convolutions.
    """

    def create_conv(
        self,
        conv_type: str,
        **kwargs: Any,
    ):
        """Factory for TCGNN convolution layers.

        Args:
            conv_type (str): 'gcn' | 'gin'.
            in_channels (int): Input feature size.
            out_channels (int): Output feature size.
            **kwargs (Any): Extra arguments passed to the underlying TCGNN layer.
        """

        feature_dim = kwargs.pop("feature_dim")
        match conv_type:
            case "gcn":
                return _GCNConv(feature_dim, feature_dim)
            case "gin":
                return _GINConv(feature_dim, feature_dim)
            case "agnn":
                return _AGNNConv(feature_dim, feature_dim)
            case _:
                raise ValueError(f"Unsupported convolution type: {conv_type}")
