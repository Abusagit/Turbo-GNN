import torch

from src.backends.fusegnn_backend.utils import fgnn_gcn

# fused get edge weight function for GAR model


class GCNGAREdgeWeight(torch.autograd.Function):
    @staticmethod
    def forward(ctx, src_index, tar_ptr, tar_index, num_nodes, edge_weight, flow):
        return fgnn_gcn.gcn_gar_edge_weight(src_index, tar_ptr, tar_index, num_nodes, edge_weight)

    @staticmethod
    def backward(ctx, *grad_output):
        return None, None, None, None, None, None


def gcn_gar_edge_weight(src_index, tar_ptr, tar_index, num_nodes, edge_weight, flow):
    edge_weight = GCNGAREdgeWeight.apply(src_index, tar_ptr, tar_index, num_nodes, edge_weight, flow)
    return edge_weight


# fused get edge weight function for GAS model


class GCNGASEdgeWeight(torch.autograd.Function):
    @staticmethod
    def forward(ctx, src_index, tar_index, num_nodes, edge_weight, flow):
        return fgnn_gcn.gcn_gas_edge_weight(src_index, tar_index, num_nodes, edge_weight)

    @staticmethod
    def backward(ctx, *grad_output):
        return None, None, None, None, None, None


def gcn_gas_edge_weight(src_index, tar_index, num_nodes, edge_weight, flow):
    edge_weight = GCNGASEdgeWeight.apply(src_index, tar_index, num_nodes, edge_weight, flow)
    return edge_weight
