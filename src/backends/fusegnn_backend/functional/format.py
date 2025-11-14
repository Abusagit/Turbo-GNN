import torch

from src.backends.fusegnn_backend.utils import fgnn_format


class Coo2Csr(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tar_index, num_node):
        tar_ptr = fgnn_format.coo2csr(tar_index, num_node)
        return tar_ptr

    @staticmethod
    def backward(ctx, *grad_output):
        return None, None


def coo2csr(src_index, tar_index, num_node, edge_weight=None, sorted=False):
    if not sorted:
        tar_index, indices = torch.sort(tar_index, dim=0)
        src_index = torch.gather(src_index, 0, indices)
        if edge_weight is not None:
            edge_weight = torch.gather(edge_weight.squeeze(), 0, indices)
    tar_ptr = Coo2Csr.apply(tar_index, num_node)
    return src_index, tar_index, tar_ptr, edge_weight


class Csr2Csc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inPtr, inInd, inVal, num_row):
        outPtr, outInd, outVal = fgnn_format.csr2csc(inPtr, inInd, inVal, num_row)
        return outPtr, outInd, outVal

    @staticmethod
    def backward(ctx, *grad_output):
        return None, None, None


csr2csc = Csr2Csc.apply
