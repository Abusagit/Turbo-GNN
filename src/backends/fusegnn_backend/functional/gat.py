import torch
import torch.nn.functional as F
import torch_scatter

from src.backends.fusegnn_backend.utils import fgnn_gat


class GATGAREdgeWeight(torch.autograd.Function):
    @staticmethod
    def forward(ctx, e_pre, src_ptr, tar_index, src_index, negative_slope):
        alpha, alpha_self, mask_lrelu, mask_lrelu_self, e_sum, e, e_self = fgnn_gat.gat_gar_edge_weight(
            e_pre, src_ptr, tar_index, negative_slope
        )
        ctx.save_for_backward(src_index, tar_index, mask_lrelu, mask_lrelu_self, e, e_self, e_sum, alpha_self, alpha)
        return alpha_self, alpha

    @staticmethod
    def backward(ctx, grad_alpha_self, grad_alpha):
        src_index, tar_index, mask_lrelu, mask_lrelu_self, e, e_self, e_sum, alpha_self, alpha = ctx.saved_tensors

        grad_e_sum, grad_e_pre = fgnn_gat.gat_gar_edge_weight_b(
            grad_alpha_self,
            grad_alpha,
            src_index,
            tar_index,
            mask_lrelu,
            mask_lrelu_self,
            e,
            e_self,
            e_sum,
            alpha_self,
            alpha,
        )
        return grad_e_pre, None, None, None, None, None


gat_gar_edge_weight = GATGAREdgeWeight.apply


class GATGASEdgeWeight(torch.autograd.Function):
    @staticmethod
    def forward(ctx, e_pre, src_index, tar_index, negative_slope):
        alpha, alpha_self, mask_lrelu, mask_lrelu_self, e_sum, e, e_self = fgnn_gat.gat_gas_edge_weight(
            e_pre, src_index, tar_index, negative_slope
        )
        ctx.save_for_backward(src_index, tar_index, mask_lrelu, mask_lrelu_self, e, e_self, e_sum, alpha_self, alpha)
        return alpha_self, alpha

    @staticmethod
    def backward(ctx, grad_alpha_self, grad_alpha):
        src_index, tar_index, mask_lrelu, mask_lrelu_self, e, e_self, e_sum, alpha_self, alpha = ctx.saved_tensors

        grad_e_sum, grad_e_pre = fgnn_gat.gat_gar_edge_weight_b(
            grad_alpha_self,
            grad_alpha,
            src_index,
            tar_index,
            mask_lrelu,
            mask_lrelu_self,
            e,
            e_self,
            e_sum,
            alpha_self,
            alpha,
        )
        return grad_e_pre, None, None, None, None


gat_gas_edge_weight = GATGASEdgeWeight.apply


class refGATEdgeWeight(torch.autograd.Function):
    @staticmethod
    def forward(ctx, e_pre, src_ptr, tar_index, src_index, negative_slope):
        e_pre_src, e_pre_tar = torch.chunk(e_pre, chunks=2, dim=1)
        e_pre_src_expand = torch.index_select(e_pre_src, dim=0, index=src_index)
        e_pre_tar_expand = torch.index_select(e_pre_tar, dim=0, index=tar_index)
        e = e_pre_src_expand + e_pre_tar_expand
        e_self = e_pre_src + e_pre_tar
        mask_lrelu = F.leaky_relu(e, negative_slope=negative_slope) / e
        mask_lrelu_self = F.leaky_relu(e_self, negative_slope=negative_slope) / e_self
        e *= mask_lrelu
        e_self *= mask_lrelu_self
        e = torch.exp(e)
        e_self = torch.exp(e_self)
        e_sum = torch_scatter.scatter_add(src=e, index=tar_index, dim=0, dim_size=e_self.size(0))
        e_sum += e_self
        alpha_self = e_self / e_sum
        e_sum_ext = torch.index_select(e_sum, dim=0, index=tar_index)
        alpha = e / e_sum_ext

        ctx.save_for_backward(src_index, tar_index, mask_lrelu, mask_lrelu_self, e, e_self, e_sum, alpha_self, alpha)
        return alpha_self, alpha

    @staticmethod
    def backward(ctx, grad_alpha_self, grad_alpha):
        src_index, tar_index, mask_lrelu, mask_lrelu_self, e, e_self, e_sum, alpha_self, alpha = ctx.saved_tensors

        src_index = src_index.to(torch.int64)
        tar_index = tar_index.to(torch.int64)

        e_sum_ext = torch.index_select(e_sum, dim=0, index=tar_index)
        grad_e_sum = -grad_alpha * alpha / e_sum_ext
        grad_e = grad_alpha / e_sum_ext

        # e_sum_ext = torch.index_select(e_sum, dim=0, index=tar_index)
        grad_e_sum = torch_scatter.scatter_add(src=grad_e_sum, index=tar_index, dim=0, dim_size=alpha_self.size(0))

        grad_e_self = grad_alpha_self / e_sum
        grad_e_sum -= grad_alpha_self * alpha_self / e_sum

        grad_e_self += grad_e_sum
        grad_e += torch.index_select(grad_e_sum, dim=0, index=tar_index)
        grad_e_self *= e_self
        grad_e *= e

        grad_e_self *= mask_lrelu_self
        grad_e *= mask_lrelu

        grad_e_pre_src = grad_e_self + torch_scatter.scatter_add(
            src=grad_e, index=src_index, dim=0, dim_size=grad_e_self.size(0)
        )
        grad_e_pre_tar = grad_e_self + torch_scatter.scatter_add(
            src=grad_e, index=tar_index, dim=0, dim_size=grad_e_self.size(0)
        )

        grad_e_pre = torch.cat((grad_e_pre_src, grad_e_pre_tar), dim=1)
        return grad_e_pre, None, None, None, None, None
