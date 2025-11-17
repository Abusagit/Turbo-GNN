import gcnlib_cuda
import torch

# gcnlib_cuda.dropout(cooVals, 0.5, True)


class Dropout(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, rate, training):
        # what feed in is the probability of 0.
        out, mask = gcnlib_cuda.dropout(input_, rate, training, None)
        if not training:
            mask = torch.ones_like(input_)
        if training:
            ctx.save_for_backward(mask)
        return out, mask

    @staticmethod
    def backward(ctx, grad_out, grad_mask):
        mask = ctx.saved_tensors[0]
        return gcnlib_cuda.dropout_bp(grad_out, mask), None, None
