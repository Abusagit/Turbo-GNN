"""Destination-prefix handling for sampled mini-batch blocks.

In a sampled block the destination nodes are the first ``num_dst`` source nodes, so
``x_dst = x_src[:num_dst]`` is a view. Plain slicing is cheap in the forward pass, but
autograd's backward of a slice materialises a zero-filled ``[num_src, F]`` gradient and
copies the prefix into it -- once per slice, per layer. A GATv2 residual block slices the
projected features twice (destination projection, residual skip).

`dst_src_project` replaces both slices *and* both projections with one autograd node whose
backward accumulates every contribution into a single buffer:

    g          = grad_r @ W_r              # [num_src, F], fresh buffer
    g[:n]     += grad_l @ W_l              # addmm_ into the prefix, in place
    g[:n]     += grad_skip                 # residual, in place

Selected with ``ResidualBlock(prefix_grad="inplace")``; ``"autograd"`` keeps plain slicing.
Both compute the same values; the rebuttal measures the difference.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

PREFIX_GRAD_MODES = ("autograd", "inplace")


def num_dst_of(graph: Any) -> int | None:
    """Number of destination rows of a sampled block, or None for a full (square) graph."""
    is_block = getattr(graph, "is_block", None)
    if is_block is not None:  # DGL graph / block
        return graph.num_dst_nodes() if is_block else None
    num_dst, num_src = getattr(graph, "num_dst", None), getattr(graph, "num_src", None)
    if num_dst is not None and num_src is not None and num_dst != num_src:
        return num_dst
    return None


class _DstSrcProject(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, x, w_dst, b_dst, w_src, b_src, num_dst):
        x_dst = x.narrow(0, 0, num_dst)
        left = F.linear(x_dst, w_dst, b_dst)
        right = F.linear(x, w_src, b_src)
        ctx.num_dst = num_dst
        ctx.has_b_dst = b_dst is not None
        ctx.has_b_src = b_src is not None
        ctx.save_for_backward(x, w_dst, w_src)
        # The residual skip is returned as a view of the input; its gradient arrives in the
        # same backward call as grad_left/grad_right, which is what makes the fusion possible.
        return left, right, x_dst

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_left, grad_right, grad_skip):
        x, w_dst, w_src = ctx.saved_tensors
        n = ctx.num_dst
        x_dst = x.narrow(0, 0, n)
        grad_x = grad_w_dst = grad_b_dst = grad_w_src = grad_b_src = None

        if ctx.needs_input_grad[0]:
            if grad_right is not None:
                grad_x = torch.mm(grad_right, w_src.to(grad_right.dtype))
            else:
                grad_x = torch.zeros_like(x)
            head = grad_x.narrow(0, 0, n)
            if grad_left is not None:
                head.addmm_(grad_left, w_dst.to(grad_left.dtype))
            if grad_skip is not None:
                head.add_(grad_skip)

        if grad_left is not None:
            if ctx.needs_input_grad[1]:
                grad_w_dst = torch.mm(grad_left.t(), x_dst.to(grad_left.dtype))
            if ctx.has_b_dst and ctx.needs_input_grad[2]:
                grad_b_dst = grad_left.sum(0)
        if grad_right is not None:
            if ctx.needs_input_grad[3]:
                grad_w_src = torch.mm(grad_right.t(), x.to(grad_right.dtype))
            if ctx.has_b_src and ctx.needs_input_grad[4]:
                grad_b_src = grad_right.sum(0)

        return grad_x, grad_w_dst, grad_b_dst, grad_w_src, grad_b_src, None


def dst_src_project(
    x: torch.Tensor, fc_dst: nn.Linear, fc_src: nn.Linear, num_dst: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(fc_dst(x[:num_dst]), fc_src(x), x[:num_dst])`` with a fused backward."""
    return _DstSrcProject.apply(x, fc_dst.weight, fc_dst.bias, fc_src.weight, fc_src.bias, num_dst)
