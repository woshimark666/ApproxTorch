"""Internal unfold wrapper avoiding a contiguous STE gradient copy."""

import torch
import torch.nn.functional as F


class _UnfoldBFloat16(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, kernel_size, dilation, padding, stride):
        ctx.geometry = (input.shape[-2:], kernel_size, dilation, padding, stride)
        return F.unfold(
            input, kernel_size, dilation=dilation, padding=padding, stride=stride,
        )

    @staticmethod
    def backward(ctx, grad_columns):
        output_size, kernel_size, dilation, padding, stride = ctx.geometry
        if torch.is_grad_enabled() or grad_columns.is_contiguous():
            # Keep PyTorch's differentiable fold for higher derivatives. The
            # native contiguous path is also faster for a single batch.
            grad_input = F.fold(
                grad_columns, output_size, kernel_size,
                dilation=dilation, padding=padding, stride=stride,
            )
        else:
            # The multi-batch STE returns [N,K,L] backed by [N,L,K]. Read it
            # directly and fold all batches in one launch, retaining PyTorch's
            # per-pixel FP32 accumulation order and final BF16 rounding.
            grad_input = torch.ops.approxtorch._col2im_bf16.default(
                grad_columns, output_size, kernel_size, dilation, padding, stride,
            )
        return grad_input, None, None, None, None


def unfold_bf16(input, kernel_size, *, dilation, padding, stride):
    return _UnfoldBFloat16.apply(input, kernel_size, dilation, padding, stride)
