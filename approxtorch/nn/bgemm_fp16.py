"""Autograd wrapper for LUT-based FP16 BGEMM."""

import torch
from torch.autograd import Function

import approxtorch as at


__all__ = ["bgemm_fp16_ste"]


class _bgemm_fp16_base(Function):

    @staticmethod
    def forward(ctx, x, w, lut, optimized=True):
        ctx.save_for_backward(x, w)
        if optimized:
            return at.backend.ops.bgemm_fp16(x, w, lut)
        return at.backend.ops.bgemm_fp16_naive(x, w, lut)


class _bgemm_fp16_ste(_bgemm_fp16_base):
    """Use the exact-product BGEMM derivative as a straight-through estimator."""

    @staticmethod
    def backward(ctx, grad_output):
        x, w = ctx.saved_tensors
        grad_x = torch.einsum("nol,ok->nkl", grad_output, w)
        grad_w = torch.einsum("nol,nkl->ok", grad_output, x)
        return grad_x, grad_w, None, None


def bgemm_fp16_ste(x, w, lut, optimized=True):
    """Run approximate FP16 BGEMM with an STE backward pass.

    Args:
        x: Contiguous CUDA FP16 tensor with shape ``[N, K, L]``.
        w: Contiguous CUDA FP16 tensor with shape ``[O, K]``.
        lut: Contiguous CUDA uint32 LUT with shape ``[1024, 1024]``.
        optimized: Select the optimized kernel when true.
    """
    return _bgemm_fp16_ste.apply(x, w, lut, optimized)
