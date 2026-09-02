"""Autograd wrapper for LUT-based FP16 BGEMM."""

import torch
from torch.autograd import Function

import approxtorch as at


__all__ = ["bgemm_fp16_ste"]


class _bgemm_fp16_base(Function):

    @staticmethod
    def forward(ctx, x, w, lut):
        ctx.save_for_backward(x, w)
        return at.backend.ops.bgemm_fp16(x, w, lut)


class _bgemm_fp16_ste(_bgemm_fp16_base):
    """Use the exact-product BGEMM derivative as a straight-through estimator."""

    @staticmethod
    def backward(ctx, grad_output):
        x, w = ctx.saved_tensors
        grad_x = torch.einsum("nol,ok->nkl", grad_output, w)
        grad_w = torch.einsum("nol,nkl->ok", grad_output, x)
        return grad_x, grad_w, None


def bgemm_fp16_ste(x, w, lut):
    """Run approximate FP16 BGEMM with an STE backward pass.

    Args:
        x: Contiguous CUDA FP16 tensor with shape ``[N, K, L]``.
        w: Contiguous CUDA FP16 tensor with shape ``[O, K]``.
        lut: Contiguous CUDA uint16 LUT with shape ``[1024, 1024]``.
    """
    return _bgemm_fp16_ste.apply(x, w, lut)
