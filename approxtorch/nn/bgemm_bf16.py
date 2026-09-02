"""Autograd wrapper for LUT-based BF16 BGEMM."""

import torch
from torch.autograd import Function

import approxtorch as at


__all__ = ["bgemm_bf16_ste"]


class _bgemm_bf16_base(Function):

    @staticmethod
    def forward(ctx, x, w, lut):
        ctx.save_for_backward(x, w)
        return at.backend.ops.bgemm_bf16(x, w, lut)


class _bgemm_bf16_ste(_bgemm_bf16_base):
    """Use the exact-product BGEMM derivative as a straight-through estimator."""

    @staticmethod
    def backward(ctx, grad_output):
        x, w = ctx.saved_tensors
        grad_x = torch.einsum("nol,ok->nkl", grad_output, w)
        grad_w = torch.einsum("nol,nkl->ok", grad_output, x)
        return grad_x, grad_w, None


def bgemm_bf16_ste(x, w, lut):
    """Run approximate BF16 BGEMM with an STE backward pass.

    Args:
        x: Contiguous CUDA BF16 tensor with shape ``[N, K, L]``.
        w: Contiguous CUDA BF16 tensor with shape ``[O, K]``.
        lut: Contiguous CUDA uint16 LUT with shape ``[128, 128]``.
    """
    return _bgemm_bf16_ste.apply(x, w, lut)
