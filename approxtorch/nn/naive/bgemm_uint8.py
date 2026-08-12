"""Naive autograd references for raw uint8 LUT BGEMM.

The public API mirrors :mod:`approxtorch.nn.bgemm_uint8`, but forward, LRE
backward and custom-gradient backward dispatch to the deliberately simple
CUDA kernels in ``backend/csrc/naive_cuda``.  Zero-point correction remains
the caller's responsibility, exactly as in the optimized module.
"""

import torch
from torch.autograd import Function

import approxtorch as at


__all__ = [
    "bgemm_uint8",
    "bgemm_uint8_ste",
    "bgemm_uint8_lre",
    "bgemm_uint8_custom",
]


def _quantize_uint8(x: torch.Tensor) -> torch.Tensor:
    """Reproduce the optimized prepass's round-to-even and uint8 clamp."""
    if x.dtype == torch.uint8:
        return x
    return torch.clamp(torch.round(x), 0, 255).to(torch.uint8)


def _bgemm_uint8_forward(
        x: torch.Tensor, w: torch.Tensor, lut: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run naive uint8 LUT BGEMM and return output plus saved index images."""
    xq = _quantize_uint8(x)
    wq = _quantize_uint8(w)
    # The raw naive CUDA op accumulates an integer multiplier LUT into int32.
    # Cast its result back to the optimized nn API's float32 output dtype so
    # a custom autograd Function can propagate gradients through it.
    y = at.backend.ops.bgemm_uint8_naive(
        xq, wq, lut.to(dtype=torch.int32)).to(dtype=torch.float32)
    return y, xq, wq


class _bgemm_uint8_base(Function):

    @staticmethod
    def forward(ctx, x, w, lut):
        y, _, _ = _bgemm_uint8_forward(x, w, lut)
        ctx.save_for_backward(x, w)
        return y


class _bgemm_uint8_ste(_bgemm_uint8_base):

    @staticmethod
    def backward(ctx, grad_output):
        x, w = ctx.saved_tensors
        grad_x = torch.einsum("nol,ok->nkl", grad_output, w)
        grad_w = torch.einsum("nol,nkl->ok", grad_output, x)
        return grad_x, grad_w, None


def bgemm_uint8_ste(x, w, lut):
    """Naive raw uint8 LUT BGEMM with a straight-through gradient."""
    return _bgemm_uint8_ste.apply(x, w, lut)


class _bgemm_uint8_lre(Function):

    @staticmethod
    def forward(ctx, x, w, lut, dx, dw):
        y, xq, wq = _bgemm_uint8_forward(x, w, lut)
        ctx.save_for_backward(xq, wq)
        ctx.dx = dx
        ctx.dw = dw
        return y

    @staticmethod
    def backward(ctx, grad_output):
        xq, wq = ctx.saved_tensors
        grad_x, grad_w = at.backend.ops.bgemm_lre_backward_uint8_naive(
            grad_output, xq, wq, ctx.dx, ctx.dw)
        return grad_x, grad_w, None, None, None


def bgemm_uint8_lre(x, w, lut, dx, dw):
    """Naive raw uint8 LUT BGEMM with one-dimensional LRE gradient LUTs."""
    return _bgemm_uint8_lre.apply(x, w, lut, dx, dw)


class _bgemm_uint8_custom(Function):

    @staticmethod
    def forward(ctx, x, w, lut, dx_lut, dw_lut):
        y, xq, wq = _bgemm_uint8_forward(x, w, lut)
        ctx.save_for_backward(xq, wq, dx_lut, dw_lut)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        xq, wq, dx_lut, dw_lut = ctx.saved_tensors
        grad_x, grad_w = at.backend.ops.bgemm_custom_grad_uint8_naive(
            xq, wq, grad_output, dx_lut, dw_lut)
        return grad_x, grad_w, None, None, None


def bgemm_uint8_custom(x, w, lut, dx_lut, dw_lut):
    """Naive raw uint8 LUT BGEMM with pair-wise custom gradient LUTs."""
    return _bgemm_uint8_custom.apply(x, w, lut, dx_lut, dw_lut)


class _bgemm_uint8(_bgemm_uint8_base):

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError(
            "bgemm_uint8 is forward-only; use bgemm_uint8_ste, "
            "bgemm_uint8_lre or bgemm_uint8_custom for training")


def bgemm_uint8(x, w, lut):
    """Naive forward-only raw uint8 LUT BGEMM."""
    return _bgemm_uint8.apply(x, w, lut)
