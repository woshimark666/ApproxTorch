"""Naive autograd references for int8 LUT BGEMM.

These functions mirror the GEMM-level APIs in :mod:`approxtorch.nn.bgemm_int8`
while dispatching to the deliberately simple CUDA kernels in
``backend/csrc/naive_cuda``.  Inputs ``x`` and ``w`` contain quantized values
in float32 storage; the helpers below reproduce the optimized forward's
round-to-nearest and int8 saturation before calling the typed naive kernels.
"""

import torch
from torch.autograd import Function

import approxtorch as at


__all__ = ["bgemm_int8_ste", "bgemm_int8_lre", "bgemm_int8_custom"]


def _quantize_int8(x: torch.Tensor) -> torch.Tensor:
    """Convert fake-quantized values to the int8 image used by naive CUDA."""
    if x.dtype == torch.int8:
        return x
    return torch.clamp(torch.round(x), -128, 127).to(torch.int8)


def _bgemm_int8_forward(
        x: torch.Tensor, w: torch.Tensor, lut: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run naive LUT BGEMM and return its output plus quantized operands."""
    xq = _quantize_int8(x)
    wq = _quantize_int8(w)
    # The raw naive kernel deliberately uses integer storage for both its LUT
    # and accumulator.  The nn-facing API, like the optimized implementation,
    # exposes a float32 output so that autograd can propagate through it.
    y = at.backend.ops.bgemm_int8_naive(
        xq, wq, lut.to(dtype=torch.int32)).to(dtype=torch.float32)
    return y, xq, wq


class _bgemm_int8_ste(Function):

    @staticmethod
    def forward(ctx, x, w, lut):
        y, _, _ = _bgemm_int8_forward(x, w, lut)
        # Match approxtorch.nn.bgemm_int8 exactly: STE treats the supplied
        # fake-quantized float values as operands of an ordinary BGEMM.
        ctx.save_for_backward(x, w)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, w = ctx.saved_tensors
        grad_x = torch.einsum("nol,ok->nkl", grad_output, w)
        grad_w = torch.einsum("nol,nkl->ok", grad_output, x)
        return grad_x, grad_w, None


def bgemm_int8_ste(x, w, lut):
    """Naive int8 LUT BGEMM with a straight-through gradient."""
    return _bgemm_int8_ste.apply(x, w, lut)


class _bgemm_int8_lre(Function):

    @staticmethod
    def forward(ctx, x, w, lut, dx, dw):
        y, xq, wq = _bgemm_int8_forward(x, w, lut)
        ctx.save_for_backward(xq, wq)
        ctx.dx = dx
        ctx.dw = dw
        return y

    @staticmethod
    def backward(ctx, grad_output):
        xq, wq = ctx.saved_tensors
        grad_x, grad_w = at.backend.ops.bgemm_lre_backward_int8_naive(
            grad_output, xq, wq, ctx.dx, ctx.dw)
        return grad_x, grad_w, None, None, None


def bgemm_int8_lre(x, w, lut, dx, dw):
    """Naive int8 LUT BGEMM with one-dimensional LRE gradient LUTs."""
    return _bgemm_int8_lre.apply(x, w, lut, dx, dw)


class _bgemm_int8_custom(Function):

    @staticmethod
    def forward(ctx, x, w, lut, dx_lut, dw_lut):
        y, xq, wq = _bgemm_int8_forward(x, w, lut)
        ctx.save_for_backward(xq, wq, dx_lut, dw_lut)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        xq, wq, dx_lut, dw_lut = ctx.saved_tensors
        grad_x, grad_w = at.backend.ops.bgemm_custom_grad_int8_naive(
            xq, wq, grad_output, dx_lut, dw_lut)
        return grad_x, grad_w, None, None, None


def bgemm_int8_custom(x, w, lut, dx_lut, dw_lut):
    """Naive int8 LUT BGEMM with pair-wise custom gradient LUTs."""
    return _bgemm_int8_custom.apply(x, w, lut, dx_lut, dw_lut)
