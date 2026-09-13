"""Autograd wrappers and fixed FP32 gradient LUTs for BF16 multiplication."""

from __future__ import annotations

import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from approxtorch.backend import ops


__all__ = [
    "approx_mul_bf16_custom",
    "gemm_bf16_custom",
    "bgemm_bf16_custom",
]


def _validate_grad_luts(
    grad_x_lut: torch.Tensor | None,
    grad_w_lut: torch.Tensor | None,
    *,
    require_cuda: bool = True,
    device: torch.device | None = None,
) -> bool:
    if grad_x_lut is None and grad_w_lut is None:
        return False
    if grad_x_lut is None or grad_w_lut is None:
        raise ValueError("grad_x_lut and grad_w_lut must be supplied together")
    for name, lut in (("grad_x_lut", grad_x_lut), ("grad_w_lut", grad_w_lut)):
        if not isinstance(lut, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if lut.dtype != torch.float32:
            raise TypeError(f"{name} must have dtype torch.float32, got {lut.dtype}")
        if tuple(lut.shape) != (16384,):
            raise ValueError(f"{name} must have shape (16384,), got {tuple(lut.shape)}")
        if not lut.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
        if lut.requires_grad:
            raise ValueError(f"{name} must be fixed (requires_grad=False)")
        if require_cuda and not lut.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor")
        if device is not None and lut.device != device:
            raise ValueError(f"{name} must be on {device}, got {lut.device}")
    return True


class _FixedBFloat16GradientLUTs:
    """Keep module gradient buffers in FP32, including after Module.bfloat16()."""

    def _register_grad_luts(self, grad_x_lut, grad_w_lut, *, device):
        _validate_grad_luts(grad_x_lut, grad_w_lut, require_cuda=False)
        self.register_buffer(
            "grad_x_lut", None if grad_x_lut is None else grad_x_lut.to(device=device)
        )
        self.register_buffer(
            "grad_w_lut", None if grad_w_lut is None else grad_w_lut.to(device=device)
        )

    def _apply(self, fn, recurse=True):
        original_luts = {
            name: self._buffers.get(name) for name in ("grad_x_lut", "grad_w_lut")
        }
        result = super()._apply(fn, recurse=recurse)
        for name, original in original_luts.items():
            converted = self._buffers.get(name)
            if original is not None and converted.dtype != torch.float32:
                # Restore from the FP32 source, never from an already rounded LUT.
                self._buffers[name] = original.to(device=converted.device)
        return result


def _save_custom_inputs(ctx, x, w, grad_x_lut, grad_w_lut):
    if not _validate_grad_luts(grad_x_lut, grad_w_lut, device=x.device):
        raise ValueError("custom backward requires grad_x_lut and grad_w_lut")
    # Both raw fractions participate in the index, even for a single gradient.
    ctx.save_for_backward(x, w, grad_x_lut, grad_w_lut)


def _custom_backward(ctx, grad_output, backward_op):
    x, w, grad_x_lut, grad_w_lut = ctx.saved_tensors
    need_x, need_w = ctx.needs_input_grad[:2]
    return backward_op(
        x, w, grad_output, grad_x_lut, grad_w_lut, need_x, need_w
    )


class _ApproxMulBFloat16Custom(Function):
    @staticmethod
    def forward(ctx, x, w, lut, grad_x_lut, grad_w_lut):
        _save_custom_inputs(ctx, x, w, grad_x_lut, grad_w_lut)
        return ops.approx_mul_bf16(x, w, lut)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        grad_x, grad_w = _custom_backward(ctx, grad_output, ops.approx_mul_bf16_backward)
        return grad_x, grad_w, None, None, None


class _GemmBFloat16Custom(Function):
    @staticmethod
    def forward(ctx, x, w, lut, grad_x_lut, grad_w_lut, optimized):
        _save_custom_inputs(ctx, x, w, grad_x_lut, grad_w_lut)
        forward_op = ops.gemm_bf16 if optimized else ops.gemm_bf16_naive
        return forward_op(x, w, lut)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        grad_x, grad_w = _custom_backward(ctx, grad_output, ops.gemm_bf16_backward)
        return grad_x, grad_w, None, None, None, None


class _BGemmBFloat16Custom(Function):
    @staticmethod
    def forward(ctx, x, w, lut, grad_x_lut, grad_w_lut, optimized):
        _save_custom_inputs(ctx, x, w, grad_x_lut, grad_w_lut)
        forward_op = ops.bgemm_bf16 if optimized else ops.bgemm_bf16_naive
        return forward_op(x, w, lut)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        grad_x, grad_w = _custom_backward(ctx, grad_output, ops.bgemm_bf16_backward)
        return grad_x, grad_w, None, None, None, None


def approx_mul_bf16_custom(x, w, lut, grad_x_lut, grad_w_lut):
    """Multiply equal-shaped CUDA BF16 tensors with a LUT-defined backward.

    The forward uint32 LUT keeps its existing ``[128, 128]`` layout. Each
    gradient LUT is a fixed, contiguous CUDA FP32 tensor of shape ``[16384]``,
    indexed by ``(fraction_x << 7) | fraction_w``. Higher derivatives are
    unsupported. No LUT is copied or generated during forward or backward.
    """
    return _ApproxMulBFloat16Custom.apply(x, w, lut, grad_x_lut, grad_w_lut)


def gemm_bf16_custom(x, w, lut, grad_x_lut, grad_w_lut, optimized=True):
    """Compute BF16 ``[M,K] @ [K,N]`` with fixed FP32 ``[16384]`` gradient LUTs."""
    if not isinstance(optimized, bool):
        raise TypeError("optimized must be a bool")
    return _GemmBFloat16Custom.apply(x, w, lut, grad_x_lut, grad_w_lut, optimized)


def bgemm_bf16_custom(x, w, lut, grad_x_lut, grad_w_lut, optimized=True):
    """Compute BF16 BGEMM ``[N,K,L], [O,K] -> [N,O,L]`` with custom gradients.

    Gradient LUTs are fixed CUDA FP32 ``[16384]`` buffers using the same
    ``(fraction_x << 7) | fraction_w`` index as elementwise multiplication.
    Weight gradients reduce across the batch and spatial dimensions.
    """
    if not isinstance(optimized, bool):
        raise TypeError("optimized must be a bool")
    return _BGemmBFloat16Custom.apply(x, w, lut, grad_x_lut, grad_w_lut, optimized)
