"""LUT-based approximate BF16 Conv2d."""

from __future__ import annotations

from functools import partial

import torch

from . import bgemm_bf16
from ._bf16_custom_grad import _FixedBFloat16GradientLUTs, _validate_grad_luts
from ._conv2d_float import (
    ApproxConv2dBFloat16,
    conv2d_approx_float,
)


__all__ = ["Conv2d_bfloat16", "Conv2d_bf16", "conv2d_bfloat16", "conv2d_bf16"]


def _bgemm_with_custom_grad(x, w, lut, optimized, *, grad_x_lut, grad_w_lut):
    return bgemm_bf16.bgemm_bf16_custom(
        x, w, lut, grad_x_lut, grad_w_lut, optimized
    )


def conv2d_bfloat16(
    input: torch.Tensor,
    weight: torch.Tensor,
    lut: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
    groups: int = 1,
    optimized: bool = True,
    *,
    grad_x_lut: torch.Tensor | None = None,
    grad_w_lut: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply approximate BF16 Conv2d using LUT BGEMM.

    Supply both fixed CUDA FP32 ``[16384]`` gradient LUTs to use custom
    gradients. Without them the existing exact-convolution STE is used.
    """
    custom_gradient = _validate_grad_luts(
        grad_x_lut,
        grad_w_lut,
        device=input.device if isinstance(input, torch.Tensor) else None,
    )
    bgemm = (
        partial(
            _bgemm_with_custom_grad,
            grad_x_lut=grad_x_lut,
            grad_w_lut=grad_w_lut,
        )
        if custom_gradient
        else bgemm_bf16.bgemm_bf16_ste
    )
    return conv2d_approx_float(
        input,
        weight,
        lut,
        bias,
        stride,
        padding,
        dilation,
        groups,
        optimized,
        kind="bf16",
        bgemm=bgemm,
    )


class Conv2d_bfloat16(_FixedBFloat16GradientLUTs, ApproxConv2dBFloat16):
    """Approximate BF16 convolution with optional fixed FP32 gradient LUTs."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        lut: torch.Tensor,
        bias: torch.Tensor | bool | None = True,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        padding_mode: str = "zeros",
        optimized: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        *,
        grad_x_lut: torch.Tensor | None = None,
        grad_w_lut: torch.Tensor | None = None,
    ) -> None:
        _validate_grad_luts(grad_x_lut, grad_w_lut, require_cuda=False)
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            lut,
            bias,
            stride,
            padding,
            dilation,
            groups,
            padding_mode,
            optimized,
            device,
            dtype,
            kind="bf16",
            operation=conv2d_bfloat16,
        )
        self._register_grad_luts(grad_x_lut, grad_w_lut, device=self.lut.device)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self._operation(
            input,
            self.weight,
            self.lut,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.optimized,
            grad_x_lut=self.grad_x_lut,
            grad_w_lut=self.grad_w_lut,
        )


# Short spellings match the corresponding bgemm_bf16 operator name.
Conv2d_bf16 = Conv2d_bfloat16
conv2d_bf16 = conv2d_bfloat16
