"""LUT-based approximate FP16 Conv2d."""

from __future__ import annotations

import torch

from . import bgemm_fp16
from ._conv2d_float import (
    ApproxConv2dFloat16,
    conv2d_approx_float,
)


__all__ = ["Conv2d_float16", "Conv2d_fp16", "conv2d_float16", "conv2d_fp16"]


def conv2d_float16(
    input: torch.Tensor,
    weight: torch.Tensor,
    lut: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
    groups: int = 1,
    optimized: bool = True,
) -> torch.Tensor:
    """Apply approximate FP16 Conv2d using unfold and LUT BGEMM."""
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
        kind="fp16",
        bgemm=bgemm_fp16.bgemm_fp16_ste,
    )


class Conv2d_float16(ApproxConv2dFloat16):
    """Approximate FP16 convolution with an exact-convolution STE backward."""

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
    ) -> None:
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
            kind="fp16",
            operation=conv2d_float16,
        )


# Short spellings match the corresponding bgemm_fp16 operator name.
Conv2d_fp16 = Conv2d_float16
conv2d_fp16 = conv2d_float16
