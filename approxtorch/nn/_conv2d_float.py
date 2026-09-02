"""Shared implementation for LUT-based FP16 and BF16 convolutions."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}
_LUT_SIDES = {"fp16": 1024, "bf16": 128}


def _pair_parameter(
    value: int | tuple[int, int],
    name: str,
    *,
    allow_zero: bool,
) -> tuple[int, int]:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an int or a pair of ints")
    if isinstance(value, int):
        result = (value, value)
    elif isinstance(value, (tuple, list)) and len(value) == 2 and all(
        isinstance(item, int) and not isinstance(item, bool) for item in value
    ):
        result = (value[0], value[1])
    else:
        raise ValueError(f"{name} must be an int or a pair of ints")

    if allow_zero:
        if any(item < 0 for item in result):
            raise ValueError(f"{name} must be non-negative")
    elif any(item <= 0 for item in result):
        raise ValueError(f"{name} must be positive")
    return result


def _validate_lut(
    lut: torch.Tensor,
    kind: str,
    *,
    require_cuda: bool,
    device: torch.device | None = None,
) -> None:
    if not isinstance(lut, torch.Tensor):
        raise TypeError(f"lut must be a torch.Tensor, got {type(lut).__name__}")
    if lut.dtype != torch.uint16:
        raise TypeError(f"{kind} lut must have dtype torch.uint16, got {lut.dtype}")
    side = _LUT_SIDES[kind]
    if tuple(lut.shape) != (side, side):
        raise ValueError(
            f"{kind} lut must have shape ({side}, {side}), got {tuple(lut.shape)}"
        )
    if not lut.is_contiguous():
        raise ValueError("lut must be contiguous in row-major order")
    if require_cuda and not lut.is_cuda:
        raise ValueError("input, weight, lut, and bias must be CUDA tensors")
    if device is not None and lut.device != device:
        raise ValueError(f"lut must be on {device}, got {lut.device}")


def _validate_conv_tensors(
    input: torch.Tensor,
    weight: torch.Tensor,
    lut: torch.Tensor,
    bias: torch.Tensor | None,
    kind: str,
    groups: int,
) -> tuple[int, int, int, int, int, int, int]:
    tensors = (input, weight, lut) + (() if bias is None else (bias,))
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise TypeError("input, weight, lut, and bias must be torch.Tensor objects")
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("input, weight, lut, and bias must be CUDA tensors")

    dtype = _DTYPES[kind]
    if input.dtype != dtype or weight.dtype != dtype:
        raise TypeError(
            f"input and weight must have dtype {dtype}, got "
            f"{input.dtype} and {weight.dtype}"
        )
    if bias is not None and bias.dtype != dtype:
        raise TypeError(f"bias must have dtype {dtype}, got {bias.dtype}")
    if input.device != weight.device:
        raise ValueError("input and weight must be on the same CUDA device")
    _validate_lut(lut, kind, require_cuda=True, device=input.device)
    if bias is not None and bias.device != input.device:
        raise ValueError(f"bias must be on {input.device}, got {bias.device}")

    if input.ndim != 4:
        raise ValueError(f"input must be 4-dimensional, got shape {tuple(input.shape)}")
    if weight.ndim != 4:
        raise ValueError(f"weight must be 4-dimensional, got shape {tuple(weight.shape)}")
    if isinstance(groups, bool) or not isinstance(groups, int):
        raise TypeError("groups must be an integer")
    if groups != 1:
        raise NotImplementedError(
            f"{kind} convolution currently supports only groups=1, got {groups}"
        )

    batch, in_channels, height, width = input.shape
    out_channels, channels_per_group, kernel_h, kernel_w = weight.shape
    if channels_per_group != in_channels:
        raise ValueError(
            "weight.shape[1] must equal input channels when groups=1, got "
            f"{channels_per_group} and {in_channels}"
        )
    if out_channels == 0 or channels_per_group == 0:
        raise ValueError("input and output channel counts must be positive")
    if kernel_h <= 0 or kernel_w <= 0:
        raise ValueError("kernel dimensions must be positive")
    if bias is not None and tuple(bias.shape) != (out_channels,):
        raise ValueError(
            f"bias must have shape ({out_channels},), got {tuple(bias.shape)}"
        )
    return (
        batch,
        in_channels,
        height,
        width,
        out_channels,
        kernel_h,
        kernel_w,
    )


def conv2d_approx_float(
    input: torch.Tensor,
    weight: torch.Tensor,
    lut: torch.Tensor,
    bias: torch.Tensor | None,
    stride: int | tuple[int, int],
    padding: int | tuple[int, int],
    dilation: int | tuple[int, int],
    groups: int,
    optimized: bool,
    *,
    kind: str,
    bgemm: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, bool], torch.Tensor],
) -> torch.Tensor:
    """Implement a groups=1 Conv2d as unfold followed by one LUT BGEMM."""
    if not isinstance(optimized, bool):
        raise TypeError("optimized must be a bool")
    stride = _pair_parameter(stride, "stride", allow_zero=False)
    padding = _pair_parameter(padding, "padding", allow_zero=True)
    dilation = _pair_parameter(dilation, "dilation", allow_zero=False)
    (
        batch,
        _,
        height,
        width,
        out_channels,
        kernel_h,
        kernel_w,
    ) = _validate_conv_tensors(input, weight, lut, bias, kind, groups)

    out_h = (
        height + 2 * padding[0] - dilation[0] * (kernel_h - 1) - 1
    ) // stride[0] + 1
    out_w = (
        width + 2 * padding[1] - dilation[1] * (kernel_w - 1) - 1
    ) // stride[1] + 1
    if out_h <= 0 or out_w <= 0:
        raise ValueError(
            "calculated output spatial size must be positive, got "
            f"({out_h}, {out_w})"
        )

    columns = F.unfold(
        input.contiguous(),
        (kernel_h, kernel_w),
        dilation=dilation,
        padding=padding,
        stride=stride,
    )
    flat_weight = weight.reshape(out_channels, -1).contiguous()
    output = bgemm(columns.contiguous(), flat_weight, lut, optimized).reshape(
        batch, out_channels, out_h, out_w
    )
    if bias is not None:
        output = output + bias.view(1, -1, 1, 1)
    return output


class ApproxConv2dFloat(nn.Module):
    """Common nn.Module implementation for 16-bit approximate convolutions."""

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
        kind: str,
        operation: Callable[..., torch.Tensor],
    ) -> None:
        super().__init__()
        if isinstance(in_channels, bool) or not isinstance(in_channels, int):
            raise TypeError("in_channels must be an integer")
        if isinstance(out_channels, bool) or not isinstance(out_channels, int):
            raise TypeError("out_channels must be an integer")
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("in_channels and out_channels must be positive")
        if isinstance(groups, bool) or not isinstance(groups, int):
            raise TypeError("groups must be an integer")
        if groups != 1:
            raise NotImplementedError(
                f"{kind} convolution currently supports only groups=1, "
                f"got {groups}"
            )
        if padding_mode != "zeros":
            raise NotImplementedError(
                "only padding_mode='zeros' is supported, got "
                f"{padding_mode!r}"
            )
        if not isinstance(optimized, bool):
            raise TypeError("optimized must be a bool")
        expected_dtype = _DTYPES[kind]
        if dtype is not None and dtype != expected_dtype:
            raise TypeError(
                f"{kind} convolution requires dtype {expected_dtype}, got {dtype}"
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair_parameter(
            kernel_size, "kernel_size", allow_zero=False
        )
        self.stride = _pair_parameter(stride, "stride", allow_zero=False)
        self.padding = _pair_parameter(padding, "padding", allow_zero=True)
        self.dilation = _pair_parameter(dilation, "dilation", allow_zero=False)
        self.groups = groups
        self.padding_mode = padding_mode
        self.optimized = optimized
        self.kind = kind
        self._operation = operation

        if not isinstance(lut, torch.Tensor):
            raise TypeError(f"lut must be a torch.Tensor, got {type(lut).__name__}")
        lut_device = torch.device(device) if device is not None else lut.device
        # Older FP16 generators emitted uint32 LUTs, although entries use only
        # 11 bits. Normalize those LUTs to the uint16 CUDA-kernel contract.
        if kind == "fp16" and lut.dtype == torch.uint32:
            lut = lut.to(dtype=torch.uint16)
        _validate_lut(lut, kind, require_cuda=False)
        self.register_buffer("lut", lut.to(device=lut_device))

        factory_kwargs = {"device": lut_device, "dtype": expected_dtype}
        self.weight = nn.Parameter(
            torch.empty(
                out_channels,
                in_channels,
                self.kernel_size[0],
                self.kernel_size[1],
                **factory_kwargs,
            )
        )
        if isinstance(bias, bool):
            self.bias = (
                nn.Parameter(torch.empty(out_channels, **factory_kwargs))
                if bias
                else None
            )
            supplied_bias = None
        elif bias is None:
            self.bias = None
            supplied_bias = None
        elif isinstance(bias, torch.Tensor):
            if tuple(bias.shape) != (out_channels,):
                raise ValueError(
                    f"bias must have shape ({out_channels},), got {tuple(bias.shape)}"
                )
            self.bias = nn.Parameter(torch.empty(out_channels, **factory_kwargs))
            supplied_bias = bias
        else:
            raise TypeError("bias must be a bool, torch.Tensor, or None")

        self.reset_parameters()
        if supplied_bias is not None:
            with torch.no_grad():
                supplied_bias = supplied_bias.to(device=lut_device, dtype=expected_dtype)
                self.bias.copy_(supplied_bias)

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_channels * self.kernel_size[0] * self.kernel_size[1]
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

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
        )

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, dilation={self.dilation}, "
            f"groups={self.groups}, bias={self.bias is not None}, "
            f"dtype={self.kind}, optimized={self.optimized}"
        )
