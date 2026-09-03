"""LUT-based approximate BF16 Linear layer."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.autograd import Function

from approxtorch.backend import ops


__all__ = ["Linear_bfloat16", "Linear_bf16", "linear_bfloat16", "linear_bf16"]


_LUT_SIDE = 128


def _validate_lut(
    lut: torch.Tensor,
    *,
    require_cuda: bool,
    device: torch.device | None = None,
) -> None:
    if not isinstance(lut, torch.Tensor):
        raise TypeError(f"lut must be a torch.Tensor, got {type(lut).__name__}")
    if lut.dtype != torch.uint16:
        raise TypeError(f"bf16 lut must have dtype torch.uint16, got {lut.dtype}")
    if tuple(lut.shape) != (_LUT_SIDE, _LUT_SIDE):
        raise ValueError(
            f"bf16 lut must have shape ({_LUT_SIDE}, {_LUT_SIDE}), "
            f"got {tuple(lut.shape)}"
        )
    if not lut.is_contiguous():
        raise ValueError("lut must be contiguous in row-major order")
    if require_cuda and not lut.is_cuda:
        raise ValueError("input, weight, lut, and bias must be CUDA tensors")
    if device is not None and lut.device != device:
        raise ValueError(f"lut must be on {device}, got {lut.device}")


def _validate_linear_tensors(
    input: torch.Tensor,
    weight: torch.Tensor,
    lut: torch.Tensor,
    bias: torch.Tensor | None,
) -> tuple[int, int]:
    tensors = (input, weight, lut) + (() if bias is None else (bias,))
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise TypeError("input, weight, lut, and bias must be torch.Tensor objects")
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("input, weight, lut, and bias must be CUDA tensors")
    if input.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise TypeError(
            "input and weight must have dtype torch.bfloat16, got "
            f"{input.dtype} and {weight.dtype}"
        )
    if bias is not None and bias.dtype != torch.bfloat16:
        raise TypeError(f"bias must have dtype torch.bfloat16, got {bias.dtype}")
    if input.device != weight.device:
        raise ValueError("input and weight must be on the same CUDA device")
    _validate_lut(lut, require_cuda=True, device=input.device)
    if bias is not None and bias.device != input.device:
        raise ValueError(f"bias must be on {input.device}, got {bias.device}")
    if input.ndim == 0:
        raise ValueError("input must have at least one dimension")
    if weight.ndim != 2:
        raise ValueError(f"weight must be 2-dimensional, got {tuple(weight.shape)}")

    out_features, in_features = weight.shape
    if in_features <= 0 or out_features <= 0:
        raise ValueError("in_features and out_features must be positive")
    if input.shape[-1] != in_features:
        raise ValueError(
            f"input.shape[-1] must be {in_features}, got {input.shape[-1]}"
        )
    if bias is not None and tuple(bias.shape) != (out_features,):
        raise ValueError(
            f"bias must have shape ({out_features},), got {tuple(bias.shape)}"
        )
    return in_features, out_features


class _LinearBFloat16STE(Function):
    @staticmethod
    def forward(
        ctx,
        input_2d: torch.Tensor,
        weight: torch.Tensor,
        lut: torch.Tensor,
        optimized: bool,
    ) -> torch.Tensor:
        ctx.save_for_backward(input_2d, weight)
        transposed_weight = weight.transpose(0, 1).contiguous()
        if optimized:
            return ops.gemm_bf16(input_2d, transposed_weight, lut)
        return ops.gemm_bf16_naive(input_2d, transposed_weight, lut)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        input_2d, weight = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_input = (
            grad_output.matmul(weight) if ctx.needs_input_grad[0] else None
        )
        grad_weight = (
            grad_output.transpose(0, 1).matmul(input_2d)
            if ctx.needs_input_grad[1]
            else None
        )
        return grad_input, grad_weight, None, None


def linear_bfloat16(
    input: torch.Tensor,
    weight: torch.Tensor,
    lut: torch.Tensor,
    bias: torch.Tensor | None = None,
    optimized: bool = True,
) -> torch.Tensor:
    """Apply an approximate BF16 linear transform with an exact-product STE."""
    if not isinstance(optimized, bool):
        raise TypeError("optimized must be a bool")
    in_features, out_features = _validate_linear_tensors(
        input, weight, lut, bias
    )
    input_2d = input.reshape(-1, in_features).contiguous()
    output = _LinearBFloat16STE.apply(
        input_2d, weight.contiguous(), lut, optimized
    )
    output = output.reshape(*input.shape[:-1], out_features)
    if bias is not None:
        output = output + bias
    return output


class Linear_bfloat16(nn.Module):
    """Approximate BF16 Linear layer with an exact-product STE backward."""

    __constants__ = ["in_features", "out_features", "optimized"]

    def __init__(
        self,
        in_features: int,
        out_features: int,
        lut: torch.Tensor,
        bias: torch.Tensor | bool | None = True,
        optimized: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if isinstance(in_features, bool) or not isinstance(in_features, int):
            raise TypeError("in_features must be an integer")
        if isinstance(out_features, bool) or not isinstance(out_features, int):
            raise TypeError("out_features must be an integer")
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive")
        if not isinstance(optimized, bool):
            raise TypeError("optimized must be a bool")
        if dtype is not None and dtype != torch.bfloat16:
            raise TypeError(
                f"bf16 linear requires dtype torch.bfloat16, got {dtype}"
            )
        if not isinstance(lut, torch.Tensor):
            raise TypeError(f"lut must be a torch.Tensor, got {type(lut).__name__}")

        self.in_features = in_features
        self.out_features = out_features
        self.optimized = optimized

        target_device = torch.device(device) if device is not None else lut.device
        _validate_lut(lut, require_cuda=False)
        self.register_buffer("lut", lut.to(device=target_device))

        factory_kwargs = {"device": target_device, "dtype": torch.bfloat16}
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        if isinstance(bias, bool):
            self.bias = (
                nn.Parameter(torch.empty(out_features, **factory_kwargs))
                if bias
                else None
            )
            supplied_bias = None
        elif bias is None:
            self.bias = None
            supplied_bias = None
        elif isinstance(bias, torch.Tensor):
            if tuple(bias.shape) != (out_features,):
                raise ValueError(
                    f"bias must have shape ({out_features},), got {tuple(bias.shape)}"
                )
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
            supplied_bias = bias
        else:
            raise TypeError("bias must be a bool, torch.Tensor, or None")

        self.reset_parameters()
        if supplied_bias is not None:
            with torch.no_grad():
                self.bias.copy_(
                    supplied_bias.to(device=target_device, dtype=torch.bfloat16)
                )

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return linear_bfloat16(
            input,
            self.weight,
            self.lut,
            self.bias,
            self.optimized,
        )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, dtype=bf16, "
            f"optimized={self.optimized}"
        )


Linear_bf16 = Linear_bfloat16
linear_bf16 = linear_bfloat16
