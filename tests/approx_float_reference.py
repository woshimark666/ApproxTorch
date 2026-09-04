"""Independent bit-level references for ApproxTorch's 16-bit float ops."""

from __future__ import annotations

from dataclasses import dataclass
import math
import struct
from typing import Literal, Sequence

import torch
from torch import Tensor


FloatKind = Literal["fp16", "bf16"]


@dataclass(frozen=True)
class FloatSpec:
    fraction_bits: int
    exponent_bits: int
    bias: int
    min_exponent: int
    max_exponent: int
    canonical_nan: int
    dtype: torch.dtype

    @property
    def sign_mask(self) -> int:
        return 0x8000

    @property
    def fraction_mask(self) -> int:
        return (1 << self.fraction_bits) - 1

    @property
    def exponent_field_mask(self) -> int:
        return (1 << self.exponent_bits) - 1

    @property
    def exponent_mask(self) -> int:
        return self.exponent_field_mask << self.fraction_bits

    @property
    def infinity(self) -> int:
        return self.exponent_mask

    @property
    def quiet_nan_bit(self) -> int:
        return 1 << (self.fraction_bits - 1)

    @property
    def lut_side(self) -> int:
        return 1 << self.fraction_bits


SPECS = {
    "fp16": FloatSpec(10, 5, 15, -14, 15, 0x7E00, torch.float16),
    "bf16": FloatSpec(7, 8, 127, -126, 127, 0x7FC0, torch.bfloat16),
}


def unsigned_bits(tensor: Tensor) -> Tensor:
    return tensor.contiguous().view(torch.int16).to(torch.int32).bitwise_and(0xFFFF)


def tensor_from_bits(
    values: Sequence[int], shape: Sequence[int], dtype: torch.dtype
) -> Tensor:
    signed = [value if value < 0x8000 else value - 0x10000 for value in values]
    return torch.tensor(signed, dtype=torch.int16).reshape(tuple(shape)).view(dtype)


def _float_value_from_bits(bits: int, spec: FloatSpec) -> float:
    """Convert a raw FP16/BF16 pattern to an exactly representable Python float."""
    bits &= 0xFFFF
    raw_exponent = (bits & spec.exponent_mask) >> spec.fraction_bits
    fraction = bits & spec.fraction_mask
    negative = bool(bits & spec.sign_mask)
    if raw_exponent == spec.exponent_field_mask:
        if fraction:
            return math.nan
        return -math.inf if negative else math.inf
    if raw_exponent == 0:
        value = math.ldexp(
            float(fraction), spec.min_exponent - spec.fraction_bits
        )
    else:
        value = math.ldexp(
            float((1 << spec.fraction_bits) | fraction),
            raw_exponent - spec.bias - spec.fraction_bits,
        )
    return -value if negative else value


def _round_float32(value: float) -> float:
    """Round a Python binary64 value to IEEE binary32, including overflow."""
    try:
        return struct.unpack("<f", struct.pack("<f", value))[0]
    except OverflowError:
        return math.copysign(math.inf, value)


def _float32_add_product(
    accumulator: float, product_bits: int, spec: FloatSpec
) -> float:
    return _round_float32(
        accumulator + _float_value_from_bits(product_bits, spec)
    )


def _float32_to_target_bits(value: float, spec: FloatSpec) -> int:
    tensor = torch.tensor(value, dtype=torch.float32).to(spec.dtype)
    return int(tensor.view(torch.int16).item()) & 0xFFFF


def _round_shift_rne(value: int, shift: int) -> int:
    if shift <= 0:
        return value << -shift
    if shift > value.bit_length() + 1:
        return 0
    truncated = value >> shift
    remainder = value & ((1 << shift) - 1)
    halfway = 1 << (shift - 1)
    return truncated + int(
        remainder > halfway or (remainder == halfway and (truncated & 1) != 0)
    )


def _normalize_operand(raw_exponent: int, fraction: int, spec: FloatSpec) -> tuple[int, int]:
    if raw_exponent != 0:
        return raw_exponent - spec.bias, fraction
    shift = spec.fraction_bits - (fraction.bit_length() - 1)
    significand = fraction << shift
    return spec.min_exponent - shift, significand & spec.fraction_mask


def _pack_approximate_product(
    sign: int,
    exponent_a: int,
    exponent_b: int,
    product: int,
    spec: FloatSpec,
) -> int:
    if product == 0:
        return sign
    leading_bit = product.bit_length() - 1
    result_exponent = (
        exponent_a + exponent_b - 2 * spec.fraction_bits + leading_bit
    )
    if result_exponent > spec.max_exponent:
        return sign | spec.infinity
    if result_exponent >= spec.min_exponent:
        significand = _round_shift_rne(product, leading_bit - spec.fraction_bits)
        if significand >= 1 << (spec.fraction_bits + 1):
            significand >>= 1
            result_exponent += 1
            if result_exponent > spec.max_exponent:
                return sign | spec.infinity
        raw_exponent = (result_exponent + spec.bias) << spec.fraction_bits
        return sign | raw_exponent | (significand & spec.fraction_mask)

    shift = (
        spec.fraction_bits + spec.min_exponent - exponent_a - exponent_b
    )
    fraction = _round_shift_rne(product, shift)
    if fraction == 0:
        return sign
    if fraction >= 1 << spec.fraction_bits:
        return sign | (1 << spec.fraction_bits)
    return sign | fraction


def approx_mul_bits(lhs: int, rhs: int, lut: Tensor, kind: FloatKind) -> int:
    """Reference the RTL mantissa-result LUT multiplier."""
    spec = SPECS[kind]
    lhs &= 0xFFFF
    rhs &= 0xFFFF
    if (lhs & 0x7FFF) == 0 or (rhs & 0x7FFF) == 0:
        return 0

    exponent_a = (lhs >> spec.fraction_bits) & spec.exponent_field_mask
    exponent_b = (rhs >> spec.fraction_bits) & spec.exponent_field_mask
    fraction_a = lhs & spec.fraction_mask
    fraction_b = rhs & spec.fraction_mask
    entry = int(lut[fraction_a, fraction_b].item())
    normalization = (entry >> spec.fraction_bits) & 1
    exponent = (
        exponent_a + exponent_b - spec.bias + normalization
    ) & spec.exponent_field_mask
    sign = (lhs ^ rhs) & spec.sign_mask
    return sign | (exponent << spec.fraction_bits) | (
        entry & spec.fraction_mask
    )

def _finite_components(bits: int, spec: FloatSpec) -> tuple[int, int]:
    raw_exponent = (bits & spec.exponent_mask) >> spec.fraction_bits
    fraction = bits & spec.fraction_mask
    sign = -1 if bits & spec.sign_mask else 1
    if raw_exponent == 0:
        significand = fraction
        power = spec.min_exponent - spec.fraction_bits
    else:
        significand = (1 << spec.fraction_bits) | fraction
        power = raw_exponent - spec.bias - spec.fraction_bits
    return sign * significand, power


def _pack_exact_integer(significand: int, power: int, spec: FloatSpec) -> int:
    if significand == 0:
        return 0
    sign = spec.sign_mask if significand < 0 else 0
    magnitude = abs(significand)
    leading_bit = magnitude.bit_length() - 1
    result_exponent = power + leading_bit
    if result_exponent > spec.max_exponent:
        return sign | spec.infinity
    if result_exponent >= spec.min_exponent:
        rounded = _round_shift_rne(magnitude, leading_bit - spec.fraction_bits)
        if rounded >= 1 << (spec.fraction_bits + 1):
            rounded >>= 1
            result_exponent += 1
            if result_exponent > spec.max_exponent:
                return sign | spec.infinity
        raw_exponent = (result_exponent + spec.bias) << spec.fraction_bits
        return sign | raw_exponent | (rounded & spec.fraction_mask)

    subnormal_power = spec.min_exponent - spec.fraction_bits
    fraction = _round_shift_rne(magnitude, subnormal_power - power)
    if fraction == 0:
        return sign
    if fraction >= 1 << spec.fraction_bits:
        return sign | (1 << spec.fraction_bits)
    return sign | fraction


def add_bits(lhs: int, rhs: int, kind: FloatKind) -> int:
    """Round-to-nearest-even 16-bit addition with no wider accumulator."""
    spec = SPECS[kind]
    lhs &= 0xFFFF
    rhs &= 0xFFFF
    exp_a = (lhs & spec.exponent_mask) >> spec.fraction_bits
    exp_b = (rhs & spec.exponent_mask) >> spec.fraction_bits
    frac_a = lhs & spec.fraction_mask
    frac_b = rhs & spec.fraction_mask
    if exp_a == spec.exponent_field_mask and frac_a:
        return lhs | spec.quiet_nan_bit
    if exp_b == spec.exponent_field_mask and frac_b:
        return rhs | spec.quiet_nan_bit
    if exp_a == spec.exponent_field_mask:
        if exp_b == spec.exponent_field_mask and ((lhs ^ rhs) & spec.sign_mask):
            return spec.canonical_nan
        return lhs
    if exp_b == spec.exponent_field_mask:
        return rhs

    a_zero = exp_a == 0 and frac_a == 0
    b_zero = exp_b == 0 and frac_b == 0
    if a_zero and b_zero:
        return spec.sign_mask if (lhs & rhs & spec.sign_mask) else 0
    if a_zero:
        return rhs
    if b_zero:
        return lhs

    significand_a, power_a = _finite_components(lhs, spec)
    significand_b, power_b = _finite_components(rhs, spec)
    common_power = min(power_a, power_b)
    exact = (
        (significand_a << (power_a - common_power))
        + (significand_b << (power_b - common_power))
    )
    return _pack_exact_integer(exact, common_power, spec)


def strict_gemm_reference(A: Tensor, B: Tensor, lut: Tensor, kind: FloatKind) -> Tensor:
    if A.device.type != "cpu" or B.device.type != "cpu" or lut.device.type != "cpu":
        raise ValueError("strict reference inputs must be CPU tensors")
    M, K = A.shape
    N = B.shape[1]
    a_bits = unsigned_bits(A).tolist()
    b_bits = unsigned_bits(B).tolist()
    output: list[int] = []
    for row in range(M):
        for col in range(N):
            accumulator = 0.0
            for k in range(K):
                product = approx_mul_bits(a_bits[row][k], b_bits[k][col], lut, kind)
                accumulator = _float32_add_product(
                    accumulator, product, SPECS[kind]
                )
            output.append(_float32_to_target_bits(accumulator, SPECS[kind]))
    return tensor_from_bits(output, (M, N), SPECS[kind].dtype)


def strict_bgemm_reference(
    X: Tensor, W: Tensor, lut: Tensor, kind: FloatKind
) -> Tensor:
    if X.device.type != "cpu" or W.device.type != "cpu" or lut.device.type != "cpu":
        raise ValueError("strict reference inputs must be CPU tensors")
    batch, K, L = X.shape
    O = W.shape[0]
    x_bits = unsigned_bits(X).tolist()
    w_bits = unsigned_bits(W).tolist()
    output: list[int] = []
    for n in range(batch):
        for o in range(O):
            for l in range(L):
                accumulator = 0.0
                for k in range(K):
                    product = approx_mul_bits(x_bits[n][k][l], w_bits[o][k], lut, kind)
                    accumulator = _float32_add_product(
                        accumulator, product, SPECS[kind]
                    )
                output.append(
                    _float32_to_target_bits(accumulator, SPECS[kind])
                )
    return tensor_from_bits(output, (batch, O, L), SPECS[kind].dtype)


def strict_conv2d_reference(
    input: Tensor,
    weight: Tensor,
    lut: Tensor,
    kind: FloatKind,
    bias: Tensor | None = None,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
    groups: int = 1,
) -> Tensor:
    """Strict Conv2d reference with input as LUT operand A.

    Each LUT product is evaluated from raw FP16/BF16 bit patterns, converted
    to FP32, and accumulated in FP32. The complete dot product is rounded once
    to the target 16-bit format. Padding contributes positive-zero operands;
    bias is then added once in the target format. The reduction order is
    channel, kernel row, then kernel column,
    matching ``torch.nn.functional.unfold`` and ApproxTorch BGEMM.
    """
    tensors = (input, weight, lut) + (() if bias is None else (bias,))
    if any(tensor.device.type != "cpu" for tensor in tensors):
        raise ValueError("strict reference inputs must be CPU tensors")

    def pair(value: int | tuple[int, int]) -> tuple[int, int]:
        return (value, value) if isinstance(value, int) else tuple(value)

    stride_h, stride_w = pair(stride)
    pad_h, pad_w = pair(padding)
    dilation_h, dilation_w = pair(dilation)
    batch, in_channels, height, width = input.shape
    out_channels, channels_per_group, kernel_h, kernel_w = weight.shape
    if in_channels != channels_per_group * groups:
        raise ValueError("input/weight/groups channel geometry is inconsistent")
    if out_channels % groups:
        raise ValueError("output channels must be divisible by groups")
    if bias is not None and bias.shape != (out_channels,):
        raise ValueError("bias shape is inconsistent with output channels")

    out_h = (
        height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1
    ) // stride_h + 1
    out_w = (
        width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1
    ) // stride_w + 1
    input_bits = unsigned_bits(input).tolist()
    weight_bits = unsigned_bits(weight).tolist()
    bias_bits = None if bias is None else unsigned_bits(bias).tolist()
    outputs_per_group = out_channels // groups
    output: list[int] = []
    for n in range(batch):
        for out_channel in range(out_channels):
            group = out_channel // outputs_per_group
            input_channel_start = group * channels_per_group
            for out_row in range(out_h):
                input_row_base = out_row * stride_h - pad_h
                for out_col in range(out_w):
                    input_col_base = out_col * stride_w - pad_w
                    accumulator = 0.0
                    for channel_offset in range(channels_per_group):
                        input_channel = input_channel_start + channel_offset
                        for kernel_row in range(kernel_h):
                            input_row = input_row_base + kernel_row * dilation_h
                            for kernel_col in range(kernel_w):
                                input_col = input_col_base + kernel_col * dilation_w
                                if (
                                    0 <= input_row < height
                                    and 0 <= input_col < width
                                ):
                                    lhs = input_bits[n][input_channel][input_row][input_col]
                                else:
                                    lhs = 0
                                rhs = weight_bits[out_channel][channel_offset][kernel_row][kernel_col]
                                product = approx_mul_bits(lhs, rhs, lut, kind)
                                accumulator = _float32_add_product(
                                    accumulator, product, SPECS[kind]
                                )
                    result = _float32_to_target_bits(
                        accumulator, SPECS[kind]
                    )
                    if bias_bits is not None:
                        result = add_bits(
                            result, bias_bits[out_channel], kind
                        )
                    output.append(result)
    return tensor_from_bits(
        output,
        (batch, out_channels, out_h, out_w),
        SPECS[kind].dtype,
    )
