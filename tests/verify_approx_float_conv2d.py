#!/usr/bin/env python3
"""Bit-level and autograd checks for approximate FP16/BF16 Conv2d."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import approxtorch as at
from approxtorch.float_lut import load_exact_lut, validate_mantissa_lut
from approx_float_reference import (
    SPECS,
    strict_conv2d_reference,
    tensor_from_bits,
    unsigned_bits,
)


def assert_result_bits(actual: torch.Tensor, expected: torch.Tensor, kind: str) -> None:
    """Require exact non-NaN bits and matching quiet-NaN classification."""
    spec = SPECS[kind]
    actual_bits = unsigned_bits(actual.cpu())
    expected_bits = unsigned_bits(expected.cpu())
    actual_nan = (
        ((actual_bits & spec.exponent_mask) == spec.exponent_mask)
        & ((actual_bits & spec.fraction_mask) != 0)
    )
    expected_nan = (
        ((expected_bits & spec.exponent_mask) == spec.exponent_mask)
        & ((expected_bits & spec.fraction_mask) != 0)
    )
    if not torch.equal(actual_nan, expected_nan):
        raise AssertionError("NaN classification mismatch")
    compare = ~expected_nan
    if not torch.equal(actual_bits[compare], expected_bits[compare]):
        count = int((actual_bits[compare] != expected_bits[compare]).sum())
        raise AssertionError(f"{count} non-NaN Conv2d outputs differ at bit level")
    if actual_nan.any() and not torch.all(
        (actual_bits[actual_nan] & spec.quiet_nan_bit) != 0
    ):
        raise AssertionError("Conv2d produced a signaling NaN")


def assert_same_bits(lhs: torch.Tensor, rhs: torch.Tensor, message: str) -> None:
    if not torch.equal(unsigned_bits(lhs.cpu()), unsigned_bits(rhs.cpu())):
        raise AssertionError(message)


def expect_error(error_type, pattern: str, function) -> None:
    try:
        function()
    except error_type as error:
        if pattern not in str(error):
            raise AssertionError(
                f"expected error containing {pattern!r}, got {str(error)!r}"
            ) from error
    else:
        raise AssertionError(f"expected {error_type.__name__} containing {pattern!r}")


def make_asymmetric_lut(kind: str) -> torch.Tensor:
    spec = SPECS[kind]
    side = spec.lut_side
    rows = torch.arange(side, dtype=torch.int64)[:, None]
    cols = torch.arange(side, dtype=torch.int64)[None, :]
    fraction = (3 * rows + 5 * cols) & spec.fraction_mask
    normalization = ((7 * rows + 11 * cols) >> (
        spec.fraction_bits - 2
    )) & 1
    values = (normalization << spec.fraction_bits) | fraction
    lut = values.to(torch.uint32).contiguous()
    return validate_mantissa_lut(lut, kind, require_cuda=False)


def finite_tensor(shape: tuple[int, ...], dtype: torch.dtype, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return (torch.randn(shape, generator=generator) * 0.625).to(dtype)


def verify_strict_forward(
    kind: str,
    exact_cpu: torch.Tensor,
    exact_cuda: torch.Tensor,
) -> None:
    spec = SPECS[kind]
    operation = getattr(at.nn, f"conv2d_{kind}")
    asymmetric_cpu = make_asymmetric_lut(kind)
    asymmetric_cuda = asymmetric_cpu.cuda()
    cases = (
        {
            "name": "regular",
            "input_shape": (1, 2, 4, 5),
            "weight_shape": (3, 2, 2, 3),
            "stride": (1, 2),
            "padding": (1, 0),
            "dilation": (1, 1),
            "groups": 1,
            "bias": False,
        },
    )
    for lut_label, lut_cpu, lut_cuda in (
        ("exact", exact_cpu, exact_cuda),
        ("asymmetric", asymmetric_cpu, asymmetric_cuda),
    ):
        for index, case in enumerate(cases):
            input_cpu = finite_tensor(case["input_shape"], spec.dtype, 100 + index)
            weight_cpu = finite_tensor(case["weight_shape"], spec.dtype, 200 + index)
            bias_cpu = (
                finite_tensor((case["weight_shape"][0],), spec.dtype, 300 + index)
                if case["bias"]
                else None
            )
            args = dict(
                bias=bias_cpu,
                stride=case["stride"],
                padding=case["padding"],
                dilation=case["dilation"],
                groups=case["groups"],
            )
            expected = strict_conv2d_reference(
                input_cpu, weight_cpu, lut_cpu, kind, **args
            )
            cuda_args = dict(args)
            cuda_args["bias"] = None if bias_cpu is None else bias_cpu.cuda()
            naive = operation(
                input_cpu.cuda(),
                weight_cpu.cuda(),
                lut_cuda,
                optimized=False,
                **cuda_args,
            )
            optimized = operation(
                input_cpu.cuda(),
                weight_cpu.cuda(),
                lut_cuda,
                optimized=True,
                **cuda_args,
            )
            assert_result_bits(naive, expected, kind)
            assert_same_bits(
                optimized,
                naive,
                f"{kind} {lut_label} {case['name']} optimized/naive mismatch",
            )

    # Special encodings make signed zero, subnormal, infinity, and NaN behavior
    # visible, including positive-zero padding multiplied by infinite weights.
    boundary = (
        [0x0000, 0x8000, 0x0001, 0x03FF, 0x0400, 0x3C00,
         0xBC00, 0x7C00, 0xFC00, 0x7E01, 0x7C01, 0x3555]
        if kind == "fp16"
        else [0x0000, 0x8000, 0x0001, 0x007F, 0x0080, 0x3F80,
              0xBF80, 0x7F80, 0xFF80, 0x7FC1, 0x7F81, 0x3EAB]
    )
    input_cpu = tensor_from_bits(
        [boundary[index % len(boundary)] for index in range(12)],
        (1, 2, 2, 3),
        spec.dtype,
    )
    weight_cpu = tensor_from_bits(
        [boundary[(index * 5 + 3) % len(boundary)] for index in range(16)],
        (2, 2, 2, 2),
        spec.dtype,
    )
    bias_cpu = tensor_from_bits([boundary[1], boundary[4]], (2,), spec.dtype)
    expected = strict_conv2d_reference(
        input_cpu,
        weight_cpu,
        asymmetric_cpu,
        kind,
        bias=bias_cpu,
        padding=1,
    )
    actual = operation(
        input_cpu.cuda(),
        weight_cpu.cuda(),
        asymmetric_cuda,
        bias_cpu.cuda(),
        padding=1,
    )
    assert_result_bits(actual, expected, kind)
    print(f"PASS {kind} strict Conv2d reference: exact/asymmetric/special")


def verify_module_api(kind: str, exact_cpu: torch.Tensor) -> None:
    spec = SPECS[kind]
    module_type = getattr(at.nn, f"Conv2d_{kind}")
    functional = getattr(at.nn, f"conv2d_{kind}")
    supplied_bias = finite_tensor((6,), spec.dtype, 401)
    module = module_type(
        4,
        6,
        (3, 2),
        exact_cpu,
        stride=(2, 1),
        padding=(2, 1),
        dilation=(2, 1),
        bias=supplied_bias,
        optimized=True,
    )
    if module.weight.dtype != spec.dtype or module.bias.dtype != spec.dtype:
        raise AssertionError("module parameters do not use the selected dtype")
    if not torch.equal(module.bias.detach(), supplied_bias):
        raise AssertionError("tensor-valued bias was not copied")
    if set(module.state_dict()) != {"weight", "bias", "lut"}:
        raise AssertionError("module state_dict does not contain weight/bias/LUT")
    module = module.cuda()
    input_cuda = finite_tensor((2, 4, 7, 6), spec.dtype, 402).cuda()
    module_result = module(input_cuda)
    functional_result = functional(
        input_cuda,
        module.weight,
        module.lut,
        module.bias,
        module.stride,
        module.padding,
        module.dilation,
        module.groups,
    )
    assert_same_bits(module_result, functional_result, f"{kind} module mismatch")

    clone = module_type(
        4,
        6,
        (3, 2),
        exact_cpu.cuda(),
        stride=(2, 1),
        padding=(2, 1),
        dilation=(2, 1),
        bias=True,
    )
    clone.load_state_dict(module.state_dict())
    assert_same_bits(clone(input_cuda), module_result, f"{kind} state round-trip mismatch")
    if f"dtype={spec.dtype}" not in repr(module) or "optimized=True" not in repr(module):
        raise AssertionError("module repr omits approximate configuration")
    print(f"PASS {kind} nn.Module API, tensor bias, state_dict, device movement")


def verify_ste_backward(kind: str, lut: torch.Tensor) -> None:
    spec = SPECS[kind]
    operation = getattr(at.nn, f"conv2d_{kind}")
    geometry = dict(stride=(2, 1), padding=(1, 2), dilation=(1, 2))
    input_actual = finite_tensor((2, 4, 6, 7), spec.dtype, 501).cuda()
    weight_actual = finite_tensor((6, 4, 3, 2), spec.dtype, 502).cuda()
    bias_actual = finite_tensor((6,), spec.dtype, 503).cuda()
    input_actual.requires_grad_(True)
    weight_actual.requires_grad_(True)
    bias_actual.requires_grad_(True)
    output_actual = operation(
        input_actual, weight_actual, lut, bias_actual, **geometry
    )
    upstream = finite_tensor(tuple(output_actual.shape), spec.dtype, 504).cuda()
    output_actual.backward(upstream)

    input_native = input_actual.detach().clone().requires_grad_(True)
    weight_native = weight_actual.detach().clone().requires_grad_(True)
    bias_native = bias_actual.detach().clone().requires_grad_(True)
    output_native = F.conv2d(
        input_native, weight_native, bias_native, **geometry
    )
    output_native.backward(upstream)
    tolerances = dict(rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(input_actual.grad, input_native.grad, **tolerances)
    torch.testing.assert_close(weight_actual.grad, weight_native.grad, **tolerances)
    torch.testing.assert_close(bias_actual.grad, bias_native.grad, **tolerances)
    print(f"PASS {kind} STE backward equals ordinary Conv2d gradients")


def verify_model_conversion(kind: str, lut: torch.Tensor) -> None:
    spec = SPECS[kind]
    module_type = getattr(at.nn, f"Conv2d_{kind}")
    model = nn.Sequential(
        nn.Conv2d(4, 6, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(6, 6, 1),
    )
    model = at.convert_float_model(
        model,
        lut,
        qtype=kind,
        ignore_first_conv=False,
        optimized=True,
    ).to(device="cuda", dtype=spec.dtype)
    if not isinstance(model[0], module_type) or not isinstance(model[2], module_type):
        raise AssertionError("convert_float_model did not replace every selected Conv2d")
    input_cuda = finite_tensor((2, 4, 5, 5), spec.dtype, 550).cuda()
    input_cuda.requires_grad_(True)
    output = model(input_cuda)
    if output.shape != (2, 6, 5, 5) or output.dtype != spec.dtype:
        raise AssertionError("converted model output contract is incorrect")
    output.float().square().mean().backward()
    if input_cuda.grad is None or any(
        module.weight.grad is None for module in (model[0], model[2])
    ):
        raise AssertionError("converted model did not propagate STE gradients")
    print(f"PASS {kind} convert_float_model end-to-end forward/backward")


def verify_empty_and_layout(kind: str, lut: torch.Tensor) -> None:
    spec = SPECS[kind]
    operation = getattr(at.nn, f"conv2d_{kind}")
    empty = torch.empty((0, 4, 5, 5), dtype=spec.dtype, device="cuda")
    weight = finite_tensor((6, 4, 3, 3), spec.dtype, 601).cuda()
    bias = finite_tensor((6,), spec.dtype, 602).cuda()
    naive = operation(empty, weight, lut, bias, padding=1, optimized=False)
    optimized = operation(empty, weight, lut, bias, padding=1)
    if naive.shape != (0, 6, 5, 5) or optimized.shape != naive.shape:
        raise AssertionError("empty-batch output shape is incorrect")

    input_contiguous = finite_tensor((1, 4, 5, 6), spec.dtype, 603).cuda()
    input_channels_last = input_contiguous.contiguous(
        memory_format=torch.channels_last
    )
    result_a = operation(input_contiguous, weight, lut, bias, padding=1)
    result_b = operation(input_channels_last, weight, lut, bias, padding=1)
    assert_same_bits(result_a, result_b, f"{kind} channels-last mismatch")
    print(f"PASS {kind} empty batch and channels-last input")


def verify_contract(kind: str, lut: torch.Tensor) -> None:
    spec = SPECS[kind]
    operation = getattr(at.nn, f"conv2d_{kind}")
    module_type = getattr(at.nn, f"Conv2d_{kind}")
    input_cuda = torch.ones((1, 4, 5, 5), dtype=spec.dtype, device="cuda")
    weight_cuda = torch.ones((6, 4, 3, 3), dtype=spec.dtype, device="cuda")
    expect_error(
        ValueError,
        "CUDA tensors",
        lambda: operation(input_cuda.cpu(), weight_cuda.cpu(), lut.cpu()),
    )
    expect_error(
        TypeError,
        "dtype",
        lambda: operation(input_cuda.float(), weight_cuda, lut),
    )
    expect_error(
        ValueError,
        "shape",
        lambda: operation(input_cuda, weight_cuda, lut.reshape(-1)),
    )
    expect_error(
        ValueError,
        "weight.shape[1]",
        lambda: operation(input_cuda, weight_cuda[:, :1], lut),
    )
    expect_error(
        TypeError,
        "groups must be an integer",
        lambda: operation(input_cuda, weight_cuda, lut, groups=True),
    )
    expect_error(
        NotImplementedError,
        "only groups=1",
        lambda: operation(input_cuda, weight_cuda, lut, groups=2),
    )
    expect_error(
        NotImplementedError,
        "only groups=1",
        lambda: module_type(4, 6, 3, lut.cpu(), groups=2),
    )
    expect_error(
        TypeError,
        "optimized must be a bool",
        lambda: operation(input_cuda, weight_cuda, lut, optimized=1),
    )
    expect_error(
        ValueError,
        "stride must be positive",
        lambda: operation(input_cuda, weight_cuda, lut, stride=0),
    )
    expect_error(
        ValueError,
        "padding must be non-negative",
        lambda: operation(input_cuda, weight_cuda, lut, padding=-1),
    )
    expect_error(
        ValueError,
        "stride must be an int or a pair",
        lambda: operation(input_cuda, weight_cuda, lut, stride=(1, 1, 1)),
    )
    expect_error(
        ValueError,
        "calculated output spatial size",
        lambda: operation(input_cuda[:, :, :1, :1], weight_cuda, lut),
    )
    expect_error(
        TypeError,
        "bias must be a bool",
        lambda: module_type(4, 6, 3, lut.cpu(), bias=1),
    )
    expect_error(
        NotImplementedError,
        "padding_mode='zeros'",
        lambda: module_type(4, 6, 3, lut.cpu(), padding_mode="reflect"),
    )
    placed = module_type(4, 6, 3, lut.cpu(), device="cuda")
    if not placed.weight.is_cuda or not placed.lut.is_cuda:
        raise AssertionError("explicit module device did not move parameters and LUT")

    if torch.cuda.device_count() > 1:
        other = torch.device("cuda:1")
        output = operation(
            input_cuda.to(other),
            weight_cuda.to(other),
            lut.to(other),
        )
        if output.device != other:
            raise AssertionError("Conv2d did not preserve a non-current CUDA device")
        expect_error(
            ValueError,
            "same CUDA device",
            lambda: operation(input_cuda, weight_cuda.to(other), lut),
        )
        expect_error(
            ValueError,
            "must be on",
            lambda: operation(input_cuda, weight_cuda, lut.to(other)),
        )
    print(f"PASS {kind} shape/dtype/device/geometry/module contract checks")


def verify_current_stream(kind: str, lut: torch.Tensor) -> None:
    spec = SPECS[kind]
    operation = getattr(at.nn, f"conv2d_{kind}")
    input_cuda = torch.zeros((1, 4, 6, 6), dtype=spec.dtype, device="cuda")
    weight_cuda = torch.zeros((6, 4, 3, 3), dtype=spec.dtype, device="cuda")
    bias_cuda = torch.zeros((6,), dtype=spec.dtype, device="cuda")
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        torch.cuda._sleep(30_000_000)
        input_cuda.fill_(0.625)
        weight_cuda.fill_(-0.75)
        bias_cuda.fill_(0.25)
        stream_result = operation(
            input_cuda, weight_cuda, lut, bias_cuda, padding=1
        )
    stream.synchronize()
    expected = operation(
        input_cuda, weight_cuda, lut, bias_cuda, padding=1
    )
    assert_same_bits(stream_result, expected, f"{kind} current-stream mismatch")
    print(f"PASS {kind} unfold/BGEMM/bias obey the current CUDA stream")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lut-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    torch.manual_seed(20260824)
    for kind in ("fp16", "bf16"):
        exact_cuda = load_exact_lut(kind, "cuda", directory=args.lut_dir)
        exact_cpu = exact_cuda.cpu()
        verify_strict_forward(kind, exact_cpu, exact_cuda)
        verify_module_api(kind, exact_cpu)
        verify_ste_backward(kind, exact_cuda)
        verify_model_conversion(kind, exact_cuda)
        verify_empty_and_layout(kind, exact_cuda)
        verify_contract(kind, exact_cuda)
        verify_current_stream(kind, exact_cuda)
    print("ALL APPROXIMATE FP16/BF16 CONV2D CHECKS PASSED")


if __name__ == "__main__":
    main()
