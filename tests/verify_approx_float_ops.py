#!/usr/bin/env python3
"""Bit-level correctness checks for FP16/BF16 LUT GEMM and BGEMM."""

from __future__ import annotations

import argparse
from pathlib import Path
import random

import torch

import approxtorch as at
from approxtorch.float_lut import load_exact_lut, validate_mantissa_lut
from approx_float_reference import (
    SPECS,
    add_bits,
    strict_bgemm_reference,
    strict_gemm_reference,
    tensor_from_bits,
    unsigned_bits,
)


def boundary_bits(kind: str) -> list[int]:
    if kind == "fp16":
        return [
            0x0000, 0x8000, 0x0001, 0x8001, 0x0002, 0x03FF, 0x83FF,
            0x0400, 0x8400, 0x3555, 0x3BFF, 0x3C00, 0x3C01, 0x4000,
            0x7BFE, 0x7BFF, 0xFBFF, 0x7C00, 0xFC00, 0x7E00, 0xFE11,
            0x7C01, 0xFC21,
        ]
    return [
        0x0000, 0x8000, 0x0001, 0x8001, 0x0002, 0x007F, 0x807F,
        0x0080, 0x8080, 0x3EAB, 0x3F7F, 0x3F80, 0x3F81, 0x4000,
        0x7F7E, 0x7F7F, 0xFF7F, 0x7F80, 0xFF80, 0x7FC0, 0xFFC5,
        0x7F81, 0xFF91,
    ]


def assert_result_bits(actual: torch.Tensor, expected: torch.Tensor, kind: str) -> None:
    spec = SPECS[kind]
    actual_bits = unsigned_bits(actual.cpu())
    expected_bits = unsigned_bits(expected.cpu())
    actual_nan = (
        (actual_bits & spec.exponent_mask) == spec.exponent_mask
    ) & ((actual_bits & spec.fraction_mask) != 0)
    expected_nan = (
        (expected_bits & spec.exponent_mask) == spec.exponent_mask
    ) & ((expected_bits & spec.fraction_mask) != 0)
    if not torch.equal(actual_nan, expected_nan):
        raise AssertionError("NaN classification mismatch")
    finite_or_inf = ~expected_nan
    if not torch.equal(actual_bits[finite_or_inf], expected_bits[finite_or_inf]):
        mismatch = actual_bits[finite_or_inf] != expected_bits[finite_or_inf]
        raise AssertionError(
            f"bit mismatch for {int(mismatch.sum())} non-NaN outputs"
        )
    if actual_nan.any() and not torch.all(
        (actual_bits[actual_nan] & spec.quiet_nan_bit) != 0
    ):
        raise AssertionError("NaN result was not quieted")


def rtl_multiply_reference_bits(
    lhs_bits: torch.Tensor,
    rhs_bits: torch.Tensor,
    lut_cpu: torch.Tensor,
    kind: str,
) -> torch.Tensor:
    spec = SPECS[kind]
    lhs = lhs_bits.to(torch.int64) & 0xFFFF
    rhs = rhs_bits.to(torch.int64) & 0xFFFF
    entries = lut_cpu.to(torch.int64)[
        lhs & spec.fraction_mask,
        rhs & spec.fraction_mask,
    ]
    normalization = (entries >> spec.fraction_bits) & 1
    exponent = (
        ((lhs >> spec.fraction_bits) & spec.exponent_field_mask)
        + ((rhs >> spec.fraction_bits) & spec.exponent_field_mask)
        - spec.bias
        + normalization
    ) & spec.exponent_field_mask
    result = (
        ((lhs ^ rhs) & spec.sign_mask)
        | (exponent << spec.fraction_bits)
        | (entries & spec.fraction_mask)
    )
    nonzero = ((lhs & 0x7FFF) != 0) & ((rhs & 0x7FFF) != 0)
    return torch.where(nonzero, result, 0).to(torch.int32)


def verify_exhaustive_multiply(kind: str, lut: torch.Tensor) -> None:
    spec = SPECS[kind]
    operation = getattr(at.backend.ops, f"approx_mul_{kind}")
    lut_cpu = lut.cpu()
    all_bits = torch.arange(65536, dtype=torch.int64)
    all_values = tensor_from_bits(range(65536), (65536,), spec.dtype).cuda()
    for operand in boundary_bits(kind):
        scalar = tensor_from_bits([operand], (1,), spec.dtype).cuda()
        repeated = scalar.expand_as(all_values).contiguous()
        repeated_bits = torch.full_like(all_bits, operand)
        for lhs, rhs, lhs_bits, rhs_bits in (
            (all_values, repeated, all_bits, repeated_bits),
            (repeated, all_values, repeated_bits, all_bits),
        ):
            actual_bits = unsigned_bits(operation(lhs, rhs, lut).cpu())
            expected_bits = rtl_multiply_reference_bits(
                lhs_bits, rhs_bits, lut_cpu, kind
            )
            if not torch.equal(actual_bits, expected_bits):
                raise AssertionError(f"{kind} exhaustive multiply mismatch")

    generator = torch.Generator(device="cpu").manual_seed(20260824)
    lhs_bits = torch.randint(
        0, 65536, (1_000_000,), generator=generator, dtype=torch.int32
    )
    rhs_bits = torch.randint(
        0, 65536, (1_000_000,), generator=generator, dtype=torch.int32
    )
    lhs = lhs_bits.to(torch.int16).view(spec.dtype).cuda()
    rhs = rhs_bits.to(torch.int16).view(spec.dtype).cuda()
    actual_bits = unsigned_bits(operation(lhs, rhs, lut).cpu())
    expected_bits = rtl_multiply_reference_bits(
        lhs_bits, rhs_bits, lut_cpu, kind
    )
    if not torch.equal(actual_bits, expected_bits):
        raise AssertionError(f"{kind} random multiply mismatch")
    print(f"PASS {kind} exhaustive input-pattern x boundary + 1M random multiply")

def verify_add_reference(kind: str) -> None:
    spec = SPECS[kind]
    rng = random.Random(9137)
    pairs = [(rng.randrange(65536), rng.randrange(65536)) for _ in range(100_000)]
    boundaries = boundary_bits(kind)
    pairs.extend((lhs, rhs) for lhs in boundaries for rhs in boundaries)
    lhs = tensor_from_bits([item[0] for item in pairs], (len(pairs),), spec.dtype)
    rhs = tensor_from_bits([item[1] for item in pairs], (len(pairs),), spec.dtype)
    reference = tensor_from_bits(
        [add_bits(item[0], item[1], kind) for item in pairs],
        (len(pairs),),
        spec.dtype,
    )
    native = lhs.cuda() + rhs.cuda()
    assert_result_bits(native, reference, kind)
    print(f"PASS {kind} pure-integer RNE add reference (100k random + boundaries)")


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
    return values.to(torch.uint32).contiguous()


def random_bit_tensor(shape: tuple[int, ...], dtype: torch.dtype, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    storage = torch.randint(0, 65536, shape, generator=generator,
                            dtype=torch.int32).to(torch.int16)
    return storage.view(dtype)


def verify_strict_references(
    kind: str, exact_cpu: torch.Tensor, exact_cuda: torch.Tensor
) -> None:
    spec = SPECS[kind]
    gemm_naive = getattr(at.backend.ops, f"gemm_{kind}_naive")
    gemm_fast = getattr(at.backend.ops, f"gemm_{kind}")
    bgemm_naive = getattr(at.backend.ops, f"bgemm_{kind}_naive")
    bgemm_fast = getattr(at.backend.ops, f"bgemm_{kind}")

    approximate_cpu = make_asymmetric_lut(kind)
    validate_mantissa_lut(approximate_cpu, kind, require_cuda=False)
    approximate_cuda = approximate_cpu.cuda()
    for lut_cpu, lut_cuda, label in (
        (exact_cpu, exact_cuda, "exact"),
        (approximate_cpu, approximate_cuda, "asymmetric"),
    ):
        A = random_bit_tensor((4, 7), spec.dtype, 101)
        B = random_bit_tensor((7, 5), spec.dtype, 102)
        expected = strict_gemm_reference(A, B, lut_cpu, kind)
        naive = gemm_naive(A.cuda(), B.cuda(), lut_cuda)
        optimized = gemm_fast(A.cuda(), B.cuda(), lut_cuda)
        assert_result_bits(naive, expected, kind)
        if not torch.equal(unsigned_bits(optimized.cpu()), unsigned_bits(naive.cpu())):
            raise AssertionError(f"{kind} {label} optimized GEMM differs from naive")

        X = random_bit_tensor((2, 7, 6), spec.dtype, 103)
        W = random_bit_tensor((5, 7), spec.dtype, 104)
        expected_b = strict_bgemm_reference(X, W, lut_cpu, kind)
        naive_b = bgemm_naive(X.cuda(), W.cuda(), lut_cuda)
        optimized_b = bgemm_fast(X.cuda(), W.cuda(), lut_cuda)
        assert_result_bits(naive_b, expected_b, kind)
        if not torch.equal(
            unsigned_bits(optimized_b.cpu()), unsigned_bits(naive_b.cpu())
        ):
            raise AssertionError(f"{kind} {label} optimized BGEMM differs from naive")

    # A deliberately non-symmetric LUT must make operand reversal observable.
    values_a = torch.linspace(0.75, 1.75, 4096, dtype=spec.dtype, device="cuda")
    values_b = torch.linspace(1.875, 0.625, 4096, dtype=spec.dtype, device="cuda")
    mul = getattr(at.backend.ops, f"approx_mul_{kind}")
    forward = unsigned_bits(mul(values_a, values_b, approximate_cuda).cpu())
    reversed_ = unsigned_bits(mul(values_b, values_a, approximate_cuda).cpu())
    if torch.equal(forward, reversed_):
        raise AssertionError(f"{kind} asymmetric LUT did not expose operand order")
    print(f"PASS {kind} independent strict GEMM/BGEMM references + LUT[A][B]")


def verify_fp32_accumulator(kind: str, exact_lut: torch.Tensor) -> None:
    """Distinguish FP32 accumulation from rounding after every 16-bit add."""
    spec = SPECS[kind]
    K = 2050 if kind == "fp16" else 258
    expected = torch.tensor(float(K), dtype=spec.dtype)
    expected_bits = int(unsigned_bits(expected).item())

    one_bits = int(unsigned_bits(torch.tensor(1.0, dtype=spec.dtype)).item())
    legacy_bits = 0
    for _ in range(K):
        legacy_bits = add_bits(legacy_bits, one_bits, kind)
    if legacy_bits == expected_bits:
        raise AssertionError("FP32 accumulator regression case is not discriminating")

    A = torch.ones((1, K), dtype=spec.dtype, device="cuda")
    B = torch.ones((K, 1), dtype=spec.dtype, device="cuda")
    for suffix in ("_naive", ""):
        operation = getattr(at.backend.ops, f"gemm_{kind}{suffix}")
        actual = unsigned_bits(operation(A, B, exact_lut).cpu())
        if not torch.all(actual == expected_bits):
            raise AssertionError(
                f"{kind} GEMM does not accumulate LUT products in FP32"
            )

    L = 8 if kind == "fp16" else 192
    X = torch.ones((1, K, L), dtype=spec.dtype, device="cuda")
    W = torch.ones((16, K), dtype=spec.dtype, device="cuda")
    for suffix in ("_naive", ""):
        operation = getattr(at.backend.ops, f"bgemm_{kind}{suffix}")
        actual = unsigned_bits(operation(X, W, exact_lut).cpu())
        if not torch.all(actual == expected_bits):
            raise AssertionError(
                f"{kind} BGEMM does not accumulate LUT products in FP32"
            )
    print(
        f"PASS {kind} FP32 accumulator with one final {kind} conversion"
    )


def verify_shape_matrix(kind: str, exact_lut: torch.Tensor) -> None:
    spec = SPECS[kind]
    approximate = make_asymmetric_lut(kind).cuda()
    gemm_naive = getattr(at.backend.ops, f"gemm_{kind}_naive")
    gemm_fast = getattr(at.backend.ops, f"gemm_{kind}")
    bgemm_naive = getattr(at.backend.ops, f"bgemm_{kind}_naive")
    bgemm_fast = getattr(at.backend.ops, f"bgemm_{kind}")
    for lut, lut_label in ((exact_lut, "exact"), (approximate, "asymmetric")):
        for index, (M, K, N) in enumerate(
            ((1, 1, 1), (3, 5, 7), (31, 19, 63), (33, 17, 65),
             (65, 33, 17), (3, 17, 257), (3, 17, 769),
             (0, 9, 7), (5, 9, 0), (4, 0, 6), (3, 0, 769))
        ):
            A = random_bit_tensor((M, K), spec.dtype, 1000 + index).cuda()
            B = random_bit_tensor((K, N), spec.dtype, 1100 + index).cuda()
            naive = gemm_naive(A, B, lut)
            optimized = gemm_fast(A, B, lut)
            if not torch.equal(unsigned_bits(optimized.cpu()), unsigned_bits(naive.cpu())):
                raise AssertionError(
                    f"{kind} {lut_label} GEMM mismatch M={M} K={K} N={N}"
                )
            if K == 0 and naive.numel() and torch.count_nonzero(unsigned_bits(naive.cpu())):
                raise AssertionError("K=0 GEMM did not return +0")

        for index, (batch, K, L, O) in enumerate(
            ((1, 1, 1, 1), (2, 5, 7, 3), (2, 17, 65, 33),
             (3, 33, 63, 31), (2, 17, 197, 65), (1, 19, 517, 65),
             (0, 9, 7, 5), (2, 9, 0, 5), (2, 0, 7, 5),
             (1, 0, 197, 65), (1, 0, 517, 65), (2, 9, 7, 0))
        ):
            X = random_bit_tensor((batch, K, L), spec.dtype, 1200 + index).cuda()
            W = random_bit_tensor((O, K), spec.dtype, 1300 + index).cuda()
            naive = bgemm_naive(X, W, lut)
            optimized = bgemm_fast(X, W, lut)
            if not torch.equal(unsigned_bits(optimized.cpu()), unsigned_bits(naive.cpu())):
                raise AssertionError(
                    f"{kind} {lut_label} BGEMM mismatch "
                    f"batch={batch} K={K} L={L} O={O}"
                )
            if K == 0 and naive.numel() and torch.count_nonzero(unsigned_bits(naive.cpu())):
                raise AssertionError("K=0 BGEMM did not return +0")
    print(f"PASS {kind} exact/asymmetric shapes, empty tensors, tile boundaries")


def verify_contract_checks(kind: str, lut: torch.Tensor) -> None:
    spec = SPECS[kind]
    gemm = getattr(at.backend.ops, f"gemm_{kind}")
    A = torch.ones((4, 6), dtype=spec.dtype, device="cuda")
    B = torch.ones((6, 5), dtype=spec.dtype, device="cuda")
    try:
        gemm(A[:, ::2], B[::2], lut)
    except RuntimeError as error:
        if "contiguous" not in str(error):
            raise
    else:
        raise AssertionError("non-contiguous GEMM input was accepted")
    try:
        gemm(A, B, lut.reshape(-1))
    except RuntimeError as error:
        if "shape" not in str(error):
            raise
    else:
        raise AssertionError("flat LUT was accepted")

    wrong_dtype = torch.zeros(
        lut.shape, dtype=torch.uint16, device=lut.device
    )
    try:
        gemm(A, B, wrong_dtype)
    except RuntimeError as error:
        if "wrong dtype" not in str(error):
            raise
    else:
        raise AssertionError("uint16 LUT was accepted by the CUDA backend")

    if torch.cuda.device_count() > 1:
        output = gemm(A.to("cuda:1"), B.to("cuda:1"), lut.to("cuda:1"))
        assert output.device.index == 1
        try:
            gemm(A, B, lut.to("cuda:1"))
        except RuntimeError as error:
            if "same CUDA device" not in str(error):
                raise
        else:
            raise AssertionError("cross-device LUT was accepted")
    print(f"PASS {kind} dtype/shape/contiguous/device contract checks")


def verify_current_stream(kind: str, lut: torch.Tensor) -> None:
    """Make a wrong default-stream launch observe zeros instead of inputs."""
    spec = SPECS[kind]
    gemm_naive = getattr(at.backend.ops, f"gemm_{kind}_naive")
    gemm_fast = getattr(at.backend.ops, f"gemm_{kind}")
    bgemm_naive = getattr(at.backend.ops, f"bgemm_{kind}_naive")
    bgemm_fast = getattr(at.backend.ops, f"bgemm_{kind}")

    A = torch.zeros((3, 17), dtype=spec.dtype, device="cuda")
    B = torch.zeros((17, 269), dtype=spec.dtype, device="cuda")
    X = torch.zeros((1, 17, 197), dtype=spec.dtype, device="cuda")
    W = torch.zeros((33, 17), dtype=spec.dtype, device="cuda")
    torch.cuda.synchronize()

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        # Delay the fills so an accidental default-stream kernel reliably
        # consumes the initialized zeros. Correct kernels queue after fills.
        torch.cuda._sleep(30_000_000)
        A.fill_(1.25)
        B.fill_(-0.75)
        X.fill_(0.625)
        W.fill_(1.5)
        stream_gemm_naive = gemm_naive(A, B, lut)
        stream_gemm_fast = gemm_fast(A, B, lut)
        stream_bgemm_naive = bgemm_naive(X, W, lut)
        stream_bgemm_fast = bgemm_fast(X, W, lut)
    stream.synchronize()
    torch.cuda.synchronize()

    expected_gemm = gemm_fast(A, B, lut)
    expected_bgemm = bgemm_fast(X, W, lut)
    for actual in (stream_gemm_naive, stream_gemm_fast):
        if not torch.equal(unsigned_bits(actual.cpu()), unsigned_bits(expected_gemm.cpu())):
            raise AssertionError(f"{kind} GEMM did not use the current CUDA stream")
    for actual in (stream_bgemm_naive, stream_bgemm_fast):
        if not torch.equal(unsigned_bits(actual.cpu()), unsigned_bits(expected_bgemm.cpu())):
            raise AssertionError(f"{kind} BGEMM did not use the current CUDA stream")
    print(f"PASS {kind} PyTorch current CUDA stream ordering")


def finite_tensor(
    shape: tuple[int, ...], dtype: torch.dtype, seed: int
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return (torch.randn(shape, generator=generator) * 0.625).to(dtype)


def verify_linear_module(kind: str, lut: torch.Tensor) -> None:
    spec = SPECS[kind]
    module_type = getattr(at.nn, f"Linear_{kind}")
    functional = getattr(at.nn, f"linear_{kind}")

    input_cpu = finite_tensor((2, 3, 7), spec.dtype, 2101)
    weight_cpu = finite_tensor((5, 7), spec.dtype, 2102)
    bias_cpu = finite_tensor((5,), spec.dtype, 2103)
    input_cuda = input_cpu.cuda()
    weight_cuda = weight_cpu.cuda()
    bias_cuda = bias_cpu.cuda()

    naive = functional(
        input_cuda, weight_cuda, lut, bias_cuda, optimized=False
    )
    optimized = functional(
        input_cuda, weight_cuda, lut, bias_cuda, optimized=True
    )
    if not torch.equal(unsigned_bits(optimized.cpu()), unsigned_bits(naive.cpu())):
        raise AssertionError(f"{kind} Linear optimized and naive paths differ")

    expected = strict_gemm_reference(
        input_cpu.reshape(-1, 7),
        weight_cpu.transpose(0, 1).contiguous(),
        lut.cpu(),
        kind,
    ).reshape(2, 3, 5)
    expected = (expected.cuda() + bias_cuda).cpu()
    assert_result_bits(optimized, expected, kind)

    module = module_type(
        7,
        5,
        lut,
        bias=bias_cuda,
        optimized=True,
    )
    with torch.no_grad():
        module.weight.copy_(weight_cuda)
    if not isinstance(module.weight, torch.nn.Parameter):
        raise AssertionError("Linear weight is not an nn.Parameter")
    if not isinstance(module.bias, torch.nn.Parameter):
        raise AssertionError("Linear bias is not an nn.Parameter")
    if module.weight.dtype != spec.dtype or module.bias.dtype != spec.dtype:
        raise AssertionError("Linear parameters do not use the selected dtype")
    if module.lut.dtype != torch.uint32:
        raise AssertionError("Linear LUT is not uint32")
    if f"dtype={spec.dtype}" not in repr(module):
        raise AssertionError("Linear repr omits its explicit dtype")
    assert_result_bits(module(input_cuda), expected, kind)

    input_actual = finite_tensor((2, 3, 7), spec.dtype, 2201).cuda()
    input_actual.requires_grad_(True)
    grad_output = finite_tensor((2, 3, 5), spec.dtype, 2202).cuda()
    module.zero_grad(set_to_none=True)
    module(input_actual).backward(grad_output)

    input_expected = input_actual.detach().clone().requires_grad_(True)
    weight_expected = module.weight.detach().clone().requires_grad_(True)
    bias_expected = module.bias.detach().clone().requires_grad_(True)
    torch.nn.functional.linear(
        input_expected, weight_expected, bias_expected
    ).backward(grad_output)
    torch.testing.assert_close(input_actual.grad, input_expected.grad)
    torch.testing.assert_close(module.weight.grad, weight_expected.grad)
    torch.testing.assert_close(module.bias.grad, bias_expected.grad)
    print(f"PASS {kind} Linear functional/module forward and STE backward")



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
        verify_add_reference(kind)
        exact = load_exact_lut(kind, "cuda:0", directory=args.lut_dir)
        exact_cpu = exact.cpu()
        verify_exhaustive_multiply(kind, exact)
        verify_strict_references(kind, exact_cpu, exact)
        verify_fp32_accumulator(kind, exact)
        verify_shape_matrix(kind, exact)
        verify_contract_checks(kind, exact)
        verify_linear_module(kind, exact)
        verify_current_stream(kind, exact)
    print("All approximate FP16/BF16 correctness checks passed")


if __name__ == "__main__":
    main()
