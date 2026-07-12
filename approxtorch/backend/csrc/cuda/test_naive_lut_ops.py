"""Exact checks for the deliberately naive LUT GEMM/BGEMM CUDA ops.

Run after rebuilding the extension:
    python approxtorch/backend/csrc/cuda/test_naive_lut_ops.py

The asymmetric LUT is intentional: it detects accidental reversal of the two
LUT operands.  Both GEMM and BGEMM obey the invariant that the first API
operand is the LUT row operand: LUT[A][B].
"""

import torch

import approxtorch as at


def to_index(x: torch.Tensor) -> torch.Tensor:
    if x.dtype == torch.int8:
        return x.long() + 128
    return x.long()


def gemm_oracle(a: torch.Tensor, b: torch.Tensor, lut: torch.Tensor) -> torch.Tensor:
    ai = to_index(a)
    bi = to_index(b)
    values = lut.view(256, 256).long()[ai[:, :, None], bi[None, :, :]]
    return values.sum(dim=1).to(torch.int32)


def bgemm_oracle(a: torch.Tensor, b: torch.Tensor, lut: torch.Tensor) -> torch.Tensor:
    ai = to_index(a)
    bi = to_index(b)
    values = lut.view(256, 256).long()[ai[:, None, :, :],
                                      bi[None, :, :, None]]
    return values.sum(dim=2).to(torch.int32)


def check_equal(actual: torch.Tensor, expected: torch.Tensor, label: str) -> None:
    if not torch.equal(actual, expected):
        delta = (actual.long() - expected.long()).abs().max().item()
        raise AssertionError(f"{label}: mismatch, max_abs_diff={delta}")
    print(f"PASS {label}")


def make_input(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    if dtype == torch.int8:
        return torch.randint(-128, 128, shape, device="cuda", dtype=dtype)
    return torch.randint(0, 256, shape, device="cuda", dtype=dtype)


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    torch.manual_seed(0)
    # Small values keep all reference sums safely inside int32.  Row and
    # column coefficients make lut[row, col] != lut[col, row].
    rows = torch.arange(256, device="cuda", dtype=torch.int32)[:, None]
    cols = torch.arange(256, device="cuda", dtype=torch.int32)[None, :]
    lut = (37 * rows - 19 * cols + 11).contiguous().view(-1)

    for dtype, suffix in ((torch.int8, "int8"), (torch.uint8, "uint8")):
        gemm_naive = getattr(at.backend.ops, f"gemm_{suffix}_naive")
        gemm_fast = getattr(at.backend.ops, f"gemm_{suffix}")
        bgemm_naive = getattr(at.backend.ops, f"bgemm_{suffix}_naive")
        bgemm_modern = getattr(at.backend.ops, f"bgemm_fake_{suffix}_claude")

        for M, K, N in ((1, 1, 1), (3, 5, 7), (17, 33, 9)):
            a = make_input((M, K), dtype)
            b = make_input((K, N), dtype)
            expected = gemm_oracle(a, b, lut)
            actual = gemm_naive(a, b, lut)
            check_equal(actual, expected, f"gemm_{suffix}_naive M{M} K{K} N{N}")
            check_equal(gemm_fast(a, b, lut), actual,
                        f"gemm_{suffix} vs naive M{M} K{K} N{N}")

        for batch, K, L, O in ((1, 1, 1, 1), (2, 5, 7, 3), (3, 33, 9, 17)):
            a = make_input((batch, K, L), dtype)
            b = make_input((O, K), dtype)
            expected = bgemm_oracle(a, b, lut)
            actual = bgemm_naive(a, b, lut)
            check_equal(
                actual, expected,
                f"bgemm_{suffix}_naive B{batch} K{K} L{L} O{O}")
            # nn/bgemm_int8.py uses this modern family.  Its fp32-storage
            # result must have the same LUT[A][B] semantics as the naive op.
            modern = bgemm_modern(a.float(), b.float(), lut.float())
            check_equal(
                modern.to(torch.int32), actual,
                f"bgemm_fake_{suffix}_claude vs naive "
                f"B{batch} K{K} L{L} O{O}")

        # The wrappers normalize strided inputs with contiguous().
        a = make_input((4, 6), dtype)[:, ::2]
        b = make_input((6, 10), dtype)[::2, ::2]
        assert not a.is_contiguous() and not b.is_contiguous()
        check_equal(gemm_naive(a, b, lut), gemm_oracle(a, b, lut),
                    f"gemm_{suffix}_naive non-contiguous")

    print("All naive LUT GEMM/BGEMM checks passed")


if __name__ == "__main__":
    main()
