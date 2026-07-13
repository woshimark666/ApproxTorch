"""Exact oracle checks for int8/uint8 naive BGEMM LRE backward ops.

Run after rebuilding the extension:
    python approxtorch/backend/csrc/naive_cuda/test_bgemm_lre_naive.py
"""

import torch

import approxtorch as at


def to_index(x: torch.Tensor) -> torch.Tensor:
    if x.dtype == torch.int8:
        return x.long() + 128
    return x.long()


def oracle(
        grad_output: torch.Tensor,
        x: torch.Tensor,
        w: torch.Tensor,
        dx_lut: torch.Tensor,
        dw_lut: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x_idx = to_index(x)
    w_idx = to_index(w)
    # [N,O,L,None] * [1,O,1,K] -> sum O -> [N,L,K] -> [N,K,L]
    grad_x = (
        grad_output[:, :, :, None] * dx_lut[w_idx][None, :, None, :]
    ).sum(dim=1).permute(0, 2, 1).contiguous()
    # [N,O,L,None] * [N,1,L,K] -> sum N,L -> [O,K]
    x_deriv = dw_lut[x_idx].permute(0, 2, 1)
    grad_w = (
        grad_output[:, :, :, None] * x_deriv[:, None, :, :]
    ).sum(dim=(0, 2))
    return grad_x, grad_w


def make_values(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    if dtype == torch.int8:
        return torch.randint(-128, 128, shape, device="cuda", dtype=dtype)
    return torch.randint(0, 256, shape, device="cuda", dtype=dtype)


def check_close(actual: torch.Tensor, expected: torch.Tensor, label: str) -> None:
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    print(f"PASS {label}")


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    torch.manual_seed(0)

    # Distinct nonlinear tables expose wrong LUT choice and wrong signed index.
    q = torch.arange(256, device="cuda", dtype=torch.float32)
    dx_lut = (q.square() * 0.0003 - q * 0.07 + 1.25).contiguous()
    dw_lut = (torch.sin(q * 0.031) * 2.5 + q * 0.004).contiguous()

    for dtype, suffix in ((torch.int8, "int8"), (torch.uint8, "uint8")):
        op = getattr(at.backend.ops, f"bgemm_lre_backward_{suffix}_naive")
        for N, K, L, O in ((1, 1, 1, 1), (2, 5, 7, 3), (3, 17, 9, 11)):
            x = make_values((N, K, L), dtype)
            w = make_values((O, K), dtype)
            grad_output = torch.randn(N, O, L, device="cuda")
            expected_x, expected_w = oracle(
                grad_output, x, w, dx_lut, dw_lut)
            actual_x, actual_w = op(
                grad_output, x, w, dx_lut, dw_lut)
            tag = f"{suffix} N{N} K{K} L{L} O{O}"
            check_close(actual_x, expected_x, f"grad_x {tag}")
            check_close(actual_w, expected_w, f"grad_w {tag}")
            if dtype == torch.int8:
                # The existing reference accepts fp32 integer images and
                # transposed weights [K,O].  Normalize its grad_w back to
                # the new naive API's [O,K] layout before comparison.
                ref_x, ref_w_t = at.backend.ops.bgemm_lre_backward(
                    grad_output, x.float(), w.t().contiguous().float(),
                    dx_lut, dw_lut)
                check_close(actual_x, ref_x, f"grad_x existing-ref {tag}")
                check_close(actual_w, ref_w_t.t(), f"grad_w existing-ref {tag}")

        # Verify contiguous normalization for all strided model tensors.
        x_base = make_values((2, 4, 10), dtype)
        w_base = make_values((6, 4), dtype)
        gy_base = torch.randn(2, 6, 10, device="cuda")
        x = x_base[:, :, ::2]
        w = w_base[::2, :]
        grad_output = gy_base[:, ::2, ::2]
        assert not x.is_contiguous() and not w.is_contiguous()
        expected_x, expected_w = oracle(
            grad_output, x, w, dx_lut, dw_lut)
        actual_x, actual_w = op(
            grad_output, x, w, dx_lut, dw_lut)
        check_close(actual_x, expected_x, f"grad_x {suffix} non-contiguous")
        check_close(actual_w, expected_w, f"grad_w {suffix} non-contiguous")

    print("All naive BGEMM LRE backward checks passed")


if __name__ == "__main__":
    main()
