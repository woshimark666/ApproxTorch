"""Oracle checks for int8/uint8 naive BGEMM custom-gradient ops.

Run after rebuilding the extension:
    python approxtorch/backend/csrc/naive_cuda/test_bgemm_custom_grad_naive.py
"""

import torch

import approxtorch as at


def to_index(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype == torch.int8:
        return tensor.long() + 128
    return tensor.long()


def oracle(
        X: torch.Tensor,
        W: torch.Tensor,
        dY: torch.Tensor,
        dx_lut: torch.Tensor,
        dw_lut: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x_idx = to_index(X)
    w_idx = to_index(W)
    # Both derivative images are [N,O,K,L], indexed as LUT[X][W].
    pair_index = x_idx[:, None, :, :] * 256 + w_idx[None, :, :, None]
    upstream = dY[:, :, None, :]
    grad_X = (upstream * dx_lut[pair_index]).sum(dim=1)
    grad_W = (upstream * dw_lut[pair_index]).sum(dim=(0, 3))
    return grad_X, grad_W


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
    torch.manual_seed(1)

    # Non-symmetric tables catch X/W operand reversal and signed-index errors.
    row = torch.arange(256, device="cuda", dtype=torch.float32)[:, None]
    col = torch.arange(256, device="cuda", dtype=torch.float32)[None, :]
    dx_lut = (torch.sin(row * 0.019) + col * 0.0031).contiguous().view(-1)
    dw_lut = (row * 0.0023 - torch.cos(col * 0.027)).contiguous().view(-1)

    for dtype, suffix in ((torch.int8, "int8"), (torch.uint8, "uint8")):
        naive = getattr(at.backend.ops, f"bgemm_custom_grad_{suffix}_naive")
        fast_dx = getattr(at.backend.ops, f"bgemm_custom_grad_{suffix}_dx")
        fast_dw = getattr(at.backend.ops, f"bgemm_custom_grad_{suffix}_dw")

        for N, K, L, O in ((1, 1, 1, 1), (2, 5, 7, 3), (3, 17, 9, 11)):
            X = make_values((N, K, L), dtype)
            W = make_values((O, K), dtype)
            dY = torch.randn(N, O, L, device="cuda")
            expected_X, expected_W = oracle(X, W, dY, dx_lut, dw_lut)
            actual_X, actual_W = naive(X, W, dY, dx_lut, dw_lut)
            tag = f"{suffix} N{N} K{K} L{L} O{O}"
            check_close(actual_X, expected_X, f"grad_X oracle {tag}")
            check_close(actual_W, expected_W, f"grad_W oracle {tag}")
            check_close(actual_X, fast_dx(X, W, dY, dx_lut),
                        f"grad_X optimized {tag}")
            check_close(actual_W, fast_dw(X, W, dY, dw_lut),
                        f"grad_W optimized {tag}")

        X_base = make_values((2, 4, 10), dtype)
        W_base = make_values((6, 4), dtype)
        dY_base = torch.randn(2, 6, 10, device="cuda")
        X = X_base[:, :, ::2]
        W = W_base[::2, :]
        dY = dY_base[:, ::2, ::2]
        assert not X.is_contiguous() and not W.is_contiguous()
        expected_X, expected_W = oracle(X, W, dY, dx_lut, dw_lut)
        actual_X, actual_W = naive(X, W, dY, dx_lut, dw_lut)
        check_close(actual_X, expected_X, f"grad_X {suffix} non-contiguous")
        check_close(actual_W, expected_W, f"grad_W {suffix} non-contiguous")

    print("All naive BGEMM custom-gradient checks passed")


if __name__ == "__main__":
    main()
