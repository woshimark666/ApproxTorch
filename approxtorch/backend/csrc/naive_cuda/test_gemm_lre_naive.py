"""Oracle checks for int8/uint8 naive GEMM LRE backward ops."""

import torch

import approxtorch as at


def to_index(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.long() + 128 if tensor.dtype == torch.int8 else tensor.long()


def oracle(grad_output, A, B, dx_lut, dw_lut):
    a_idx, b_idx = to_index(A), to_index(B)
    # [M,N,1] * [1,N,K] -> reduce N -> [M,K]
    grad_A = (grad_output[:, :, None] * dx_lut[b_idx].t()[None]).sum(dim=1)
    # [M,N,1] * [M,1,K] -> reduce M -> [N,K] -> [K,N]
    grad_B = (
        grad_output[:, :, None] * dw_lut[a_idx][:, None, :]
    ).sum(dim=0).t().contiguous()
    return grad_A, grad_B


def make_values(shape, dtype):
    low, high = (-128, 128) if dtype == torch.int8 else (0, 256)
    return torch.randint(low, high, shape, device="cuda", dtype=dtype)


def check(actual, expected, label):
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    print(f"PASS {label}")


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    torch.manual_seed(2)
    q = torch.arange(256, device="cuda", dtype=torch.float32)
    dx_lut = (q.square() * 0.0002 - q * 0.03 + 0.7).contiguous()
    dw_lut = (torch.sin(q * 0.023) + q * 0.002).contiguous()

    for dtype, suffix in ((torch.int8, "int8"), (torch.uint8, "uint8")):
        op = getattr(at.backend.ops, f"gemm_lre_backward_{suffix}_naive")
        for M, K, N in ((1, 1, 1), (3, 5, 7), (11, 17, 9)):
            A, B = make_values((M, K), dtype), make_values((K, N), dtype)
            grad_output = torch.randn(M, N, device="cuda")
            expected_A, expected_B = oracle(
                grad_output, A, B, dx_lut, dw_lut)
            actual_A, actual_B = op(grad_output, A, B, dx_lut, dw_lut)
            tag = f"{suffix} M{M} K{K} N{N}"
            check(actual_A, expected_A, f"grad_A {tag}")
            check(actual_B, expected_B, f"grad_B {tag}")

        A0, B0 = make_values((6, 8), dtype), make_values((8, 10), dtype)
        g0 = torch.randn(6, 10, device="cuda")
        A, B, grad_output = A0[::2, ::2], B0[::2, ::2], g0[::2, ::2]
        assert not A.is_contiguous() and not B.is_contiguous()
        expected_A, expected_B = oracle(grad_output, A, B, dx_lut, dw_lut)
        actual_A, actual_B = op(grad_output, A, B, dx_lut, dw_lut)
        check(actual_A, expected_A, f"grad_A {suffix} non-contiguous")
        check(actual_B, expected_B, f"grad_B {suffix} non-contiguous")

    print("All naive GEMM LRE backward checks passed")


if __name__ == "__main__":
    main()
