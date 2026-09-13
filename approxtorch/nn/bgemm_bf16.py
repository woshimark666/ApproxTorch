"""Autograd wrapper for LUT-based BF16 BGEMM."""

import torch
from torch.autograd import Function

import approxtorch as at


__all__ = ["bgemm_bf16_ste"]


class _bgemm_bf16_base(Function):

    @staticmethod
    def forward(ctx, x, w, lut, optimized=True):
        # dx reads w and dw reads x. Save only the operands needed by the
        # requested gradients, so frozen weights do not retain activations.
        ctx.save_for_backward(
            x if ctx.needs_input_grad[1] else None,
            w if ctx.needs_input_grad[0] else None,
        )
        if optimized:
            return at.backend.ops.bgemm_bf16(x, w, lut)
        return at.backend.ops.bgemm_bf16_naive(x, w, lut)


class _bgemm_bf16_ste(_bgemm_bf16_base):
    """Use the exact-product BGEMM derivative as a straight-through estimator."""

    @staticmethod
    def backward(ctx, grad_output):
        x, w = ctx.saved_tensors
        need_x, need_w = ctx.needs_input_grad[:2]
        n, o, length = grad_output.shape
        k = w.shape[1] if need_x else x.shape[1]
        # Singleton/empty contractions have special einsum lowerings (some
        # are elementwise). Preserve their signed-zero/autocast behavior, and
        # the existing differentiable graph when higher derivatives are wanted.
        use_einsum = torch.is_grad_enabled() or n == 0 or min(o, length, k) <= 1
        grad_x = grad_w = None
        if need_x:
            if use_einsum:
                grad_x = torch.einsum("nol,ok->nkl", grad_output, w)
            elif n == 1:
                grad_x = torch.bmm(w.t().unsqueeze(0), grad_output)
            else:
                grad_x = torch.bmm(
                    grad_output.transpose(1, 2).reshape(1, n * length, o),
                    w.unsqueeze(0),
                ).view(n, length, k).transpose(1, 2)
        if need_w:
            if use_einsum:
                grad_w = torch.einsum("nol,nkl->ok", grad_output, x)
            elif n == 1:
                grad_w = torch.bmm(grad_output, x.transpose(1, 2)).squeeze(0)
            else:
                # Einsum contracts missing labels alphabetically: (l,n), not
                # (n,l). Keep its GEMM operands and order for BF16 bit parity.
                grad_w = torch.bmm(
                    grad_output.permute(1, 2, 0).reshape(1, o, length * n),
                    x.permute(2, 0, 1).reshape(1, length * n, k),
                ).view(o, k)
        return grad_x, grad_w, None, None


def bgemm_bf16_ste(x, w, lut, optimized=True):
    """Run approximate BF16 BGEMM with an STE backward pass.

    Args:
        x: Contiguous CUDA BF16 tensor with shape ``[N, K, L]``.
        w: Contiguous CUDA BF16 tensor with shape ``[O, K]``.
        lut: Contiguous CUDA uint32 LUT with shape ``[128, 128]``.
        optimized: Select the optimized kernel when true.
    """
    return _bgemm_bf16_ste.apply(x, w, lut, optimized)
