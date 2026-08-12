import torch
from torch import Tensor
from torch.nn.modules.utils import _pair


__all__ = ['im2col_int8', 
           'im2col_uint8', 
           'gemm_int8', 
           'gemm_uint8',
           'gemm_int8_naive',
           'gemm_uint8_naive',
           'gemm_lre_backward_int8_naive',
           'gemm_lre_backward_uint8_naive',
           'gemm_custom_grad_int8_naive',
           'gemm_custom_grad_uint8_naive',
           'bgemm_int8', 
           'bgemm_uint8',
           'bgemm_int8_naive',
           'bgemm_uint8_naive',
           'bgemm_lre_backward_int8_naive',
           'bgemm_lre_backward_uint8_naive',
           'bgemm_custom_grad_int8_naive',
           'bgemm_custom_grad_uint8_naive',
           'bgemm_custom_grad_int8_dx',
           'bgemm_custom_grad_int8_dw',
           'bgemm_custom_grad_uint8_dx',
           'bgemm_custom_grad_uint8_dw']

def im2col_int8(feature: Tensor, kernel_size, stride=1, padding=0, dilation=1) -> Tensor:
    kernel_size = _pair(kernel_size)
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)
    
    return torch.ops.approxtorch.im2col_int8.default(feature, 
                                    kernel_size[0], 
                                    kernel_size[1], 
                                    stride[0], 
                                    stride[1], 
                                    padding[0], 
                                    padding[1], 
                                    dilation[0], 
                                    dilation[1])


def im2col_uint8(feature: Tensor, kernel_size, stride=1, padding=0, dilation=1) -> Tensor:
    kernel_size = _pair(kernel_size)
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)
    
    return torch.ops.approxtorch.im2col_uint8.default(feature, 
                                    kernel_size[0], 
                                    kernel_size[1], 
                                    stride[0], 
                                    stride[1], 
                                    padding[0], 
                                    padding[1],
                                    dilation[0], 
                                    dilation[1])
    
def gemm_int8(A: Tensor, B: Tensor, lut: Tensor) -> Tensor:
    return torch.ops.approxtorch.gemm_int8.default(A, B, lut)

def gemm_uint8(A: Tensor, B: Tensor, lut: Tensor) -> Tensor:
    return torch.ops.approxtorch.gemm_uint8.default(A, B, lut)

def gemm_int8_naive(A: Tensor, B: Tensor, lut: Tensor) -> Tensor:
    return torch.ops.approxtorch.gemm_int8_naive.default(A, B, lut)

def gemm_uint8_naive(A: Tensor, B: Tensor, lut: Tensor) -> Tensor:
    return torch.ops.approxtorch.gemm_uint8_naive.default(A, B, lut)

def gemm_lre_backward_int8_naive(
        grad_output: Tensor, A: Tensor, B: Tensor,
        dx_lut: Tensor, dw_lut: Tensor) -> tuple[Tensor, Tensor]:
    return torch.ops.approxtorch.gemm_lre_backward_int8_naive.default(
        grad_output, A, B, dx_lut, dw_lut)

def gemm_lre_backward_uint8_naive(
        grad_output: Tensor, A: Tensor, B: Tensor,
        dx_lut: Tensor, dw_lut: Tensor) -> tuple[Tensor, Tensor]:
    return torch.ops.approxtorch.gemm_lre_backward_uint8_naive.default(
        grad_output, A, B, dx_lut, dw_lut)

def gemm_custom_grad_int8_naive(
        A: Tensor, B: Tensor, dY: Tensor,
        dx_lut: Tensor, dw_lut: Tensor) -> tuple[Tensor, Tensor]:
    return torch.ops.approxtorch.gemm_custom_grad_int8_naive.default(
        A, B, dY, dx_lut, dw_lut)

def gemm_custom_grad_uint8_naive(
        A: Tensor, B: Tensor, dY: Tensor,
        dx_lut: Tensor, dw_lut: Tensor) -> tuple[Tensor, Tensor]:
    return torch.ops.approxtorch.gemm_custom_grad_uint8_naive.default(
        A, B, dY, dx_lut, dw_lut)

def bgemm_int8(A: Tensor, B: Tensor, lut: Tensor) -> Tensor:
    return torch.ops.approxtorch.bgemm_int8.default(A, B, lut)

def bgemm_uint8(A: Tensor, B: Tensor, lut: Tensor) -> Tensor:
    return torch.ops.approxtorch.bgemm_uint8.default(A, B, lut)

def bgemm_int8_naive(A: Tensor, B: Tensor, lut: Tensor) -> Tensor:
    return torch.ops.approxtorch.bgemm_int8_naive.default(A, B, lut)

def bgemm_uint8_naive(A: Tensor, B: Tensor, lut: Tensor) -> Tensor:
    return torch.ops.approxtorch.bgemm_uint8_naive.default(A, B, lut)

def bgemm_lre_backward_int8_naive(
        grad_output: Tensor, x: Tensor, w: Tensor,
        dx_lut: Tensor, dw_lut: Tensor) -> tuple[Tensor, Tensor]:
    return torch.ops.approxtorch.bgemm_lre_backward_int8_naive.default(
        grad_output, x, w, dx_lut, dw_lut)

def bgemm_lre_backward_uint8_naive(
        grad_output: Tensor, x: Tensor, w: Tensor,
        dx_lut: Tensor, dw_lut: Tensor) -> tuple[Tensor, Tensor]:
    return torch.ops.approxtorch.bgemm_lre_backward_uint8_naive.default(
        grad_output, x, w, dx_lut, dw_lut)


def bgemm_custom_grad_uint8_naive(
        X: Tensor, W: Tensor, dY: Tensor,
        dx_lut: Tensor, dw_lut: Tensor) -> tuple[Tensor, Tensor]:
    return torch.ops.approxtorch.bgemm_custom_grad_uint8_naive.default(
        X, W, dY, dx_lut, dw_lut)

def bgemm_custom_grad_int8_naive(
        X: Tensor, W: Tensor, dY: Tensor,
        dx_lut: Tensor, dw_lut: Tensor) -> tuple[Tensor, Tensor]:
    return torch.ops.approxtorch.bgemm_custom_grad_int8_naive.default(
        X, W, dY, dx_lut, dw_lut)

# def bgemm_custom_grad_uint8(X: Tensor, W: Tensor, dY: Tensor, dx_lut: Tensor, dw_lut: Tensor) -> Tensor:
#     return torch.ops.approxtorch.bgemm_custom_grad_uint8.default(X, W, dY, dx_lut, dw_lut)


def bgemm_custom_grad_uint8_dx(X: Tensor, W: Tensor, dY: Tensor, dx_lut: Tensor) -> Tensor:
    return torch.ops.approxtorch.bgemm_custom_grad_uint8_dx.default(X, W, dY, dx_lut)

def bgemm_custom_grad_uint8_dw(X: Tensor, W: Tensor, dY: Tensor, dW_lut: Tensor) -> Tensor:
    return torch.ops.approxtorch.bgemm_custom_grad_uint8_dw.default(X, W, dY, dW_lut)


def bgemm_custom_grad_int8_dx(X: Tensor, W: Tensor, dY: Tensor, dx_lut: Tensor) -> Tensor:
    return torch.ops.approxtorch.bgemm_custom_grad_int8_dx.default(X, W, dY, dx_lut)

def bgemm_custom_grad_int8_dw(X: Tensor, W: Tensor, dY: Tensor, dW_lut: Tensor) -> Tensor:
    return torch.ops.approxtorch.bgemm_custom_grad_int8_dw.default(X, W, dY, dW_lut)


def lut_lookup_int8(x: Tensor, lut: Tensor) -> Tensor:
    return torch.ops.approxtorch.lut_lookup_int8.default(x, lut)


# def bgemm_gradual_int8(X: Tensor, W: Tensor, lut: Tensor, alpha: float) -> Tensor:
#     return torch.ops.approxtorch.bgemm_gradual_int8.default(X, W, lut, alpha)


def bgemm_fake_int8_gpt(X: Tensor, W: Tensor, lut: Tensor) -> Tensor:
    return torch.ops.approxtorch.bgemm_fake_int8_forward_cuda.default(X, W, lut)


# claude-optimized forward, bit-identical to bgemm_fake_int8_gpt (csrc/claude/NOTES.md)
def bgemm_fake_int8_claude(X: Tensor, W: Tensor, lut: Tensor) -> Tensor:
    return torch.ops.approxtorch.bgemm_fake_int8_forward_cuda_claude.default(X, W, lut)


# same forward, but also returns the internal uint8 quantized images
# (xq [N,K,L], wq [O,K]) for the autograd Function to save (4x less memory)
def bgemm_fake_int8_claude_save(X: Tensor, W: Tensor, lut: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    return torch.ops.approxtorch.bgemm_fake_int8_forward_cuda_claude_save.default(X, W, lut)


# uint8 variant (uint8 x uint8 approximate multiplier, asymmetric quant):
#   y[n,o,l] = sum_k lut[X[n,k,l] * 256 + W[o,k]]
# both operands hold unsigned quantized values in [0,255] (X: fp32 integer
# values, or uint8 dtype = already LUT indices; W: fp32 integer values).
# LUT layout: lut[x_u8][w_u8], raw values as indices on both sides.
# Zero-point corrections are the caller's job. Same kernel/modes as int8.
def bgemm_fake_uint8_claude(X: Tensor, W: Tensor, lut: Tensor) -> Tensor:
    return torch.ops.approxtorch.bgemm_fake_uint8_forward_cuda_claude.default(X, W, lut)


def bgemm_fake_uint8_claude_save(X: Tensor, W: Tensor, lut: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    return torch.ops.approxtorch.bgemm_fake_uint8_forward_cuda_claude_save.default(X, W, lut)


# claude LUT GEMM, torch.mm layout: A [M,K] @ B [K,N] -> y [M,N].
#   int8:  y[m,n] = sum_k lut[(A[m,k]+128)*256 + (B[k,n]+128)]
#   uint8: y[m,n] = sum_k lut[A[m,k]*256 + B[k,n]]        (raw values)
# A: fp32 integer values or uint8 (already LUT indices). B: fp32 or uint8,
# any strides — the [K,N]->[N,K] transpose is fused into the u8 quantize
# prepass, so B = W.t() views cost the same as contiguous B.
# _save also returns the u8 images (aq [M,K], bq [N,K] — bq is transposed).
def gemm_fake_int8_claude(A: Tensor, B: Tensor, lut: Tensor) -> Tensor:
    return torch.ops.approxtorch.gemm_fake_int8_forward_cuda_claude.default(A, B, lut)


def gemm_fake_int8_claude_save(A: Tensor, B: Tensor, lut: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    return torch.ops.approxtorch.gemm_fake_int8_forward_cuda_claude_save.default(A, B, lut)


def gemm_fake_uint8_claude(A: Tensor, B: Tensor, lut: Tensor) -> Tensor:
    return torch.ops.approxtorch.gemm_fake_uint8_forward_cuda_claude.default(A, B, lut)


def gemm_fake_uint8_claude_save(A: Tensor, B: Tensor, lut: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    return torch.ops.approxtorch.gemm_fake_uint8_forward_cuda_claude_save.default(A, B, lut)


def bgemm_lre_backward(grad_output: Tensor, x: Tensor, w: Tensor, dx: Tensor, dw: Tensor) -> tuple[Tensor, Tensor]:
    return torch.ops.approxtorch.bgemm_lre_backward.default(grad_output, x, w, dx, dw)


# claude-optimized LRE backward (LUT prepass + cuBLAS, csrc/claude/NOTES.md)
def bgemm_lre_backward_claude(grad_output: Tensor, x: Tensor, w: Tensor, dx: Tensor, dw: Tensor) -> tuple[Tensor, Tensor]:
    return torch.ops.approxtorch.bgemm_lre_backward_claude.default(grad_output, x, w, dx, dw)


# u8 im2col: unfolds a quantized image [N,C,H,W] (int8/uint8/float) straight
# to uint8 LUT indices [N, C*kh*kw, L] (padding -> index 128). Feeds
# bgemm_fake_int8_claude(_save) directly, skipping both the fp32 unfold and
# the forward kernel's fp32->u8 prepass.
def im2col_u8(feature: Tensor, kernel_size, stride=1, padding=0, dilation=1) -> Tensor:
    kh, kw = _pair(kernel_size)
    sh, sw = _pair(stride)
    ph, pw = _pair(padding)
    dilh, dilw = _pair(dilation)
    return torch.ops.approxtorch.im2col_u8.default(
        feature, kh, kw, sh, sw, ph, pw, dilh, dilw)


# out[i] = lut[idx(x[i])] for a [256] LUT; x int8 (idx = v+128), uint8
# (idx = v) or float32 (idx = round+clamp+128)
def lut_map(x: Tensor, lut: Tensor) -> Tensor:
    return torch.ops.approxtorch.lut_map.default(x, lut)


# lut_map for an image [N,C,H,W] with constant border: output
# [N,C,H+2ph,W+2pw], border = lut[128] (what unfold's zero padding maps to)
def lut_map_pad(x: Tensor, lut: Tensor, padding) -> Tensor:
    ph, pw = _pair(padding)
    return torch.ops.approxtorch.lut_map_pad.default(x, lut, ph, pw)


# depthwise approximate-mult conv forward (groups == C, channel multiplier 1):
# x quantized image [N,C,H,W] (int8/uint8/float), w fp32 [C, kh*kw] quantized
# values. Returns (y [N,C,OH,OW] fp32, wq [C, kh*kw] u8 LUT indices). y is
# bit-identical to running the LUT-BGEMM per channel (same tap order).
def dwconv_fake_int8_claude(x: Tensor, w: Tensor, lut: Tensor,
                            kernel_size, stride=1, padding=0,
                            dilation=1) -> tuple[Tensor, Tensor]:
    kh, kw = _pair(kernel_size)
    sh, sw = _pair(stride)
    ph, pw = _pair(padding)
    dilh, dilw = _pair(dilation)
    return torch.ops.approxtorch.dwconv_fake_int8_claude.default(
        x, w, lut, kh, kw, sh, sw, ph, pw, dilh, dilw)


# implicit-im2col variant: x is the PRE-unfold quantized image [N,C,H,W]
# (int8/uint8/float); im2col indices are computed on the fly inside the
# X' build kernel, the unfolded tensor is never materialized
def bgemm_lre_backward_claude_im2col(grad_output: Tensor, x: Tensor, w: Tensor,
                                     dx: Tensor, dw: Tensor,
                                     kernel_size, stride, padding, dilation) -> tuple[Tensor, Tensor]:
    kh, kw = _pair(kernel_size)
    sh, sw = _pair(stride)
    ph, pw = _pair(padding)
    dilh, dilw = _pair(dilation)
    return torch.ops.approxtorch.bgemm_lre_backward_claude_im2col.default(
        grad_output, x, w, dx, dw, kh, kw, sh, sw, ph, pw, dilh, dilw)


def elementwise_mul(a: Tensor, b: Tensor, lut: Tensor) -> Tensor:
    return torch.ops.approxtorch.elementwise_mul.default(a, b, lut)
