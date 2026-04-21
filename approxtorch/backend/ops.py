import torch
from torch import Tensor
from torch.nn.modules.utils import _pair


__all__ = ['im2col_int8', 
           'im2col_uint8', 
           'gemm_int8', 
           'gemm_uint8',
           'gemm_int8_naive', 
           'bgemm_int8', 
           'bgemm_uint8', 
           'bgemm_gradual_approx_int8']

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

def bgemm_int8(A: Tensor, B: Tensor, lut: Tensor) -> Tensor:
    return torch.ops.approxtorch.bgemm_int8.default(A, B, lut)

def bgemm_uint8(A: Tensor, B: Tensor, lut: Tensor) -> Tensor:
    return torch.ops.approxtorch.bgemm_uint8.default(A, B, lut)


def bgemm_custom_grad_uint8_naive(X: Tensor, W: Tensor, dY: Tensor, dx_lut: Tensor, dw_lut: Tensor) -> Tensor:
    return torch.ops.approxtorch.bgemm_custom_grad_uint8_naive.default(X, W, dY, dx_lut, dw_lut)

def bgemm_custom_grad_int8_naive(X: Tensor, W: Tensor, dY: Tensor, dx_lut: Tensor, dw_lut: Tensor) -> Tensor:
    return torch.ops.approxtorch.bgemm_custom_grad_int8_naive.default(X, W, dY, dx_lut, dw_lut)

def bgemm_custom_grad_uint8(X: Tensor, W: Tensor, dY: Tensor, dx_lut: Tensor, dw_lut: Tensor) -> Tensor:
    return torch.ops.approxtorch.bgemm_custom_grad_uint8.default(X, W, dY, dx_lut, dw_lut)


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


def bgemm_gradual_int8(X: Tensor, W: Tensor, lut: Tensor, alpha: float) -> Tensor:
    return torch.ops.approxtorch.bgemm_gradual_int8.default(X, W, lut, alpha)