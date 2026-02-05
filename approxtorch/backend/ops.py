import torch
from torch import Tensor
from torch.nn.modules.utils import _pair


__all__ = ['im2col_int8', 'im2col_uint8', 'gemm_int8', 'gemm_uint8',
           'gemm_int8_naive', 'bgemm_int8', 'bgemm_uint8']

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