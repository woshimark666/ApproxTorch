import torch
from torch import Tensor


__all__ = ['gemm_int8', 'batch_gemm_int8', 'gemm_int8_naive', 'gemm_int8_old', 'gemm_int8_gradient', 'gemm_uint8', 
            'depthwise_gemm_int8', 'depthwise_gemm_int8_gradient', 'gemm_int4', 'gemm_uint8_gradient']

def gemm_int8(A: Tensor, B: Tensor, C: Tensor) -> Tensor:
    return torch.ops.approxtorch.gemm_int8.default(A, B, C)

def gemm_int4(A: Tensor, B: Tensor, C: Tensor) -> Tensor:
    return torch.ops.approxtorch.gemm_int4.default(A, B, C)

@torch.library.register_fake("approxtorch::gemm_int8")
def _(a, b, c):
    # shape a[M, K], b[K, N]
    
    torch._check(a.dtype == torch.int8)
    torch._check(b.dtype == torch.int8)
    torch._check(c.dtype == torch.int32)
    torch._check(a.device == b.device == c.device)
    
    # output same device as a, has shape [M, N], dtype int32,
    M, K = a.shape
    N = b.shape[1]
    result = torch.empty((M, N), dtype=torch.int32, device=a.device)
    return result



def gemm_int8_naive(A: Tensor, B: Tensor, C: Tensor) -> Tensor:
    return torch.ops.approxtorch.gemm_int8_naive.default(A, B, C)


def gemm_int8_gradient(A: Tensor, B: Tensor, grad_A_lut: Tensor, grad_B_lut: Tensor) -> tuple[Tensor, Tensor]:
    return torch.ops.approxtorch.gemm_int8_gradient.default(A, B, grad_A_lut, grad_B_lut)

def gemm_uint8_gradient(A: Tensor, B: Tensor, grad_A_lut: Tensor, grad_B_lut: Tensor) -> tuple[Tensor, Tensor]:
    return torch.ops.approxtorch.gemm_uint8_gradient.default(A, B, grad_A_lut, grad_B_lut)


# def gemm_int8_opt(A: Tensor, B: Tensor, lut: Tensor) -> Tensor:
#     return torch.ops.approxtorch.gemm_int8_opt.default(A, B, lut)

def gemm_uint8(A: Tensor, B: Tensor, lut: Tensor) -> Tensor:
    return torch.ops.approxtorch.gemm_uint8.default(A, B, lut)


def depthwise_gemm_int8(X: Tensor, W: Tensor, lut: Tensor) -> Tensor:
    return torch.ops.approxtorch.depthwise_gemm_int8.default(X, W, lut)

def depthwise_gemm_int8_gradient(X: Tensor, W: Tensor, grad_X_lut: Tensor, grad_W_lut: Tensor) -> tuple[Tensor, Tensor]:
    return torch.ops.approxtorch.depthwise_gemm_int8_gradient.default(X, W, grad_X_lut, grad_W_lut)


def gemm_custom_grad_uint8_tt(A: Tensor, B: Tensor, upstream_grad: Tensor, grad_lut_dx: Tensor, grad_lut_dy: Tensor, scale_A: Tensor, zero_A: Tensor, scale_B: Tensor, zero_B: Tensor) -> tuple[Tensor, Tensor]:
    return torch.ops.approxtorch.gemm_custom_grad_uint8_tt.default(A, B, upstream_grad, grad_lut_dx, grad_lut_dy, scale_A, zero_A, scale_B, zero_B)

def gemm_custom_grad_uint8_tc(A: Tensor, B: Tensor, upstream_grad: Tensor, grad_lut_dx: Tensor, grad_lut_dy: Tensor, scale_A: Tensor, zero_A: Tensor, scale_B: Tensor, zero_B: Tensor) -> tuple[Tensor, Tensor]:
    return torch.ops.approxtorch.gemm_custom_grad_uint8_tc.default(A, B, upstream_grad, grad_lut_dx, grad_lut_dy, scale_A, zero_A, scale_B, zero_B)

def gemm_custom_grad_uint8_dw_only(A: Tensor, B: Tensor, upstream_grad: Tensor, grad_lut_dy: Tensor, scale_A: Tensor, zero_A: Tensor) -> Tensor:
    return torch.ops.approxtorch.gemm_custom_grad_uint8_dw_only.default(A, B, upstream_grad, grad_lut_dy, scale_A, zero_A)