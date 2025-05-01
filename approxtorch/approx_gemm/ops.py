import torch
from torch import Tensor


__all__ = ['gemm_int8', 'batch_gemm_int8', 'gemm_int8_naive', 'gemm_int8_old', 'gemm_int8_gradient']

def gemm_int8(A: Tensor, B: Tensor, C: Tensor) -> Tensor:
    return torch.ops.approxtorch.gemm_int8.default(A, B, C)


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


def gemm_int8_gradient(A: Tensor, B: Tensor, upstream_grad: Tensor, grad_lut: Tensor) -> (Tensor, Tensor):
    return torch.ops.approxtorch.gemm_int8_gradient.default(A, B, upstream_grad, grad_lut)
