import torch
import approxtorch as at
import numpy as np

device = torch.device("cuda")
grad_lut = at.load_gradient_lut('exact_grad.txt').to(device)
grad_lut_dx, grad_lut_dy = torch.unbind(grad_lut, dim=1)

M = 1024; N = 256; K = 512

A = torch.randint(-127, 127, (M, K), device=device, dtype=torch.int8)
B = torch.randint(-127, 127, (K, N), device=device, dtype=torch.int8)
a = A.clone().detach().to(torch.float32).requires_grad_(True)
b = B.clone().detach().to(torch.float32).requires_grad_(True)

upstream_grad_gpu = torch.ones((M, N), device=device, dtype=torch.float)
upstream_grad_cpu = upstream_grad_gpu.detach().clone()


grad_A, grad_B = at.approx_gemm.ops.gemm_int8_gradient(A, B, grad_lut_dx, grad_lut_dy)

grad_A = upstream_grad_gpu.matmul(B.to(torch.float).t())
grad_B = A.to(torch.float).t().matmul(upstream_grad_gpu)


c = torch.matmul(a, b)
c.backward(upstream_grad_cpu)

grad_a = a.grad
grad_b = b.grad

print(grad_A - grad_a)
print(grad_B - grad_b)