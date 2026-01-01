import torch
import numpy as np

M = 100
N = 1024
K = 87

device = torch.device("cuda:0")
da_lut = np.loadtxt('exact_int4_grad_a.txt', dtype=np.float32)
db_lut = np.loadtxt('exact_int4_grad_b.txt', dtype=np.float32)
da_lut = torch.tensor(da_lut, device=device, dtype=torch.float32)
db_lut = torch.tensor(db_lut, device=device, dtype=torch.float32)


up_grad = torch.ones((M, N), device=device, dtype=torch.float)
# up_grad = torch.randint(-10, 10, (M, N), device=device, dtype=torch.float



A = torch.randint(-8, 7, (M, K), device=device, dtype=torch.int8)
B = torch.randint(-8, 7, (K, N), device=device, dtype=torch.int8)

a = A.clone().detach().to(torch.float).requires_grad_(True)
b = B.clone().detach().to(torch.float).requires_grad_(True)


A = A.to(torch.int)
B = B.to(torch.int)
dA = torch.zeros((M, K), device=device, dtype=torch.float)
dB = torch.zeros((K, N), device=device, dtype=torch.float)
for k in range(K):
    A_k = A[:, k] + 8
    B_k = B[k, :] + 8
    
    ga = da_lut[A_k[:, None], B_k[None, :]]
    gb = db_lut[A_k[:, None], B_k[None, :]]
    
    dA[:, k] = (up_grad * ga).sum(dim=1)
    dB[k, :] = (up_grad * gb).sum(dim=0)


c = torch.matmul(a, b)
c.backward(up_grad)
print(a.grad)
print(b.grad)


print(dA)
print(dB)



print(a.grad - dA)
print(b.grad - dB)