import numpy as np
import torch
import approxtorch as at

torch.manual_seed(42)
np.random.seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)


def checkone(M, K, N):
    lut = at.load_uint8_lut('exact_uint8.txt').to(device='cuda')
    A = torch.randint(0, 255, (M, K), dtype=torch.uint8, device='cuda')
    B = torch.randint(0, 255, (K, N), dtype=torch.uint8, device='cuda')
    C = at.approx_gemm.ops.gemm_uint8(A, B, lut).to(torch.float)
    real_C = torch.matmul(A.to(torch.float), B.to(torch.float))
    return (C - real_C).abs().max()

def check_uint8_gemm():
    lut = at.load_uint8_lut('exact_uint8.txt').to(device='cuda')
    # for i in range(0, 256):
    #     for j in range(0, 256):
    #         if lut[i*256 + j] != i * j:
    #             print(f"worng at {i}, {j}")


    cpu = torch.device('cpu')
    for M in range(10, 5000, 100):
        for K in range(10, 5000, 100):
            for N in range(10, 5000, 100):
                A = torch.randint(0, 255, (M, K), dtype=torch.uint8, device='cuda')
                B = torch.randint(0, 255, (K, N), dtype=torch.uint8, device='cuda')





                C = at.approx_gemm.ops.gemm_uint8(A, B, lut).to(cpu)

                real_C = torch.matmul(A.to(torch.int).to(cpu), B.to(torch.int).to(cpu)).to(torch.int)

                if (C - real_C).abs().max() < 1:
                    print(f'{M} {K} {N} is ok')
                else:
                    print(f'{M} {K} {N} is wrong')
                    print('warning !!!!!!!!!!!!!!!!!!!!!!')
                    return 


def check_int8_gemm():
    lut = at.load_lut('exact.txt').to(device='cuda')
    for M in range(10, 5000, 100):
        for K in range(10, 5000, 100):
            for N in range(10, 5000, 100):
                A = torch.randint(-128, 127, (M, K), dtype=torch.int8, device='cuda')
                B = torch.randint(-128, 127, (K, N), dtype=torch.int8, device='cuda')

                C = at.approx_gemm.ops.gemm_int8(A, B, lut)
                real_C = torch.matmul(A.to(torch.float), B.to(torch.float)).to(torch.int)
                
                if (C - real_C).abs().max() > 1:
                    print(C-real_C)
                    break
                
if __name__ == '__main__':
    # check_int8_gemm()
    check_uint8_gemm()
    # checkone(510, 4810, 10)