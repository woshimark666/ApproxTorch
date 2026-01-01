import torch
import approxtorch as at

B = 128
C = 32
KK = 9
L = 8*8

device = torch.device("cuda:0")
cpu = torch.device("cpu")



for B in [256, 512]:
    for C in [512]:
        for KK in [36, 49]:
            for L in [64, 128, 256, 512]:
            
                X = torch.randint(-128, 127, (B, C, KK, L), device=device, dtype=torch.int8)
                W = torch.randint(-128, 127, (C, 1, KK), device=device, dtype=torch.int8)
                lut = at.load_lut("exact.txt").to(device)


                Y = at.approx_gemm.ops.depthwise_gemm_int8(X, W, lut)
                Y = Y.view(B, C, L)
                Y = Y.to(cpu)
                X_cpu = X.detach().clone().cpu()
                W_cpu = W.detach().clone().cpu()



                Y_cpu = torch.einsum('bckl,cok->bcol', X_cpu.to(torch.int32), W_cpu.to(torch.int32))
                Y_cpu = Y_cpu.view(B, C, L)


                is_true = torch.allclose(Y_cpu, Y)
                print(is_true, B, C, KK, L)
                
                if not is_true:
                    print(Y_cpu)
                    print(Y)
                    break

