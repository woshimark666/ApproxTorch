#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
const int WARPSIZE = 32; 
using int4_t = int4;


namespace approxtorch {
template <const int BM, const int BN, const int BK, const int rowStrideA, const int rowStrideB>
__device__ void loadFromGmemUint8_Robust(
                uint M, uint N, uint K,
                uint k_offset,  // 【新增参数】当前 K 维度的偏移量
                uint blockRow, uint blockCol,
                const uint8_t *__restrict__ A,
                const uint8_t *__restrict__ B,
                uint8_t *As,
                uint8_t *Bs,
                uint innerRowA, uint innerColA,
                uint innerRowB, uint innerColB) 
{
    // ==========================================================
    // 1. 加载矩阵 A (Robust Version)
    //    Global: [M, K] -> Shared: [BK][BM] (Transposed)
    // ==========================================================
#pragma unroll
    for (uint offset = 0; offset < BM; offset += rowStrideA) 
    {
        if (innerRowA + offset >= BM) continue;

        uint global_row = blockRow * BM + innerRowA + offset;
        
        // 【修改点】计算 K 时加上 k_offset
        uint k_start = k_offset + innerColA * 16; 

        const uint8_t* src_ptr = &A[global_row * K + k_start];
        
        bool is_aligned = (reinterpret_cast<uintptr_t>(src_ptr) % 16 == 0);
        // 【修改点】边界检查也要基于加上偏移后的 k_start
        bool is_safe_range = (global_row < M) && (k_start + 16 <= K);

        if (is_aligned && is_safe_range) {
            int4_t loaded_vec = *reinterpret_cast<const int4_t*>(src_ptr);
            uint8_t* val_ptr = reinterpret_cast<uint8_t*>(&loaded_vec);
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                As[(innerColA * 16 + i) * BM + (innerRowA + offset)] = val_ptr[i];
            }
        } 
        else {
            // Scalar fallback
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                uint current_k = k_start + i;
                uint8_t val = 0;
                if (global_row < M && current_k < K) {
                    val = A[global_row * K + current_k];
                }
                As[(innerColA * 16 + i) * BM + (innerRowA + offset)] = val; // Index within As stays relative
            }
        }
    }

    // ==========================================================
    // 2. 加载矩阵 B (Robust Version)
    //    Global: [K, N] -> Shared: [BK][BN]
    // ==========================================================
#pragma unroll
    for (uint offset = 0; offset < BK; offset += rowStrideB) 
    {
        if (innerRowB + offset >= BK) continue;

        uint local_k = innerRowB + offset; 
        uint local_n_start = innerColB * 16;
        
        // 【修改点】计算 global_k 时加上 k_offset
        uint global_k = k_offset + local_k; 
        uint global_n_start = blockCol * BN + local_n_start;

        const uint8_t* src_ptr = &B[global_k * N + global_n_start];
        
        bool is_aligned = (reinterpret_cast<uintptr_t>(src_ptr) % 16 == 0);
        bool is_safe_range = (global_k < K) && (global_n_start + 16 <= N);

        if (is_aligned && is_safe_range) {
            *reinterpret_cast<int4_t*>(&Bs[local_k * BN + local_n_start]) = 
                *reinterpret_cast<const int4_t*>(src_ptr);
        } else {
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                uint current_n = global_n_start + i;
                uint8_t val = 0;
                if (global_k < K && current_n < N) {
                    val = B[global_k * N + current_n];
                }
                Bs[local_k * BN + (local_n_start + i)] = val;
            }
        }
    }
}

// --- 2. 带逻辑屏蔽的计算核心 (Masked Compute) ---
template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WMITER, const int WNITER, const int WSUBM, const int WSUBN,
          const int TM, const int TN>
__device__ void
processFromSmemMasked(
    uint8_t *regM, uint8_t *regN, int *threadResults, 
    const uint8_t *As, const uint8_t *Bs, 
    const uint warpRow, const uint warpCol,
    const uint threadRowInWarp, const uint threadColInWarp,
    uint validK,         // 当前 Block 的有效 K 长度
    const int* lut // 传入 LUT 指针
) {
  // 遍历 K 维度 (Block Tile K)
  for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      
    // 【核心屏蔽逻辑】：如果 dotIdx 处于 Padding 区域，直接跳过计算
    if (dotIdx >= validK) continue; 
    // 注意：如果是 Warp 同步的架构，continue 可能导致不同步？
    // 在 Volta/Ampere 架构后最好用 if 包裹整个体：
    
    // 1. Load Registers (即便无效数据也无妨，因为后面不累加)
#pragma unroll
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
#pragma unroll
      for (uint i = 0; i < TM; ++i) {
        // 强制转 float 或 int 都可以，看你 LUT 输入要求
        regM[wSubRowIdx * TM + i] = As[(dotIdx * BM) + warpRow * WM + wSubRowIdx * WSUBM + threadRowInWarp * TM + i];
      }
    }
#pragma unroll
    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
#pragma unroll
      for (uint i = 0; i < TN; ++i) {
        regN[wSubColIdx * TN + i] = Bs[(dotIdx * BN) + warpCol * WN + wSubColIdx * WSUBN + threadColInWarp * TN + i];
      }
    }

    // 2. Compute (只有 validK 范围内才累加)
#pragma unroll
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
#pragma unroll
      for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
#pragma unroll
        for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
#pragma unroll
          for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
            
            // 获取操作数
            uint8_t valA = regM[wSubRowIdx * TM + resIdxM];
            uint8_t valB = regN[wSubColIdx * TN + resIdxN];
            
            // 查表索引计算
            int lut_idx = (int)valB * 256 + (int)valA; // 假设是平铺的表
            
            // 累加
            threadResults[(wSubRowIdx * TM + resIdxM) * (WNITER * TN) + (wSubColIdx * TN) + resIdxN] 
                += __ldg(&lut[lut_idx]);
          }
        }
      }
    }
  }
}


// --- 3. Kernel 入口 (支持 Batched Gemm: C[batch] = A_shared * B_batched[batch]) ---
// 注意：为了复用代码逻辑，我们将用户的 B 传给 Kernel 的 A，用户的 A 传给 Kernel 的 B
template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int TM, const int TN, const int NUM_THREADS>
__global__ void approx_bgemm_kernel(
    uint M, uint N, uint K, 
    const uint8_t* __restrict__ A_weight, // Kernel A: 对应 User B [O, CKK] -> [M, K]
    const uint8_t* __restrict__ B_feature,  // Kernel B: 对应 User A [Batch, CKK, L] -> [Batch, K, N]
    const int32_t* __restrict__ lut,      
    int* __restrict__ C_batch             // Kernel C: 对应 User C [Batch, O, L] -> [Batch, M, N]
) 
{
    // 【新增】处理 Batch 索引
    // blockIdx.z 对应 User A 的 N (Batch Size) 维度
    const uint batchIdx = blockIdx.z;

    // 【新增】计算当前 Batch 的指针偏移
    // Kernel B (User A) 的每个 Batch 大小是 K * N (即 CKK * L)
    const uint8_t* B_ptr = B_feature + batchIdx * (K * N);
    
    // Kernel C (User C) 的每个 Batch 大小是 M * N (即 O * L)
    int* C_ptr = C_batch + batchIdx * (M * N);

    // Kernel A (User B) 是共享的，不需要偏移
    const uint8_t* A_ptr = A_weight;

    // --- 以下逻辑与之前的 GEMM 完全一致，只是指针换成了上面计算后的 ---
    
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;
  
    // Placement of the warp in the threadblock tile
    const uint warpIdx = threadIdx.x / WARPSIZE; 
    const uint warpCol = warpIdx % (BN / WN);
    const uint warpRow = warpIdx / (BN / WN);
  
    // size of the warp subtile
    constexpr uint WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
    constexpr uint WSUBM = WM / WMITER; 
    constexpr uint WSUBN = WN / WNITER; 
  
    const uint threadIdxInWarp = threadIdx.x % WARPSIZE;        
    const uint threadColInWarp = threadIdxInWarp % (WSUBN / TN); 
    const uint threadRowInWarp = threadIdxInWarp / (WSUBN / TN); 
  
    __shared__ uint8_t As[BK * BM];
    __shared__ uint8_t Bs[BK * BN];

    const uint innerRowA = threadIdx.x / (BK / 16);
    const uint innerColA = threadIdx.x % (BK / 16);
    constexpr uint rowStrideA = (NUM_THREADS * 16) / BK;
    const uint innerRowB = threadIdx.x / (BN / 16);
    const uint innerColB = threadIdx.x % (BN / 16);
    constexpr uint rowStrideB = NUM_THREADS / (BN / 16);

    int threadResults[WMITER * TM * WNITER * TN] = {0};
    uint8_t regM[WMITER * TM] = {0};
    uint8_t regN[WNITER * TN] = {0};

    // --- Main Loop ---
    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) 
    {
        // 注意：这里传入的是计算好偏移量的 A_ptr 和 B_ptr
        approxtorch::loadFromGmemUint8_Robust<BM, BN, BK, rowStrideA, rowStrideB>(
            M, N, K, bkIdx, cRow, cCol, A_ptr, B_ptr, 
            As, Bs, innerRowA, innerColA, innerRowB, innerColB);
        
        __syncthreads();
    
        uint validK = K - bkIdx;
        if (validK > BK) validK = BK;

        approxtorch::processFromSmemMasked<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM, TN>(
            regM, regN, threadResults, As, Bs, 
            warpRow, warpCol, threadRowInWarp, threadColInWarp, 
            validK, lut);  
        __syncthreads();
    }
    
    // ==========================================================
    //  Write Back Logic (写入 C_ptr)
    // ==========================================================
    uint globalBaseRow = cRow * BM + warpRow * WM;
    uint globalBaseCol = cCol * BN + warpCol * WN;

#pragma unroll
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
#pragma unroll
        for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
        
            uint subTileRowOffset = wSubRowIdx * WSUBM;
            uint subTileColOffset = wSubColIdx * WSUBN;

#pragma unroll
            for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
                uint globalRow = globalBaseRow + subTileRowOffset + threadRowInWarp * TM + resIdxM;

#pragma unroll
                for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
                    uint globalCol = globalBaseCol + subTileColOffset + threadColInWarp * TN + resIdxN;

                    uint i = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) + (wSubColIdx * TN) + resIdxN;
                    int4 val;
                    val.x = threadResults[i + 0];
                    val.y = threadResults[i + 1];
                    val.z = threadResults[i + 2];
                    val.w = threadResults[i + 3];

                    // 使用 C_ptr (已偏移)
                    int* dst_ptr = &C_ptr[globalRow * N + globalCol];

                    bool can_use_int4 = (globalRow < M) && 
                                        (globalCol + 3 < N) && 
                                        (reinterpret_cast<uintptr_t>(dst_ptr) % 16 == 0);

                    if (can_use_int4) {
                        *reinterpret_cast<int4*>(dst_ptr) = val;
                    } 
                    else if (globalRow < M) {
                        if (globalCol + 0 < N) dst_ptr[0] = val.x;
                        if (globalCol + 1 < N) dst_ptr[1] = val.y;
                        if (globalCol + 2 < N) dst_ptr[2] = val.z;
                        if (globalCol + 3 < N) dst_ptr[3] = val.w;
                    }
                }
            }
        }
    }
}

torch::Tensor 
approx_bgemm(const torch::Tensor& user_A, const torch::Tensor& user_B, const torch::Tensor& lut) 
{
    // user_A shape: [Batch, CKK, L] -> 对应 Kernel 的 B (Batched)
    // user_B shape: [O, CKK]        -> 对应 Kernel 的 A (Shared)
    
    const at::cuda::OptionalCUDAGuard device_guard(device_of(user_A));
    
    // 1. 提取维度
    uint BatchSize = user_A.size(0); // N in your description
    uint CKK       = user_A.size(1); // K
    uint L         = user_A.size(2); // N (Kernel perspective)
    
    uint O         = user_B.size(0); // M (Kernel perspective)
    // Check: user_B.size(1) must be CKK

    // Kernel Perspective Dimensions:
    uint M_gemm = O;    // Rows of Left Matrix
    uint K_gemm = CKK;  // Reduction Dimension
    uint N_gemm = L;    // Cols of Right Matrix

    // Output Shape: [Batch, O, L]
    auto tensor_options = torch::TensorOptions().device(user_A.device()).dtype(torch::kInt32);
    auto C = torch::empty({BatchSize, O, L}, tensor_options);

    // ------------------------------------------------------------------
    // Config
    // ------------------------------------------------------------------
    constexpr uint BM = 64; 
    constexpr uint BN = 64; 
    constexpr uint BK = 64; 
    constexpr uint WM = 32; 
    constexpr uint WN = 16; 
    constexpr uint WNITER = 1;
    constexpr uint TM = 4; 
    constexpr uint TN = 4; 
    constexpr uint NUM_THREADS = 256; 

    // 【修改】Grid 的 Z 维度设为 BatchSize
    dim3 grid(CEIL_DIV(N_gemm, BN), CEIL_DIV(M_gemm, BM), BatchSize);
    dim3 block(BM * BN / (TM * TN));

    // 【关键调用】交换 user_A 和 user_B 的位置
    // Kernel 参数顺序: M, N, K, A_ptr, B_ptr, ...
    // 我们传入: O, L, CKK, user_B.data, user_A.data
    approx_bgemm_kernel<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS>
    <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>
    (
        M_gemm, N_gemm, K_gemm, 
        user_B.data_ptr<uint8_t>(), // Kernel A (Left) <- User B
        user_A.data_ptr<uint8_t>(), // Kernel B (Right) <- User A
        lut.data_ptr<int32_t>(), 
        C.data_ptr<int32_t>()       // Kernel C
    );

    return C;
}

// ... Binding logic ...
TORCH_LIBRARY_FRAGMENT(approxtorch, m){
    m.def("approx_bgemm_uint8(Tensor A, Tensor B, Tensor lut) -> Tensor");
}
TORCH_LIBRARY_IMPL(approxtorch, CUDA, m){
    m.impl("approx_bgemm_uint8", &approx_bgemm);
}



} // namespace end