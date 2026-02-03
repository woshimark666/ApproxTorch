#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>
using int4_t = int4;

#ifdef __CUDACC__

template <typename T, 
const int BM, const int BN, const int BK, 
const int rowStrideA, const int rowStrideB>
__device__ void loadFromGmem_Robust(
                uint M, uint N, uint K,
                uint k_offset,  // 【新增参数】当前 K 维度的偏移量
                uint blockRow, uint blockCol,
                const T *__restrict__ A,
                const T *__restrict__ B,
                T *As,
                T *Bs,
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

        const T* src_ptr = &A[global_row * K + k_start];
        
        bool is_aligned = (reinterpret_cast<uintptr_t>(src_ptr) % 16 == 0);
        // 【修改点】边界检查也要基于加上偏移后的 k_start
        bool is_safe_range = (global_row < M) && (k_start + 16 <= K);

        if (is_aligned && is_safe_range) {
            int4_t loaded_vec = *reinterpret_cast<const int4_t*>(src_ptr);
            T* val_ptr = reinterpret_cast<T*>(&loaded_vec);
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
                T val = 0;
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

        const T* src_ptr = &B[global_k * N + global_n_start];
        
        bool is_aligned = (reinterpret_cast<uintptr_t>(src_ptr) % 16 == 0);
        bool is_safe_range = (global_k < K) && (global_n_start + 16 <= N);

        if (is_aligned && is_safe_range) {
            *reinterpret_cast<int4_t*>(&Bs[local_k * BN + local_n_start]) = 
                *reinterpret_cast<const int4_t*>(src_ptr);
        } else {
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                uint current_n = global_n_start + i;
                T val = 0;
                if (global_k < K && current_n < N) {
                    val = B[global_k * N + current_n];
                }
                Bs[local_k * BN + (local_n_start + i)] = val;
            }
        }
    }
}


// --- 2. 带逻辑屏蔽的计算核心 (Masked Compute) --- for int8
template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WMITER, const int WNITER, const int WSUBM, const int WSUBN,
          const int TM, const int TN>
__device__ void
processFromSmemMasked_int8(
    int8_t *regM, int8_t *regN, int *threadResults, 
    const int8_t *As, const int8_t *Bs, 
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
            int8_t valA = regM[wSubRowIdx * TM + resIdxM];
            int8_t valB = regN[wSubColIdx * TN + resIdxN];
            
            // 查表索引计算
            //  in bgemm: valB is the feature, valA is the weight, 
            // we always put the feature in front
            int lut_idx = (int)valB * 256 + (int)valA + 32896; // 假设是平铺的表
            
            // 累加
            threadResults[(wSubRowIdx * TM + resIdxM) * (WNITER * TN) + (wSubColIdx * TN) + resIdxN] 
                += __ldg(&lut[lut_idx]);
          }
        }
      }
    }
  }
}


#endif