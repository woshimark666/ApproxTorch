#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
const int WARPSIZE = 32; 
using int4_t = int4;

// 有一个细节，我想要把寻址的那个变量用uint，这样肯定就不会溢出了
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
            int lut_idx = (int)valA * 256 + (int)valB; // 假设是平铺的表
            
            // 累加
            threadResults[(wSubRowIdx * TM + resIdxM) * (WNITER * TN) + (wSubColIdx * TN) + resIdxN] 
                += __ldg(&lut[lut_idx]);
          }
        }
      }
    }
  }
}



// --- 3. Kernel 入口 ---
template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int TM, const int TN, const int NUM_THREADS>
__global__ void approx_gemm_kernel(
    uint M, uint N, uint K, 
    const uint8_t* __restrict__ A, // matrix A [M, K]
    const uint8_t* __restrict__ B, // matrix B [K, N]
    const int32_t* __restrict__ lut,            // LUT [256, 256]
    int* __restrict__ C            // matrix C [M, N]
) 
{
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;
  
    // Placement of the warp in the threadblock tile
    const uint warpIdx = threadIdx.x / WARPSIZE; // the warp this thread is in
    const uint warpCol = warpIdx % (BN / WN);
    const uint warpRow = warpIdx / (BN / WN);
  
    // size of the warp subtile
    constexpr uint WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
    constexpr uint WSUBM = WM / WMITER; // 64/2=32
    constexpr uint WSUBN = WN / WNITER; // 32/2=16
  
    // Placement of the thread in the warp subtile
    const uint threadIdxInWarp = threadIdx.x % WARPSIZE;         // [0, 31]
    const uint threadColInWarp = threadIdxInWarp % (WSUBN / TN); // i%(16/4)
    const uint threadRowInWarp = threadIdxInWarp / (WSUBN / TN); // i/4
  
    // allocate space for the current blocktile in SMEM
    __shared__ uint8_t As[BK * BM];
    __shared__ uint8_t Bs[BK * BN];

    // calculating the indices that this thread will load into SMEM
    // we'll load 128bit / 32bit = 4 elements per thread at each step
    const uint innerRowA = threadIdx.x / (BK / 16);
    const uint innerColA = threadIdx.x % (BK / 16);
    constexpr uint rowStrideA = (NUM_THREADS * 16) / BK;
    const uint innerRowB = threadIdx.x / (BN / 16);
    const uint innerColB = threadIdx.x % (BN / 16);
    constexpr uint rowStrideB = NUM_THREADS / (BN / 16);

    // 寄存器缓存
    int threadResults[WMITER * TM * WNITER * TN] = {0};
    uint8_t regM[WMITER * TM] = {0};
    uint8_t regN[WNITER * TN] = {0};

  // --- Main Loop ---
    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) 
    {
        approxtorch::loadFromGmemUint8_Robust<BM, BN, BK, rowStrideA, rowStrideB>(
            M, N, K, bkIdx, cRow, cCol, A, B, 
            As, Bs, innerRowA, innerColA, innerRowB, innerColB);
        __syncthreads();
    
        // 2. 计算当前 Block 的有效 K 长度
        uint validK = K - bkIdx;
        if (validK > BK) validK = BK;

        // 3. process the shared memory
        approxtorch::processFromSmemMasked<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM, TN>(
            regM, regN, threadResults, As, Bs, 
            warpRow, warpCol, threadRowInWarp, threadColInWarp, 
            validK, lut);  
        __syncthreads();
    }
    
    // ==========================================================
    //  Write Back Logic (边界安全版)
    // ==========================================================

    // 计算当前 Block 和 Warp 在全局 C 矩阵中的基准偏移
    // C_base_row = (BlockIdx.y * BM) + (WarpRow * WM)
    uint globalBaseRow = cRow * BM + warpRow * WM;
    // C_base_col = (BlockIdx.x * BN) + (WarpCol * WN)
    uint globalBaseCol = cCol * BN + warpCol * WN;

    // 遍历当前 Warp 负责的每个 Sub-Tile
#pragma unroll
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
#pragma unroll
        for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
        
            // 当前 Sub-Tile 的像素偏移
            uint subTileRowOffset = wSubRowIdx * WSUBM;
            uint subTileColOffset = wSubColIdx * WSUBN;

            // 遍历线程负责的 TM x TN 小块
#pragma unroll
            for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
                uint globalRow = globalBaseRow + subTileRowOffset + threadRowInWarp * TM + resIdxM;

#pragma unroll
                for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
                    uint globalCol = globalBaseCol + subTileColOffset + threadColInWarp * TN + resIdxN;

                    // 1. 准备数据
                    uint i = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) + (wSubColIdx * TN) + resIdxN;
                    int4 val;
                    val.x = threadResults[i + 0];
                    val.y = threadResults[i + 1];
                    val.z = threadResults[i + 2];
                    val.w = threadResults[i + 3];

                    // 2. 计算目标地址
                    int* dst_ptr = &C[globalRow * N + globalCol];

                    // 3. 【核心修复】不仅检查边界，还要检查地址对齐！
                    // 条件：(行不越界) AND (列剩余空间 >= 4) AND (地址是 16 字节对齐的)
                    bool can_use_int4 = (globalRow < M) && 
                                        (globalCol + 3 < N) && 
                                        (reinterpret_cast<uintptr_t>(dst_ptr) % 16 == 0);

                    if (can_use_int4) {
                        // Fast Path: 只有完全对齐且不越界时才用向量写
                        *reinterpret_cast<int4*>(dst_ptr) = val;
                    } 
                    else if (globalRow < M) {
                        // Slow Path: 只要不对齐，或者到了边界，就乖乖一个一个写
                        if (globalCol + 0 < N) dst_ptr[0] = val.x;
                        if (globalCol + 1 < N) dst_ptr[1] = val.y;
                        if (globalCol + 2 < N) dst_ptr[2] = val.z;
                        if (globalCol + 3 < N) dst_ptr[3] = val.w;
                    }
                }
            }
        }
    }
    //  写回结束


}




torch::Tensor 
approx_gemm(const torch::Tensor& A, const torch::Tensor& B, const torch::Tensor& lut) 
{
    const at::cuda::OptionalCUDAGuard device_guard(device_of(A));
    int device_id = A.get_device();
    uint M = A.size(0);
    uint N = B.size(1);
    uint K = A.size(1);
    auto tensor_options = torch::TensorOptions().device(A.device()).dtype(torch::kInt32);
    auto C = torch::empty({M, N}, tensor_options);

    // ------------------------------------------------------------------
    // 完美适配 int4 的参数组合
    // ------------------------------------------------------------------
    constexpr uint BM = 64;   // 保持不变
    constexpr uint BN = 64;   // 保持不变
    constexpr uint BK = 64;   // 【关键修改】从 32 改为 64，为了容纳 int4 数据量

    // Warp Tiling 参数 (适配 256 线程 / 8 Warps)
    constexpr uint WM = 32;   // 【修改】
    constexpr uint WN = 16;   // 【修改】让 8 个 Warp 拼成 64x64
    constexpr uint WNITER = 1;// 【修改】减少寄存器压力

    constexpr uint TM = 4;    // 保持不变，计算效率高
    constexpr uint TN = 4;    // 保持不变
    constexpr uint NUM_THREADS = 256; // 保持不变
    dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 block(BM * BN / (TM * TN));


    // cudaMemAdvise(lut.data_ptr<int32_t>(), 4*256*256, cudaMemAdviseSetReadMostly , device_id);
    approx_gemm_kernel<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS>
    <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>
    (M, N, K, A.data_ptr<uint8_t>(), B.data_ptr<uint8_t>(), lut.data_ptr<int32_t>(), C.data_ptr<int32_t>());
    // cudaMemAdvise(lut.data_ptr<int32_t>(), 4*256*256, cudaMemAdviseUnsetReadMostly , device_id);
    return C;
}
// ... Binding logic ...
TORCH_LIBRARY_FRAGMENT(approxtorch, m){
    m.def("approx_gemm_uint8(Tensor A, Tensor B, Tensor lut) -> Tensor");
}
TORCH_LIBRARY_IMPL(approxtorch, CUDA, m){
    m.impl("approx_gemm_uint8", &approx_gemm);
}
} // namespace approx_gemm