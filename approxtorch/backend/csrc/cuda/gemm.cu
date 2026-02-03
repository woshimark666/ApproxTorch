#include "gemm_utils.cuh"


#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
const int WARPSIZE = 32; 
using int4_t = int4;


namespace approxtorch {

//---------------------------------- int8 ************************************************
template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int TM, const int TN, const int NUM_THREADS>
__global__ void approx_gemm_kernel_int8(
    uint M, uint N, uint K, 
    const int8_t* __restrict__ A, // matrix A [M, K]
    const int8_t* __restrict__ B, // matrix B [K, N]
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
    __shared__ int8_t As[BK * BM];
    __shared__ int8_t Bs[BK * BN];

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
    int8_t regM[WMITER * TM] = {0};
    int8_t regN[WNITER * TN] = {0};

  // --- Main Loop ---
    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) 
    {
        loadFromGmem_Robust<int8_t, BM, BN, BK, rowStrideA, rowStrideB>(
            M, N, K, bkIdx, cRow, cCol, A, B, 
            As, Bs, innerRowA, innerColA, innerRowB, innerColB);
        __syncthreads();
    
        // 2. 计算当前 Block 的有效 K 长度
        uint validK = K - bkIdx;
        if (validK > BK) validK = BK;

        // 3. process the shared memory
        processFromSmemMasked_int8<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM, TN>(
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
gemm_int8_cuda(const torch::Tensor& A, const torch::Tensor& B, const torch::Tensor& lut) 
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
    approx_gemm_kernel_int8<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS>
    <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>
    (M, N, K, A.data_ptr<int8_t>(), B.data_ptr<int8_t>(), 
    lut.data_ptr<int32_t>(), C.data_ptr<int32_t>());
    // cudaMemAdvise(lut.data_ptr<int32_t>(), 4*256*256, cudaMemAdviseUnsetReadMostly , device_id);
    return C;
}
//---------------------------------- int8 ************************************************


//---------------------------------- uint8 ************************************************
template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int TM, const int TN, const int NUM_THREADS>
__global__ void approx_gemm_kernel_uint8(
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
        loadFromGmem_Robust<uint8_t, BM, BN, BK, rowStrideA, rowStrideB>(
            M, N, K, bkIdx, cRow, cCol, A, B, 
            As, Bs, innerRowA, innerColA, innerRowB, innerColB);
        __syncthreads();
    
        // 2. 计算当前 Block 的有效 K 长度
        uint validK = K - bkIdx;
        if (validK > BK) validK = BK;

        // 3. process the shared memory
        processFromSmemMasked_uint8<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM, TN>(
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
gemm_uint8_cuda(const torch::Tensor& A, const torch::Tensor& B, const torch::Tensor& lut) 
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
    approx_gemm_kernel_uint8<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS>
    <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>
    (M, N, K, A.data_ptr<uint8_t>(), B.data_ptr<uint8_t>(), lut.data_ptr<int32_t>(), C.data_ptr<int32_t>());
    // cudaMemAdvise(lut.data_ptr<int32_t>(), 4*256*256, cudaMemAdviseUnsetReadMostly , device_id);
    return C;
}
//---------------------------------- uint8 ************************************************


TORCH_LIBRARY_IMPL(approxtorch, CUDA, m){
    m.impl("gemm_int8", &gemm_int8_cuda);
    m.impl("gemm_uint8", &gemm_uint8_cuda);
}


}