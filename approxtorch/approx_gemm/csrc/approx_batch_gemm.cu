#include "gemm.cuh"

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
const int WARPSIZE = 32; 
using int4_t = int4;


namespace approxtorch {
// ******************************** uint8 ************************************************
// --- 3. Kernel 入口 (支持 Batched Gemm: C[batch] = A_shared * B_batched[batch]) ---
// 注意：为了复用代码逻辑，我们将用户的 B 传给 Kernel 的 A，用户的 A 传给 Kernel 的 B
template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int TM, const int TN, const int NUM_THREADS>
__global__ void approx_bgemm_kernel_uint8(
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
        approxtorch::loadFromGmem_Robust<uint8_t, BM, BN, BK, rowStrideA, rowStrideB>(
            M, N, K, bkIdx, cRow, cCol, A_ptr, B_ptr, 
            As, Bs, innerRowA, innerColA, innerRowB, innerColB);
        
        __syncthreads();
    
        uint validK = K - bkIdx;
        if (validK > BK) validK = BK;

        approxtorch::processFromSmemMasked_uint8
        <BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM, TN>(
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
approx_bgemm_uint8
(const torch::Tensor& user_A, const torch::Tensor& user_B, const torch::Tensor& lut) 
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
    approx_bgemm_kernel_uint8<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS>
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
// ********************************* uint8 ************************************************




// *********************************** int8 ************************************************
// --- 3. Kernel 入口 (支持 Batched Gemm: C[batch] = A_shared * B_batched[batch]) ---
// 注意：为了复用代码逻辑，我们将用户的 B 传给 Kernel 的 A，用户的 A 传给 Kernel 的 B
template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int TM, const int TN, const int NUM_THREADS>
__global__ void approx_bgemm_kernel_int8(
    uint M, uint N, uint K, 
    const int8_t* __restrict__ A_weight, // Kernel A: 对应 User B [O, CKK] -> [M, K]
    const int8_t* __restrict__ B_feature,  // Kernel B: 对应 User A [Batch, CKK, L] -> [Batch, K, N]
    const int32_t* __restrict__ lut,      
    int* __restrict__ C_batch             // Kernel C: 对应 User C [Batch, O, L] -> [Batch, M, N]
) 
{
    // 【新增】处理 Batch 索引
    // blockIdx.z 对应 User A 的 N (Batch Size) 维度
    const uint batchIdx = blockIdx.z;

    // 【新增】计算当前 Batch 的指针偏移
    // Kernel B (User A) 的每个 Batch 大小是 K * N (即 CKK * L)
    const int8_t* B_ptr = B_feature + batchIdx * (K * N);
    
    // Kernel C (User C) 的每个 Batch 大小是 M * N (即 O * L)
    int* C_ptr = C_batch + batchIdx * (M * N);

    // Kernel A (User B) 是共享的，不需要偏移
    const int8_t* A_ptr = A_weight;

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
  
    __shared__ int8_t As[BK * BM];
    __shared__ int8_t Bs[BK * BN];

    const uint innerRowA = threadIdx.x / (BK / 16);
    const uint innerColA = threadIdx.x % (BK / 16);
    constexpr uint rowStrideA = (NUM_THREADS * 16) / BK;
    const uint innerRowB = threadIdx.x / (BN / 16);
    const uint innerColB = threadIdx.x % (BN / 16);
    constexpr uint rowStrideB = NUM_THREADS / (BN / 16);

    int threadResults[WMITER * TM * WNITER * TN] = {0};
    int8_t regM[WMITER * TM] = {0};
    int8_t regN[WNITER * TN] = {0};

    // --- Main Loop ---
    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) 
    {
        // 注意：这里传入的是计算好偏移量的 A_ptr 和 B_ptr
        approxtorch::loadFromGmem_Robust<int8_t, BM, BN, BK, rowStrideA, rowStrideB>(
            M, N, K, bkIdx, cRow, cCol, A_ptr, B_ptr, 
            As, Bs, innerRowA, innerColA, innerRowB, innerColB);
        
        __syncthreads();
    
        uint validK = K - bkIdx;
        if (validK > BK) validK = BK;

        approxtorch::processFromSmemMasked_int8
        <BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM, TN>(
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
approx_bgemm_int8
(const torch::Tensor& user_A, const torch::Tensor& user_B, const torch::Tensor& lut) 
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
    approx_bgemm_kernel_int8<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS>
    <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>
    (
        M_gemm, N_gemm, K_gemm, 
        user_B.data_ptr<int8_t>(), // Kernel A (Left) <- User B
        user_A.data_ptr<int8_t>(), // Kernel B (Right) <- User A
        lut.data_ptr<int32_t>(), 
        C.data_ptr<int32_t>()       // Kernel C
    );

    return C;
}


// ********************************** int8 ************************************************
// ... Binding logic ...
TORCH_LIBRARY_FRAGMENT(approxtorch, m){
    m.def("approx_bgemm_uint8(Tensor A, Tensor B, Tensor lut) -> Tensor");
    m.def("approx_bgemm_int8(Tensor A, Tensor B, Tensor lut) -> Tensor");
}
TORCH_LIBRARY_IMPL(approxtorch, CUDA, m){
    m.impl("approx_bgemm_uint8", &approx_bgemm_uint8);
    m.impl("approx_bgemm_int8", &approx_bgemm_int8);
}



} // namespace end