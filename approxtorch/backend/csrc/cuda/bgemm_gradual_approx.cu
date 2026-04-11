#include "gemm_utils.cuh"

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
const int WARPSIZE = 32;
using int4_t = int4;

namespace approxtorch {

// ==========================================================
//  核心计算函数: 使用 float 累加 + alpha 混合
//  val = (1 - alpha) * q_x * q_w + alpha * ApproxLUT(q_x, q_w)
// ==========================================================
template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WMITER, const int WNITER, const int WSUBM, const int WSUBN,
          const int TM, const int TN>
__device__ void processFromSmemMasked_int8_gradual(
    int8_t *regM, int8_t *regN,
    float *threadResults,  // float 累加
    const int8_t *As, const int8_t *Bs,
    const uint warpRow, const uint warpCol,
    const uint threadRowInWarp, const uint threadColInWarp,
    const uint validK,
    const int32_t *__restrict__ lut,
    const float alpha  // 混合系数
) {
    const float one_minus_alpha = 1.0f - alpha;

    for (uint dotIdx = 0; dotIdx < validK; ++dotIdx) {
        // Load A tile registers
#pragma unroll
        for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
#pragma unroll
            for (uint i = 0; i < TM; ++i) {
                uint row = warpRow * WM + wSubRowIdx * WSUBM + threadRowInWarp * TM + i;
                regM[wSubRowIdx * TM + i] = As[dotIdx * BM + row];
            }
        }

        // Load B tile registers
#pragma unroll
        for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
#pragma unroll
            for (uint i = 0; i < TN; ++i) {
                uint col = warpCol * WN + wSubColIdx * WSUBN + threadColInWarp * TN + i;
                regN[wSubColIdx * TN + i] = Bs[dotIdx * BN + col];
            }
        }

        // Compute outer product with alpha blending
#pragma unroll
        for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
#pragma unroll
            for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
#pragma unroll
                for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
#pragma unroll
                    for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
                        int8_t q_x = regM[wSubRowIdx * TM + resIdxM];
                        int8_t q_w = regN[wSubColIdx * TN + resIdxN];

                        // 精确乘法: q_x * q_w (int)
                        float exact_val = (float)((int)q_x * (int)q_w);

                        // 近似查表: ApproxLUT(q_x, q_w)
                        // LUT 索引: 将 int8 [-128,127] 映射到 [0,255]
                        uint8_t uq_x = (uint8_t)(q_x + 128);
                        uint8_t uq_w = (uint8_t)(q_w + 128);
                        float approx_val = (float)lut[uq_x * 256 + uq_w];

                        // 混合: (1-alpha)*exact + alpha*approx
                        float blended = one_minus_alpha * exact_val + alpha * approx_val;

                        uint idx = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                                   (wSubColIdx * TN) + resIdxN;
                        threadResults[idx] += blended;
                    }
                }
            }
        }
    }
}

// ==========================================================
//  Kernel 入口 (Batched Gemm, float 累加, alpha 混合)
// ==========================================================
template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int TM, const int TN, const int NUM_THREADS>
__global__ void approx_bgemm_kernel_gradual_int8(
    uint M, uint N, uint K,
    const int8_t *__restrict__ A_weight,   // Kernel A: User B [O, CKK]
    const int8_t *__restrict__ B_feature,  // Kernel B: User A [Batch, CKK, L]
    const int32_t *__restrict__ lut,
    float *__restrict__ C_batch,           // 改为 float 输出
    const float alpha                      // 混合系数
) {
    const uint batchIdx = blockIdx.z;
    const int8_t *B_ptr = B_feature + batchIdx * (K * N);
    float *C_ptr = C_batch + batchIdx * (M * N);  // float 指针
    const int8_t *A_ptr = A_weight;

    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    const uint warpIdx = threadIdx.x / WARPSIZE;
    const uint warpCol = warpIdx % (BN / WN);
    const uint warpRow = warpIdx / (BN / WN);

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

    // ===== float 累加器 =====
    float threadResults[WMITER * TM * WNITER * TN] = {0.0f};
    int8_t regM[WMITER * TM] = {0};
    int8_t regN[WNITER * TN] = {0};

    // --- Main Loop ---
    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        loadFromGmem_Robust<int8_t, BM, BN, BK, rowStrideA, rowStrideB>(
            M, N, K, bkIdx, cRow, cCol, A_ptr, B_ptr,
            As, Bs, innerRowA, innerColA, innerRowB, innerColB);

        __syncthreads();

        uint validK = K - bkIdx;
        if (validK > BK) validK = BK;

        processFromSmemMasked_int8_gradual
            <BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM, TN>(
                regM, regN, threadResults, As, Bs,
                warpRow, warpCol, threadRowInWarp, threadColInWarp,
                validK, lut, alpha);  // 传入 alpha

        __syncthreads();
    }

    // ==========================================================
    //  Write Back (float4 向量化写回)
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
                uint globalRow = globalBaseRow + subTileRowOffset +
                                 threadRowInWarp * TM + resIdxM;

#pragma unroll
                for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
                    uint globalCol = globalBaseCol + subTileColOffset +
                                     threadColInWarp * TN + resIdxN;

                    uint i = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                             (wSubColIdx * TN) + resIdxN;

                    float4 val;
                    val.x = threadResults[i + 0];
                    val.y = threadResults[i + 1];
                    val.z = threadResults[i + 2];
                    val.w = threadResults[i + 3];

                    // float 输出指针
                    float *dst_ptr = &C_ptr[globalRow * N + globalCol];

                    bool can_use_float4 =
                        (globalRow < M) &&
                        (globalCol + 3 < N) &&
                        (reinterpret_cast<uintptr_t>(dst_ptr) % 16 == 0);

                    if (can_use_float4) {
                        *reinterpret_cast<float4 *>(dst_ptr) = val;
                    } else if (globalRow < M) {
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

// ==========================================================
//  Host 接口
// ==========================================================
torch::Tensor approx_bgemm_int8_gradual(
    const torch::Tensor &user_A,
    const torch::Tensor &user_B,
    const torch::Tensor &lut,
    const double alpha
) {
    // user_A: [Batch, CKK, L] -> Kernel B (Batched)
    // user_B: [O, CKK]        -> Kernel A (Shared)

    const at::cuda::OptionalCUDAGuard device_guard(device_of(user_A));

    uint BatchSize = user_A.size(0);
    uint CKK       = user_A.size(1);
    uint L         = user_A.size(2);
    uint O         = user_B.size(0);

    uint M_gemm = O;
    uint K_gemm = CKK;
    uint N_gemm = L;

    // ===== 输出 float32 =====
    auto tensor_options = torch::TensorOptions().device(user_A.device()).dtype(torch::kFloat32);
    auto C = torch::empty({(long)BatchSize, (long)O, (long)L}, tensor_options);

    constexpr uint BM = 64;
    constexpr uint BN = 64;
    constexpr uint BK = 64;
    constexpr uint WM = 32;
    constexpr uint WN = 16;
    constexpr uint WNITER = 1;
    constexpr uint TM = 4;
    constexpr uint TN = 4;
    constexpr uint NUM_THREADS = 256;

    dim3 grid(CEIL_DIV(N_gemm, BN), CEIL_DIV(M_gemm, BM), BatchSize);
    dim3 block(NUM_THREADS);

    approx_bgemm_kernel_gradual_int8<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS>
        <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            M_gemm, N_gemm, K_gemm,
            user_B.data_ptr<int8_t>(),   // Kernel A <- User B
            user_A.data_ptr<int8_t>(),   // Kernel B <- User A
            lut.data_ptr<int32_t>(),
            C.data_ptr<float>(),         // float 输出
            (float)alpha                 // 传入 alpha
        );

    return C;
}

TORCH_LIBRARY_IMPL(approxtorch, CUDA, m) {
    m.impl("bgemm_int8_gradual", &approx_bgemm_int8_gradual);
}

} // namespace approxtorch
