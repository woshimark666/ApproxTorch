#include <torch/extension.h>
// #include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>
using int4_t = int4;

#ifdef __CUDACC__
namespace approxtorch{
template<const int BM, const int BN, const int BK, const int NUM_THREADS>
__device__ void load_to_shared_memory(
                            const torch::PackedTensorAccessor32<int8_t, 2>  A,
                            const torch::PackedTensorAccessor32<int8_t, 2>  B,
                            int8_t As[BM][BK],
                            int8_t Bs[BK][BN],
                            int thread_block_tile_idx,
                            int thread_linear_idx,
                            int m, int n, int k
                            )
{
    // load data from A on global memory to As on shared memory 
// #pragma unroll
    for(size_t load_idx = 0; load_idx < (BM * BK + NUM_THREADS - 1) / NUM_THREADS; ++load_idx)
    {
        size_t const A_thread_block_tile_row_idx = 
                (thread_linear_idx + load_idx * NUM_THREADS) / BK;
        size_t const A_thread_block_tile_col_idx = 
                (thread_linear_idx + load_idx * NUM_THREADS) % BK;
        
        size_t const A_row_idx = blockIdx.y * BM + A_thread_block_tile_row_idx;
        size_t const A_col_idx = thread_block_tile_idx * BK + A_thread_block_tile_col_idx;

        int8_t val = 0;
        if (A_row_idx < m && A_col_idx < k)
        {
            val = A[A_row_idx][A_col_idx];
        }

        static_assert(BK * BM % NUM_THREADS == 0);

        As[A_thread_block_tile_row_idx][A_thread_block_tile_col_idx] = val;
    }

#pragma unroll
    for(size_t load_idx = 0; load_idx < (BN * BK + NUM_THREADS - 1) / NUM_THREADS; ++load_idx)
    {
        size_t const B_thread_block_tile_row_idx = 
                (thread_linear_idx + load_idx * NUM_THREADS) / BN;
        size_t const B_thread_block_tile_col_idx = 
                (thread_linear_idx + load_idx * NUM_THREADS) % BN;
        size_t const B_row_idx = thread_block_tile_idx * BK + B_thread_block_tile_row_idx;
        size_t const B_col_idx = blockIdx.x * BN + B_thread_block_tile_col_idx;

        int8_t val = 0;
        if (B_row_idx < k && B_col_idx < n)
        {
            val = B[B_row_idx][B_col_idx];
        }

        static_assert(BN * BK % NUM_THREADS == 0);

        Bs[B_thread_block_tile_row_idx][B_thread_block_tile_col_idx] = val;
    }
}

// for int8 big size A
template<const int BM, const int BN, const int BK, const int NUM_THREADS>
__device__ void load_to_shared_memory(
                            const torch::PackedTensorAccessor64<int8_t, 2>  A,
                            const torch::PackedTensorAccessor32<int8_t, 2>  B,
                            int8_t As[BM][BK],
                            int8_t Bs[BK][BN],
                            int thread_block_tile_idx,
                            int thread_linear_idx,
                            int m, int n, int k
                            )
{
    // load data from A on global memory to As on shared memory 
#pragma unroll
    for(size_t load_idx = 0; load_idx < (BM * BK + NUM_THREADS - 1) / NUM_THREADS; ++load_idx)
    {
        size_t const A_thread_block_tile_row_idx = 
                (thread_linear_idx + load_idx * NUM_THREADS) / BK;
        size_t const A_thread_block_tile_col_idx = 
                (thread_linear_idx + load_idx * NUM_THREADS) % BK;
        
        size_t const A_row_idx = blockIdx.y * BM + A_thread_block_tile_row_idx;
        size_t const A_col_idx = thread_block_tile_idx * BK + A_thread_block_tile_col_idx;

        int8_t val = 0;
        if (A_row_idx < m && A_col_idx < k)
        {
            val = A[A_row_idx][A_col_idx];
        }

        static_assert(BK * BM % NUM_THREADS == 0);

        As[A_thread_block_tile_row_idx][A_thread_block_tile_col_idx] = val;
    }

#pragma unroll
    for(size_t load_idx = 0; load_idx < (BN * BK + NUM_THREADS - 1) / NUM_THREADS; ++load_idx)
    {
        size_t const B_thread_block_tile_row_idx = 
                (thread_linear_idx + load_idx * NUM_THREADS) / BN;
        size_t const B_thread_block_tile_col_idx = 
                (thread_linear_idx + load_idx * NUM_THREADS) % BN;
        size_t const B_row_idx = thread_block_tile_idx * BK + B_thread_block_tile_row_idx;
        size_t const B_col_idx = blockIdx.x * BN + B_thread_block_tile_col_idx;

        int8_t val = 0;
        if (B_row_idx < k && B_col_idx < n)
        {
            val = B[B_row_idx][B_col_idx];
        }

        static_assert(BN * BK % NUM_THREADS == 0);

        Bs[B_thread_block_tile_row_idx][B_thread_block_tile_col_idx] = val;
    }
}


template
<size_t BM, size_t BN, size_t BK, size_t NUM_THREADS>
__device__ void load_to_shared_memory_transposed(
                            const torch::PackedTensorAccessor32<int8_t, 2>  A,
                            const torch::PackedTensorAccessor32<int8_t, 2>  B,
                            int8_t As[BK][BM],
                            int8_t Bs[BK][BN],
                            int thread_block_tile_idx,
                            int thread_linear_idx,
                            int m, int n, int k
                            )
{
    // load data from A on global memory to As on shared memory
    //  As is transposed

#pragma unroll
    for(size_t load_idx = 0; load_idx < (BM * BK + NUM_THREADS -1) / NUM_THREADS; ++load_idx)
    {
        size_t const A_thread_block_tile_row_idx = (thread_linear_idx + load_idx * NUM_THREADS) / BK;
        size_t const A_thread_block_tile_col_idx = (thread_linear_idx + load_idx * NUM_THREADS) % BK;
        
        size_t const A_row_idx = blockIdx.y * BM + A_thread_block_tile_row_idx;
        size_t const A_col_idx = thread_block_tile_idx * BK + A_thread_block_tile_col_idx;

        int8_t val = 0;
        if (A_row_idx < m && A_col_idx < k)
        {
            val = A[A_row_idx][A_col_idx];
        } 

        static_assert(BK * BM % NUM_THREADS == 0);

        As[A_thread_block_tile_col_idx][A_thread_block_tile_row_idx] = val;

    }

#pragma unroll
    for (size_t load_idx = 0; load_idx < (BK * BN + NUM_THREADS - 1) / NUM_THREADS; ++load_idx)
    {
        size_t const B_thread_block_tile_row_idx = (thread_linear_idx + load_idx * NUM_THREADS) / BN;
        size_t const B_thread_block_tile_col_idx = (thread_linear_idx + load_idx * NUM_THREADS) % BN;
        size_t const B_row_idx = thread_block_tile_idx * BK + B_thread_block_tile_row_idx;
        size_t const B_col_idx = blockIdx.x * BN + B_thread_block_tile_col_idx;

        int8_t val = 0;
        if (B_row_idx < k && B_col_idx < n)
        {
            val = B[B_row_idx][B_col_idx];
        }

        static_assert(BK * BN % NUM_THREADS == 0);

        Bs[B_thread_block_tile_row_idx][B_thread_block_tile_col_idx] = val;
    }

}


// 重载 for bigger kernel 
template
<size_t BM, size_t BN, size_t BK, size_t NUM_THREADS>
__device__ void load_to_shared_memory_transposed(
                            const torch::PackedTensorAccessor64<int8_t, 2>  A,
                            const torch::PackedTensorAccessor32<int8_t, 2>  B,
                            int8_t As[BK][BM],
                            int8_t Bs[BK][BN],
                            int thread_block_tile_idx,
                            int thread_linear_idx,
                            int m, int n, int k
                            )
{
    // load data from A on global memory to As on shared memory
    //  As is transposed

#pragma unroll
    for(size_t load_idx = 0; load_idx < (BM * BK + NUM_THREADS -1) / NUM_THREADS; ++load_idx)
    {
        size_t const A_thread_block_tile_row_idx = (thread_linear_idx + load_idx * NUM_THREADS) / BK;
        size_t const A_thread_block_tile_col_idx = (thread_linear_idx + load_idx * NUM_THREADS) % BK;
        
        size_t const A_row_idx = blockIdx.y * BM + A_thread_block_tile_row_idx;
        size_t const A_col_idx = thread_block_tile_idx * BK + A_thread_block_tile_col_idx;

        int8_t val = 0;
        if (A_row_idx < m && A_col_idx < k)
        {
            val = A[A_row_idx][A_col_idx];
        } 

        static_assert(BK * BM % NUM_THREADS == 0);

        As[A_thread_block_tile_col_idx][A_thread_block_tile_row_idx] = val;

    }

#pragma unroll
    for (size_t load_idx = 0; load_idx < (BK * BN + NUM_THREADS - 1) / NUM_THREADS; ++load_idx)
    {
        size_t const B_thread_block_tile_row_idx = (thread_linear_idx + load_idx * NUM_THREADS) / BN;
        size_t const B_thread_block_tile_col_idx = (thread_linear_idx + load_idx * NUM_THREADS) % BN;
        size_t const B_row_idx = thread_block_tile_idx * BK + B_thread_block_tile_row_idx;
        size_t const B_col_idx = blockIdx.x * BN + B_thread_block_tile_col_idx;

        int8_t val = 0;
        if (B_row_idx < k && B_col_idx < n)
        {
            val = B[B_row_idx][B_col_idx];
        }

        static_assert(BK * BN % NUM_THREADS == 0);

        Bs[B_thread_block_tile_row_idx][B_thread_block_tile_col_idx] = val;
    }

}


// load to memory for batched gemm
template
<size_t BM, size_t BN, size_t BK, size_t NUM_THREADS>
__device__ void load_to_shared_memory_transposed_batched(
                            const torch::PackedTensorAccessor32<int8_t, 3>  F,
                            const torch::PackedTensorAccessor32<int8_t, 2>  W,
                            int8_t Fs[BK][BN],
                            int8_t Ws[BK][BM],
                            int thread_block_tile_idx,
                            int thread_linear_idx,
                            int m, int n, int k
                            )
{
    // load data from W on global memory to Ws on shared memory
    //  Ws is transposed
#pragma unroll
    for(size_t load_idx = 0; load_idx < (BM * BK + NUM_THREADS -1) / NUM_THREADS; ++load_idx)
    {
        size_t const W_thread_block_tile_row_idx = (thread_linear_idx + load_idx * NUM_THREADS) / BK;
        size_t const W_thread_block_tile_col_idx = (thread_linear_idx + load_idx * NUM_THREADS) % BK;
        
        size_t const W_row_idx = blockIdx.y * BM + W_thread_block_tile_row_idx;
        size_t const W_col_idx = thread_block_tile_idx * BK + W_thread_block_tile_col_idx;

        int8_t val = 0;
        if (W_row_idx < m && W_col_idx < k)
        {
            val = W[W_row_idx][W_col_idx];
        } 

        static_assert(BK * BM % NUM_THREADS == 0);

        Ws[W_thread_block_tile_col_idx][W_thread_block_tile_row_idx] = val;

    }

#pragma unroll
    for (size_t load_idx = 0; load_idx < (BK * BN + NUM_THREADS - 1) / NUM_THREADS; ++load_idx)
    {
        size_t const F_thread_block_tile_row_idx = (thread_linear_idx + load_idx * NUM_THREADS) / BN;
        size_t const F_thread_block_tile_col_idx = (thread_linear_idx + load_idx * NUM_THREADS) % BN;
        size_t const F_row_idx = thread_block_tile_idx * BK + F_thread_block_tile_row_idx;
        size_t const F_col_idx = blockIdx.x * BN + F_thread_block_tile_col_idx;
        size_t const batch_idx = blockIdx.z;
        int8_t val = 0;
        if (F_row_idx < k && F_col_idx < n)
        {
            val = F[batch_idx][F_row_idx][F_col_idx];
        }

        static_assert(BK * BN % NUM_THREADS == 0);

        Fs[F_thread_block_tile_row_idx][F_thread_block_tile_col_idx] = val;
    }

}

// load to memory for batched gemm, 重载 for bigger tensor
template
<size_t BM, size_t BN, size_t BK, size_t NUM_THREADS>
__device__ void load_to_shared_memory_transposed_batched(
                            const torch::PackedTensorAccessor64<int8_t, 3>  F,
                            const torch::PackedTensorAccessor32<int8_t, 2>  W,
                            int8_t Fs[BK][BN],
                            int8_t Ws[BK][BM],
                            int thread_block_tile_idx,
                            int thread_linear_idx,
                            int m, int n, int k
                            )
{
    // load data from W on global memory to Ws on shared memory
    //  Ws is transposed
#pragma unroll
    for(size_t load_idx = 0; load_idx < (BM * BK + NUM_THREADS -1) / NUM_THREADS; ++load_idx)
    {
        size_t const W_thread_block_tile_row_idx = (thread_linear_idx + load_idx * NUM_THREADS) / BK;
        size_t const W_thread_block_tile_col_idx = (thread_linear_idx + load_idx * NUM_THREADS) % BK;
        
        size_t const W_row_idx = blockIdx.y * BM + W_thread_block_tile_row_idx;
        size_t const W_col_idx = thread_block_tile_idx * BK + W_thread_block_tile_col_idx;

        int8_t val = 0;
        if (W_row_idx < m && W_col_idx < k)
        {
            val = W[W_row_idx][W_col_idx];
        } 

        static_assert(BK * BM % NUM_THREADS == 0);

        Ws[W_thread_block_tile_col_idx][W_thread_block_tile_row_idx] = val;

    }

#pragma unroll
    for (size_t load_idx = 0; load_idx < (BK * BN + NUM_THREADS - 1) / NUM_THREADS; ++load_idx)
    {
        size_t const F_thread_block_tile_row_idx = (thread_linear_idx + load_idx * NUM_THREADS) / BN;
        size_t const F_thread_block_tile_col_idx = (thread_linear_idx + load_idx * NUM_THREADS) % BN;
        size_t const F_row_idx = thread_block_tile_idx * BK + F_thread_block_tile_row_idx;
        size_t const F_col_idx = blockIdx.x * BN + F_thread_block_tile_col_idx;
        size_t const batch_idx = blockIdx.z;
        int8_t val = 0;
        if (F_row_idx < k && F_col_idx < n)
        {
            val = F[batch_idx][F_row_idx][F_col_idx];
        }

        static_assert(BK * BN % NUM_THREADS == 0);

        Fs[F_thread_block_tile_row_idx][F_thread_block_tile_col_idx] = val;
    }

}

template
<size_t BM, size_t BN, size_t BK, size_t NUM_THREADS>
__device__ void load_to_shared_memory_transposed_uint8(
                            const torch::PackedTensorAccessor32<uint8_t, 2>  A,
                            const torch::PackedTensorAccessor32<uint8_t, 2>  B,
                            uint8_t As[BK][BM],
                            uint8_t Bs[BK][BN],
                            int thread_block_tile_idx,
                            int thread_linear_idx,
                            int m, int n, int k
                            )
{
    // load data from A on global memory to As on shared memory
    //  As is transposed

#pragma unroll
    for(size_t load_idx = 0; load_idx < (BM * BK + NUM_THREADS -1) / NUM_THREADS; ++load_idx)
    {
        size_t const A_thread_block_tile_row_idx = (thread_linear_idx + load_idx * NUM_THREADS) / BK;
        size_t const A_thread_block_tile_col_idx = (thread_linear_idx + load_idx * NUM_THREADS) % BK;
        
        size_t const A_row_idx = blockIdx.y * BM + A_thread_block_tile_row_idx;
        size_t const A_col_idx = thread_block_tile_idx * BK + A_thread_block_tile_col_idx;

        uint8_t val = 0;
        if (A_row_idx < m && A_col_idx < k)
        {
            val = A[A_row_idx][A_col_idx];
        } 

        static_assert(BK * BM % NUM_THREADS == 0);

        As[A_thread_block_tile_col_idx][A_thread_block_tile_row_idx] = val;

    }

#pragma unroll
    for (size_t load_idx = 0; load_idx < (BK * BN + NUM_THREADS - 1) / NUM_THREADS; ++load_idx)
    {
        size_t const B_thread_block_tile_row_idx = (thread_linear_idx + load_idx * NUM_THREADS) / BN;
        size_t const B_thread_block_tile_col_idx = (thread_linear_idx + load_idx * NUM_THREADS) % BN;
        size_t const B_row_idx = thread_block_tile_idx * BK + B_thread_block_tile_row_idx;
        size_t const B_col_idx = blockIdx.x * BN + B_thread_block_tile_col_idx;

        uint8_t val = 0;
        if (B_row_idx < k && B_col_idx < n)
        {
            val = B[B_row_idx][B_col_idx];
        }

        static_assert(BK * BN % NUM_THREADS == 0);

        Bs[B_thread_block_tile_row_idx][B_thread_block_tile_col_idx] = val;
    }

}

template
<size_t BM, size_t BN, size_t BK, size_t NUM_THREADS>
__device__ void load_to_shared_memory_transposed_uint8(
                            const torch::PackedTensorAccessor64<uint8_t, 2>  A,
                            const torch::PackedTensorAccessor32<uint8_t, 2>  B,
                            uint8_t As[BK][BM],
                            uint8_t Bs[BK][BN],
                            int thread_block_tile_idx,
                            int thread_linear_idx,
                            int m, int n, int k
                            )
{
    // load data from A on global memory to As on shared memory
    //  As is transposed

#pragma unroll
    for(size_t load_idx = 0; load_idx < (BM * BK + NUM_THREADS -1) / NUM_THREADS; ++load_idx)
    {
        size_t const A_thread_block_tile_row_idx = (thread_linear_idx + load_idx * NUM_THREADS) / BK;
        size_t const A_thread_block_tile_col_idx = (thread_linear_idx + load_idx * NUM_THREADS) % BK;
        
        size_t const A_row_idx = blockIdx.y * BM + A_thread_block_tile_row_idx;
        size_t const A_col_idx = thread_block_tile_idx * BK + A_thread_block_tile_col_idx;

        uint8_t val = 0;
        if (A_row_idx < m && A_col_idx < k)
        {
            val = A[A_row_idx][A_col_idx];
        } 

        static_assert(BK * BM % NUM_THREADS == 0);

        As[A_thread_block_tile_col_idx][A_thread_block_tile_row_idx] = val;

    }

#pragma unroll
    for (size_t load_idx = 0; load_idx < (BK * BN + NUM_THREADS - 1) / NUM_THREADS; ++load_idx)
    {
        size_t const B_thread_block_tile_row_idx = (thread_linear_idx + load_idx * NUM_THREADS) / BN;
        size_t const B_thread_block_tile_col_idx = (thread_linear_idx + load_idx * NUM_THREADS) % BN;
        size_t const B_row_idx = thread_block_tile_idx * BK + B_thread_block_tile_row_idx;
        size_t const B_col_idx = blockIdx.x * BN + B_thread_block_tile_col_idx;

        uint8_t val = 0;
        if (B_row_idx < k && B_col_idx < n)
        {
            val = B[B_row_idx][B_col_idx];
        }

        static_assert(BK * BN % NUM_THREADS == 0);

        Bs[B_thread_block_tile_row_idx][B_thread_block_tile_col_idx] = val;
    }

}



template
<size_t BM, size_t BN, size_t BK, size_t NUM_THREADS>
__device__ void load_to_shared_memory_transposed_uint8(
                            const uint8_t* __restrict__ A,
                            const uint8_t* __restrict__ B,
                            uint8_t As[BK][BM],
                            uint8_t Bs[BK][BN],
                            int thread_block_tile_idx,
                            int thread_linear_idx,
                            int m, int n, int k
                            )
{
    // load data from A on global memory to As on shared memory
    //  As is transposed

#pragma unroll
    for(size_t load_idx = 0; load_idx < (BM * BK + NUM_THREADS -1) / NUM_THREADS; ++load_idx)
    {
        size_t const A_thread_block_tile_row_idx = (thread_linear_idx + load_idx * NUM_THREADS) / BK;
        size_t const A_thread_block_tile_col_idx = (thread_linear_idx + load_idx * NUM_THREADS) % BK;
        
        size_t const A_row_idx = blockIdx.y * BM + A_thread_block_tile_row_idx;
        size_t const A_col_idx = thread_block_tile_idx * BK + A_thread_block_tile_col_idx;

        uint8_t val = 0;
        if (A_row_idx < m && A_col_idx < k)
        {
            val = A[A_row_idx * m + A_col_idx];
        } 

        static_assert(BK * BM % NUM_THREADS == 0);

        As[A_thread_block_tile_col_idx][A_thread_block_tile_row_idx] = val;

    }

#pragma unroll
    for (size_t load_idx = 0; load_idx < (BK * BN + NUM_THREADS - 1) / NUM_THREADS; ++load_idx)
    {
        size_t const B_thread_block_tile_row_idx = (thread_linear_idx + load_idx * NUM_THREADS) / BN;
        size_t const B_thread_block_tile_col_idx = (thread_linear_idx + load_idx * NUM_THREADS) % BN;
        size_t const B_row_idx = thread_block_tile_idx * BK + B_thread_block_tile_row_idx;
        size_t const B_col_idx = blockIdx.x * BN + B_thread_block_tile_col_idx;

        uint8_t val = 0;
        if (B_row_idx < k && B_col_idx < n)
        {
            val = B[B_row_idx * k + B_col_idx];
        }

        static_assert(BK * BN % NUM_THREADS == 0);

        Bs[B_thread_block_tile_row_idx][B_thread_block_tile_col_idx] = val;
    }

}


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


// --- 2. 带逻辑屏蔽的计算核心 (Masked Compute) --- for uint8
template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WMITER, const int WNITER, const int WSUBM, const int WSUBN,
          const int TM, const int TN>
__device__ void
processFromSmemMasked_uint8(
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
                // += lut[lut_idx];
          }
        }
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


torch::Tensor gemm_int8(
    torch::Tensor& A,
    torch::Tensor& B,
    const torch::Tensor& lut
);


torch::Tensor batch_gemm_int8(
    torch::Tensor& F,
    torch::Tensor& W,
    const torch::Tensor& lut
);

torch::Tensor naive_gemm_int8(
    torch::Tensor& A,
    torch::Tensor& B,
    const torch::Tensor& lut
);


//  The LRE gradient for uint8 and int8
std::tuple<torch::Tensor, torch::Tensor> 
gemm_int8_gradient( 
    torch::Tensor& A, torch::Tensor& B, 
    torch::Tensor& grad_A_lut, torch::Tensor& grad_B_lut
);

std::tuple<torch::Tensor, torch::Tensor> 
gemm_uint8_gradient( 
    torch::Tensor& A, torch::Tensor& B, 
    torch::Tensor& grad_A_lut, torch::Tensor& grad_B_lut
);

torch::Tensor gemm_uint8(
    torch::Tensor& A,
    torch::Tensor& B,
    const torch::Tensor& lut);

// depthwise gemm for depthwise conv
torch::Tensor depthwise_gemm_int8(
    torch::Tensor& X,
    torch::Tensor& W,
    torch::Tensor& lut
);

std::tuple<torch::Tensor, torch::Tensor> 
depthwise_gemm_int8_gradient( 
    torch::Tensor& X, torch::Tensor& W, 
    torch::Tensor& grad_X_lut, torch::Tensor& grad_W_lut
);

torch::Tensor gemm_int4(
    torch::Tensor& A,
    torch::Tensor& B,
    const torch::Tensor& lut
);

std::tuple<torch::Tensor, torch::Tensor>
gemm_custom_grad_uint8_tt(const torch::Tensor& A, const torch::Tensor& B,
                const torch::Tensor& upstream_grad,
                const torch::Tensor& grad_lut_dx,
                const torch::Tensor& grad_lut_dy,
                const torch::Tensor& scale_A, const torch::Tensor& zero_A,
                const torch::Tensor& scale_B, const torch::Tensor& zero_B);

}



