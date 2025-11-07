#include <torch/extension.h>
// #include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>

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


}



