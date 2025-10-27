
#include "gemm.cuh"
namespace approxtorch{
template <size_t BM, size_t BN, size_t BK, size_t TM, size_t TN>
__global__ void apgemm_int4_small(int M , int N, int K, 
                    const torch::PackedTensorAccessor32<int8_t, 2>  A,
                    const torch::PackedTensorAccessor32<int8_t, 2>  B,
                    const torch::PackedTensorAccessor32<int32_t, 1>  lut,
                    torch::PackedTensorAccessor32<int, 2>  C)
{
    constexpr size_t NUM_THREADS = BM * BN / (TM * TN);
    size_t const thread_linear_idx = threadIdx.y * blockDim.x + threadIdx.x;


    __shared__ int8_t As[BK][BM];
    __shared__ int8_t Bs[BK][BN];

    size_t const num_threads_block_tiles = (K + BK - 1) / BK;

    int32_t C_thread_results[TM][TN] = {0};
    int8_t A_vals[TM] = {0};
    int8_t B_vals[TN] = {0};

    for (size_t thread_block_tile_idx = 0; 
                thread_block_tile_idx < num_threads_block_tiles;
                ++thread_block_tile_idx)
    {
        load_to_shared_memory_transposed<BM, BN, BK, NUM_THREADS>(
            A, B, As, Bs, thread_block_tile_idx, thread_linear_idx, M, N, K
        );
        __syncthreads();

#pragma unroll
        for (size_t k_i = 0; k_i < BK; ++k_i)
        {
            size_t const A_thread_block_tile_row_idx = 
                    thread_linear_idx / (BN / TN) * TM;
            size_t const A_thread_block_tile_col_idx = k_i;

#pragma unroll
            for (size_t thread_tile_row_idx = 0; 
                    thread_tile_row_idx < TM; ++thread_tile_row_idx)
            {
                A_vals[thread_tile_row_idx] = 
                    As[A_thread_block_tile_col_idx][A_thread_block_tile_row_idx + thread_tile_row_idx];
            }

            size_t const B_thread_block_tile_row_idx = k_i;
            size_t const B_thread_block_tile_col_idx = 
                        thread_linear_idx % (BN / TN) * TN;
#pragma unroll
            for (size_t thread_tile_col_idx = 0; 
                    thread_tile_col_idx < TN; ++thread_tile_col_idx)
            {
                B_vals[thread_tile_col_idx] = 
                    Bs[B_thread_block_tile_row_idx][B_thread_block_tile_col_idx + thread_tile_col_idx];
            }

            for (size_t thread_tile_row_idx = 0; 
                    thread_tile_row_idx < TM; ++thread_tile_row_idx)
            {
                for (size_t thread_tile_col_idx = 0; 
                        thread_tile_col_idx < TN; ++thread_tile_col_idx)
                {
                    
                    uint const lut_idx = 16 * A_vals[thread_tile_row_idx] + B_vals[thread_tile_col_idx] + 136;
                    C_thread_results[thread_tile_row_idx][thread_tile_col_idx] +=  lut[lut_idx];   
                }
            }
        }
        __syncthreads();

    }

    for (size_t thread_tile_row_idx = 0; 
            thread_tile_row_idx < TM; ++thread_tile_row_idx)
    {
        for (size_t thread_tile_col_idx = 0; 
                thread_tile_col_idx < TN; ++thread_tile_col_idx)
        {
            size_t const C_row_idx = blockIdx.y * BM + threadIdx.x / (BN / TN) * TM + thread_tile_row_idx;
            size_t const C_col_idx = blockIdx.x * BN + threadIdx.x % (BN / TN) * TN + thread_tile_col_idx;

            if (C_row_idx < M && C_col_idx < N)
            {
                C[C_row_idx][C_col_idx] = C_thread_results[thread_tile_row_idx][thread_tile_col_idx];
            }
        }
    }

}


template <size_t BM, size_t BN, size_t BK, size_t TM, size_t TN>
__global__ void apgemm_int4_big(int M , int N, int K, 
                    const torch::PackedTensorAccessor64<int8_t, 2>  A,
                    const torch::PackedTensorAccessor32<int8_t, 2>  B,
                    const torch::PackedTensorAccessor32<int32_t, 1>  lut,
                    torch::PackedTensorAccessor64<int, 2>  C)
{
    constexpr size_t NUM_THREADS = BM * BN / (TM * TN);
    size_t const thread_linear_idx = threadIdx.y * blockDim.x + threadIdx.x;


    __shared__ int8_t As[BK][BM];
    __shared__ int8_t Bs[BK][BN];

    size_t const num_threads_block_tiles = (K + BK - 1) / BK;

    int32_t C_thread_results[TM][TN] = {0};
    int8_t A_vals[TM] = {0};
    int8_t B_vals[TN] = {0};

    for (size_t thread_block_tile_idx = 0; 
                thread_block_tile_idx < num_threads_block_tiles;
                ++thread_block_tile_idx)
    {
        load_to_shared_memory_transposed<BM, BN, BK, NUM_THREADS>(
            A, B, As, Bs, thread_block_tile_idx, thread_linear_idx, M, N, K
        );
        __syncthreads();

#pragma unroll
        for (size_t k_i = 0; k_i < BK; ++k_i)
        {
            size_t const A_thread_block_tile_row_idx = 
                    thread_linear_idx / (BN / TN) * TM;
            size_t const A_thread_block_tile_col_idx = k_i;

#pragma unroll
            for (size_t thread_tile_row_idx = 0; 
                    thread_tile_row_idx < TM; ++thread_tile_row_idx)
            {
                A_vals[thread_tile_row_idx] = 
                    As[A_thread_block_tile_col_idx][A_thread_block_tile_row_idx + thread_tile_row_idx];
            }

            size_t const B_thread_block_tile_row_idx = k_i;
            size_t const B_thread_block_tile_col_idx = 
                        thread_linear_idx % (BN / TN) * TN;
#pragma unroll
            for (size_t thread_tile_col_idx = 0; 
                    thread_tile_col_idx < TN; ++thread_tile_col_idx)
            {
                B_vals[thread_tile_col_idx] = 
                    Bs[B_thread_block_tile_row_idx][B_thread_block_tile_col_idx + thread_tile_col_idx];
            }

            for (size_t thread_tile_row_idx = 0; 
                    thread_tile_row_idx < TM; ++thread_tile_row_idx)
            {
                for (size_t thread_tile_col_idx = 0; 
                        thread_tile_col_idx < TN; ++thread_tile_col_idx)
                {
                    
                    uint const lut_idx = 16 * A_vals[thread_tile_row_idx] + B_vals[thread_tile_col_idx] + 136;
                    C_thread_results[thread_tile_row_idx][thread_tile_col_idx] +=  lut[lut_idx];   
                }
            }
        }
        __syncthreads();

    }

    for (size_t thread_tile_row_idx = 0; 
            thread_tile_row_idx < TM; ++thread_tile_row_idx)
    {
        for (size_t thread_tile_col_idx = 0; 
                thread_tile_col_idx < TN; ++thread_tile_col_idx)
        {
            size_t const C_row_idx = blockIdx.y * BM + threadIdx.x / (BN / TN) * TM + thread_tile_row_idx;
            size_t const C_col_idx = blockIdx.x * BN + threadIdx.x % (BN / TN) * TN + thread_tile_col_idx;

            if (C_row_idx < M && C_col_idx < N)
            {
                C[C_row_idx][C_col_idx] = C_thread_results[thread_tile_row_idx][thread_tile_col_idx];
            }
        }
    }

}


torch::Tensor gemm_int4(
    torch::Tensor& A,
    torch::Tensor& B,
    const torch::Tensor& lut
)
{
    constexpr uint BM=64, BN=64, BK=16, TM=4, TN=4;
    constexpr uint NUM_THREADS_PER_BLOCK = BM * BN / (TM * TN);
    
    // auto which_gpu = A.device().index();
    // auto options = torch::TensorOptions().dtype(torch::kInt32).device(A.device());

    const at::cuda::OptionalCUDAGuard device_guard(device_of(A));
    auto options = torch::TensorOptions().dtype(torch::kInt32).device(A.device());

    
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    torch::Tensor C = torch::empty({M, N}, options);
    int64_t MK = M * K; 

    dim3 const block_dim(NUM_THREADS_PER_BLOCK, 1, 1);
    dim3 const grid_dim((N + BN - 1) / BN, (M + BM - 1) / BM, 1);
    if( MK < INT32_MAX)
        apgemm_int4_small<BM, BN, BK, TM, TN><<<grid_dim, block_dim,0, at::cuda::getCurrentCUDAStream()>>>(
            M, N, K,
            A.packed_accessor32<int8_t, 2>(),
            B.packed_accessor32<int8_t, 2>(),
            lut.packed_accessor32<int32_t, 1>(),
            C.packed_accessor32<int, 2>()
        );
    else apgemm_int4_big<BM, BN, BK, TM, TN><<<grid_dim, block_dim,0, at::cuda::getCurrentCUDAStream()>>>(
            M, N, K,
            A.packed_accessor64<int8_t, 2>(),
            B.packed_accessor32<int8_t, 2>(),
            lut.packed_accessor32<int32_t, 1>(),
            C.packed_accessor64<int, 2>()
        );

    // cudaDeviceSynchronize();
    // Check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }
    return C;
}


TORCH_LIBRARY_FRAGMENT(approxtorch, m){
    m.def("gemm_int4(Tensor A, Tensor B, Tensor lut) -> Tensor");
}
TORCH_LIBRARY_IMPL(approxtorch, CUDA, m){
    m.impl("gemm_int4", &gemm_int4);
}


}