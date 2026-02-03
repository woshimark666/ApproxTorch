#include <torch/extension.h>
// #include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>



namespace approxtorch{

__global__ void gemm_int8_naive_kernel(int M, int N, int K,
        const int8_t* __restrict__ A,
        const int8_t* __restrict__ B,
        const int32_t* __restrict__ lut,
        int32_t* __restrict__ C)
{
    int64_t row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t col_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (row_idx < M && col_idx < N){
        int32_t acc = 0;
        for (int64_t k = 0; k < K; ++k){
            int64_t lut_idx = 256 * A[row_idx * K + k] + B[k * N + col_idx] + 32896;
            acc += lut[lut_idx];
        }
        C[row_idx * N + col_idx] = acc;
    }
}



torch::Tensor gemm_int8_naive(
    torch::Tensor& A,
    torch::Tensor& B,
    const torch::Tensor& lut
){
    int M = A.size(0);
    int N = B.size(1);
    int K = A.size(1);

    constexpr int BM = 16, BN = 16;
    dim3 const block_dim(BM, BN);
    dim3 const grid_dim((M + BM - 1) / BM, (N + BN - 1) / BN, 1);
    auto options = torch::TensorOptions().dtype(torch::kInt32).device(A.device());
    torch::Tensor C = torch::empty({M, N}, options);
    gemm_int8_naive_kernel<<<grid_dim, block_dim, 0, at::cuda::getCurrentCUDAStream()>>>(
        M, N, K,
        A.data_ptr<int8_t>(),
        B.data_ptr<int8_t>(),
        lut.data_ptr<int32_t>(),
        C.data_ptr<int32_t>()
    );
    return C;
}



__global__ void gemm_uint8_naive_kernel(int M, int N, int K,
    const uint8_t* __restrict__ A,
    const uint8_t* __restrict__ B,
    const int32_t* __restrict__ lut,
    int32_t* __restrict__ C)
{
    int64_t row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t col_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (row_idx < M && col_idx < N){
        int32_t acc = 0;
        for (int64_t k = 0; k < K; ++k){
            int64_t lut_idx = 256 * A[row_idx * K + k] + B[k * N + col_idx];
            acc += lut[lut_idx];
        }
        C[row_idx * N + col_idx] = acc;
    }
}



torch::Tensor gemm_uint8_naive(
    torch::Tensor& A,
    torch::Tensor& B,
    const torch::Tensor& lut
){
    int M = A.size(0);
    int N = B.size(1);
    int K = A.size(1);

    constexpr int BM = 16, BN = 16;
    dim3 const block_dim(BM, BN);
    dim3 const grid_dim((M + BM - 1) / BM, (N + BN - 1) / BN, 1);
    auto options = torch::TensorOptions().dtype(torch::kInt32).device(A.device());
    torch::Tensor C = torch::empty({M, N}, options);
    gemm_uint8_naive_kernel<<<grid_dim, block_dim, 0, at::cuda::getCurrentCUDAStream()>>>(
        M, N, K,
        A.data_ptr<uint8_t>(),
        B.data_ptr<uint8_t>(),
        lut.data_ptr<int32_t>(),
        C.data_ptr<int32_t>()
    );

    return C;
}


TORCH_LIBRARY_FRAGMENT(approxtorch, m){
    m.def("gemm_int8_naive(Tensor A, Tensor B, Tensor lut) -> Tensor");
    m.def("gemm_uint8_naive(Tensor A, Tensor B, Tensor lut) -> Tensor");
}

TORCH_LIBRARY_IMPL(approxtorch, CUDA, m){
    m.impl("gemm_int8_naive", &gemm_int8_naive);
    m.impl("gemm_uint8_naive", &gemm_uint8_naive);
}


}