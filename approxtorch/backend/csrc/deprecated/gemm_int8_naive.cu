#include "gemm.cuh"

namespace approxtorch{

__global__ void naive_gemm_int8_small_kernel(int M, int N, int K,
                const torch::PackedTensorAccessor32<int8_t, 2> A,
                const torch::PackedTensorAccessor32<int8_t, 2> B,
                const torch::PackedTensorAccessor32<int32_t, 1> lut,
                torch::PackedTensorAccessor32<int32_t, 2> C)
{
    int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int col_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (row_idx < M && col_idx < N){
        int32_t acc = 0;
        for (int k = 0; k < K; ++k){
            int lut_idx = 256 * A[row_idx][k] + B[k][col_idx] + 32896;
            acc += lut[lut_idx];
        }
        C[row_idx][col_idx] = acc;
    }
}

__global__ void naive_gemm_int8_big_kernel(int M, int N, int K,
                const torch::PackedTensorAccessor64<int8_t, 2> A,
                const torch::PackedTensorAccessor32<int8_t, 2> B,
                const torch::PackedTensorAccessor32<int32_t, 1> lut,
                torch::PackedTensorAccessor64<int32_t, 2> C)
{
    int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int col_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (row_idx < M && col_idx < N){
        int32_t acc = 0;
        for (int k = 0; k < K; ++k){
            int lut_idx = 256 * A[row_idx][k] + B[k][col_idx] + 32896;
            acc += lut[lut_idx];
        }
        C[row_idx][col_idx] = acc;
    }
}

    torch::Tensor naive_gemm_int8(
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
    int64_t MK = M * K;
    auto options = torch::TensorOptions().dtype(torch::kInt32).device(A.device());
    torch::Tensor C = torch::empty({M, N}, options);
    if (MK <= INT32_MAX)
        naive_gemm_int8_small_kernel<<<grid_dim, block_dim>>>(
            M, N, K,
            A.packed_accessor32<int8_t, 2>(),
            B.packed_accessor32<int8_t, 2>(),
            lut.packed_accessor32<int32_t, 1>(),
            C.packed_accessor32<int32_t, 2>()
        );
    else naive_gemm_int8_big_kernel<<<grid_dim, block_dim>>>(
            M, N, K,
            A.packed_accessor64<int8_t, 2>(),
            B.packed_accessor32<int8_t, 2>(),
            lut.packed_accessor32<int32_t, 1>(),
            C.packed_accessor64<int32_t, 2>());

    cudaDeviceSynchronize();

    // check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }
    return C;
}

TORCH_LIBRARY_FRAGMENT(approxtorch, m){
    m.def("gemm_int8_naive(Tensor A, Tensor B, Tensor lut) -> Tensor");
}

TORCH_LIBRARY_IMPL(approxtorch, CUDA, m){
    m.impl("gemm_int8_naive", &naive_gemm_int8);
}

}