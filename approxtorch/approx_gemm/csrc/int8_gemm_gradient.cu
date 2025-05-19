
#include "gemm.cuh"
namespace approxtorch{


__global__ void gradient_compute_kernel(
    const int M,
    const int N,
    const int K,
    const torch::PackedTensorAccessor64<int8_t, 2> A,
    const torch::PackedTensorAccessor64<int8_t, 2> B,
    const torch::PackedTensorAccessor64<float, 2> upstream_gradient,
    const torch::PackedTensorAccessor64<float, 2> gradient_lut,
    torch::PackedTensorAccessor64<float, 2> grad_A,
    torch::PackedTensorAccessor64<float, 2> grad_B
)
{
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    float grad_a = 0;
    float grad_b = 0;
    float upstream_grad = upstream_gradient[row_idx][col_idx];
    if (row_idx < M && col_idx < N){
        for(int k = 0; k < K; ++k){
            int lut_idx = 256 * (int(A[row_idx][k]) + 128) + int(B[k][col_idx]) + 128;
            grad_a = gradient_lut[lut_idx][0];
            grad_b = gradient_lut[lut_idx][1];
            
            atomicAdd(&grad_A[row_idx][k], upstream_grad * grad_a);
            atomicAdd(&grad_B[k][col_idx], upstream_grad * grad_b);
        }


    }
}

//  input is martix A[M, K] int8 and matrix B[K, N] int8, 
// upsteam_gradient[M, N] float, gradient_lut[127*127, 2] float
std::tuple<torch::Tensor, torch::Tensor> gemm_int8_gradient(
    torch::Tensor& A,
    torch::Tensor& B,
    torch::Tensor& upstream_gradient,
    const torch::Tensor& gradient_lut
)
{
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    constexpr uint BM=32, BN =32;
    dim3 const block_dim(BN, BM);
    dim3 const grid_dim((N + BN - 1) / BN, (M + BM - 1) / BM, 1);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(A));

    auto options = torch::TensorOptions().dtype(torch::kFloat).device(A.device());
    torch::Tensor grad_A = torch::zeros({M, K}, options);  // dA
    torch::Tensor grad_B = torch::zeros({K, N}, options);  // dB

    gradient_compute_kernel<<<grid_dim, block_dim>>>(
        M, N, K,
        A.packed_accessor64<int8_t, 2>(),
        B.packed_accessor64<int8_t, 2>(),
        upstream_gradient.packed_accessor64<float, 2>(),
        gradient_lut.packed_accessor64<float, 2>(),
        grad_A.packed_accessor64<float, 2>(),
        grad_B.packed_accessor64<float, 2>()
    );


    cudaDeviceSynchronize();

    // check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }

    return std::make_tuple(grad_A, grad_B);
}

TORCH_LIBRARY_FRAGMENT(approxtorch, m){
    m.def("gemm_int8_gradient(Tensor A, Tensor B, Tensor upstream_grad, Tensor grad_lut) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(approxtorch, CUDA, m){
    m.impl("gemm_int8_gradient", &gemm_int8_gradient);
}


}