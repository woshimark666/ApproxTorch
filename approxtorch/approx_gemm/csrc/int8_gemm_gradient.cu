
#include "gemm.cuh"
namespace approxtorch{


// look up helper
__device__ __forceinline__ 
float get_gradient(int8_t index, const float* __restrict__ gradient_lut){
    int idx = (int)index + 128;
    return __ldg(gradient_lut + idx);
}

__global__ void get_gradient_kernel(
    const int tensor_size,
    const int8_t* __restrict__ tensor,
    const float* __restrict__ gradient_lut,
    float* __restrict__ grad_tensor
)
{
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_idx < tensor_size){
        grad_tensor[thread_idx] = get_gradient(tensor[thread_idx], gradient_lut);
    }
}

std::tuple<torch::Tensor, torch::Tensor> 
gemm_int8_gradient( 
    torch::Tensor& A, torch::Tensor& B, 
    torch::Tensor& grad_A_lut, torch::Tensor& grad_B_lut
)
{
    const at::cuda::OptionalCUDAGuard device_guard(device_of(A));
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    constexpr uint block_size = 256;
    
    uint grid_size_A = (K*N + block_size - 1) / block_size;
    uint grid_size_B = (M*K + block_size - 1) / block_size;

    

    auto options = torch::TensorOptions().dtype(torch::kFloat).device(A.device());
    torch::Tensor grad_A = torch::zeros({K, N}, options);  // dA 应该是B的形状
    torch::Tensor grad_B = torch::zeros({M, K}, options);  // dB 应该是A的形状

    const int8_t* B_ptr = B.data_ptr<int8_t>();
    const int8_t* A_ptr = A.data_ptr<int8_t>();
    const float* grad_B_lut_ptr = grad_B_lut.data_ptr<float>();
    const float* grad_A_lut_ptr = grad_A_lut.data_ptr<float>();
    
    // use B to get grad_A 
    get_gradient_kernel
        <<<grid_size_A, block_size, 0, at::cuda::getCurrentCUDAStream()>>>
            (N*K, B_ptr, grad_A_lut_ptr, grad_A.data_ptr<float>());

    // use A to get grad_B
    get_gradient_kernel
        <<<grid_size_B, block_size, 0, at::cuda::getCurrentCUDAStream()>>>
            (M*K, A_ptr, grad_B_lut_ptr, grad_B.data_ptr<float>());
    


    // cudaDeviceSynchronize();

    // check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }

    return std::make_tuple(grad_A, grad_B);
}


std::tuple<torch::Tensor, torch::Tensor> 
depthwise_gemm_int8_gradient( 
    torch::Tensor& X, torch::Tensor& W, 
    torch::Tensor& grad_X_lut, torch::Tensor& grad_W_lut
)
{
    /*
     X shape [B,C,KK,L]
     W shape [C,1,KK]
    */
    const at::cuda::OptionalCUDAGuard device_guard(device_of(X));
    uint B = X.size(0);
    uint C = X.size(1);
    uint KK = X.size(2);
    uint L = X.size(3);
    constexpr uint block_size = 256;
    
    uint grid_size_A = (C*KK + block_size - 1) / block_size;
    uint grid_size_B = (B*C*KK*L + block_size - 1) / block_size;

    

    auto options = torch::TensorOptions().dtype(torch::kFloat).device(X.device());
    torch::Tensor grad_X = torch::zeros({C, 1, KK}, options);  // dX 应该是W的形状
    torch::Tensor grad_W = torch::zeros({B, C, KK, L}, options);  // dW 应该是X的形状

    const int8_t* X_ptr = X.data_ptr<int8_t>();
    const int8_t* W_ptr = W.data_ptr<int8_t>();
    const float* grad_W_lut_ptr = grad_W_lut.data_ptr<float>();
    const float* grad_X_lut_ptr = grad_X_lut.data_ptr<float>();
    
    // use W to get grad_X
    get_gradient_kernel
        <<<grid_size_A, block_size, 0, at::cuda::getCurrentCUDAStream()>>>
            (C*KK, W_ptr, grad_X_lut_ptr, grad_X.data_ptr<float>());

    // use X to get grad_W
    get_gradient_kernel
        <<<grid_size_B, block_size, 0, at::cuda::getCurrentCUDAStream()>>>
            (B*C*KK*L, X_ptr, grad_W_lut_ptr, grad_W.data_ptr<float>());
    


    // cudaDeviceSynchronize();

    // check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }

    return std::make_tuple(grad_X, grad_W);
}



TORCH_LIBRARY_FRAGMENT(approxtorch, m){
    m.def("gemm_int8_gradient(Tensor A, Tensor B, Tensor grad_A_lut, Tensor grad_B_lut) -> (Tensor, Tensor)");
    m.def("depthwise_gemm_int8_gradient(Tensor X, Tensor W, Tensor grad_X_lut, Tensor grad_W_lut) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(approxtorch, CUDA, m){
    m.impl("gemm_int8_gradient", &gemm_int8_gradient);
    m.impl("depthwise_gemm_int8_gradient", &depthwise_gemm_int8_gradient);
}

}
// __global__ void gradient_compute_kernel(
//     const int M,
//     const int N,
//     const int K,
//     const torch::PackedTensorAccessor64<int8_t, 2> A,
//     const torch::PackedTensorAccessor64<int8_t, 2> B,
//     const torch::PackedTensorAccessor64<float, 2> upstream_gradient,
//     const torch::PackedTensorAccessor64<float, 2> gradient_lut,
//     torch::PackedTensorAccessor64<float, 2> grad_A,
//     torch::PackedTensorAccessor64<float, 2> grad_B
// )
// {
//     int row_idx = blockIdx.y * blockDim.y + threadIdx.y;
//     int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
//     float grad_a = 0;
//     float grad_b = 0;
//     float upstream_grad = upstream_gradient[row_idx][col_idx];
//     if (row_idx < M && col_idx < N){
//         for(int k = 0; k < K; ++k){
//             int lut_idx = 256 * (int(A[row_idx][k]) + 128) + int(B[k][col_idx]) + 128;
//             grad_a = gradient_lut[lut_idx][0];
//             grad_b = gradient_lut[lut_idx][1];
            
//             atomicAdd(&grad_A[row_idx][k], upstream_grad * grad_a);
//             atomicAdd(&grad_B[k][col_idx], upstream_grad * grad_b);
//         }


//     }
// }

//  input is martix A[M, K] int8 and matrix B[K, N] int8, 
// upsteam_gradient[M, N] float, gradient_lut[127*127, 2] float
// std::tuple<torch::Tensor, torch::Tensor> gemm_int8_gradient(
//     torch::Tensor& A,
//     torch::Tensor& B,
//     const torch::Tensor& gradient_lut_A,
//     const torch::Tensor& gradient_lut_B
// )
// {
//     int M = A.size(0);
//     int K = A.size(1);
//     int N = B.size(1);
//     constexpr uint BM=32, BN =32;
//     int const block_size = 256;
//     int const grid_size_A = (M*K + block_size - 1) / block_size;
//     int const grid_size_B = (K*N + block_size - 1) / block_size;

//     const at::cuda::OptionalCUDAGuard device_guard(device_of(A));

//     auto options = torch::TensorOptions().dtype(torch::kFloat).device(A.device());
//     torch::Tensor grad_A = torch::zeros({M, K}, options);  // dA
//     torch::Tensor grad_B = torch::zeros({K, N}, options);  // dB

//     gradient_compute_kernel<<<grid_dim, block_dim>>>(
//         M, N, K,
//         A.packed_accessor64<int8_t, 2>(),
//         B.packed_accessor64<int8_t, 2>(),
//         upstream_gradient.packed_accessor64<float, 2>(),
//         gradient_lut.packed_accessor64<float, 2>(),
//         grad_A.packed_accessor64<float, 2>(),
//         grad_B.packed_accessor64<float, 2>()
//     );


//     cudaDeviceSynchronize();

//     // check for errors
//     cudaError_t error = cudaGetLastError();
//     if (error != cudaSuccess) {
//         printf("CUDA error: %s\n", cudaGetErrorString(error));
//     }

//     return std::make_tuple(grad_A, grad_B);
// }

