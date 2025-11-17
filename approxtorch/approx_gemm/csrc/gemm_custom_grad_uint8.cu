#include "gemm.cuh"
namespace approxtorch{

// lookup the gradlut
__device__ __forceinline__ 
float fetch_grad(int index, const float* __restrict__ grad_lut)
{
    return __ldg(grad_lut + index);
}


__global__ void fetch_custom_grad_kernel(int tensor_size, 
                        const int* __restrict__ index,
                        const float* __restrict__ grad_lut,
                        float* __restrict__ grad_output)
{
    uint thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_idx < tensor_size){
        grad_output[thread_idx] = fetch_grad(index[thread_idx], grad_lut);
    }
}






torch::Tensor fetch_gemm_custom_grad(torch::Tensor& index_tensor, torch::Tensor& grad_lut_tensor)
{
    int M = index_tensor.size(0);
    int N = index_tensor.size(1);
    int K = index_tensor.size(2);
    int tensor_size = M * N * K;

    auto options = torch::TensorOptions().dtype(torch::kFloat).device(index_tensor.device());
    torch::Tensor grad_output = torch::zeros({M, N, K}, options);

    int block_size = 512;
    int grid_size = (tensor_size + block_size - 1) / block_size;

    const at::cuda::OptionalCUDAGuard device_guard(device_of(index_tensor));
    fetch_custom_grad_kernel<<<grid_size, block_size, 0, at::cuda::getCurrentCUDAStream()>>>
        (tensor_size, index_tensor.data_ptr<int>(), grad_lut_tensor.data_ptr<float>(), grad_output.data_ptr<float>());

    return grad_output;
}

TORCH_LIBRARY_FRAGMENT(approxtorch, m){
    m.def("fetch_gemm_custom_grad(Tensor index_tensor, Tensor grad_lut_tensor) -> Tensor");
}

TORCH_LIBRARY_IMPL(approxtorch, CUDA, m){
    m.impl("fetch_gemm_custom_grad", &fetch_gemm_custom_grad);
}

}