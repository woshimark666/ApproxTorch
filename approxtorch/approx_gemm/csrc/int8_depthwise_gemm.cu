#include "gemm.cuh"

namespace approxtorch{

__device__ __forceinline__ 
int get_lut(int index, const int* __restrict__ lut)
{
    return __ldg(lut + index);
}

template<int KK_STATIC>
__global__ void depthwise_gemm_int8_kernel_static(
    const int8_t* __restrict__ X,
    const int8_t* __restrict__ W,
    const int* __restrict__ lut,
    int* __restrict__ Y,
    uint B, uint C, uint KK, uint L
)
/*
    input tensor X [B, C, KK, L]
    input tensor W [C, 1, KK]
    input tensor lut [256*256]
    output tensor Y [B, C, 1, L]
*/
{
    uint b = blockIdx.y;
    uint c = blockIdx.x;
    uint tid = threadIdx.x;

    // base address
    const int8_t* w = W + c*KK;
    const int8_t* x = X + b*C*KK*L + c*KK*L;
    int* y = Y + b*C*L + c*L;

    __shared__ int8_t w_sh[KK_STATIC];

    for (int kk = tid; kk < KK_STATIC; kk += blockDim.x)
    {
        w_sh[kk] = w[kk];
    }

    __syncthreads();

    for (int l = tid; l < L; l += blockDim.x)
    {
        int acc = 0;
        for (int kk = 0; kk < KK_STATIC; ++kk)
        {
            int index = 256 * (x[kk*L + l] + 128) + w_sh[kk] + 128;
            acc += get_lut(index, lut);
        }
        y[l] = acc;
    }
}



torch::Tensor depthwise_gemm_int8(
    torch::Tensor& X,
    torch::Tensor& W,
    torch::Tensor& lut
)
/*
    input tensor X [B, C, KK, L]
    input tensor W [C, 1, KK]
    input tensor lut [256*256]
    output tensor should be [B, C, 1, L]
*/
{
    uint B = X.size(0);
    uint C = X.size(1);
    uint KK = X.size(2);
    uint L = X.size(3);

    dim3 grid(C, B);
    int threads = 256;

    const at::cuda::OptionalCUDAGuard device_guard(device_of(X));
    auto options = torch::TensorOptions().dtype(torch::kInt32).device(X.device());

    torch::Tensor Y = torch::empty({B, C, 1, L}, options);

    auto stream = at::cuda::getCurrentCUDAStream();

    const int8_t* X_ptr = X.data_ptr<int8_t>();
    const int8_t* W_ptr = W.data_ptr<int8_t>();
    const int* lut_ptr = lut.data_ptr<int>();
    int* Y_ptr = Y.data_ptr<int>();

    switch (KK)
    {
        case 1: depthwise_gemm_int8_kernel_static<1><<<grid, threads, 0, stream>>>(X_ptr, W_ptr, lut_ptr, Y_ptr, B, C, KK, L); break;
        case 4: depthwise_gemm_int8_kernel_static<4><<<grid, threads, 0, stream>>>(X_ptr, W_ptr, lut_ptr, Y_ptr, B, C, KK, L); break;
        case 9: depthwise_gemm_int8_kernel_static<9><<<grid, threads, 0, stream>>>(X_ptr, W_ptr, lut_ptr, Y_ptr, B, C, KK, L); break;
        case 16: depthwise_gemm_int8_kernel_static<16><<<grid, threads, 0, stream>>>(X_ptr, W_ptr, lut_ptr, Y_ptr, B, C, KK, L); break;
        case 25: depthwise_gemm_int8_kernel_static<25><<<grid, threads, 0, stream>>>(X_ptr, W_ptr, lut_ptr, Y_ptr, B, C, KK, L); break;
        case 36: depthwise_gemm_int8_kernel_static<36><<<grid, threads, 0, stream>>>(X_ptr, W_ptr, lut_ptr, Y_ptr, B, C, KK, L); break;
        case 49: depthwise_gemm_int8_kernel_static<49><<<grid, threads, 0, stream>>>(X_ptr, W_ptr, lut_ptr, Y_ptr, B, C, KK, L); break;
        default:
            printf("Unsupported KK: %d\n", KK);
            assert(false);
            break;
    }

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }

    return Y;
}

TORCH_LIBRARY_FRAGMENT(approxtorch, m){
    m.def("depthwise_gemm_int8(Tensor X, Tensor W, Tensor lut) -> Tensor");
}

TORCH_LIBRARY_IMPL(approxtorch, CUDA, m){
    m.impl("depthwise_gemm_int8", &depthwise_gemm_int8);
}

}