#include "gemm.cuh"
namespace approxtorch{


constexpr int TILE_BL = 32;  // BL 方向
constexpr int TILE_O = 32;  // O 方向

// lookup the gradlut
__device__ __forceinline__ 
float fetch_grad(uint index, const float* __restrict__ grad_lut)
{
    return __ldg(grad_lut + index);
}

__global__ void gemm_custom_grad_uint8_kernel_tt(
    uint BL, uint CKK, uint O,
    const uint8_t* __restrict__ A,
    const uint8_t* __restrict__ B,
    const float* __restrict__ grad_lut_dx,
    const float* __restrict__ grad_lut_dy,
    const float* __restrict__ upstream_grad,
    const float scale_A, const float zero_A,
    const float scale_B, const float zero_B,
    float* __restrict__ grad_A,
    float* __restrict__ grad_B
){
    uint tile_i = blockIdx.x * TILE_BL;
    uint tile_o = blockIdx.y * TILE_O;
    uint k = blockIdx.z;

    if (k >= CKK) return;

    uint local_i = threadIdx.x;
    uint local_o = threadIdx.y;

    uint i = tile_i + local_i;
    uint o = tile_o + local_o;
    // define shared memory, and load chunks of A and B
    __shared__ uint8_t As[TILE_BL];
    __shared__ uint8_t Bs[TILE_O];

    // load to shared memory
    if (local_o == 0 && local_o < BL){
        As[local_i] = A[i * CKK + k];
    }
    if (local_i == 0 && o < O){
        Bs[local_o] = B[k * O + o];
    }
    __syncthreads();

    if (i >= BL || o >= O) return;

    uint index = 256*uint(As[local_i]) + uint(Bs[local_o]);
    float dfdx = fetch_grad(index, grad_lut_dx);
    float dfdy = fetch_grad(index, grad_lut_dy);

    // upstream_grad(i,o)
    float g = upstream_grad[i * O + o];
    float gA = g * scale_B * (g - zero_B);
    float gB = g * scale_A * (g - zero_A);

    atomicAdd(&grad_A[i*CKK + k], gA);
    atomicAdd(&grad_B[o * CKK + k], gB);


}


std::tuple<torch::Tensor, torch::Tensor>
gemm_custom_grad_uint8_tt(const torch::Tensor& A, const torch::Tensor& B,
                const torch::Tensor& upstream_grad,
                const torch::Tensor& grad_lut_dx,
                const torch::Tensor& grad_lut_dy,
                const torch::Tensor& scale_A, const torch::Tensor& zero_A,
                const torch::Tensor& scale_B, const torch::Tensor& zero_B)
{
    const at::cuda::OptionalCUDAGuard device_guard(device_of(A));

    uint BL = A.size(0);
    uint CKK = A.size(1);
    uint O = B.size(1); 

    auto options = torch::TensorOptions().dtype(torch::kFloat).device(A.device());
    torch::Tensor grad_A = torch::zeros({BL, CKK}, options);
    torch::Tensor grad_B = torch::zeros({CKK, O}, options);

    const dim3 block(TILE_BL, TILE_O);
    const dim3 grid(
        (BL + TILE_BL - 1) / TILE_BL,
        (O + TILE_O - 1) / TILE_O,
        CKK
    );

    float scale_A_f = scale_A.item<float>();
    float zero_A_f = zero_A.item<float>();
    float scale_B_f = scale_B.item<float>();
    float zero_B_f = zero_B.item<float>();
    gemm_custom_grad_uint8_kernel_tt<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>
        (BL, CKK, O,
        A.data_ptr<uint8_t>(),
        B.data_ptr<uint8_t>(),
        grad_lut_dx.data_ptr<float>(),
        grad_lut_dy.data_ptr<float>(),
        upstream_grad.data_ptr<float>(),
        scale_A_f, zero_A_f, scale_B_f, zero_B_f,
        grad_A.data_ptr<float>(), grad_B.data_ptr<float>());

    // check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }

    return std::make_tuple(grad_A, grad_B);
}

TORCH_LIBRARY_FRAGMENT(approxtorch, m){
    m.def("gemm_custom_grad_uint8_tt(Tensor A, Tensor B, Tensor upstream_grad, Tensor grad_lut_dx, Tensor grad_lut_dy, Tensor scale_A, Tensor zero_A, Tensor scale_B, Tensor zero_B) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(approxtorch, CUDA, m){
    m.impl("gemm_custom_grad_uint8_tt", &gemm_custom_grad_uint8_tt);
}

}