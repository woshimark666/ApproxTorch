#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>


namespace approxtorch{


__global__ void bgemm_custom_grad_dx_uint8_naive_kernel(
    const uint8_t* __restrict__ X,
    const uint8_t* __restrict__ W,
    const float* __restrict__ dY,
    const float* __restrict__ lut_dx,
    float* __restrict__ grad_X,
    int N, int CKK, int O, int L
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * CKK * L;
    if (idx >= total) return;

    int l = idx % L;
    int c = (idx / L) % CKK;
    int n = idx / (L * CKK);

    int x_val = (int)X[idx];

    float sum = 0.0f;
    for (int o = 0; o < O; ++o) {
        int w_val = (int)W[o * CKK + c];
        float dy  = dY[(n * O + o) * L + l];
        float lut = lut_dx[x_val * 256 + w_val];
        sum += dy * lut;
    }

    grad_X[idx] = sum;
}


__global__ void bgemm_custom_grad_dw_uint8_naive_kernel(
    const uint8_t* __restrict__ X,      // (N, CKK, L)
    const uint8_t* __restrict__ W,      // (O, CKK)
    const float*   __restrict__ dY,     // (N, O, L)
    const float*   __restrict__ dw_lut, // (65536,)
    float*         __restrict__ grad_W,  // (O, CKK)
    int N, int CKK, int O, int L
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = O * CKK;
    if (idx >= total) return;

    int c = idx % CKK;
    int o = idx / CKK;

    int w_val = (int)W[idx];

    float sum = 0.0f;
    for (int n = 0; n < N; ++n) {
        for (int l = 0; l < L; ++l) {
            int x_val = (int)X[(n * CKK + c) * L + l];
            float dy  = dY[(n * O + o) * L + l];
            float lut = dw_lut[x_val * 256 + w_val];
            sum += dy * lut;
        }
    }

    grad_W[idx] = sum;
}

// ===========================================================================
// Python binding
// ===========================================================================
std::vector<torch::Tensor> bgemm_custom_grad_uint8_naive(
    torch::Tensor X,       // (N, CKK, L) uint8
    torch::Tensor W,       // (O, CKK) uint8
    torch::Tensor dY,      // (N, O, L) float32
    torch::Tensor dx_lut,  // (256, 256) float32
    torch::Tensor dw_lut   // (256, 256) float32
) {
    TORCH_CHECK(X.is_cuda() && W.is_cuda() && dY.is_cuda());
    TORCH_CHECK(X.dtype() == torch::kUInt8);
    TORCH_CHECK(W.dtype() == torch::kUInt8);
    TORCH_CHECK(dY.dtype() == torch::kFloat32);

    X = X.contiguous();
    W = W.contiguous();
    dY = dY.contiguous();
    dx_lut = dx_lut.contiguous().view({-1});
    dw_lut = dw_lut.contiguous().view({-1});

    int N   = X.size(0);
    int CKK = X.size(1);
    int L   = X.size(2);
    int O   = W.size(0);

    TORCH_CHECK(W.size(1) == CKK);
    TORCH_CHECK(dY.size(0) == N && dY.size(1) == O && dY.size(2) == L);

    auto grad_X = torch::empty({N, CKK, L}, torch::dtype(torch::kFloat32).device(X.device()));
    auto grad_W = torch::empty({O, CKK}, torch::dtype(torch::kFloat32).device(W.device()));

    int threads = 256;

    // Launch grad_X
    {
        int total = N * CKK * L;
        int blocks = (total + threads - 1) / threads;
        bgemm_custom_grad_dx_uint8_naive_kernel<<<blocks, threads>>>(
            X.data_ptr<uint8_t>(),
            W.data_ptr<uint8_t>(),
            dY.data_ptr<float>(),
            dx_lut.data_ptr<float>(),
            grad_X.data_ptr<float>(),
            N, CKK, O, L
        );
    }

    // Launch grad_W
    {
        int total = O * CKK;
        int blocks = (total + threads - 1) / threads;
        bgemm_custom_grad_dw_uint8_naive_kernel<<<blocks, threads>>>(
            X.data_ptr<uint8_t>(),
            W.data_ptr<uint8_t>(),
            dY.data_ptr<float>(),
            dw_lut.data_ptr<float>(),
            grad_W.data_ptr<float>(),
            N, CKK, O, L
        );
    }

    return {grad_X, grad_W};
}

TORCH_LIBRARY_FRAGMENT(approxtorch, m){
    // m.def("im2col_uint8(Tensor input, int k_h, int k_w, int stride_h, int stride_w, int padding_h, int padding_w, int dilation_h, int dilation_w) -> Tensor");
    m.def("bgemm_custom_grad_uint8_naive(Tensor X, Tensor W, Tensor dY, Tensor dx_lut, Tensor dw_lut) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(approxtorch, CUDA, m){
    m.impl("bgemm_custom_grad_uint8_naive", &bgemm_custom_grad_uint8_naive);
}

}