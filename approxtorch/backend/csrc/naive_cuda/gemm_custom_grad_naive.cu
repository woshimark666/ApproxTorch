#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <algorithm>
#include <cstdint>

namespace approxtorch {
namespace {

constexpr int kThreads = 256;
constexpr int64_t kMaxBlocks = 65535;

template <typename scalar_t, bool Signed>
__device__ __forceinline__ int operand_index(scalar_t value) {
    if constexpr (Signed) return static_cast<int>(value) + 128;
    return static_cast<int>(value);
}

// grad_A[m,k] = sum_n dY[m,n] * dx_lut[A[m,k]][B[k,n]]
template <typename scalar_t, bool Signed>
__global__ void gemm_custom_grad_a_naive_kernel(
    const scalar_t* __restrict__ A, const scalar_t* __restrict__ B,
    const float* __restrict__ dY, const float* __restrict__ dx_lut,
    float* __restrict__ grad_A, int64_t M, int64_t K, int64_t N) {
    const int64_t total = M * K;
    for (int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x
                        + threadIdx.x;
         linear < total;
         linear += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const int64_t k = linear % K;
        const int64_t m = linear / K;
        const int a_idx = operand_index<scalar_t, Signed>(A[linear]);
        float acc = 0.0f;
        for (int64_t n = 0; n < N; ++n) {
            const int b_idx = operand_index<scalar_t, Signed>(B[k * N + n]);
            acc += dY[m * N + n] * dx_lut[a_idx * 256 + b_idx];
        }
        grad_A[linear] = acc;
    }
}

// grad_B[k,n] = sum_m dY[m,n] * dw_lut[A[m,k]][B[k,n]]
template <typename scalar_t, bool Signed>
__global__ void gemm_custom_grad_b_naive_kernel(
    const scalar_t* __restrict__ A, const scalar_t* __restrict__ B,
    const float* __restrict__ dY, const float* __restrict__ dw_lut,
    float* __restrict__ grad_B, int64_t M, int64_t K, int64_t N) {
    const int64_t total = K * N;
    for (int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x
                        + threadIdx.x;
         linear < total;
         linear += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const int64_t n = linear % N;
        const int64_t k = linear / N;
        const int b_idx = operand_index<scalar_t, Signed>(B[linear]);
        float acc = 0.0f;
        for (int64_t m = 0; m < M; ++m) {
            const int a_idx = operand_index<scalar_t, Signed>(A[m * K + k]);
            acc += dY[m * N + n] * dw_lut[a_idx * 256 + b_idx];
        }
        grad_B[linear] = acc;
    }
}

template <typename scalar_t, bool Signed>
std::tuple<torch::Tensor, torch::Tensor> gemm_custom_grad_naive_cuda(
    const torch::Tensor& A, const torch::Tensor& B,
    const torch::Tensor& dY, const torch::Tensor& dx_lut,
    const torch::Tensor& dw_lut, torch::ScalarType expected_dtype,
    const char* op_name) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda() && dY.is_cuda()
                && dx_lut.is_cuda() && dw_lut.is_cuda(),
                op_name, ": all inputs must be CUDA tensors");
    TORCH_CHECK(A.device() == B.device() && A.device() == dY.device()
                && A.device() == dx_lut.device() && A.device() == dw_lut.device(),
                op_name, ": all tensors must be on the same CUDA device");
    TORCH_CHECK(A.scalar_type() == expected_dtype,
                op_name, ": A has the wrong dtype");
    TORCH_CHECK(B.scalar_type() == expected_dtype,
                op_name, ": B has the wrong dtype");
    TORCH_CHECK(dY.scalar_type() == torch::kFloat32,
                op_name, ": dY must have dtype torch.float32");
    TORCH_CHECK(dx_lut.scalar_type() == torch::kFloat32,
                op_name, ": dx_lut must have dtype torch.float32");
    TORCH_CHECK(dw_lut.scalar_type() == torch::kFloat32,
                op_name, ": dw_lut must have dtype torch.float32");
    TORCH_CHECK(A.dim() == 2, op_name, ": A must have shape [M, K]");
    TORCH_CHECK(B.dim() == 2, op_name, ": B must have shape [K, N]");
    TORCH_CHECK(dY.dim() == 2, op_name, ": dY must have shape [M, N]");
    TORCH_CHECK(A.size(1) == B.size(0),
                op_name, ": A.shape[1] must equal B.shape[0] (K)");
    TORCH_CHECK(dY.size(0) == A.size(0) && dY.size(1) == B.size(1),
                op_name, ": dY must have shape [M, N] = [",
                A.size(0), ", ", B.size(1), "]");
    TORCH_CHECK(dx_lut.numel() == 256 * 256,
                op_name, ": dx_lut must have 65536 elements");
    TORCH_CHECK(dw_lut.numel() == 256 * 256,
                op_name, ": dw_lut must have 65536 elements");

    const at::cuda::OptionalCUDAGuard device_guard(device_of(A));
    const auto Ac = A.contiguous(), Bc = B.contiguous(), go = dY.contiguous();
    const auto dx = dx_lut.contiguous().view({-1});
    const auto dw = dw_lut.contiguous().view({-1});
    const int64_t M = Ac.size(0), K = Ac.size(1), N = Bc.size(1);
    auto grad_A = torch::empty({M, K}, dY.options());
    auto grad_B = torch::empty({K, N}, dY.options());
    const auto stream = at::cuda::getCurrentCUDAStream();

    const int64_t total_A = M * K;
    if (total_A > 0) {
        const int blocks = static_cast<int>(std::min<int64_t>(
            (total_A + kThreads - 1) / kThreads, kMaxBlocks));
        gemm_custom_grad_a_naive_kernel<scalar_t, Signed>
            <<<blocks, kThreads, 0, stream>>>(
                Ac.data_ptr<scalar_t>(), Bc.data_ptr<scalar_t>(),
                go.data_ptr<float>(), dx.data_ptr<float>(),
                grad_A.data_ptr<float>(), M, K, N);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    const int64_t total_B = K * N;
    if (total_B > 0) {
        const int blocks = static_cast<int>(std::min<int64_t>(
            (total_B + kThreads - 1) / kThreads, kMaxBlocks));
        gemm_custom_grad_b_naive_kernel<scalar_t, Signed>
            <<<blocks, kThreads, 0, stream>>>(
                Ac.data_ptr<scalar_t>(), Bc.data_ptr<scalar_t>(),
                go.data_ptr<float>(), dw.data_ptr<float>(),
                grad_B.data_ptr<float>(), M, K, N);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return std::make_tuple(grad_A, grad_B);
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> gemm_custom_grad_int8_naive_cuda(
    const torch::Tensor& A, const torch::Tensor& B, const torch::Tensor& dY,
    const torch::Tensor& dx_lut, const torch::Tensor& dw_lut) {
    return gemm_custom_grad_naive_cuda<int8_t, true>(
        A, B, dY, dx_lut, dw_lut, torch::kInt8,
        "gemm_custom_grad_int8_naive");
}

std::tuple<torch::Tensor, torch::Tensor> gemm_custom_grad_uint8_naive_cuda(
    const torch::Tensor& A, const torch::Tensor& B, const torch::Tensor& dY,
    const torch::Tensor& dx_lut, const torch::Tensor& dw_lut) {
    return gemm_custom_grad_naive_cuda<uint8_t, false>(
        A, B, dY, dx_lut, dw_lut, torch::kUInt8,
        "gemm_custom_grad_uint8_naive");
}

TORCH_LIBRARY_FRAGMENT(approxtorch, m) {
    m.def("gemm_custom_grad_int8_naive(Tensor A, Tensor B, Tensor dY, "
          "Tensor dx_lut, Tensor dw_lut) -> (Tensor grad_A, Tensor grad_B)");
    m.def("gemm_custom_grad_uint8_naive(Tensor A, Tensor B, Tensor dY, "
          "Tensor dx_lut, Tensor dw_lut) -> (Tensor grad_A, Tensor grad_B)");
}

TORCH_LIBRARY_IMPL(approxtorch, CUDA, m) {
    m.impl("gemm_custom_grad_int8_naive", &gemm_custom_grad_int8_naive_cuda);
    m.impl("gemm_custom_grad_uint8_naive", &gemm_custom_grad_uint8_naive_cuda);
}

}  // namespace approxtorch
