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
    if constexpr (Signed) {
        return static_cast<int>(value) + 128;
    } else {
        return static_cast<int>(value);
    }
}

// One thread computes one grad_X element and serially reduces over O:
//   grad_X[n,k,l] = sum_o dY[n,o,l] * dx_lut[X[n,k,l]][W[o,k]]
template <typename scalar_t, bool Signed>
__global__ void bgemm_custom_grad_x_naive_kernel(
    const scalar_t* __restrict__ X,
    const scalar_t* __restrict__ W,
    const float* __restrict__ dY,
    const float* __restrict__ dx_lut,
    float* __restrict__ grad_X,
    int64_t N,
    int64_t K,
    int64_t L,
    int64_t O) {
    const int64_t total = N * K * L;
    for (int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x
                        + threadIdx.x;
         linear < total;
         linear += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const int64_t l = linear % L;
        const int64_t nk = linear / L;
        const int64_t k = nk % K;
        const int64_t n = nk / K;
        const int x_idx = operand_index<scalar_t, Signed>(X[linear]);

        float acc = 0.0f;
        for (int64_t o = 0; o < O; ++o) {
            const int w_idx = operand_index<scalar_t, Signed>(W[o * K + k]);
            const int lut_idx = x_idx * 256 + w_idx;
            acc += dY[(n * O + o) * L + l] * dx_lut[lut_idx];
        }
        grad_X[linear] = acc;
    }
}

// One thread computes one grad_W element and serially reduces over N and L:
//   grad_W[o,k] = sum_{n,l} dY[n,o,l] * dw_lut[X[n,k,l]][W[o,k]]
template <typename scalar_t, bool Signed>
__global__ void bgemm_custom_grad_w_naive_kernel(
    const scalar_t* __restrict__ X,
    const scalar_t* __restrict__ W,
    const float* __restrict__ dY,
    const float* __restrict__ dw_lut,
    float* __restrict__ grad_W,
    int64_t N,
    int64_t K,
    int64_t L,
    int64_t O) {
    const int64_t total = O * K;
    for (int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x
                        + threadIdx.x;
         linear < total;
         linear += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const int64_t k = linear % K;
        const int64_t o = linear / K;
        const int w_idx = operand_index<scalar_t, Signed>(W[linear]);

        float acc = 0.0f;
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t l = 0; l < L; ++l) {
                const int64_t x_offset = (n * K + k) * L + l;
                const int x_idx = operand_index<scalar_t, Signed>(X[x_offset]);
                const int lut_idx = x_idx * 256 + w_idx;
                acc += dY[(n * O + o) * L + l] * dw_lut[lut_idx];
            }
        }
        grad_W[linear] = acc;
    }
}

template <typename scalar_t, bool Signed>
std::tuple<torch::Tensor, torch::Tensor> bgemm_custom_grad_naive_cuda(
    const torch::Tensor& X,
    const torch::Tensor& W,
    const torch::Tensor& dY,
    const torch::Tensor& dx_lut,
    const torch::Tensor& dw_lut,
    torch::ScalarType expected_dtype,
    const char* op_name) {
    TORCH_CHECK(X.is_cuda(), op_name, ": X must be a CUDA tensor");
    TORCH_CHECK(W.is_cuda(), op_name, ": W must be a CUDA tensor");
    TORCH_CHECK(dY.is_cuda(), op_name, ": dY must be a CUDA tensor");
    TORCH_CHECK(dx_lut.is_cuda(), op_name, ": dx_lut must be a CUDA tensor");
    TORCH_CHECK(dw_lut.is_cuda(), op_name, ": dw_lut must be a CUDA tensor");
    TORCH_CHECK(X.device() == W.device()
                && X.device() == dY.device()
                && X.device() == dx_lut.device()
                && X.device() == dw_lut.device(),
                op_name, ": all tensors must be on the same CUDA device");
    TORCH_CHECK(X.scalar_type() == expected_dtype,
                op_name, ": X has the wrong dtype");
    TORCH_CHECK(W.scalar_type() == expected_dtype,
                op_name, ": W has the wrong dtype");
    TORCH_CHECK(dY.scalar_type() == torch::kFloat32,
                op_name, ": dY must have dtype torch.float32");
    TORCH_CHECK(dx_lut.scalar_type() == torch::kFloat32,
                op_name, ": dx_lut must have dtype torch.float32");
    TORCH_CHECK(dw_lut.scalar_type() == torch::kFloat32,
                op_name, ": dw_lut must have dtype torch.float32");
    TORCH_CHECK(X.dim() == 3, op_name, ": X must have shape [N, K, L]");
    TORCH_CHECK(W.dim() == 2, op_name, ": W must have shape [O, K]");
    TORCH_CHECK(dY.dim() == 3, op_name, ": dY must have shape [N, O, L]");
    TORCH_CHECK(dx_lut.numel() == 256 * 256,
                op_name, ": dx_lut must have 65536 elements");
    TORCH_CHECK(dw_lut.numel() == 256 * 256,
                op_name, ": dw_lut must have 65536 elements");

    const int64_t N = X.size(0);
    const int64_t K = X.size(1);
    const int64_t L = X.size(2);
    const int64_t O = W.size(0);
    TORCH_CHECK(W.size(1) == K,
                op_name, ": W.shape[1] must equal X.shape[1] (K)");
    TORCH_CHECK(dY.size(0) == N && dY.size(1) == O && dY.size(2) == L,
                op_name, ": dY must have shape [N, O, L] = [",
                N, ", ", O, ", ", L, "]");

    const at::cuda::OptionalCUDAGuard device_guard(device_of(X));
    const auto X_contig = X.contiguous();
    const auto W_contig = W.contiguous();
    const auto dY_contig = dY.contiguous();
    const auto dx_lut_contig = dx_lut.contiguous().view({-1});
    const auto dw_lut_contig = dw_lut.contiguous().view({-1});
    auto grad_X = torch::empty({N, K, L}, dY.options());
    auto grad_W = torch::empty({O, K}, dY.options());
    const auto stream = at::cuda::getCurrentCUDAStream();

    const int64_t grad_X_total = N * K * L;
    if (grad_X_total > 0) {
        const int blocks = static_cast<int>(std::min<int64_t>(
            (grad_X_total + kThreads - 1) / kThreads, kMaxBlocks));
        bgemm_custom_grad_x_naive_kernel<scalar_t, Signed>
            <<<blocks, kThreads, 0, stream>>>(
                X_contig.data_ptr<scalar_t>(), W_contig.data_ptr<scalar_t>(),
                dY_contig.data_ptr<float>(), dx_lut_contig.data_ptr<float>(),
                grad_X.data_ptr<float>(), N, K, L, O);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    const int64_t grad_W_total = O * K;
    if (grad_W_total > 0) {
        const int blocks = static_cast<int>(std::min<int64_t>(
            (grad_W_total + kThreads - 1) / kThreads, kMaxBlocks));
        bgemm_custom_grad_w_naive_kernel<scalar_t, Signed>
            <<<blocks, kThreads, 0, stream>>>(
                X_contig.data_ptr<scalar_t>(), W_contig.data_ptr<scalar_t>(),
                dY_contig.data_ptr<float>(), dw_lut_contig.data_ptr<float>(),
                grad_W.data_ptr<float>(), N, K, L, O);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    return std::make_tuple(grad_X, grad_W);
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> bgemm_custom_grad_int8_naive_cuda(
    const torch::Tensor& X,
    const torch::Tensor& W,
    const torch::Tensor& dY,
    const torch::Tensor& dx_lut,
    const torch::Tensor& dw_lut) {
    return bgemm_custom_grad_naive_cuda<int8_t, true>(
        X, W, dY, dx_lut, dw_lut, torch::kInt8,
        "bgemm_custom_grad_int8_naive");
}

std::tuple<torch::Tensor, torch::Tensor> bgemm_custom_grad_uint8_naive_cuda(
    const torch::Tensor& X,
    const torch::Tensor& W,
    const torch::Tensor& dY,
    const torch::Tensor& dx_lut,
    const torch::Tensor& dw_lut) {
    return bgemm_custom_grad_naive_cuda<uint8_t, false>(
        X, W, dY, dx_lut, dw_lut, torch::kUInt8,
        "bgemm_custom_grad_uint8_naive");
}

TORCH_LIBRARY_FRAGMENT(approxtorch, m) {
    m.def("bgemm_custom_grad_int8_naive(Tensor X, Tensor W, Tensor dY, "
          "Tensor dx_lut, Tensor dw_lut) -> (Tensor grad_X, Tensor grad_W)");
    m.def("bgemm_custom_grad_uint8_naive(Tensor X, Tensor W, Tensor dY, "
          "Tensor dx_lut, Tensor dw_lut) -> (Tensor grad_X, Tensor grad_W)");
}

TORCH_LIBRARY_IMPL(approxtorch, CUDA, m) {
    m.impl("bgemm_custom_grad_int8_naive",
           &bgemm_custom_grad_int8_naive_cuda);
    m.impl("bgemm_custom_grad_uint8_naive",
           &bgemm_custom_grad_uint8_naive_cuda);
}

}  // namespace approxtorch
