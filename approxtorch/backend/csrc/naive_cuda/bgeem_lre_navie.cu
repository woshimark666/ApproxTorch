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
__device__ __forceinline__ int lre_index(scalar_t value) {
    if constexpr (Signed) {
        return static_cast<int>(value) + 128;
    } else {
        return static_cast<int>(value);
    }
}

// One thread computes one grad_x element and serially reduces over O:
//   grad_x[n,k,l] = sum_o grad_output[n,o,l] * dx_lut[w[o,k]]
template <typename scalar_t, bool Signed>
__global__ void bgemm_lre_grad_x_naive_kernel(
    const float* __restrict__ grad_output,
    const scalar_t* __restrict__ w,
    const float* __restrict__ dx_lut,
    float* __restrict__ grad_x,
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

        float acc = 0.0f;
        for (int64_t o = 0; o < O; ++o) {
            const float gy = grad_output[(n * O + o) * L + l];
            const int idx = lre_index<scalar_t, Signed>(w[o * K + k]);
            acc += gy * dx_lut[idx];
        }
        grad_x[linear] = acc;
    }
}

// One thread computes one grad_w element and serially reduces over N and L:
//   grad_w[o,k] = sum_{n,l} grad_output[n,o,l] * dw_lut[x[n,k,l]]
template <typename scalar_t, bool Signed>
__global__ void bgemm_lre_grad_w_naive_kernel(
    const float* __restrict__ grad_output,
    const scalar_t* __restrict__ x,
    const float* __restrict__ dw_lut,
    float* __restrict__ grad_w,
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

        float acc = 0.0f;
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t l = 0; l < L; ++l) {
                const float gy = grad_output[(n * O + o) * L + l];
                const int idx = lre_index<scalar_t, Signed>(
                    x[(n * K + k) * L + l]);
                acc += gy * dw_lut[idx];
            }
        }
        grad_w[linear] = acc;
    }
}

template <typename scalar_t, bool Signed>
std::tuple<torch::Tensor, torch::Tensor> bgemm_lre_backward_naive_cuda(
    const torch::Tensor& grad_output,
    const torch::Tensor& x,
    const torch::Tensor& w,
    const torch::Tensor& dx_lut,
    const torch::Tensor& dw_lut,
    torch::ScalarType expected_dtype,
    const char* op_name) {
    TORCH_CHECK(grad_output.is_cuda(), op_name, ": grad_output must be CUDA");
    TORCH_CHECK(x.is_cuda(), op_name, ": x must be a CUDA tensor");
    TORCH_CHECK(w.is_cuda(), op_name, ": w must be a CUDA tensor");
    TORCH_CHECK(dx_lut.is_cuda(), op_name, ": dx_lut must be a CUDA tensor");
    TORCH_CHECK(dw_lut.is_cuda(), op_name, ": dw_lut must be a CUDA tensor");
    TORCH_CHECK(grad_output.device() == x.device()
                && x.device() == w.device()
                && x.device() == dx_lut.device()
                && x.device() == dw_lut.device(),
                op_name, ": all tensors must be on the same CUDA device");
    TORCH_CHECK(grad_output.scalar_type() == torch::kFloat32,
                op_name, ": grad_output must have dtype torch.float32");
    TORCH_CHECK(x.scalar_type() == expected_dtype,
                op_name, ": x has the wrong dtype");
    TORCH_CHECK(w.scalar_type() == expected_dtype,
                op_name, ": w has the wrong dtype");
    TORCH_CHECK(dx_lut.scalar_type() == torch::kFloat32,
                op_name, ": dx_lut must have dtype torch.float32");
    TORCH_CHECK(dw_lut.scalar_type() == torch::kFloat32,
                op_name, ": dw_lut must have dtype torch.float32");
    TORCH_CHECK(grad_output.dim() == 3,
                op_name, ": grad_output must have shape [N, O, L]");
    TORCH_CHECK(x.dim() == 3, op_name, ": x must have shape [N, K, L]");
    TORCH_CHECK(w.dim() == 2, op_name, ": w must have shape [O, K]");
    TORCH_CHECK(dx_lut.numel() == 256,
                op_name, ": dx_lut must have 256 elements");
    TORCH_CHECK(dw_lut.numel() == 256,
                op_name, ": dw_lut must have 256 elements");

    const int64_t N = x.size(0);
    const int64_t K = x.size(1);
    const int64_t L = x.size(2);
    const int64_t O = w.size(0);
    TORCH_CHECK(w.size(1) == K,
                op_name, ": w.shape[1] must equal x.shape[1] (K)");
    TORCH_CHECK(grad_output.size(0) == N
                && grad_output.size(1) == O
                && grad_output.size(2) == L,
                op_name, ": grad_output must have shape [N, O, L] = [",
                N, ", ", O, ", ", L, "]");

    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
    const auto gy_contig = grad_output.contiguous();
    const auto x_contig = x.contiguous();
    const auto w_contig = w.contiguous();
    const auto dx_lut_contig = dx_lut.contiguous().view({-1});
    const auto dw_lut_contig = dw_lut.contiguous().view({-1});
    auto grad_x = torch::empty({N, K, L}, grad_output.options());
    auto grad_w = torch::empty({O, K}, grad_output.options());
    const auto stream = at::cuda::getCurrentCUDAStream();

    const int64_t gx_total = N * K * L;
    if (gx_total > 0) {
        const int blocks = static_cast<int>(std::min<int64_t>(
            (gx_total + kThreads - 1) / kThreads, kMaxBlocks));
        bgemm_lre_grad_x_naive_kernel<scalar_t, Signed>
            <<<blocks, kThreads, 0, stream>>>(
                gy_contig.data_ptr<float>(), w_contig.data_ptr<scalar_t>(),
                dx_lut_contig.data_ptr<float>(), grad_x.data_ptr<float>(),
                N, K, L, O);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    const int64_t gw_total = O * K;
    if (gw_total > 0) {
        const int blocks = static_cast<int>(std::min<int64_t>(
            (gw_total + kThreads - 1) / kThreads, kMaxBlocks));
        bgemm_lre_grad_w_naive_kernel<scalar_t, Signed>
            <<<blocks, kThreads, 0, stream>>>(
                gy_contig.data_ptr<float>(), x_contig.data_ptr<scalar_t>(),
                dw_lut_contig.data_ptr<float>(), grad_w.data_ptr<float>(),
                N, K, L, O);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    return std::make_tuple(grad_x, grad_w);
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> bgemm_lre_backward_int8_naive_cuda(
    const torch::Tensor& grad_output,
    const torch::Tensor& x,
    const torch::Tensor& w,
    const torch::Tensor& dx_lut,
    const torch::Tensor& dw_lut) {
    return bgemm_lre_backward_naive_cuda<int8_t, true>(
        grad_output, x, w, dx_lut, dw_lut, torch::kInt8,
        "bgemm_lre_backward_int8_naive");
}

std::tuple<torch::Tensor, torch::Tensor> bgemm_lre_backward_uint8_naive_cuda(
    const torch::Tensor& grad_output,
    const torch::Tensor& x,
    const torch::Tensor& w,
    const torch::Tensor& dx_lut,
    const torch::Tensor& dw_lut) {
    return bgemm_lre_backward_naive_cuda<uint8_t, false>(
        grad_output, x, w, dx_lut, dw_lut, torch::kUInt8,
        "bgemm_lre_backward_uint8_naive");
}

TORCH_LIBRARY_FRAGMENT(approxtorch, m) {
    m.def("bgemm_lre_backward_int8_naive(Tensor grad_output, Tensor x, "
          "Tensor w, Tensor dx_lut, Tensor dw_lut) -> "
          "(Tensor grad_x, Tensor grad_w)");
    m.def("bgemm_lre_backward_uint8_naive(Tensor grad_output, Tensor x, "
          "Tensor w, Tensor dx_lut, Tensor dw_lut) -> "
          "(Tensor grad_x, Tensor grad_w)");
}

TORCH_LIBRARY_IMPL(approxtorch, CUDA, m) {
    m.impl("bgemm_lre_backward_int8_naive",
           &bgemm_lre_backward_int8_naive_cuda);
    m.impl("bgemm_lre_backward_uint8_naive",
           &bgemm_lre_backward_uint8_naive_cuda);
}

}  // namespace approxtorch
