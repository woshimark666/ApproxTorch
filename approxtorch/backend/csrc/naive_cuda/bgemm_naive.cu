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
__device__ __forceinline__ int lut_index(scalar_t lhs, scalar_t rhs) {
    if constexpr (Signed) {
        return (static_cast<int>(lhs) + 128) * 256
             + (static_cast<int>(rhs) + 128);
    } else {
        return static_cast<int>(lhs) * 256 + static_cast<int>(rhs);
    }
}

// Deliberately naive reference kernel:
//   A: [batch, K, L], B: [O, K], C: [batch, O, L]
//   C[b, o, l] = sum_k LUT[A[b, k, l]][B[o, k]]
// The first API operand is always the LUT row operand, matching
// approxtorch/nn/bgemm_int8.py and the bgemm_fake_*_claude family.
template <typename scalar_t, bool Signed>
__global__ void bgemm_lut_naive_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    const int32_t* __restrict__ lut,
    int32_t* __restrict__ C,
    int64_t batch,
    int64_t O,
    int64_t L,
    int64_t K) {
    const int64_t total = batch * O * L;
    for (int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x
                        + threadIdx.x;
         linear < total;
         linear += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const int64_t l = linear % L;
        const int64_t bo = linear / L;
        const int64_t o = bo % O;
        const int64_t b = bo / O;
        int32_t acc = 0;
        for (int64_t k = 0; k < K; ++k) {
            const scalar_t lhs = A[(b * K + k) * L + l];
            const scalar_t rhs = B[o * K + k];
            acc += lut[lut_index<scalar_t, Signed>(lhs, rhs)];
        }
        C[linear] = acc;
    }
}

template <typename scalar_t, bool Signed>
torch::Tensor bgemm_lut_naive_cuda(
    const torch::Tensor& A,
    const torch::Tensor& B,
    const torch::Tensor& lut,
    torch::ScalarType expected_dtype,
    const char* op_name) {
    TORCH_CHECK(A.is_cuda(), op_name, ": A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), op_name, ": B must be a CUDA tensor");
    TORCH_CHECK(lut.is_cuda(), op_name, ": lut must be a CUDA tensor");
    TORCH_CHECK(A.device() == B.device() && A.device() == lut.device(),
                op_name, ": A, B, and lut must be on the same CUDA device");
    TORCH_CHECK(A.scalar_type() == expected_dtype,
                op_name, ": A has the wrong dtype");
    TORCH_CHECK(B.scalar_type() == expected_dtype,
                op_name, ": B has the wrong dtype");
    TORCH_CHECK(lut.scalar_type() == torch::kInt32,
                op_name, ": lut must have dtype torch.int32");
    TORCH_CHECK(A.dim() == 3, op_name, ": A must have shape [batch, K, L]");
    TORCH_CHECK(B.dim() == 2, op_name, ": B must have shape [O, K]");
    TORCH_CHECK(A.size(1) == B.size(1),
                op_name, ": K dimensions must match, got A.shape[1]=",
                A.size(1), " and B.shape[1]=", B.size(1));
    TORCH_CHECK(lut.numel() == 256 * 256,
                op_name, ": lut must contain exactly 65536 elements");

    const at::cuda::OptionalCUDAGuard device_guard(device_of(A));
    const auto A_contig = A.contiguous();
    const auto B_contig = B.contiguous();
    const auto lut_contig = lut.contiguous().view({-1});
    const int64_t batch = A_contig.size(0);
    const int64_t K = A_contig.size(1);
    const int64_t L = A_contig.size(2);
    const int64_t O = B_contig.size(0);
    auto C = torch::empty({batch, O, L}, A.options().dtype(torch::kInt32));

    const int64_t total = batch * O * L;
    if (total == 0) {
        return C;
    }
    const int blocks = static_cast<int>(
        std::min<int64_t>((total + kThreads - 1) / kThreads, kMaxBlocks));
    bgemm_lut_naive_kernel<scalar_t, Signed>
        <<<blocks, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
            A_contig.data_ptr<scalar_t>(),
            B_contig.data_ptr<scalar_t>(),
            lut_contig.data_ptr<int32_t>(),
            C.data_ptr<int32_t>(), batch, O, L, K);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return C;
}

}  // namespace

torch::Tensor bgemm_int8_naive_cuda(
    const torch::Tensor& A,
    const torch::Tensor& B,
    const torch::Tensor& lut) {
    return bgemm_lut_naive_cuda<int8_t, true>(
        A, B, lut, torch::kInt8, "bgemm_int8_naive");
}

torch::Tensor bgemm_uint8_naive_cuda(
    const torch::Tensor& A,
    const torch::Tensor& B,
    const torch::Tensor& lut) {
    return bgemm_lut_naive_cuda<uint8_t, false>(
        A, B, lut, torch::kUInt8, "bgemm_uint8_naive");
}

TORCH_LIBRARY_FRAGMENT(approxtorch, m) {
    m.def("bgemm_int8_naive(Tensor A, Tensor B, Tensor lut) -> Tensor");
    m.def("bgemm_uint8_naive(Tensor A, Tensor B, Tensor lut) -> Tensor");
}

TORCH_LIBRARY_IMPL(approxtorch, CUDA, m) {
    m.impl("bgemm_int8_naive", &bgemm_int8_naive_cuda);
    m.impl("bgemm_uint8_naive", &bgemm_uint8_naive_cuda);
}

}  // namespace approxtorch
