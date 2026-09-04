#include "approx_float_cuda_common.cuh"
#include "approx_mul_fp16.cuh"

namespace approxtorch {
namespace {

using float_cuda_detail::check_input;
using float_cuda_detail::check_lut;
using float_cuda_detail::elementwise_blocks;
using float_cuda_detail::kThreads;

__global__ void approx_mul_fp16_kernel(
    const __half* __restrict__ lhs,
    const __half* __restrict__ rhs,
    const uint32_t* __restrict__ lut,
    __half* __restrict__ output,
    int64_t count) {
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
       index < count;
       index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    output[index] = float_backend::approx_mul_fp16(
        lhs[index], rhs[index], lut);
  }
}

__global__ void gemm_fp16_direct_kernel(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    const uint32_t* __restrict__ lut,
    __half* __restrict__ C,
    int64_t M,
    int64_t N,
    int64_t K) {
  const int64_t total = M * N;
  for (int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                        threadIdx.x;
       linear < total;
       linear += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t row = linear / N;
    const int64_t col = linear - row * N;
    float accumulator = 0.0f;
#pragma unroll 4
    for (int64_t k = 0; k < K; ++k) {
      const __half product = float_backend::approx_mul_fp16(
          A[row * K + k], B[k * N + col], lut);
      accumulator = __fadd_rn(accumulator, __half2float(product));
    }
    C[linear] = __float2half_rn(accumulator);
  }
}

torch::Tensor approx_mul_fp16_cuda(
    const torch::Tensor& lhs,
    const torch::Tensor& rhs,
    const torch::Tensor& lut) {
  constexpr const char* kOpName = "approx_mul_fp16";
  TORCH_CHECK(lhs.is_cuda() && rhs.is_cuda(),
              kOpName, ": lhs and rhs must be CUDA tensors");
  TORCH_CHECK(
      lhs.scalar_type() == torch::kFloat16 &&
          rhs.scalar_type() == torch::kFloat16,
      kOpName, ": lhs and rhs have the wrong dtype");
  TORCH_CHECK(lhs.sizes() == rhs.sizes(),
              kOpName, ": lhs and rhs must have the same shape");
  TORCH_CHECK(lhs.is_contiguous() && rhs.is_contiguous(),
              kOpName, ": lhs and rhs must be contiguous");
  TORCH_CHECK(lhs.device() == rhs.device(),
              kOpName, ": lhs and rhs must be on the same CUDA device");
  check_lut(lut, torch::kUInt32, 1024, lhs.device(), kOpName);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(lhs));
  auto output = torch::empty_like(lhs);
  const int64_t count = lhs.numel();
  if (count == 0) {
    return output;
  }
  approx_mul_fp16_kernel
      <<<elementwise_blocks(count), kThreads, 0,
         at::cuda::getCurrentCUDAStream()>>>(
          reinterpret_cast<const __half*>(lhs.data_ptr<at::Half>()),
          reinterpret_cast<const __half*>(rhs.data_ptr<at::Half>()),
          lut.data_ptr<uint32_t>(),
          reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
          count);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor launch_gemm_fp16(
    const torch::Tensor& A,
    const torch::Tensor& B,
    const torch::Tensor& lut,
    bool optimized,
    const char* op_name) {
  check_input(A, torch::kFloat16, 2, "A", op_name);
  check_input(B, torch::kFloat16, 2, "B", op_name);
  TORCH_CHECK(A.device() == B.device(),
              op_name, ": A and B must be on the same CUDA device");
  TORCH_CHECK(A.size(1) == B.size(0), op_name,
              ": inner dimensions must match, got A[", A.size(0), ", ",
              A.size(1), "] and B[", B.size(0), ", ", B.size(1), "]");
  check_lut(lut, torch::kUInt32, 1024, A.device(), op_name);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(A));
  const int64_t M = A.size(0);
  const int64_t K = A.size(1);
  const int64_t N = B.size(1);
  auto C = torch::empty({M, N}, A.options());
  const int64_t total = M * N;
  if (total == 0) {
    return C;
  }

  const auto* A_ptr =
      reinterpret_cast<const __half*>(A.data_ptr<at::Half>());
  const auto* B_ptr =
      reinterpret_cast<const __half*>(B.data_ptr<at::Half>());
  auto* C_ptr = reinterpret_cast<__half*>(C.data_ptr<at::Half>());
  const auto* lut_ptr = lut.data_ptr<uint32_t>();
  const auto stream = at::cuda::getCurrentCUDAStream();

  // The one-thread-per-output mapping is the fastest measured GEMM path:
  // B loads are coalesced, A is warp-broadcast, and every warp stays on one
  // LUT row. Shared-memory/register tiling remains counterproductive because
  // the ordered FP32 accumulator exposes little cross-K ILP.
  (void)optimized;
  gemm_fp16_direct_kernel
      <<<elementwise_blocks(total), kThreads, 0, stream>>>(
          A_ptr, B_ptr, lut_ptr, C_ptr, M, N, K);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return C;
}

torch::Tensor gemm_fp16_naive_cuda(
    const torch::Tensor& A,
    const torch::Tensor& B,
    const torch::Tensor& lut) {
  return launch_gemm_fp16(A, B, lut, false, "gemm_fp16_naive");
}

torch::Tensor gemm_fp16_cuda(
    const torch::Tensor& A,
    const torch::Tensor& B,
    const torch::Tensor& lut) {
  return launch_gemm_fp16(A, B, lut, true, "gemm_fp16");
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(approxtorch, m) {
  m.def("approx_mul_fp16(Tensor lhs, Tensor rhs, Tensor lut) -> Tensor");
  m.def("gemm_fp16_naive(Tensor A, Tensor B, Tensor lut) -> Tensor");
  m.def("gemm_fp16(Tensor A, Tensor B, Tensor lut) -> Tensor");
}

TORCH_LIBRARY_IMPL(approxtorch, CUDA, m) {
  m.impl("approx_mul_fp16", &approx_mul_fp16_cuda);
  m.impl("gemm_fp16_naive", &gemm_fp16_naive_cuda);
  m.impl("gemm_fp16", &gemm_fp16_cuda);
}

}  // namespace approxtorch
