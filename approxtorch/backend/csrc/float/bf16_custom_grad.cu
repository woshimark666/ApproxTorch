#include "approx_float_cuda_common.cuh"
#include "approx_mul_bf16.cuh"
#include "bf16_grad_elementwise.cuh"
#include "bf16_grad_scale.cuh"

#include <optional>
#include <tuple>

namespace approxtorch {
namespace {

using namespace float_backend::bf16_detail;
using float_cuda_detail::check_input;
using float_cuda_detail::elementwise_blocks;
using float_cuda_detail::kThreads;

constexpr int kGradientLutSize = 1 << (2 * kFractionBits);
constexpr int kWarpSize = 32;
using Gradients =
    std::tuple<std::optional<torch::Tensor>, std::optional<torch::Tensor>>;

__device__ __forceinline__ bool zero_product(uint16_t x, uint16_t w) {
  return (x & kMagnitudeMask) == 0 || (w & kMagnitudeMask) == 0;
}

__device__ __forceinline__ unsigned int gradient_index(uint16_t x, uint16_t w) {
  return ((x & kFractionMask) << kFractionBits) | (w & kFractionMask);
}

// Scale the LUT partial directly, avoiding a BF16 conversion or a temporary
// 2^128 overflow. Nonzero subnormals/Inf/NaN retain their raw fields, just as
// in the RTL forward: no IEEE special handling or flush-to-zero is added.
__device__ __forceinline__ float scale_partial(float partial, uint16_t bits) {
  return float_backward_detail::scale_partial_fast(partial, bits);
}

#include "bf16_gemm_grad_x.cuh"
#include "bf16_bgemm_grad_x.cuh"
#include "bf16_custom_grad_w.cuh"

template <typename grad_t>
__global__ void approx_mul_bf16_backward_kernel(
    const __nv_bfloat16* __restrict__ X,
    const __nv_bfloat16* __restrict__ W,
    const grad_t* __restrict__ dY,
    const float* __restrict__ dx_lut,
    const float* __restrict__ dw_lut,
    __nv_bfloat16* __restrict__ dX,
    __nv_bfloat16* __restrict__ dW,
    int64_t count) {
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const uint16_t x = __bfloat16_as_ushort(X[i]);
    const uint16_t w = __bfloat16_as_ushort(W[i]);
    // Differentiate the forward's constant-zero branch as zero for both
    // operands, including -0. Do not read either LUT for this branch.
    float dx = 0.0f, dw = 0.0f;
    if (!zero_product(x, w)) {
      const unsigned int index = gradient_index(x, w);
      const float dy = static_cast<float>(dY[i]);
      if (dX) dx = dy * scale_partial(__ldg(dx_lut + index), w);
      if (dW) dw = dy * scale_partial(__ldg(dw_lut + index), x);
    }
    if (dX) dX[i] = __float2bfloat16_rn(dx);
    if (dW) dW[i] = __float2bfloat16_rn(dw);
  }
}

// GEMM:  X[M,K], W[K,O], dY[M,O]. Here batch=1 and L=M.
// BGEMM: X[N,K,L], W[O,K], dY[N,O,L].
// Keep each native layout; no operand transpose or derivative tensor is needed.
template <bool gemm, typename grad_t>
__global__ void bf16_backward_x_kernel(
    const __nv_bfloat16* __restrict__ X,
    const __nv_bfloat16* __restrict__ W,
    const grad_t* __restrict__ dY,
    const float* __restrict__ dx_lut,
    __nv_bfloat16* __restrict__ dX,
    int64_t batch, int64_t K, int64_t L, int64_t O) {
  const int64_t total = batch * K * L;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < total; i += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t k = gemm ? i % K : (i / L) % K;
    const int64_t l = gemm ? i / K : i % L;
    const int64_t n = gemm ? 0 : i / (K * L);
    const uint16_t x = __bfloat16_as_ushort(X[i]);
    float acc = 0.0f;
    if ((x & kMagnitudeMask) != 0) {
      // Same O contraction as the existing custom-gradient/STE BGEMM.
      for (int64_t o = 0; o < O; ++o) {
        const uint16_t w = __bfloat16_as_ushort(W[gemm ? k * O + o : o * K + k]);
        if ((w & kMagnitudeMask) != 0) {
          const float partial = scale_partial(
              __ldg(dx_lut + gradient_index(x, w)), w);
          const float dy = static_cast<float>(dY[gemm ? l * O + o : (n * O + o) * L + l]);
          acc = __fadd_rn(acc, __fmul_rn(dy, partial));
        }
      }
    }
    dX[i] = __float2bfloat16_rn(acc);
  }
}

template <bool gemm, typename grad_t>
__global__ void bf16_backward_w_kernel(
    const __nv_bfloat16* __restrict__ X,
    const __nv_bfloat16* __restrict__ W,
    const grad_t* __restrict__ dY,
    const float* __restrict__ dw_lut,
    __nv_bfloat16* __restrict__ dW,
    int64_t batch, int64_t K, int64_t L, int64_t O) {
  // Follow bgemm_custom_grad_optimize.cu: one block owns one weight and
  // reduces all N*L contributions using warp shuffles and shared warp sums.
  // The LUT column is fixed for that weight; stage its 128 FP32 entries.
  const int64_t i = blockIdx.x;
  const int64_t k = gemm ? i / O : i % K;
  const int64_t o = gemm ? i % O : i / K;
  const uint16_t w = __bfloat16_as_ushort(W[i]);
  if ((w & kMagnitudeMask) == 0) {
    if (threadIdx.x == 0) dW[i] = __ushort_as_bfloat16(0);
    return;
  }
  __shared__ float lut_column[1 << kFractionBits];
  __shared__ float warp_sums[kThreads / kWarpSize];
  if (threadIdx.x < (1 << kFractionBits)) {
    lut_column[threadIdx.x] = __ldg(
        dw_lut + (threadIdx.x << kFractionBits) + (w & kFractionMask));
  }
  __syncthreads();

  float acc = 0.0f;
  for (int64_t nl = threadIdx.x; nl < batch * L; nl += blockDim.x) {
    const int64_t n = gemm ? 0 : nl / L;
    const int64_t l = gemm ? nl : nl % L;
    const uint16_t x = __bfloat16_as_ushort(
        X[gemm ? l * K + k : (n * K + k) * L + l]);
    if ((x & kMagnitudeMask) != 0) {
      const float partial = scale_partial(lut_column[x & kFractionMask], x);
      const float dy = static_cast<float>(dY[gemm ? l * O + o : (n * O + o) * L + l]);
      acc = __fadd_rn(acc, __fmul_rn(dy, partial));
    }
  }
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    acc = __fadd_rn(acc, __shfl_down_sync(0xffffffffu, acc, offset));
  }
  const int lane = threadIdx.x % kWarpSize;
  const int warp = threadIdx.x / kWarpSize;
  if (lane == 0) warp_sums[warp] = acc;
  __syncthreads();
  if (warp == 0) {
    acc = lane < kThreads / kWarpSize ? warp_sums[lane] : 0.0f;
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
      acc = __fadd_rn(acc, __shfl_down_sync(0xffffffffu, acc, offset));
    }
    if (lane == 0) dW[i] = __float2bfloat16_rn(acc);
  }
}

void check_gradient_lut(const torch::Tensor& lut, const torch::Device& device,
                        const char* argument, const char* op_name) {
  check_input(lut, torch::kFloat32, 1, argument, op_name);
  TORCH_CHECK(lut.numel() == kGradientLutSize, op_name, ": ", argument,
              " must have shape [16384]");
  TORCH_CHECK(lut.device() == device, op_name,
              ": all tensors must be on the same CUDA device");
  TORCH_CHECK(!lut.requires_grad(), op_name, ": ", argument,
              " must be fixed (requires_grad=False)");
}

void check_backward(const torch::Tensor& X, const torch::Tensor& W,
                    const torch::Tensor& dY, const torch::Tensor& dx_lut,
                    const torch::Tensor& dw_lut, const char* op_name) {
  check_input(X, torch::kBFloat16, X.dim(), "X", op_name);
  check_input(W, torch::kBFloat16, W.dim(), "W", op_name);
  TORCH_CHECK(dY.is_cuda(), op_name, ": grad_output must be a CUDA tensor");
  TORCH_CHECK(dY.scalar_type() == torch::kBFloat16 ||
              dY.scalar_type() == torch::kFloat32, op_name,
              ": grad_output must have dtype torch.bfloat16 or torch.float32");
  TORCH_CHECK(X.device() == W.device() && X.device() == dY.device(),
              op_name, ": all tensors must be on the same CUDA device");
  check_gradient_lut(dx_lut, X.device(), "grad_x_lut", op_name);
  check_gradient_lut(dw_lut, X.device(), "grad_w_lut", op_name);
}

const __nv_bfloat16* bf16_data(const torch::Tensor& tensor) {
  return reinterpret_cast<const __nv_bfloat16*>(tensor.data_ptr<at::BFloat16>());
}

__nv_bfloat16* gradient_data(const torch::Tensor& tensor) {
  return tensor.defined()
      ? reinterpret_cast<__nv_bfloat16*>(tensor.data_ptr<at::BFloat16>())
      : nullptr;
}

Gradients optional_gradients(const torch::Tensor& dx, const torch::Tensor& dw) {
  return {dx.defined() ? std::make_optional(dx) : std::nullopt,
          dw.defined() ? std::make_optional(dw) : std::nullopt};
}

template <typename grad_t>
void launch_elementwise_backward(
    const torch::Tensor& X, const torch::Tensor& W, const torch::Tensor& dY,
    const torch::Tensor& dx_lut, const torch::Tensor& dw_lut,
    const torch::Tensor& dx, const torch::Tensor& dw) {
  const auto* x_ptr = bf16_data(X);
  const auto* w_ptr = bf16_data(W);
  const auto* dy_ptr = dY.data_ptr<grad_t>();
  auto* dx_ptr = gradient_data(dx);
  auto* dw_ptr = gradient_data(dw);
  const auto* dx_table = dx_lut.data_ptr<float>();
  const auto* dw_table = dw_lut.data_ptr<float>();
  const auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t count = X.numel();
  // Contiguous slices can still start at an odd BF16 storage offset.
  const uintptr_t alignment = reinterpret_cast<uintptr_t>(x_ptr) |
      reinterpret_cast<uintptr_t>(w_ptr) | reinterpret_cast<uintptr_t>(dy_ptr) |
      reinterpret_cast<uintptr_t>(dx_ptr) | reinterpret_cast<uintptr_t>(dw_ptr);
  if (count >= 1048576 && (alignment & 3u) == 0) {
    const int blocks = elementwise_blocks((count + 1) / 2);
    using float_backward_detail::approx_mul_bf16_backward_packed_kernel;
    if (dx.defined() && dw.defined()) {
      approx_mul_bf16_backward_packed_kernel<grad_t, true, true>
          <<<blocks, kThreads, 0, stream>>>(
              x_ptr, w_ptr, dy_ptr, dx_table, dw_table, dx_ptr, dw_ptr, count);
    } else if (dx.defined()) {
      approx_mul_bf16_backward_packed_kernel<grad_t, true, false>
          <<<blocks, kThreads, 0, stream>>>(
              x_ptr, w_ptr, dy_ptr, dx_table, dw_table, dx_ptr, dw_ptr, count);
    } else {
      approx_mul_bf16_backward_packed_kernel<grad_t, false, true>
          <<<blocks, kThreads, 0, stream>>>(
              x_ptr, w_ptr, dy_ptr, dx_table, dw_table, dx_ptr, dw_ptr, count);
    }
  } else {
    approx_mul_bf16_backward_kernel<grad_t>
        <<<elementwise_blocks(count), kThreads, 0, stream>>>(
            x_ptr, w_ptr, dy_ptr, dx_table, dw_table, dx_ptr, dw_ptr, count);
  }
}

Gradients approx_mul_bf16_backward_cuda(
    const torch::Tensor& X_, const torch::Tensor& W_,
    const torch::Tensor& dY_, const torch::Tensor& dx_lut_,
    const torch::Tensor& dw_lut_, bool need_x, bool need_w) {
  constexpr const char* kOpName = "approx_mul_bf16_backward";
  check_backward(X_, W_, dY_, dx_lut_, dw_lut_, kOpName);
  TORCH_CHECK(X_.sizes() == W_.sizes() && X_.sizes() == dY_.sizes(),
              kOpName, ": operands and grad_output must have the same shape");
  const at::cuda::OptionalCUDAGuard device_guard(device_of(X_));
  auto dx = need_x ? torch::empty_like(X_) : torch::Tensor();
  auto dw = need_w ? torch::empty_like(W_) : torch::Tensor();
  if ((!need_x && !need_w) || X_.numel() == 0) return optional_gradients(dx, dw);
  const auto X = X_.resolve_neg(), W = W_.resolve_neg();
  const auto dY = dY_.resolve_neg().contiguous();
  // Normal resident LUT buffers are returned by reference, without copies.
  const auto dx_lut = dx_lut_.resolve_neg(), dw_lut = dw_lut_.resolve_neg();
  if (dY.scalar_type() == torch::kFloat32) {
    launch_elementwise_backward<float>(X, W, dY, dx_lut, dw_lut, dx, dw);
  } else {
    launch_elementwise_backward<at::BFloat16>(X, W, dY, dx_lut, dw_lut, dx, dw);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return optional_gradients(dx, dw);
}

template <bool gemm, typename grad_t>
void launch_weight_direct(
    const torch::Tensor& X, const torch::Tensor& W, const torch::Tensor& dY,
    const torch::Tensor& lut, const torch::Tensor& dw,
    int64_t batch, int64_t K, int64_t L, int64_t O, const char* op_name) {
  bf16_backward_w_kernel<gemm, grad_t>
      <<<float_cuda_detail::checked_grid_x(dw.numel(), op_name),
         kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
          bf16_data(X), bf16_data(W), dY.data_ptr<grad_t>(),
          lut.data_ptr<float>(), gradient_data(dw), batch, K, L, O);
}

template <bool gemm, bool transposed, typename grad_t,
          bool transposed_lut = true, int parts = 1>
void launch_weight_warp(
    const torch::Tensor& X, const torch::Tensor& W, const torch::Tensor& dY,
    const torch::Tensor& lut, const torch::Tensor& dw,
    int64_t batch, int64_t K, int64_t L, int64_t O, const char* op_name) {
  if constexpr (parts < 8) {
    if (batch * L > parts * kWarpSize) {
      launch_weight_warp<gemm, transposed, grad_t, transposed_lut, parts * 2>(
          X, W, dY, lut, dw, batch, K, L, O, op_name);
      return;
    }
  }
  const auto blocks = float_cuda_detail::checked_grid_x(
      (dw.numel() + kThreads / kWarpSize - 1) / (kThreads / kWarpSize), op_name);
  bf16_backward_w_warp_kernel<gemm, transposed, grad_t, transposed_lut, parts>
      <<<blocks, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
          bf16_data(X), bf16_data(W), dY.data_ptr<grad_t>(),
          lut.data_ptr<float>(), gradient_data(dw), batch, K, L, O);
}

template <bool gemm, bool transposed, bool transposed_lut,
          typename grad_t, int outputs>
void launch_weight_tiled(
    const torch::Tensor& X, const torch::Tensor& W, const torch::Tensor& dY,
    const torch::Tensor& lut, const torch::Tensor& dw,
    int64_t batch, int64_t K, int64_t L, int64_t O, const char* op_name) {
  const auto blocks = float_cuda_detail::checked_grid_x(
      K * ((O + outputs - 1) / outputs), op_name);
  bf16_backward_w_tiled_kernel<gemm, transposed, grad_t, outputs, transposed_lut>
      <<<blocks, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
          bf16_data(X), bf16_data(W), dY.data_ptr<grad_t>(),
          lut.data_ptr<float>(), gradient_data(dw), batch, K, L, O);
}

template <bool gemm, typename grad_t>
void launch_weight_backward(
    const torch::Tensor& X, const torch::Tensor& W, const torch::Tensor& dY,
    const torch::Tensor& lut, const torch::Tensor& dw,
    int64_t batch, int64_t K, int64_t L, int64_t O, const char* op_name) {
  if (dw.numel() < 1024) {
    launch_weight_direct<gemm, grad_t>(
        X, W, dY, lut, dw, batch, K, L, O, op_name);
  } else if constexpr (gemm) {
    // Small reductions cannot amortize even the LUT transpose launch.
    if (L <= 64 && dw.numel() <= 4096) {
      launch_weight_warp<true, false, grad_t, false>(
          X, W, dY, lut, dw, batch, K, L, O, op_name);
      return;
    }
    // Scratch copies preserve values and operand order. Coalesced reads pay
    // back their cost on matrix reductions; small M keeps the native operands.
    const auto table = lut.view({128, 128}).transpose(0, 1).contiguous();
    if (L <= 16 || (L <= 64 && dw.numel() <= 8192)) {
      launch_weight_warp<true, false, grad_t>(
          X, W, dY, table, dw, batch, K, L, O, op_name);
    } else {
      const auto xt = X.transpose(0, 1).contiguous();
      const auto gt = dY.transpose(0, 1).contiguous();
      if (dw.numel() >= 4096) {
        launch_weight_warp<true, true, grad_t>(
            xt, W, gt, table, dw, batch, K, L, O, op_name);
      } else {
        launch_weight_tiled<true, true, true, grad_t, 1>(
            xt, W, gt, table, dw, batch, K, L, O, op_name);
      }
    }
  } else {
    const int64_t reduction = batch * L;
    if (reduction <= 128 || dw.numel() >= 8192) {
      const auto table = lut.view({128, 128}).transpose(0, 1).contiguous();
      if (reduction <= 1024) {
        launch_weight_warp<false, false, grad_t>(
            X, W, dY, table, dw, batch, K, L, O, op_name);
      } else {
        launch_weight_tiled<false, false, true, grad_t, 4>(
            X, W, dY, table, dw, batch, K, L, O, op_name);
      }
    } else {
      launch_weight_tiled<false, false, false, grad_t, 1>(
          X, W, dY, lut, dw, batch, K, L, O, op_name);
    }
  }
}

template <bool gemm, typename grad_t>
void launch_matrix_backward(
    const torch::Tensor& X, const torch::Tensor& W, const torch::Tensor& dY,
    const torch::Tensor& dx_lut, const torch::Tensor& dw_lut,
    const torch::Tensor& dx, const torch::Tensor& dw,
    int64_t batch, int64_t K, int64_t L, int64_t O, const char* op_name) {
  const auto stream = at::cuda::getCurrentCUDAStream();
  if (dx.defined() && dx.numel() > 0) {
    if (gemm && dx.numel() <= 8192 && O >= 16) {
      bf16_gemm_backward_x_warp_kernel<grad_t>
          <<<float_cuda_detail::checked_grid_x(dx.numel(), op_name),
             kWarpSize, 0, stream>>>(
              bf16_data(X), bf16_data(W), dY.data_ptr<grad_t>(),
              dx_lut.data_ptr<float>(), gradient_data(dx), L, K, O);
    } else if (gemm && O >= 32) {
      const int64_t blocks = ((L + 15) / 16) * ((K + 15) / 16);
      bf16_gemm_backward_x_tiled_kernel<grad_t>
          <<<float_cuda_detail::checked_grid_x(blocks, op_name),
             kThreads, 0, stream>>>(
              bf16_data(X), bf16_data(W), dY.data_ptr<grad_t>(),
              dx_lut.data_ptr<float>(), gradient_data(dx), L, K, O);
    } else if (!gemm && dx.numel() <= 16384 && L >= 32 && O >= 16) {
      const int64_t kt = (K + 3) / 4, lt = (L + 63) / 64;
      bf16_bgemm_backward_x_tiled_kernel<grad_t>
          <<<float_cuda_detail::checked_grid_x(batch * kt * lt, op_name),
             kThreads, 0, stream>>>(
              bf16_data(X), bf16_data(W), dY.data_ptr<grad_t>(),
              dx_lut.data_ptr<float>(), gradient_data(dx), K, L, O, kt, lt);
    } else if (!gemm && dx.numel() >= 262144 && L >= 64 && O >= 32) {
      if (O <= 128) {
        const int64_t total = batch * K * ((L + 3) / 4);
        bf16_bgemm_backward_x_register_kernel<grad_t, 4>
            <<<elementwise_blocks(total), kThreads, 0, stream>>>(
                bf16_data(X), bf16_data(W), dY.data_ptr<grad_t>(),
                dx_lut.data_ptr<float>(), gradient_data(dx), batch, K, L, O);
      } else {
        const int64_t total = batch * K * ((L + 1) / 2);
        bf16_bgemm_backward_x_register_kernel<grad_t, 2>
            <<<elementwise_blocks(total), kThreads, 0, stream>>>(
                bf16_data(X), bf16_data(W), dY.data_ptr<grad_t>(),
                dx_lut.data_ptr<float>(), gradient_data(dx), batch, K, L, O);
      }
    } else {
      bf16_backward_x_kernel<gemm, grad_t>
          <<<elementwise_blocks(dx.numel()), kThreads, 0, stream>>>(
              bf16_data(X), bf16_data(W), dY.data_ptr<grad_t>(),
              dx_lut.data_ptr<float>(), gradient_data(dx), batch, K, L, O);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  if (dw.defined() && dw.numel() > 0) {
    launch_weight_backward<gemm, grad_t>(
        X, W, dY, dw_lut, dw, batch, K, L, O, op_name);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

template <bool gemm>
Gradients matrix_backward_cuda(
    const torch::Tensor& X_, const torch::Tensor& W_,
    const torch::Tensor& dY_, const torch::Tensor& dx_lut_,
    const torch::Tensor& dw_lut_, bool need_x, bool need_w) {
  const char* op_name = gemm ? "gemm_bf16_backward" : "bgemm_bf16_backward";
  check_backward(X_, W_, dY_, dx_lut_, dw_lut_, op_name);
  TORCH_CHECK(X_.dim() == (gemm ? 2 : 3) && W_.dim() == 2 &&
              dY_.dim() == X_.dim(), op_name, ": incorrect tensor dimensions");
  const int64_t batch = gemm ? 1 : X_.size(0);
  const int64_t K = X_.size(1);
  const int64_t L = gemm ? X_.size(0) : X_.size(2);
  const int64_t O = W_.size(gemm ? 1 : 0);
  TORCH_CHECK(W_.size(gemm ? 0 : 1) == K, op_name,
              ": operand K dimensions must match");
  if constexpr (gemm) {
    TORCH_CHECK(dY_.size(0) == L && dY_.size(1) == O, op_name,
                ": grad_output must have shape [M, N]");
  } else {
    TORCH_CHECK(dY_.size(0) == batch && dY_.size(1) == O && dY_.size(2) == L,
                op_name, ": grad_output must have shape [N, O, L]");
  }
  const at::cuda::OptionalCUDAGuard device_guard(device_of(X_));
  auto dx = need_x ? torch::empty_like(X_) : torch::Tensor();
  auto dw = need_w ? torch::empty_like(W_) : torch::Tensor();
  if (!need_x && !need_w) return optional_gradients(dx, dw);
  const auto X = X_.resolve_neg(), W = W_.resolve_neg();
  const auto dY = dY_.resolve_neg().contiguous();
  const auto dx_lut = dx_lut_.resolve_neg(), dw_lut = dw_lut_.resolve_neg();
  if (dY.scalar_type() == torch::kFloat32) {
    launch_matrix_backward<gemm, float>(
        X, W, dY, dx_lut, dw_lut, dx, dw, batch, K, L, O, op_name);
  } else {
    launch_matrix_backward<gemm, at::BFloat16>(
        X, W, dY, dx_lut, dw_lut, dx, dw, batch, K, L, O, op_name);
  }
  return optional_gradients(dx, dw);
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(approxtorch, m) {
  m.def("approx_mul_bf16_backward(Tensor lhs, Tensor rhs, Tensor grad_output, "
        "Tensor grad_x_lut, Tensor grad_w_lut, bool need_x=True, "
        "bool need_w=True) -> (Tensor?, Tensor?)");
  m.def("gemm_bf16_backward(Tensor A, Tensor B, Tensor grad_output, "
        "Tensor grad_x_lut, Tensor grad_w_lut, bool need_x=True, "
        "bool need_w=True) -> (Tensor?, Tensor?)");
  m.def("bgemm_bf16_backward(Tensor X, Tensor W, Tensor grad_output, "
        "Tensor grad_x_lut, Tensor grad_w_lut, bool need_x=True, "
        "bool need_w=True) -> (Tensor?, Tensor?)");
}

TORCH_LIBRARY_IMPL(approxtorch, CUDA, m) {
  m.impl("approx_mul_bf16_backward", &approx_mul_bf16_backward_cuda);
  m.impl("gemm_bf16_backward", &matrix_backward_cuda<true>);
  m.impl("bgemm_bf16_backward", &matrix_backward_cuda<false>);
}

}  // namespace approxtorch
