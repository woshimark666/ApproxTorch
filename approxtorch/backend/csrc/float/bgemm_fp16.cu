#include "approx_float_cuda_common.cuh"
#include "approx_mul_fp16.cuh"

namespace approxtorch {
namespace {

using float_cuda_detail::check_input;
using float_cuda_detail::check_lut;
using float_cuda_detail::checked_grid_x;
using float_cuda_detail::elementwise_blocks;
using float_cuda_detail::kThreads;

constexpr int kKTile = 32;
constexpr int kWarpSize = 32;
constexpr int kLengthTile = 8;
constexpr int kSharedWPadding = 2;

__device__ __forceinline__ __half fp16_zero() {
  return __ushort_as_half(0);
}

__global__ void bgemm_fp16_naive_kernel(
    const __half* __restrict__ X,
    const __half* __restrict__ W,
    const uint16_t* __restrict__ lut,
    __half* __restrict__ Y,
    int64_t batch,
    int64_t O,
    int64_t L,
    int64_t K) {
  const int64_t total = batch * O * L;
  for (int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                        threadIdx.x;
       linear < total;
       linear += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t l = linear % L;
    const int64_t batch_output = linear / L;
    const int64_t output = batch_output % O;
    const int64_t n = batch_output / O;
    float accumulator = 0.0f;
    for (int64_t k = 0; k < K; ++k) {
      // X remains the LUT row operand; approximate LUTs need not be symmetric.
      const __half product = float_backend::approx_mul_fp16(
          X[(n * K + k) * L + l], W[output * K + k], lut);
      accumulator = __fadd_rn(accumulator, __half2float(product));
    }
    Y[linear] = __float2half_rn(accumulator);
  }
}

// One warp fixes an L position, so its lanes share the X operand and one LUT
// row while spanning adjacent output channels. W is loaded coalesced in its
// native [O, K] layout, then transposed into padded shared memory. The padding
// removes the bank conflict caused by warp lanes reading different O values
// at the same K.
template <int outputs_per_lane>
__global__ __launch_bounds__(kThreads)
void bgemm_fp16_tiled_kernel(
    const __half* __restrict__ X,
    const __half* __restrict__ W,
    const uint16_t* __restrict__ lut,
    __half* __restrict__ Y,
    int64_t O,
    int64_t L,
    int64_t K,
    int64_t length_tiles,
    int64_t output_tiles) {
  constexpr int kOutputTile = kWarpSize * outputs_per_lane;
  constexpr int kSharedWStride = kOutputTile + kSharedWPadding;
  __shared__ __align__(16) __half shared_x[kKTile * kLengthTile];
  __shared__ __align__(16) __half shared_w[kKTile * kSharedWStride];

  const int thread = threadIdx.x;
  const int warp = thread / kWarpSize;
  const int lane = thread & (kWarpSize - 1);
  const int64_t tiles_per_batch = output_tiles * length_tiles;
  const int64_t n = static_cast<int64_t>(blockIdx.x) / tiles_per_batch;
  const int64_t batch_tile =
      static_cast<int64_t>(blockIdx.x) - n * tiles_per_batch;
  const int64_t output_tile = batch_tile / length_tiles;
  const int64_t length_tile =
      batch_tile - output_tile * length_tiles;
  const int64_t first_output = output_tile * kOutputTile;
  const int64_t first_l = length_tile * kLengthTile;
  const int64_t global_l = first_l + warp;

  float accumulators[outputs_per_lane];
#pragma unroll
  for (int item = 0; item < outputs_per_lane; ++item) {
    accumulators[item] = 0.0f;
  }

  int64_t tile_k = 0;
  for (; tile_k + kKTile <= K; tile_k += kKTile) {
    for (int index = thread; index < kKTile * kLengthTile;
         index += blockDim.x) {
      const int inner = index / kLengthTile;
      const int local_l = index - inner * kLengthTile;
      const int64_t source_l = first_l + local_l;
      shared_x[index] = source_l < L
          ? X[(n * K + tile_k + inner) * L + source_l]
          : fp16_zero();
    }
    for (int index = thread; index < kOutputTile * kKTile;
         index += blockDim.x) {
      const int local_output = index / kKTile;
      const int inner = index - local_output * kKTile;
      const int64_t source_output = first_output + local_output;
      shared_w[inner * kSharedWStride + local_output] =
          source_output < O
          ? W[source_output * K + tile_k + inner]
          : fp16_zero();
    }
    __syncthreads();

    if (global_l < L) {
#pragma unroll
      for (int inner = 0; inner < kKTile; ++inner) {
        const __half lhs =
            shared_x[inner * kLengthTile + warp];
#pragma unroll
        for (int item = 0; item < outputs_per_lane; ++item) {
          const int local_output = lane + item * kWarpSize;
          const __half rhs =
              shared_w[inner * kSharedWStride + local_output];
          const __half product =
              float_backend::approx_mul_fp16(lhs, rhs, lut);
          accumulators[item] = __fadd_rn(
              accumulators[item], __half2float(product));
        }
      }
    }
    __syncthreads();
  }

  if (tile_k < K) {
    for (int index = thread; index < kKTile * kLengthTile;
         index += blockDim.x) {
      const int inner = index / kLengthTile;
      const int local_l = index - inner * kLengthTile;
      const int64_t source_l = first_l + local_l;
      const int64_t source_k = tile_k + inner;
      shared_x[index] = source_k < K && source_l < L
          ? X[(n * K + source_k) * L + source_l]
          : fp16_zero();
    }
    for (int index = thread; index < kOutputTile * kKTile;
         index += blockDim.x) {
      const int local_output = index / kKTile;
      const int inner = index - local_output * kKTile;
      const int64_t source_output = first_output + local_output;
      const int64_t source_k = tile_k + inner;
      shared_w[inner * kSharedWStride + local_output] =
          source_output < O && source_k < K
          ? W[source_output * K + source_k]
          : fp16_zero();
    }
    __syncthreads();

    const int valid_k = static_cast<int>(K - tile_k);
    if (global_l < L) {
#pragma unroll
      for (int inner = 0; inner < kKTile; ++inner) {
        if (inner < valid_k) {
          const __half lhs =
              shared_x[inner * kLengthTile + warp];
#pragma unroll
          for (int item = 0; item < outputs_per_lane; ++item) {
            const int local_output = lane + item * kWarpSize;
            const __half rhs =
                shared_w[inner * kSharedWStride + local_output];
            const __half product =
                float_backend::approx_mul_fp16(lhs, rhs, lut);
            accumulators[item] = __fadd_rn(
                accumulators[item], __half2float(product));
          }
        }
      }
    }
    __syncthreads();
  }

  if (global_l < L) {
#pragma unroll
    for (int item = 0; item < outputs_per_lane; ++item) {
      const int64_t output =
          first_output + lane + item * kWarpSize;
      if (output < O) {
        Y[(n * O + output) * L + global_l] =
            __float2half_rn(accumulators[item]);
      }
    }
  }
}

template <int outputs_per_lane>
void launch_bgemm_fp16_tiled(
    const __half* X,
    const __half* W,
    const uint16_t* lut,
    __half* Y,
    int64_t batch,
    int64_t O,
    int64_t L,
    int64_t K,
    cudaStream_t stream,
    const char* op_name) {
  constexpr int kOutputTile = kWarpSize * outputs_per_lane;
  const int64_t length_tiles =
      (L + kLengthTile - 1) / kLengthTile;
  const int64_t output_tiles =
      (O + kOutputTile - 1) / kOutputTile;
  const unsigned int blocks = checked_grid_x(
      batch * output_tiles * length_tiles, op_name);
  bgemm_fp16_tiled_kernel<outputs_per_lane>
      <<<blocks, kThreads, 0, stream>>>(
          X, W, lut, Y, O, L, K, length_tiles, output_tiles);
}

torch::Tensor launch_bgemm_fp16(
    const torch::Tensor& X,
    const torch::Tensor& W,
    const torch::Tensor& lut,
    bool optimized,
    const char* op_name) {
  check_input(X, torch::kFloat16, 3, "X", op_name);
  check_input(W, torch::kFloat16, 2, "W", op_name);
  TORCH_CHECK(X.device() == W.device(),
              op_name, ": X and W must be on the same CUDA device");
  TORCH_CHECK(X.size(1) == W.size(1), op_name,
              ": K dimensions must match, got X.shape[1]=", X.size(1),
              " and W.shape[1]=", W.size(1));
  check_lut(lut, torch::kUInt16, 1024, X.device(), op_name);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(X));
  const int64_t batch = X.size(0);
  const int64_t K = X.size(1);
  const int64_t L = X.size(2);
  const int64_t O = W.size(0);
  auto Y = torch::empty({batch, O, L}, X.options());
  const int64_t total = batch * O * L;
  if (total == 0) {
    return Y;
  }

  const auto* X_ptr =
      reinterpret_cast<const __half*>(X.data_ptr<at::Half>());
  const auto* W_ptr =
      reinterpret_cast<const __half*>(W.data_ptr<at::Half>());
  auto* Y_ptr = reinterpret_cast<__half*>(Y.data_ptr<at::Half>());
  const auto* lut_ptr = lut.data_ptr<uint16_t>();
  const auto stream = at::cuda::getCurrentCUDAStream();

  if (!optimized || K < kKTile || L < kLengthTile || O < 16) {
    bgemm_fp16_naive_kernel
        <<<elementwise_blocks(total), kThreads, 0, stream>>>(
            X_ptr, W_ptr, lut_ptr, Y_ptr, batch, O, L, K);
  } else if (O <= 32) {
    launch_bgemm_fp16_tiled<1>(
        X_ptr, W_ptr, lut_ptr, Y_ptr, batch, O, L, K, stream, op_name);
  } else if (O <= 64) {
    launch_bgemm_fp16_tiled<2>(
        X_ptr, W_ptr, lut_ptr, Y_ptr, batch, O, L, K, stream, op_name);
  } else {
    launch_bgemm_fp16_tiled<4>(
        X_ptr, W_ptr, lut_ptr, Y_ptr, batch, O, L, K, stream, op_name);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return Y;
}

torch::Tensor bgemm_fp16_naive_cuda(
    const torch::Tensor& X,
    const torch::Tensor& W,
    const torch::Tensor& lut) {
  return launch_bgemm_fp16(X, W, lut, false, "bgemm_fp16_naive");
}

torch::Tensor bgemm_fp16_cuda(
    const torch::Tensor& X,
    const torch::Tensor& W,
    const torch::Tensor& lut) {
  return launch_bgemm_fp16(X, W, lut, true, "bgemm_fp16");
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(approxtorch, m) {
  m.def("bgemm_fp16_naive(Tensor X, Tensor W, Tensor lut) -> Tensor");
  m.def("bgemm_fp16(Tensor X, Tensor W, Tensor lut) -> Tensor");
}

TORCH_LIBRARY_IMPL(approxtorch, CUDA, m) {
  m.impl("bgemm_fp16_naive", &bgemm_fp16_naive_cuda);
  m.impl("bgemm_fp16", &bgemm_fp16_cuda);
}

}  // namespace approxtorch
