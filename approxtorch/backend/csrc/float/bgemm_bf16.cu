#include "approx_float_cuda_common.cuh"
#include "approx_mul_bf16.cuh"

namespace approxtorch {
namespace {

using float_cuda_detail::check_input;
using float_cuda_detail::check_lut;
using float_cuda_detail::checked_grid_x;
using float_cuda_detail::elementwise_blocks;
using float_cuda_detail::kThreads;

constexpr int kKTile = 64;
constexpr int kWarpSize = 32;
constexpr int kLengthTile = 8;
constexpr int kSharedWPadding = 2;

__device__ __forceinline__ __nv_bfloat16 bf16_zero() {
  return __ushort_as_bfloat16(0);
}

template <bool shared_lookup>
__device__ __forceinline__ __nv_bfloat16 tiled_mul_bf16(
    __nv_bfloat16 a,
    __nv_bfloat16 b,
    const uint32_t* __restrict__ lut,
    const uint8_t* __restrict__ shared_lut) {
  if constexpr (!shared_lookup) {
    return float_backend::approx_mul_bf16(a, b, lut);
  } else {
    const uint16_t bits_a = __bfloat16_as_ushort(a);
    const uint16_t bits_b = __bfloat16_as_ushort(b);
    const uint32_t both_nonzero =
        static_cast<uint32_t>((bits_a & 0x7fffu) != 0) &
        static_cast<uint32_t>((bits_b & 0x7fffu) != 0);
    const uint16_t result_mask =
        static_cast<uint16_t>(uint32_t{0} - both_nonzero);
    // Input remains the LUT row operand, including asymmetric LUTs.
    const uint32_t index = ((bits_a & 0x7fu) << 7) | (bits_b & 0x7fu);
    const uint32_t entry = shared_lut[index];
    // With entry = 128*n + f, this is ((Ea+Eb-127+n)*128+f) mod
    // 32768: the original wrapped 8-bit exponent and 7-bit fraction.
    const uint32_t magnitude = ((bits_a & 0x7f80u) +
        (bits_b & 0x7f80u) - 0x3f80u + entry) & 0x7fffu;
    const uint16_t packed = static_cast<uint16_t>(
        ((bits_a ^ bits_b) & 0x8000u) | magnitude);
    return __ushort_as_bfloat16(static_cast<uint16_t>(packed & result_mask));
  }
}

__global__ void bgemm_bf16_naive_kernel(
    const __nv_bfloat16* __restrict__ X,
    const __nv_bfloat16* __restrict__ W,
    const uint32_t* __restrict__ lut,
    __nv_bfloat16* __restrict__ Y,
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
      const __nv_bfloat16 product = float_backend::approx_mul_bf16(
          X[(n * K + k) * L + l], W[output * K + k], lut);
      accumulator = __fadd_rn(accumulator, __bfloat162float(product));
    }
    Y[linear] = __float2bfloat16_rn(accumulator);
  }
}

// One warp fixes an L position, so its lanes share the X operand and one LUT
// row while spanning adjacent output channels. W is loaded coalesced in its
// native [O, K] layout, then transposed into padded shared memory. The padding
// removes the bank conflict caused by warp lanes reading different O values
// at the same K.
template <int outputs_per_lane, bool shared_lookup = false,
          int reduction_tile = kKTile>
__global__ __launch_bounds__(kThreads)
void bgemm_bf16_tiled_kernel(
    const __nv_bfloat16* __restrict__ X,
    const __nv_bfloat16* __restrict__ W,
    const uint32_t* __restrict__ lut,
    __nv_bfloat16* __restrict__ Y,
    int64_t O,
    int64_t L,
    int64_t K,
    int64_t length_tiles,
    int64_t output_tiles) {
  constexpr int kOutputTile = kWarpSize * outputs_per_lane;
  constexpr int kSharedWStride = kOutputTile + kSharedWPadding;
  __shared__ __align__(16) __nv_bfloat16 shared_x[reduction_tile * kLengthTile];
  __shared__ __align__(16) __nv_bfloat16 shared_w[reduction_tile * kSharedWStride];
  __shared__ uint8_t shared_lut[shared_lookup ? 128 * 128 : 1];

  const int thread = threadIdx.x;
  if constexpr (shared_lookup) {
    // Only the low byte is significant. Reload each CTA so in-place LUT
    // updates are visible without a cache or a separate preprocessing kernel.
    for (int index = thread; index < 128 * 128; index += blockDim.x) {
      shared_lut[index] = static_cast<uint8_t>(lut[index]);
    }
    __syncthreads();
  }
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
  for (; tile_k + reduction_tile <= K; tile_k += reduction_tile) {
    for (int index = thread; index < reduction_tile * kLengthTile;
         index += blockDim.x) {
      const int inner = index / kLengthTile;
      const int local_l = index - inner * kLengthTile;
      const int64_t source_l = first_l + local_l;
      shared_x[index] = source_l < L
          ? X[(n * K + tile_k + inner) * L + source_l]
          : bf16_zero();
    }
    for (int index = thread; index < kOutputTile * reduction_tile;
         index += blockDim.x) {
      const int local_output = index / reduction_tile;
      const int inner = index - local_output * reduction_tile;
      const int64_t source_output = first_output + local_output;
      shared_w[inner * kSharedWStride + local_output] =
          source_output < O
          ? W[source_output * K + tile_k + inner]
          : bf16_zero();
    }
    __syncthreads();

    if (global_l < L) {
#pragma unroll
      for (int inner = 0; inner < reduction_tile; ++inner) {
        const __nv_bfloat16 lhs =
            shared_x[inner * kLengthTile + warp];
#pragma unroll
        for (int item = 0; item < outputs_per_lane; ++item) {
          const int local_output = lane + item * kWarpSize;
          const __nv_bfloat16 rhs =
              shared_w[inner * kSharedWStride + local_output];
          const __nv_bfloat16 product =
              tiled_mul_bf16<shared_lookup>(lhs, rhs, lut, shared_lut);
          accumulators[item] = __fadd_rn(
              accumulators[item], __bfloat162float(product));
        }
      }
    }
    __syncthreads();
  }

  if (tile_k < K) {
    for (int index = thread; index < reduction_tile * kLengthTile;
         index += blockDim.x) {
      const int inner = index / kLengthTile;
      const int local_l = index - inner * kLengthTile;
      const int64_t source_l = first_l + local_l;
      const int64_t source_k = tile_k + inner;
      shared_x[index] = source_k < K && source_l < L
          ? X[(n * K + source_k) * L + source_l]
          : bf16_zero();
    }
    for (int index = thread; index < kOutputTile * reduction_tile;
         index += blockDim.x) {
      const int local_output = index / reduction_tile;
      const int inner = index - local_output * reduction_tile;
      const int64_t source_output = first_output + local_output;
      const int64_t source_k = tile_k + inner;
      shared_w[inner * kSharedWStride + local_output] =
          source_output < O && source_k < K
          ? W[source_output * K + source_k]
          : bf16_zero();
    }
    __syncthreads();

    const int valid_k = static_cast<int>(K - tile_k);
    if (global_l < L) {
#pragma unroll
      for (int inner = 0; inner < reduction_tile; ++inner) {
        if (inner < valid_k) {
          const __nv_bfloat16 lhs =
              shared_x[inner * kLengthTile + warp];
#pragma unroll
          for (int item = 0; item < outputs_per_lane; ++item) {
            const int local_output = lane + item * kWarpSize;
            const __nv_bfloat16 rhs =
                shared_w[inner * kSharedWStride + local_output];
            const __nv_bfloat16 product =
                tiled_mul_bf16<shared_lookup>(lhs, rhs, lut, shared_lut);
            accumulators[item] = __fadd_rn(
                accumulators[item], __bfloat162float(product));
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
            __float2bfloat16_rn(accumulators[item]);
      }
    }
  }
}

template <int outputs_per_lane, bool shared_lookup = false,
          int reduction_tile = kKTile>
void launch_bgemm_bf16_tiled(
    const __nv_bfloat16* X,
    const __nv_bfloat16* W,
    const uint32_t* lut,
    __nv_bfloat16* Y,
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
  bgemm_bf16_tiled_kernel<outputs_per_lane, shared_lookup, reduction_tile>
      <<<blocks, kThreads, 0, stream>>>(
          X, W, lut, Y, O, L, K, length_tiles, output_tiles);
}

torch::Tensor launch_bgemm_bf16(
    const torch::Tensor& X,
    const torch::Tensor& W,
    const torch::Tensor& lut,
    bool optimized,
    const char* op_name) {
  check_input(X, torch::kBFloat16, 3, "X", op_name);
  check_input(W, torch::kBFloat16, 2, "W", op_name);
  TORCH_CHECK(X.device() == W.device(),
              op_name, ": X and W must be on the same CUDA device");
  TORCH_CHECK(X.size(1) == W.size(1), op_name,
              ": K dimensions must match, got X.shape[1]=", X.size(1),
              " and W.shape[1]=", W.size(1));
  check_lut(lut, torch::kUInt32, 128, X.device(), op_name);

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
      reinterpret_cast<const __nv_bfloat16*>(X.data_ptr<at::BFloat16>());
  const auto* W_ptr =
      reinterpret_cast<const __nv_bfloat16*>(W.data_ptr<at::BFloat16>());
  auto* Y_ptr = reinterpret_cast<__nv_bfloat16*>(Y.data_ptr<at::BFloat16>());
  const auto* lut_ptr = lut.data_ptr<uint32_t>();
  const auto stream = at::cuda::getCurrentCUDAStream();

  // Short spatial maps with long reductions also benefit from reusing X/W
  // across a tile. Keep genuinely small reductions on the direct kernel.
  const bool tiled = K >= kKTile && O >= 16 &&
      (L >= 192 || (K >= 512 && O >= 64 && L >= 32));
  if (!optimized || !tiled) {
    bgemm_bf16_naive_kernel
        <<<elementwise_blocks(total), kThreads, 0, stream>>>(
            X_ptr, W_ptr, lut_ptr, Y_ptr, batch, O, L, K);
  } else if (K >= 512 && batch * L > 2048 && O >= 32 && L >= 192) {
    // Long reductions amortize staging the 16 KiB byte LUT. The smaller K
    // tile limits shared-memory usage while retaining output-channel reuse.
    if (O <= 32) {
      launch_bgemm_bf16_tiled<1, true, 32>(
          X_ptr, W_ptr, lut_ptr, Y_ptr, batch, O, L, K, stream, op_name);
    } else if (O <= 64) {
      launch_bgemm_bf16_tiled<2, true, 32>(
          X_ptr, W_ptr, lut_ptr, Y_ptr, batch, O, L, K, stream, op_name);
    } else {
      launch_bgemm_bf16_tiled<4, true, 32>(
          X_ptr, W_ptr, lut_ptr, Y_ptr, batch, O, L, K, stream, op_name);
    }
  } else if (O <= 32 || batch * L <= 2048) {
    // Smaller output tiles expose more independent CTAs on small spatial
    // workloads, and avoid spending most lanes on output-channel tails.
    launch_bgemm_bf16_tiled<1>(
        X_ptr, W_ptr, lut_ptr, Y_ptr, batch, O, L, K, stream, op_name);
  } else {
    // Keep weight reuse for large spatial workloads, but cap the tile at 64.
    // The fully unrolled 128-output tile uses 211 registers/thread on sm_89,
    // restricting residency; the 32/64-output variants use 40 without spills.
    launch_bgemm_bf16_tiled<2>(
        X_ptr, W_ptr, lut_ptr, Y_ptr, batch, O, L, K, stream, op_name);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return Y;
}

torch::Tensor bgemm_bf16_naive_cuda(
    const torch::Tensor& X,
    const torch::Tensor& W,
    const torch::Tensor& lut) {
  return launch_bgemm_bf16(X, W, lut, false, "bgemm_bf16_naive");
}

torch::Tensor bgemm_bf16_cuda(
    const torch::Tensor& X,
    const torch::Tensor& W,
    const torch::Tensor& lut) {
  return launch_bgemm_bf16(X, W, lut, true, "bgemm_bf16");
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(approxtorch, m) {
  m.def("bgemm_bf16_naive(Tensor X, Tensor W, Tensor lut) -> Tensor");
  m.def("bgemm_bf16(Tensor X, Tensor W, Tensor lut) -> Tensor");
}

TORCH_LIBRARY_IMPL(approxtorch, CUDA, m) {
  m.impl("bgemm_bf16_naive", &bgemm_bf16_naive_cuda);
  m.impl("bgemm_bf16", &bgemm_bf16_cuda);
}

}  // namespace approxtorch
