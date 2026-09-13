#pragma once

// Included inside the BF16 backward's anonymous namespace, after its bit-field
// helpers. Each block owns several output channels of a single input channel.
// Reusing X saves loads and index arithmetic without changing the 256-thread
// accumulation order of any individual weight.
// For GEMM, transposed means X[K,M] and dY[O,M] scratch tensors; W stays[K,O].
// transposed_lut uses a per-call transpose of the original 128x128 table, so
// columns are contiguous. Both public LUT arguments remain independent 1D
// FP32 buffers and are never modified or cached by these kernels.
template <bool gemm, bool transposed, typename grad_t, int kOutputs, bool transposed_lut = false>
__global__ void bf16_backward_w_tiled_kernel(
    const __nv_bfloat16* __restrict__ X,
    const __nv_bfloat16* __restrict__ W,
    const grad_t* __restrict__ dY,
    const float* __restrict__ dw_lut,
    __nv_bfloat16* __restrict__ dW,
    int64_t batch, int64_t K, int64_t L, int64_t O) {
  const int64_t groups = (O + kOutputs - 1) / kOutputs;
  const int64_t k = blockIdx.x / groups;
  const int64_t first_o = (blockIdx.x % groups) * kOutputs;
  uint16_t weights[kOutputs];
  bool any_nonzero = false;
#pragma unroll
  for (int j = 0; j < kOutputs; ++j) {
    const int64_t o = first_o + j;
    weights[j] = o < O
        ? __bfloat16_as_ushort(W[gemm ? k * O + o : o * K + k])
        : 0;
    any_nonzero |= (weights[j] & kMagnitudeMask) != 0;
  }
  if (!any_nonzero) {
    if (threadIdx.x < kOutputs && first_o + threadIdx.x < O) {
      const int64_t o = first_o + threadIdx.x;
      dW[gemm ? k * O + o : o * K + k] = __ushort_as_bfloat16(0);
    }
    return;
  }
  __shared__ float columns[kOutputs][1 << kFractionBits];
  __shared__ float warp_sums[kOutputs][kThreads / kWarpSize];
  for (int p = threadIdx.x; p < kOutputs * (1 << kFractionBits); p += kThreads) {
    const int j = p >> kFractionBits;
    const int fraction = p & kFractionMask;
    columns[j][fraction] = (weights[j] & kMagnitudeMask) != 0
        ? __ldg(dw_lut + (transposed_lut
              ? ((weights[j] & kFractionMask) << kFractionBits) + fraction
              : (fraction << kFractionBits) + (weights[j] & kFractionMask)))
        : 0.0f;
  }
  __syncthreads();
  float accum[kOutputs] = {};
  int64_t n = gemm ? 0 : (L ? threadIdx.x / L : 0);
  int64_t l = gemm ? threadIdx.x : (L ? threadIdx.x % L : 0);
  const int64_t step_n = gemm ? 0 : (L ? kThreads / L : 0);
  const int64_t step_l = gemm ? kThreads : (L ? kThreads % L : 0);
  for (int64_t nl = threadIdx.x; nl < batch * L; nl += kThreads) {
    const uint16_t x = __bfloat16_as_ushort(
        X[gemm && !transposed ? l * K + k : (n * K + k) * L + l]);
    if ((x & kMagnitudeMask) != 0) {
#pragma unroll
      for (int j = 0; j < kOutputs; ++j) {
        if ((weights[j] & kMagnitudeMask) != 0) {
          const int64_t o = first_o + j;
          const float partial = scale_partial(columns[j][x & kFractionMask], x);
          const float dy = static_cast<float>(
              dY[gemm && !transposed ? l * O + o : (n * O + o) * L + l]);
          accum[j] = __fadd_rn(accum[j], __fmul_rn(dy, partial));
        }
      }
    }
    n += step_n;
    l += step_l;
    if constexpr (!gemm) {
      if (l >= L) {
        l -= L;
        ++n;
      }
    }
  }
  const int lane = threadIdx.x % kWarpSize;
  const int warp = threadIdx.x / kWarpSize;
#pragma unroll
  for (int j = 0; j < kOutputs; ++j) {
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
      accum[j] = __fadd_rn(
          accum[j], __shfl_down_sync(0xffffffffu, accum[j], offset));
    }
    if (lane == 0) warp_sums[j][warp] = accum[j];
  }
  __syncthreads();
  if (warp == 0) {
#pragma unroll
    for (int j = 0; j < kOutputs; ++j) {
      float acc = lane < kThreads / kWarpSize ? warp_sums[j][lane] : 0.0f;
#pragma unroll
      for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        acc = __fadd_rn(acc, __shfl_down_sync(0xffffffffu, acc, offset));
      }
      const int64_t o = first_o + j;
      if (lane == 0 && o < O) {
        dW[gemm ? k * O + o : o * K + k] = __float2bfloat16_rn(acc);
      }
    }
  }
}

// One warp emulates the eight independent warps of the original one-weight
// block. Its eight FP32 accumulators keep nl = lane + 32*r + 256*j exactly;
// the final shuffle tree is consequently identical to the original reduction.
// kParts=1/2/4 is valid only when batch*L<=32/64/128 respectively; the omitted
// original warp accumulators are then all +0. Use kParts=8 for longer inputs.
template <bool gemm, bool transposed, typename grad_t, bool transposed_lut = false,
          int kParts = kThreads / kWarpSize>
__global__ void bf16_backward_w_warp_kernel(
    const __nv_bfloat16* __restrict__ X,
    const __nv_bfloat16* __restrict__ W,
    const grad_t* __restrict__ dY,
    const float* __restrict__ dw_lut,
    __nv_bfloat16* __restrict__ dW,
    int64_t batch, int64_t K, int64_t L, int64_t O) {
  constexpr int kWarps = kThreads / kWarpSize;
  static_assert(kThreads == 256 && kWarpSize == 32,
                "Keep the original 256-thread weight reduction order");
  static_assert(kParts == 1 || kParts == 2 || kParts == 4 || kParts == 8,
                "The original reduction has eight warps");
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int64_t i = static_cast<int64_t>(blockIdx.x) * kWarps + warp;
  if (i >= K * O) return;
  const int64_t k = gemm ? i / O : i % K;
  const int64_t o = gemm ? i % O : i / K;
  const uint16_t w = __bfloat16_as_ushort(W[i]);
  if ((w & kMagnitudeMask) == 0) {
    if (lane == 0) dW[i] = __ushort_as_bfloat16(0);
    return;
  }
  __shared__ float columns[kWarps][1 << kFractionBits];
#pragma unroll
  for (int m = lane; m < (1 << kFractionBits); m += kWarpSize) {
    columns[warp][m] = __ldg(dw_lut + (transposed_lut
        ? ((w & kFractionMask) << kFractionBits) + m
        : (m << kFractionBits) + (w & kFractionMask)));
  }
  __syncwarp();
  float accum[kParts] = {};
  int64_t n = gemm ? 0 : (L ? lane / L : 0);
  int64_t l = gemm ? lane : (L ? lane % L : 0);
  const int64_t step_n = gemm ? 0 : (L ? kWarpSize / L : 0);
  const int64_t step_l = gemm ? kWarpSize : (L ? kWarpSize % L : 0);
  for (int64_t first = lane; first < batch * L; first += kThreads) {
#pragma unroll
    for (int r = 0; r < kParts; ++r) {
      const int64_t nl = first + r * kWarpSize;
      if (nl < batch * L) {
        const uint16_t x = __bfloat16_as_ushort(
            X[gemm && !transposed ? l * K + k : (n * K + k) * L + l]);
        if ((x & kMagnitudeMask) != 0) {
          const float partial = scale_partial(columns[warp][x & kFractionMask], x);
          const float dy = static_cast<float>(
              dY[gemm && !transposed ? l * O + o : (n * O + o) * L + l]);
          accum[r] = __fadd_rn(accum[r], __fmul_rn(dy, partial));
        }
      }
      n += step_n;
      l += step_l;
      if constexpr (!gemm) {
        if (l >= L) {
          l -= L;
          ++n;
        }
      }
    }
  }
#pragma unroll
  for (int r = 0; r < kParts; ++r) {
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
      accum[r] = __fadd_rn(
          accum[r], __shfl_down_sync(0xffffffffu, accum[r], offset));
    }
  }
  if (lane == 0) {
#pragma unroll
    for (int offset = kParts / 2; offset > 0; offset >>= 1) {
#pragma unroll
      for (int r = 0; r < offset; ++r) {
        accum[r] = __fadd_rn(accum[r], accum[r + offset]);
      }
    }
    dW[i] = __float2bfloat16_rn(accum[0]);
  }
}
