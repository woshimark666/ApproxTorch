#pragma once

// Included inside the backward implementation's anonymous namespace, after
// gradient_index() and scale_partial(). Forward kernels do not use this file.

// Cooperatively load O-contiguous slices, then reuse them across an M/K tile.
// Each thread retains one output's exact, increasing-O FP32 reduction order.
template <typename grad_t, int kTileM = 16, int kTileK = 16, int kTileO = 16>
__global__ void bf16_gemm_backward_x_tiled_kernel(
    const __nv_bfloat16* __restrict__ X,
    const __nv_bfloat16* __restrict__ W,
    const grad_t* __restrict__ dY,
    const float* __restrict__ dx_lut,
    __nv_bfloat16* __restrict__ dX,
    int64_t M, int64_t K, int64_t O) {
  constexpr int kBlockThreads = kTileM * kTileK;
  __shared__ uint16_t shared_w[kTileO][kTileK + 2];
  __shared__ float shared_dy[kTileM][kTileO + 1];

  const int thread = threadIdx.x;
  const int local_k = thread % kTileK;
  const int local_m = thread / kTileK;
  const int64_t k_tiles = (K + kTileK - 1) / kTileK;
  const int64_t first_m = (static_cast<int64_t>(blockIdx.x) / k_tiles) * kTileM;
  const int64_t first_k = (static_cast<int64_t>(blockIdx.x) % k_tiles) * kTileK;
  const int64_t m = first_m + local_m;
  const int64_t k = first_k + local_k;
  const bool valid = m < M && k < K;
  const uint16_t x = valid ? __bfloat16_as_ushort(X[m * K + k]) : 0;
  const unsigned int row = (x & kFractionMask) << kFractionBits;
  float accumulator = 0.0f;

  for (int64_t first_o = 0; first_o < O; first_o += kTileO) {
    for (int i = thread; i < kTileK * kTileO; i += kBlockThreads) {
      const int tile_k = i / kTileO;
      const int tile_o = i % kTileO;
      const int64_t source_k = first_k + tile_k;
      const int64_t source_o = first_o + tile_o;
      shared_w[tile_o][tile_k] = source_k < K && source_o < O
          ? __bfloat16_as_ushort(W[source_k * O + source_o]) : 0;
    }
    for (int i = thread; i < kTileM * kTileO; i += kBlockThreads) {
      const int tile_m = i / kTileO;
      const int tile_o = i % kTileO;
      const int64_t source_m = first_m + tile_m;
      const int64_t source_o = first_o + tile_o;
      shared_dy[tile_m][tile_o] = source_m < M && source_o < O
          ? static_cast<float>(dY[source_m * O + source_o]) : 0.0f;
    }
    __syncthreads();

    if ((x & kMagnitudeMask) != 0) {
#pragma unroll
      for (int tile_o = 0; tile_o < kTileO; ++tile_o) {
        const uint16_t w = shared_w[tile_o][local_k];
        if ((w & kMagnitudeMask) != 0) {
          const float partial = scale_partial(
              __ldg(dx_lut + row + (w & kFractionMask)), w);
          accumulator = __fadd_rn(
              accumulator, __fmul_rn(shared_dy[local_m][tile_o], partial));
        }
      }
    }
    __syncthreads();
  }
  if (valid) dX[m * K + k] = __float2bfloat16_rn(accumulator);
}

// Small GEMMs expose too few independent outputs to occupy the GPU. A warp
// prepares 32 products concurrently, then adds them in their original order.
// Every lane participates in the shuffles, and lane zero stores the result.
template <typename grad_t>
__global__ void bf16_gemm_backward_x_warp_kernel(
    const __nv_bfloat16* __restrict__ X,
    const __nv_bfloat16* __restrict__ W,
    const grad_t* __restrict__ dY,
    const float* __restrict__ dx_lut,
    __nv_bfloat16* __restrict__ dX,
    int64_t M, int64_t K, int64_t O) {
  const int64_t i = blockIdx.x;
  const int64_t m = i / K;
  const int64_t k = i % K;
  const int lane = threadIdx.x;
  const uint16_t x = __bfloat16_as_ushort(X[i]);
  float accumulator = 0.0f;
  if ((x & kMagnitudeMask) != 0) {
    const unsigned int row = (x & kFractionMask) << kFractionBits;
    for (int64_t first_o = 0; first_o < O; first_o += kWarpSize) {
      const int64_t o = first_o + lane;
      const uint16_t w = o < O ? __bfloat16_as_ushort(W[k * O + o]) : 0;
      const bool contributes = (w & kMagnitudeMask) != 0;
      float product = 0.0f;
      if (contributes) {
        const float partial = scale_partial(
            __ldg(dx_lut + row + (w & kFractionMask)), w);
        product = __fmul_rn(static_cast<float>(dY[m * O + o]), partial);
      }
      const unsigned int active = __ballot_sync(0xffffffffu, contributes);
#pragma unroll
      for (int source = 0; source < kWarpSize; ++source) {
        const float term = __shfl_sync(0xffffffffu, product, source);
        if (active & (1u << source)) {
          accumulator = __fadd_rn(accumulator, term);
        }
      }
    }
  }
  if (lane == 0) dX[i] = __float2bfloat16_rn(accumulator);
}
