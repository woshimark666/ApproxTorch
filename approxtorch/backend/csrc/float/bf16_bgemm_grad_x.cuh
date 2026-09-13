#pragma once

// Independent adjacent-L accumulators expose instruction-level parallelism
// while sharing W decoding. Each output still sums in increasing O order.
template <typename grad_t, int values_per_thread>
__global__ void bf16_bgemm_backward_x_register_kernel(
    const __nv_bfloat16* __restrict__ X,
    const __nv_bfloat16* __restrict__ W,
    const grad_t* __restrict__ dY,
    const float* __restrict__ dx_lut,
    __nv_bfloat16* __restrict__ dX,
    int64_t batch, int64_t K, int64_t L, int64_t O) {
  const int64_t groups_l = (L + values_per_thread - 1) / values_per_thread;
  const int64_t total = batch * K * groups_l;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < total; i += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t first_l = (i % groups_l) * values_per_thread;
    const int64_t nk = i / groups_l;
    const int64_t n = nk / K, k = nk % K;
    uint16_t x[values_per_thread];
    float acc[values_per_thread] = {};
#pragma unroll
    for (int v = 0; v < values_per_thread; ++v) {
      x[v] = first_l + v < L ? __bfloat16_as_ushort(X[nk * L + first_l + v]) : 0;
    }
    for (int64_t o = 0; o < O; ++o) {
      const uint16_t w = __bfloat16_as_ushort(W[o * K + k]);
      if ((w & kMagnitudeMask) != 0) {
#pragma unroll
        for (int v = 0; v < values_per_thread; ++v) {
          if ((x[v] & kMagnitudeMask) != 0) {
            const float partial = scale_partial(__ldg(dx_lut + gradient_index(x[v], w)), w);
            const float dy = static_cast<float>(dY[(n * O + o) * L + first_l + v]);
            acc[v] = __fadd_rn(acc[v], __fmul_rn(dy, partial));
          }
        }
      }
    }
#pragma unroll
    for (int v = 0; v < values_per_thread; ++v) {
      if (first_l + v < L) dX[nk * L + first_l + v] = __float2bfloat16_rn(acc[v]);
    }
  }
}

// Included inside the backward implementation's anonymous namespace.
// Each CTA reuses an O tile of dY across four K rows, and W across 64 L
// positions. The O accumulation order is identical to the direct kernel.
template <typename grad_t, int k_tile = 4, int l_tile = 64, int o_tile = 16>
__global__ void bf16_bgemm_backward_x_tiled_kernel(
    const __nv_bfloat16* __restrict__ X,
    const __nv_bfloat16* __restrict__ W,
    const grad_t* __restrict__ dY,
    const float* __restrict__ dx_lut,
    __nv_bfloat16* __restrict__ dX,
    int64_t K, int64_t L, int64_t O,
    int64_t k_tiles, int64_t l_tiles) {
  __shared__ uint16_t shared_w[o_tile][k_tile];
  __shared__ float shared_dy[o_tile][l_tile];
  const int tid = threadIdx.x;
  const int local_k = tid / l_tile;
  const int local_l = tid % l_tile;
  const int64_t tile = blockIdx.x;
  const int64_t n = tile / (k_tiles * l_tiles);
  const int64_t first_k = ((tile / l_tiles) % k_tiles) * k_tile;
  const int64_t first_l = (tile % l_tiles) * l_tile;
  const int64_t k = first_k + local_k;
  const int64_t l = first_l + local_l;
  const bool valid = k < K && l < L;
  const int64_t x_index = (n * K + k) * L + l;
  const uint16_t x = valid ? __bfloat16_as_ushort(X[x_index]) : 0;
  const unsigned int row = (x & kFractionMask) << kFractionBits;
  float acc = 0.0f;
  for (int64_t first_o = 0; first_o < O; first_o += o_tile) {
    for (int j = tid; j < o_tile * k_tile; j += blockDim.x) {
      const int o = j / k_tile, kk = j % k_tile;
      shared_w[o][kk] = first_o + o < O && first_k + kk < K
          ? __bfloat16_as_ushort(W[(first_o + o) * K + first_k + kk]) : 0;
    }
    for (int j = tid; j < o_tile * l_tile; j += blockDim.x) {
      const int o = j / l_tile, ll = j % l_tile;
      shared_dy[o][ll] = first_o + o < O && first_l + ll < L
          ? static_cast<float>(dY[(n * O + first_o + o) * L + first_l + ll]) : 0.0f;
    }
    __syncthreads();
    if ((x & kMagnitudeMask) != 0) {
#pragma unroll
      for (int o = 0; o < o_tile; ++o) {
        const uint16_t w = shared_w[o][local_k];
        if ((w & kMagnitudeMask) != 0) {
          const float partial = scale_partial(__ldg(dx_lut + row + (w & kFractionMask)), w);
          acc = __fadd_rn(acc, __fmul_rn(shared_dy[o][local_l], partial));
        }
      }
    }
    __syncthreads();
  }
  if (valid) dX[x_index] = __float2bfloat16_rn(acc);
}
