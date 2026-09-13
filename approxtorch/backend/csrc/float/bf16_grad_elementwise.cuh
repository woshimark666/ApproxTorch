#pragma once

#include "bf16_grad_scale.cuh"

#include <cuda_bf16.h>

namespace approxtorch {
namespace float_backward_detail {

// Two adjacent BF16 values share one aligned 32-bit load/store. Call only
// when X, W, dY (for BF16), dX and dW are at least four-byte aligned; an odd
// final element is handled by the same kernel without an out-of-bounds load.
template <typename grad_t, bool need_x, bool need_w>
__global__ void approx_mul_bf16_backward_packed_kernel(
    const __nv_bfloat16* __restrict__ X,
    const __nv_bfloat16* __restrict__ W,
    const grad_t* __restrict__ dY,
    const float* __restrict__ dx_lut,
    const float* __restrict__ dw_lut,
    __nv_bfloat16* __restrict__ dX,
    __nv_bfloat16* __restrict__ dW,
    int64_t count) {
  using namespace float_backend::bf16_detail;
  const int64_t pairs = count / 2;
  const int64_t step = static_cast<int64_t>(blockDim.x) * gridDim.x;
  const int64_t first = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  for (int64_t pair = first; pair < pairs; pair += step) {
    const unsigned int xs = reinterpret_cast<const unsigned int*>(X)[pair];
    const unsigned int ws = reinterpret_cast<const unsigned int*>(W)[pair];
    float dy[2];
    if constexpr (sizeof(grad_t) == sizeof(__nv_bfloat16)) {
      const unsigned int ys = reinterpret_cast<const unsigned int*>(dY)[pair];
      dy[0] = __bfloat162float(__ushort_as_bfloat16(static_cast<uint16_t>(ys)));
      dy[1] = __bfloat162float(__ushort_as_bfloat16(static_cast<uint16_t>(ys >> 16)));
    } else {
      dy[0] = static_cast<float>(dY[2 * pair]);
      dy[1] = static_cast<float>(dY[2 * pair + 1]);
    }
    float dx[2] = {0.0f, 0.0f}, dw[2] = {0.0f, 0.0f};
#pragma unroll
    for (int lane = 0; lane < 2; ++lane) {
      const uint16_t x = static_cast<uint16_t>(xs >> (16 * lane));
      const uint16_t w = static_cast<uint16_t>(ws >> (16 * lane));
      if ((x & kMagnitudeMask) != 0 && (w & kMagnitudeMask) != 0) {
        const unsigned int index =
            ((x & kFractionMask) << kFractionBits) | (w & kFractionMask);
        if constexpr (need_x) {
          dx[lane] = dy[lane] * scale_partial_fast(__ldg(dx_lut + index), w);
        }
        if constexpr (need_w) {
          dw[lane] = dy[lane] * scale_partial_fast(__ldg(dw_lut + index), x);
        }
      }
    }
    if constexpr (need_x) {
      reinterpret_cast<__nv_bfloat162*>(dX)[pair] = __floats2bfloat162_rn(dx[0], dx[1]);
    }
    if constexpr (need_w) {
      reinterpret_cast<__nv_bfloat162*>(dW)[pair] = __floats2bfloat162_rn(dw[0], dw[1]);
    }
  }
  if ((count & 1) && first == 0) {
    const int64_t i = count - 1;
    const uint16_t x = __bfloat16_as_ushort(X[i]);
    const uint16_t w = __bfloat16_as_ushort(W[i]);
    float dx = 0.0f, dw = 0.0f;
    if ((x & kMagnitudeMask) != 0 && (w & kMagnitudeMask) != 0) {
      const unsigned int index =
          ((x & kFractionMask) << kFractionBits) | (w & kFractionMask);
      const float dy = static_cast<float>(dY[i]);
      if constexpr (need_x) dx = dy * scale_partial_fast(__ldg(dx_lut + index), w);
      if constexpr (need_w) dw = dy * scale_partial_fast(__ldg(dw_lut + index), x);
    }
    if constexpr (need_x) dX[i] = __float2bfloat16_rn(dx);
    if constexpr (need_w) dW[i] = __float2bfloat16_rn(dw);
  }
}

}  // namespace float_backward_detail
}  // namespace approxtorch
