#pragma once

#include "approx_mul_bf16.cuh"

#include <cuda_runtime.h>
#include <stdint.h>

namespace approxtorch {
namespace float_backward_detail {

// Multiplication by an integral power of two does not round while the input
// and output are both normal FP32. Adjust that exponent field directly, and
// leave the complete IEEE boundary handling to the original ldexpf path.
__device__ __forceinline__ float scale_partial_fast(float partial, uint16_t bits) {
  using namespace float_backend::bf16_detail;
  const unsigned int raw = __float_as_uint(partial);
  const unsigned int source_exponent = (raw >> 23) & 0xffu;
  const int exponent =
      static_cast<int>((bits >> kFractionBits) & kExponentFieldMask) - kExponentBias;
  const unsigned int result_exponent = source_exponent + exponent;
  if (source_exponent - 1u < 254u && result_exponent - 1u < 254u) {
    const unsigned int scaled = raw + (static_cast<unsigned int>(exponent) << 23);
    return __uint_as_float(scaled ^ ((static_cast<unsigned int>(bits) & kSignMask) << 16));
  }
  return ldexpf((bits & kSignMask) ? -partial : partial, exponent);
}

}  // namespace float_backward_detail
}  // namespace approxtorch
