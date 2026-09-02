#pragma once

#include <cuda_bf16.h>
#include <stdint.h>

namespace approxtorch {
namespace float_backend {
namespace bf16_detail {

constexpr int kFractionBits = 7;
constexpr int kExponentBias = 127;

constexpr uint16_t kSignMask = 0x8000u;
constexpr uint16_t kMagnitudeMask = 0x7fffu;
constexpr uint16_t kFractionMask = 0x007fu;
constexpr uint32_t kExponentFieldMask = 0x00ffu;
constexpr uint16_t kLutNormalizationBit = 0x0080u;

}  // namespace bf16_detail

// RTL-compatible BF16 multiplication using a mantissa-result LUT.
//
// LUT contract:
//   - shape: [128, 128], flattened row-major;
//   - index: lut[fraction(a) * 128 + fraction(b)];
//   - entry type: uint16_t, with only bits 7:0 used;
//   - bit 7: add one to the result exponent;
//   - bits 6:0: result fraction.
//
// This deliberately models the corresponding RTL rather than IEEE-754.
// Exponent arithmetic wraps to eight bits. NaN, infinity, overflow,
// underflow, subnormal normalization, and rounding receive no special
// handling. The only special case is signed zero: if either operand is +0 or
// -0, the result is +0.
//
// For non-symmetric approximate multipliers the operand order is significant:
// a selects the LUT row and b selects the LUT column.
__device__ __forceinline__ __nv_bfloat16 approx_mul_bf16(
    __nv_bfloat16 a,
    __nv_bfloat16 b,
    const uint16_t* __restrict__ mantissa_lut) {
  using namespace bf16_detail;

  const uint16_t bits_a = __bfloat16_as_ushort(a);
  const uint16_t bits_b = __bfloat16_as_ushort(b);

  // Turn the zero rule into a final bit mask so every lane follows the same
  // LUT and result-packing path. True converts to 1; unsigned subtraction
  // therefore produces either 0xffff (both non-zero) or 0x0000 (any zero).
  const uint32_t both_nonzero =
      static_cast<uint32_t>((bits_a & kMagnitudeMask) != 0) &
      static_cast<uint32_t>((bits_b & kMagnitudeMask) != 0);
  const uint16_t result_mask =
      static_cast<uint16_t>(uint32_t{0} - both_nonzero);

  const uint32_t fraction_a = bits_a & kFractionMask;
  const uint32_t fraction_b = bits_b & kFractionMask;
  const uint32_t lut_index =
      (fraction_a << kFractionBits) | fraction_b;
  const uint16_t lut_entry = __ldg(mantissa_lut + lut_index);

  const uint16_t fraction_out = lut_entry & kFractionMask;
  const uint32_t normalization =
      (lut_entry & kLutNormalizationBit) >> kFractionBits;
  const uint32_t exponent_a =
      (static_cast<uint32_t>(bits_a) >> kFractionBits) & kExponentFieldMask;
  const uint32_t exponent_b =
      (static_cast<uint32_t>(bits_b) >> kFractionBits) & kExponentFieldMask;

  // Conversion to unsigned followed by the mask reproduces the RTL's
  // eight-bit exponent wire, including wraparound for negative intermediates.
  const int exponent_sum = static_cast<int>(exponent_a + exponent_b) -
                           kExponentBias + static_cast<int>(normalization);
  const uint16_t exponent_out = static_cast<uint16_t>(
      static_cast<uint32_t>(exponent_sum) & kExponentFieldMask);
  const uint16_t sign_out =
      static_cast<uint16_t>((bits_a ^ bits_b) & kSignMask);
  const uint16_t result = static_cast<uint16_t>(
      sign_out | (exponent_out << kFractionBits) | fraction_out);

  return __ushort_as_bfloat16(static_cast<uint16_t>(result & result_mask));
}

}  // namespace float_backend
}  // namespace approxtorch
