#pragma once

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <algorithm>
#include <cstdint>
#include <limits>

namespace approxtorch {
namespace float_cuda_detail {

constexpr int kThreads = 256;
constexpr int64_t kMaxElementwiseBlocks = 65535;
constexpr int64_t kMaxGridX =
    static_cast<int64_t>(std::numeric_limits<int32_t>::max());

inline void check_lut(
    const torch::Tensor& lut,
    torch::ScalarType dtype,
    int64_t side,
    const torch::Device& device,
    const char* op_name) {
  TORCH_CHECK(lut.is_cuda(), op_name, ": lut must be a CUDA tensor");
  TORCH_CHECK(lut.device() == device,
              op_name, ": all tensors must be on the same CUDA device");
  TORCH_CHECK(lut.scalar_type() == dtype,
              op_name, ": lut has the wrong dtype");
  TORCH_CHECK(lut.dim() == 2 && lut.size(0) == side && lut.size(1) == side,
              op_name, ": lut must have shape [", side, ", ", side, "]");
  TORCH_CHECK(lut.numel() == side * side,
              op_name, ": lut has the wrong element count");
  TORCH_CHECK(lut.is_contiguous(),
              op_name, ": lut must be contiguous in row-major order");
}

inline void check_input(
    const torch::Tensor& tensor,
    torch::ScalarType dtype,
    int64_t dimensions,
    const char* argument,
    const char* op_name) {
  TORCH_CHECK(tensor.is_cuda(), op_name, ": ", argument,
              " must be a CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == dtype, op_name, ": ", argument,
              " has the wrong dtype");
  TORCH_CHECK(tensor.dim() == dimensions, op_name, ": ", argument,
              " must be ", dimensions, "-dimensional");
  TORCH_CHECK(tensor.is_contiguous(), op_name, ": ", argument,
              " must be contiguous");
}

inline int elementwise_blocks(int64_t count) {
  return static_cast<int>(std::min<int64_t>(
      (count + kThreads - 1) / kThreads, kMaxElementwiseBlocks));
}

inline unsigned int checked_grid_x(int64_t blocks, const char* op_name) {
  TORCH_CHECK(blocks > 0 && blocks <= kMaxGridX,
              op_name, ": CUDA grid is too large");
  return static_cast<unsigned int>(blocks);
}

}  // namespace float_cuda_detail
}  // namespace approxtorch
