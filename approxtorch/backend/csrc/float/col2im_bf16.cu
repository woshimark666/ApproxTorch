// BF16 STE fold with strided column reads. The col2im traversal is adapted
// from PyTorch 2.9.1, aten/src/ATen/native/cuda/im2col.cuh
// (col2im_device and col2im_batched_kernel). The original per-pixel FP32
// accumulation order and 512-thread launch are preserved. Only tensor
// addressing and allocation differ; no gradient formula is registered here.
// PyTorch's BSD-style license and required notices follow.
/*
From PyTorch:

Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
Copyright (c) 2011-2013 NYU                      (Clement Farabet)
Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)

From Caffe2:

Copyright (c) 2016-present, Facebook Inc. All rights reserved.

All contributions by Facebook:
Copyright (c) 2016 Facebook Inc.

All contributions by Google:
Copyright (c) 2015 Google Inc.
All rights reserved.

All contributions by Yangqing Jia:
Copyright (c) 2015 Yangqing Jia
All rights reserved.

All contributions by Kakao Brain:
Copyright 2019-2020 Kakao Brain

All contributions by Cruise LLC:
Copyright (c) 2022 Cruise LLC.
All rights reserved.

All contributions by Tri Dao:
Copyright (c) 2024 Tri Dao.
All rights reserved.

All contributions by Arm:
Copyright (c) 2021, 2023-2024 Arm Limited and/or its affiliates

All contributions from Caffe:
Copyright(c) 2013, 2014, 2015, the respective contributors
All rights reserved.

All other contributions:
Copyright(c) 2015, 2016 the respective contributors
All rights reserved.

Caffe2 uses a copyright model similar to Caffe: each contributor holds
copyright over their contributions to Caffe2. The project versioning records
all such contribution and copyright details. If a contributor wants to further
mark their specific copyright on a particular contribution, they should
indicate their copyright solely in the commit message of the change when it is
committed.

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
   and IDIAP Research Institute nor the names of its contributors may be
   used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
*/

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/KernelUtils.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>

#include <limits>

namespace approxtorch {
namespace {
using at::cuda::detail::GET_BLOCKS;

int64_t checked_product(int64_t left, int64_t right) {
  TORCH_CHECK(right == 0 || left <= std::numeric_limits<int64_t>::max() / right,
      "_col2im_bf16: geometry is too large");
  return left * right;
}

int64_t checked_sum(int64_t left, int64_t right) {
  TORCH_CHECK(left <= std::numeric_limits<int64_t>::max() - right,
      "_col2im_bf16: geometry is too large");
  return left + right;
}

C10_LAUNCH_BOUNDS_1(512)
__global__ void col2im_bf16_kernel(
    const int64_t per_image, const int64_t total,
    const at::BFloat16* data_col,
    const int64_t batch_stride, const int64_t reduction_stride,
    const int64_t position_stride,
    const int64_t height, const int64_t width,
    const int64_t kernel_h, const int64_t kernel_w,
    const int64_t pad_h, const int64_t pad_w,
    const int64_t stride_h, const int64_t stride_w,
    const int64_t dilation_h, const int64_t dilation_w,
    const int64_t height_col, const int64_t width_col,
    at::BFloat16* data_im) {
  using accT = at::acc_type<at::BFloat16, /*is_cuda=*/true>;
  CUDA_KERNEL_LOOP_TYPE(index, total, int64_t) {
    const int64_t ibatch = index / per_image;
    const int64_t slice_index = index % per_image;
    accT val = static_cast<accT>(0);
    const int64_t w_im = slice_index % width + pad_w;
    const int64_t h_im = (slice_index / width) % height + pad_h;
    const int64_t c_im = slice_index / (width * height);
    const int64_t kernel_extent_w = (kernel_w - 1) * dilation_w + 1;
    const int64_t kernel_extent_h = (kernel_h - 1) * dilation_h + 1;
    const int64_t w_col_start = w_im < kernel_extent_w
        ? 0 : (w_im - kernel_extent_w) / stride_w + 1;
    const int64_t w_col_end = ::min(w_im / stride_w + 1, width_col);
    const int64_t h_col_start = h_im < kernel_extent_h
        ? 0 : (h_im - kernel_extent_h) / stride_h + 1;
    const int64_t h_col_end = ::min(h_im / stride_h + 1, height_col);
    for (int64_t h_col = h_col_start; h_col < h_col_end; ++h_col) {
      for (int64_t w_col = w_col_start; w_col < w_col_end; ++w_col) {
        int64_t h_k = h_im - h_col * stride_h;
        int64_t w_k = w_im - w_col * stride_w;
        if (h_k % dilation_h == 0 && w_k % dilation_w == 0) {
          h_k /= dilation_h;
          w_k /= dilation_w;
          const int64_t reduction = (c_im * kernel_h + h_k) * kernel_w + w_k;
          const int64_t position = h_col * width_col + w_col;
          // ATen's original address is reduction * L + position.
          // FP32 += BF16 and the loop order are deliberately unchanged.
          val += data_col[ibatch * batch_stride +
              reduction * reduction_stride + position * position_stride];
        }
      }
    }
    data_im[index] = static_cast<at::BFloat16>(val);
  }
}

at::Tensor col2im_bf16(
    const at::Tensor& columns_, at::IntArrayRef output_size,
    at::IntArrayRef kernel_size, at::IntArrayRef dilation,
    at::IntArrayRef padding, at::IntArrayRef stride) {
  TORCH_CHECK(columns_.is_cuda() && columns_.scalar_type() == at::kBFloat16,
      "_col2im_bf16: columns must be a CUDA BF16 tensor");
  TORCH_CHECK(columns_.layout() == at::kStrided && columns_.dim() == 3,
      "_col2im_bf16: columns must be a strided tensor with shape [N,K,L]");
  TORCH_CHECK(output_size.size() == 2 && kernel_size.size() == 2 &&
      dilation.size() == 2 && padding.size() == 2 && stride.size() == 2,
      "_col2im_bf16: geometry arguments must be pairs");
  for (int d = 0; d < 2; ++d) {
    TORCH_CHECK(output_size[d] > 0 && kernel_size[d] > 0 && dilation[d] > 0 &&
        stride[d] > 0 && padding[d] >= 0, "_col2im_bf16: invalid geometry");
  }
  const c10::cuda::CUDAGuard device_guard(columns_.device());
  // Resolve lazy-negative views before raw storage access. Higher derivatives
  // use the Python wrapper's native fold fallback.
  const at::Tensor columns = columns_.resolve_neg();
  const int64_t n = columns.size(0), k = columns.size(1);
  const int64_t h = output_size[0], w = output_size[1];
  const int64_t kh = kernel_size[0], kw = kernel_size[1];
  const int64_t kernel_elements = checked_product(kh, kw);
  const int64_t extent_h = checked_sum(checked_product(dilation[0], kh - 1), 1);
  const int64_t extent_w = checked_sum(checked_product(dilation[1], kw - 1), 1);
  const int64_t nh = checked_sum(h, checked_product(2, padding[0])) - extent_h;
  const int64_t nw = checked_sum(w, checked_product(2, padding[1])) - extent_w;
  TORCH_CHECK(nh >= 0 && nw >= 0 && k > 0 && k % kernel_elements == 0,
      "_col2im_bf16: invalid column geometry");
  const int64_t hc = nh / stride[0] + 1, wc = nw / stride[1] + 1;
  TORCH_CHECK(columns.size(2) == checked_product(hc, wc),
      "_col2im_bf16: incorrect number of columns");
  const int64_t c = k / kernel_elements;
  const int64_t per_image = checked_product(c, checked_product(h, w));
  checked_product(n, per_image);
  auto output = at::empty({n, c, h, w}, columns.options());
  if (output.numel() == 0) return output;
  col2im_bf16_kernel<<<GET_BLOCKS(output.numel(), 512), 512, 0,
      at::cuda::getCurrentCUDAStream()>>>(
      per_image, output.numel(), columns.const_data_ptr<at::BFloat16>(),
      columns.stride(0), columns.stride(1), columns.stride(2), h, w, kh, kw,
      padding[0], padding[1], stride[0], stride[1], dilation[0], dilation[1],
      hc, wc, output.mutable_data_ptr<at::BFloat16>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
} // namespace

TORCH_LIBRARY_FRAGMENT(approxtorch, m) {
  m.def("_col2im_bf16(Tensor columns, int[] output_size, int[] kernel_size, "
        "int[] dilation, int[] padding, int[] stride) -> Tensor");
}
TORCH_LIBRARY_IMPL(approxtorch, CUDA, m) {
  m.impl("_col2im_bf16", &col2im_bf16);
}

} // namespace approxtorch
