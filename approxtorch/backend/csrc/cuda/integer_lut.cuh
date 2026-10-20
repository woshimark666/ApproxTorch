#pragma once

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

namespace approxtorch_integer {

// Explicit two's-complement wrap, including split-K partial sums. Avoid
// signed-overflow UB; no floating-point rounding occurs until the output.
__device__ __forceinline__ int32_t add(int32_t a, int32_t b) {
    return static_cast<int32_t>(static_cast<uint32_t>(a) + static_cast<uint32_t>(b));
}

template<bool TRANSPOSE>
__global__ void prepare_kernel(
    const float* __restrict__ in,
    int32_t* __restrict__ out,
    int16_t* __restrict__ out16,
    int* __restrict__ bad)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const float v = in[i];
    // The upper bound is exclusive: INT32_MAX rounds UP to 2^31 in FP32.
    const bool valid = v >= -2147483648.0f && v < 2147483648.0f && v == truncf(v);
    const bool i16_ok = valid && v >= -32768.0f && v <= 32767.0f;
    const bool u16_ok = valid && v >= 0.0f && v <= 65535.0f;
    // One atomic per invalid warp/category, rather than racing plain stores.
    if (__any_sync(0xffffffff, !i16_ok) && (threadIdx.x & 31) == 0) atomicExch(bad, 1);
    if (__any_sync(0xffffffff, !u16_ok) && (threadIdx.x & 31) == 0) atomicExch(bad + 1, 1);
    if (__any_sync(0xffffffff, !valid) && (threadIdx.x & 31) == 0) atomicExch(bad + 2, 1);
    const int o = TRANSPOSE ? (((i & 255) << 8) | (i >> 8)) : i;
    const int32_t value = valid ? static_cast<int32_t>(v) : 0;
    out[o] = value;
    out16[o] = static_cast<int16_t>(value);
}

struct Lut {
    torch::Tensor values;    // int32, optionally transposed
    torch::Tensor values16;  // shared bit pattern for int16 / uint16
    torch::Tensor bad;       // [int16 invalid, uint16 invalid, int32 invalid]
};

// Caller validates shape, dtype and device and holds the CUDA device guard.
// Inspecting the validity scalar deliberately synchronizes the current stream
// so invalid LUTs raise a recoverable host exception before the main kernel.
static inline Lut prepare(const torch::Tensor& lut, bool transpose, cudaStream_t stream) {
    auto src = lut.contiguous();
    Lut result{
        torch::empty({65536}, lut.options().dtype(torch::kInt32)),
        torch::empty({65536}, lut.options().dtype(torch::kInt16)),
        torch::zeros({3}, lut.options().dtype(torch::kInt32))
    };
    if (transpose) {
        prepare_kernel<true><<<256, 256, 0, stream>>>(
            src.data_ptr<float>(), result.values.data_ptr<int32_t>(),
            result.values16.data_ptr<int16_t>(), result.bad.data_ptr<int>());
    } else {
        prepare_kernel<false><<<256, 256, 0, stream>>>(
            src.data_ptr<float>(), result.values.data_ptr<int32_t>(),
            result.values16.data_ptr<int16_t>(), result.bad.data_ptr<int>());
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    TORCH_CHECK(result.bad.select(0, 2).item<int>() == 0,
                "forward LUT must contain only finite integer values in the INT32 range "
                "[-2147483648, 2147483647]");
    return result;
}

} // namespace approxtorch_integer
