#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
/*
 * Shared-memory LUT lookup for int8 tensors.
 *
 * Why shared memory over __constant__:
 *   - constant memory serializes divergent accesses within a warp
 *     (only 1 unique address broadcast per cycle).
 *   - LUT lookup is scatter-read: 32 threads in a warp hit ~32 different indices.
 *   - shared memory has 32 banks, 4 bytes each → 256 bytes spans 64 words
 *     across all 32 banks. Random int8 reads land on different banks → parallel.
 *
 * Vectorization: reinterpret int8* as int32*, process 4 elements per thread.
 * Block size fixed at 256 so cooperative LUT load is exactly 1 byte/thread.
 */
namespace approxtorch{

/*
 * int8 → float LUT lookup via shared memory.
 *
 * x:   int8  tensor, arbitrary shape (flattened internally)
 * lut: float tensor, size [256], where lut[x+128] is the mapped value
 * y:   float tensor, same shape as x
 *
 * Key insight: reinterpret int8 as uint8 == (int8 + 128), zero cost.
 *
 * Shared memory: 256 floats = 1024 bytes per block.
 * 32 banks × 4 bytes = each float lands on a unique bank index (idx % 32).
 * Random uint8 indices → good bank distribution → mostly parallel reads.
 *
 * Vectorization: load 4 × int8 as one int32, then do 4 LUT lookups,
 * write 4 floats via float4.
 */

 __global__ void lut_lookup_f32_vec4(
    const int32_t* __restrict__ x,
    float4* __restrict__ y,
    const float* __restrict__ lut,
    const int64_t n4
) {
    __shared__ float s_lut[256];

    s_lut[threadIdx.x] = lut[threadIdx.x];
    __syncthreads();

    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (int64_t i = idx; i < n4; i += stride) {
        int32_t pack = x[i];

        // 提取每个 int8，显式 +128 得到 [0,255] 索引，和 scalar kernel 一致
        int b0 = (int)((int8_t)(pack))       + 128;
        int b1 = (int)((int8_t)(pack >> 8))   + 128;
        int b2 = (int)((int8_t)(pack >> 16))  + 128;
        int b3 = (int)((int8_t)(pack >> 24))  + 128;

        float4 out;
        out.x = s_lut[b0];
        out.y = s_lut[b1];
        out.z = s_lut[b2];
        out.w = s_lut[b3];

        y[i] = out;
    }
}

__global__ void lut_lookup_f32_scalar(
    const int8_t* __restrict__ x,
    float* __restrict__ y,
    const float* __restrict__ lut,
    const int64_t offset,
    const int64_t n
) {
    __shared__ float s_lut[256];
    s_lut[threadIdx.x] = lut[threadIdx.x];
    __syncthreads();

    int64_t idx = offset + (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int ux = int(x[idx]) + 128;
        y[idx] = s_lut[ux];
    }
}

torch::Tensor lut_lookup_int8(torch::Tensor x, torch::Tensor lut) {
    TORCH_CHECK(x.dtype() == torch::kInt8, "x must be int8");
    TORCH_CHECK(lut.dtype() == torch::kFloat32, "lut must be float32");
    TORCH_CHECK(lut.numel() == 256, "lut must have 256 elements");
    TORCH_CHECK(x.is_cuda(), "x must be on CUDA");
    TORCH_CHECK(lut.is_cuda(), "lut must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(lut.is_contiguous(), "lut must be contiguous");

    // Output: same shape as x, but float32
    auto y = torch::empty(x.sizes(), x.options().dtype(torch::kFloat32));
    int64_t n = x.numel();
    if (n == 0) return y;

    const int threads = 256;

    // Vectorized: 4 int8 → 4 float per thread
    int64_t n4 = n / 4;
    if (n4 > 0) {
        int blocks = std::min((int)((n4 + threads - 1) / threads), 65535);
        lut_lookup_f32_vec4<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const int32_t*>(x.data_ptr<int8_t>()),
            reinterpret_cast<float4*>(y.data_ptr<float>()),
            lut.data_ptr<float>(),
            n4
        );
    }

    // Scalar tail for remaining 0-3 elements
    int64_t tail_offset = n4 * 4;
    int64_t tail_n = n - tail_offset;
    if (tail_n > 0) {
        int blocks = (tail_n + threads - 1) / threads;
        lut_lookup_f32_scalar<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            x.data_ptr<int8_t>(),
            y.data_ptr<float>(),
            lut.data_ptr<float>(),
            tail_offset,
            n
        );
    }

    return y;
}


TORCH_LIBRARY_FRAGMENT(approxtorch, m){
    // m.def("im2col_uint8(Tensor input, int k_h, int k_w, int stride_h, int stride_w, int padding_h, int padding_w, int dilation_h, int dilation_w) -> Tensor");
    m.def("lut_lookup_int8(Tensor x, Tensor lut) -> Tensor");
}

TORCH_LIBRARY_IMPL(approxtorch, CUDA, m){
    m.impl("lut_lookup_int8", &lut_lookup_int8);
}




}