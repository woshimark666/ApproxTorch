#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

template <typename scalar_t>
__global__ void elementwise_mul_kernel(
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    const scalar_t* __restrict__ lut,   // 长度必须为 256*256 = 65536
    scalar_t* __restrict__ out,
    const int64_t n) {

    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // float -> int (截断到零)。想要四舍五入可以用 __float2int_rn
    int ai = static_cast<int>(a[idx]);
    int bi = static_cast<int>(b[idx]);

    // clamp 到 int8 范围 [-128, 127]，防止越界访问
    ai = max(-128, min(127, ai));
    bi = max(-128, min(127, bi));

    // LUT 寻址：(a+128)*256 + (b+128)
    const int lut_idx = (ai + 128) * 256 + (bi + 128);

    out[idx] = lut[lut_idx];
}

torch::Tensor elementwise_mul_cuda(torch::Tensor a, torch::Tensor b, torch::Tensor lut) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda() && lut.is_cuda(),
                "a, b, lut must all be CUDA tensors");
    TORCH_CHECK(a.sizes() == b.sizes(),
                "a and b must have the same shape");
    TORCH_CHECK(lut.numel() == 256 * 256,
                "lut must have exactly 65536 elements");
    TORCH_CHECK(a.scalar_type() == b.scalar_type() &&
                a.scalar_type() == lut.scalar_type(),
                "a, b, lut must have the same dtype");

    a = a.contiguous();
    b = b.contiguous();
    lut = lut.contiguous();

    auto out = torch::empty_like(a);
    const int64_t n = a.numel();

    const int threads = 256;
    const int blocks  = static_cast<int>((n + threads - 1) / threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    elementwise_mul_kernel<float><<<blocks, threads, 0, stream>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        lut.data_ptr<float>(),
        out.data_ptr<float>(),
        n
    );

    return out;
}





TORCH_LIBRARY_FRAGMENT(approxtorch, m) {   
    m.def("elementwise_mul(Tensor a, Tensor b, Tensor lut) -> Tensor");
}

TORCH_LIBRARY_IMPL(approxtorch, CUDA, m) {
    m.impl("elementwise_mul", &elementwise_mul_cuda);
}
