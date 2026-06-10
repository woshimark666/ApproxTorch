// claude-optimized fused per-tensor symmetric fake-quantization.
//
// Reference (python, nn/fakequant.py::_symmetric_static_quantize_int8_per_tensor):
//   forward:  s = clamp(scale, 1e-12); v = x/s; save v;
//             q = clamp(round(v), qmin, qmax)            (3 big kernels,
//                                                          saves fp32 v)
//   backward: mask = (v >= qmin) & (v <= qmax);
//             gx = go * mask / s                          (~5 big kernels)
//
// Fused version:
//   forward:  ONE kernel -> (q, bool mask); saves the 1-byte mask instead
//             of the fp32 pre-round tensor (4x less saved-activation memory)
//   backward: ONE kernel -> gx = (mask ? go : 0) / s
//
// Bit-identical to the reference: same fp32 division, round-half-even
// (nearbyintf == torch.round), NaN-preserving comparison-based clamp, same
// clamped scale in forward and backward.

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace {

constexpr int FQ_THREADS = 256;

inline int fq_grid(long long work) {
    long long b = (work + FQ_THREADS - 1) / FQ_THREADS;
    return (int)std::min<long long>(b, 4096);
}

// float4/uchar4 global accesses require 16/4-byte aligned addresses, but
// contiguous() can return a view with nonzero storage_offset whose data_ptr
// is only element-aligned. Vectorize only when every pointer qualifies
// (fresh allocations from the caching allocator always do).
inline bool fq_aligned(const void* p, uintptr_t bytes) {
    return (reinterpret_cast<uintptr_t>(p) & (bytes - 1)) == 0;
}

__device__ __forceinline__ float fq_one(float xv, float s,
                                        float qmin, float qmax, bool& m) {
    float v = xv / s;
    m = (v >= qmin) && (v <= qmax);
    float r = nearbyintf(v);             // round half to even == torch.round
    r = r < qmin ? qmin : r;             // NaN falls through both (== torch.clamp)
    r = r > qmax ? qmax : r;
    return r;
}

template <int VEC>
__global__ void fakequant_pt_fwd_kernel(
    const float* __restrict__ x,
    const float* __restrict__ scale,     // 1 element
    float* __restrict__ q,
    bool* __restrict__ mask,
    long long numel, float qmin, float qmax
) {
    const float s = fmaxf(scale[0], 1e-12f);
    const long long stride = (long long)gridDim.x * blockDim.x;
    const long long base = (long long)blockIdx.x * blockDim.x + threadIdx.x;

    const long long n4 = (VEC == 4) ? (numel >> 2) : 0;
    if constexpr (VEC == 4) {
        const float4* x4 = reinterpret_cast<const float4*>(x);
        float4* q4 = reinterpret_cast<float4*>(q);
        uchar4* m4 = reinterpret_cast<uchar4*>(mask);
        for (long long i = base; i < n4; i += stride) {
            float4 v = x4[i];
            float4 r; bool m0, m1, m2, m3;
            r.x = fq_one(v.x, s, qmin, qmax, m0);
            r.y = fq_one(v.y, s, qmin, qmax, m1);
            r.z = fq_one(v.z, s, qmin, qmax, m2);
            r.w = fq_one(v.w, s, qmin, qmax, m3);
            q4[i] = r;
            m4[i] = make_uchar4(m0, m1, m2, m3);
        }
    }
    for (long long i = (n4 << 2) + base; i < numel; i += stride) {
        bool m;
        q[i] = fq_one(x[i], s, qmin, qmax, m);
        mask[i] = m;
    }
}

template <int VEC>
__global__ void fakequant_pt_bwd_kernel(
    const float* __restrict__ go,
    const bool* __restrict__ mask,
    const float* __restrict__ scale,     // 1 element
    float* __restrict__ gx,
    long long numel
) {
    const float s = fmaxf(scale[0], 1e-12f);
    const long long stride = (long long)gridDim.x * blockDim.x;
    const long long base = (long long)blockIdx.x * blockDim.x + threadIdx.x;

    const long long n4 = (VEC == 4) ? (numel >> 2) : 0;
    if constexpr (VEC == 4) {
        const float4* g4 = reinterpret_cast<const float4*>(go);
        const uchar4* m4 = reinterpret_cast<const uchar4*>(mask);
        float4* o4 = reinterpret_cast<float4*>(gx);
        for (long long i = base; i < n4; i += stride) {
            float4 g = g4[i];
            uchar4 m = m4[i];
            float4 r;
            r.x = (m.x ? g.x : 0.0f) / s;
            r.y = (m.y ? g.y : 0.0f) / s;
            r.z = (m.z ? g.z : 0.0f) / s;
            r.w = (m.w ? g.w : 0.0f) / s;
            o4[i] = r;
        }
    }
    for (long long i = (n4 << 2) + base; i < numel; i += stride) {
        gx[i] = (mask[i] ? go[i] : 0.0f) / s;
    }
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> fakequant_per_tensor_claude(
    torch::Tensor x, torch::Tensor scale, int64_t qmin, int64_t qmax
) {
    TORCH_CHECK(x.is_cuda() && scale.is_cuda(), "x and scale must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(scale.scalar_type() == torch::kFloat32, "scale must be float32");
    TORCH_CHECK(scale.numel() == 1, "scale must be a scalar tensor");

    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
    auto xc = x.contiguous();
    auto sc = scale.contiguous();
    auto q = at::empty_like(xc);
    auto mask = at::empty(xc.sizes(), xc.options().dtype(torch::kBool));

    const long long n = xc.numel();
    if (n > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        const bool vec = fq_aligned(xc.data_ptr<float>(), 16) &&
                         fq_aligned(q.data_ptr<float>(), 16) &&
                         fq_aligned(mask.data_ptr<bool>(), 4);
        if (vec) {
            fakequant_pt_fwd_kernel<4><<<fq_grid(n / 4 + 1), FQ_THREADS, 0, stream>>>(
                xc.data_ptr<float>(), sc.data_ptr<float>(),
                q.data_ptr<float>(), mask.data_ptr<bool>(),
                n, (float)qmin, (float)qmax);
        } else {
            fakequant_pt_fwd_kernel<1><<<fq_grid(n), FQ_THREADS, 0, stream>>>(
                xc.data_ptr<float>(), sc.data_ptr<float>(),
                q.data_ptr<float>(), mask.data_ptr<bool>(),
                n, (float)qmin, (float)qmax);
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return std::make_tuple(q, mask);
}

torch::Tensor fakequant_per_tensor_backward_claude(
    torch::Tensor grad_output, torch::Tensor mask, torch::Tensor scale
) {
    TORCH_CHECK(grad_output.is_cuda() && mask.is_cuda() && scale.is_cuda(),
                "all tensors must be CUDA");
    TORCH_CHECK(grad_output.scalar_type() == torch::kFloat32, "grad_output must be float32");
    TORCH_CHECK(mask.scalar_type() == torch::kBool, "mask must be bool");
    TORCH_CHECK(scale.scalar_type() == torch::kFloat32 && scale.numel() == 1,
                "scale must be a float32 scalar tensor");
    TORCH_CHECK(grad_output.sizes() == mask.sizes(), "grad_output/mask shape mismatch");

    const at::cuda::OptionalCUDAGuard device_guard(device_of(grad_output));
    auto go = grad_output.contiguous();
    auto mc = mask.contiguous();
    auto sc = scale.contiguous();
    auto gx = at::empty_like(go);

    const long long n = go.numel();
    if (n > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        const bool vec = fq_aligned(go.data_ptr<float>(), 16) &&
                         fq_aligned(mc.data_ptr<bool>(), 4) &&
                         fq_aligned(gx.data_ptr<float>(), 16);
        if (vec) {
            fakequant_pt_bwd_kernel<4><<<fq_grid(n / 4 + 1), FQ_THREADS, 0, stream>>>(
                go.data_ptr<float>(), mc.data_ptr<bool>(), sc.data_ptr<float>(),
                gx.data_ptr<float>(), n);
        } else {
            fakequant_pt_bwd_kernel<1><<<fq_grid(n), FQ_THREADS, 0, stream>>>(
                go.data_ptr<float>(), mc.data_ptr<bool>(), sc.data_ptr<float>(),
                gx.data_ptr<float>(), n);
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return gx;
}

TORCH_LIBRARY_FRAGMENT(approxtorch, m) {
    m.def("fakequant_per_tensor_claude(Tensor x, Tensor scale, int qmin, int qmax) -> (Tensor, Tensor)");
    m.def("fakequant_per_tensor_backward_claude(Tensor grad_output, Tensor mask, Tensor scale) -> Tensor");
}

TORCH_LIBRARY_IMPL(approxtorch, CUDA, m) {
    m.impl("fakequant_per_tensor_claude", &fakequant_per_tensor_claude);
    m.impl("fakequant_per_tensor_backward_claude", &fakequant_per_tensor_backward_claude);
}
