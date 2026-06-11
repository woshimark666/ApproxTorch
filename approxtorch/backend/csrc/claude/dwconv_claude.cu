// claude depthwise approximate-multiplication conv2d forward (LUT-based).
//
//   y[n,c,oh,ow] = sum_{i,j} lut[ q(x[n,c, oh*sh-ph+i*dh, ow*sw-pw+j*dw]),
//                                 q(w[c,i,j]) ]
//
// Out-of-bounds x positions reproduce the im2col pipeline's zero padding:
// quantized zero == LUT row index 128 (lut[128][w] is generally nonzero for
// an approximate multiplier), NOT a zero contribution. Accumulation walks
// the taps in (i, j) row-major ascending order with plain float adds — the
// same order a per-group im2col GEMM would use — so y is bit-identical to
// running the main LUT-BGEMM per channel.
//
// Why a dedicated kernel: with groups == C the im2col GEMM degenerates to
// K = kh*kw and O = 1 per group, which the tiled BGEMM cannot tile. Here a
// warp's lanes share the channel (same weight index) but differ in pixel
// (different x index), so gathers from the TRANSPOSED table lutT[w][x] stay
// inside one 1KB row (512B for the int16 image): the per-channel hot set is
// just kh*kw rows and lives in L1. The int16 image is used when the table
// is integer-valued (validated on device, grid-uniform flag, no host sync —
// same scheme as the main bgemm).
//
// Registered op:
//   dwconv_fake_int8_claude(x [N,C,H,W] f32|i8|u8, w [C, kh*kw] f32
//                           (quantized integer values), lut [65536] f32,
//                           kh, kw, sh, sw, ph, pw, dilh, dilw)
//       -> (y [N,C,OH,OW] f32, wq [C, kh*kw] u8 LUT indices)
// wq is returned so the training Function can save it for the backward
// instead of keeping the fp32 weight alive (same contract as the bgemm
// _save op).

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace claude_dwconv {

constexpr int THREADS = 256;

__device__ __forceinline__ int lut_idx_of(float v) {
    int q = __float2int_rn(v);
    return max(-128, min(127, q)) + 128;
}
__device__ __forceinline__ int lut_idx_of(int8_t v)  { return (int)v + 128; }
__device__ __forceinline__ int lut_idx_of(uint8_t v) { return (int)v; }

// fp32 quantized values -> u8 LUT index (round + clamp + 128), same formula
// as the main bgemm prepass
__global__ void quantize_w_kernel(const float* __restrict__ w,
                                  uint8_t* __restrict__ wq, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const int q = __float2int_rn(w[i]) + 128;
        wq[i] = (uint8_t)max(0, min(255, q));
    }
}

// Transposed LUT images ([x][w] -> [w][x]) + int16 validity flag. Same pass
// as bgemm_float_claude.cu's prepare_lut_kernel<true>; grid covers 65536.
__global__ void prepare_lutT_kernel(const float* __restrict__ in,
                                    float* __restrict__ outf,
                                    short* __restrict__ out16,
                                    int* __restrict__ bad) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const float v = in[i];
    const float r = rintf(v);
    const bool good = (v == r) && (fabsf(v) <= 32767.0f);
    if (!good) *bad = 1;
    const int o = ((i & 255) << 8) | (i >> 8);
    outf[o]  = v;
    out16[o] = good ? (short)r : 0;
}

// TKH/TKW > 0: compile-time kernel size (3x3 hot case: tap loop fully
// unrolled, weight rows hoisted to registers). TKH == 0: runtime sizes.
template <typename T, typename LUT_T, int TKH, int TKW>
__device__ __forceinline__ void dw_mainloop(
    const T* __restrict__ x, const uint8_t* __restrict__ wq,
    const LUT_T* __restrict__ lutT, float* __restrict__ y,
    int N, int C, int H, int W, int c,
    int KH, int KW, int SH, int SW, int PH, int PW, int DH, int DW,
    int OH, int OW)
{
    const int L = OH * OW;
    const long long NL = (long long)N * L;
    const long long rstride = (long long)gridDim.x * blockDim.x;

    if constexpr (TKH > 0) {
        int wrow[TKH * TKW];
#pragma unroll
        for (int t = 0; t < TKH * TKW; ++t) {
            wrow[t] = (int)__ldg(wq + (long long)c * TKH * TKW + t) << 8;
        }
        for (long long r = (long long)blockIdx.x * blockDim.x + threadIdx.x;
             r < NL; r += rstride) {
            const int n = (int)(r / L);
            const int l = (int)(r - (long long)n * L);
            const int oh = l / OW;
            const int ow = l - oh * OW;
            const T* xn = x + ((long long)n * C + c) * H * W;
            const int ih0 = oh * SH - PH;
            const int iw0 = ow * SW - PW;
            float acc = 0.0f;
#pragma unroll
            for (int i = 0; i < TKH; ++i) {
                const int ih = ih0 + i * DH;
                const bool rowok = (unsigned)ih < (unsigned)H;
                const T* xrow = xn + (long long)ih * W;
#pragma unroll
                for (int j = 0; j < TKW; ++j) {
                    const int iw = iw0 + j * DW;
                    int xi = 128;  // zero padding -> quantized zero
                    if (rowok && (unsigned)iw < (unsigned)W) {
                        xi = lut_idx_of(xrow[iw]);
                    }
                    acc += (float)__ldg(lutT + wrow[i * TKW + j] + xi);
                }
            }
            y[((long long)n * C + c) * L + l] = acc;
        }
    } else {
        const uint8_t* wc = wq + (long long)c * KH * KW;
        for (long long r = (long long)blockIdx.x * blockDim.x + threadIdx.x;
             r < NL; r += rstride) {
            const int n = (int)(r / L);
            const int l = (int)(r - (long long)n * L);
            const int oh = l / OW;
            const int ow = l - oh * OW;
            const T* xn = x + ((long long)n * C + c) * H * W;
            const int ih0 = oh * SH - PH;
            const int iw0 = ow * SW - PW;
            float acc = 0.0f;
            for (int i = 0; i < KH; ++i) {
                const int ih = ih0 + i * DH;
                const bool rowok = (unsigned)ih < (unsigned)H;
                const T* xrow = xn + (long long)ih * W;
                for (int j = 0; j < KW; ++j) {
                    const int iw = iw0 + j * DW;
                    int xi = 128;
                    if (rowok && (unsigned)iw < (unsigned)W) {
                        xi = lut_idx_of(xrow[iw]);
                    }
                    const int wr = (int)__ldg(wc + i * KW + j) << 8;
                    acc += (float)__ldg(lutT + wr + xi);
                }
            }
            y[((long long)n * C + c) * L + l] = acc;
        }
    }
}

template <typename T, int TKH, int TKW>
__global__ void __launch_bounds__(THREADS)
dwconv_lut_kernel(
    const T* __restrict__ x, const uint8_t* __restrict__ wq,
    const float* __restrict__ lutTf, const short* __restrict__ lutT16,
    const int* __restrict__ lutT16_bad,
    float* __restrict__ y,
    int N, int C, int H, int W,
    int KH, int KW, int SH, int SW, int PH, int PW, int DH, int DW,
    int OH, int OW)
{
    const int c = blockIdx.y;
    // grid-uniform branch: the flag is identical for every block
    if (__ldg(lutT16_bad) == 0) {
        dw_mainloop<T, short, TKH, TKW>(x, wq, lutT16, y, N, C, H, W, c,
            KH, KW, SH, SW, PH, PW, DH, DW, OH, OW);
    } else {
        dw_mainloop<T, float, TKH, TKW>(x, wq, lutTf, y, N, C, H, W, c,
            KH, KW, SH, SW, PH, PW, DH, DW, OH, OW);
    }
}

std::tuple<torch::Tensor, torch::Tensor> dwconv_impl(
    const torch::Tensor& x_in, const torch::Tensor& w_in,
    const torch::Tensor& lut_in,
    int64_t kh, int64_t kw, int64_t sh, int64_t sw,
    int64_t ph, int64_t pw, int64_t dilh, int64_t dilw)
{
    TORCH_CHECK(x_in.is_cuda() && w_in.is_cuda() && lut_in.is_cuda(),
                "all tensors must be CUDA");
    TORCH_CHECK(x_in.scalar_type() == torch::kFloat32
                || x_in.scalar_type() == torch::kChar
                || x_in.scalar_type() == torch::kByte,
                "x must be float32, int8 or uint8");
    TORCH_CHECK(w_in.scalar_type() == torch::kFloat32, "w must be float32");
    TORCH_CHECK(lut_in.scalar_type() == torch::kFloat32, "lut must be float32");
    TORCH_CHECK(x_in.dim() == 4, "x must have shape [N, C, H, W]");
    TORCH_CHECK(lut_in.numel() == 256 * 256, "lut must have 65536 elements");
    TORCH_CHECK(kh >= 1 && kw >= 1 && sh >= 1 && sw >= 1 && ph >= 0 && pw >= 0
                && dilh >= 1 && dilw >= 1, "invalid conv geometry");

    const at::cuda::OptionalCUDAGuard device_guard(device_of(x_in));
    auto x = x_in.contiguous();
    auto w = w_in.contiguous();
    auto lut = lut_in.contiguous();

    const int64_t N = x.size(0);
    const int64_t C = x.size(1);
    const int64_t H = x.size(2);
    const int64_t W = x.size(3);
    const int64_t kk = kh * kw;
    TORCH_CHECK(w.dim() == 2 && w.size(0) == C && w.size(1) == kk,
                "w must have shape [C, kh*kw] (depthwise, channel multiplier 1), got ",
                w.sizes());
    TORCH_CHECK(C <= 65535, "C too large");

    const int64_t OH = (H + 2 * ph - dilh * (kh - 1) - 1) / sh + 1;
    const int64_t OW = (W + 2 * pw - dilw * (kw - 1) - 1) / sw + 1;
    TORCH_CHECK(OH >= 1 && OW >= 1, "conv geometry yields empty output");
    const int64_t L = OH * OW;
    TORCH_CHECK(N * L <= INT32_MAX * 32LL, "N * L too large");

    auto fopts = x.options().dtype(torch::kFloat32);
    auto u8opts = x.options().dtype(torch::kUInt8);
    auto y = torch::empty({N, C, OH, OW}, fopts);
    auto wq = torch::empty({C, kk}, u8opts);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    {
        const int n = (int)(C * kk);
        quantize_w_kernel<<<(n + THREADS - 1) / THREADS, THREADS, 0, stream>>>(
            w.data_ptr<float>(), wq.data_ptr<uint8_t>(), n);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    if (y.numel() == 0) return std::make_tuple(y, wq);

    auto lutT = torch::empty({256 * 256}, fopts);
    auto lutT16 = torch::empty({256 * 256}, x.options().dtype(torch::kInt16));
    auto lutT16_bad = torch::zeros({1}, x.options().dtype(torch::kInt32));
    prepare_lutT_kernel<<<256, 256, 0, stream>>>(
        lut.data_ptr<float>(), lutT.data_ptr<float>(),
        lutT16.data_ptr<short>(), lutT16_bad.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // gathers hit L1 (kh*kw hot LUT rows per channel), so the launch only
    // needs enough blocks to fill the GPU: cap the total and grid-stride
    // over the flattened (n, oh, ow) axis per channel
    const int64_t rtiles = (N * L + THREADS - 1) / THREADS;
    const int64_t gx = std::min(rtiles, std::max<int64_t>(1, 16384 / C));
    dim3 grid((unsigned)gx, (unsigned)C);

    auto launch = [&](auto* xp) {
        using T = std::remove_const_t<std::remove_pointer_t<decltype(xp)>>;
        auto args = [&](auto kern) {
            kern<<<grid, THREADS, 0, stream>>>(
                xp, wq.data_ptr<uint8_t>(),
                lutT.data_ptr<float>(), lutT16.data_ptr<short>(),
                lutT16_bad.data_ptr<int>(), y.data_ptr<float>(),
                (int)N, (int)C, (int)H, (int)W,
                (int)kh, (int)kw, (int)sh, (int)sw,
                (int)ph, (int)pw, (int)dilh, (int)dilw, (int)OH, (int)OW);
        };
        if (kh == 3 && kw == 3) args(dwconv_lut_kernel<T, 3, 3>);
        else                    args(dwconv_lut_kernel<T, 0, 0>);
    };
    if (x.scalar_type() == torch::kChar) launch(x.data_ptr<int8_t>());
    else if (x.scalar_type() == torch::kByte) launch(x.data_ptr<uint8_t>());
    else launch(x.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return std::make_tuple(y, wq);
}

}  // namespace claude_dwconv

std::tuple<torch::Tensor, torch::Tensor> dwconv_fake_int8_claude(
    torch::Tensor x, torch::Tensor w, torch::Tensor lut,
    int64_t kh, int64_t kw, int64_t sh, int64_t sw,
    int64_t ph, int64_t pw, int64_t dilh, int64_t dilw)
{
    return claude_dwconv::dwconv_impl(x, w, lut, kh, kw, sh, sw, ph, pw,
                                      dilh, dilw);
}

TORCH_LIBRARY_FRAGMENT(approxtorch, m) {
    m.def("dwconv_fake_int8_claude(Tensor x, Tensor w, Tensor lut, int kh, int kw, int sh, int sw, int ph, int pw, int dilh, int dilw) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(approxtorch, CUDA, m) {
    m.impl("dwconv_fake_int8_claude", &dwconv_fake_int8_claude);
}
