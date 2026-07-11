// Optimized LUT-based approximate-multiplication BGEMM (fake-int8, float storage).
//
// Functional reference: ../cuda/bgemm_float_gpt.cu
//   y[n, o, l] = sum_k lut[(round(x[n,k,l]) + 128) * 256 + (round(w[o,k]) + 128)]
//
// Interface is identical to the reference:
//   bgemm_fake_int8_forward_cuda_claude(Tensor x[N,K,L] f32, Tensor w[O,K] f32,
//                                       Tensor lut[65536] f32) -> Tensor y[N,O,L] f32
//
// uint8 variant (uint8 x uint8 approximate multiplier, asymmetric quant):
//   bgemm_fake_uint8_forward_cuda_claude(x, w, lut)
//     y[n, o, l] = sum_k lut[round(x[n,k,l]) * 256 + round(w[o,k])]
//   BOTH operands hold unsigned quantized values in [0, 255]; the LUT index
//   is the raw value itself (no +128 offset on either side), layout
//   lut[x_u8][w_u8]. Zero-point corrections are the caller's job (nn layer).
//   Only the prepass offsets differ (0/0 vs 128/128); the main kernel is
//   byte-index based and shared verbatim, so every mode / tiling / split-K
//   property applies to both ops.
//
// Key optimizations vs the reference (kernels + tuning live in
// bgemm_lut_kernels.cuh, shared with gemm.cu; details in NOTES.md):
//  1. One cheap prepass quantizes x and w to uint8 once. The main kernel then
//     re-reads 1-byte instead of 4-byte elements, the float->int conversion
//     leaves the hot loop, and the per-call `w.t().contiguous()` transpose of
//     the reference disappears.
//  2. Register tiling: each thread computes TM x TN outputs.
//  3. Warp layout: 32 threads of a warp span the column direction, so for a
//     fixed (k, tm) every lane gathers from the SAME 1KB LUT row -> minimal
//     sector divergence and high L1 reuse on the LUT gathers (the bottleneck).
//     The wider the block's column tile, the better each fetched LUT row is
//     amortized, so the warp/column direction is put along the LARGEST
//     available dimension:
//       NFLAT: rows = N*L (flattened), cols = O     (large O)
//       SFLAT: rows = O, cols = N*L (flattened), transposed LUT (small O)
//       XMK:   rows = N, cols = O                   (L == 1, GEMV-like)
//     Flattening (n, l) into one axis means narrow per-image L never strands
//     threads or block tiles.
//  4. Templated tile sizes dispatched by shape (see pick_cfg / NOTES.md).
//  5. Split-K when the natural grid is too small to fill the GPU: partial
//     sums go to a workspace and a deterministic reduction produces y.
//
// Without split-K the accumulation order over k matches the reference
// (ascending k, plain float adds), so results are bit-identical to it. With
// split-K the association changes (chunk partials are reduced in ascending
// chunk order); results stay deterministic and are exact whenever the LUT
// holds integers and intermediate sums stay below 2^24 (the common case for
// 8x8 approximate-multiplier tables).
//
// The 16-bit L1-resident LUT image covers both signed tables (int16, values
// in [-32767, 32767]) and unsigned uint8 x uint8 tables (uint16, values up
// to 65535 >= 255*255); see prepare_lut_kernel. Only mixed-sign tables
// exceeding int16 fall back to the 256KB float image.

#include "bgemm_lut_kernels.cuh"

namespace claude_bgemm {

// also returns the internal u8 quantized images (xq [N,K,L], wq [O,K]) so a
// training Function can save them for the LRE backward instead of re-casting
// the fp32 activations (4x smaller payload, zero extra kernels).
//
// x is accepted as float32 (quantized integer values; prepass converts to u8
// LUT indices) OR uint8 (already LUT indices, e.g. from im2col_u8; the x
// prepass is skipped entirely and the returned xq aliases the input).
//
// x_offset / w_offset select the operand semantics: 128 = signed int8 values
// [-128,127], 0 = unsigned uint8 values [0,255]. int8 op: (128, 128);
// uint8 op: (0, 0). Only the fp32 prepasses use them; a uint8-dtype x is
// already LUT indices under either convention.
static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> bgemm_forward_save_cfg_impl(
    const torch::Tensor& x,
    const torch::Tensor& w,
    const torch::Tensor& lut,
    int64_t cfg,
    int x_offset,
    int w_offset)
{
    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));

    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(w.is_cuda(), "w must be a CUDA tensor");
    TORCH_CHECK(lut.is_cuda(), "lut must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32
                || x.scalar_type() == torch::kByte,
                "x must be float32 or uint8 (LUT indices)");
    TORCH_CHECK(w.scalar_type() == torch::kFloat32, "w must be float32");
    TORCH_CHECK(lut.scalar_type() == torch::kFloat32, "lut must be float32");
    TORCH_CHECK(x.dim() == 3, "x must have shape [N, CKK, L]");
    TORCH_CHECK(w.dim() == 2, "w must have shape [O, CKK]");
    TORCH_CHECK(lut.numel() == 256 * 256, "lut must have 65536 elements");
    TORCH_CHECK(x.size(1) == w.size(1), "x.size(1) must equal w.size(1)");

    auto xc = x.contiguous();
    auto wc = w.contiguous();
    auto lutc = lut.contiguous();

    const int N = static_cast<int>(xc.size(0));
    const int K = static_cast<int>(xc.size(1));
    const int L = static_cast<int>(xc.size(2));
    const int O = static_cast<int>(wc.size(0));
    const long long NL = static_cast<long long>(N) * L;
    TORCH_CHECK(NL <= INT32_MAX, "N * L too large");

    auto stream = at::cuda::getCurrentCUDAStream();
    auto f32opts = xc.options().dtype(torch::kFloat32);
    auto u8opts = xc.options().dtype(torch::kUInt8);

    // prepass: quantize x and w to uint8 LUT indices (skipped for x when the
    // caller already provides the u8 index image)
    torch::Tensor xq;
    auto wq = torch::empty({O, K}, u8opts);
    {
        const int threads = 256;
        auto launch_quant = [&](const float* in, uint8_t* out, long long n, int offset) {
            const bool vec = ptr_aligned(in, 16) && ptr_aligned(out, 4);
            const int blocks = static_cast<int>(
                std::min<long long>(n / (vec ? 4 : 1) / threads + 1, 4096));
            if (vec) quantize_to_u8_kernel<4><<<blocks, threads, 0, stream>>>(in, out, n, offset);
            else     quantize_to_u8_kernel<1><<<blocks, threads, 0, stream>>>(in, out, n, offset);
        };
        if (xc.scalar_type() == torch::kByte) {
            xq = xc;
        } else {
            xq = torch::empty({N, K, L}, u8opts);
            launch_quant(xc.data_ptr<float>(), xq.data_ptr<uint8_t>(),
                         static_cast<long long>(N) * K * L, x_offset);
        }
        launch_quant(wc.data_ptr<float>(), wq.data_ptr<uint8_t>(),
                     static_cast<long long>(O) * K, w_offset);
    }

    auto y = torch::empty({N, O, L}, f32opts);

    // mode selection: cfg -1 = auto; 0..14 = force NFLAT/XMK cfg;
    // 100..114 = force SFLAT cfg (tuning hooks for the bench harness)
    int mode;
    int c = static_cast<int>(cfg);
    if (L == 1) {
        mode = XMK;
        if (c >= 100) c -= 100;
    } else if (c >= 100) {
        mode = SFLAT;
        c -= 100;
    } else if (c >= 0) {
        mode = NFLAT;
    } else {
        // auto: put the warp along the larger of (O, N*L)
        mode = (NL >= 2 * O) ? SFLAT : NFLAT;
    }
    if (mode == NFLAT && (NL + 7) / 8 > 65535) {
        mode = SFLAT;   // NFLAT row-tile count would overflow grid.y
        c = (c >= 0) ? -1 : c;
    }

    LaunchArgs a{
        nullptr, nullptr, lutc.data_ptr<float>(), nullptr, nullptr,
        y.data_ptr<float>(),
        K, 0, 0, L, O, f32opts, stream.stream()
    };

    // LUT images: 16-bit (+ transposed copies for SFLAT) and validity flags
    // [0]: int16 interpretation invalid, [1]: uint16 interpretation invalid
    auto lut16 = torch::empty({256 * 256}, u8opts.dtype(torch::kInt16));
    auto lut16_bad = torch::zeros({2}, u8opts.dtype(torch::kInt32));
    torch::Tensor lutT;   // keep alive until kernel runs
    a.lut16 = lut16.data_ptr<short>();
    a.lut16_bad = lut16_bad.data_ptr<int>();

    if (mode == XMK) {
        a.rowsrc = xq.data_ptr<uint8_t>();
        a.colsrc = wq.data_ptr<uint8_t>();
        a.R = N; a.C = O;
    } else if (mode == NFLAT) {
        a.rowsrc = xq.data_ptr<uint8_t>();
        a.colsrc = wq.data_ptr<uint8_t>();
        a.R = static_cast<int>(NL); a.C = O;
    } else {
        a.rowsrc = wq.data_ptr<uint8_t>();
        a.colsrc = xq.data_ptr<uint8_t>();
        a.R = O; a.C = static_cast<int>(NL);
    }
    if (mode == SFLAT) {
        lutT = torch::empty_like(lutc);
        prepare_lut_kernel<true><<<256, 256, 0, stream>>>(
            lutc.data_ptr<float>(), lutT.data_ptr<float>(),
            lut16.data_ptr<short>(), lut16_bad.data_ptr<int>());
        a.lut = lutT.data_ptr<float>();
    } else {
        prepare_lut_kernel<false><<<256, 256, 0, stream>>>(
            lutc.data_ptr<float>(), nullptr,
            lut16.data_ptr<short>(), lut16_bad.data_ptr<int>());
    }



    if (c < 0) c = (mode == NFLAT) ? pick_cfg_nflat(a.R, a.C)
                                   : pick_cfg_sflat(a.R, a.C);

    if (mode == XMK)         dispatch_cfg<XMK>(c, a);
    else if (mode == NFLAT)  dispatch_cfg<NFLAT>(c, a);
    else                     dispatch_cfg<SFLAT>(c, a);

    CHECK_CUDA_ERROR();
    return std::make_tuple(y, xq, wq);
}

// ---- int8 entry points (x offset 128, unchanged behavior) ----

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> bgemm_lut_forward_cuda_claude_save_cfg(
    const torch::Tensor& x,
    const torch::Tensor& w,
    const torch::Tensor& lut,
    int64_t cfg)
{
    return bgemm_forward_save_cfg_impl(x, w, lut, cfg, 128, 128);
}

torch::Tensor bgemm_lut_forward_cuda_claude_cfg(
    const torch::Tensor& x,
    const torch::Tensor& w,
    const torch::Tensor& lut,
    int64_t cfg)
{
    return std::get<0>(bgemm_lut_forward_cuda_claude_save_cfg(x, w, lut, cfg));
}

torch::Tensor bgemm_lut_forward_cuda_claude(
    const torch::Tensor& x,
    const torch::Tensor& w,
    const torch::Tensor& lut)
{
    return bgemm_lut_forward_cuda_claude_cfg(x, w, lut, -1);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> bgemm_lut_forward_cuda_claude_save(
    const torch::Tensor& x,
    const torch::Tensor& w,
    const torch::Tensor& lut)
{
    return bgemm_lut_forward_cuda_claude_save_cfg(x, w, lut, -1);
}

// ---- uint8 entry points (offsets 0/0: uint8 x uint8, raw values as indices) ----

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> bgemm_lut_forward_cuda_claude_u8_save_cfg(
    const torch::Tensor& x,
    const torch::Tensor& w,
    const torch::Tensor& lut,
    int64_t cfg)
{
    return bgemm_forward_save_cfg_impl(x, w, lut, cfg, 0, 0);
}

torch::Tensor bgemm_lut_forward_cuda_claude_u8_cfg(
    const torch::Tensor& x,
    const torch::Tensor& w,
    const torch::Tensor& lut,
    int64_t cfg)
{
    return std::get<0>(bgemm_lut_forward_cuda_claude_u8_save_cfg(x, w, lut, cfg));
}

torch::Tensor bgemm_lut_forward_cuda_claude_u8(
    const torch::Tensor& x,
    const torch::Tensor& w,
    const torch::Tensor& lut)
{
    return bgemm_lut_forward_cuda_claude_u8_cfg(x, w, lut, -1);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> bgemm_lut_forward_cuda_claude_u8_save(
    const torch::Tensor& x,
    const torch::Tensor& w,
    const torch::Tensor& lut)
{
    return bgemm_lut_forward_cuda_claude_u8_save_cfg(x, w, lut, -1);
}

} // namespace claude_bgemm

TORCH_LIBRARY_FRAGMENT(approxtorch, m){
    m.def("bgemm_fake_int8_forward_cuda_claude(Tensor x, Tensor w, Tensor lut) -> Tensor");
    m.def("bgemm_fake_int8_forward_cuda_claude_cfg(Tensor x, Tensor w, Tensor lut, int cfg) -> Tensor");
    m.def("bgemm_fake_int8_forward_cuda_claude_save(Tensor x, Tensor w, Tensor lut) -> (Tensor, Tensor, Tensor)");
    m.def("bgemm_fake_uint8_forward_cuda_claude(Tensor x, Tensor w, Tensor lut) -> Tensor");
    m.def("bgemm_fake_uint8_forward_cuda_claude_cfg(Tensor x, Tensor w, Tensor lut, int cfg) -> Tensor");
    m.def("bgemm_fake_uint8_forward_cuda_claude_save(Tensor x, Tensor w, Tensor lut) -> (Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(approxtorch, CUDA, m){
    m.impl("bgemm_fake_int8_forward_cuda_claude", &claude_bgemm::bgemm_lut_forward_cuda_claude);
    m.impl("bgemm_fake_int8_forward_cuda_claude_cfg", &claude_bgemm::bgemm_lut_forward_cuda_claude_cfg);
    m.impl("bgemm_fake_int8_forward_cuda_claude_save", &claude_bgemm::bgemm_lut_forward_cuda_claude_save);
    m.impl("bgemm_fake_uint8_forward_cuda_claude", &claude_bgemm::bgemm_lut_forward_cuda_claude_u8);
    m.impl("bgemm_fake_uint8_forward_cuda_claude_cfg", &claude_bgemm::bgemm_lut_forward_cuda_claude_u8_cfg);
    m.impl("bgemm_fake_uint8_forward_cuda_claude_save", &claude_bgemm::bgemm_lut_forward_cuda_claude_u8_save);
}
