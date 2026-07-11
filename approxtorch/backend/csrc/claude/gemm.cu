// LUT-based approximate-multiplication GEMM (fake-quant, float storage),
// built on the shared BGEMM machinery in bgemm_lut_kernels.cuh (tuned tile
// configs, warp-along-largest-dim layout, split-K, 16-bit L1-resident LUT
// images — design notes and measurements in NOTES.md).
//
//   gemm_fake_int8_forward_cuda_claude (A, B, lut):    int8 x int8
//       y[m, n] = sum_k lut[(round(A[m,k]) + 128) * 256 + (round(B[k,n]) + 128)]
//   gemm_fake_uint8_forward_cuda_claude(A, B, lut):    uint8 x uint8
//       y[m, n] = sum_k lut[round(A[m,k]) * 256 + round(B[k,n])]
//
// torch.mm layout: A [M, K], B [K, N] -> y [M, N] fp32.
//   A: float32 (quantized integer values) or uint8 (already LUT indices;
//      the prepass is skipped and the u8 image aliases the input).
//   B: float32 or uint8, ANY 2-D strides. The kernel wants both operands
//      K-contiguous, so the required [K,N] -> [N,K] transpose is FUSED into
//      the quantize prepass (tiled through smem, one 1-byte write per
//      element) instead of a fp32 B.t().contiguous(). A transposed weight
//      view B = W.t() therefore costs exactly the same as a contiguous B.
//
// Mode selection puts the warp/column tile along the LARGER of (M, N), the
// same trick the BGEMM uses for (O, N*L) — the wider the column tile, the
// better each fetched LUT row is amortized:
//   XMK:          rows = M (rowsrc Aq [M,K]), cols = N (colsrc Bq [N,K])
//   SFLAT (L=1):  rows = N (rowsrc Bq),       cols = M (colsrc Aq),
//                 transposed LUT keeps summands identical; its epilogue with
//                 L=1 writes off = c*O + r = m*N + n, i.e. row-major y [M,N].
// grid.y caps rows at 65535 * BM; if one orientation exceeds it the other is
// forced, and only min(M, N) > ~524k is rejected outright (TORCH_CHECK).
//
// Degenerate shapes are handled, not crashed on: M == 0 / N == 0 return an
// empty y before any launch; K == 0 runs the main kernel with an empty
// k-range and writes exact zeros.
//
// _cfg variants: cfg -1 = auto, 0..18 = force XMK with that tile config,
// 100+ = force the swapped (SFLAT) orientation (bench hooks, ids match the
// BGEMM). _save variants also return the u8 operand images for a training
// Function to keep for its backward: aq [M,K] and bq [N,K] — note bq is the
// TRANSPOSED quantized image of B.

#include "bgemm_lut_kernels.cuh"

namespace claude_gemm {

using claude_bgemm::ceil_div;
using claude_bgemm::ptr_aligned;
using claude_bgemm::quantize_one;
using claude_bgemm::quantize_to_u8_kernel;
using claude_bgemm::prepare_lut_kernel;
using claude_bgemm::LaunchArgs;
using claude_bgemm::dispatch_cfg;
using claude_bgemm::pick_cfg_sflat;
using claude_bgemm::XMK;
using claude_bgemm::SFLAT;

__device__ __forceinline__ uint8_t to_idx(float v, int offset)
{
    return quantize_one(v, offset);
}
__device__ __forceinline__ uint8_t to_idx(uint8_t v, int) { return v; }

// Fused quantize + transpose: B [K, N] with arbitrary strides -> u8 LUT
// index image [N, K]. 32x32 tile staged through smem so global reads run
// along n and global writes along k (both coalesced for a contiguous B; for
// a transposed view B = W.t() the roles swap and it is the writes that
// follow the fast axis — either way one side is always coalesced and the
// traffic is 1 byte out per element). +1 smem padding kills bank conflicts.
template<typename T>
static __global__ void quantize_transpose_u8_kernel(
    const T* __restrict__ in,
    uint8_t* __restrict__ out,
    int K, int N,
    long long sk, long long sn,
    int offset)
{
    __shared__ uint8_t tile[32][33];
    const int k0 = blockIdx.y * 32;
    const int n0 = blockIdx.x * 32;

    for (int dy = threadIdx.y; dy < 32; dy += 8) {
        const int k = k0 + dy;
        const int n = n0 + threadIdx.x;
        if (k < K && n < N) {
            tile[dy][threadIdx.x] =
                to_idx(in[k * sk + n * sn], offset);
        }
    }
    __syncthreads();
    for (int dy = threadIdx.y; dy < 32; dy += 8) {
        const int n = n0 + dy;
        const int k = k0 + threadIdx.x;
        if (n < N && k < K) {
            out[static_cast<long long>(n) * K + k] = tile[threadIdx.x][dy];
        }
    }
}

static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> gemm_forward_save_cfg_impl(
    const torch::Tensor& A,
    const torch::Tensor& B,
    const torch::Tensor& lut,
    int64_t cfg,
    int offset)
{
    const at::cuda::OptionalCUDAGuard device_guard(device_of(A));

    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(lut.is_cuda(), "lut must be a CUDA tensor");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32
                || A.scalar_type() == torch::kByte,
                "A must be float32 or uint8 (LUT indices)");
    TORCH_CHECK(B.scalar_type() == torch::kFloat32
                || B.scalar_type() == torch::kByte,
                "B must be float32 or uint8 (LUT indices)");
    TORCH_CHECK(lut.scalar_type() == torch::kFloat32, "lut must be float32");
    TORCH_CHECK(A.dim() == 2, "A must have shape [M, K]");
    TORCH_CHECK(B.dim() == 2, "B must have shape [K, N]");
    TORCH_CHECK(lut.numel() == 256 * 256, "lut must have 65536 elements");
    TORCH_CHECK(A.size(1) == B.size(0),
                "A.size(1) must equal B.size(0), got ", A.size(1), " vs ", B.size(0));
    TORCH_CHECK(A.size(0) <= INT32_MAX && A.size(1) <= INT32_MAX
                && B.size(1) <= INT32_MAX, "dimensions too large");

    const int M = static_cast<int>(A.size(0));
    const int K = static_cast<int>(A.size(1));
    const int N = static_cast<int>(B.size(1));

    auto stream = at::cuda::getCurrentCUDAStream();
    auto f32opts = A.options().dtype(torch::kFloat32);
    auto u8opts = A.options().dtype(torch::kUInt8);

    // ---- operand images ----
    // A [M,K]: elementwise quantize (or alias a u8 input)
    torch::Tensor aq;
    if (A.scalar_type() == torch::kByte) {
        aq = A.contiguous();
    } else {
        auto ac = A.contiguous();
        aq = torch::empty({M, K}, u8opts);
        const long long n = static_cast<long long>(M) * K;
        if (n > 0) {
            const int threads = 256;
            const float* in = ac.data_ptr<float>();
            uint8_t* outp = aq.data_ptr<uint8_t>();
            const bool vec = ptr_aligned(in, 16) && ptr_aligned(outp, 4);
            const int blocks = static_cast<int>(
                std::min<long long>(n / (vec ? 4 : 1) / threads + 1, 4096));
            if (vec) quantize_to_u8_kernel<4><<<blocks, threads, 0, stream>>>(in, outp, n, offset);
            else     quantize_to_u8_kernel<1><<<blocks, threads, 0, stream>>>(in, outp, n, offset);
        }
    }
    // B [K,N] (any strides): fused quantize + transpose -> bq [N,K]
    auto bq = torch::empty({N, K}, u8opts);
    if (static_cast<long long>(N) * K > 0) {
        dim3 block(32, 8);
        dim3 grid(ceil_div(N, 32), ceil_div(K, 32));
        if (B.scalar_type() == torch::kByte) {
            quantize_transpose_u8_kernel<uint8_t><<<grid, block, 0, stream>>>(
                B.data_ptr<uint8_t>(), bq.data_ptr<uint8_t>(),
                K, N, B.stride(0), B.stride(1), offset);
        } else {
            quantize_transpose_u8_kernel<float><<<grid, block, 0, stream>>>(
                B.data_ptr<float>(), bq.data_ptr<uint8_t>(),
                K, N, B.stride(0), B.stride(1), offset);
        }
    }

    auto y = torch::empty({M, N}, f32opts);
    if (y.numel() == 0) {           // M == 0 or N == 0: nothing to launch
        CHECK_CUDA_ERROR();
        return std::make_tuple(y, aq, bq);
    }

    // ---- orientation ----
    // swapped=false: XMK  rows=M cols=N (warp along N)
    // swapped=true:  SFLAT rows=N cols=M (warp along M), transposed LUT
    int c = static_cast<int>(cfg);
    bool swapped;
    if (c >= 100) {
        swapped = true;
        c -= 100;
    } else if (c >= 0) {
        swapped = false;
    } else {
        swapped = (static_cast<long long>(M) >= 2LL * N);
    }
    // grid.y holds ceil(rows / BM) tiles; BM >= 8 for every cfg
    const bool xmk_fits  = (static_cast<long long>(M) + 7) / 8 <= 65535;
    const bool swap_fits = (static_cast<long long>(N) + 7) / 8 <= 65535;
    TORCH_CHECK(xmk_fits || swap_fits,
                "both M and N exceed the row-tile grid limit (", M, ", ", N, ")");
    if (!swapped && !xmk_fits) { swapped = true;  c = -1; }
    if (swapped && !swap_fits) { swapped = false; c = -1; }

    // ---- LUT images (16-bit dual-interpretation + validity flags) ----
    auto lut16 = torch::empty({256 * 256}, u8opts.dtype(torch::kInt16));
    auto lut16_bad = torch::zeros({2}, u8opts.dtype(torch::kInt32));
    auto lutc = lut.contiguous();
    torch::Tensor lutT;   // keep alive until the kernel runs

    LaunchArgs a{
        nullptr, nullptr, lutc.data_ptr<float>(),
        lut16.data_ptr<short>(), lut16_bad.data_ptr<int>(),
        y.data_ptr<float>(),
        K, 0, 0, /*L=*/1, /*O=*/N, f32opts, stream.stream()
    };

    if (swapped) {
        lutT = torch::empty_like(lutc);
        prepare_lut_kernel<true><<<256, 256, 0, stream>>>(
            lutc.data_ptr<float>(), lutT.data_ptr<float>(),
            lut16.data_ptr<short>(), lut16_bad.data_ptr<int>());
        a.lut = lutT.data_ptr<float>();
        a.rowsrc = bq.data_ptr<uint8_t>();
        a.colsrc = aq.data_ptr<uint8_t>();
        a.R = N; a.C = M;
    } else {
        prepare_lut_kernel<false><<<256, 256, 0, stream>>>(
            lutc.data_ptr<float>(), nullptr,
            lut16.data_ptr<short>(), lut16_bad.data_ptr<int>());
        a.rowsrc = aq.data_ptr<uint8_t>();
        a.colsrc = bq.data_ptr<uint8_t>();
        a.R = M; a.C = N;
    }

    if (c < 0) c = pick_cfg_sflat(a.R, a.C);
    if (swapped) dispatch_cfg<SFLAT>(c, a);
    else         dispatch_cfg<XMK>(c, a);

    CHECK_CUDA_ERROR();
    return std::make_tuple(y, aq, bq);
}

// ---- int8 entry points (offset 128: signed values in [-128, 127]) ----

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> gemm_lut_i8_save_cfg(
    const torch::Tensor& A, const torch::Tensor& B, const torch::Tensor& lut,
    int64_t cfg)
{
    return gemm_forward_save_cfg_impl(A, B, lut, cfg, 128);
}

torch::Tensor gemm_lut_i8_cfg(
    const torch::Tensor& A, const torch::Tensor& B, const torch::Tensor& lut,
    int64_t cfg)
{
    return std::get<0>(gemm_lut_i8_save_cfg(A, B, lut, cfg));
}

torch::Tensor gemm_lut_i8(
    const torch::Tensor& A, const torch::Tensor& B, const torch::Tensor& lut)
{
    return gemm_lut_i8_cfg(A, B, lut, -1);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> gemm_lut_i8_save(
    const torch::Tensor& A, const torch::Tensor& B, const torch::Tensor& lut)
{
    return gemm_lut_i8_save_cfg(A, B, lut, -1);
}

// ---- uint8 entry points (offset 0: raw unsigned values in [0, 255]) ----

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> gemm_lut_u8_save_cfg(
    const torch::Tensor& A, const torch::Tensor& B, const torch::Tensor& lut,
    int64_t cfg)
{
    return gemm_forward_save_cfg_impl(A, B, lut, cfg, 0);
}

torch::Tensor gemm_lut_u8_cfg(
    const torch::Tensor& A, const torch::Tensor& B, const torch::Tensor& lut,
    int64_t cfg)
{
    return std::get<0>(gemm_lut_u8_save_cfg(A, B, lut, cfg));
}

torch::Tensor gemm_lut_u8(
    const torch::Tensor& A, const torch::Tensor& B, const torch::Tensor& lut)
{
    return gemm_lut_u8_cfg(A, B, lut, -1);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> gemm_lut_u8_save(
    const torch::Tensor& A, const torch::Tensor& B, const torch::Tensor& lut)
{
    return gemm_lut_u8_save_cfg(A, B, lut, -1);
}

} // namespace claude_gemm

TORCH_LIBRARY_FRAGMENT(approxtorch, m){
    m.def("gemm_fake_int8_forward_cuda_claude(Tensor A, Tensor B, Tensor lut) -> Tensor");
    m.def("gemm_fake_int8_forward_cuda_claude_cfg(Tensor A, Tensor B, Tensor lut, int cfg) -> Tensor");
    m.def("gemm_fake_int8_forward_cuda_claude_save(Tensor A, Tensor B, Tensor lut) -> (Tensor, Tensor, Tensor)");
    m.def("gemm_fake_uint8_forward_cuda_claude(Tensor A, Tensor B, Tensor lut) -> Tensor");
    m.def("gemm_fake_uint8_forward_cuda_claude_cfg(Tensor A, Tensor B, Tensor lut, int cfg) -> Tensor");
    m.def("gemm_fake_uint8_forward_cuda_claude_save(Tensor A, Tensor B, Tensor lut) -> (Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(approxtorch, CUDA, m){
    m.impl("gemm_fake_int8_forward_cuda_claude", &claude_gemm::gemm_lut_i8);
    m.impl("gemm_fake_int8_forward_cuda_claude_cfg", &claude_gemm::gemm_lut_i8_cfg);
    m.impl("gemm_fake_int8_forward_cuda_claude_save", &claude_gemm::gemm_lut_i8_save);
    m.impl("gemm_fake_uint8_forward_cuda_claude", &claude_gemm::gemm_lut_u8);
    m.impl("gemm_fake_uint8_forward_cuda_claude_cfg", &claude_gemm::gemm_lut_u8_cfg);
    m.impl("gemm_fake_uint8_forward_cuda_claude_save", &claude_gemm::gemm_lut_u8_save);
}
