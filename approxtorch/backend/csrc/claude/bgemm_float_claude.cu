// Optimized LUT-based approximate-multiplication BGEMM (fake-int8, float storage).
//
// Functional reference: ../cuda/bgemm_float_gpt.cu
//   y[n, o, l] = sum_k lut[(round(x[n,k,l]) + 128) * 256 + (round(w[o,k]) + 128)]
//
// Interface is identical to the reference:
//   bgemm_fake_int8_forward_cuda_claude(Tensor x[N,K,L] f32, Tensor w[O,K] f32,
//                                       Tensor lut[65536] f32) -> Tensor y[N,O,L] f32
//
// Key optimizations vs the reference:
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

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#ifndef CHECK_CUDA_ERROR
#define CHECK_CUDA_ERROR()                                      \
    do {                                                        \
        cudaError_t err = cudaGetLastError();                   \
        if (err != cudaSuccess) {                               \
            printf("CUDA kernel failed: %s\n",                  \
                   cudaGetErrorString(err));                    \
        }                                                       \
    } while (0)
#endif

namespace claude_bgemm {

// split-K engages only below this many natural blocks (RTX 6000 Ada: 142 SMs)
constexpr int kSplitMinBlocks = 192;
constexpr int kSplitTargetBlocks = 284;

enum Mode { XMK = 0, NFLAT = 1, SFLAT = 2 };

static inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

// ---------------------------------------------------------------------------
// Prepass: quantize float (integers in [-128,127] stored as float) -> uint8
// index (value + 128, clamped to [0,255]), vectorized float4 -> uchar4.
// ---------------------------------------------------------------------------

__device__ __forceinline__ uint8_t quantize_one(float v)
{
    int q = __float2int_rn(v) + 128;
    q = max(0, min(255, q));
    return static_cast<uint8_t>(q);
}

__global__ void quantize_to_u8_kernel(
    const float* __restrict__ in,
    uint8_t* __restrict__ out,
    long long numel)
{
    const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
    const long long tid0   = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long long n4     = numel >> 2;

    const float4* in4 = reinterpret_cast<const float4*>(in);
    uchar4* out4      = reinterpret_cast<uchar4*>(out);

    for (long long i = tid0; i < n4; i += stride) {
        float4 v = in4[i];
        uchar4 q;
        q.x = quantize_one(v.x);
        q.y = quantize_one(v.y);
        q.z = quantize_one(v.z);
        q.w = quantize_one(v.w);
        out4[i] = q;
    }
    for (long long i = (n4 << 2) + tid0; i < numel; i += stride) {
        out[i] = quantize_one(in[i]);
    }
}

// LUT preprocessing (one kernel pass over 65536 entries, ~microseconds).
//
// int16 image: a float LUT is 256KB (twice the 128KB L1 of sm_89);
// approximate-multiplier tables are integer-valued and fit int16, in which
// case the whole table is 128KB, gathers pull half the sectors, and most of
// it stays L1-resident. `bad` (zero-initialized) is set if any entry is not
// exactly representable; the main kernel reads it as a grid-uniform flag, so
// no host synchronization is needed and results stay bit-identical (the
// int16 -> float conversion is exact). Non-integer LUTs fall back to the
// float image.
//
// TRANSPOSE=true additionally produces the transposed float/int16 images
// SFLAT mode needs, fused into the same launch.
template<bool TRANSPOSE>
__global__ void prepare_lut_kernel(
    const float* __restrict__ in,
    float* __restrict__ outf,    // transposed float image (TRANSPOSE only)
    short* __restrict__ out16,
    int* __restrict__ bad)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;   // grid covers 65536
    const float v = in[i];
    const float r = rintf(v);
    const bool good = (v == r) && (fabsf(v) <= 32767.0f);
    if (!good) *bad = 1;
    const short v16 = good ? static_cast<short>(r) : 0;
    if constexpr (TRANSPOSE) {
        const int o = ((i & 255) << 8) | (i >> 8);   // [x][w] -> [w][x]
        outf[o]  = v;
        out16[o] = v16;
    } else {
        out16[i] = v16;
    }
}

// ---------------------------------------------------------------------------
// Split-K reduction: y[i] = sum_s ws[s * numel + i], ascending s.
// ---------------------------------------------------------------------------

__global__ void reduce_splits_kernel(
    const float* __restrict__ ws,
    float* __restrict__ y,
    long long numel,
    int S)
{
    const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
    for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < numel; i += stride) {
        float a = 0.0f;
        for (int s = 0; s < S; ++s) {
            a += ws[static_cast<long long>(s) * numel + i];
        }
        y[i] = a;
    }
}

// ---------------------------------------------------------------------------
// Main kernel. Generic over a (rows R, cols C) view of the problem:
//
//   XMK:   R = N, C = O.    rowsrc = x viewed as [M=N, K], colsrc = w[O, K]
//   NFLAT: R = N*L, C = O.  rowsrc = x[N, K, L] (rows flatten (n, l)),
//                           colsrc = w[O, K]
//   SFLAT: R = O, C = N*L.  rowsrc = w[O, K], colsrc = x[N, K, L] (cols
//                           flatten (n, l)), lut transposed -> summands
//                           identical, warp spans N*L.
//
// Block tile: BM rows x BN cols, k-tile BK. Threads: (BN/TN) x (BM/TM) =
// 32 x 8; threadIdx.x spans cols so each warp shares one row value per
// (k, tm) -> one LUT row per gather instruction.
//
// Split-K: grid.z = S; split s handles k-tiles [s * kt_split, ...) and
// writes to workspace slice s. S == 1 -> out == y.
// ---------------------------------------------------------------------------

template<int BM, int BN, int BK, int TM, int TN, int MODE, typename LUT_T>
__device__ __forceinline__ void bgemm_mainloop(
    const uint8_t* __restrict__ rowsrc,
    const uint8_t* __restrict__ colsrc,
    const LUT_T*  __restrict__ lut,
    uint8_t (&srow)[BK][BM],
    uint8_t (&scol)[BK][BN],
    float (&acc)[TM][TN],
    int K, int R, int C, int L,
    int kbeg, int kend,
    int r0, int c0, int tx, int ty, int tid)
{
    constexpr int TX = BN / TN;
    constexpr int TY = BM / TM;
    constexpr int NT = TX * TY;

    for (int k0 = kbeg; k0 < kend; k0 += BK) {
        // ---- row tile [BK][BM] ----
        if constexpr (MODE == NFLAT) {
            // x[N, K, L], row index r = n * L + l
#pragma unroll
            for (int idx = tid; idx < BK * BM; idx += NT) {
                const int kk = idx / BM;
                const int m  = idx % BM;
                const int gk = min(k0 + kk, K - 1);   // clamped: padded values
                const int gr = min(r0 + m,  R - 1);   // never reach valid outputs
                const int n  = gr / L;
                const int l  = gr - n * L;
                srow[kk][m] = rowsrc[(static_cast<long long>(n) * K + gk) * L + l];
            }
        } else {
            // row-major [R, K] (x[M,K] for XMK, w[O,K] for SFLAT)
#pragma unroll
            for (int idx = tid; idx < BK * BM; idx += NT) {
                const int m  = idx / BK;
                const int kk = idx % BK;
                const int gk = min(k0 + kk, K - 1);
                const int gr = min(r0 + m,  R - 1);
                srow[kk][m] = rowsrc[static_cast<long long>(gr) * K + gk];
            }
        }
        // ---- col tile [BK][BN] ----
        if constexpr (MODE == SFLAT) {
            // x[N, K, L], col index c = n * L + l
#pragma unroll
            for (int idx = tid; idx < BK * BN; idx += NT) {
                const int kk = idx / BN;
                const int c  = idx % BN;
                const int gk = min(k0 + kk, K - 1);
                const int gc = min(c0 + c,  C - 1);
                const int n  = gc / L;
                const int l  = gc - n * L;
                scol[kk][c] = colsrc[(static_cast<long long>(n) * K + gk) * L + l];
            }
        } else {
            // w[O, K] row-major: BK consecutive bytes per o row
#pragma unroll
            for (int idx = tid; idx < BK * BN; idx += NT) {
                const int c  = idx / BK;
                const int kk = idx % BK;
                const int gk = min(k0 + kk, K - 1);
                const int gc = min(c0 + c,  C - 1);
                scol[kk][c] = colsrc[static_cast<long long>(gc) * K + gk];
            }
        }
        __syncthreads();

        const int klim = min(BK, kend - k0);
        if (klim == BK) {
#pragma unroll
            for (int kk = 0; kk < BK; ++kk) {
                uint8_t xb[TM];
                uint8_t wb[TN];
                if constexpr (TM % 4 == 0) {
#pragma unroll
                    for (int t = 0; t < TM / 4; ++t)
                        *reinterpret_cast<uchar4*>(xb + 4 * t) =
                            *reinterpret_cast<const uchar4*>(&srow[kk][ty * TM + 4 * t]);
                } else {
#pragma unroll
                    for (int t = 0; t < TM; ++t)
                        xb[t] = srow[kk][ty * TM + t];
                }
                if constexpr (TN % 4 == 0) {
#pragma unroll
                    for (int t = 0; t < TN / 4; ++t)
                        *reinterpret_cast<uchar4*>(wb + 4 * t) =
                            *reinterpret_cast<const uchar4*>(&scol[kk][tx * TN + 4 * t]);
                } else {
#pragma unroll
                    for (int t = 0; t < TN; ++t)
                        wb[t] = scol[kk][tx * TN + t];
                }
#pragma unroll
                for (int tm = 0; tm < TM; ++tm) {
                    const unsigned row = static_cast<unsigned>(xb[tm]) << 8;
#pragma unroll
                    for (int tn = 0; tn < TN; ++tn) {
                        acc[tm][tn] += static_cast<float>(__ldg(lut + (row | wb[tn])));
                    }
                }
            }
        } else {
            // K tail: same body, runtime bound (taken at most once per block)
            for (int kk = 0; kk < klim; ++kk) {
                uint8_t xb[TM];
                uint8_t wb[TN];
#pragma unroll
                for (int t = 0; t < TM; ++t) xb[t] = srow[kk][ty * TM + t];
#pragma unroll
                for (int t = 0; t < TN; ++t) wb[t] = scol[kk][tx * TN + t];
#pragma unroll
                for (int tm = 0; tm < TM; ++tm) {
                    const unsigned row = static_cast<unsigned>(xb[tm]) << 8;
#pragma unroll
                    for (int tn = 0; tn < TN; ++tn) {
                        acc[tm][tn] += static_cast<float>(__ldg(lut + (row | wb[tn])));
                    }
                }
            }
        }
        __syncthreads();
    }
}

template<int BM, int BN, int BK, int TM, int TN, int MODE>
__global__ void __launch_bounds__(256)
bgemm_lut_u8_kernel(
    const uint8_t* __restrict__ rowsrc,
    const uint8_t* __restrict__ colsrc,
    const float* __restrict__ lut,
    const short* __restrict__ lut16,
    const int*   __restrict__ lut16_bad,
    float* __restrict__ out,
    int K, int R, int C, int L, int O,
    int kt_split)
{
    constexpr int TX = BN / TN;

    __shared__ uint8_t srow[BK][BM];
    __shared__ uint8_t scol[BK][BN];

    const int s  = blockIdx.z;
    const int r0 = blockIdx.y * BM;
    const int c0 = blockIdx.x * BN;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * TX + tx;

    const int kbeg = s * kt_split * BK;
    const int kend = min(K, kbeg + kt_split * BK);

    out += static_cast<long long>(s) * R * C;

    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
            acc[i][j] = 0.0f;

    // grid-uniform branch (flag identical for every block -> sync-safe)
    if (__ldg(lut16_bad) == 0) {
        bgemm_mainloop<BM, BN, BK, TM, TN, MODE>(
            rowsrc, colsrc, lut16, srow, scol, acc,
            K, R, C, L, kbeg, kend, r0, c0, tx, ty, tid);
    } else {
        bgemm_mainloop<BM, BN, BK, TM, TN, MODE>(
            rowsrc, colsrc, lut, srow, scol, acc,
            K, R, C, L, kbeg, kend, r0, c0, tx, ty, tid);
    }

    // ---- epilogue: y[n, o, l] ----
#pragma unroll
    for (int tm = 0; tm < TM; ++tm) {
        const int r = r0 + ty * TM + tm;
        if (r < R) {
#pragma unroll
            for (int tn = 0; tn < TN; ++tn) {
                const int c = c0 + tx * TN + tn;
                if (c < C) {
                    long long off;
                    if constexpr (MODE == XMK) {
                        off = static_cast<long long>(r) * C + c;          // y[m, o]
                    } else if constexpr (MODE == NFLAT) {
                        const int n = r / L;
                        const int l = r - n * L;
                        off = (static_cast<long long>(n) * O + c) * L + l;
                    } else {
                        const int n = c / L;
                        const int l = c - n * L;
                        off = (static_cast<long long>(n) * O + r) * L + l;
                    }
                    out[off] = acc[tm][tn];
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------

struct LaunchArgs {
    const uint8_t* rowsrc;
    const uint8_t* colsrc;
    const float* lut;
    const short* lut16;
    const int* lut16_bad;
    float* y;
    int K, R, C, L, O;
    torch::TensorOptions f32opts;
    cudaStream_t stream;
};

template<int BM, int BN, int BK, int TM, int TN, int MODE>
void launch_cfg(const LaunchArgs& a)
{
    dim3 block(BN / TN, BM / TM);
    const int gx = ceil_div(a.C, BN);
    const int gy = ceil_div(a.R, BM);
    TORCH_CHECK(gy <= 65535, "row tile count exceeds grid.y limit");

    const int ktiles = ceil_div(a.K, BK);
    const long long natural_blocks = static_cast<long long>(gx) * gy;

    int S = 1;
    if (natural_blocks < kSplitMinBlocks && ktiles > 1) {
        S = static_cast<int>(
            std::min<long long>(
                (kSplitTargetBlocks + natural_blocks - 1) / natural_blocks,
                ktiles));
        const int chunk = ceil_div(ktiles, S);
        S = ceil_div(ktiles, chunk);
    }
    const int kt_split = ceil_div(ktiles, S);
    dim3 grid(gx, gy, S);

    if (S == 1) {
        bgemm_lut_u8_kernel<BM, BN, BK, TM, TN, MODE><<<grid, block, 0, a.stream>>>(
            a.rowsrc, a.colsrc, a.lut, a.lut16, a.lut16_bad,
            a.y, a.K, a.R, a.C, a.L, a.O, kt_split);
    } else {
        const long long numel = static_cast<long long>(a.R) * a.C;
        auto ws = torch::empty({S * numel}, a.f32opts);
        bgemm_lut_u8_kernel<BM, BN, BK, TM, TN, MODE><<<grid, block, 0, a.stream>>>(
            a.rowsrc, a.colsrc, a.lut, a.lut16, a.lut16_bad,
            ws.data_ptr<float>(), a.K, a.R, a.C, a.L, a.O, kt_split);
        const int threads = 256;
        const int blocks = static_cast<int>(
            std::min<long long>((numel + threads - 1) / threads, 4096));
        reduce_splits_kernel<<<blocks, threads, 0, a.stream>>>(
            ws.data_ptr<float>(), a.y, numel, S);
    }
}

// cfg ids, kept stable so the bench harness can sweep them.
// (BM, BN, BK, TM, TN)
template<int MODE>
static void dispatch_cfg(int cfg, const LaunchArgs& a)
{
    switch (cfg) {
        case 0:  launch_cfg<32, 128, 32, 4, 4, MODE>(a); break;
        case 1:  launch_cfg<64, 128, 32, 8, 4, MODE>(a); break;
        case 2:  launch_cfg<16, 128, 32, 2, 4, MODE>(a); break;
        case 3:  launch_cfg< 8, 128, 32, 1, 4, MODE>(a); break;
        case 7:  launch_cfg<32, 256, 32, 4, 8, MODE>(a); break;
        case 9:  launch_cfg<16,  32, 32, 2, 1, MODE>(a); break;
        case 10: launch_cfg< 8,  32, 32, 1, 1, MODE>(a); break;
        case 11: launch_cfg<64, 256, 32, 8, 8, MODE>(a); break;
        case 13: launch_cfg<16, 256, 32, 2, 8, MODE>(a); break;
        case 14: launch_cfg< 8, 256, 32, 1, 8, MODE>(a); break;
        case 15: launch_cfg<16, 512, 32, 2, 16, MODE>(a); break;
        case 16: launch_cfg< 8, 512, 32, 1, 16, MODE>(a); break;
        case 17: launch_cfg<32, 128, 64, 4, 4, MODE>(a); break;
        case 18: launch_cfg<64, 128, 64, 8, 4, MODE>(a); break;
        default: TORCH_CHECK(false, "unknown cfg id: ", cfg);
    }
}

// Heuristics tuned by sweeping cfgs on RTX 6000 Ada (sm_89); see NOTES.md.
// R = block-row dimension, C = block-column (warp) dimension.
// Small BM wins when many blocks share an SM: the per-SM working set of hot
// LUT rows (~BM KB per resident block) has to stay within the 128KB L1.

// NFLAT / XMK: R = N*L (or N), C = O
static int pick_cfg_nflat(long long R, long long C)
{
    if (C >= 192) {
        if (R >= 512) return 11;
        if (R >= 24)  return 7;
        if (R >= 12)  return 13;
        return 14;
    }
    if (C >= 48) {
        if (R >= 24) return 0;
        if (R >= 12) return 2;
        return 3;
    }
    if (R >= 12) return 9;
    return 10;
}

// SFLAT / XMK: R = O (or N), C = N*L (or O); C is almost always large here.
// Tuned for the int16-LUT path (the common case): the table is mostly
// L1-resident, so medium tiles with dense per-thread gather ILP win over
// the wide-BN row-amortization configs that the float-LUT path prefers.
static int pick_cfg_sflat(long long R, long long C)
{
    if (C >= 96) {
        if (R >= 24) return 0;
        if (R >= 12) return 2;
        return 3;
    }
    if (R >= 12) return 9;
    return 10;
}

torch::Tensor bgemm_lut_forward_cuda_claude_cfg(
    const torch::Tensor& x,
    const torch::Tensor& w,
    const torch::Tensor& lut,
    int64_t cfg)
{
    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));

    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(w.is_cuda(), "w must be a CUDA tensor");
    TORCH_CHECK(lut.is_cuda(), "lut must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
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
    auto u8opts = xc.options().dtype(torch::kUInt8);

    // prepass: quantize x and w to uint8 LUT indices
    auto xq = torch::empty({N, K, L}, u8opts);
    auto wq = torch::empty({O, K}, u8opts);
    {
        const long long xn = static_cast<long long>(N) * K * L;
        const long long wn = static_cast<long long>(O) * K;
        const int threads = 256;
        const int xblocks = static_cast<int>(std::min<long long>(xn / 4 / threads + 1, 4096));
        const int wblocks = static_cast<int>(std::min<long long>(wn / 4 / threads + 1, 4096));
        quantize_to_u8_kernel<<<xblocks, threads, 0, stream>>>(
            xc.data_ptr<float>(), xq.data_ptr<uint8_t>(), xn);
        quantize_to_u8_kernel<<<wblocks, threads, 0, stream>>>(
            wc.data_ptr<float>(), wq.data_ptr<uint8_t>(), wn);
    }

    auto y = torch::empty({N, O, L}, xc.options());

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
        K, 0, 0, L, O, xc.options(), stream.stream()
    };

    // LUT images: int16 (+ transposed copies for SFLAT) and validity flag
    auto lut16 = torch::empty({256 * 256}, u8opts.dtype(torch::kInt16));
    auto lut16_bad = torch::zeros({1}, u8opts.dtype(torch::kInt32));
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
    return y;
}

torch::Tensor bgemm_lut_forward_cuda_claude(
    const torch::Tensor& x,
    const torch::Tensor& w,
    const torch::Tensor& lut)
{
    return bgemm_lut_forward_cuda_claude_cfg(x, w, lut, -1);
}

} // namespace claude_bgemm

TORCH_LIBRARY_FRAGMENT(approxtorch, m){
    m.def("bgemm_fake_int8_forward_cuda_claude(Tensor x, Tensor w, Tensor lut) -> Tensor");
    m.def("bgemm_fake_int8_forward_cuda_claude_cfg(Tensor x, Tensor w, Tensor lut, int cfg) -> Tensor");
}

TORCH_LIBRARY_IMPL(approxtorch, CUDA, m){
    m.impl("bgemm_fake_int8_forward_cuda_claude", &claude_bgemm::bgemm_lut_forward_cuda_claude);
    m.impl("bgemm_fake_int8_forward_cuda_claude_cfg", &claude_bgemm::bgemm_lut_forward_cuda_claude_cfg);
}
