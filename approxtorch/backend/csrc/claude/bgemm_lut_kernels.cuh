// Shared machinery for the LUT-based approximate-multiplication GEMM family.
// Used by bgemm_float_claude.cu (batched: x[N,K,L] * w[O,K]) and gemm.cu
// (plain mm: A[M,K] * B[K,N]). Everything here is quantization-agnostic:
// operands are uint8 LUT indices, the semantics (int8 +128 / uint8 raw) live
// in the prepass offsets and the LUT layout chosen by the caller.
//
// Contents: u8 quantize prepass, 16-bit LUT imaging (int16 / uint16 dual
// interpretation + fp32 fallback), the tiled main kernel with its three
// row/col addressing modes, split-K launch + reduction, the tuned tile-config
// dispatch table and shape heuristics. Design rationale and measured numbers
// live in NOTES.md; the main kernel's contract is documented at
// bgemm_lut_u8_kernel below.
//
// Everything is header-safe for multiple TUs: templates and static functions
// only (identical template instantiations dedup at link; statics stay
// TU-local).

#pragma once

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

// float4/uchar4 global accesses require 16/4-byte aligned addresses, but
// contiguous() can return a view with nonzero storage_offset whose data_ptr
// is only element-aligned. Vectorize only when every pointer qualifies
// (fresh allocations from the caching allocator always do).
static inline bool ptr_aligned(const void* p, uintptr_t bytes) {
    return (reinterpret_cast<uintptr_t>(p) & (bytes - 1)) == 0;
}

// ---------------------------------------------------------------------------
// Prepass: quantize float (integer values stored as float) -> uint8 LUT index
// (value + offset, clamped to [0,255]), vectorized float4 -> uchar4.
// offset 128: signed operand in [-128,127] (int8 op). offset 0: unsigned
// operand in [0,255] (uint8 op, both x and w).
// ---------------------------------------------------------------------------

__device__ __forceinline__ uint8_t quantize_one(float v, int offset)
{
    int q = __float2int_rn(v) + offset;
    q = max(0, min(255, q));
    return static_cast<uint8_t>(q);
}

template<int VEC>
__global__ void quantize_to_u8_kernel(
    const float* __restrict__ in,
    uint8_t* __restrict__ out,
    long long numel,
    int offset)
{
    const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
    const long long tid0   = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long long n4     = (VEC == 4) ? (numel >> 2) : 0;

    if constexpr (VEC == 4) {
        const float4* in4 = reinterpret_cast<const float4*>(in);
        uchar4* out4      = reinterpret_cast<uchar4*>(out);

        for (long long i = tid0; i < n4; i += stride) {
            float4 v = in4[i];
            uchar4 q;
            q.x = quantize_one(v.x, offset);
            q.y = quantize_one(v.y, offset);
            q.z = quantize_one(v.z, offset);
            q.w = quantize_one(v.w, offset);
            out4[i] = q;
        }
    }
    for (long long i = (n4 << 2) + tid0; i < numel; i += stride) {
        out[i] = quantize_one(in[i], offset);
    }
}

// LUT preprocessing (one kernel pass over 65536 entries, ~microseconds).
//
// 16-bit image: a float LUT is 256KB (twice the 128KB L1 of sm_89);
// approximate-multiplier tables are integer-valued and fit 16 bits, in which
// case the whole table is 128KB, gathers pull half the sectors, and most of
// it stays L1-resident. Two interpretations share one image: signed int16
// (8x8 signed tables, values in [-32767, 32767]) and uint16 (uint8 x uint8
// tables, values in [0, 65535] — up to 255*255 = 65025, beyond int16). The
// stored low 16 bits of the integer value are the correct bit pattern under
// EITHER interpretation; `bad` is int[2] (zero-initialized), bad[0] set if
// some entry is not exact int16, bad[1] if not exact uint16. The main kernel
// reads both as grid-uniform flags (no host sync) and picks int16 -> uint16
// -> float32; the 16-bit -> float conversion is exact either way, so results
// stay bit-identical. Mixed-sign tables exceeding int16 fall back to float.
//
// TRANSPOSE=true additionally produces the transposed float/16-bit images
// SFLAT mode needs, fused into the same launch.
template<bool TRANSPOSE>
__global__ void prepare_lut_kernel(
    const float* __restrict__ in,
    float* __restrict__ outf,    // transposed float image (TRANSPOSE only)
    short* __restrict__ out16,
    int* __restrict__ bad)       // int[2]: [0] int16 invalid, [1] uint16 invalid
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;   // grid covers 65536
    const float v = in[i];
    const float r = rintf(v);
    const bool is_int = (v == r);
    const bool i16_ok = is_int && (fabsf(v) <= 32767.0f);
    const bool u16_ok = is_int && (v >= 0.0f) && (v <= 65535.0f);
    if (!i16_ok) bad[0] = 1;
    if (!u16_ok) bad[1] = 1;
    // low 16 bits of the integer value: valid pattern for both interpretations
    const short v16 = (i16_ok || u16_ok)
        ? static_cast<short>(static_cast<unsigned short>(static_cast<int>(r)))
        : 0;
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

static __global__ void reduce_splits_kernel(
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

    // grid-uniform branch (flags identical for every block -> sync-safe):
    // int16 image -> uint16 image (same bits, unsigned read) -> float LUT
    if (__ldg(lut16_bad + 0) == 0) {
        bgemm_mainloop<BM, BN, BK, TM, TN, MODE>(
            rowsrc, colsrc, lut16, srow, scol, acc,
            K, R, C, L, kbeg, kend, r0, c0, tx, ty, tid);
    } else if (__ldg(lut16_bad + 1) == 0) {
        bgemm_mainloop<BM, BN, BK, TM, TN, MODE>(
            rowsrc, colsrc, reinterpret_cast<const unsigned short*>(lut16),
            srow, scol, acc,
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
// Host-side launch helpers
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

} // namespace claude_bgemm
