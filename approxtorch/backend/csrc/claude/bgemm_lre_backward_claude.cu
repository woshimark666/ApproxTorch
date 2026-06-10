// claude-optimized LUT-based linear-regression-estimate (LRE) backward for
// the approximate-multiplication BGEMM.
//
// Reference: ../cuda/bgemm_lre_backward.cu
//
//   grad_x[n,k,l] = sum_o grad_output[n,o,l] * DX[q(w[k,o])]
//   grad_w[k,o]   = sum_{n,l} grad_output[n,o,l] * DW[q(x[n,k,l])]
//
// Key observation: unlike the forward (a true 2-D LUT gather per (x,w) pair),
// the backward FACTORIZES. DX[q(w[k,o])] does not depend on x at all, so it
// can be materialized once as a dense matrix W' = DX∘q(w) ([K,O], K*O cheap
// elementwise lookups). After that
//
//   grad_x[n] = W' @ grad_output[n]            -> one strided-batched SGEMM
//   grad_w    = X'p @ go_nlo over P = N*L      -> one SGEMM
//                (X' = DW∘q(x), one elementwise lookup pass over x)
//
// so the heavy lifting goes to cuBLAS instead of a hand-rolled
// one-output-per-thread GEMM, and the reference's full
// grad_output.permute(0,2,1).contiguous() copy is not needed for grad_x at
// all (cuBLAS consumes the natural [N,O,L] layout via op/stride flags).
//
// Two ops are registered:
//
// 1. bgemm_lre_backward_claude(go, x[N,K,L], w[K,O], dx, dw): x is the
//    already-unfolded (im2col) tensor. _cfg variant adds a tuning hook
//    (cfg -1 auto, 1 = single-gemm grad_w, 2 = batched+sum grad_w; measured:
//    single wins for every N>1 shape, batched only for N==1 where it is one
//    GEMM straight into grad_w).
//
// 2. bgemm_lre_backward_claude_im2col(go, x[N,C,H,W], w[K,O], dx, dw,
//    kh, kw, sh, sw, ph, pw, dilh, dilw): implicit-im2col variant. x is the
//    PRE-unfold quantized image; the X' build kernel computes the im2col
//    gather indices on the fly (out-of-bounds == unfold zero padding ==
//    LUT index 128). The unfolded tensor is never materialized: the image is
//    ~9x smaller than the unfolded form, lives in L2, and the patch overlap
//    is read essentially for free — so this is both the memory saver (the
//    autograd Function keeps only the small image) and slightly faster than
//    reading the unfolded X' source from HBM. grad_x is still returned in
//    im2col space [N,K,L] (autograd folds it back through unfold's backward).
//
// Numerics: bit-identical LUT indexing (__float2int_rn + clamp) and fp32
// accumulation, but cuBLAS reduction order differs from the reference's
// ascending-k loop, so results match to fp32 round-off (allclose), not
// bit-exactly. TF32 is only used if torch.backends.cuda.matmul.allow_tf32
// is enabled (default off => full fp32).
//
// x and w are accepted as float32 OR int8 (quantized values, idx = v+128)
// OR uint8 (the forward op's LUT-index images, idx = v). Fake-quantized
// values are exact integers, so the autograd Function saves those instead
// of fp32 (4x less activation memory) and gradients are identical
// bit-for-bit across input dtypes.

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>

namespace {

constexpr int QMIN = -128;
constexpr int QMAX = 127;
constexpr int MAP_THREADS = 256;

__device__ __forceinline__ int qfloat_to_idx(float v) {
    int q = __float2int_rn(v);
    return max(QMIN, min(QMAX, q)) + 128;
}

// int8 inputs hold already-quantized values: index is just v+128, no
// rounding/clamping needed (the dtype guarantees [-128, 127]). uint8 inputs
// are the forward op's quantized LUT-index images: index is the value itself.
__device__ __forceinline__ int lut_idx_of(float v)   { return qfloat_to_idx(v); }
__device__ __forceinline__ int lut_idx_of(int8_t v)  { return (int)v + 128; }
__device__ __forceinline__ int lut_idx_of(uint8_t v) { return (int)v; }

template <typename T> struct Vec4;
template <> struct Vec4<float>   { using type = float4; };
template <> struct Vec4<int8_t>  { using type = char4;  };
template <> struct Vec4<uint8_t> { using type = uchar4; };

// Vec4 global accesses require 4*sizeof(T)-byte aligned addresses, but
// contiguous() can return a view with nonzero storage_offset whose data_ptr
// is only element-aligned. Vectorize only when every pointer qualifies
// (fresh allocations from the caching allocator always do).
inline bool ptr_aligned(const void* p, uintptr_t bytes) {
    return (reinterpret_cast<uintptr_t>(p) & (bytes - 1)) == 0;
}

// out[i] = lut[q(in[i])], flat layout. 4-wide main loop + scalar tail
// (VEC == 1: scalar loop only, for misaligned pointers).
template <typename T, int VEC>
__global__ void lut_map_flat_kernel(
    const T* __restrict__ in,
    const float* __restrict__ lut,
    float* __restrict__ out,
    long long numel
) {
    __shared__ float slut[256];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) slut[i] = lut[i];
    __syncthreads();

    const long long stride = (long long)gridDim.x * blockDim.x;
    const long long base = (long long)blockIdx.x * blockDim.x + threadIdx.x;

    const long long n4 = (VEC == 4) ? (numel >> 2) : 0;
    if constexpr (VEC == 4) {
        using V4 = typename Vec4<T>::type;
        const V4* in4 = reinterpret_cast<const V4*>(in);
        float4* out4 = reinterpret_cast<float4*>(out);
        for (long long i = base; i < n4; i += stride) {
            V4 v = in4[i];
            float4 r;
            r.x = slut[lut_idx_of(v.x)];
            r.y = slut[lut_idx_of(v.y)];
            r.z = slut[lut_idx_of(v.z)];
            r.w = slut[lut_idx_of(v.w)];
            out4[i] = r;
        }
    }
    for (long long i = (n4 << 2) + base; i < numel; i += stride) {
        out[i] = slut[lut_idx_of(in[i])];
    }
}

// xp[k, n*L + l] = lut[q(x[n, k, l])]  ([N,K,L] -> [K, N*L])
// Writes are fully coalesced (consecutive threads span p = n*L+l); reads are
// coalesced within each L-segment.
template <typename T>
__global__ void lut_map_kp_kernel(
    const T* __restrict__ x,
    const float* __restrict__ lut,
    float* __restrict__ xp,
    int N, int K, int L
) {
    __shared__ float slut[256];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) slut[i] = lut[i];
    __syncthreads();

    const int P = N * L;
    const int pstride = gridDim.x * blockDim.x;
    for (int k = blockIdx.z; k < K; k += gridDim.z) {
        const T* xk = x + (long long)k * L;              // + n*K*L below
        float* xpk = xp + (long long)k * P;
        for (int p = blockIdx.x * blockDim.x + threadIdx.x; p < P; p += pstride) {
            int n = p / L;
            int l = p - n * L;
            T v = xk[(long long)n * K * L + l];
            xpk[p] = slut[lut_idx_of(v)];
        }
    }
}

// Implicit-im2col X' build: xp[k, n*L + l] = lut[q(unfold(x)[n, k, l])]
// computed straight from the pre-unfold image x [N, C, H, W] — the unfolded
// tensor is never materialized. k = (c, kh, kw), l = (oh, ow); out-of-bounds
// positions reproduce unfold's zero padding (quantized zero == index 128).
template <typename T>
__global__ void lut_map_kp_im2col_kernel(
    const T* __restrict__ x,        // [N, C, H, W]
    const float* __restrict__ lut,
    float* __restrict__ xp,         // [K = C*KH*KW, P = N*OH*OW]
    int N, int C, int H, int W,
    int KH, int KW, int SH, int SW,
    int PH, int PW, int DH, int DW,
    int OH, int OW
) {
    __shared__ float slut[256];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) slut[i] = lut[i];
    __syncthreads();

    const int L = OH * OW;
    const long long P = (long long)N * L;
    const int K = C * KH * KW;
    const int lstride = gridDim.x * blockDim.x;
    // grid: x spans l (one divide per element), y spans n, z spans k —
    // GPU integer division is emulated (~20 instructions), so hoisting
    // n = p/L out of the per-element path matters on this write-bound kernel
    for (int k = blockIdx.z; k < K; k += gridDim.z) {
        const int c   = k / (KH * KW);
        const int r   = k - c * (KH * KW);
        const int kh  = r / KW;
        const int kw  = r - kh * KW;
        const int ih0 = kh * DH - PH;
        const int iw0 = kw * DW - PW;
        for (int n = blockIdx.y; n < N; n += gridDim.y) {
            const T* xnc = x + ((long long)n * C + c) * H * W;
            float* xpkn = xp + (long long)k * P + (long long)n * L;
            for (int l = blockIdx.x * blockDim.x + threadIdx.x; l < L; l += lstride) {
                const int oh = l / OW;
                const int ow = l - oh * OW;
                const int ih = oh * SH + ih0;
                const int iw = ow * SW + iw0;
                int idx = 128;  // zero padding -> quantized zero
                if ((unsigned)ih < (unsigned)H && (unsigned)iw < (unsigned)W) {
                    idx = lut_idx_of(xnc[(long long)ih * W + iw]);
                }
                xpkn[l] = slut[idx];
            }
        }
    }
}

// Batched 2-D transpose: out[n, l, o] = in[n, o, l]  (32x32 smem tiles)
constexpr int TR_TILE = 32;
constexpr int TR_ROWS = 8;

__global__ void transpose_ol_batched_kernel(
    const float* __restrict__ in,   // [N, O, L]
    float* __restrict__ out,        // [N, L, O]
    int N, int O, int L
) {
    __shared__ float tile[TR_TILE][TR_TILE + 1];

    const int l0 = blockIdx.x * TR_TILE;
    const int o0 = blockIdx.y * TR_TILE;

    for (int n = blockIdx.z; n < N; n += gridDim.z) {
        const float* gin = in + (long long)n * O * L;
        float* gout = out + (long long)n * O * L;

        #pragma unroll
        for (int dy = 0; dy < TR_TILE; dy += TR_ROWS) {
            int o = o0 + threadIdx.y + dy;
            int l = l0 + threadIdx.x;
            if (o < O && l < L) {
                tile[threadIdx.y + dy][threadIdx.x] = gin[(long long)o * L + l];
            }
        }
        __syncthreads();
        #pragma unroll
        for (int dy = 0; dy < TR_TILE; dy += TR_ROWS) {
            int l = l0 + threadIdx.y + dy;
            int o = o0 + threadIdx.x;
            if (l < L && o < O) {
                gout[(long long)l * O + o] = tile[threadIdx.x][threadIdx.y + dy];
            }
        }
        __syncthreads();
    }
}

inline int map_grid(long long work, int threads) {
    long long b = (work + threads - 1) / threads;
    return (int)std::min<long long>(b, 4096);
}

bool qdtype_ok(torch::ScalarType t) {
    return t == torch::kFloat32 || t == torch::kChar || t == torch::kByte;
}

// ---- dtype-dispatched kernel launchers ----

template <typename T>
void launch_lut_map_flat_typed(cudaStream_t stream, const T* in,
                               const float* lut, float* out, long long n) {
    const bool vec = ptr_aligned(in, 4 * sizeof(T)) && ptr_aligned(out, 16);
    const int grid = map_grid(n, MAP_THREADS * (vec ? 4 : 1));
    if (vec) {
        lut_map_flat_kernel<T, 4><<<grid, MAP_THREADS, 0, stream>>>(in, lut, out, n);
    } else {
        lut_map_flat_kernel<T, 1><<<grid, MAP_THREADS, 0, stream>>>(in, lut, out, n);
    }
}

void launch_lut_map_flat(cudaStream_t stream, const torch::Tensor& in,
                         const torch::Tensor& lut, torch::Tensor& out) {
    const long long n = in.numel();
    if (in.scalar_type() == torch::kChar) {
        launch_lut_map_flat_typed(stream,
            in.data_ptr<int8_t>(), lut.data_ptr<float>(), out.data_ptr<float>(), n);
    } else if (in.scalar_type() == torch::kByte) {
        launch_lut_map_flat_typed(stream,
            in.data_ptr<uint8_t>(), lut.data_ptr<float>(), out.data_ptr<float>(), n);
    } else {
        launch_lut_map_flat_typed(stream,
            in.data_ptr<float>(), lut.data_ptr<float>(), out.data_ptr<float>(), n);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_lut_map_kp(cudaStream_t stream, const torch::Tensor& in,
                       const torch::Tensor& lut, torch::Tensor& out,
                       int64_t N, int64_t K, int64_t L) {
    dim3 grid(map_grid(N * L, MAP_THREADS), 1,
              (unsigned)std::min<int64_t>(K, 65535));
    if (in.scalar_type() == torch::kChar) {
        lut_map_kp_kernel<int8_t><<<grid, MAP_THREADS, 0, stream>>>(
            in.data_ptr<int8_t>(), lut.data_ptr<float>(), out.data_ptr<float>(),
            (int)N, (int)K, (int)L);
    } else if (in.scalar_type() == torch::kByte) {
        lut_map_kp_kernel<uint8_t><<<grid, MAP_THREADS, 0, stream>>>(
            in.data_ptr<uint8_t>(), lut.data_ptr<float>(), out.data_ptr<float>(),
            (int)N, (int)K, (int)L);
    } else {
        lut_map_kp_kernel<float><<<grid, MAP_THREADS, 0, stream>>>(
            in.data_ptr<float>(), lut.data_ptr<float>(), out.data_ptr<float>(),
            (int)N, (int)K, (int)L);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

struct ConvGeom {
    int C, H, W, KH, KW, SH, SW, PH, PW, DH, DW, OH, OW;
};

void launch_lut_map_kp_im2col(cudaStream_t stream, const torch::Tensor& in,
                              const torch::Tensor& lut, torch::Tensor& out,
                              int64_t N, const ConvGeom& g) {
    const long long L = (long long)g.OH * g.OW;
    const long long K = (long long)g.C * g.KH * g.KW;
    dim3 grid(map_grid(L, MAP_THREADS),
              (unsigned)std::min<int64_t>(N, 65535),
              (unsigned)std::min<long long>(K, 65535));
    if (in.scalar_type() == torch::kChar) {
        lut_map_kp_im2col_kernel<int8_t><<<grid, MAP_THREADS, 0, stream>>>(
            in.data_ptr<int8_t>(), lut.data_ptr<float>(), out.data_ptr<float>(),
            (int)N, g.C, g.H, g.W, g.KH, g.KW, g.SH, g.SW, g.PH, g.PW,
            g.DH, g.DW, g.OH, g.OW);
    } else if (in.scalar_type() == torch::kByte) {
        lut_map_kp_im2col_kernel<uint8_t><<<grid, MAP_THREADS, 0, stream>>>(
            in.data_ptr<uint8_t>(), lut.data_ptr<float>(), out.data_ptr<float>(),
            (int)N, g.C, g.H, g.W, g.KH, g.KW, g.SH, g.SW, g.PH, g.PW,
            g.DH, g.DW, g.OH, g.OW);
    } else {
        lut_map_kp_im2col_kernel<float><<<grid, MAP_THREADS, 0, stream>>>(
            in.data_ptr<float>(), lut.data_ptr<float>(), out.data_ptr<float>(),
            (int)N, g.C, g.H, g.W, g.KH, g.KW, g.SH, g.SW, g.PH, g.PW,
            g.DH, g.DW, g.OH, g.OW);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---- cuBLAS pieces ----

// grad_x[n] = W' @ go[n]   (go in natural [N,O,L] layout, strideB=0
// broadcasts W'; L==1 collapses to one GEMM instead of N gemv batches)
void compute_grad_x(cublasHandle_t handle, const torch::Tensor& go,
                    const torch::Tensor& wp, torch::Tensor& grad_x,
                    int64_t N, int64_t K, int64_t L, int64_t O) {
    const float one = 1.0f, zero = 0.0f;
    if (L == 1) {
        // grad_x [N,K] = go [N,O] @ W'^T
        TORCH_CUDABLAS_CHECK(cublasSgemm(
            handle, CUBLAS_OP_T, CUBLAS_OP_N,
            (int)K, (int)N, (int)O,
            &one,
            wp.data_ptr<float>(), (int)O,
            go.data_ptr<float>(), (int)O,
            &zero,
            grad_x.data_ptr<float>(), (int)K));
    } else {
        // row-major C[K,L] = A[K,O] @ B[O,L]  ==  column-major C' = B' @ A'
        TORCH_CUDABLAS_CHECK(cublasSgemmStridedBatched(
            handle, CUBLAS_OP_N, CUBLAS_OP_N,
            (int)L, (int)K, (int)O,
            &one,
            go.data_ptr<float>(), (int)L, (long long)(O * L),
            wp.data_ptr<float>(), (int)O, 0LL,
            &zero,
            grad_x.data_ptr<float>(), (int)L, (long long)(K * L),
            (int)N));
    }
}

// grad_w[K,O] from X' in [K, P] layout (P = N*L). For N==1 (and unless the
// cfg hook forces the single-gemm route) one batch-1 call goes straight into
// grad_w with no transpose; otherwise go is transposed to [P,O] once and a
// single GEMM contracts over P.
void compute_grad_w_from_kp(cublasHandle_t handle, cudaStream_t stream,
                            const torch::Tensor& go, const torch::Tensor& xp,
                            torch::Tensor& grad_w,
                            int64_t N, int64_t K, int64_t L, int64_t O,
                            bool allow_direct_n1) {
    const float one = 1.0f, zero = 0.0f;
    const int64_t P = N * L;

    if (N == 1 && allow_direct_n1) {
        // row-major grad_w[K,O] = X'[K,L] @ (go[O,L])^T
        TORCH_CUDABLAS_CHECK(cublasSgemmStridedBatched(
            handle, CUBLAS_OP_T, CUBLAS_OP_N,
            (int)O, (int)K, (int)L,
            &one,
            go.data_ptr<float>(), (int)L, (long long)(O * L),
            xp.data_ptr<float>(), (int)L, (long long)(K * L),
            &zero,
            grad_w.data_ptr<float>(), (int)O, (long long)(K * O),
            1));
        return;
    }

    torch::Tensor go_nlo;  // [P, O]
    const float* go_nlo_ptr;
    if (L == 1 || O == 1) {
        // [N,O,1] and [N,1,L] are already (n,l)-major: no copy needed
        go_nlo_ptr = go.data_ptr<float>();
    } else {
        go_nlo = at::empty({N, L, O}, go.options());
        dim3 block(TR_TILE, TR_ROWS);
        dim3 grid((unsigned)((L + TR_TILE - 1) / TR_TILE),
                  (unsigned)((O + TR_TILE - 1) / TR_TILE),
                  (unsigned)std::min<int64_t>(N, 65535));
        transpose_ol_batched_kernel<<<grid, block, 0, stream>>>(
            go.data_ptr<float>(), go_nlo.data_ptr<float>(),
            (int)N, (int)O, (int)L);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        go_nlo_ptr = go_nlo.data_ptr<float>();
    }

    // row-major C[K,O] = A[K,P] @ B[P,O]
    TORCH_CUDABLAS_CHECK(cublasSgemm(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        (int)O, (int)K, (int)P,
        &one,
        go_nlo_ptr, (int)O,
        xp.data_ptr<float>(), (int)P,
        &zero,
        grad_w.data_ptr<float>(), (int)O));
}

cublasHandle_t setup_cublas(cudaStream_t stream, cublasMath_t* prev_math) {
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    TORCH_CUDABLAS_CHECK(cublasSetStream(handle, stream));
    TORCH_CUDABLAS_CHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
    TORCH_CUDABLAS_CHECK(cublasGetMathMode(handle, prev_math));
    TORCH_CUDABLAS_CHECK(cublasSetMathMode(handle,
        at::globalContext().allowTF32CuBLAS() ? CUBLAS_TF32_TENSOR_OP_MATH
                                              : CUBLAS_DEFAULT_MATH));
    return handle;
}

// ---- op implementations ----

std::tuple<torch::Tensor, torch::Tensor> bgemm_lre_backward_claude_impl(
    const torch::Tensor& grad_output_in,  // [N, O, L]
    const torch::Tensor& x_in,            // [N, K, L]
    const torch::Tensor& w_in,            // [K, O] (transposed weight)
    const torch::Tensor& dx_in,           // [256]
    const torch::Tensor& dw_in,           // [256]
    int64_t cfg                           // -1 auto, 1 single-gemm grad_w, 2 batched grad_w
) {
    TORCH_CHECK(grad_output_in.is_cuda() && x_in.is_cuda() && w_in.is_cuda()
                && dx_in.is_cuda() && dw_in.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(grad_output_in.scalar_type() == torch::kFloat32
                && dx_in.scalar_type() == torch::kFloat32
                && dw_in.scalar_type() == torch::kFloat32,
                "grad_output/dx/dw must be float32");
    TORCH_CHECK(qdtype_ok(x_in.scalar_type()), "x must be float32, int8 or uint8");
    TORCH_CHECK(qdtype_ok(w_in.scalar_type()), "w must be float32, int8 or uint8");
    TORCH_CHECK(grad_output_in.dim() == 3, "grad_output must have shape [N, O, L]");
    TORCH_CHECK(x_in.dim() == 3, "x must have shape [N, K, L]");
    TORCH_CHECK(w_in.dim() == 2, "w must have shape [K, O]");
    TORCH_CHECK(dx_in.numel() == 256, "dx must have 256 elements");
    TORCH_CHECK(dw_in.numel() == 256, "dw must have 256 elements");

    const at::cuda::OptionalCUDAGuard device_guard(device_of(x_in));

    auto go = grad_output_in.contiguous();
    auto x = x_in.contiguous();
    auto w = w_in.contiguous();
    auto dx = dx_in.contiguous();
    auto dw = dw_in.contiguous();

    const int64_t N = x.size(0);
    const int64_t K = x.size(1);
    const int64_t L = x.size(2);
    const int64_t O = w.size(1);

    TORCH_CHECK(w.size(0) == K, "w.shape[0] must equal K");
    TORCH_CHECK(go.size(0) == N && go.size(1) == O && go.size(2) == L,
                "grad_output shape must be [N, O, L]");

    const int64_t P = N * L;
    TORCH_CHECK(P <= INT32_MAX && K <= INT32_MAX && O <= INT32_MAX,
                "dimensions too large");

    // gradients are always float32 regardless of x/w dtype
    auto fopts = x.options().dtype(torch::kFloat32);

    if (N == 0 || K == 0 || L == 0 || O == 0) {
        return std::make_tuple(at::zeros({N, K, L}, fopts), at::zeros({K, O}, fopts));
    }

    auto grad_x = at::empty({N, K, L}, fopts);
    auto grad_w = at::empty({K, O}, fopts);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cublasMath_t prev_math;
    cublasHandle_t handle = setup_cublas(stream, &prev_math);

    // ---- W' = DX[q(w)]  ([K, O]) ----
    auto wp = at::empty({K, O}, fopts);
    launch_lut_map_flat(stream, w, dx, wp);

    // ---- grad_x ----
    compute_grad_x(handle, go, wp, grad_x, N, K, L, O);

    // ---- grad_w ----
    // measured (RTX 6000 Ada sweep): the single big GEMM beats the
    // batched+sum path on every N>1 shape tested, even when K << L
    // (one large GEMM amortizes better than N small ones + reduce).
    bool batched;
    if (cfg == 1) {
        batched = false;
    } else if (cfg == 2) {
        batched = true;
    } else {
        batched = (N == 1);
    }

    if (batched && N > 1) {
        // cfg==2 tuning path: X' in natural [N,K,L]; ws[n] = X'[n] @ go[n]^T;
        // grad_w = sum_n ws[n]
        auto xp = at::empty({N, K, L}, fopts);
        launch_lut_map_flat(stream, x, dw, xp);

        auto ws = at::empty({N, K, O}, fopts);
        const float one = 1.0f, zero = 0.0f;
        TORCH_CUDABLAS_CHECK(cublasSgemmStridedBatched(
            handle, CUBLAS_OP_T, CUBLAS_OP_N,
            (int)O, (int)K, (int)L,
            &one,
            go.data_ptr<float>(), (int)L, (long long)(O * L),
            xp.data_ptr<float>(), (int)L, (long long)(K * L),
            &zero,
            ws.data_ptr<float>(), (int)O, (long long)(K * O),
            (int)N));
        at::sum_out(grad_w, ws, {0});
    } else {
        // X' in [K, P] layout (for N==1 the natural layout coincides, so the
        // cheaper flat kernel applies)
        auto xp = at::empty({K, P}, fopts);
        if (N == 1) launch_lut_map_flat(stream, x, dw, xp);
        else launch_lut_map_kp(stream, x, dw, xp, N, K, L);
        compute_grad_w_from_kp(handle, stream, go, xp, grad_w, N, K, L, O,
                               /*allow_direct_n1=*/cfg != 1);
    }

    TORCH_CUDABLAS_CHECK(cublasSetMathMode(handle, prev_math));

    return std::make_tuple(grad_x, grad_w);
}

std::tuple<torch::Tensor, torch::Tensor> bgemm_lre_backward_claude_im2col_impl(
    const torch::Tensor& grad_output_in,  // [N, O, L = OH*OW]
    const torch::Tensor& x_in,            // [N, C, H, W] pre-unfold image
    const torch::Tensor& w_in,            // [K = C*kh*kw, O]
    const torch::Tensor& dx_in,           // [256]
    const torch::Tensor& dw_in,           // [256]
    int64_t kh, int64_t kw, int64_t sh, int64_t sw,
    int64_t ph, int64_t pw, int64_t dilh, int64_t dilw
) {
    TORCH_CHECK(grad_output_in.is_cuda() && x_in.is_cuda() && w_in.is_cuda()
                && dx_in.is_cuda() && dw_in.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(grad_output_in.scalar_type() == torch::kFloat32
                && dx_in.scalar_type() == torch::kFloat32
                && dw_in.scalar_type() == torch::kFloat32,
                "grad_output/dx/dw must be float32");
    TORCH_CHECK(qdtype_ok(x_in.scalar_type()), "x must be float32, int8 or uint8");
    TORCH_CHECK(qdtype_ok(w_in.scalar_type()), "w must be float32, int8 or uint8");
    TORCH_CHECK(grad_output_in.dim() == 3, "grad_output must have shape [N, O, L]");
    TORCH_CHECK(x_in.dim() == 4, "x must have shape [N, C, H, W]");
    TORCH_CHECK(w_in.dim() == 2, "w must have shape [K, O]");
    TORCH_CHECK(dx_in.numel() == 256, "dx must have 256 elements");
    TORCH_CHECK(dw_in.numel() == 256, "dw must have 256 elements");
    TORCH_CHECK(kh >= 1 && kw >= 1 && sh >= 1 && sw >= 1 && ph >= 0 && pw >= 0
                && dilh >= 1 && dilw >= 1, "invalid conv geometry");

    const at::cuda::OptionalCUDAGuard device_guard(device_of(x_in));

    auto go = grad_output_in.contiguous();
    auto x = x_in.contiguous();
    auto w = w_in.contiguous();
    auto dx = dx_in.contiguous();
    auto dw = dw_in.contiguous();

    const int64_t N = x.size(0);
    const int64_t C = x.size(1);
    const int64_t H = x.size(2);
    const int64_t W = x.size(3);
    const int64_t K = C * kh * kw;
    const int64_t O = w.size(1);

    TORCH_CHECK(w.size(0) == K, "w.shape[0] must equal C*kh*kw");

    const int64_t OH = (H + 2 * ph - dilh * (kh - 1) - 1) / sh + 1;
    const int64_t OW = (W + 2 * pw - dilw * (kw - 1) - 1) / sw + 1;
    TORCH_CHECK(OH >= 1 && OW >= 1, "conv geometry yields empty output");
    const int64_t L = OH * OW;

    TORCH_CHECK(go.size(0) == N && go.size(1) == O && go.size(2) == L,
                "grad_output shape must be [N, O, OH*OW] for this geometry, got ",
                go.sizes(), " expected L=", L);

    const int64_t P = N * L;
    TORCH_CHECK(P <= INT32_MAX && K <= INT32_MAX && O <= INT32_MAX
                && N * C * H * W <= INT32_MAX * 4LL, "dimensions too large");

    auto fopts = x.options().dtype(torch::kFloat32);

    if (N == 0 || K == 0 || L == 0 || O == 0) {
        return std::make_tuple(at::zeros({N, K, L}, fopts), at::zeros({K, O}, fopts));
    }

    auto grad_x = at::empty({N, K, L}, fopts);
    auto grad_w = at::empty({K, O}, fopts);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cublasMath_t prev_math;
    cublasHandle_t handle = setup_cublas(stream, &prev_math);

    // ---- W' + grad_x (identical to the unfolded-x op) ----
    auto wp = at::empty({K, O}, fopts);
    launch_lut_map_flat(stream, w, dx, wp);
    compute_grad_x(handle, go, wp, grad_x, N, K, L, O);

    // ---- grad_w: implicit-im2col X' build + single GEMM over P ----
    ConvGeom g{(int)C, (int)H, (int)W, (int)kh, (int)kw, (int)sh, (int)sw,
               (int)ph, (int)pw, (int)dilh, (int)dilw, (int)OH, (int)OW};
    auto xp = at::empty({K, P}, fopts);
    launch_lut_map_kp_im2col(stream, x, dw, xp, N, g);
    compute_grad_w_from_kp(handle, stream, go, xp, grad_w, N, K, L, O,
                           /*allow_direct_n1=*/true);

    TORCH_CUDABLAS_CHECK(cublasSetMathMode(handle, prev_math));

    return std::make_tuple(grad_x, grad_w);
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> bgemm_lre_backward_claude(
    torch::Tensor grad_output, torch::Tensor x, torch::Tensor w,
    torch::Tensor dx, torch::Tensor dw
) {
    return bgemm_lre_backward_claude_impl(grad_output, x, w, dx, dw, -1);
}

std::tuple<torch::Tensor, torch::Tensor> bgemm_lre_backward_claude_cfg(
    torch::Tensor grad_output, torch::Tensor x, torch::Tensor w,
    torch::Tensor dx, torch::Tensor dw, int64_t cfg
) {
    return bgemm_lre_backward_claude_impl(grad_output, x, w, dx, dw, cfg);
}

std::tuple<torch::Tensor, torch::Tensor> bgemm_lre_backward_claude_im2col(
    torch::Tensor grad_output, torch::Tensor x, torch::Tensor w,
    torch::Tensor dx, torch::Tensor dw,
    int64_t kh, int64_t kw, int64_t sh, int64_t sw,
    int64_t ph, int64_t pw, int64_t dilh, int64_t dilw
) {
    return bgemm_lre_backward_claude_im2col_impl(
        grad_output, x, w, dx, dw, kh, kw, sh, sw, ph, pw, dilh, dilw);
}

TORCH_LIBRARY_FRAGMENT(approxtorch, m) {
    m.def("bgemm_lre_backward_claude(Tensor grad_output, Tensor x, Tensor w, Tensor dx, Tensor dw) -> (Tensor, Tensor)");
    m.def("bgemm_lre_backward_claude_cfg(Tensor grad_output, Tensor x, Tensor w, Tensor dx, Tensor dw, int cfg) -> (Tensor, Tensor)");
    m.def("bgemm_lre_backward_claude_im2col(Tensor grad_output, Tensor x, Tensor w, Tensor dx, Tensor dw, int kh, int kw, int sh, int sw, int ph, int pw, int dilh, int dilw) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(approxtorch, CUDA, m) {
    m.impl("bgemm_lre_backward_claude", &bgemm_lre_backward_claude);
    m.impl("bgemm_lre_backward_claude_cfg", &bgemm_lre_backward_claude_cfg);
    m.impl("bgemm_lre_backward_claude_im2col", &bgemm_lre_backward_claude_im2col);
}
