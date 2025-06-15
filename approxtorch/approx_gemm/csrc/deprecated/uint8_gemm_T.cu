// gemm_uint8.cu — uint8 GEMM with vectorized + tail‑scalar path
// -----------------------------------------------------------------------------
// This version adds **Scheme 2**: scalar fallback for the remaining K elements
// when K is **not** divisible by 4. The main vectorized path is unchanged.
//
// Build (example):
//   from torch.utils.cpp_extension import load
//   gemm_uint8 = load(name="gemm_uint8", sources=["gemm_uint8.cu"], extra_cuda_cflags=["-O3"])
//
// Usage in Python:
//   import torch, gemm_uint8
//   A = torch.randint(0, 256, (M, K), dtype=torch.uint8, device="cuda")
//   B = torch.randint(0, 256, (K, N), dtype=torch.uint8, device="cuda")
//   C = gemm_uint8.gemm_uint8(A, B)  # C is int32 of shape (M, N)
// -----------------------------------------------------------------------------
#include "gemm.cuh"

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

using uint8 = uint8_t;

// ---- Tunables ---------------------------------------------------------------
constexpr int BLOCK_M = 128;
constexpr int BLOCK_N = 128;
constexpr int BLOCK_K = 32;     // must be multiple of 4 (vector width)

constexpr int THREADS_X = 32;   // along N (cols)
constexpr int THREADS_Y = 8;    // along M (rows)
constexpr int VEC_WIDTH = 4;    // uchar4

// ---- Helpers ----------------------------------------------------------------
__device__ __forceinline__ uint32_t vload4(const uint8* p) {
    return *reinterpret_cast<const uint32_t*>(p);  // address must be >=4‑byte aligned
}

// ---- Kernel -----------------------------------------------------------------
__global__ void gemm_uint8_kernel(const uint8* __restrict__ A,
                                  const uint8* __restrict__ B,
                                  int32_t*      __restrict__ C,
                                  int M, int N, int K,
                                  int lda, int ldb, int ldc)
{
    // Block/tile origin -------------------------------------------------------
    const int block_row = blockIdx.y;
    const int block_col = blockIdx.x;

    const int tid_x = threadIdx.x;   // 0‥THREADS_X‑1 along N
    const int tid_y = threadIdx.y;   // 0‥THREADS_Y‑1 along M

    constexpr int TM = 4;  // rows per thread
    constexpr int TN = 4;  // cols per thread

    const int row_start = block_row * BLOCK_M + tid_y * TM;
    const int col_start = block_col * BLOCK_N + tid_x * TN;

    int32_t acc[TM][TN] = {0};

    // Shared memory for one (A_tile, B_tile) pair ---------------------------
    extern __shared__ uint8 smem[];
    uint8* As = smem;                               // BLOCK_M × BLOCK_K
    uint8* Bs = As + BLOCK_M * BLOCK_K;             // BLOCK_K × BLOCK_N

    // ---- MAIN VECTOR LOOP (uchar4 path) -----------------------------------
    const int K_vec_end = (K & ~0x3);   // floor to multiple of 4

    for (int k0 = 0; k0 < K_vec_end; k0 += BLOCK_K) {
        // Load A -------------------------------------------------------------
        #pragma unroll
        for (int i = 0; i < TM; ++i) {
            const int g_row = row_start + i;
            const int g_col_vec = k0 + tid_x * VEC_WIDTH;
            if (g_row < M && g_col_vec + VEC_WIDTH - 1 < K) {
                uint32_t vec = vload4(&A[g_row * lda + g_col_vec]);
                *reinterpret_cast<uint32_t*>(&As[(tid_y * TM + i) * BLOCK_K + tid_x * VEC_WIDTH]) = vec;
            }
        }
        // Load B -------------------------------------------------------------
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int g_row_vec = k0 + tid_y * VEC_WIDTH;
            const int g_col = col_start + j;
            if (g_row_vec + VEC_WIDTH - 1 < K && g_col < N) {
                uint32_t vec = vload4(&B[g_row_vec * ldb + g_col]);
                *reinterpret_cast<uint32_t*>(&Bs[(tid_y * VEC_WIDTH) * BLOCK_N + tid_x * TN + j]) = vec;
            }
        }
        __syncthreads();

        // Compute ------------------------------------------------------------
        #pragma unroll
        for (int k_inner = 0; k_inner < BLOCK_K && k0 + k_inner < K_vec_end; ++k_inner) {
            uint8 a_frag[TM];
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                a_frag[i] = As[(tid_y * TM + i) * BLOCK_K + k_inner];
            }
            uint8 b_frag[TN];
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                b_frag[j] = Bs[k_inner * BLOCK_N + tid_x * TN + j];
            }
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] += static_cast<int32_t>(a_frag[i]) * static_cast<int32_t>(b_frag[j]);
                }
            }
        }
        __syncthreads();
    } // end main vector loop

    // ---- TAIL SCALAR LOOP (K % 4) ----------------------------------------
    const int K_tail = K & 0x3;  // 0‑3
    if (K_tail) {
        const int k_scalar_start = K_vec_end;       // first remainder index
        // For each scalar k tail, directly read A & B from **global** memory
        for (int k = 0; k < K_tail; ++k) {
            const int kk = k_scalar_start + k;
            // Gather A scalars for TM rows ----------------------------------
            uint8 a_tail[TM];
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                const int g_row = row_start + i;
                a_tail[i] = (g_row < M) ? A[g_row * lda + kk] : 0;
            }
            // Gather B scalars for TN cols ----------------------------------
            uint8 b_tail[TN];
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                const int g_col = col_start + j;
                b_tail[j] = (g_col < N) ? B[kk * ldb + g_col] : 0;
            }
            // FMA -----------------------------------------------------------
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] += static_cast<int32_t>(a_tail[i]) * static_cast<int32_t>(b_tail[j]);
                }
            }
        }
    }

    // ---- Store C -----------------------------------------------------------
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int g_row = row_start + i;
        if (g_row < M) {
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                const int g_col = col_start + j;
                if (g_col < N) {
                    C[g_row * ldc + g_col] = acc[i][j];
                }
            }
        }
    }
}

// ---- Host wrapper -----------------------------------------------------------
static torch::Tensor gemm_uint8(torch::Tensor A, torch::Tensor B, torch::Tensor lut) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    CHECK_INPUT(lut);
    TORCH_CHECK(A.dtype() == torch::kUInt8 && B.dtype() == torch::kUInt8,
                "A and B must be uint8");
    TORCH_CHECK(A.size(1) == B.size(0), "A.cols must equal B.rows");

    const int64_t M = A.size(0);
    const int64_t K = A.size(1);
    const int64_t N = B.size(1);

    auto C = torch::empty({M, N}, torch::TensorOptions().dtype(torch::kInt32).device(A.device()));

    dim3 block(THREADS_X, THREADS_Y);
    dim3 grid((N + BLOCK_N - 1) / BLOCK_N,
              (M + BLOCK_M - 1) / BLOCK_M);

    const size_t smem_bytes = (BLOCK_M * BLOCK_K + BLOCK_K * BLOCK_N) * sizeof(uint8);

    gemm_uint8_kernel<<<grid, block, smem_bytes, at::cuda::getCurrentCUDAStream()>>>(
        A.data_ptr<uint8>(),
        B.data_ptr<uint8>(),
        C.data_ptr<int32_t>(),
        static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
        static_cast<int>(K), static_cast<int>(N), static_cast<int>(N));

    return C;
}

// ---- PyBind -----------------------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_uint8", &gemm_uint8, "uint8 GEMM with vectorized & scalar tail (CUDA)");
}
