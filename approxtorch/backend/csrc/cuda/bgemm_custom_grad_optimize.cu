#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

namespace approxtorch {

// ============================================================================
// grad_X kernel — shared memory tiling over O
//
// Each block handles a tile of (ckk, l) for one batch n.
//   blockDim = (TILE_L, TILE_CKK)  e.g. (64, 4) = 256 threads
//   gridDim  = (ceil(L/TILE_L), ceil(CKK/TILE_CKK), N)
//
// For each O-tile:
//   1. Cooperatively load W[o_tile:o_tile+TILE_O, ckk_base:ckk_base+TILE_CKK]
//      into shared memory — reused by all L threads in the block
//   2. Cooperatively load dY[n, o_tile:o_tile+TILE_O, l_base:l_base+TILE_L]
//      into shared memory — reused by all CKK threads in the block
//   3. Each thread accumulates: sum_o dY[n,o,l] * dx_lut[256*x_val + W[o,ckk]]
//      where x_val is loaded once and stays in a register
// ============================================================================

#define DX_TILE_L   64
#define DX_TILE_CKK 4
#define DX_TILE_O   32

__global__ void bgemm_custom_grad_dx_kernel(
    const uint8_t* __restrict__ X,       // (N, CKK, L)
    const uint8_t* __restrict__ W,       // (O, CKK)
    const float*   __restrict__ dY,      // (N, O, L)
    const float*   __restrict__ dx_lut,  // (65536,)
    float*         __restrict__ grad_X,  // (N, CKK, L)
    int N, int CKK, int O, int L)
{
    int n        = blockIdx.z;
    int ckk_base = blockIdx.y * DX_TILE_CKK;
    int l_base   = blockIdx.x * DX_TILE_L;

    int l_local   = threadIdx.x;  // 0..DX_TILE_L-1
    int ckk_local = threadIdx.y;  // 0..DX_TILE_CKK-1

    int l_idx   = l_base + l_local;
    int ckk_idx = ckk_base + ckk_local;

    bool valid = (l_idx < L) && (ckk_idx < CKK);

    // Load x_val once — fixed across the entire O summation
    uint8_t x_val = 0;
    if (valid) {
        x_val = X[(size_t)n * CKK * L + (size_t)ckk_idx * L + l_idx];
    }
    int lut_row_base = (int)x_val << 8;  // x_val * 256

    // Shared memory
    __shared__ uint8_t s_W[DX_TILE_O][DX_TILE_CKK];
    __shared__ float   s_dY[DX_TILE_O][DX_TILE_L];

    int flat_tid = ckk_local * DX_TILE_L + l_local;
    int block_threads = DX_TILE_L * DX_TILE_CKK;  // 256

    float acc = 0.0f;

    for (int o_tile = 0; o_tile < O; o_tile += DX_TILE_O) {

        // --- Load W tile: [DX_TILE_O, DX_TILE_CKK] ---
        int w_count = DX_TILE_O * DX_TILE_CKK;
        for (int i = flat_tid; i < w_count; i += block_threads) {
            int o_loc = i / DX_TILE_CKK;
            int c_loc = i % DX_TILE_CKK;
            int o_idx = o_tile + o_loc;
            int c_idx = ckk_base + c_loc;
            if (o_idx < O && c_idx < CKK) {
                s_W[o_loc][c_loc] = W[(size_t)o_idx * CKK + c_idx];
            } else {
                s_W[o_loc][c_loc] = 0;
            }
        }

        // --- Load dY tile: [DX_TILE_O, DX_TILE_L] ---
        int dy_count = DX_TILE_O * DX_TILE_L;
        for (int i = flat_tid; i < dy_count; i += block_threads) {
            int o_loc = i / DX_TILE_L;
            int l_loc = i % DX_TILE_L;
            int o_idx = o_tile + o_loc;
            int l_real = l_base + l_loc;
            if (o_idx < O && l_real < L) {
                s_dY[o_loc][l_loc] = dY[(size_t)n * O * L + (size_t)o_idx * L + l_real];
            } else {
                s_dY[o_loc][l_loc] = 0.0f;
            }
        }

        __syncthreads();

        // --- Compute: accumulate over this O tile ---
        if (valid) {
            int tile_end = min(DX_TILE_O, O - o_tile);
            #pragma unroll 8
            for (int o_loc = 0; o_loc < tile_end; ++o_loc) {
                float dy_val  = s_dY[o_loc][l_local];
                uint8_t w_val = s_W[o_loc][ckk_local];
                float lut_val = __ldg(&dx_lut[lut_row_base + (int)w_val]);
                acc += dy_val * lut_val;
            }
        }

        __syncthreads();
    }

    if (valid) {
        grad_X[(size_t)n * CKK * L + (size_t)ckk_idx * L + l_idx] = acc;
    }
}


// ============================================================================
// grad_W kernel — LUT column preload + large tile + warp shuffle reduction
//
// Each block handles one (o, ckk) pair.
//   gridDim  = (O, CKK)        — one block per output element
//   blockDim = (DW_BLOCK_SIZE)  e.g. 256
//
// Key optimizations:
//   1. w_val = W[o, ckk] is constant for the block. Preload the entire LUT
//      column dw_lut[i*256 + w_val] for i=0..255 into shared memory (1KB).
//      All subsequent lookups become fast shared memory reads.
//   2. Each block sweeps over ALL N*L elements (no atomicAdd needed).
//   3. Final reduction uses warp shuffle + shared memory.
// ============================================================================

#define DW_BLOCK_SIZE 256

__global__ void bgemm_custom_grad_dw_kernel(
    const uint8_t* __restrict__ X,       // (N, CKK, L)
    const uint8_t* __restrict__ W,       // (O, CKK)
    const float*   __restrict__ dY,      // (N, O, L)
    const float*   __restrict__ dw_lut,  // (65536,)
    float*         __restrict__ grad_W,  // (O, CKK)
    int N, int CKK, int O, int L)
{
    int o_idx   = blockIdx.x;
    int ckk_idx = blockIdx.y;

    if (o_idx >= O || ckk_idx >= CKK) return;

    int tid = threadIdx.x;

    // Load w_val — constant for this entire block
    uint8_t w_val = W[(size_t)o_idx * CKK + ckk_idx];

    // Preload LUT column into shared memory:
    // s_lut[i] = dw_lut[i * 256 + w_val]  for i = 0..255
    // Only 256 floats = 1KB — tiny, fits easily
    __shared__ float s_lut[256];
    if (tid < 256) {
        s_lut[tid] = dw_lut[(int)tid * 256 + (int)w_val];
    }
    __syncthreads();

    // Sweep over all (n, l) pairs, accumulate into register
    int total_nl = N * L;
    float acc = 0.0f;

    for (int nl = tid; nl < total_nl; nl += DW_BLOCK_SIZE) {
        int n = nl / L;
        int l = nl % L;

        uint8_t x_val = __ldg(&X[(size_t)n * CKK * L + (size_t)ckk_idx * L + l]);
        float dy_val  = __ldg(&dY[(size_t)n * O * L + (size_t)o_idx * L + l]);

        // LUT lookup from shared memory — very fast
        acc += dy_val * s_lut[x_val];
    }

    // --- Warp-level reduction ---
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    // --- Block-level reduction via shared memory ---
    __shared__ float s_reduce[32];  // one slot per warp
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    if (lane_id == 0) {
        s_reduce[warp_id] = acc;
    }
    __syncthreads();

    // Final reduction by first warp
    if (warp_id == 0) {
        acc = (lane_id < (DW_BLOCK_SIZE / 32)) ? s_reduce[lane_id] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
        }
        if (lane_id == 0) {
            grad_W[(size_t)o_idx * CKK + ckk_idx] = acc;
        }
    }
}


// ============================================================================
// Host functions
// ============================================================================

torch::Tensor bgemm_custom_grad_uint8_dx(
    torch::Tensor X,       // (N, CKK, L) uint8
    torch::Tensor W,       // (O, CKK) uint8
    torch::Tensor dY,      // (N, O, L) float32
    torch::Tensor dx_lut   // (65536,) float32
) {
    TORCH_CHECK(X.is_cuda() && W.is_cuda() && dY.is_cuda());
    TORCH_CHECK(X.dtype() == torch::kUInt8);
    TORCH_CHECK(W.dtype() == torch::kUInt8);
    TORCH_CHECK(dY.dtype() == torch::kFloat32);
    TORCH_CHECK(dx_lut.dtype() == torch::kFloat32);

    X = X.contiguous();
    W = W.contiguous();
    dY = dY.contiguous();
    dx_lut = dx_lut.contiguous().view({-1});

    int N   = X.size(0);
    int CKK = X.size(1);
    int L   = X.size(2);
    int O   = W.size(0);

    TORCH_CHECK(W.size(1) == CKK);
    TORCH_CHECK(dY.size(0) == N && dY.size(1) == O && dY.size(2) == L);
    TORCH_CHECK(dx_lut.numel() == 65536);

    auto grad_X = torch::empty({N, CKK, L}, torch::dtype(torch::kFloat32).device(X.device()));

    dim3 block(DX_TILE_L, DX_TILE_CKK);  // (64, 4) = 256 threads
    dim3 grid(
        (L   + DX_TILE_L   - 1) / DX_TILE_L,
        (CKK + DX_TILE_CKK - 1) / DX_TILE_CKK,
        N
    );

    bgemm_custom_grad_dx_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        X.data_ptr<uint8_t>(),
        W.data_ptr<uint8_t>(),
        dY.data_ptr<float>(),
        dx_lut.data_ptr<float>(),
        grad_X.data_ptr<float>(),
        N, CKK, O, L
    );

    return grad_X;
}

torch::Tensor bgemm_custom_grad_uint8_dw(
    torch::Tensor X,       // (N, CKK, L) uint8
    torch::Tensor W,       // (O, CKK) uint8
    torch::Tensor dY,      // (N, O, L) float32
    torch::Tensor dw_lut   // (65536,) float32
) {
    TORCH_CHECK(X.is_cuda() && W.is_cuda() && dY.is_cuda());
    TORCH_CHECK(X.dtype() == torch::kUInt8);
    TORCH_CHECK(W.dtype() == torch::kUInt8);
    TORCH_CHECK(dY.dtype() == torch::kFloat32);
    TORCH_CHECK(dw_lut.dtype() == torch::kFloat32);

    X = X.contiguous();
    W = W.contiguous();
    dY = dY.contiguous();
    dw_lut = dw_lut.contiguous().view({-1});

    int N   = X.size(0);
    int CKK = X.size(1);
    int L   = X.size(2);
    int O   = W.size(0);

    TORCH_CHECK(W.size(1) == CKK);
    TORCH_CHECK(dY.size(0) == N && dY.size(1) == O && dY.size(2) == L);
    TORCH_CHECK(dw_lut.numel() == 65536);

    // No need for zeros — no atomicAdd, each block writes exactly one element
    auto grad_W = torch::empty({O, CKK}, torch::dtype(torch::kFloat32).device(W.device()));

    dim3 block(DW_BLOCK_SIZE);  // 256
    dim3 grid(O, CKK);          // one block per (o, ckk)

    bgemm_custom_grad_dw_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        X.data_ptr<uint8_t>(),
        W.data_ptr<uint8_t>(),
        dY.data_ptr<float>(),
        dw_lut.data_ptr<float>(),
        grad_W.data_ptr<float>(),
        N, CKK, O, L
    );

    return grad_W;
}

// ============================================================================
// Registration
// ============================================================================

TORCH_LIBRARY_FRAGMENT(approxtorch, m) {
    m.def("bgemm_custom_grad_uint8_dx(Tensor X, Tensor W, Tensor dY, Tensor dx_lut) -> Tensor");
    m.def("bgemm_custom_grad_uint8_dw(Tensor X, Tensor W, Tensor dY, Tensor dw_lut) -> Tensor");
}

TORCH_LIBRARY_IMPL(approxtorch, CUDA, m) {
    m.impl("bgemm_custom_grad_uint8_dx", &bgemm_custom_grad_uint8_dx);
    m.impl("bgemm_custom_grad_uint8_dw", &bgemm_custom_grad_uint8_dw);
}

}  // namespace approxtorch
