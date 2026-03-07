#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>



namespace approxtorch{



// TILE sizes
#define GX_TILE_CKK 4
#define GX_TILE_L   64
#define GX_TILE_O   64

__global__ void grad_x_kernel_uint8(
    const uint8_t* __restrict__ X,      // [N, CKK, L]
    const uint8_t* __restrict__ W,      // [O, CKK]
    const float*   __restrict__ upstream,// [N, O, L]
    const float*   __restrict__ dx_table,// [256*256]
    float*         __restrict__ grad_X,  // [N, CKK, L]
    int N, int CKK, int L, int O)
{
    // Block handles a tile of (ckk, l) for batch n
    int n   = blockIdx.z;
    int ckk_base = blockIdx.y * GX_TILE_CKK;
    int l_base   = blockIdx.x * GX_TILE_L;

    int tid = threadIdx.x; // blockDim.x = GX_TILE_L (e.g., 64)
    int ckk_local = threadIdx.y; // blockDim.y = GX_TILE_CKK (e.g., 4)

    int ckk_idx = ckk_base + ckk_local;
    int l_idx   = l_base + tid;

    if (ckk_idx >= CKK) return;

    // Load x value (fixed for this thread across all o)
    uint8_t x_val = 0;
    bool valid_l = (l_idx < L);
    if (valid_l) {
        x_val = X[(size_t)n * CKK * L + (size_t)ckk_idx * L + l_idx];
    }

    // Shared memory for W tile: [GX_TILE_O][GX_TILE_CKK]
    __shared__ uint8_t s_W[GX_TILE_O][GX_TILE_CKK];
    // Shared memory for upstream tile: [GX_TILE_O][GX_TILE_L]
    __shared__ float s_up[GX_TILE_O][GX_TILE_L];
    // Shared memory for dx_table rows: for each ckk_local, a row of 256 floats
    // Since x_val can differ per thread in the L dimension, we can't share a single row.
    // Instead, we'll use registers + direct LUT access.
    // 
    // Better approach: preload the row for x_val into registers (256 is too many).
    // Alternative: just do the lookup from global memory — dx_table is 256KB, fits in L2.
    // With the tiling of W into shared memory, the W access pattern becomes coalesced.

    float acc = 0.0f;

    // Precompute the base offset into dx_table for this thread's x_val
    int lut_row_base = (int)x_val * 256;

    for (int o_tile = 0; o_tile < O; o_tile += GX_TILE_O) {
        // Collaboratively load W tile into shared memory
        // W[o, ckk] — we need W[o_tile..o_tile+TILE_O-1, ckk_base..ckk_base+TILE_CKK-1]
        {
            int load_count = GX_TILE_O * GX_TILE_CKK; // e.g., 64*4 = 256
            int threads_per_block = GX_TILE_L * GX_TILE_CKK; // e.g., 64*4 = 256
            int flat_tid = ckk_local * GX_TILE_L + tid;
            // Each thread loads one element
            for (int i = flat_tid; i < load_count; i += threads_per_block) {
                int o_local = i / GX_TILE_CKK;
                int c_local = i % GX_TILE_CKK;
                int o_idx = o_tile + o_local;
                int c_idx = ckk_base + c_local;
                if (o_idx < O && c_idx < CKK) {
                    s_W[o_local][c_local] = W[(size_t)o_idx * CKK + c_idx];
                } else {
                    s_W[o_local][c_local] = 0;
                }
            }
        }

        // Collaboratively load upstream tile into shared memory
        // upstream[n, o, l] — we need upstream[n, o_tile..., l_base...]
        {
            int load_count = GX_TILE_O * GX_TILE_L;
            int threads_per_block = GX_TILE_L * GX_TILE_CKK;
            int flat_tid = ckk_local * GX_TILE_L + tid;
            for (int i = flat_tid; i < load_count; i += threads_per_block) {
                int o_local = i / GX_TILE_L;
                int l_local = i % GX_TILE_L;
                int o_idx = o_tile + o_local;
                int l_real = l_base + l_local;
                if (o_idx < O && l_real < L) {
                    s_up[o_local][l_local] = upstream[(size_t)n * O * L + (size_t)o_idx * L + l_real];
                } else {
                    s_up[o_local][l_local] = 0.0f;
                }
            }
        }

        __syncthreads();

        if (valid_l) {
            int tile_end = min(GX_TILE_O, O - o_tile);
            // Unrolled inner loop
            #pragma unroll 8
            for (int o_local = 0; o_local < tile_end; o_local++) {
                uint8_t w_val = s_W[o_local][ckk_local];
                float up_val = s_up[o_local][tid];
                float lut_val = dx_table[lut_row_base + w_val];
                acc += up_val * lut_val;
            }
        }

        __syncthreads();
    }

    if (valid_l && ckk_idx < CKK) {
        grad_X[(size_t)n * CKK * L + (size_t)ckk_idx * L + l_idx] = acc;
    }
}


// ============================================================================
// Kernel 2: grad_W  — for each (o, ckk), sum over (N, L)
//
// grad_W[o, ckk] = sum_{n,l} upstream[n, o, l] * dw_table[256 * X[n,ckk,l] + W[o,ckk]]
//
// Strategy:
//   - Each thread block handles one or a few (o, ckk) pairs
//   - For a fixed (o, ckk), w_val = W[o, ckk] is a constant
//   - The LUT lookup becomes dw_table[256*x_val + w_val] where only x_val varies
//   - We preload the column of the LUT (dw_table[256*i + w_val] for i=0..255) 
//     into shared memory — only 256 floats = 1KB!
//   - Then iterate over (n, l) in tiles, doing coalesced reads of X and upstream
// ============================================================================

#define GW_BLOCK_SIZE 256
#define GW_TILE_NL    1024  // number of (n*L + l) elements per tile

__global__ void grad_w_kernel_uint8(
    const uint8_t* __restrict__ X,        // [N, CKK, L]
    const uint8_t* __restrict__ W,        // [O, CKK]
    const float*   __restrict__ upstream,  // [N, O, L]
    const float*   __restrict__ dw_table,  // [256*256]
    float*         __restrict__ grad_W,    // [O, CKK]
    int N, int CKK, int L, int O)
{
    int o_idx   = blockIdx.x;
    int ckk_idx = blockIdx.y;

    if (o_idx >= O || ckk_idx >= CKK) return;

    int tid = threadIdx.x;

    // Load w_val for this (o, ckk) — constant across all (n, l)
    uint8_t w_val = W[(size_t)o_idx * CKK + ckk_idx];

    // Preload the LUT column: dw_table[256*i + w_val] for i = 0..255
    __shared__ float s_lut[256];
    if (tid < 256) {
        s_lut[tid] = dw_table[tid * 256 + w_val];
    }
    __syncthreads();

    // Now iterate over all (n, l) pairs, accumulate
    // Total elements: N * L
    // X[n, ckk_idx, l] is at offset n*CKK*L + ckk_idx*L + l
    // upstream[n, o_idx, l] is at offset n*O*L + o_idx*L + l

    float acc = 0.0f;
    int total_nl = N * L;

    for (int base = 0; base < total_nl; base += GW_BLOCK_SIZE) {
        int idx = base + tid;
        if (idx < total_nl) {
            int n = idx / L;
            int l = idx % L;
            uint8_t x_val = X[(size_t)n * CKK * L + (size_t)ckk_idx * L + l];
            float up_val  = upstream[(size_t)n * O * L + (size_t)o_idx * L + l];
            acc += up_val * s_lut[x_val];
        }
    }

    // Warp-level reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    // Block-level reduction using shared memory
    __shared__ float s_reduce[32]; // one per warp
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    if (lane_id == 0) {
        s_reduce[warp_id] = acc;
    }
    __syncthreads();

    // Final reduction by first warp
    if (warp_id == 0) {
        acc = (lane_id < (GW_BLOCK_SIZE / 32)) ? s_reduce[lane_id] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
        }
        if (lane_id == 0) {
            grad_W[(size_t)o_idx * CKK + ckk_idx] = acc;
        }
    }
}



torch::Tensor bgemm_custom_grad_uint8_dx(
    torch::Tensor& X,
    torch::Tensor& W,
    torch::Tensor& dY,
    torch::Tensor& dx_lut)
{
    X = X.contiguous();
    W = W.contiguous();
    dY = dY.contiguous();
    dx_lut = dx_lut.contiguous().view({-1});  // flatten to (65536,)

    int N = X.size(0);
    int CKK = X.size(1);
    int L = X.size(2);
    int O = W.size(0);

    TORCH_CHECK(W.size(1) == CKK);
    TORCH_CHECK(dY.size(0) == N && dY.size(1) == O && dY.size(2) == L);
    TORCH_CHECK(dx_lut.numel() == 65536);

    auto grad_X = torch::empty({N, CKK, L}, 
        torch::dtype(torch::kFloat32).device(X.device()));

    dim3 block(GX_TILE_L, GX_TILE_CKK);  // (64, 4) = 256 threads
    dim3 grid(
        (L + GX_TILE_L - 1) / GX_TILE_L,
        (CKK + GX_TILE_CKK - 1) / GX_TILE_CKK,
        N
    );
    grad_x_kernel_uint8<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        X.data_ptr<uint8_t>(),
        W.data_ptr<uint8_t>(),
        dY.data_ptr<float>(),
        dx_lut.data_ptr<float>(),
        grad_X.data_ptr<float>(),
        N, CKK, L, O
    );
    return grad_X;
}



torch::Tensor bgemm_custom_grad_uint8_dw(
    torch::Tensor& X,
    torch::Tensor& W,
    torch::Tensor& dY,
    torch::Tensor& dW_lut)
{
    X = X.contiguous();
    W = W.contiguous();
    dY = dY.contiguous();
    dW_lut = dW_lut.contiguous().view({-1});  // flatten to (65536,)

    int N = X.size(0);
    int CKK = X.size(1);
    int L = X.size(2);
    int O = W.size(0);

    TORCH_CHECK(W.size(1) == CKK);
    TORCH_CHECK(dY.size(0) == N && dY.size(1) == O && dY.size(2) == L);
    TORCH_CHECK(dW_lut.numel() == 65536);

    auto grad_W = torch::empty({O, CKK}, 
        torch::dtype(torch::kFloat32).device(X.device()));

    dim3 block(GW_BLOCK_SIZE);
    dim3 grid(O, CKK);
    grad_w_kernel_uint8<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        X.data_ptr<uint8_t>(),
        W.data_ptr<uint8_t>(),
        dY.data_ptr<float>(),
        dW_lut.data_ptr<float>(),
        grad_W.data_ptr<float>(),
        N, CKK, L, O
    );
    return grad_W;
}




TORCH_LIBRARY_FRAGMENT(approxtorch, m){
    // m.def("im2col_uint8(Tensor input, int k_h, int k_w, int stride_h, int stride_w, int padding_h, int padding_w, int dilation_h, int dilation_w) -> Tensor");
    m.def(" bgemm_custom_grad_uint8_dx(Tensor X, Tensor W, Tensor dY, Tensor dx_lut) -> Tensor");
    m.def(" bgemm_custom_grad_uint8_dw(Tensor X, Tensor W, Tensor dY, Tensor dW_lut) -> Tensor");
}

TORCH_LIBRARY_IMPL(approxtorch, CUDA, m){
    m.impl("bgemm_custom_grad_uint8_dx", &bgemm_custom_grad_uint8_dx);
    m.impl("bgemm_custom_grad_uint8_dw", &bgemm_custom_grad_uint8_dw);
}






}