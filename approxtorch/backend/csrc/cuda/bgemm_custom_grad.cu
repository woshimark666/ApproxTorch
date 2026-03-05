#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

#define TILE_O 32
#define BLOCK_SIZE 256
namespace approxtorch{

// ===========================================================================
// Fused kernel v3: reduce syncthreads overhead by accumulating multiple o
// values between reductions. Instead of reducing after every o, accumulate
// TILE_O partial sums per thread in registers, then reduce TILE_O times.
// But we need per-o sums... so we store partial sums in registers array.
// ===========================================================================
__global__ void bgemm_custom_grad_uint8_kernel(
    const uint8_t* __restrict__ X,      // (N, CKK, L)
    const uint8_t* __restrict__ W,      // (O, CKK)
    const float*   __restrict__ dY,     // (N, O, L)
    const float*   __restrict__ dx_lut, // (65536,)
    const float*   __restrict__ dw_lut, // (65536,)
    float*         __restrict__ grad_X,  // (N, CKK, L)
    float*         __restrict__ grad_W,  // (O, CKK)
    int N, int CKK, int O, int L
) {
    int c = blockIdx.x;
    if (c >= CKK) return;

    int NL = N * L;
    int nl_base = blockIdx.y * BLOCK_SIZE;
    int nl = nl_base + threadIdx.x;

    int n = -1, l = -1;
    uint8_t x_val = 0;
    int lut_row = 0;
    bool valid = (nl < NL);

    if (valid) {
        n = nl / L;
        l = nl % L;
        x_val = X[(n * CKK + c) * L + l];
        lut_row = (int)x_val << 8;
    }

    __shared__ uint8_t smem_w[TILE_O];
    // For batched reduction: TILE_O rows x BLOCK_SIZE
    // That's TILE_O * 256 * 4 = 32KB for TILE_O=32 -- fits.
    __shared__ float smem_reduce[TILE_O * BLOCK_SIZE];

    float grad_x_accum = 0.0f;

    for (int o_base = 0; o_base < O; o_base += TILE_O) {
        int tile_end = min(o_base + TILE_O, O);
        int tile_size = tile_end - o_base;

        // Load W tile
        if (threadIdx.x < tile_size) {
            smem_w[threadIdx.x] = W[(o_base + threadIdx.x) * CKK + c];
        }
        __syncthreads();

        // Each thread computes its partial products for all o in tile
        // and stores them in smem_reduce[t * BLOCK_SIZE + threadIdx.x]
        for (int t = 0; t < tile_size; ++t) {
            uint8_t w_val = smem_w[t];
            int lut_idx = lut_row + (int)w_val;

            float dy_val = 0.0f;
            float dx_val = 0.0f;
            float dw_val = 0.0f;

            if (valid) {
                int o = o_base + t;
                dy_val = __ldg(&dY[(n * O + o) * L + l]);
                dx_val = __ldg(&dx_lut[lut_idx]);
                dw_val = __ldg(&dw_lut[lut_idx]);
                grad_x_accum += dy_val * dx_val;
            }

            smem_reduce[t * BLOCK_SIZE + threadIdx.x] = valid ? (dy_val * dw_val) : 0.0f;
        }
        // Zero out unused slots
        for (int t = tile_size; t < TILE_O; ++t) {
            smem_reduce[t * BLOCK_SIZE + threadIdx.x] = 0.0f;
        }
        __syncthreads();

        // Now reduce each of the tile_size rows
        for (int t = 0; t < tile_size; ++t) {
            float* row = &smem_reduce[t * BLOCK_SIZE];
            for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
                if (threadIdx.x < s) {
                    row[threadIdx.x] += row[threadIdx.x + s];
                }
                __syncthreads();
            }
            if (threadIdx.x == 0) {
                int o = o_base + t;
                atomicAdd(&grad_W[o * CKK + c], row[0]);
            }
        }
        __syncthreads();
    }

    if (valid) {
        grad_X[(n * CKK + c) * L + l] = grad_x_accum;
    }
}


std::tuple<torch::Tensor, torch::Tensor> bgemm_custom_grad_uint8(
    torch::Tensor X,       // (N, CKK, L) uint8
    torch::Tensor W,       // (O, CKK) uint8
    torch::Tensor dY,      // (N, O, L) float32
    torch::Tensor dx_lut,  // (256, 256) float32
    torch::Tensor dw_lut  // (256, 256) float32
) {
    TORCH_CHECK(X.is_cuda() && W.is_cuda() && dY.is_cuda());
    TORCH_CHECK(X.dtype() == torch::kUInt8);
    TORCH_CHECK(W.dtype() == torch::kUInt8);
    TORCH_CHECK(dY.dtype() == torch::kFloat32);

    X = X.contiguous();
    W = W.contiguous();
    dY = dY.contiguous();
    dx_lut = dx_lut.contiguous().view({-1});
    dw_lut = dw_lut.contiguous().view({-1});

    int N   = X.size(0);
    int CKK = X.size(1);
    int L   = X.size(2);
    int O   = W.size(0);

    TORCH_CHECK(W.size(1) == CKK);
    TORCH_CHECK(dY.size(0) == N && dY.size(1) == O && dY.size(2) == L);

    auto grad_X = torch::empty({N, CKK, L}, torch::dtype(torch::kFloat32).device(X.device()));
    auto grad_W = torch::zeros({O, CKK}, torch::dtype(torch::kFloat32).device(W.device()));

    int NL = N * L;
    dim3 grid(CKK, (NL + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE);

    auto x_ptr  = X.data_ptr<uint8_t>();
    auto w_ptr  = W.data_ptr<uint8_t>();
    auto dy_ptr = dY.data_ptr<float>();
    auto dx_ptr = dx_lut.data_ptr<float>();
    auto dw_ptr = dw_lut.data_ptr<float>();
    auto gx_ptr = grad_X.data_ptr<float>();
    auto gw_ptr = grad_W.data_ptr<float>();

    bgemm_custom_grad_uint8_kernel<<<grid, block>>>(
        x_ptr, w_ptr, dy_ptr, dx_ptr, dw_ptr, gx_ptr, gw_ptr,
        N, CKK, O, L);

    return std::make_tuple(grad_X, grad_W);
}

// ================================================================================
// ================================version2==============================================
// ================================================================================
// ================================================================================
// ================================================================================


// #define GRAD_W_BLOCK 256
// #define GRAD_W_NL_TILE 256   // each block handles this many (n*l) elements
// #define TILE_O 32

// __global__ void bgemm_custom_grad_dx_uint8_kernel(
//     const uint8_t* __restrict__ X,      // (N, CKK, L)
//     const uint8_t* __restrict__ W,      // (O, CKK)
//     const float*   __restrict__ dY,     // (N, O, L)
//     const float*   __restrict__ dx_lut, // (65536,)
//     float*         __restrict__ grad_X,  // (N, CKK, L)
//     int N, int CKK, int O, int L
// ) {
//     // blockDim.x threads, each handles one (n,c,l)
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     int total = N * CKK * L;

//     // Decode (n, c, l) -- need c for W indexing
//     int l_idx = (idx < total) ? (idx % L) : 0;
//     int tmp   = (idx < total) ? (idx / L) : 0;
//     int c_idx = tmp % CKK;
//     int n_idx = tmp / CKK;

//     // Load x_val once
//     uint8_t x_val = (idx < total) ? X[idx] : 0;
//     int lut_row = (int)x_val << 8; // x_val * 256

//     float sum = 0.0f;

//     // Shared memory for a tile of W values: W[o_tile..o_tile+TILE_O-1, c]
//     // But c varies per thread... so shmem for W doesn't help directly.
//     // Instead, use shmem to cache dY tiles: for a given n, l, load dY[n, o_tile:o_tile+TILE_O, l]
//     // But n and l vary per thread too.
//     //
//     // Better: just rely on L1 cache and __ldg. The simple kernel with __ldg
//     // is already quite good. Let's use a different optimization:
//     // Process multiple (n,c,l) per block but share the LUT row loading.

//     // Actually, the simplest high-perf approach: just use the straightforward
//     // kernel with __ldg and let L1 cache handle locality.
//     if (idx >= total) return;

//     int dy_base = n_idx * O * L + l_idx;

//     for (int o = 0; o < O; ++o) {
//         float dy_val = __ldg(&dY[dy_base + o * L]);
//         uint8_t w_val = __ldg(&W[o * CKK + c_idx]);
//         float lut_val = __ldg(&dx_lut[lut_row + (int)w_val]);
//         sum += dy_val * lut_val;
//     }

//     grad_X[idx] = sum;
// }

// __global__ void bgemm_custom_grad_dw_uint8_kernel(
//     const uint8_t* __restrict__ X,      // (N, CKK, L)
//     const uint8_t* __restrict__ W,      // (O, CKK)
//     const float*   __restrict__ dY,     // (N, O, L)
//     const float*   __restrict__ dw_lut, // (65536,)
//     float*         __restrict__ grad_W,  // (O, CKK)
//     int N, int CKK, int O, int L
// ) {
//     // blockIdx.x: which (o,c) element
//     // blockIdx.y: which chunk of NL
//     int oc_idx = blockIdx.x;
//     if (oc_idx >= O * CKK) return;

//     int c = oc_idx % CKK;
//     int o = oc_idx / CKK;

//     uint8_t w_val = W[oc_idx];
//     int lut_col = (int)w_val;

//     int NL = N * L;
//     int nl_start = blockIdx.y * GRAD_W_NL_TILE;

//     float local_sum = 0.0f;

//     for (int i = threadIdx.x; i < GRAD_W_NL_TILE && (nl_start + i) < NL; i += blockDim.x) {
//         int nl = nl_start + i;
//         int n = nl / L;
//         int l = nl % L;

//         uint8_t x_val = __ldg(&X[(n * CKK + c) * L + l]);
//         float dy_val = __ldg(&dY[(n * O + o) * L + l]);
//         float lut_val = __ldg(&dw_lut[(int)x_val * 256 + lut_col]);
//         local_sum += dy_val * lut_val;
//     }

//     // Block-level reduction
//     __shared__ float sdata[GRAD_W_BLOCK];
//     sdata[threadIdx.x] = local_sum;
//     __syncthreads();

//     for (int s = blockDim.x / 2; s > 0; s >>= 1) {
//         if (threadIdx.x < s) {
//             sdata[threadIdx.x] += sdata[threadIdx.x + s];
//         }
//         __syncthreads();
//     }

//     if (threadIdx.x == 0) {
//         atomicAdd(&grad_W[oc_idx], sdata[0]);
//     }
// }

// // version 2 
// std::tuple<torch::Tensor, torch::Tensor>
// bgemm_custom_grad_uint8(
//     torch::Tensor X,       // (N, CKK, L) uint8
//     torch::Tensor W,       // (O, CKK) uint8
//     torch::Tensor dY,      // (N, O, L) float32
//     torch::Tensor dx_lut,  // (65536,) float32
//     torch::Tensor dw_lut   // (65536,) float32
// ) {
//     TORCH_CHECK(X.is_cuda() && W.is_cuda() && dY.is_cuda());
//     TORCH_CHECK(X.dtype() == torch::kUInt8);
//     TORCH_CHECK(W.dtype() == torch::kUInt8);
//     TORCH_CHECK(dY.dtype() == torch::kFloat32);
//     TORCH_CHECK(dx_lut.dtype() == torch::kFloat32);
//     TORCH_CHECK(dw_lut.dtype() == torch::kFloat32);

//     // Ensure contiguous
//     X = X.contiguous();
//     W = W.contiguous();
//     dY = dY.contiguous();
//     dx_lut = dx_lut.contiguous().view({-1});  // flatten to (65536,)
//     dw_lut = dw_lut.contiguous().view({-1});

//     int N   = X.size(0);
//     int CKK = X.size(1);
//     int L   = X.size(2);
//     int O   = W.size(0);

//     TORCH_CHECK(W.size(1) == CKK);
//     TORCH_CHECK(dY.size(0) == N && dY.size(1) == O && dY.size(2) == L);
//     TORCH_CHECK(dx_lut.numel() == 65536);
//     TORCH_CHECK(dw_lut.numel() == 65536);

//     auto grad_X = torch::empty_like(X, torch::dtype(torch::kFloat32).device(X.device()));
//     // Must zero grad_W because we use atomicAdd
//     auto grad_W = torch::zeros({O, CKK}, torch::dtype(torch::kFloat32).device(W.device()));

//     // ---- Launch grad_X kernel ----
//     {
//         int total = N * CKK * L;
//         int threads = 256;
//         int blocks = (total + threads - 1) / threads;

//         bgemm_custom_grad_dx_uint8_kernel<<<blocks, threads>>>(
//             X.data_ptr<uint8_t>(),
//             W.data_ptr<uint8_t>(),
//             dY.data_ptr<float>(),
//             dx_lut.data_ptr<float>(),
//             grad_X.data_ptr<float>(),
//             N, CKK, O, L
//         );
//     }

//     // ---- Launch grad_W kernel (parallel reduction version) ----
//     {
//         int NL = N * L;
//         int grid_y = (NL + GRAD_W_NL_TILE - 1) / GRAD_W_NL_TILE;
//         int grid_x = O * CKK;
//         dim3 grid(grid_x, grid_y);
//         dim3 block(GRAD_W_BLOCK);

//         bgemm_custom_grad_dw_uint8_kernel<<<grid, block>>>(
//             X.data_ptr<uint8_t>(),
//             W.data_ptr<uint8_t>(),
//             dY.data_ptr<float>(),
//             dw_lut.data_ptr<float>(),
//             grad_W.data_ptr<float>(),
//             N, CKK, O, L
//         );
//     }

//     return std::make_tuple(grad_X, grad_W);
// }





TORCH_LIBRARY_FRAGMENT(approxtorch, m){
    // m.def("im2col_uint8(Tensor input, int k_h, int k_w, int stride_h, int stride_w, int padding_h, int padding_w, int dilation_h, int dilation_w) -> Tensor");
    m.def("bgemm_custom_grad_uint8(Tensor X, Tensor W, Tensor dY, Tensor dx_lut, Tensor dw_lut) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(approxtorch, CUDA, m){
    m.impl("bgemm_custom_grad_uint8", &bgemm_custom_grad_uint8);
}

}