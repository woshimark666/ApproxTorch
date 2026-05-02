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

// tile 参数
// BM: 一次算多少个 M = N * L 的 row
// BN: 一次算多少个 O
// BK: CKK 方向 tile

template<int BM, int BN, int BK>
__global__ void bgemm_lut_fake_int8_forward_kernel(
    const float* __restrict__ x,       // [N, K, L]
    const float* __restrict__ wt,      // [K, O], w 的转置
    const float* __restrict__ lut,     // [65536]
    float* __restrict__ y,             // [N, O, L]
    int N,
    int K,
    int L,
    int O
) {
    // blockIdx.y -> M tile, M = N * L
    // blockIdx.x -> O tile
    const int tx = threadIdx.x;  // O direction
    const int ty = threadIdx.y;  // M direction

    const int row = blockIdx.y * BM + ty;  // row in M = N * L
    const int col = blockIdx.x * BN + tx;  // output channel O

    // shared memory 存 uint16_t 的量化值 index，即 q + 128 后的 0..255
    __shared__ unsigned short sx[BM][BK + 1];
    __shared__ unsigned short sw[BK][BN + 1];

    float acc = 0.0f;

    const int tid = ty * BN + tx;
    constexpr int NUM_THREADS = BM * BN;

    for (int k0 = 0; k0 < K; k0 += BK) {
        // load x tile: [BM, BK]
        for (int idx = tid; idx < BM * BK; idx += NUM_THREADS) {
            const int r = idx / BK;
            const int kk = idx - r * BK;

            const int global_row = blockIdx.y * BM + r;
            const int global_k = k0 + kk;

            unsigned short q = 0;

            if (global_row < N * L && global_k < K) {
                const int n = global_row / L;
                const int l = global_row - n * L;

                // x shape: [N, K, L]
                float xv = __ldg(x + ((n * K + global_k) * L + l));

                // x 里面存的是 -128 到 127 的整数 float
                int xi = __float2int_rn(xv) + 128;

                // 如果你能保证 x/w 一定在 [-128,127]，可以去掉这两行 clamp
                xi = max(0, min(255, xi));

                q = static_cast<unsigned short>(xi);
            }

            sx[r][kk] = q;
        }

        // load w tile，使用 wt: [K, O]
        // 这样 O 方向连续，读取比原始 [O, K] 更 coalesced
        for (int idx = tid; idx < BK * BN; idx += NUM_THREADS) {
            const int kk = idx / BN;
            const int c = idx - kk * BN;

            const int global_k = k0 + kk;
            const int global_col = blockIdx.x * BN + c;

            unsigned short q = 0;

            if (global_k < K && global_col < O) {
                float wv = __ldg(wt + global_k * O + global_col);

                int wi = __float2int_rn(wv) + 128;

                // 如果你能保证 x/w 一定在 [-128,127]，可以去掉这两行 clamp
                wi = max(0, min(255, wi));

                q = static_cast<unsigned short>(wi);
            }

            sw[kk][c] = q;
        }

        __syncthreads();

        if (row < N * L && col < O) {
#pragma unroll
            for (int kk = 0; kk < BK; ++kk) {
                if (k0 + kk < K) {
                    const int xi = static_cast<int>(sx[ty][kk]);
                    const int wi = static_cast<int>(sw[kk][tx]);

                    // LUT[(x + 128) * 256 + (w + 128)]
                    const int lut_idx = (xi << 8) | wi;

                    acc += __ldg(lut + lut_idx);
                }
            }
        }

        __syncthreads();
    }

    if (row < N * L && col < O) {
        const int n = row / L;
        const int l = row - n * L;

        // y shape: [N, O, L]
        y[(n * O + col) * L + l] = acc;
    }
}

torch::Tensor bgemm_lut_forward_cuda(
    const torch::Tensor& x,
    const torch::Tensor& w,
    const torch::Tensor& lut
) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));

    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(w.is_cuda(), "w must be a CUDA tensor");
    TORCH_CHECK(lut.is_cuda(), "lut must be a CUDA tensor");

    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(w.scalar_type() == torch::kFloat32, "w must be float32");
    TORCH_CHECK(lut.scalar_type() == torch::kFloat32, "lut must be float32");

    TORCH_CHECK(x.dim() == 3, "x must have shape [N, CKK, L]");
    TORCH_CHECK(w.dim() == 2, "w must have shape [O, CKK]");
    TORCH_CHECK(lut.dim() == 1, "lut must have shape [256 * 256]");
    TORCH_CHECK(lut.numel() == 256 * 256, "lut must have 65536 elements");

    TORCH_CHECK(x.size(1) == w.size(1), "x.size(1) must equal w.size(1)");
    
    const int N = static_cast<int>(x.size(0));
    const int K = static_cast<int>(x.size(1));
    const int L = static_cast<int>(x.size(2));
    const int O = static_cast<int>(w.size(0));

    auto y = torch::empty({N, O, L}, x.options());

    // 为了让 w 读取 coalesced，把 [O, K] 转成 [K, O]
    // 如果 w 在多次调用中不变，可以在外面提前转置，进一步减少开销
    auto wt = w.t().contiguous();

    constexpr int BM = 16;
    constexpr int BN = 16;
    constexpr int BK = 32;

    dim3 block(BN, BM);
    dim3 grid(
        (O + BN - 1) / BN,
        (N * L + BM - 1) / BM
    );

    bgemm_lut_fake_int8_forward_kernel<BM, BN, BK>
    <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(),
        wt.data_ptr<float>(),
        lut.data_ptr<float>(),
        y.data_ptr<float>(),
        N,
        K,
        L,
        O
    );

    CHECK_CUDA_ERROR();

    return y;
}


TORCH_LIBRARY_FRAGMENT(approxtorch, m){
    m.def("bgemm_fake_int8_forward_cuda(Tensor x, Tensor w, Tensor lut) -> Tensor");
}

TORCH_LIBRARY_IMPL(approxtorch, CUDA, m){
    m.impl("bgemm_fake_int8_forward_cuda", &bgemm_lut_forward_cuda);
}

