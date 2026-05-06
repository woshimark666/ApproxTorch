#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == torch::kFloat32, #x " must be float32")

constexpr int QMIN = -128;
constexpr int QMAX = 127;

__device__ __forceinline__ int qfloat_to_idx(float v) {
    int q = __float2int_rn(v);
    q = max(QMIN, min(QMAX, q));
    return q + 128;
}


/**
 * grad_x[n, k, l] =
 *     sum_o grad_output[n, o, l] * dx[w[k, o] + 128]
 *
 * grad_output_nlo: [N, L, O], contiguous
 * x layout:         [N, K, L]
 * w layout:         [K, O]
 * grad_x layout:    [N, K, L]
 */
template<int BP, int BK, int TO>
__global__ void bgemm_lre_grad_x_kernel_shared_lut(
    const float* __restrict__ grad_output_nlo,
    const float* __restrict__ w,
    const float* __restrict__ dx,
    float* __restrict__ grad_x,
    int N,
    int K,
    int L,
    int O
) {
    const int P = N * L;

    // blockIdx.x: p = n * L + l
    // blockIdx.y: k
    const int p0 = blockIdx.x * BP;
    const int k0 = blockIdx.y * BK;

    // threadIdx.x 对应 p，方便 grad_x 写回时尽量沿 L 连续
    const int tx = threadIdx.x;  // local p
    const int ty = threadIdx.y;  // local k

    const int p = p0 + tx;
    const int k = k0 + ty;

    const int tid = ty * blockDim.x + tx;
    constexpr int BLOCK_THREADS = BP * BK;

    __shared__ float sh_dx[256];

    // grad_output tile: [BP, TO]
    __shared__ float sh_go[BP][TO + 1];

    // dx(w) tile: [BK, TO]
    __shared__ float sh_dfdx[BK][TO + 1];

    // load LUT dx into shared memory
    for (int i = tid; i < 256; i += BLOCK_THREADS) {
        sh_dx[i] = dx[i];
    }

    __syncthreads();

    float acc = 0.0f;

    for (int o0 = 0; o0 < O; o0 += TO) {
        // load grad_output_nlo[p, o]
        for (int i = tid; i < BP * TO; i += BLOCK_THREADS) {
            int local_p = i / TO;
            int local_o = i % TO;

            int pp = p0 + local_p;
            int oo = o0 + local_o;

            float v = 0.0f;
            if (pp < P && oo < O) {
                v = grad_output_nlo[pp * O + oo];
            }

            sh_go[local_p][local_o] = v;
        }

        // load dx[w[k, o]]
        // w shape: [K, O]
        for (int i = tid; i < BK * TO; i += BLOCK_THREADS) {
            int local_k = i / TO;
            int local_o = i % TO;

            int kk = k0 + local_k;
            int oo = o0 + local_o;

            float v = 0.0f;
            if (kk < K && oo < O) {
                float w_val = w[kk * O + oo];
                int idx = qfloat_to_idx(w_val);
                v = sh_dx[idx];
            }

            sh_dfdx[local_k][local_o] = v;
        }

        __syncthreads();

#pragma unroll
        for (int oo = 0; oo < TO; ++oo) {
            acc += sh_go[tx][oo] * sh_dfdx[ty][oo];
        }

        __syncthreads();
    }

    if (p < P && k < K) {
        int n = p / L;
        int l = p - n * L;

        grad_x[n * K * L + k * L + l] = acc;
    }
}


/**
 * grad_w[k, o] =
 *     sum_{n,l} grad_output[n, o, l] * dw[x[n, k, l] + 128]
 *
 * grad_output_nlo: [N, L, O], contiguous
 * x layout:         [N, K, L]
 * w layout:         [K, O]
 * grad_w layout:    [K, O]
 */
template<int BO, int BK, int TP>
__global__ void bgemm_lre_grad_w_kernel_shared_lut(
    const float* __restrict__ grad_output_nlo,
    const float* __restrict__ x,
    const float* __restrict__ dw,
    float* __restrict__ grad_w,
    int N,
    int K,
    int L,
    int O
) {
    const int P = N * L;

    // blockIdx.x: o
    // blockIdx.y: k
    const int o0 = blockIdx.x * BO;
    const int k0 = blockIdx.y * BK;

    // threadIdx.x 对应 o，方便 grad_w[k, o] 写回连续
    const int tx = threadIdx.x;  // local o
    const int ty = threadIdx.y;  // local k

    const int o = o0 + tx;
    const int k = k0 + ty;

    const int tid = ty * blockDim.x + tx;
    constexpr int BLOCK_THREADS = BO * BK;

    __shared__ float sh_dw[256];

    // grad_output tile: [TP, BO]
    __shared__ float sh_go[TP][BO + 1];

    // dw(x) tile: [BK, TP]
    __shared__ float sh_dfdw[BK][TP + 1];

    // load LUT dw into shared memory
    for (int i = tid; i < 256; i += BLOCK_THREADS) {
        sh_dw[i] = dw[i];
    }

    __syncthreads();

    float acc = 0.0f;

    for (int p0 = 0; p0 < P; p0 += TP) {
        // load grad_output_nlo[p, o]
        for (int i = tid; i < TP * BO; i += BLOCK_THREADS) {
            int local_p = i / BO;
            int local_o = i % BO;

            int pp = p0 + local_p;
            int oo = o0 + local_o;

            float v = 0.0f;
            if (pp < P && oo < O) {
                v = grad_output_nlo[pp * O + oo];
            }

            sh_go[local_p][local_o] = v;
        }

        // load dw[x[n, k, l]]
        for (int i = tid; i < BK * TP; i += BLOCK_THREADS) {
            int local_k = i / TP;
            int local_p = i % TP;

            int kk = k0 + local_k;
            int pp = p0 + local_p;

            float v = 0.0f;
            if (kk < K && pp < P) {
                int n = pp / L;
                int l = pp - n * L;

                float x_val = x[n * K * L + kk * L + l];
                int idx = qfloat_to_idx(x_val);
                v = sh_dw[idx];
            }

            sh_dfdw[local_k][local_p] = v;
        }

        __syncthreads();

#pragma unroll
        for (int pp = 0; pp < TP; ++pp) {
            acc += sh_go[pp][tx] * sh_dfdw[ty][pp];
        }

        __syncthreads();
    }

    if (k < K && o < O) {
        grad_w[k * O + o] = acc;
    }
}


std::tuple<torch::Tensor, torch::Tensor> bgemm_lre_backward(
    const torch::Tensor& grad_output,  // [N, O, L]
    const torch::Tensor& x,            // [N, K, L]
    const torch::Tensor& w,            // [K, O], w 的转置
    const torch::Tensor& dx,           // [256]
    const torch::Tensor& dw,          // [256]
    const torch::Tensor& s_x,      // [1]
    const torch::Tensor& s_w       // [O]
) {
    CHECK_CUDA(grad_output);
    CHECK_CUDA(x);
    CHECK_CUDA(w);
    CHECK_CUDA(dx);
    CHECK_CUDA(dw);

    CHECK_FLOAT(grad_output);
    CHECK_FLOAT(x);
    CHECK_FLOAT(w);
    CHECK_FLOAT(dx);
    CHECK_FLOAT(dw);

    TORCH_CHECK(grad_output.dim() == 3, "grad_output must have shape [N, O, L]");
    TORCH_CHECK(x.dim() == 3, "x must have shape [N, K, L]");
    TORCH_CHECK(w.dim() == 2, "w must have shape [K, O]");
    TORCH_CHECK(dx.numel() == 256, "dx must have 256 elements");
    TORCH_CHECK(dw.numel() == 256, "dw must have 256 elements");

    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));

    grad_output = grad_output.contiguous();
    x = x.contiguous();
    w = w.contiguous();  // 转成 [K, O]，方便按照 k 做 reduction
    dx = dx.contiguous();
    dw = dw.contiguous();

    const int N = x.size(0);
    const int K = x.size(1);
    const int L = x.size(2);

    const int WK = w.size(0);
    const int O = w.size(1);

    TORCH_CHECK(WK == K, "w.shape[0] must equal K");
    TORCH_CHECK(grad_output.size(0) == N, "grad_output.shape[0] must equal N");
    TORCH_CHECK(grad_output.size(1) == O, "grad_output.shape[1] must equal O");
    TORCH_CHECK(grad_output.size(2) == L, "grad_output.shape[2] must equal L");

    auto grad_x = torch::empty_like(x);
    auto grad_w = torch::empty_like(w);

    // grad_output 原始布局是 [N, O, L]
    // 这里转成 [N, L, O]，让 O 维连续，方便按照 o 做 reduction
    auto grad_output_nlo = grad_output.permute({0, 2, 1}).contiguous();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const int P = N * L;

    {
        constexpr int BP = 16;
        constexpr int BK = 16;
        constexpr int TO = 32;

        dim3 block(BP, BK);
        dim3 grid(
            (P + BP - 1) / BP,
            (K + BK - 1) / BK
        );
        
        torch::Tensor grad_output_scaled_w = grad_output_nlo * s_w.view({1, 1, O});
        bgemm_lre_grad_x_kernel_shared_lut<BP, BK, TO>
            <<<grid, block, 0, stream>>>(
                grad_output_scaled_w.data_ptr<float>(),
                w.data_ptr<float>(),
                dx.data_ptr<float>(),
                grad_x.data_ptr<float>(),
                N, K, L, O
            );
    }

    {
        constexpr int BO = 16;
        constexpr int BK = 16;
        constexpr int TP = 32;

        dim3 block(BO, BK);
        dim3 grid(
            (O + BO - 1) / BO,
            (K + BK - 1) / BK
        );
        torch::Tensor grad_output_scaled_x = grad_output_nlo * s_x;
        bgemm_lre_grad_w_kernel_shared_lut<BO, BK, TP>
            <<<grid, block, 0, stream>>>(
                grad_output_scaled_x.data_ptr<float>(),
                x.data_ptr<float>(),
                dw.data_ptr<float>(),
                grad_w.data_ptr<float>(),
                N, K, L, O
            );
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return std::make_tuple(grad_x, grad_w);
}



TORCH_LIBRARY_FRAGMENT(approxtorch, m){
    m.def("bgemm_lre_backward(Tensor grad_output, Tensor x, Tensor w, Tensor dx, Tensor dw, Tensor s_x, Tensor s_w) -> (Tensor grad_x, Tensor grad_w)");
}

TORCH_LIBRARY_IMPL(approxtorch, CUDA, m){
    m.impl("bgemm_lre_backward", &bgemm_lre_backward);
}