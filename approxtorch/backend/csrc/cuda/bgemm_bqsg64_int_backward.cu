// bgemm_bqsg_backward.cu
//
// BQSG backward for BGEMM:
//   grad_output: [N, O, L], float32 CUDA
//   x:           [N, K, L], float32 CUDA, values are integer-like int8 in [-128, 127]
//   w:           [O, K],    float32 CUDA, values are integer-like int8 in [-128, 127]
//   coeff:       [16, 5],   float32 CPU or CUDA
//
// Each 64x64 block stores 5 surface coefficients:
//   coeff[bid] = [p10, p01, p20, p02, p11]
//
// Original fitted surface:
//   f(qx, qw) = p00 + p01*qw + p10*qx + p02*qw^2 + p20*qx^2 + p11*qx*qw
//
// Backward only needs:
//   d f / d x = p10 + 2*p20*qx + p11*qw
//   d f / d w = p01 + 2*p02*qw + p11*qx

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <tuple>
#include <limits>

// 16 blocks * 5 BQSG coefficients = 80 floats.
__constant__ float c_bqsg_coeff[16 * 5];

constexpr int kCoeffPerBlock = 5;
constexpr int kCoeffNumel = 16 * kCoeffPerBlock;
constexpr int kWarpSize = 32;

__device__ __forceinline__ int clamp_int(int v, int lo, int hi) {
    return max(lo, min(hi, v));
}

// q is stored as float, but represents integer-like int8 value in [-128, 127].
// 64-wide partition:
//   [-128, -65] -> 0
//   [ -64,  -1] -> 1
//   [   0,  63] -> 2
//   [  64, 127] -> 3
__device__ __forceinline__ int q_to_block_id_1d(float q) {
    int qi = __float2int_rn(q) + 128;  // 0 ~ 255 ideally
    qi = clamp_int(qi, 0, 255);
    return qi >> 6;                    // 0 ~ 3
}

// If you later want continuous float block assignment instead of rounding first,
// replace q_to_block_id_1d with this version:
// __device__ __forceinline__ int q_to_block_id_1d(float q) {
//     int b = __float2int_rd((q + 128.0f) * 0.015625f);  // floor((q+128)/64)
//     return clamp_int(b, 0, 3);
// }

__device__ __forceinline__ int q_pair_to_block_id(float qx, float qw) {
    int bx = q_to_block_id_1d(qx);
    int bw = q_to_block_id_1d(qw);
    return (bx << 2) | bw;             // bx * 4 + bw, 0 ~ 15
}

__device__ __forceinline__ float bqsg_dfdx(float qx, float qw) {
    int bid = q_pair_to_block_id(qx, qw);
    const float* c = c_bqsg_coeff + bid * kCoeffPerBlock;

    float p10 = c[0];
    float p20 = c[2];
    float p11 = c[4];

    // d f / d x = p10 + 2*p20*qx + p11*qw
    return fmaf(2.0f * p20, qx, fmaf(p11, qw, p10));
}

__device__ __forceinline__ float bqsg_dfdw(float qx, float qw) {
    int bid = q_pair_to_block_id(qx, qw);
    const float* c = c_bqsg_coeff + bid * kCoeffPerBlock;

    float p01 = c[1];
    float p02 = c[3];
    float p11 = c[4];

    // d f / d w = p01 + 2*p02*qw + p11*qx
    return fmaf(2.0f * p02, qw, fmaf(p11, qx, p01));
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float block_reduce_sum(float val) {
    // Supports up to 1024 threads/block.
    __shared__ float shared[32];

    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;

    val = warp_reduce_sum(val);

    if (lane == 0) {
        shared[wid] = val;
    }

    __syncthreads();

    int num_warps = (blockDim.x + 31) >> 5;
    val = (threadIdx.x < num_warps) ? shared[lane] : 0.0f;

    if (wid == 0) {
        val = warp_reduce_sum(val);
    }

    return val;
}

// One thread computes one grad_x[n, k, l].
// grad_x[n,k,l] = sum_o grad_output[n,o,l] * d f(x[n,k,l], w[o,k]) / d x
__global__ void bqsg64_backward_x_kernel(
    const float* __restrict__ grad_output,  // [N, O, L]
    const float* __restrict__ x,            // [N, K, L]
    const float* __restrict__ w,            // [O, K]
    float* __restrict__ grad_x,             // [N, K, L]
    int N,
    int O,
    int K,
    int L
) {
    int l = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.z;

    if (n >= N || k >= K || l >= L) {
        return;
    }

    float qx = x[(n * K + k) * L + l];
    float acc = 0.0f;

    for (int o = 0; o < O; ++o) {
        float go = grad_output[(n * O + o) * L + l];
        float qw = w[o * K + k];
        float local_grad = bqsg_dfdx(qx, qw);
        acc = fmaf(go, local_grad, acc);
    }

    grad_x[(n * K + k) * L + l] = acc;
}

// One CUDA block computes one grad_w[o, k].
// grad_w[o,k] = sum_{n,l} grad_output[n,o,l] * d f(x[n,k,l], w[o,k]) / d w
__global__ void bqsg64_backward_w_kernel(
    const float* __restrict__ grad_output,  // [N, O, L]
    const float* __restrict__ x,            // [N, K, L]
    const float* __restrict__ w,            // [O, K]
    float* __restrict__ grad_w,             // [O, K]
    int N,
    int O,
    int K,
    int L
) {
    int k = blockIdx.x;
    int o = blockIdx.y;
    int tid = threadIdx.x;

    if (k >= K || o >= O) {
        return;
    }

    float qw = w[o * K + k];
    int total = N * L;

    float sum = 0.0f;

    for (int idx = tid; idx < total; idx += blockDim.x) {
        int n = idx / L;
        int l = idx - n * L;

        float go = grad_output[(n * O + o) * L + l];
        float qx = x[(n * K + k) * L + l];
        float local_grad = bqsg_dfdw(qx, qw);
        sum = fmaf(go, local_grad, sum);
    }

    sum = block_reduce_sum(sum);

    if (tid == 0) {
        grad_w[o * K + k] = sum;
    }
}

void check_bqsg_inputs(
    const torch::Tensor& grad_output, // [N, O, L]
    const torch::Tensor& x,           // [N, K, L]
    const torch::Tensor& w,           // [O, K]
    const torch::Tensor& coeff        // [16, 5]
) {
    TORCH_CHECK(grad_output.is_cuda(), "grad_output must be a CUDA tensor");
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(w.is_cuda(), "w must be a CUDA tensor");

    TORCH_CHECK(grad_output.scalar_type() == torch::kFloat32, "grad_output must be float32");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(w.scalar_type() == torch::kFloat32, "w must be float32");
    TORCH_CHECK(coeff.scalar_type() == torch::kFloat32, "coeff must be float32");

    TORCH_CHECK(grad_output.dim() == 3, "grad_output must have shape [N, O, L]");
    TORCH_CHECK(x.dim() == 3, "x must have shape [N, K, L]");
    TORCH_CHECK(w.dim() == 2, "w must have shape [O, K]");

    TORCH_CHECK(coeff.numel() == kCoeffNumel,
                "coeff must have 80 elements, usually shape [16, 5], ordered as [p10, p01, p20, p02, p11]");

    int64_t N = grad_output.size(0);
    int64_t O = grad_output.size(1);
    int64_t L = grad_output.size(2);
    int64_t K = x.size(1);

    TORCH_CHECK(x.size(0) == N, "x.size(0) must equal grad_output.size(0)");
    TORCH_CHECK(x.size(2) == L, "x.size(2) must equal grad_output.size(2)");
    TORCH_CHECK(w.size(0) == O, "w.size(0) must equal grad_output.size(1)");
    TORCH_CHECK(w.size(1) == K, "w.size(1) must equal x.size(1)");

    TORCH_CHECK(N <= std::numeric_limits<int>::max(), "N is too large for int indexing");
    TORCH_CHECK(O <= std::numeric_limits<int>::max(), "O is too large for int indexing");
    TORCH_CHECK(K <= std::numeric_limits<int>::max(), "K is too large for int indexing");
    TORCH_CHECK(L <= std::numeric_limits<int>::max(), "L is too large for int indexing");
}

void copy_coeff_to_constant(const torch::Tensor& coeff) {
    auto coeff_contig = coeff.contiguous().view({kCoeffNumel});

    if (coeff_contig.is_cuda()) {
        cudaMemcpyToSymbol(
            c_bqsg_coeff,
            coeff_contig.data_ptr<float>(),
            kCoeffNumel * sizeof(float),
            0,
            cudaMemcpyDeviceToDevice
        );
    } else {
        cudaMemcpyToSymbol(
            c_bqsg_coeff,
            coeff_contig.data_ptr<float>(),
            kCoeffNumel * sizeof(float),
            0,
            cudaMemcpyHostToDevice
        );
    }

    C10_CUDA_CHECK(cudaGetLastError());
}

std::tuple<torch::Tensor, torch::Tensor> bgemm_bqsg64_backward_cuda(
    torch::Tensor& grad_output,  // [N, O, L]
    torch::Tensor& x,            // [N, K, L]
    torch::Tensor& w,            // [O, K]
    torch::Tensor& coeff       // [16, 5], ordered as [p10, p01, p20, p02, p11]
) {
    check_bqsg_inputs(grad_output, x, w, coeff);

    const c10::cuda::CUDAGuard device_guard(grad_output.device());

    grad_output = grad_output.contiguous();
    x = x.contiguous();
    w = w.contiguous();

    int N = static_cast<int>(grad_output.size(0));
    int O = static_cast<int>(grad_output.size(1));
    int L = static_cast<int>(grad_output.size(2));
    int K = static_cast<int>(x.size(1));

    copy_coeff_to_constant(coeff);

    auto grad_x = torch::empty_like(x);
    auto grad_w = torch::empty_like(w);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    {
        // grad_x: one thread computes one [n,k,l].
        // 32 along L gives coalesced x/grad_x writes for neighboring l.
        // 8 along K gives 256 threads/block.
        dim3 block_x(32, 8, 1);
        dim3 grid_x(
            (L + block_x.x - 1) / block_x.x,
            (K + block_x.y - 1) / block_x.y,
            N
        );

        bqsg64_backward_x_kernel<<<grid_x, block_x, 0, stream>>>(
            grad_output.data_ptr<float>(),
            x.data_ptr<float>(),
            w.data_ptr<float>(),
            grad_x.data_ptr<float>(),
            N,
            O,
            K,
            L
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    {
        // grad_w: one block computes one [o,k], reducing over N*L.
        constexpr int threads_w = 256;
        dim3 block_w(threads_w, 1, 1);
        dim3 grid_w(K, O, 1);


        bqsg64_backward_w_kernel<<<grid_w, block_w, 0, stream>>>(
            grad_output.data_ptr<float>(),
            x.data_ptr<float>(),
            w.data_ptr<float>(),
            grad_w.data_ptr<float>(),
            N,
            O,
            K,
            L
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    return std::make_tuple(grad_x, grad_w);
}

// Optional helper: copy BQSG coefficients to constant memory once.
// If you call this from Python before training and then write another wrapper that does not copy
// coeff each backward, you can remove copy overhead entirely.
void set_bqsg_coeff_cuda(torch::Tensor coeff) {
    TORCH_CHECK(coeff.scalar_type() == torch::kFloat32, "coeff must be float32");
    TORCH_CHECK(coeff.numel() == kCoeffNumel,
                "coeff must have 80 elements, usually shape [16, 5], ordered as [p10, p01, p20, p02, p11]");

    if (coeff.is_cuda()) {
        const c10::cuda::CUDAGuard device_guard(coeff.device());
        copy_coeff_to_constant(coeff);
    } else {
        copy_coeff_to_constant(coeff);
    }
}

TORCH_LIBRARY_FRAGMENT(approxtorch, m) {
    m.def("bgemm_bqsg64_backward(Tensor grad_output, Tensor x, Tensor w, Tensor coeff) -> (Tensor grad_x, Tensor grad_w)");
    m.def("set_bqsg_coeff(Tensor coeff) -> ()");
}

TORCH_LIBRARY_IMPL(approxtorch, CUDA, m) {
    m.impl("bgemm_bqsg64_backward", &bgemm_bqsg64_backward_cuda);
    m.impl("set_bqsg_coeff", &set_bqsg_coeff_cuda);
}
