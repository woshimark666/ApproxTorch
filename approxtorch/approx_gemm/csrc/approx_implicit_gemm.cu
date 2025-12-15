#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

// #define BM 128       // Block M
// #define BN 128       // Block N
// #define BK 32        // K 维度步长 (FP32时是8，INT8时改为32，增加流水线深度)

// #define WM 64        // Warp M
// #define WN 32        // Warp N

// #define TM 8         // Thread M
// #define TN 8         // Thread N
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

namespace approxtorch{
template 
<const int BM, const int BN, const int BK, 
const int WM, const int WN, 
const int TM, const int TN, const int NUM_THREADS>
__global__ void implicit_gemm_uint8_kernel(
    const uint8_t* __restrict__ input,  // [N, C, H, W]
    const uint8_t* __restrict__ weight, // [K, C, R, S]
    const int32_t* __restrict__ lut,    // LUT [256 * 256], 扁平化存储
    int32_t* __restrict__ output, // [N, K, P, Q] (通常累加用 int32))
    int N, int C, int H, int W,
    int K, int R, int S,
    int P, int Q,
    int pad_h, int pad_w, int stride_h, int stride_w,
    int dilation_h, int dilation_w
) {
    // --------------------------------------------------------
    // 1. 坐标计算
    // --------------------------------------------------------
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int warpId = tid / 32;
    int laneId = tid % 32;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int global_m_offset = by * BM;
    int global_n_offset = bx * BN;

    int M_gemm = N * P * Q;
    int N_gemm = K;
    int K_gemm = C * R * S;

    // Warp 布局 (2x4)
    int warp_row = warpId / 4;
    int warp_col = warpId % 4;
    
    // Thread 在 Warp Tile 中的布局 (8x4)
    int thread_row = laneId / 4;
    int thread_col = laneId % 4;

    int thread_m_base = warp_row * WM + thread_row * TM;
    int thread_n_base = warp_col * WN + thread_col * TN;

    // --------------------------------------------------------
    // 2. Shared Memory (存放 uint8)
    // --------------------------------------------------------
    // 同样大小的 Smem 可以存更多数据，或者保持原样
    __shared__ uint8_t smem_a[BM][BK];
    __shared__ uint8_t smem_b[BK][BN];

    // 寄存器累加器 (int32)
    int32_t accum[TM][TN] = {0};

    // 寄存器片段 (用于计算)
    uint8_t frag_a[TM];
    uint8_t frag_b[TN];

    // --------------------------------------------------------
    // 3. 主循环
    // --------------------------------------------------------
    for (int k_step = 0; k_step < K_gemm; k_step += BK) {

        // ===========================================
        // Phase A: 加载 Input -> Smem A (Implicit Logic)
        // ===========================================
        // 由于 Implicit 寻址很难向量化(非连续)，我们仍然按 byte 处理，
        // 但为了效率，每个线程处理多个 byte。
        // BM*BK = 128*32 = 4096 bytes. 256 threads -> 16 bytes per thread.
        
        #pragma unroll
        for (int i = 0; i < 16; ++i) { // 16 bytes per thread
            int idx = tid + i * NUM_THREADS;
            if (idx < BM * BK) {
                int r = idx / BK;
                int c = idx % BK;
                
                int global_m = global_m_offset + r;
                int global_k = k_step + c;
                
                uint8_t val = 0;
                if (global_m < M_gemm && global_k < K_gemm) {
                    // Implicit Address Calculation
                    int n_idx = global_m / (P * Q);
                    int rem = global_m % (P * Q);
                    int p_idx = rem / Q;
                    int q_idx = rem % Q;
                    
                    int c_idx = global_k / (R * S);
                    rem = global_k % (R * S);
                    int r_idx = rem / S;
                    int s_idx = rem % S;
                    
                    int h_in = p_idx * stride_h - pad_h + r_idx * dilation_h;
                    int w_in = q_idx * stride_w - pad_w + s_idx * dilation_w;
                    
                    if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                        int in_off = ((n_idx * C + c_idx) * H + h_in) * W + w_in;
                        val = input[in_off];
                    }
                }
                smem_a[r][c] = val;
            }
        }

        // ===========================================
        // Phase B: 加载 Weight -> Smem B
        // ===========================================
        // Weight 是连续的，可以使用 int4 向量化加载优化，这里简化逻辑保持一致性
        // BK*BN = 32*128 = 4096 bytes. 256 threads -> 16 bytes per thread.
        
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            int idx = tid + i * NUM_THREADS;
            if (idx < BK * BN) {
                int r = idx / BN;
                int c = idx % BN;
                
                int global_k = k_step + r;
                int global_n = global_n_offset + c;
                
                uint8_t val = 0;
                if (global_k < K_gemm && global_n < N_gemm) {
                    // 假设 Weight 格式: [N(Output), K(Input*R*S)] 转置或直接映射
                    // 这里按标准 Row Major: global_n * K_total + global_k
                    int w_off = global_n * K_gemm + global_k;
                    val = weight[w_off];
                }
                smem_b[r][c] = val;
            }
        }

        __syncthreads();

        // ===========================================
        // Phase C: 计算 (LUT Lookup + Masking)
        // ===========================================
        
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            // 1. Load regs
            #pragma unroll
            for (int m = 0; m < TM; ++m) frag_a[m] = smem_a[thread_m_base + m][k];
            
            #pragma unroll
            for (int n = 0; n < TN; ++n) frag_b[n] = smem_b[k][thread_n_base + n];
            
            // 2. Compute
            #pragma unroll
            for (int m = 0; m < TM; ++m) {
                #pragma unroll
                for (int n = 0; n < TN; ++n) {
                    uint8_t va = frag_a[m];
                    uint8_t vb = frag_b[n];
                    
                    // ------------------------------------------------
                    // 核心逻辑: 数据遮蔽 (Data Masking)
                    // ------------------------------------------------
                    
                    // 方法 1: 分支逻辑 (简单易懂，编译器通常会优化成 CMOV)
                    /*
                    int32_t prod = 0;
                    if (va != 0 && vb != 0) {
                        // 利用 __ldg 强制走 Read-Only Cache (L1/Texture)
                        // 避免普通的 Global Memory 延迟
                        prod = (int32_t)__ldg(&lut[(int)va * 256 + (int)vb]);
                    }
                    accum[m][n] += prod;
                    */
                    
                    // 方法 2: 无分支逻辑 (通常更快)
                    // 计算 LUT 索引
                    int lut_idx = (int)va * 256 + (int)vb;
                    
                    // 查表 (使用 __ldg 极速读取)
                    // 注意：lut 指针必须是全局内存指针
                    int32_t lut_val = (int32_t)__ldg(&lut[lut_idx]);
                    
                    // 制作 Mask: 如果 va 或 vb 为 0，mask 为 0，否则为 1
                    // 这里的 bool 到 int 转换是隐式的
                    int mask = (va != 0) && (vb != 0);
                    
                    accum[m][n] += lut_val * mask;
                }
            }
        }
        __syncthreads();
    }

    // --------------------------------------------------------
    // 4. 写回 Output
    // --------------------------------------------------------
    #pragma unroll
    for (int m = 0; m < TM; ++m) {
        #pragma unroll
        for (int n = 0; n < TN; ++n) {
            int local_m = thread_m_base + m;
            int local_n = thread_n_base + n;
            int global_m = global_m_offset + local_m;
            int global_n = global_n_offset + local_n;

            if (global_m < M_gemm && global_n < N_gemm) {
                // 简单的坐标逆映射
                int n_idx = global_m / (P * Q);
                int rem = global_m % (P * Q);
                int p_idx = rem / Q;
                int q_idx = rem % Q;
                
                int out_idx = ((n_idx * N_gemm + global_n) * P + p_idx) * Q + q_idx;
                output[out_idx] = accum[m][n];
            }
        }
    }
}

torch::Tensor implicit_gemm_uint8(
    const torch::Tensor& feature,
    const torch::Tensor& weight,
    const torch::Tensor& lut,
    int64_t pad_h, int64_t pad_w, 
    int64_t stride_h, int64_t stride_w, 
    int64_t dilation_h, int64_t dilation_w
){

    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 32;
    constexpr int WM = 64;
    constexpr int WN = 32;
    constexpr int TM = 8;
    constexpr int TN = 8;
    constexpr int NUM_THREADS = 256;

    // constexpr int BM = 64;
    // constexpr int BN = 64;
    // constexpr int BK = 64;
    // constexpr int WM = 32;
    // constexpr int WN = 16;
    // constexpr int TM = 4;
    // constexpr int TN = 4;
    // constexpr int NUM_THREADS = 256;


    const at::cuda::OptionalCUDAGuard device_guard(device_of(feature));

    int64_t N = feature.size(0);
    int64_t C = feature.size(1);
    int64_t H = feature.size(2);
    int64_t W = feature.size(3);

    int64_t K = weight.size(0);
    int64_t R = weight.size(2);
    int64_t S = weight.size(3);
    
    int64_t P = (H + 2 * pad_h - dilation_h * (R - 1) - 1) / stride_h + 1;
    int64_t Q = (W + 2 * pad_w - dilation_w * (S - 1) - 1) / stride_w + 1;

    auto tensor_options = torch::TensorOptions().device(feature.device()).dtype(torch::kInt32);
    auto output = torch::empty({N, K, P, Q}, tensor_options);
    
    int64_t M_gemm = N * P * Q;
    int64_t N_gemm = K;
    dim3 grid(CEIL_DIV(N_gemm, BN), CEIL_DIV(M_gemm, BM));
    dim3 block(256);
    implicit_gemm_uint8_kernel<BM, BN, BK, WM, WN, TM, TN, NUM_THREADS>
    <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>
    (feature.data_ptr<uint8_t>(), 
    weight.data_ptr<uint8_t>(), 
    lut.data_ptr<int32_t>(),
    output.data_ptr<int32_t>(),
    N, C, H, W, K, R, S, P, Q, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w
    );
    return output;
}

// ... Binding logic ...
TORCH_LIBRARY_FRAGMENT(approxtorch, m){
    m.def("approx_implicit_gemm_uint8(Tensor feature, Tensor weight, Tensor lut, int pad_h, int pad_w, int stride_h, int stride_w, int dilation_h, int dilation_w) -> Tensor");
}
TORCH_LIBRARY_IMPL(approxtorch, CUDA, m){
    m.impl("approx_implicit_gemm_uint8", &implicit_gemm_uint8);
}


} // namespace approx_gemm