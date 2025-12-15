#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

namespace approxtorch {

template <typename T>
__global__ void im2col_gpu_kernel_nchw_optimized(
    const int64_t n, 
    const T* __restrict__ data_im, // input feature [N, C, H, W]
    T* __restrict__ data_col,   // output feature [N, C x Kh x Kw, H_out x W_out]
    const int64_t height, const int64_t width, 
    const int64_t kernel_h, const int64_t kernel_w,
    const int64_t pad_h, const int64_t pad_w,
    const int64_t stride_h, const int64_t stride_w,
    const int64_t dilation_h, const int64_t dilation_w,
    const int64_t height_col, const int64_t width_col
) {
    // index 范围: [0, N * C * Kh * Kw * H_out * W_out)
    int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n) return;

    // 假设 data_col 的布局是 (N, C, Kh, Kw, H_out, W_out) 的扁平化
    // 这种布局使得同一个 kernel patch 的数据不连续，不利于 GEMM。
    // 标准 GEMM 需要的布局通常是：
    // Matrix Rows: (C * Kh * Kw)
    // Matrix Cols: (N * H_out * W_out)  <-- 也就是 spatial_size
    // 
    // 为了写得快，我们让 threads 连续对应 data_col 的线性索引。
    // data_col 索引 i 对应:
    // row = i / (N * H_out * W_out)
    // col = i % (N * H_out * W_out)
    // 
    // 但计算 row 和 col 需要除法。
    
    // 优化技巧：利用 Caffe 原始实现的变体，将每个 pixel 的拷贝独立出来
    // 这种方法虽然看起来计算量大，但对于 uint8 这种小数据，能最大化合并写。

    // 计算 output 里的坐标
    // 这种写法是为了适配 data_col 形状: (C * Kh * Kw) * (N * H_out * W_out)
    // 这是一个巨大的 2D 矩阵。
    // index 是线性偏移。
    
    int64_t spatial_size = height_col * width_col; // H_out * W_out
    int64_t batch_spatial_size = n * spatial_size; // 如果把 N 融合进 spatial
    
    // 这里我们假设 N=1 的情况，或者 N 在外部作为 offset。
    // 为了代码清晰，这里只写 Single Batch (N=1) 的逻辑，多 Batch 建议用 Block.z 控制 batch index。
    
    int64_t w_out = index % width_col;
    int64_t h_index = index / width_col;
    int64_t h_out = h_index % height_col;
    int64_t channel_in = h_index / height_col;
    
    int64_t channel = channel_in / (kernel_h * kernel_w);
    int64_t kernel_idx = channel_in % (kernel_h * kernel_w);
    int64_t k_h_idx = kernel_idx / kernel_w;
    int64_t k_w_idx = kernel_idx % kernel_w;

    int64_t h_in = h_out * stride_h - pad_h + k_h_idx * dilation_h;
    int64_t w_in = w_out * stride_w - pad_w + k_w_idx * dilation_w;

    T val = 0;
    if (h_in >= 0 && w_in >= 0 && h_in < height && w_in < width) {
        val = __ldg(data_im + (channel * height + h_in) * width + w_in);
    }
    data_col[index] = val;
}




// 辅助函数：计算输出尺寸
__host__ __device__ int64_t get_out_dim(int64_t input_dim, int64_t padding, int64_t kernel, int64_t stride, int64_t dilation) {
    return (input_dim + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1;
}

torch::Tensor im2col_uint8(
    const torch::Tensor& feature, 
    int64_t k_h, int64_t k_w,
    int64_t pad_h, int64_t pad_w,
    int64_t stride_h, int64_t stride_w,
    int64_t dil_h, int64_t dil_w
) {
    int64_t N = feature.size(0);
    int64_t C = feature.size(1);
    int64_t H = feature.size(2);
    int64_t W = feature.size(3);
    int64_t h_col = get_out_dim(H, pad_h, k_h, stride_h, dil_h);
    int64_t w_col = get_out_dim(W, pad_w, k_w, stride_w, dil_w);
    
    // 这里的并行策略：把 Batch N 视为独立的任务，或者展平。
    // 为了简单且高效，我们在 kernel 里不处理 N，而是在 Host 端循环 N 或者 launch 一个巨大的 grid。
    // 如果显存够大，一次 launch 所有 N 是最好的。
    
    int64_t total_elements = N * C * k_h * k_w * h_col * w_col; 
    torch::Tensor feature_col = torch::empty({N, C * k_h * k_w, h_col*w_col}, 
        feature.options());
    
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    im2col_gpu_kernel_nchw_optimized<uint8_t>
    <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>
    (total_elements, feature.data_ptr<uint8_t>(), feature_col.data_ptr<uint8_t>(),
     H, W, k_h, k_w, pad_h, pad_w, stride_h, stride_w, dil_h, dil_w, h_col, w_col);

    return feature_col;
}
// ... Binding logic ...
TORCH_LIBRARY_FRAGMENT(approxtorch, m){
    m.def("im2col_uint8(Tensor feature, int k_h, int k_w, int pad_h, int pad_w, int stride_h, int stride_w, int dil_h, int dil_w) -> Tensor");
}
TORCH_LIBRARY_IMPL(approxtorch, CUDA, m){
    m.impl("im2col_uint8", &im2col_uint8);
}

} // namespace end 