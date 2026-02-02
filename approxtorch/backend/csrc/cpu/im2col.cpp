#include <torch/extension.h>

// 计算输出的空间维度 L = output_h * output_w
inline int64_t compute_output_size(int64_t input_size, int64_t kernel_size, 
                                    int64_t stride, int64_t padding, int64_t dilation) {
    return (input_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
}

torch::Tensor im2col_cpu(
    const torch::Tensor& input,      // (N, C, H, W)
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t stride_h,
    int64_t stride_w,
    int64_t padding_h,
    int64_t padding_w,
    int64_t dilation_h,
    int64_t dilation_w
) {
    // 检查输入
    TORCH_CHECK(input.dim() == 4, "Input must be 4D tensor (N, C, H, W)");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    
    const int64_t N = input.size(0);
    const int64_t C = input.size(1);
    const int64_t H = input.size(2);
    const int64_t W = input.size(3);
    
    // 计算输出尺寸
    const int64_t out_h = compute_output_size(H, kernel_h, stride_h, padding_h, dilation_h);
    const int64_t out_w = compute_output_size(W, kernel_w, stride_w, padding_w, dilation_w);
    const int64_t L = out_h * out_w;
    const int64_t kernel_size = kernel_h * kernel_w;
    const int64_t col_channels = C * kernel_size;  // C × ∏(kernel_size)
    
    TORCH_CHECK(out_h > 0 && out_w > 0, 
                "Output size is invalid. Check kernel/stride/padding/dilation parameters.");
    
    // 创建输出张量 (N, C*kernel_h*kernel_w, L)
    auto output = torch::zeros({N, col_channels, L}, input.options());
    
    // 使用 AT_DISPATCH 支持多种数据类型
    AT_DISPATCH_ALL_TYPES(input.scalar_type(), "im2col_cpu", [&] {
        const scalar_t* input_data = input.data_ptr<scalar_t>();
        scalar_t* output_data = output.data_ptr<scalar_t>();
        
        // 遍历 batch
        for (int64_t n = 0; n < N; ++n) {
            const scalar_t* input_n = input_data + n * C * H * W;
            scalar_t* output_n = output_data + n * col_channels * L;
            
            // 遍历输出位置
            for (int64_t oh = 0; oh < out_h; ++oh) {
                for (int64_t ow = 0; ow < out_w; ++ow) {
                    const int64_t l_idx = oh * out_w + ow;  // 在 L 维度上的索引
                    
                    // 遍历通道
                    for (int64_t c = 0; c < C; ++c) {
                        const scalar_t* input_c = input_n + c * H * W;
                        
                        // 遍历 kernel
                        for (int64_t kh = 0; kh < kernel_h; ++kh) {
                            for (int64_t kw = 0; kw < kernel_w; ++kw) {
                                // 计算输入位置 (考虑 padding 和 dilation)
                                const int64_t ih = oh * stride_h - padding_h + kh * dilation_h;
                                const int64_t iw = ow * stride_w - padding_w + kw * dilation_w;
                                
                                // 计算输出通道索引: c * kernel_size + kh * kernel_w + kw
                                const int64_t col_c = c * kernel_size + kh * kernel_w + kw;
                                
                                // 边界检查
                                scalar_t val = 0;
                                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                    val = input_c[ih * W + iw];
                                }
                                
                                output_n[col_c * L + l_idx] = val;
                            }
                        }
                    }
                }
            }
        }
    });
    
    return output;
}

// 绑定到 Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("im2col", &im2col_cpu, "Im2Col CPU implementation",
          py::arg("input"),
          py::arg("kernel_h"),
          py::arg("kernel_w"),
          py::arg("stride_h"),
          py::arg("stride_w"),
          py::arg("padding_h"),
          py::arg("padding_w"),
          py::arg("dilation_h"),
          py::arg("dilation_w"));
}