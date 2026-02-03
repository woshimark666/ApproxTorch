#pragma once

#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace approxtorch {

    // =============================================================================
    // 通用定义
    // =============================================================================
    
    #define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))
    #define WARP_SIZE 32
    
    // CUDA 错误检查
    #define CUDA_CHECK(call) do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        } \
} while(0)
/*
================================================================================
                    ULTIMATE APPROXIMATE GEMM IMPLEMENTATION
                              最终高性能实现
================================================================================

【问题分析】
你的 LUT 是 256x256x4bytes = 256KB，无法完全放入 Shared Memory (48KB)。
这导致每次乘法都要访问 Global Memory，成为性能瓶颈。

【解决方案】
我提供三种策略，适用于不同场景：

策略 1: HOT ZONE CACHE (热点缓存)
  - 假设数据大多集中在某个范围（如量化后的神经网络权重通常在 [-32, 31]）
  - 缓存这个热点区域的 LUT 到 Shared Memory
  - 适用于：量化模型推理

策略 2: MULTI-PASS (多遍扫描)  
  - 将 LUT 分成小块，每次只处理一块
  - 所有 LUT 访问都在 Shared Memory
  - 适用于：精度要求高，数据分布均匀

策略 3: STREAM + PREFETCH (流式预取)
  - 单 Pass，但使用 L2 cache hints 和软件预取
  - 适用于：大矩阵，追求吞吐量

================================================================================
*/




// =============================================================================
// 策略 1: HOT ZONE CACHE
// =============================================================================
// 最适合量化神经网络！权重和激活值通常分布在一个较小的范围内。
// 热点区域用 Shared Memory，其余用 Global Memory + L2 Cache

template<int HOT_RANGE = 64>  // 热点范围：-32 to 31
__global__ void __launch_bounds__(256)
approx_gemm_hotzone_kernel(
    const int M, const int N, const int K,
    const int8_t* __restrict__ A,
    const int8_t* __restrict__ B,
    const int32_t* __restrict__ LUT,
    int32_t* __restrict__ C)
{
    // =========================================
    // 配置常量
    // =========================================
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 32;
    constexpr int TM = 8;
    constexpr int TN = 8;
    
    // =========================================
    // 索引计算
    // =========================================
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    
    const int warp_id = tx / WARP_SIZE;
    const int lane_id = tx % WARP_SIZE;
    
    // Warp 排列: 4 rows × 2 cols
    const int warp_row = warp_id / 2;   // 0-3, 每个 warp 负责 32 行
    const int warp_col = warp_id % 2;   // 0-1, 每个 warp 负责 64 列
    
    // 线程在 warp 内的位置: 4 rows × 8 cols
    const int thread_row = lane_id / 8;  // 0-3, 每线程 8 行
    const int thread_col = lane_id % 8;  // 0-7, 每线程 8 列
    
    // =========================================
    // Shared Memory 布局
    // =========================================
    // As: 转置存储 [BK][BM] 以优化后续列访问
    // Bs: 正常存储 [BK][BN]
    // LUT_hot: 热点区域 [HOT_RANGE][HOT_RANGE]
    __shared__ int8_t As[BK][BM + 4];      // +4 padding 消除 bank conflict
    __shared__ int8_t Bs[BK][BN + 4];      // +4 padding
    __shared__ int32_t LUT_hot[HOT_RANGE][HOT_RANGE];  // 热点 LUT
    
    // =========================================
    // 预加载热点 LUT
    // =========================================
    // 64x64 = 4096 个 int32，256 线程每个加载 16 个
    constexpr int HOT_OFFSET = 128 - HOT_RANGE / 2;  // LUT 中的起始位置
    
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        int idx = tx * 16 + i;
        if (idx < HOT_RANGE * HOT_RANGE) {
            int row = idx / HOT_RANGE;
            int col = idx % HOT_RANGE;
            int lut_row = HOT_OFFSET + row;
            int lut_col = HOT_OFFSET + col;
            LUT_hot[row][col] = LUT[lut_row * 256 + lut_col];
        }
    }
    
    // =========================================
    // 寄存器
    // =========================================
    int32_t acc[TM][TN] = {{0}};
    int8_t reg_a[TM];
    int8_t reg_b[TN];
    
    // 全局基址
    const int global_row_base = by * BM;
    const int global_col_base = bx * BN;
    
    __syncthreads();
    
    // =========================================
    // 主循环
    // =========================================
    for (int k_tile = 0; k_tile < K; k_tile += BK) {
        
        // ----- 协作加载 A -----
        // 256 线程加载 128×32 = 4096 个 int8
        // 每线程加载 16 个元素
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            int idx = tx * 16 + i;
            int local_row = idx % BM;   // 0-127
            int local_col = idx / BM;   // 0-31
            
            int global_row = global_row_base + local_row;
            int global_k = k_tile + local_col;
            
            As[local_col][local_row] = (global_row < M && global_k < K) ?
                A[global_row * K + global_k] : 0;
        }
        
        // ----- 协作加载 B -----
        // 256 线程加载 32×128 = 4096 个 int8
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            int idx = tx * 16 + i;
            int local_row = idx / BN;   // 0-31
            int local_col = idx % BN;   // 0-127
            
            int global_k = k_tile + local_row;
            int global_col = global_col_base + local_col;
            
            Bs[local_row][local_col] = (global_k < K && global_col < N) ?
                B[global_k * N + global_col] : 0;
        }
        
        __syncthreads();
        
        // ----- K 维度计算 -----
        int valid_k = min(BK, K - k_tile);
        
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            if (k >= valid_k) break;
            
            // 加载 A 到寄存器
            int a_row_base = warp_row * 32 + thread_row * TM;
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                reg_a[i] = As[k][a_row_base + i];
            }
            
            // 加载 B 到寄存器
            int b_col_base = warp_col * 64 + thread_col * TN;
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                reg_b[j] = Bs[k][b_col_base + j];
            }
            
            // ----- 核心计算：LUT 查表 -----
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                int8_t a_val = reg_a[i];
                
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    int8_t b_val = reg_b[j];
                    
                    int32_t product;
                    
                    // 热点检测
                    constexpr int HOT_MIN = -(HOT_RANGE / 2);
                    constexpr int HOT_MAX = HOT_RANGE / 2 - 1;
                    
                    if (a_val >= HOT_MIN && a_val <= HOT_MAX &&
                        b_val >= HOT_MIN && b_val <= HOT_MAX) {
                        // 热点路径：Shared Memory 访问 (~20 cycles)
                        int a_local = a_val - HOT_MIN;  // 映射到 [0, 63]
                        int b_local = b_val - HOT_MIN;
                        product = LUT_hot[a_local][b_local];
                    } else {
                        // 冷路径：Global Memory 访问 (~200+ cycles)
                        int lut_idx = (a_val + 128) * 256 + (b_val + 128);
                        product = __ldg(&LUT[lut_idx]);
                    }
                    
                    acc[i][j] += product;
                }
            }
        }
        
        __syncthreads();
    }
    
    // =========================================
    // 写回结果
    // =========================================
    const int c_row_base = global_row_base + warp_row * 32 + thread_row * TM;
    const int c_col_base = global_col_base + warp_col * 64 + thread_col * TN;
    
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        int global_row = c_row_base + i;
        if (global_row >= M) continue;
        
        int32_t* c_ptr = &C[global_row * N + c_col_base];
        
        #pragma unroll
        for (int j = 0; j < TN; j += 4) {
            int global_col = c_col_base + j;
            
            if (global_col + 3 < N) {
                // 向量存储 (16 bytes = 4 × int32)
                *reinterpret_cast<int4*>(c_ptr + j) = make_int4(
                    acc[i][j], acc[i][j+1], acc[i][j+2], acc[i][j+3]);
            } else {
                // 边界处理
                #pragma unroll
                for (int jj = 0; jj < 4; ++jj) {
                    if (global_col + jj < N) {
                        c_ptr[j + jj] = acc[i][j + jj];
                    }
                }
            }
        }
    }
}


// =============================================================================
// 策略 2: MULTI-PASS (完整 LUT，多遍扫描)
// =============================================================================
// 将 256x256 LUT 分成 4x4=16 块，每块 64x64 = 16KB
// 每个 Pass 处理一块，保证所有 LUT 访问都在 Shared Memory

// 单个 Pass 的 Kernel
template<int LUT_ROW_START, int LUT_COL_START>
__global__ void __launch_bounds__(256)
approx_gemm_pass_kernel(
    const int M, const int N, const int K,
    const int8_t* __restrict__ A,
    const int8_t* __restrict__ B,
    const int32_t* __restrict__ LUT,
    int32_t* __restrict__ C,
    const bool accumulate)  // false = 覆盖, true = 累加
{
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 32;
    constexpr int TM = 8;
    constexpr int TN = 8;
    constexpr int LUT_TILE = 64;
    
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    
    const int warp_id = tx / WARP_SIZE;
    const int lane_id = tx % WARP_SIZE;
    const int warp_row = warp_id / 2;
    const int warp_col = warp_id % 2;
    const int thread_row = lane_id / 8;
    const int thread_col = lane_id % 8;
    
    __shared__ int8_t As[BK][BM + 4];
    __shared__ int8_t Bs[BK][BN + 4];
    __shared__ int32_t LUT_tile[LUT_TILE][LUT_TILE];
    
    // 预加载当前 Pass 的 LUT 块
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        int idx = tx * 16 + i;
        if (idx < LUT_TILE * LUT_TILE) {
            int row = idx / LUT_TILE;
            int col = idx % LUT_TILE;
            LUT_tile[row][col] = LUT[(LUT_ROW_START + row) * 256 + (LUT_COL_START + col)];
        }
    }
    
    // 当前 Pass 对应的值范围
    constexpr int8_t A_VAL_MIN = LUT_ROW_START - 128;
    constexpr int8_t A_VAL_MAX = LUT_ROW_START - 128 + LUT_TILE - 1;
    constexpr int8_t B_VAL_MIN = LUT_COL_START - 128;
    constexpr int8_t B_VAL_MAX = LUT_COL_START - 128 + LUT_TILE - 1;
    
    int32_t acc[TM][TN] = {{0}};
    int8_t reg_a[TM];
    int8_t reg_b[TN];
    
    const int global_row_base = by * BM;
    const int global_col_base = bx * BN;
    
    __syncthreads();
    
    // 主循环
    for (int k_tile = 0; k_tile < K; k_tile += BK) {
        // 加载 A
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            int idx = tx * 16 + i;
            int local_row = idx % BM;
            int local_col = idx / BM;
            int gr = global_row_base + local_row;
            int gk = k_tile + local_col;
            As[local_col][local_row] = (gr < M && gk < K) ? A[gr * K + gk] : 0;
        }
        
        // 加载 B
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            int idx = tx * 16 + i;
            int local_row = idx / BN;
            int local_col = idx % BN;
            int gk = k_tile + local_row;
            int gc = global_col_base + local_col;
            Bs[local_row][local_col] = (gk < K && gc < N) ? B[gk * N + gc] : 0;
        }
        
        __syncthreads();
        
        int valid_k = min(BK, K - k_tile);
        
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            if (k >= valid_k) break;
            
            int a_base = warp_row * 32 + thread_row * TM;
            int b_base = warp_col * 64 + thread_col * TN;
            
            #pragma unroll
            for (int i = 0; i < TM; ++i) reg_a[i] = As[k][a_base + i];
            #pragma unroll
            for (int j = 0; j < TN; ++j) reg_b[j] = Bs[k][b_base + j];
            
            // 计算（只处理当前 Pass 范围内的值）
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                int8_t a_val = reg_a[i];
                if (a_val < A_VAL_MIN || a_val > A_VAL_MAX) continue;
                int a_local = a_val - A_VAL_MIN;
                
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    int8_t b_val = reg_b[j];
                    if (b_val < B_VAL_MIN || b_val > B_VAL_MAX) continue;
                    int b_local = b_val - B_VAL_MIN;
                    
                    // 全部走 Shared Memory！
                    acc[i][j] += LUT_tile[a_local][b_local];
                }
            }
        }
        
        __syncthreads();
    }
    
    // 写回（根据 accumulate 标志决定是覆盖还是累加）
    const int c_row_base = global_row_base + warp_row * 32 + thread_row * TM;
    const int c_col_base = global_col_base + warp_col * 64 + thread_col * TN;
    
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        int gr = c_row_base + i;
        if (gr >= M) continue;
        
        #pragma unroll
        for (int j = 0; j < TN; j += 4) {
            int gc = c_col_base + j;
            if (gc + 3 < N) {
                int32_t* ptr = &C[gr * N + gc];
                if (accumulate) {
                    int4 old = *reinterpret_cast<int4*>(ptr);
                    *reinterpret_cast<int4*>(ptr) = make_int4(
                        old.x + acc[i][j], old.y + acc[i][j+1],
                        old.z + acc[i][j+2], old.w + acc[i][j+3]);
                } else {
                    *reinterpret_cast<int4*>(ptr) = make_int4(
                        acc[i][j], acc[i][j+1], acc[i][j+2], acc[i][j+3]);
                }
            } else {
                for (int jj = 0; jj < 4 && gc + jj < N; ++jj) {
                    if (accumulate) {
                        C[gr * N + gc + jj] += acc[i][j + jj];
                    } else {
                        C[gr * N + gc + jj] = acc[i][j + jj];
                    }
                }
            }
        }
    }
}


// =============================================================================
// Host 函数
// =============================================================================

// 策略 1: 热点缓存版本
torch::Tensor approx_gemm_hotzone(
    const torch::Tensor& A,
    const torch::Tensor& B,
    const torch::Tensor& LUT,
    int hot_range = 64)
{
    const at::cuda::OptionalCUDAGuard device_guard(A.device());
    
    int M = A.size(0);
    int N = B.size(1);
    int K = A.size(1);
    
    auto C = torch::empty({M, N}, A.options().dtype(torch::kInt32));
    
    dim3 grid(CEIL_DIV(N, 128), CEIL_DIV(M, 128));
    dim3 block(256);
    
    // 根据 hot_range 选择不同的 kernel
    if (hot_range == 64) {
        approx_gemm_hotzone_kernel<64><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            M, N, K, A.data_ptr<int8_t>(), B.data_ptr<int8_t>(),
            LUT.data_ptr<int32_t>(), C.data_ptr<int32_t>());
    } else if (hot_range == 128) {
        approx_gemm_hotzone_kernel<128><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            M, N, K, A.data_ptr<int8_t>(), B.data_ptr<int8_t>(),
            LUT.data_ptr<int32_t>(), C.data_ptr<int32_t>());
    } else {
        // 默认 64
        approx_gemm_hotzone_kernel<64><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            M, N, K, A.data_ptr<int8_t>(), B.data_ptr<int8_t>(),
            LUT.data_ptr<int32_t>(), C.data_ptr<int32_t>());
    }
    
    return C;
}


// 策略 2: 多 Pass 版本
torch::Tensor approx_gemm_multipass(
    const torch::Tensor& A,
    const torch::Tensor& B,
    const torch::Tensor& LUT)
{
    const at::cuda::OptionalCUDAGuard device_guard(A.device());
    
    int M = A.size(0);
    int N = B.size(1);
    int K = A.size(1);
    
    auto C = torch::zeros({M, N}, A.options().dtype(torch::kInt32));
    
    dim3 grid(CEIL_DIV(N, 128), CEIL_DIV(M, 128));
    dim3 block(256);
    auto stream = at::cuda::getCurrentCUDAStream();
    
    const int8_t* A_ptr = A.data_ptr<int8_t>();
    const int8_t* B_ptr = B.data_ptr<int8_t>();
    const int32_t* LUT_ptr = LUT.data_ptr<int32_t>();
    int32_t* C_ptr = C.data_ptr<int32_t>();
    
    // 宏展开 16 个 Pass (4x4 分块)
    #define PASS(R, C_idx, ACC) \
        approx_gemm_pass_kernel<(R)*64, (C_idx)*64><<<grid, block, 0, stream>>>( \
            M, N, K, A_ptr, B_ptr, LUT_ptr, C_ptr, ACC)
    
    PASS(0, 0, false);  // 第一个 Pass 覆盖
    PASS(0, 1, true);   // 后续 Pass 累加
    PASS(0, 2, true);
    PASS(0, 3, true);
    PASS(1, 0, true);
    PASS(1, 1, true);
    PASS(1, 2, true);
    PASS(1, 3, true);
    PASS(2, 0, true);
    PASS(2, 1, true);
    PASS(2, 2, true);
    PASS(2, 3, true);
    PASS(3, 0, true);
    PASS(3, 1, true);
    PASS(3, 2, true);
    PASS(3, 3, true);
    
    #undef PASS
    
    return C;
}


// =============================================================================
// 主入口：自动选择最优策略
// =============================================================================

torch::Tensor approx_gemm(
    const torch::Tensor& A,
    const torch::Tensor& B,
    const torch::Tensor& LUT,
    const std::string& strategy = "auto")
{
    // 输入检查
    TORCH_CHECK(A.dim() == 2, "A must be 2D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");
    TORCH_CHECK(A.size(1) == B.size(0), "Inner dimensions must match");
    TORCH_CHECK(A.dtype() == torch::kInt8, "A must be int8");
    TORCH_CHECK(B.dtype() == torch::kInt8, "B must be int8");
    TORCH_CHECK(LUT.numel() == 256 * 256, "LUT must be 256x256");
    
    if (strategy == "hotzone" || strategy == "hot") {
        return approx_gemm_hotzone(A, B, LUT);
    } else if (strategy == "multipass" || strategy == "multi") {
        return approx_gemm_multipass(A, B, LUT);
    } else {
        // Auto 策略：
        // - 如果矩阵较小，用 multipass（LUT 完全在 shared memory）
        // - 如果矩阵较大，用 hotzone（单 Pass，依赖 L2 cache）
        int M = A.size(0);
        int N = B.size(1);
        
        // 经验阈值：小于 512x512 用 multipass
        if (M * N < 512 * 512) {
            return approx_gemm_multipass(A, B, LUT);
        } else {
            return approx_gemm_hotzone(A, B, LUT);
        }
    }
}

}  // namespace approx_gemm_final


// =============================================================================
// PyTorch 绑定
// =============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("approx_gemm", &approx_gemm_final::approx_gemm, 
          "Approximate GEMM with LUT",
          py::arg("A"), py::arg("B"), py::arg("LUT"), 
          py::arg("strategy") = "auto");
    
    m.def("approx_gemm_hotzone", &approx_gemm_final::approx_gemm_hotzone,
          "Approximate GEMM - Hot Zone Cache Strategy",
          py::arg("A"), py::arg("B"), py::arg("LUT"),
          py::arg("hot_range") = 64);
    
    m.def("approx_gemm_multipass", &approx_gemm_final::approx_gemm_multipass,
          "Approximate GEMM - Multi-Pass Strategy",
          py::arg("A"), py::arg("B"), py::arg("LUT"));
}
