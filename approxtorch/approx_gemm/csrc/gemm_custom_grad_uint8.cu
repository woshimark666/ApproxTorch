#include "gemm.cuh"
namespace approxtorch{

constexpr int TILE_DIM = 16;
// lookup the gradlut
__device__ __forceinline__ 
float fetch_grad(uint index, const float* __restrict__ grad_lut)
{
    return __ldg(grad_lut + index);
}

// ================================================================
// Kernel 1: 计算 Feature 的梯度 (Backward w.r.t Input)
// 逻辑类似于: Grad_Input = Grad_Output @ Transpose(Weight)
// 形状: [BL, CKK] = [BL, O] x [O, CKK]
// ================================================================
__global__ void grad_feature_uint8_tt_kernel(
    float* __restrict__ grad_feature,      // Output: (BL, CKK)
    const uint8_t* __restrict__ mat_feature,   // Input: (BL, CKK) -> 用于计算索引
    const uint8_t* __restrict__ mat_weight,    // Input: (O, CKK)  -> 用于计算索引
    const float* __restrict__ grad_output, // Input: (BL, O), upstream gradient
    const float* __restrict__ lut_dx,      // LUT of df(x,y) / dx, one dimension
    float scale_w, float zero_w,           // scale and zero of weight,
    uint M, uint N, uint K)                   // M=BL, N=O, K=CKK
{
    // Block索引
    uint by = blockIdx.y;    // CKK
    uint bx = blockIdx.x;     // BL
    
    // Thread索引
    uint ty = threadIdx.y;
    uint tx = threadIdx.x;

    // 当前线程计算的目标坐标 (row, col) -> (BL 维度, CKK 维度)
    uint row = by * TILE_DIM + ty;   // BL 维度
    uint col = bx * TILE_DIM + tx;   // CKK 维度

    // 累加器
    float acc = 0.0f;

    // 共享内存，用于缓存 grad_output 和 mat_weight 的分块
    // ds_grad_out: [TILE, TILE]
    // ds_weight:   [TILE, TILE]
    __shared__ float ds_grad_out[TILE_DIM][TILE_DIM];
    __shared__ uint8_t   ds_weight[TILE_DIM][TILE_DIM];

    // 既然我们要计算 grad_feature[row, col]，我们需要用到 mat_feature[row, col]
    // 这个值对于内部的循环是不变的，我们可以提前加载到寄存器
    uint my_feat_val = 0;
    if (row < M && col < K) {
        my_feat_val = mat_feature[row * K + col];
    }

    // 循环遍历 N (Output Channels) 维度
    for (uint t = 0; t < (N + TILE_DIM - 1) / TILE_DIM; ++t) {
        // 1. 加载 grad_output 到共享内存 (Row 对应 M, Col 对应 N)
        uint tiled_n_idx = t * TILE_DIM + tx; 
        if (row < M && tiled_n_idx < N) {
            ds_grad_out[ty][tx] = grad_output[row * N + tiled_n_idx];
        } else {
            ds_grad_out[ty][tx] = 0.0f;
        }

        // 2. 加载 mat_weight 到共享内存 
        // 注意：我们需要转置的访问模式，因为是 @ Weight.T
        // Weight 形状是 (CKK, O) -> (K, N)
        // 我们需要加载 Weight[col, tiled_n_idx] 吗？
        // 让我们看公式: Sum_over_n ( Grad_out[m, n] * LUT(Feat[m,k], Weight[k,n]) )
        // 这里 col 对应 k。tiled_n 对应 n。
        // 所以我们需要 Weight[col, tiled_n]
        // 为了利用 shared memory 的 broadcasting，我们将 chunk 内的 weight 加载进来
        // 这里 ds_weight 存储 (K维度切片, N维度切片)
        uint tiled_k_idx = bx * TILE_DIM + ty; // 注意这里为了合并访问可能需要调整
        uint tiled_n_idx_for_w = t * TILE_DIM + tx;
        
        if (tiled_k_idx < K && tiled_n_idx_for_w < N) {
             ds_weight[ty][tx] = mat_weight[tiled_k_idx * N + tiled_n_idx_for_w];
        } else {
             ds_weight[ty][tx] = 0;  // Padding
        }

        __syncthreads();

        // 3. 计算乘积
        // 我们需要计算 Row(grad_out) dot Col(Weight_Transposed)
        // 实际上是遍历当前的 Tile 长度
        #pragma unroll
        for (uint k = 0; k < TILE_DIM; ++k) {
            // 这里的 k 对应 N 维度的分块索引
            
            // 取出 grad_out 的值: ds_grad_out[ty][k] -> grad_out[row, current_n]
            float g_val = ds_grad_out[ty][k];
            
            // 取出 weight 的值: 
            // 我们需要 weight[col, current_n]
            // 在加载阶段，ds_weight[y][x] 存的是 weight[bx*TILE+y, t*TILE+x]
            // 也就是说 ds_weight[local_k][k] 对应 weight[local_k_global, current_n]
            // 我们的 col (全局K索引) 是 bx*TILE + tx
            // 等等，这里的 Shared Memory 映射有点 tricky。
            // 为了简化逻辑且保证正确性，我们可以只缓存 Weight，
            // 但是因为 Weight 取决于 col (tx)，而 col 在 Block 内是变化的。
            // 简单方案：直接从 Shared Memory 读。
            // ds_weight 的行是 K (对应我们的 col), 列是 N (对应循环变量 k)
            // 但是 ds_weight 是按 [ty][tx] 加载的。
            // 我们需要的 weight 索引是 [tx][k] (相对于 block)
            
            uint w_val = ds_weight[tx][k]; 

            // 计算 LUT 索引
            uint idx = my_feat_val * 256 + w_val;
            
            // 查表 + 变换
            float lut_val = fetch_grad(idx, lut_dx);
            float deriv = (lut_val - zero_w) * scale_w;

            acc += g_val * deriv;
        }
        __syncthreads();
    }

    if (row < M && col < K) {
        grad_feature[row * K + col] = acc;
    }
}

// ================================================================
// Kernel 2: 计算 Weight 的梯度 (Backward w.r.t Weight)
// 逻辑类似于: Grad_Weight = Transpose(Input) @ Grad_Output
// 形状: [CKK, O] = [CKK, BL] x [BL, O]
// ================================================================
__global__ void grad_weight_uint8_tt_kernel(
    float* __restrict__ grad_weight,       // Output: (CKK, O)
    const uint8_t* __restrict__ mat_feature,   // Input: (BL, CKK)
    const uint8_t* __restrict__ mat_weight,    // Input: (O, CKK)
    const float* __restrict__ grad_output, // Input: (BL, O)
    const float* __restrict__ lut_dy,      // LUT
    float scale_f, float zero_f,
    uint K, uint N, uint M)                   // K=CKK, N=O, M=BL
{
    uint by = blockIdx.y;
    uint bx = blockIdx.x;
    uint ty = threadIdx.y;
    uint tx = threadIdx.x;

    // 目标坐标: row -> CKK, col -> O
    uint row = by * TILE_DIM + ty;
    uint col = bx * TILE_DIM + tx;

    float acc = 0.0f;

    __shared__ uint8_t   ds_feat[TILE_DIM][TILE_DIM];
    __shared__ float ds_grad_out[TILE_DIM][TILE_DIM];

    // 当前线程对应的 Weight 值 (固定)
    uint8_t my_w_val = 0;
    if (row < K && col < N) {
        my_w_val = mat_weight[row * N + col];
    }

    // 循环遍历 M (Batch * L) 维度
    for (uint t = 0; t < (M + TILE_DIM - 1) / TILE_DIM; ++t) {
        
        // 1. 加载 mat_feature (需要转置: 我们需要 Feat.T)
        // Feat 形状 (M, K)。我们需要加载块 (M_chunk, K_chunk)
        // 我们目标是行 row(K), 遍历维度 M
        // ds_feat[ty][tx] -> 加载 mat_feature[t*TILE + ty, row_base + tx] ?
        // 让我们按照标准 GEMM: C = A.T * B
        // A (Feat) is (M, K). B (Grad) is (M, N).
        // Loop index k (represents M).
        // Load A tile: (k, row) -> ds_feat
        // Load B tile: (k, col) -> ds_grad_out
        
        uint tiled_m = t * TILE_DIM + ty;
        uint global_k = by * TILE_DIM + tx; // Transposed loading for feature
        
        if (tiled_m < M && global_k < K) {
             ds_feat[ty][tx] = mat_feature[tiled_m * K + global_k];
        } else {
             ds_feat[ty][tx] = 0;
        }

        // 2. 加载 grad_output (M, N)
        // 这里的行是 tiled_m (ty), 列是 global_col (bx*TILE + tx)
        uint global_n = bx * TILE_DIM + tx;
        if (tiled_m < M && global_n < N) {
            ds_grad_out[ty][tx] = grad_output[tiled_m * N + global_n];
        } else {
            ds_grad_out[ty][tx] = 0.0f;
        }

        __syncthreads();

        // 3. 累加
        #pragma unroll
        for (uint k = 0; k < TILE_DIM; ++k) {
            // k 代表 M 维度的局部索引
            
            // Feat: ds_feat[k][ty] (因为我们加载时做了技巧，或者直接按行列取)
            // 让我们理一下：
            // ds_feat[k][ty] 对应 mat_feature[t*TILE+k, row]
            // 此时 row 是我们的 output row (CKK维度)
            // 这里的 ty 对应 output row 的局部偏移
            uint f_val = ds_feat[k][ty];
            
            // Grad: ds_grad_out[k][tx] 对应 grad_output[t*TILE+k, col]
            float g_val = ds_grad_out[k][tx];

            uint idx = f_val * 256 + my_w_val;
            
            float lut_val = fetch_grad(idx, lut_dy);
            float deriv = (lut_val - zero_f) * scale_f;

            acc += g_val * deriv;
        }
        __syncthreads();
    }

    if (row < K && col < N) {
        grad_weight[row * N + col] = acc;
    }
}

std::tuple<torch::Tensor, torch::Tensor>
gemm_custom_grad_uint8_tt(const torch::Tensor& A, const torch::Tensor& B,
                const torch::Tensor& upstream_grad,
                const torch::Tensor& grad_lut_dx,
                const torch::Tensor& grad_lut_dy,
                const torch::Tensor& scale_A, const torch::Tensor& zero_A,
                const torch::Tensor& scale_B, const torch::Tensor& zero_B)
{
    const at::cuda::OptionalCUDAGuard device_guard(device_of(A));

    uint BL = A.size(0);
    uint CKK = A.size(1);
    uint O = B.size(1); 

    auto options = torch::TensorOptions().dtype(torch::kFloat).device(A.device());
    torch::Tensor grad_A = torch::zeros({BL, CKK}, options);
    torch::Tensor grad_B = torch::zeros({CKK, O}, options);



    float scale_A_f = scale_A.item<float>();
    float zero_A_f = zero_A.item<float>();
    float scale_B_f = scale_B.item<float>();
    float zero_B_f = zero_B.item<float>();


    const dim3 block(TILE_DIM, TILE_DIM);
    const dim3 grid_feature(
        (CKK + TILE_DIM - 1) / TILE_DIM,
        (BL + TILE_DIM - 1) / TILE_DIM);

    grad_feature_uint8_tt_kernel<<<grid_feature, block, 0>>>
    (grad_A.data_ptr<float>(), 
        A.data_ptr<uint8_t>(), B.data_ptr<uint8_t>(), 
        upstream_grad.data_ptr<float>(), 
        grad_lut_dx.data_ptr<float>(), 
        scale_B_f, zero_B_f, BL, O, CKK);
    cudaDeviceSynchronize();

    dim3 grid_weight(
        (O + TILE_DIM - 1) / TILE_DIM,
        (CKK + TILE_DIM - 1) / TILE_DIM);

    grad_weight_uint8_tt_kernel<<<grid_weight, block, 0, at::cuda::getCurrentCUDAStream()>>>
    (grad_B.data_ptr<float>(), 
        A.data_ptr<uint8_t>(), B.data_ptr<uint8_t>(), 
        upstream_grad.data_ptr<float>(), 
        grad_lut_dy.data_ptr<float>(), scale_A_f, zero_A_f, CKK, O, BL);

    // check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }

    return std::make_tuple(grad_A, grad_B);
}



// ================================================================
// Kernel 1: 计算 Feature 的梯度 (Backward w.r.t Input)
// 逻辑类似于: Grad_Input = Grad_Output @ Transpose(Weight)
// 形状: [BL, CKK] = [BL, O] x [O, CKK]
// ================================================================
__global__ void grad_feature_uint8_tc_kernel(
    float* __restrict__ grad_feature,      // Output: (BL, CKK)
    const uint8_t* __restrict__ mat_feature,   // Input: (BL, CKK) -> 用于计算索引
    const uint8_t* __restrict__ mat_weight,    // Input: (O, CKK)  -> 用于计算索引
    const float* __restrict__ grad_output, // Input: (BL, O), upstream gradient
    const float* __restrict__ lut_dx,      // LUT of df(x,y) / dx, one dimension
    const float* __restrict__ scale_w, // (O,)
    const float* __restrict__ zero_w,  // scale and zero of weight, shape is (O,)
    uint M, uint N, uint K)                   // M=BL, N=O, K=CKK
{
    // Block索引
    uint by = blockIdx.y;    // CKK
    uint bx = blockIdx.x;     // BL
    
    // Thread索引
    uint ty = threadIdx.y;
    uint tx = threadIdx.x;

    // 当前线程计算的目标坐标 (row, col) -> (BL 维度, CKK 维度)
    uint row = by * TILE_DIM + ty;   // BL 维度
    uint col = bx * TILE_DIM + tx;   // CKK 维度

    // 累加器
    float acc = 0.0f;

    // 共享内存，用于缓存 grad_output 和 mat_weight 的分块
    // ds_grad_out: [TILE, TILE]
    // ds_weight:   [TILE, TILE]
    __shared__ float ds_grad_out[TILE_DIM][TILE_DIM];
    __shared__ uint8_t   ds_weight[TILE_DIM][TILE_DIM];

    // [新增] 1. 增加 Shared Memory 来存当前 Tile 的 Scale 和 Zero
    // 因为是 Per-Channel，只需要存 N 维度的 TILE_DIM 个值
    __shared__ float ds_scale[TILE_DIM];
    __shared__ float ds_zero[TILE_DIM];
    
    // 既然我们要计算 grad_feature[row, col]，我们需要用到 mat_feature[row, col]
    // 这个值对于内部的循环是不变的，我们可以提前加载到寄存器
    uint my_feat_val = 0;
    if (row < M && col < K) {
        my_feat_val = mat_feature[row * K + col];
    }

    // 循环遍历 N (Output Channels) 维度
    for (uint t = 0; t < (N + TILE_DIM - 1) / TILE_DIM; ++t) {
        // 1. 加载 grad_output 到共享内存 (Row 对应 M, Col 对应 N)
        uint tiled_n_idx = t * TILE_DIM + tx; 
        if (row < M && tiled_n_idx < N) {
            ds_grad_out[ty][tx] = grad_output[row * N + tiled_n_idx];
        } else {
            ds_grad_out[ty][tx] = 0.0f;
        }

        // 2. 加载 mat_weight 到共享内存 
        // 注意：我们需要转置的访问模式，因为是 @ Weight.T
        // Weight 形状是 (CKK, O) -> (K, N)
        // 我们需要加载 Weight[col, tiled_n_idx] 吗？
        // 让我们看公式: Sum_over_n ( Grad_out[m, n] * LUT(Feat[m,k], Weight[k,n]) )
        // 这里 col 对应 k。tiled_n 对应 n。
        // 所以我们需要 Weight[col, tiled_n]
        // 为了利用 shared memory 的 broadcasting，我们将 chunk 内的 weight 加载进来
        // 这里 ds_weight 存储 (K维度切片, N维度切片)
        uint tiled_k_idx = bx * TILE_DIM + ty; // 注意这里为了合并访问可能需要调整
        uint tiled_n_idx_for_w = t * TILE_DIM + tx;
        
        if (tiled_k_idx < K && tiled_n_idx_for_w < N) {
             ds_weight[ty][tx] = mat_weight[tiled_k_idx * N + tiled_n_idx_for_w];
        } else {
             ds_weight[ty][tx] = 0;  // Padding
        }
        
        // [新增] 2. 加载 scale_w 和 zero_w 到 Shared Memory
        // scale_w 是一维向量，长度为 N。当前 Tile 对应的 N 索引范围是 [t*TILE, t*TILE + TILE]
        // 我们只需要利用 block 内的一行线程 (例如 ty==0) 来加载即可
        if (ty == 0) {
            if (tiled_n_idx < N) { // tiled_n_idx = t * TILE + tx
                ds_scale[tx] = scale_w[tiled_n_idx];
                ds_zero[tx]  = zero_w[tiled_n_idx];
            } else {
                ds_scale[tx] = 1.0f; // Padding: 防止计算出 NaN/Inf，虽然后面 grad=0 也会抵消
                ds_zero[tx]  = 0.0f;
            }
        }

        __syncthreads();

        // 3. 计算乘积
        // 我们需要计算 Row(grad_out) dot Col(Weight_Transposed)
        // 实际上是遍历当前的 Tile 长度
        #pragma unroll
        for (uint k = 0; k < TILE_DIM; ++k) {
            // 这里的 k 对应 N 维度的分块索引
            
            // 取出 grad_out 的值: ds_grad_out[ty][k] -> grad_out[row, current_n]
            float g_val = ds_grad_out[ty][k];
            
            // 取出 weight 的值: 
            // 我们需要 weight[col, current_n]
            // 在加载阶段，ds_weight[y][x] 存的是 weight[bx*TILE+y, t*TILE+x]
            // 也就是说 ds_weight[local_k][k] 对应 weight[local_k_global, current_n]
            // 我们的 col (全局K索引) 是 bx*TILE + tx
            // 等等，这里的 Shared Memory 映射有点 tricky。
            // 为了简化逻辑且保证正确性，我们可以只缓存 Weight，
            // 但是因为 Weight 取决于 col (tx)，而 col 在 Block 内是变化的。
            // 简单方案：直接从 Shared Memory 读。
            // ds_weight 的行是 K (对应我们的 col), 列是 N (对应循环变量 k)
            // 但是 ds_weight 是按 [ty][tx] 加载的。
            // 我们需要的 weight 索引是 [tx][k] (相对于 block)
            
            uint w_val = ds_weight[tx][k]; 

            // 计算 LUT 索引
            uint idx = my_feat_val * 256 + w_val;
            
            // 查表 + 变换
            float lut_val = fetch_grad(idx, lut_dx);
            float current_scale = ds_scale[k];
            float current_zero  = ds_zero[k];
            float deriv = (lut_val - current_zero) * current_scale;

            acc += g_val * deriv;
        }
        __syncthreads();
    }

    if (row < M && col < K) {
        grad_feature[row * K + col] = acc;
    }
}

std::tuple<torch::Tensor, torch::Tensor>
gemm_custom_grad_uint8_tc(const torch::Tensor& A, const torch::Tensor& B,
                const torch::Tensor& upstream_grad,
                const torch::Tensor& grad_lut_dx,
                const torch::Tensor& grad_lut_dy,
                const torch::Tensor& scale_A, const torch::Tensor& zero_A,
                const torch::Tensor& scale_B, const torch::Tensor& zero_B)
{
    const at::cuda::OptionalCUDAGuard device_guard(device_of(A));

    uint BL = A.size(0);
    uint CKK = A.size(1);
    uint O = B.size(1); 

    auto options = torch::TensorOptions().dtype(torch::kFloat).device(A.device());
    torch::Tensor grad_A = torch::zeros({BL, CKK}, options);
    torch::Tensor grad_B = torch::zeros({CKK, O}, options);



    float scale_A_f = scale_A.item<float>();
    float zero_A_f = zero_A.item<float>();
    float scale_B_f = scale_B.item<float>();
    float zero_B_f = zero_B.item<float>();


    const dim3 block(TILE_DIM, TILE_DIM);
    const dim3 grid_feature(
        (CKK + TILE_DIM - 1) / TILE_DIM,
        (BL + TILE_DIM - 1) / TILE_DIM);

    grad_feature_uint8_tc_kernel<<<grid_feature, block, 0>>>
    (grad_A.data_ptr<float>(), 
        A.data_ptr<uint8_t>(), B.data_ptr<uint8_t>(), 
        upstream_grad.data_ptr<float>(), 
        grad_lut_dx.data_ptr<float>(), 
        scale_B.data_ptr<float>(), zero_B.data_ptr<float>(), BL, O, CKK);
    cudaDeviceSynchronize();

    dim3 grid_weight(
        (O + TILE_DIM - 1) / TILE_DIM,
        (CKK + TILE_DIM - 1) / TILE_DIM);

    grad_weight_uint8_tt_kernel<<<grid_weight, block, 0, at::cuda::getCurrentCUDAStream()>>>
    (grad_B.data_ptr<float>(), 
        A.data_ptr<uint8_t>(), B.data_ptr<uint8_t>(), 
        upstream_grad.data_ptr<float>(), 
        grad_lut_dy.data_ptr<float>(), scale_A_f, zero_A_f, CKK, O, BL);

    // check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }

    return std::make_tuple(grad_A, grad_B);
}


TORCH_LIBRARY_FRAGMENT(approxtorch, m){
    m.def("gemm_custom_grad_uint8_tt(Tensor A, Tensor B, Tensor upstream_grad, Tensor grad_lut_dx, Tensor grad_lut_dy, Tensor scale_A, Tensor zero_A, Tensor scale_B, Tensor zero_B) -> (Tensor, Tensor)");
    m.def("gemm_custom_grad_uint8_tc(Tensor A, Tensor B, Tensor upstream_grad, Tensor grad_lut_dx, Tensor grad_lut_dy, Tensor scale_A, Tensor zero_A, Tensor scale_B, Tensor zero_B) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(approxtorch, CUDA, m){
    m.impl("gemm_custom_grad_uint8_tt", &gemm_custom_grad_uint8_tt);
    m.impl("gemm_custom_grad_uint8_tc", &gemm_custom_grad_uint8_tc);
}

}
