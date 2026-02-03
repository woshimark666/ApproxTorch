#include "torch/types.h"
#include <Python.h>
#include <ATen/Operators.h>
#include <cstdint>
#include <torch/all.h>
#include <torch/library.h>


namespace approxtorch {


torch::Tensor gemm_int8(
    const torch::Tensor& A,
    const torch::Tensor& B,
    const torch::Tensor& lut
) {
    // Input validation (same as above)
    TORCH_CHECK(A.dtype() == torch::kInt8, "A must be int8");
    TORCH_CHECK(B.dtype() == torch::kInt8, "B must be int8");
    TORCH_CHECK(lut.dtype() == torch::kInt32, "LUT must be int32");
    
    TORCH_CHECK(A.dim() == 2, "A must be 2D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be 2D tensor");
    TORCH_CHECK(lut.dim() == 1, "LUT must be 1D tensor");
    
    TORCH_CHECK(A.size(1) == B.size(0), "Inner dimensions must match: A[M,K] x B[K,N]");
    TORCH_CHECK(lut.size(0) == 256 * 256, "LUT must be 256 * 256");
    
    const int64_t M = A.size(0);
    const int64_t K = A.size(1);
    const int64_t N = B.size(1);
    
    // Transpose B for better cache access: B_T[N, K]
    auto B_T = B.t().contiguous();
    
    // Create output tensor
    auto C = torch::zeros({M, N}, torch::TensorOptions().dtype(torch::kInt32));
    
    // Get raw pointers
    const int8_t* A_ptr = A.data_ptr<int8_t>();
    const int8_t* B_T_ptr = B_T.data_ptr<int8_t>();
    const int32_t* lut_ptr = lut.data_ptr<int32_t>();
    int32_t* C_ptr = C.data_ptr<int32_t>();
    
    // Perform GEMM with better cache locality
    #pragma omp parallel for collapse(2) if(M * N > 1000)
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            int32_t sum = 0;
            const int8_t* A_row = A_ptr + i * K;
            const int8_t* B_col = B_T_ptr + j * K;  // Now contiguous!
            
            for (int64_t k = 0; k < K; ++k) {
                int a_idx = static_cast<int>(A_row[k]);
                int b_idx = static_cast<int>(B_col[k]);
                sum += lut_ptr[a_idx * 256 + b_idx + 32896];
            }
            C_ptr[i * N + j] = sum;
        }
    }
    
    return C;
}


torch::Tensor gemm_uint8(
    const torch::Tensor& A,
    const torch::Tensor& B,
    const torch::Tensor& lut
) {
    // Input validation (same as above)
    TORCH_CHECK(A.dtype() == torch::kUInt8, "A must be int8");
    TORCH_CHECK(B.dtype() == torch::kUInt8, "B must be int8");
    TORCH_CHECK(lut.dtype() == torch::kInt32, "LUT must be int32");
    
    TORCH_CHECK(A.dim() == 2, "A must be 2D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be 2D tensor");
    TORCH_CHECK(lut.dim() == 1, "LUT must be 1D tensor");
    
    TORCH_CHECK(A.size(1) == B.size(0), "Inner dimensions must match: A[M,K] x B[K,N]");
    TORCH_CHECK(lut.size(0) == 256 * 256, "LUT must be 256 * 256");
    
    const int64_t M = A.size(0);
    const int64_t K = A.size(1);
    const int64_t N = B.size(1);
    
    // Transpose B for better cache access: B_T[N, K]
    auto B_T = B.t().contiguous();
    
    // Create output tensor
    auto C = torch::zeros({M, N}, torch::TensorOptions().dtype(torch::kInt32));
    
    // Get raw pointers
    const uint8_t* A_ptr = A.data_ptr<uint8_t>();
    const uint8_t* B_T_ptr = B_T.data_ptr<uint8_t>();
    const int32_t* lut_ptr = lut.data_ptr<int32_t>();
    int32_t* C_ptr = C.data_ptr<int32_t>();
    
    // Perform GEMM with better cache locality
    #pragma omp parallel for collapse(2) if(M * N > 1000)
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            int32_t sum = 0;
            const uint8_t* A_row = A_ptr + i * K;
            const uint8_t* B_col = B_T_ptr + j * K;  // Now contiguous!
            
            for (int64_t k = 0; k < K; ++k) {
                int64_t a_idx = static_cast<int64_t>(A_row[k]);
                int64_t b_idx = static_cast<int64_t>(B_col[k]);
                sum += lut_ptr[a_idx * 256 + b_idx];
            }
            C_ptr[i * N + j] = sum;
        }
    }
    
    return C;
}

TORCH_LIBRARY_FRAGMENT(approxtorch, m){
    // m.def("im2col_uint8(Tensor input, int k_h, int k_w, int stride_h, int stride_w, int padding_h, int padding_w, int dilation_h, int dilation_w) -> Tensor");
    m.def("gemm_int8(Tensor A, Tensor B, Tensor lut) -> Tensor");
    m.def("gemm_uint8(Tensor A, Tensor B, Tensor lut) -> Tensor");
}

TORCH_LIBRARY_IMPL(approxtorch, CPU, m){
    m.impl("gemm_int8", &gemm_int8);
    m.impl("gemm_uint8", &gemm_uint8);
}

}