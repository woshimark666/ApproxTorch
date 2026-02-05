#include "torch/types.h"
#include <Python.h>
#include <ATen/Operators.h>
#include <cstdint>
#include <torch/all.h>
#include <torch/library.h>




namespace approxtorch {


torch::Tensor bgemm_int8_cpu(
    const torch::Tensor& A,      // [B, CKK, L]
    const torch::Tensor& B,      // [O, CKK]
    const torch::Tensor& lut     // [256 * 256]
) {
    // Input validation
    TORCH_CHECK(A.dtype() == torch::kInt8, "A must be int8");
    TORCH_CHECK(B.dtype() == torch::kInt8, "B must be int8");
    TORCH_CHECK(lut.dtype() == torch::kInt32, "LUT must be int32");
    
    TORCH_CHECK(A.dim() == 3, "A must be 3D tensor [B, CKK, L]");
    TORCH_CHECK(B.dim() == 2, "B must be 2D tensor [O, CKK]");
    TORCH_CHECK(lut.dim() == 1, "LUT must be 1D tensor");
    
    TORCH_CHECK(A.size(1) == B.size(1), "A[B, CKK, L] and B[O, CKK] must have matching CKK dimension");
    TORCH_CHECK(lut.size(0) == 256 * 256, "LUT must be 256 * 256");
    
    const int64_t batch = A.size(0);   // B
    const int64_t CKK = A.size(1);     // CKK (K dimension for matmul)
    const int64_t L = A.size(2);       // L
    const int64_t O = B.size(0);       // O
    
    // Make A contiguous if needed
    auto A_contig = A.contiguous();
    
    // B is [O, CKK], already in good layout for row access
    auto B_contig = B.contiguous();
    
    // Create output tensor [B, O, L]
    auto C = torch::zeros({batch, O, L}, torch::TensorOptions().dtype(torch::kInt32));
    
    // Get raw pointers
    const int8_t* A_ptr = A_contig.data_ptr<int8_t>();
    const int8_t* B_ptr = B_contig.data_ptr<int8_t>();
    const int32_t* lut_ptr = lut.data_ptr<int32_t>();
    int32_t* C_ptr = C.data_ptr<int32_t>();
    
    // Strides
    const int64_t A_batch_stride = CKK * L;  // stride for batch dimension in A
    const int64_t C_batch_stride = O * L;    // stride for batch dimension in C
    
    // For each batch: C[b, :, :] = B @ A[b, :, :]
    // B[O, CKK] @ A[b, CKK, L] = C[b, O, L]
    // C[b, o, l] = sum_k B[o, k] * A[b, k, l]
    
    #pragma omp parallel for collapse(3) if(batch * O * L > 1000)
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t o = 0; o < O; ++o) {
            for (int64_t l = 0; l < L; ++l) {
                int32_t sum = 0;
                
                const int8_t* B_row = B_ptr + o * CKK;           // B[o, :]
                const int8_t* A_slice = A_ptr + b * A_batch_stride;  // A[b, :, :]
                
                for (int64_t k = 0; k < CKK; ++k) {
                    // B[o, k] and A[b, k, l]
                    int b_val = static_cast<int>(B_row[k]);
                    int a_val = static_cast<int>(A_slice[k * L + l]);
                    sum += lut_ptr[b_val * 256 + a_val + 32896];
                }
                
                C_ptr[b * C_batch_stride + o * L + l] = sum;
            }
        }
    }
    
    return C;
}
    

torch::Tensor bgemm_uint8_cpu(
    const torch::Tensor& A,      // [B, CKK, L]
    const torch::Tensor& B,      // [O, CKK]
    const torch::Tensor& lut     // [256 * 256]
) {
    // Input validation
    TORCH_CHECK(A.dtype() == torch::kUInt8, "A must be uint8");
    TORCH_CHECK(B.dtype() == torch::kUInt8, "B must be uint8");
    TORCH_CHECK(lut.dtype() == torch::kInt32, "LUT must be int32");
    
    TORCH_CHECK(A.dim() == 3, "A must be 3D tensor [B, CKK, L]");
    TORCH_CHECK(B.dim() == 2, "B must be 2D tensor [O, CKK]");
    TORCH_CHECK(lut.dim() == 1, "LUT must be 1D tensor");
    
    TORCH_CHECK(A.size(1) == B.size(1), "A[B, CKK, L] and B[O, CKK] must have matching CKK dimension");
    TORCH_CHECK(lut.size(0) == 256 * 256, "LUT must be 256 * 256");
    
    const int64_t batch = A.size(0);   // B
    const int64_t CKK = A.size(1);     // CKK (K dimension for matmul)
    const int64_t L = A.size(2);       // L
    const int64_t O = B.size(0);       // O
    
    // Make tensors contiguous if needed
    auto A_contig = A.contiguous();
    auto B_contig = B.contiguous();
    
    // Create output tensor [B, O, L]
    auto C = torch::zeros({batch, O, L}, torch::TensorOptions().dtype(torch::kInt32));
    
    // Get raw pointers
    const uint8_t* A_ptr = A_contig.data_ptr<uint8_t>();
    const uint8_t* B_ptr = B_contig.data_ptr<uint8_t>();
    const int32_t* lut_ptr = lut.data_ptr<int32_t>();
    int32_t* C_ptr = C.data_ptr<int32_t>();
    
    // Strides
    const int64_t A_batch_stride = CKK * L;  // stride for batch dimension in A
    const int64_t C_batch_stride = O * L;    // stride for batch dimension in C
    
    // For each batch: C[b, :, :] = B @ A[b, :, :]
    // B[O, CKK] @ A[b, CKK, L] = C[b, O, L]
    // C[b, o, l] = sum_k B[o, k] * A[b, k, l]
    
    #pragma omp parallel for collapse(3) if(batch * O * L > 1000)
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t o = 0; o < O; ++o) {
            for (int64_t l = 0; l < L; ++l) {
                int32_t sum = 0;
                
                const uint8_t* B_row = B_ptr + o * CKK;              // B[o, :]
                const uint8_t* A_slice = A_ptr + b * A_batch_stride; // A[b, :, :]
                
                for (int64_t k = 0; k < CKK; ++k) {
                    // B[o, k] and A[b, k, l]
                    // uint8 范围是 [0, 255]，直接索引即可，无需偏移
                    unsigned int b_val = static_cast<unsigned int>(B_row[k]);
                    unsigned int a_val = static_cast<unsigned int>(A_slice[k * L + l]);
                    sum += lut_ptr[b_val * 256 + a_val];
                }
                
                C_ptr[b * C_batch_stride + o * L + l] = sum;
            }
        }
    }
    
    return C;
}

TORCH_LIBRARY_FRAGMENT(approxtorch, m){
    // m.def("im2col_uint8(Tensor input, int k_h, int k_w, int stride_h, int stride_w, int padding_h, int padding_w, int dilation_h, int dilation_w) -> Tensor");
    m.def("bgemm_int8(Tensor A, Tensor B, Tensor lut) -> Tensor");
    m.def("bgemm_uint8(Tensor A, Tensor B, Tensor lut) -> Tensor");
}

TORCH_LIBRARY_IMPL(approxtorch, CPU, m){
    m.impl("bgemm_int8", &bgemm_int8_cpu);
    m.impl("bgemm_uint8", &bgemm_uint8_cpu);
}

}