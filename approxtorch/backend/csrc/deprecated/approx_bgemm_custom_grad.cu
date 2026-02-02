#include "gemm.cuh"
namespace approxtorch{

__global__ void bgemm_custom_grad_feature_uint8(
    const float* __restrict__ upstream_grad,
    const uint8_t* __restrict__ feature, // shape [B, CKK, L]
    const uint8_t* __restrict__ weight, // shape [O, CKK]
    const float* __restrict__ grad_lut_A,
    float* 
)




} // namespace approxtorch