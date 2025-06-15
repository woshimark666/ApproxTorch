// approx_gemm_int8_double_buffering.cu
// ------------------------------------------------------------
// High-performance int8 (A[M,K] * B[N,K]^T) → int32 C[M,N]
// Featuring:
//   • Double-buffered global→shared transfer with cp.async (Ampere+)
//   • Register tiling (4×4 per thread, 128×128×32 CTA tile)
//   • Warp-local LUT cache (32 rows × 256 cols per CTA-row) in shared mem
//   • PyTorch extension entry point (build with setup.py)
// ------------------------------------------------------------
// Compile example (stand-alone):
// nvcc -O3 -std=c++20 -arch=sm_80 -lineinfo \
//      -Xptxas -v approx_gemm_int8_double_buffering.cu \
//      --compiler-options "-fPIC" -shared -o libapprox_gemm.so
// ------------------------------------------------------------

#include "gemm.cuh"


// approx_gemm_int8_double_buffering.cu  – ILLEGAL ACCESS FIXED
// ------------------------------------------------------------
// 1. Keeps all previous optimisations (double-buffer cp.async, 4×4 register tile)
// 2. **Fixes illegal shared-mem access** by:
//    • Copying full 32-byte rows for A/B in cp.async_bulk                
//    • Storing only 32 LUT rows in shared but index via (key & 31) slot  
//      so we never exceed allocated 32×256 int16 panel per warp-row
// 3. Adds cudaFuncSetAttribute(MaxDynamicSharedMemorySize, smem_bytes)   
//    so launch no longer fails when smem > 48 KB.
// ------------------------------------------------------------

namespace approxtorch{
// ---------------- Tunables ----------------
#define TILE_M 128
#define TILE_N 128
#define TILE_K  32
#define WARP_SIZE 32
#define WARPS_PER_M (TILE_M/32)   // 4
#define WARPS_PER_N (TILE_N/32)   // 4
#define THREADS_PER_BLOCK (WARPS_PER_M*WARPS_PER_N*WARP_SIZE) // 512

// ---------------- Small helpers ----------------
__device__ __forceinline__ int32_t lut_fetch_gmem(const int16_t* lut, int8_t a, int8_t b){
    return (int32_t)__ldg(lut + (((int)a + 128)<<8) + ((int)b + 128)); // ((a+128)*256 + b+128)
}

// ---------------- Asynchronous row copy (32 B) ----------------
__device__ __forceinline__ void cp_async_row32(void* dst, const void* src){
    // First 16 bytes
    asm volatile ("cp.async.cg.shared.global [%0], [%1], 16;" :: "l"(dst), "l"(src));
    // Second 16 bytes
    asm volatile ("cp.async.cg.shared.global [%0], [%1], 16;" :: "l"((char*)dst + 16), "l"((char*)src + 16));
}

// ---------------- Kernel ----------------
__global__ void gemm_int8_lut_db_kernel(
        const int8_t*  __restrict__ A,
        const int8_t*  __restrict__ B,
        const int16_t* __restrict__ lut,
        int32_t*       __restrict__ C,
        int M,int N,int K){

    // ---- CTA-level coords ----
    const int base_m = blockIdx.y * TILE_M;
    const int base_n = blockIdx.x * TILE_N;

    // ---- Warp / lane ----
    const int warp_id = threadIdx.x / WARP_SIZE;   // 0..15
    const int lane_id = threadIdx.x & 31;          // 0..31
    const int warp_m  = warp_id / WARPS_PER_N;     // 0..3
    const int warp_n  = warp_id % WARPS_PER_N;     // 0..3

    // ---- Per-thread 4×4 tile ----
    const int tm4 = (lane_id/8) & 3;      // 0..3 (row-slot within warp)
    const int tn4 =  lane_id & 7;         // 0..7 (col-slot)

    const int row_start = base_m + warp_m*32 + tm4*4;  // first of 4 rows this thread owns
    const int col_start = base_n + warp_n*32 + tn4*4;  // first of 4 cols this thread owns

    int32_t acc[4][4] = {0};

    // ---- Shared memory layout ----
    extern __shared__ char smem[];
    int8_t*  As = (int8_t*)smem;                                    // 2*128*32   =  8 192 B
    int8_t*  Bs = As + 2*TILE_M*TILE_K;                             // +8 192 B = 16 384 B
    int16_t* lut_panel = (int16_t*)(Bs + 2*TILE_N*TILE_K);          // 4*32*256*2 = 65 536 B

    auto A_buf = [&](int b){ return As + b*TILE_M*TILE_K; };
    auto B_buf = [&](int b){ return Bs + b*TILE_N*TILE_K; };

    // ---- Load first A/B tiles (buf=0) ----
    const int lda = K;
    const int ldb = K;

    for(int r = lane_id; r < TILE_M; r += WARP_SIZE){
        const int8_t* srcA = A + (base_m + r) * lda;
        cp_async_row32(A_buf(0) + r*TILE_K, srcA);
    }
    for(int r = lane_id; r < TILE_N; r += WARP_SIZE){
        const int8_t* srcB = B + (base_n + r) * ldb;
        cp_async_row32(B_buf(0) + r*TILE_K, srcB);
    }
    asm volatile ("cp.async.commit_group;"::);
    asm volatile ("cp.async.wait_group 0;"::);
    __syncthreads();

    // ---- K-loop ----
    int buf=0;
    for(int k0=0;k0<K;k0+=TILE_K){
        // Preload next tile (buf^1) unless last
        if(k0+TILE_K<K){
            int next = buf^1;
            // copy rows – A
            for(int r=lane_id; r<TILE_M; r+=WARP_SIZE){
                const int8_t* srcA = A + (base_m + r)*lda + (k0+TILE_K);
                cp_async_row32(A_buf(next)+r*TILE_K, srcA);
            }
            // copy rows – B
            for(int r=lane_id; r<TILE_N; r+=WARP_SIZE){
                const int8_t* srcB = B + (base_n + r)*ldb + (k0+TILE_K);
                cp_async_row32(B_buf(next)+r*TILE_K, srcB);
            }
            asm volatile("cp.async.commit_group;"::);
        }

        __syncthreads(); // current buf ready
        const int8_t* At = A_buf(buf);
        const int8_t* Bt = B_buf(buf);

        // ---- unroll over kk ----
        #pragma unroll
        for(int kk=0;kk<TILE_K;++kk){
            // gather a & b fragments
            int8_t a_frag[4];
            int8_t b_frag[4];
            #pragma unroll
            for(int i=0;i<4;++i)
                a_frag[i] = At[(tm4*4+i)*TILE_K + kk];
            #pragma unroll
            for(int j=0;j<4;++j)
                b_frag[j] = Bt[(tn4*4+j)*TILE_K + kk];

            // ---- Warp-local LUT cache in shared (32 rows) ----
            int a_key0 = (int)a_frag[0] + 128;
            int row_slot = a_key0 & 31;          // maps 0-255 → 0-31
            if(lane_id < 256)
                lut_panel[(warp_m*32 + row_slot)*256 + lane_id] = __ldg(lut + (a_key0<<8) + lane_id);
            __syncwarp();

            #pragma unroll
            for(int i=0;i<4;++i){
                int a_key = (int)a_frag[i] + 128;
                int slot   = a_key & 31;
                const int16_t* row_ptr = lut_panel + (warp_m*32 + slot)*256;
                #pragma unroll
                for(int j=0;j<4;++j){
                    acc[i][j] += (int32_t)row_ptr[(int)b_frag[j]+128];
                }
            }
        }

        buf ^= 1;
        asm volatile ("cp.async.wait_group 0;"::);
        __syncthreads();
    }

    // ---- Writeback ----
    for(int i=0;i<4;++i){
        int row = row_start + i;
        if(row < M){
            int32_t* Crow = C + row*N;
            #pragma unroll
            for(int j=0;j<4;++j){
                int col = col_start + j;
                if(col < N) Crow[col] = acc[i][j];
            }
        }
    }
}

// ---------------- Launcher ----------------
static void launch_gemm(torch::Tensor A, torch::Tensor B, torch::Tensor lut, torch::Tensor C){
    const int M=A.size(0), K=A.size(1), N=B.size(0);
    dim3 block(THREADS_PER_BLOCK);
    dim3 grid((N+TILE_N-1)/TILE_N,(M+TILE_M-1)/TILE_M);
    size_t smem_bytes = (2*TILE_M*TILE_K + 2*TILE_N*TILE_K)*sizeof(int8_t)
                      + WARPS_PER_M*32*256*sizeof(int16_t);

    cudaFuncSetAttribute(gemm_int8_lut_db_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_bytes));

    gemm_int8_lut_db_kernel<<<grid,block,smem_bytes,at::cuda::getCurrentCUDAStream()>>>(
        A.data_ptr<int8_t>(), B.data_ptr<int8_t>(), lut.data_ptr<int16_t>(),
        C.data_ptr<int32_t>(), M,N,K);
}

torch::Tensor gemm_int8_opt(torch::Tensor A, torch::Tensor B, torch::Tensor lut){
    TORCH_CHECK(A.dtype()==torch::kInt8 && B.dtype()==torch::kInt8);
    TORCH_CHECK(lut.dtype()==torch::kInt16 && lut.numel()==256*256);
    TORCH_CHECK(A.size(1)==B.size(1));
    auto C = torch::empty({A.size(0),B.size(0)}, torch::dtype(torch::kInt32).device(A.device()));
    launch_gemm(A,B,lut,C);
    return C;
}
//--------------------------------------------------------------
// PyBind11 registration
//--------------------------------------------------------------
TORCH_LIBRARY_FRAGMENT(approxtorch, m){
    m.def("gemm_int8_opt(Tensor A, Tensor B, Tensor lut) -> Tensor");
}


TORCH_LIBRARY_IMPL(approxtorch, CUDA, m){
    m.impl("gemm_int8_opt", &gemm_int8_opt);
}

}