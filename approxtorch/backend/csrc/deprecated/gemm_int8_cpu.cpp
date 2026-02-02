#include <Python.h>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

extern "C" {
  /* Creates a dummy empty _C module that can be imported from Python.
     The import from Python will load the .so consisting of this file
     in this extension, so that the TORCH_LIBRARY static initializers
     below are run. */
  PyObject* PyInit__C(void)
  {
      static struct PyModuleDef module_def = {
          PyModuleDef_HEAD_INIT,
          "_C",   /* name of module */
          NULL,   /* module documentation, may be NULL */
          -1,     /* size of per-interpreter state of the module,
                     or -1 if the module keeps state in global variables. */
          NULL,   /* methods */
      };
      return PyModule_Create(&module_def);
  }
}

namespace approxtorch{
torch::Tensor gemm_int8_cpu(torch::Tensor& A, 
                            torch::Tensor& B,
                            const torch::Tensor& lut)
{
    // A (M, K)
    // B (K, N)
    // lut (256, 256)
    // output (M, N)

    // A is int8
    // B is int8
    // lut is int32
    // output is int32

    int M = A.size(0);
    int N = B.size(1);
    int K = A.size(1);

    // create results tensor
    torch::Tensor output = torch::empty({M, N}, A.options().dtype(torch::kInt32));
    const int8_t* A_ptr = A.data_ptr<int8_t>();
    const int8_t* B_ptr = B.data_ptr<int8_t>();
    const int32_t* lut_ptr = lut.data_ptr<int32_t>();
    int32_t* output_ptr = output.data_ptr<int32_t>();

    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n)
        {
            int temp = 0;
            for (int k = 0; k < K; ++k)
            {
                temp += lut_ptr[256 * A_ptr[m * K + k] + B_ptr[k * N + n] + 32896];
            }
            output_ptr[m * N + n] = temp;
        }


    return output;
}


TORCH_LIBRARY_FRAGMENT(approxtorch, m){
    m.def("gemm_int8(Tensor A, Tensor B, Tensor lut) -> Tensor");
}

TORCH_LIBRARY_IMPL(approxtorch, CPU, m){
    m.impl("gemm_int8", &gemm_int8_cpu);
}

}