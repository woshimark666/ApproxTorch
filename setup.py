from setuptools import setup, Extension, find_packages
from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension, include_paths



cuda_functions = CUDAExtension('approxtorch.approx_gemm._C',[
        "./approxtorch/approx_gemm/csrc/int8_gemm.cu",
        "./approxtorch/approx_gemm/csrc/int8_batch_gemm.cu",
        "./approxtorch/approx_gemm/csrc/gemm_int8_cpu.cpp",
        "./approxtorch/approx_gemm/csrc/gemm_int8_naive.cu",
    ],                   
    include_dirs = ['approxtorch/approx_gemm/csrc'],
    extra_compile_args={'nvcc': ['-arch=native', '-std=c++17', "-O3"],
                        "cxx": ["-O3","-fdiagnostics-color=always",
                                "-DPy_LIMITED_API=0x03090000",  # min CPython version 3.9
                                ]},
)


setup(
    name="approxtorch",
    version="1.0",
    description="a pytorch extension for 8-bit int based conv2d and gemm",
    packages=find_packages(),
    ext_modules=[cuda_functions],    # extensions to be compiled
    install_requires=['torch'],
    cmdclass={'build_ext': BuildExtension},
    options={'bdist_whell': {"py_limited_api": "cp39"}}
)

