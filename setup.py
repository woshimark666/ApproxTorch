from setuptools import setup, Extension, find_packages
from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension, include_paths



cuda_functions = CUDAExtension('approxtorch.backend._C',[
        # dummy module
        './approxtorch/backend/csrc/dummy_module.cpp',
    
        # cpu backend
        './approxtorch/backend/csrc/cpu/im2col.cpp',
        './approxtorch/backend/csrc/cpu/gemm.cpp',
        './approxtorch/backend/csrc/cpu/bgemm.cpp',
        # cuda backednd
        './approxtorch/backend/csrc/cuda/im2col.cu',
        './approxtorch/backend/csrc/cuda/gemm.cu',
        './approxtorch/backend/csrc/cuda/gemm_navie.cu',
        './approxtorch/backend/csrc/cuda/bgemm.cu',
    ],                   
    include_dirs = ['./approxtorch/backend/csrc/cuda'],
    extra_compile_args={'nvcc': ['-arch=native', '-std=c++17', "-O3"],
                        "cxx": ["-O3","-fdiagnostics-color=always",
                                "-DPy_LIMITED_API=0x03090000",  # min CPython version 3.9
                                ]},
)


setup(
    name="approxtorch",
    version="2.0",
    description="a simulation framework for 8-bit (u)signed approximate multiplier in CNNs",
    packages=find_packages(),
    ext_modules=[cuda_functions],    # extensions to be compiled
    install_requires=['torch'],
    cmdclass={'build_ext': BuildExtension},
    options={'bdist_whell': {"py_limited_api": "cp39"}}
)

# setup(
#     ext_modules=[cuda_functions],
#     cmdclass={'build_ext': BuildExtension},
# )