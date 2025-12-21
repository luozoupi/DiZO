"""
Setup script for DiZO Fused Kernels CUDA Extension

Build with:
    python setup.py install
    
Or for development:
    python setup.py develop
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Get CUDA architecture from environment or default to sm_80 (Ampere)
cuda_arch = os.environ.get('TORCH_CUDA_ARCH_LIST', '8.0')
if cuda_arch:
    # Convert to nvcc format (e.g., "8.0" -> "sm_80")
    arch_list = cuda_arch.split(';')
    nvcc_archs = [f'-arch=sm_{arch.replace(".", "")}' for arch in arch_list]
else:
    nvcc_archs = ['-arch=sm_80']  # Default to Ampere

setup(
    name='dizo_fused_kernels_cuda',
    ext_modules=[
        CUDAExtension(
            name='dizo_fused_kernels_cuda',
            sources=['dizo_fused_kernels.cu'],
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17'],
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    '-lineinfo',  # For profiling
                    '-DUSE_TORCH',
                    '--expt-relaxed-constexpr',
                    '--expt-extended-lambda',
                ] + nvcc_archs
            }
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    py_modules=['dizo_fused_kernels_cuda_wrapper'],
)
