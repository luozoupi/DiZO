"""
Setup script for Single-GPU Optimization CUDA Extension

Build with:
    python setup_single_gpu_opt.py install

Or for development:
    python setup_single_gpu_opt.py develop

For specific GPU architecture:
    TORCH_CUDA_ARCH_LIST="8.0 8.9 9.0" python setup_single_gpu_opt.py install
"""

import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Detect GPU architecture if not specified
cuda_arch_flags = []
try:
    import torch
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()
        arch = f"{capability[0]}{capability[1]}"
        # Blackwell is sm_100 but falls back to sm_90
        if capability[0] >= 9:
            cuda_arch_flags = ['-arch=sm_90']
        elif capability[0] == 8:
            cuda_arch_flags = [f'-arch=sm_{arch}']
        else:
            cuda_arch_flags = ['-arch=sm_80']  # Default to Ampere
except:
    cuda_arch_flags = ['-arch=sm_80']

print(f"Building with CUDA arch flags: {cuda_arch_flags}")

setup(
    name='single_gpu_opt_cuda',
    version='1.0.0',
    description='CUDA kernels for single-GPU ZO optimization',
    ext_modules=[
        CUDAExtension(
            name='single_gpu_opt_cuda',
            sources=['single_gpu_optimizations.cu'],
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17'],
                'nvcc': [
                    '-O3',
                    *cuda_arch_flags,
                    '-DUSE_TORCH',
                    '--use_fast_math',
                    '-lineinfo',  # For profiling with nsys/ncu
                    '--ptxas-options=-v',  # Show register usage
                    '-Xcompiler', '-fPIC',
                ]
            }
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
