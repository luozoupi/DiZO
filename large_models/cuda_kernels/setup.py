"""
Setup script for CUDA Fused Perturb Extension

Build with:
    python setup.py install
    
Or for development:
    python setup.py develop
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='fused_perturb_cuda',
    ext_modules=[
        CUDAExtension(
            name='fused_perturb_cuda',
            sources=['fused_perturb.cu'],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '-arch=sm_80',  # Ampere architecture (A100/A6000)
                    '-DUSE_TORCH',
                    '--use_fast_math',
                    '-lineinfo',  # For profiling
                ]
            }
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
