"""
Build script for optimized Philox CUDA kernels.

Usage:
    # Install as package:
    cd kernel_optimization && python setup_v2_cuda.py install

    # Build in-place (for development):
    cd kernel_optimization && python setup_v2_cuda.py build_ext --inplace

    # Or use JIT compilation (no setup.py needed):
    from torch.utils.cpp_extension import load
    philox_v2_cuda_ext = load(
        name='philox_v2_cuda_ext',
        sources=['philox_v2_cuda.cu'],
        extra_cuda_cflags=['-O3', '--use_fast_math'],
    )
"""

import os
import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Detect GPU architecture
_SM_FLAG = '-arch=sm_120'  # Blackwell (RTX PRO 6000)
try:
    cap = torch.cuda.get_device_capability(0)
    sm = f'{cap[0]}{cap[1]}0' if cap[1] >= 10 else f'{cap[0]}{cap[1]}'
    _SM_FLAG = f'-gencode=arch=compute_{sm},code=sm_{sm}'
except Exception:
    pass

setup(
    name='philox_v2_cuda_ext',
    ext_modules=[
        CUDAExtension(
            name='philox_v2_cuda_ext',
            sources=[
                os.path.join(os.path.dirname(__file__) or '.', 'philox_v2_cuda.cu'),
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    _SM_FLAG,
                    '--use_fast_math',
                    '-lineinfo',          # Keep line info for NCU profiling
                    '--ptxas-options=-v',  # Show register/smem usage at compile time
                ],
            },
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)
