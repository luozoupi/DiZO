"""
Setup script for DiZO Fused Kernels CUDA Extension V2

Optimized version with:
- Float4 vectorized memory access
- Multi-block parallelism
- Warp-level reductions

Build: TORCH_CUDA_ARCH_LIST="9.0" python setup_v2.py install
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

cuda_arch = os.environ.get('TORCH_CUDA_ARCH_LIST', '9.0')
arch_list = cuda_arch.split(';')
nvcc_archs = [f'-arch=sm_{arch.replace(".", "")}' for arch in arch_list]

setup(
    name='dizo_fused_kernels_cuda_v2',
    ext_modules=[
        CUDAExtension(
            name='dizo_fused_kernels_cuda_v2',
            sources=['dizo_fused_kernels_v2.cu'],
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17'],
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    '-lineinfo',
                    '-DUSE_TORCH',
                    '--expt-relaxed-constexpr',
                    '--expt-extended-lambda',
                    '-Xptxas', '-v',  # Show register usage
                ] + nvcc_archs
            }
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)
