from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

cuda_arch = os.environ.get('TORCH_CUDA_ARCH_LIST', '9.0')

setup(
    name='dizo_fused_kernels_cuda_v5',
    ext_modules=[
        CUDAExtension(
            name='dizo_fused_kernels_cuda_v5',
            sources=['dizo_fused_kernels_v5.cu'],
            extra_compile_args={
                'cxx': ['-O3', '-march=native'],
                'nvcc': [
                    '-O3',
                    '-use_fast_math',
                    '--expt-relaxed-constexpr',
                    '-lineinfo',
                    f'-arch=sm_{cuda_arch.replace(".", "")}',
                    '-DUSE_TORCH',
                    '-Xptxas', '-dlcm=ca',  # Cache all loads
                    '-Xptxas', '-dscm=wb',  # Write-back cache
                ]
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
