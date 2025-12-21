from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

cuda_arch = os.environ.get('TORCH_CUDA_ARCH_LIST', '9.0')

setup(
    name='dizo_fused_kernels_cuda_v4',
    ext_modules=[
        CUDAExtension(
            name='dizo_fused_kernels_cuda_v4',
            sources=['dizo_fused_kernels_v4.cu'],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '-use_fast_math',
                    '-lineinfo',
                    f'-arch=sm_{cuda_arch.replace(".", "")}',
                    '-DUSE_TORCH',
                ]
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
