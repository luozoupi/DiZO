# Building CUDA Kernels for DiZO zo_forward

## Prerequisites

- CUDA toolkit (11.0+)
- PyTorch with CUDA support
- Python 3.10+
- Setuptools

## Build Instructions

### Option 1: Install (Recommended)

```bash
cd /home/luo00466/luo00466_data1/DiZO_old/DiZO/large_models/cuda_kernels/zo_foward_wise
conda activate py310
python setup.py install
```

### Option 2: Develop Mode

```bash
cd /home/luo00466/luo00466_data1/DiZO_old/DiZO/large_models/cuda_kernels/zo_foward_wise
conda activate py310
python setup.py develop
```

### Option 3: Build for Specific CUDA Architecture

```bash
# For H200 (Hopper, compute capability 9.0)
TORCH_CUDA_ARCH_LIST="9.0" python setup.py install

# For A100 (Ampere, compute capability 8.0)
TORCH_CUDA_ARCH_LIST="8.0" python setup.py install

# For multiple architectures
TORCH_CUDA_ARCH_LIST="8.0;9.0" python setup.py install
```

## Verify Installation

```python
import dizo_fused_kernels_cuda
print("CUDA kernels installed successfully!")
```

## Usage

The CUDA kernels are automatically used via the wrapper:

```python
from dizo_fused_kernels_cuda_wrapper import (
    fused_compute_norms_cuda,
    fused_apply_constraints_cuda,
    fused_reverse_constraints_cuda,
    fused_update_gamma_cuda,
)

# These functions automatically use CUDA kernels if available,
# otherwise fall back to Triton kernels
```

## Troubleshooting

### Build Errors

1. **CUDA not found**: Ensure CUDA is in PATH
   ```bash
   which nvcc
   export PATH=/usr/local/cuda/bin:$PATH
   ```

2. **PyTorch CUDA mismatch**: Ensure PyTorch CUDA version matches system CUDA
   ```python
   import torch
   print(torch.version.cuda)
   ```

3. **Architecture mismatch**: Set TORCH_CUDA_ARCH_LIST for your GPU
   ```bash
   # Check GPU compute capability
   nvidia-smi --query-gpu=compute_cap --format=csv
   ```

### Runtime Errors

If CUDA kernels fail at runtime, the wrapper automatically falls back to Triton kernels. Check the console for warnings.

## Performance

CUDA kernels typically provide:
- **2-3x speedup** over Triton kernels for large parameter groups
- **Better memory bandwidth utilization**
- **Lower kernel launch overhead**

Benchmark with:
```bash
python test_kernels.py  # Compares CUDA vs Triton
```
