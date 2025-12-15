# Fused RNG + Perturbation Kernels for MeZO Optimization

## Overview

This directory contains optimized CUDA and Triton kernels for zero-order optimization (MeZO). 
The key insight is that we can **fuse random number generation with perturbation application** 
to eliminate the need for storing the random tensor `z`, achieving both speed and memory benefits.

## Algorithm

Standard MeZO perturbation:
```python
z = torch.randn_like(params)  # Allocates ~1.3GB for OPT-350M
params += alpha * z            # Another kernel launch
```

Fused approach:
```cuda
// Single kernel - no z storage needed!
for each element i:
    z_i = philox_randn(seed, i)  // Generate inline
    params[i] += alpha * z_i      // Apply immediately
```

## Files

| File | Description |
|------|-------------|
| `fused_perturb.cu` | CUDA kernel implementation with Philox RNG |
| `triton_fused_perturb.py` | Triton kernel implementation |
| `benchmark_kernels.py` | Comprehensive benchmark comparing all approaches |
| `setup.py` | Build script for CUDA extension |

## Benchmark Results (OPT-350M: 331M params)

```
Method                            Time (ms)    Speedup          Memory
-------------------------------------------------------------------
cuda_ext                              1.85      2.05x    0 MB (fused) ★
pytorch_baseline                      3.81      1.00x         1263 MB
triton_v2_b2048                       4.00      0.95x    0 MB (fused)
chunked_64mb                          4.24      0.90x           64 MB
```

### Key Findings

1. **CUDA Kernel is Fastest**: 2.05x speedup over PyTorch baseline
2. **Zero Memory Overhead**: No z_flat buffer needed (saves 1.26GB for OPT-350M)
3. **Triton is Memory-Efficient but Slower**: ~0.95x due to less optimized RNG
4. **Chunked is a Good Compromise**: 0.90x speed with only 64MB overhead

## Usage

### CUDA Extension

```bash
# Build
cd cuda_kernels
python setup.py install

# Use
import fused_perturb_cuda

# Apply perturbation: params = params + alpha * randn(seed)
fused_perturb_cuda.fused_perturb(params, seed, alpha)

# Restore and update: params = params + (eps - lr*grad) * randn(seed)
fused_perturb_cuda.fused_restore_update(params, seed, eps, projected_grad, lr)
```

### Triton Kernels

```python
from triton_fused_perturb import FusedPerturbMeZO

# Create trainer
trainer = FusedPerturbMeZO(model, eps=1e-3, lr=1e-5)

# Training step
loss, grad = trainer.step(batch)
```

## Memory Comparison

For OPT-350M (331,196,416 parameters):

| Approach | Memory Overhead | Speed | Use Case |
|----------|-----------------|-------|----------|
| Original MeZO | ~1.26 GB (GC-dependent) | 1.0x | Baseline |
| Flat Buffer | +1.26 GB (fixed) | 1.5x | Speed-focused |
| **CUDA Fused** | **0 GB** | **2.0x** | **Optimal** |
| Triton Fused | 0 GB | 0.95x | Easy to modify |
| Chunked (64MB) | +64 MB | 0.9x | Memory-constrained |

## Technical Details

### Philox RNG

Both CUDA and Triton kernels use Philox4x32-10 RNG (same as PyTorch/cuRAND):
- **Deterministic**: Same seed + index = same random value
- **Reproducible**: Can regenerate z for -ε perturbation without storing it
- **High quality**: Passes all standard randomness tests

### CUDA Kernel Optimizations

1. **Vectorized loads/stores**: Process 4 elements per thread using `float4`
2. **Coalesced memory access**: Sequential thread access patterns
3. **Fast math**: `--use_fast_math` for faster transcendentals
4. **Philox inline**: RNG directly in kernel without cuRAND state overhead

### Triton Kernel Notes

- Uses `tl.randn(seed, offsets)` which calls Philox internally
- Autotuning helps find optimal block size for different tensor sizes
- Less optimized than CUDA due to higher-level abstraction

## Running Benchmarks

```bash
cd cuda_kernels

# Benchmark all approaches
python benchmark_kernels.py

# Benchmark Triton kernels only
python triton_fused_perturb.py
```

## Integration with MeZO Training

See `../minimal_kernel_mezo.py` for full integration examples showing how these
kernels can be used in a complete zeroth-order optimization training loop.

## References

- [MeZO: Fine-Tuning Language Models with Just Forward Passes](https://arxiv.org/abs/2305.17333)
- [Philox RNG](https://www.thesalmons.org/john/random123/papers/random123sc11.pdf)
- [Triton Documentation](https://triton-lang.org/main/index.html)
