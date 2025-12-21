# DiZO zo_forward Optimization

This directory contains optimized CUDA/Triton kernels for the `zo_forward` method in DiZO, designed to reduce kernel launch overhead and CPU-GPU synchronization bottlenecks.

## Overview

The optimization focuses on:
1. **Batching operations**: Flatten all parameters into a single tensor to enable batch operations
2. **Fused kernels**: Combine multiple operations into single kernel launches
3. **Reduced synchronization**: Minimize CPU-GPU synchronization points

## Key Components

### 1. Parameter Flattening (`param_utils.py`)
- `ParameterFlattener`: Manages flattening/unflattening of model parameters
- Enables batch operations across all parameters

### 2. Fused Kernels (`dizo_fused_kernels.py`)
- `fused_compute_norms`: Batch L2 norm computation
- `fused_apply_constraints`: Fused constraint application
- `fused_reverse_constraints`: Fused constraint reversal
- `fused_perturb_gamma`: Gamma perturbation (uses PyTorch for small arrays)
- `fused_update_gamma`: Gamma update with clipping

### 3. Optimized Implementation (`optimized_zo_forward.py`)
- `OptimizedDiZO`: Drop-in replacement for `DiZO.zo_forward`
- Uses fused kernels to minimize overhead

## Usage

### Basic Usage

```python
from optimized_zo_forward import OptimizedDiZO

# Initialize
optimized_dizo = OptimizedDiZO(dizo_instance, model, anchor_model)

# Use optimized zo_forward
optimized_dizo.zo_forward_optimized(
    new=model,
    pre_trained=anchor_model,
    x=data,
    apply=False,
    args=args
)
```

### Integration with Trainer

Modify `trainer.py` to use optimized version:

```python
# In dizo_trainer.__init__ or similar
from cuda_kernels.zo_foward_wise.optimized_zo_forward import OptimizedDiZO

self.optimized_dizo = OptimizedDiZO(self.dizo, model, base_model)

# Replace zo_forward calls
self.optimized_dizo.zo_forward_optimized(model, base_model, x=data, args=args)
```

## Testing

Run the test suite:

```bash
cd /home/luo00466/luo00466_data1/DiZO_old/DiZO/large_models/cuda_kernels/zo_foward_wise
conda activate py310
CUDA_VISIBLE_DEVICES=6 python test_kernels.py
```

## Performance Expectations

Based on profiler analysis:
- **Kernel launch overhead**: 25.48% → ~5% (5x reduction)
- **CPU-GPU sync**: 25.72% → ~10% (fewer sync points)
- **Memory copies**: 18.20% → ~5% (fused operations)
- **Overall speedup**: 1.5-2x for `zo_forward` operation

## Dependencies

- PyTorch (with CUDA support)
- Triton
- CUDA-capable GPU (tested on GPU 6)

## Notes

- The gamma perturbation uses PyTorch operations since the number of constraints is typically small (< 1000)
- All kernels operate on flattened parameter tensors
- Memory is pre-allocated to avoid allocation overhead

## Future Improvements

1. CUDA kernels for even better performance (if Triton becomes bottleneck)
2. Stream-based overlapping of operations
3. Further fusion of constraint application + forward pass

