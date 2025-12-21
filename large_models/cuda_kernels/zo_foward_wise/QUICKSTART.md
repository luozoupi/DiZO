# Quick Start Guide: DiZO zo_forward Optimization

## Overview

This optimization reduces kernel launch overhead and CPU-GPU synchronization in `zo_forward` by:
1. Flattening all parameters into a single tensor
2. Using fused Triton kernels for batch operations
3. Minimizing synchronization points

## Quick Test

```bash
cd /home/luo00466/luo00466_data1/DiZO_old/DiZO/large_models/cuda_kernels/zo_foward_wise
conda activate py310
CUDA_VISIBLE_DEVICES=6 python test_kernels.py
```

## Integration Steps

### Option 1: Patch Existing Trainer (Easiest)

```python
from cuda_kernels.zo_foward_wise.integration_example import patch_trainer_zo_forward

# After creating dizo_trainer instance
trainer = dizo_trainer(...)
patch_trainer_zo_forward(trainer)

# Now trainer.dizo_zo_iters uses optimized kernels automatically
```

### Option 2: Manual Integration

Modify `trainer.py`, in the `dizo_trainer` class:

```python
from cuda_kernels.zo_foward_wise import OptimizedDiZO

class dizo_trainer():
    def __init__(self, ...):
        # ... existing code ...
        self._optimized_dizo = None  # Will be initialized lazily
    
    def dizo_zo_iters(self, model, base_model, apply=False, args=None):
        if not apply:
            # ... existing data loading code ...
            
            # Initialize optimized DiZO if needed
            if self._optimized_dizo is None:
                self._optimized_dizo = OptimizedDiZO(
                    self.dizo, model, base_model
                )
            
            # Use optimized version
            self._optimized_dizo.zo_forward_optimized(
                new=model,
                pre_trained=base_model,
                x=data,
                apply=False,
                args=args
            )
        else:
            if self._optimized_dizo is None:
                self._optimized_dizo = OptimizedDiZO(
                    self.dizo, model, self.pre_trained
                )
            self._optimized_dizo.zo_forward_optimized(
                new=model,
                pre_trained=self.pre_trained,
                x=None,
                apply=True,
                args=args
            )
```

## Expected Performance

Based on profiler analysis (OPT-13B):
- **Kernel launches**: 54,990 → ~100 (550x reduction)
- **CPU-GPU sync**: 512ms → ~200ms (2.5x reduction)
- **Overall zo_forward**: ~1.5-2x speedup

## Troubleshooting

### Import Errors
Make sure the path is correct:
```python
import sys
sys.path.append('/home/luo00466/luo00466_data1/DiZO_old/DiZO/large_models/cuda_kernels')
```

### CUDA Out of Memory
The optimization uses pre-allocated buffers. If OOM occurs:
- Reduce batch size
- Use gradient checkpointing
- Consider chunked processing for very large models

### Numerical Differences
Small numerical differences (< 1e-4) are expected due to:
- Different reduction order in norm computation
- Floating point associativity

If differences are larger, check:
- Parameter flattening is correct
- Constraint values match

## Files Created

- `param_utils.py`: Parameter flattening utilities
- `dizo_fused_kernels.py`: Triton kernels
- `optimized_zo_forward.py`: Optimized implementation
- `test_kernels.py`: Test suite
- `integration_example.py`: Integration examples

## Next Steps

1. Run tests to verify correctness
2. Benchmark on your model
3. Integrate into training loop
4. Profile to measure actual speedup

