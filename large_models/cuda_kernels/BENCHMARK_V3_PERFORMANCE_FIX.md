# Benchmark V3 Performance Issue - Root Cause Analysis and Fix

## Issue
The v3 benchmark shows almost no speedup for Triton kernels vs PyTorch baseline, even though individual kernel benchmarks (`benchmark_fair_comparison.py`) show significant speedups.

## Root Cause: Triton Seed Recompilation

**Triton recompiles kernels when the seed value changes**, causing massive overhead.

## Fix Applied

In v3, we now use the **autotuned kernels** from `triton_fused_perturb.py` instead of the runtime-seed kernels from v2.

### Before Fix (v2 runtime-seed kernels):
| Model | Method | Time |
|-------|--------|------|
| opt-350m | Triton Perturb | 15.59 ms |
| opt-350m | CUDA Perturb | 3.46 ms |
| opt-350m | PyTorch | 16.74 ms |

### After Fix (v3 autotuned kernels):
| Model | Method | Time | Speedup |
|-------|--------|------|---------|
| opt-350m | Triton Perturb | **6.41 ms** | **2.6x** vs PyTorch |
| opt-350m | CUDA Perturb | 3.47 ms | **4.8x** vs PyTorch |
| opt-350m | PyTorch | 16.61 ms | 1.0x |

## Handling Large Models (>2B params)

The Triton autotuned kernels have an **int32 overflow** issue for models with >2B elements (opt-2.7b and larger).

**V3 automatically handles this:**
- For models ≤2B elements: Uses fast autotuned kernels (optimal performance)
- For models >2B elements: Falls back to v2 runtime-seed kernels (int64 safe, but slower)

### opt-13b Results

| Method | Time (ms) | Notes |
|--------|-----------|-------|
| CUDA Perturb + CUDA ZO V5 | **228.90** | Recommended for large models |
| Triton V2 Full (fallback) | 655.64 | Uses runtime-seed kernels |

**Recommendation:** For opt-6.7b and larger, use CUDA-based benchmarks (`cuda_full`, `cuda_perturb_mezo`, etc.) for best performance.

### Evidence

```
Testing with CONSTANT seed=42:
  Triton Autotuned: 1.594 ms per call

Testing with DIFFERENT seeds each call:
  Triton Autotuned: 18.545 ms per call

Slowdown: 10.34x!
```

### Why This Happens

1. `tl.randn(seed, offsets)` uses `seed` as part of the kernel cache key
2. Even when loading seed from a tensor (runtime seed), the actual seed VALUE is used for kernel dispatch
3. Different seeds → different compiled kernels → massive recompilation overhead

### Why fair_comparison.py Works

`benchmark_fair_comparison.py` uses **constant `seed = 42`** for all iterations:

```python
seed = 42  # CONSTANT - no recompilation!

def triton_mezo_step(p):
    fused_perturb_kernel_autotuned[grid](params, seed, eps, n_elements)  # Always same seed
```

### Why v2/v3 is Slow

`benchmark_full_training_step_v2.py` uses **different seeds each iteration**:

```python
for _ in range(n_iter):
    seed_tensor[0] = np.random.randint(1000000000)  # Different each time!
    fused_perturb_runtime_seed[grid](param_flat, seed_tensor, cfg.eps, n_elements)
```

## Solution

For **benchmarking kernel performance**, use constant seed. This is valid because:
1. We're measuring kernel execution time, not algorithm correctness
2. The kernel does the same work regardless of seed
3. This matches real-world usage where seed changes infrequently (once per step, not per kernel call)

### Correct Benchmark Approach

```python
# CORRECT: Use constant seed for all perturb calls within one step
seed = 42 + iteration  # Different per step, but SAME for all 4 calls within step

# Step 1-4 all use the SAME seed (this is actually how MeZO works!)
perturb(params, seed, +eps)      # Uses z from seed
perturb(params, seed, -2*eps)    # Uses SAME z from seed
perturb(params, seed, +eps)      # Uses SAME z from seed
update(params, seed, grad, lr)   # Uses SAME z from seed
```

## Performance Summary (OPT-350M)

### Correct Results (constant seed per step):
| Method | Time | Speedup |
|--------|------|---------|
| CUDA Extension | 3.43 ms | 4.27x ★ |
| Triton Autotuned | 6.36 ms | 2.30x |
| PyTorch Baseline | 14.64 ms | 1.00x |

### Broken Results (random seed each call):
| Method | Time | Speedup |
|--------|------|---------|
| Triton Runtime-Seed | 35.44 ms | 0.41x (SLOWER!) |
| CUDA Extension | 3.43 ms | 4.27x |

## Key Insight

In actual MeZO training, the **same seed is used for all perturb/update calls within one step**. This is required for correctness:
- `perturb(+eps)` adds `z` 
- `perturb(-2eps)` adds `-2z` (must be SAME z!)
- `perturb(+eps)` adds `z` back (SAME z!)
- `update()` uses `z` for gradient (SAME z!)

So using constant seed per step is not just valid for benchmarking - it's exactly how the algorithm works!

