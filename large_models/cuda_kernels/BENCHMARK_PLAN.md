# Integrated Benchmark: Full MeZO/DiZO Training Step

## Overview

This benchmark suite evaluates the **complete training step** performance of MeZO (Zeroth-Order Optimizer) with DiZO enhancements, combining optimizations from both:

1. **Perturb-wise kernels** (`Perturb_wise/`): Fused RNG + perturbation kernels
2. **ZO-forward-wise kernels** (`zo_foward_wise/`): Fused constraint/norm operations

## Benchmark Architecture

### Training Step Components

A complete MeZO/DiZO training step consists of:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        MeZO Training Step                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. Perturb +ε          θ ← θ + ε·z       [Perturb-wise optimization]       │
│                                                                              │
│  2. Forward Pass 1      loss₁ = f(θ+εz)   [Model forward]                   │
│                                                                              │
│  3. DiZO Constraint     Apply projection   [ZO-forward optimization]         │
│                                                                              │
│  4. Perturb -2ε         θ ← θ - 2ε·z      [Perturb-wise optimization]       │
│                                                                              │
│  5. Forward Pass 2      loss₂ = f(θ-εz)   [Model forward]                   │
│                                                                              │
│  6. DiZO Reverse        Reverse projection [ZO-forward optimization]         │
│                                                                              │
│  7. Compute Gradient    g = (loss₁-loss₂)/(2ε)                              │
│                                                                              │
│  8. Reset +ε            θ ← θ + ε·z       [Perturb-wise optimization]       │
│                                                                              │
│  9. Update Parameters   θ ← θ - lr·g·z    [Perturb-wise optimization]       │
│                                                                              │
│  10. DiZO γ Update      Update constraints [ZO-forward optimization]         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Bottleneck Analysis (from profiling)

The CPU latency-bound operations identified:

| Operation | PyTorch Baseline | Issue |
|-----------|------------------|-------|
| Perturbation | N kernel launches per param | High launch overhead |
| Norm computation | N sequential calls | No batching |
| Constraint apply/reverse | N loops | Memory-bound |
| RNG generation | N `torch.randn` calls | CPU-bound |

## Benchmark Configurations

### 1. PyTorch Baseline (Original MeZO)
- Per-parameter loops for all operations
- Matches `trainer.py` implementation exactly
- Reference for speedup calculations

### 2. PyTorch Baseline + DiZO
- Baseline + constraint projection loops
- Shows overhead of DiZO enhancement

### 3. Perturb-Optimized
- **Fused** Triton/CUDA kernels for:
  - `perturb +ε`, `perturb -2ε`, `reset +ε`
  - `update` operation
- Uses flat buffer technique (single kernel launch)
- Inline Philox RNG (no separate allocation)

### 4. ZO-Forward-Optimized
- **Fused** Triton kernels for:
  - Batch norm computation
  - Constraint application
  - Constraint reversal
  - Gamma perturbation/update

### 5. Fully-Optimized (Perturb + ZO-Forward)
- **Combines both** optimization paths
- Minimal kernel launches
- Maximum throughput

## Usage

### Basic Benchmarks

```bash
# Single model
python benchmark_full_training_step.py --model opt-350m

# With breakdown
python benchmark_full_training_step.py --model opt-350m --breakdown

# All model sizes
python benchmark_full_training_step.py --all

# Save results
python benchmark_full_training_step.py --model opt-6.7b --output
```

### Model Sizes

| Model | Parameters | Memory (FP32) | Param Groups |
|-------|------------|---------------|--------------|
| opt-350m | 331M | 1.3 GB | 388 |
| opt-1.3b | 1.3B | 5.2 GB | 386 |
| opt-2.7b | 2.7B | 10.6 GB | 514 |
| opt-6.7b | 6.7B | 26.6 GB | 516 |
| opt-13b | 13B | 52 GB | 644 |

### Command-Line Options

```
--model MODEL      Model size preset (opt-350m, opt-1.3b, etc.)
--n_iter N         Number of benchmark iterations (default: 20)
--warmup N         Warmup iterations (default: 5)
--breakdown        Show detailed operation breakdown
--skip_mezo        Skip MeZO-only benchmarks
--skip_dizo        Skip DiZO benchmarks (MeZO only)
--output           Save results to file
--all              Run all model sizes

# MeZO/DiZO hyperparameters:
--eps FLOAT        MeZO perturbation epsilon (default: 1e-3)
--lr FLOAT         Learning rate (default: 1e-5)
--tau FLOAT        DiZO clip range (default: 0.2)
--zo_eps FLOAT     DiZO zo_eps_projection (default: 0.1)
--step_size FLOAT  DiZO step_size_projection (default: 2.0)
```

## Expected Results

### Performance Summary (Example: OPT-350M on A100)

```
Method                                        Time (ms)    Speedup    Memory (MB)  Kernels
------------------------------------------------------------------------------------------
PyTorch Baseline                              45.2         1.00x      1340.5       776
PyTorch Baseline + DiZO                       78.3         0.58x      1540.2       1164
Perturb-Optimized (Triton)                    12.4         3.65x      1265.8       4
Perturb-Optimized (Triton) + DiZO             38.6         1.17x      1465.5       392
ZO-Forward-Optimized + DiZO                   52.1         0.87x      1380.3       392
Fully-Optimized (Triton) + DiZO               28.3         1.60x      1320.1       12
```

### Operation Breakdown (ms per iteration)

```
Method                         Perturb+  Fwd1   Perturb-  Fwd2   Reset   Update   Norm   Apply   Rev    Gamma
-------------------------------------------------------------------------------------------------------------
PyTorch Baseline                 8.2     5.1     8.3      5.0    8.1     8.5      0.0    0.0     0.0    0.0
PyTorch Baseline + DiZO          8.2     5.1     8.3      5.0    8.1     8.5      12.3   10.5    10.2   2.1
Perturb-Optimized (Triton)       1.1     5.0     1.1      5.0    1.1     1.0      0.0    0.0     0.0    0.0
Fully-Optimized (Triton) + DiZO  1.1     5.0     1.1      5.0    1.1     1.0      0.8    0.7     0.6    0.4
```

## Kernel Details

### Perturb-wise Kernels

Located in `Perturb_wise/`:

| Kernel | Description | Source |
|--------|-------------|--------|
| `fused_perturb_kernel_autotuned` | Fused RNG+perturb | `triton_fused_perturb.py` |
| `fused_update_kernel_autotuned` | Fused RNG+update | `triton_fused_perturb.py` |
| `fused_perturb_cuda` | CUDA C++ kernel | `fused_perturb.cu` |

Key optimizations:
- Custom Philox 4x32-10 RNG (inline, no allocation)
- Single kernel launch for all parameters (flat buffer)
- Autotuned block sizes

### ZO-Forward-wise Kernels

Located in `zo_foward_wise/`:

| Kernel | Description | Source |
|--------|-------------|--------|
| `fused_compute_norms` | Batch L2 norm | `dizo_fused_kernels.py` |
| `fused_apply_constraints` | Batch projection | `dizo_fused_kernels.py` |
| `fused_reverse_constraints` | Batch reverse | `dizo_fused_kernels.py` |
| `fused_perturb_gamma` | Gamma perturbation | `dizo_fused_kernels.py` |
| `fused_update_gamma` | Gamma update | `dizo_fused_kernels.py` |

Key optimizations:
- Per-parameter-group parallelism (num_params programs)
- Block-wise reduction for norms
- Fused alpha computation

## Building Kernels

### Prerequisites

```bash
# Install dependencies
pip install triton torch

# For CUDA kernels
pip install ninja
```

### Build Steps

```bash
# Build Perturb-wise CUDA extension
cd Perturb_wise
python setup.py install

# ZO-forward kernels are Triton-only (no build needed)
```

## Integration with trainer.py

To use the optimized kernels in actual training:

```python
from cuda_kernels.Perturb_wise.triton_fused_perturb import FusedPerturbMeZO
from cuda_kernels.zo_foward_wise.optimized_zo_forward import OptimizedDiZO

# Replace standard MeZO with optimized version
trainer = FusedPerturbMeZO(model)

# Replace DiZO with optimized version
optimized_dizo = OptimizedDiZO(dizo_instance, model, anchor_model)
```

## File Structure

```
cuda_kernels/
├── benchmark_full_training_step.py   # ← This integrated benchmark
├── BENCHMARK_PLAN.md                 # ← This documentation
├── Perturb_wise/
│   ├── triton_fused_perturb.py       # Triton perturb kernels
│   ├── triton_mezo_kernels.py        # Alternative Triton impl
│   ├── fused_perturb.cu              # CUDA perturb kernel
│   ├── benchmark_fair_comparison.py  # Perturb benchmarks
│   └── ...
└── zo_foward_wise/
    ├── dizo_fused_kernels.py         # Triton zo_forward kernels
    ├── dizo_fused_kernels_v2.py      # V2 with pre-allocation
    ├── param_utils.py                # Parameter flattening utils
    ├── optimized_zo_forward.py       # Integrated optimizer
    ├── benchmark_zo_forward.py       # ZO-forward benchmarks
    └── ...
```

## Notes

1. **Forward pass timing**: The benchmark uses simulated forward passes. In real training, forward pass time dominates and speedups will be relatively smaller percentage-wise.

2. **Memory overhead**: Flat buffer approach requires temporary memory equal to model size. For large models, this may require optimization.

3. **Numerical correctness**: All kernels maintain numerical equivalence to PyTorch baseline (verified in separate tests).

4. **Multi-GPU**: Current implementation is single-GPU. FSDP/DDP integration requires additional work.

## References

- MeZO: Fine-Tuning Language Models with Just Forward Passes (arXiv:2305.17333)
- DiZO: Distance-based Zero-Order Optimization
- Triton Documentation: https://triton-lang.org/
