# Benchmark V3 Usage Guide

## Overview

The v3 script supports **TWO modes** of operation:

1. **All Methods at Once** (default): Runs all available benchmarks, each generating its own Chrome trace file
2. **Individual Method**: Run a single benchmark method with profiling (PyTorch, nsys, or ncu)

## Mode 1: All Methods at Once (Default)

When you run without `--method`, the script runs ALL available benchmarks:

```bash
# Run all methods with PyTorch Profiler
conda activate py310
CUDA_VISIBLE_DEVICES=5 python benchmark_full_training_step_v3.py \
    --model opt-350m \
    --pytorch_profile \
    --profile_iter 5
```

**Behavior:**
- Runs all 10 benchmark methods in sequence
- Each method generates its own Chrome trace file
- Trace files are named descriptively: `{model}_{method}_iter{n_iter}_{timestamp}.pt.trace.json`
- Each trace file is saved in: `benchmark_results/profiler_logs/{model}/{method}/`

**Example output:**
```
Benchmarking: PyTorch Baseline (MeZO only)...
  Chrome trace exported: opt-350m_PyTorch_Baseline_MeZO_only_iter5_20260103_212235.pt.trace.json

Benchmarking: PyTorch Baseline + DiZO...
  Chrome trace exported: opt-350m_PyTorch_Baseline_plus_DiZO_iter5_20260103_212242.pt.trace.json

Benchmarking: Triton Perturb (MeZO only)...
  Chrome trace exported: opt-350m_Triton_Perturb_MeZO_only_iter5_20260103_212250.pt.trace.json

... (continues for all methods)
```

## Mode 2: Individual Method Selection

To run a single benchmark method, use `--method`:

### Available Methods

- `pytorch_baseline_mezo` - PyTorch Baseline (MeZO only)
- `pytorch_baseline_dizo` - PyTorch Baseline + DiZO
- `triton_perturb_mezo` - Triton Perturb (MeZO only)
- `triton_perturb_dizo` - Triton Perturb + PyTorch DiZO
- `cuda_perturb_mezo` - CUDA Perturb (MeZO only)
- `cuda_perturb_dizo` - CUDA Perturb + PyTorch DiZO
- `triton_v1_full` - Triton Perturb + Triton ZO V1
- `triton_v2_full` - Triton Perturb + Triton ZO V2
- `cuda_triton_zo` - CUDA Perturb + Triton ZO V2
- `cuda_full` - CUDA Perturb + CUDA ZO V5

### PyTorch Profiler (Individual Method)

```bash
python benchmark_full_training_step_v3.py \
    --model opt-350m \
    --method pytorch_baseline_dizo \
    --pytorch_profile \
    --profile_iter 5
```

Generates: `opt-350m_PyTorch_Baseline_plus_DiZO_iter5_TIMESTAMP.pt.trace.json`

### nsys Profiling (Individual Method)

```bash
python benchmark_full_training_step_v3.py \
    --model opt-350m \
    --method triton_perturb_dizo \
    --nsys \
    --profile_iter 5
```

Generates: `opt-350m_Triton_Perturb_plus_PyTorch_DiZO_iter5.nsys-rep`

### ncu Profiling (Individual Method)

```bash
python benchmark_full_training_step_v3.py \
    --model opt-350m \
    --method cuda_full \
    --ncu \
    --profile_iter 5
```

Generates: `opt-350m_CUDA_Perturb_plus_CUDA_ZO_V5_iter5.ncu-rep`

## Trace File Naming Convention

All trace files follow this naming pattern:

**PyTorch Profiler:**
```
{model}_{method}_iter{n_iter}_{timestamp}.pt.trace.json
```

**nsys:**
```
{model}_{method}_iter{n_iter}.nsys-rep
```

**ncu:**
```
{model}_{method}_iter{n_iter}.ncu-rep
```

Where:
- `{model}`: Model name (e.g., `opt-350m`)
- `{method}`: Method name with spaces/special chars replaced (e.g., `PyTorch_Baseline_plus_DiZO`)
- `{n_iter}`: Number of iterations (e.g., `iter5`)
- `{timestamp}`: Timestamp in format `YYYYMMDD_HHMMSS` (for PyTorch only)

## Output Directory Structure

```
benchmark_results/
├── profiler_logs/
│   └── opt-350m/
│       ├── PyTorch_Baseline_MeZO_only/
│       │   └── opt-350m_PyTorch_Baseline_MeZO_only_iter5_TIMESTAMP.pt.trace.json
│       ├── PyTorch_Baseline_plus_DiZO/
│       │   └── opt-350m_PyTorch_Baseline_plus_DiZO_iter5_TIMESTAMP.pt.trace.json
│       ├── Triton_Perturb_MeZO_only/
│       │   └── opt-350m_Triton_Perturb_MeZO_only_iter5_TIMESTAMP.pt.trace.json
│       └── ... (one directory per method)
│       ├── nsys/  (if using --nsys)
│       │   └── opt-350m/
│       │       └── opt-350m_METHOD_iter5.nsys-rep
│       └── ncu/   (if using --ncu)
│           └── opt-350m/
│               └── opt-350m_METHOD_iter5.ncu-rep
```

## Quick Reference

| Command | Behavior |
|---------|----------|
| `--pytorch_profile` (no `--method`) | Run all methods, each gets PyTorch trace |
| `--method X --pytorch_profile` | Run method X only, get PyTorch trace |
| `--method X --nsys` | Run method X only, get nsys profile |
| `--method X --ncu` | Run method X only, get ncu profile |
| (no profiling flags) | Run all methods, timing only (no traces) |

## Notes

1. **All methods mode**: When running all methods, PyTorch Profiler generates separate trace files for each method automatically
2. **Individual mode**: Use `--method` to select a specific benchmark for targeted profiling
3. **nsys/ncu**: Only available for individual method runs (they require `--method`)
4. **Iterations**: Use `--profile_iter` (default 5) for profiling, `--n_iter` (default 20) for timing-only benchmarks

