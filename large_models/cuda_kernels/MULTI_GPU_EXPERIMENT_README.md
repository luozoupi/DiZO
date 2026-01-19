# Multi-GPU Parallel MeZO Experiment

This experiment tests the multi-GPU parallel approach for MeZO training, where the two forward passes (f(θ+εz) and f(θ-εz)) run simultaneously on separate GPUs.

## NEW: Real OPT Model Support

The benchmark now supports **real OPT model forward passes** with proper flat buffer ↔ model parameter mapping!

### Quick Start with Real Models

```bash
# Use real OPT-350M model (requires transformers library)
CUDA_VISIBLE_DEVICES=1,2 python benchmark_multi_gpu_parallel.py \
    --model opt-350m --use_real_model --compare_all

# With larger batch for better multi-GPU utilization
CUDA_VISIBLE_DEVICES=1,2 python benchmark_multi_gpu_parallel.py \
    --model opt-350m --use_real_model --batch_size 4 --seq_len 256 --compare_all

# Using environment variables with the shell script
USE_REAL_MODEL=1 BATCH_SIZE=4 SEQ_LEN=256 bash run_multi_gpu_experiment.sh opt-350m
```

### How Flat Buffer ↔ Model Mapping Works

1. **Model Loading**: Loads real OPT model from HuggingFace
2. **Flat Buffer Creation**: Flattens all model parameters into a single contiguous buffer
3. **Parameter Views**: Creates views into the flat buffer that map to `model.param.data`
4. **Perturb + Forward**: When we perturb the flat buffer, model params automatically see the perturbed values

```python
# The mapping is set up like this:
for name, param in model.named_parameters():
    # Create a view into flat_buffer at the correct offset
    view = flat_buffer[offset:offset + numel].view(param.shape)
    # Set param.data to point to this view
    param.data = view
```

### Real Model Benchmark Results (OPT-350M, batch=1, seq=128)

| Method | Total Time | Forward Time | Notes |
|--------|-----------|--------------|-------|
| **Sequential Baseline** | 28.71ms | 21.18ms | Standard MeZO |
| **Single-GPU Parallel** | 30.13ms | 20.81ms | 2 buffers, 2 streams |
| **Single-GPU Pipelined** | 32.22ms | 17.50ms | Overlapped prep |
| **Multi-GPU Parallel** | 2678ms | 23.18ms | 50ms model sync overhead |

**Key insight**: Multi-GPU approach has high overhead due to model sync (~50ms per iteration).
For multi-GPU to be beneficial, forward passes need to exceed the sync overhead.

## Overview

This experiment tests **three parallel/pipelined approaches** for MeZO training:

### 1. Multi-GPU Parallel
- **Main GPU**: Handles θ₀ + εz → forward1 → loss1
- **Side GPU**: Handles θ₀ - εz → forward2 → loss2
- Both forward passes run simultaneously on separate GPUs
- **Expected speedup**: ~1.8-1.9x (near 2x if forwards dominate)

### 2. Single-GPU Parallel (Two Buffers)
- Uses **two buffers** on the same GPU
- **Buffer 1**: θ₀ + εz → forward1 (stream1)
- **Buffer 2**: θ₀ - εz → forward2 (stream2)
- Both forwards run in parallel using CUDA streams
- **Memory**: 2x parameter memory (base + plus + minus buffers)
- **Expected speedup**: ~1.5-1.8x (limited by single GPU bandwidth)

### 3. Single-GPU Pipelined (One Buffer + Temp)
- Uses **one buffer + one temp buffer**
- **Phase 1**: Prepare θ₀ + εz → forward1 (stream1)
- **Phase 2**: While forward1 runs, prepare θ₀ - εz in temp buffer (stream2) - **overlapped!**
- **Phase 3**: After forward1, swap temp → main buffer → forward2
- **Memory**: ~2x parameter memory (base + perturbed + temp)
- **Expected speedup**: ~1.2-1.5x (partial overlap, less than full parallel)

## Files

- `benchmark_multi_gpu_parallel.py`: Main benchmark script
- `run_multi_gpu_experiment.sh`: Convenience script for running with GPUs 1,2 and conda py310

## Usage

### Basic Usage

```bash
# Run with default GPUs (0 and 1)
python benchmark_multi_gpu_parallel.py --model opt-350m

# Use specific GPUs (for debugging)
python benchmark_multi_gpu_parallel.py --model opt-350m --main_gpu 1 --side_gpu 2

# Compare with sequential baseline (includes single-GPU parallel)
python benchmark_multi_gpu_parallel.py --model opt-350m --compare_baseline

# Compare ALL methods (multi-GPU, single-GPU parallel, pipelined, sequential)
python benchmark_multi_gpu_parallel.py --model opt-350m --compare_all

# Debug mode with detailed timing
python benchmark_multi_gpu_parallel.py --model opt-350m --debug
```

### Using the Convenience Script

For debugging with GPUs 1,2 and conda venv py310:

```bash
# Basic run
./run_multi_gpu_experiment.sh opt-350m

# With custom iterations
./run_multi_gpu_experiment.sh opt-350m 0 1 50
```

The script automatically:
- Activates conda environment `py310`
- Sets `CUDA_VISIBLE_DEVICES=1,2` (so GPUs 1,2 appear as 0,1 in the script)
- Runs the benchmark with comparison to sequential baseline

## Arguments

- `--model`: Model to benchmark (opt-350m, opt-1.3b, opt-2.7b, opt-6.7b, opt-13b)
- `--main_gpu`: Main GPU device ID (default: 0)
- `--side_gpu`: Side GPU device ID (default: 1)
- `--n_iter`: Number of benchmark iterations (default: 20)
- `--eps`: Perturbation scale (default: 1e-3)
- `--lr`: Learning rate (default: 1e-5)
- `--compare_baseline`: Compare with sequential baseline + single-GPU parallel
- `--compare_all`: Compare ALL methods (multi-GPU, single-GPU parallel, pipelined, sequential)
- `--debug`: Enable detailed timing breakdown
- `--use_triton`: Use Triton kernels instead of CUDA (slower but more compatible)
- `--use_real_model`: **NEW** Use real OPT model for forward passes (requires transformers)
- `--batch_size`: Batch size for real model forward (default: 1)
- `--seq_len`: Sequence length for real model forward (default: 128)

## Output

The benchmark reports:

1. **Total time**: End-to-end time for one training step
2. **Memory usage**: Peak memory on both GPUs
3. **Timing breakdown**:
   - Model synchronization (main → side)
   - Perturbation times (main and side)
   - Forward pass times (main and side)
   - Loss synchronization (side → main)
   - Parameter update time
4. **Speedup analysis** (if `--compare_baseline` is used):
   - Speedup vs sequential execution
   - Efficiency (percentage of theoretical 2x speedup)
   - Forward pass speedup breakdown

## Expected Results

### Multi-GPU Parallel
- **Best case**: ~1.8-1.9x speedup when forward passes dominate
- **Overhead**: Model sync ~5-10ms, loss sync ~0.1ms
- **When it helps**: Forward passes >50ms each, sync overhead << forward time

### Single-GPU Parallel (Two Buffers)
- **Best case**: ~1.5-1.8x speedup (limited by single GPU)
- **Memory**: 2x parameter memory
- **When it helps**: Single GPU available, forward passes are bottleneck

### Single-GPU Pipelined (One Buffer + Temp)
- **Best case**: ~1.2-1.5x speedup (partial overlap)
- **Memory**: ~2x parameter memory (base + perturbed + temp)
- **When it helps**: Memory constrained, forward time >> perturbation prep time

### Comparison Summary
| Method | Speedup | Memory | Best For |
|--------|---------|--------|----------|
| Multi-GPU | ~1.8-1.9x | 2x (2 GPUs) | Multiple GPUs available |
| Single-GPU Parallel | ~1.5-1.8x | 2x (1 GPU) | Single GPU, memory available |
| Single-GPU Pipelined | ~1.2-1.5x | ~2x (1 GPU) | Memory constrained |
| Sequential | 1.0x | 1x | Baseline |

## Implementation Details

### Model Synchronization

Before each step, parameters are synchronized from main GPU to side GPU:
```python
param_flat_side.copy_(param_flat_main, non_blocking=True)
anchor_flat_side.copy_(anchor_flat_main, non_blocking=True)
```

### Random Vector Generation

The same random vector `z` is generated on both GPUs:
1. Generate on CPU with fixed seed
2. Copy to both GPUs
3. Ensures mathematical equivalence: both compute f(θ+εz) and f(θ-εz)

### Parallel Execution

Uses CUDA streams for true parallelism:
```python
# Main GPU stream
with torch.cuda.device(device_main):
    with torch.cuda.stream(stream_main):
        # Perturb + forward1
        
# Side GPU stream (runs simultaneously!)
with torch.cuda.device(device_side):
    with torch.cuda.stream(stream_side):
        # Perturb + forward2
```

## Troubleshooting

### Out of Memory

If you get OOM errors:
- Use a smaller model (opt-350m instead of opt-13b)
- Reduce batch size in dummy_forward() if using real model
- Check that both GPUs have sufficient free memory

### Kernel Import Errors

If CUDA/Triton kernels fail to import:
- Check that kernels are built in `Perturb_wise/` and `zo_foward_wise/`
- Use `--use_triton` flag to fall back to Triton kernels
- Ensure conda environment has required dependencies

### Synchronization Issues

If results are incorrect:
- Ensure both GPUs complete their work before computing gradient
- Check that model sync happens before perturbations
- Verify random seed synchronization

## Future Work

- [ ] Add DiZO constraint support for multi-GPU
- [ ] Optimize model sync with peer-to-peer (P2P) transfers
- [ ] Support for more than 2 GPUs
- [x] ~~Integration with real model forward passes~~ **DONE!**

## Notes

- Currently only supports MeZO (no DiZO constraints)
- Supports both synthetic and **real OPT model** forward passes
- Model sync is done via flat buffer copy (efficient for large models)
- Loss transfer is minimal overhead (~0.1ms)
- Real model mode uses `transformers` library (install with `pip install transformers`)
- Flat buffer ↔ model param mapping uses `view()` for zero-copy access