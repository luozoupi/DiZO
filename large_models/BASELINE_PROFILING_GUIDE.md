# Baseline AdamW Profiling - Quick Start Guide

## Overview

The `profile_baseline_adamw.py` script generates a PyTorch profiler trace for conventional first-order (AdamW) training, allowing you to compare it with MeZO/DiZO using the same `analyze_timeline.py` tool.

## Quick Usage

### 1. Generate Baseline Trace

Run with default settings (OPT-350M, SST2 task, 20 steps):

```bash
cd /mnt/data1/luo00466/DiZO_old/DiZO/large_models
python profile_baseline_adamw.py
```

### 2. Analyze the Trace

```bash
python analyze_timeline.py profiler_logs/baseline_adamw.pt.trace.json
```

This will output:
- `timeline_gantt.png` - Visual Gantt chart
- Terminal report with GPU utilization, gaps, and bottlenecks

## Custom Configuration

Match your MeZO profiling setup:

```bash
python profile_baseline_adamw.py \
    --model_name facebook/opt-1.3b \
    --task_name SST2 \
    --num_steps 20 \
    --warmup_steps 5 \
    --batch_size 4 \
    --seq_length 128 \
    --lr 1e-5 \
    --output_trace profiler_logs/baseline_opt1.3b.pt.trace.json \
    --device cuda
```

## Comparing MeZO vs Baseline

### Step 1: Analyze MeZO trace
```bash
python analyze_timeline.py profiler_logs/cs-u-converge_3680154.1764723598761079115.pt.trace.json
```

Output example:
```
GPU Utilization: 32.6%
GPU Idle Gaps: 12056 gaps, 261.22 ms total (40.0%)
Top CPU Operations:
   zo_gradient_estimation: 10 calls, 558.82 ms
   zo_perturb_+eps: 10 calls, 227.12 ms
```

### Step 2: Analyze baseline trace
```bash
python analyze_timeline.py profiler_logs/baseline_adamw.pt.trace.json
```

Expected output:
```
GPU Utilization: ~70-85% (higher than MeZO)
GPU Idle Gaps: fewer gaps, ~10-20% idle
Top CPU Operations:
   aten::linear: XXX calls, YYY ms
   aten::addmm: XXX calls, YYY ms
   (no zo_* operations)
```

### Step 3: Compare

Key metrics to compare:
- **GPU Utilization %**: Baseline should be higher (~70-85% vs MeZO's ~33%)
- **Gap Time %**: Baseline should have less idle time
- **Kernel Launches**: Baseline may have more but better pipelined
- **Top Operations**: MeZO has `zo_*` ops, baseline has standard `aten::*` ops

## Parameters Explained

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model_name` | `facebook/opt-350m` | HuggingFace model ID |
| `--task_name` | `SST2` | Task (SST2, RTE, CB, etc.) |
| `--num_steps` | `20` | Total profiling steps |
| `--warmup_steps` | `5` | Steps before recording starts |
| `--batch_size` | `4` | Batch size |
| `--seq_length` | `128` | Max sequence length |
| `--lr` | `1e-5` | AdamW learning rate |
| `--output_trace` | `profiler_logs/baseline_adamw.pt.trace.json` | Output file |
| `--device` | `cuda` | Device (cuda/cpu) |

## Troubleshooting

### Issue: "Task samples not available"
The script will use synthetic data for profiling. This is fine for performance comparison.

### Issue: Out of memory
Reduce `--batch_size` or `--seq_length`:
```bash
python profile_baseline_adamw.py --batch_size 2 --seq_length 64
```

### Issue: Wrong model path
Ensure the model exists locally or is a valid HuggingFace ID:
```bash
python profile_baseline_adamw.py --model_name ./path/to/local/model
```

## Expected Workflow

1. **Profile MeZO** (already done in your case):
   - Output: `profiler_logs/cs-u-converge_*.pt.trace.json`

2. **Profile Baseline** (new):
   ```bash
   python profile_baseline_adamw.py --model_name facebook/opt-1.3b --num_steps 20
   ```

3. **Analyze Both**:
   ```bash
   python analyze_timeline.py profiler_logs/baseline_adamw.pt.trace.json > baseline_report.txt
   python analyze_timeline.py profiler_logs/cs-u-converge_3680154.1764723598761079115.pt.trace.json > mezo_report.txt
   ```

4. **Compare**:
   ```bash
   diff baseline_report.txt mezo_report.txt
   # Or manually compare GPU util %, gap %, and top operations
   ```

## Notes

- The baseline uses **AdamW optimizer** (standard first-order training)
- MeZO traces show **zeroth-order operations** (`zo_perturb`, `zo_gradient_estimation`, etc.)
- Baseline typically has **higher GPU utilization** due to backpropagation parallelism
- MeZO has **CPU overhead** from per-parameter perturbations (visible in gaps)
