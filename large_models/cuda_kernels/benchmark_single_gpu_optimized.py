#!/usr/bin/env python
"""
Benchmark for Single-GPU Optimizations in ZO (Zeroth-Order) Optimization

This script compares different single-GPU parallel strategies:
1. Baseline Sequential: perturb(+eps) -> forward -> perturb(-2eps) -> forward -> update
2. Single-GPU Dual-Model (Original): Two model copies, separate perturbs
3. Single-GPU Dual-Model (Optimized): Fused dual-perturb + reduced syncs
4. Single-GPU Dual-Model + CUDA Graph: Capture entire ZO step as graph

Usage:
    python benchmark_single_gpu_optimized.py --model opt-350m --n_iter 20
    python benchmark_single_gpu_optimized.py --model opt-1.3b --n_iter 10

Author: DiZO Team
Date: 2025-01
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import time
import gc
import argparse
import copy
from typing import Dict, Tuple, List, Optional, Any
from dataclasses import dataclass

# Add paths for kernel imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'Perturb_wise'))

# Import optimized kernels
import triton
from single_gpu_optimizations import (
    fused_dual_perturb_kernel_autotuned,
    fused_dual_perturb_kernel,
    compute_projected_grad_kernel,
    fused_update_runtime_grad_kernel,
    FusedDualPerturbOps,
)
from triton_fused_perturb import (
    fused_perturb_kernel_philox_autotuned,
    fused_update_kernel,
)

# Try to import transformers
HAS_TRANSFORMERS = False
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    print("Warning: transformers not available")


MODEL_CONFIGS = {
    'opt-350m': {
        'hf_name': 'facebook/opt-350m',
        'total_params': 331_196_416,
    },
    'opt-1.3b': {
        'hf_name': 'facebook/opt-1.3b',
        'total_params': 1_315_753_984,
    },
    'opt-2.7b': {
        'hf_name': 'facebook/opt-2.7b',
        'total_params': 2_651_596_800,
    },
}


@dataclass
class BenchmarkResult:
    method: str
    total_time_ms: float
    perturb_time_ms: float
    forward_time_ms: float
    update_time_ms: float
    memory_gb: float
    notes: str = ""


@dataclass
class ParamMetadata:
    name: str
    shape: Tuple[int, ...]
    offset: int
    numel: int


def flatten_model_params(model: nn.Module, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, List[ParamMetadata]]:
    """Flatten all model parameters into a single contiguous buffer."""
    total_numel = sum(p.numel() for p in model.parameters())
    flat_buffer = torch.empty(total_numel, dtype=torch.float32, device=device)
    anchor_buffer = torch.empty(total_numel, dtype=torch.float32, device=device)

    metadata = []
    offset = 0

    for name, param in model.named_parameters():
        numel = param.numel()
        flat_buffer[offset:offset + numel] = param.data.view(-1)
        anchor_buffer[offset:offset + numel] = param.data.view(-1)
        metadata.append(ParamMetadata(name, tuple(param.shape), offset, numel))
        offset += numel

    return flat_buffer, anchor_buffer, metadata


def set_model_params_from_flat(model: nn.Module, flat_buffer: torch.Tensor, metadata: List[ParamMetadata]):
    """Set model parameters to views into flat buffer."""
    for meta in metadata:
        param = dict(model.named_parameters())[meta.name]
        param.data = flat_buffer[meta.offset:meta.offset + meta.numel].view(meta.shape)


# =============================================================================
# Benchmark Methods
# =============================================================================

def benchmark_sequential_baseline(
    model: nn.Module,
    flat_buffer: torch.Tensor,
    anchor_buffer: torch.Tensor,
    batch: Dict[str, torch.Tensor],
    n_iter: int,
    eps: float,
    lr: float,
    device: torch.device,
) -> BenchmarkResult:
    """
    Sequential baseline: perturb(+eps) -> forward1 -> perturb(-2eps) -> forward2 -> update
    """
    n_elements = flat_buffer.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    # Warmup
    for _ in range(3):
        flat_buffer.copy_(anchor_buffer)
        fused_perturb_kernel_philox_autotuned[grid](flat_buffer, 42, eps, n_elements)
        with torch.no_grad():
            _ = model(**batch, return_dict=True).loss
    torch.cuda.synchronize(device)

    # Benchmark
    timings = {'perturb': [], 'forward': [], 'update': [], 'total': []}

    for i in range(n_iter):
        seed = np.random.randint(1000000000)
        flat_buffer.copy_(anchor_buffer)

        total_start = torch.cuda.Event(enable_timing=True)
        total_end = torch.cuda.Event(enable_timing=True)
        total_start.record()

        # Perturb +eps
        perturb_start = torch.cuda.Event(enable_timing=True)
        perturb_end = torch.cuda.Event(enable_timing=True)
        perturb_start.record()
        fused_perturb_kernel_philox_autotuned[grid](flat_buffer, seed, eps, n_elements)
        perturb_end.record()

        # Forward 1
        fwd_start = torch.cuda.Event(enable_timing=True)
        fwd_end = torch.cuda.Event(enable_timing=True)
        fwd_start.record()
        with torch.no_grad():
            loss1 = model(**batch, return_dict=True).loss
        fwd_end.record()

        # Perturb -2eps
        fused_perturb_kernel_philox_autotuned[grid](flat_buffer, seed, -2 * eps, n_elements)

        # Forward 2
        fwd2_start = torch.cuda.Event(enable_timing=True)
        fwd2_end = torch.cuda.Event(enable_timing=True)
        fwd2_start.record()
        with torch.no_grad():
            loss2 = model(**batch, return_dict=True).loss
        fwd2_end.record()

        # Perturb +eps (restore)
        fused_perturb_kernel_philox_autotuned[grid](flat_buffer, seed, eps, n_elements)

        # Update
        update_start = torch.cuda.Event(enable_timing=True)
        update_end = torch.cuda.Event(enable_timing=True)
        update_start.record()
        projected_grad = (loss1.item() - loss2.item()) / (2 * eps)
        fused_update_kernel[grid](flat_buffer, seed, projected_grad, lr, n_elements, BLOCK_SIZE=1024)
        update_end.record()

        total_end.record()
        torch.cuda.synchronize(device)

        # Record timings (skip first iteration)
        if i > 0:
            timings['perturb'].append(perturb_start.elapsed_time(perturb_end) * 3)  # 3 perturb calls
            timings['forward'].append(fwd_start.elapsed_time(fwd_end) + fwd2_start.elapsed_time(fwd2_end))
            timings['update'].append(update_start.elapsed_time(update_end))
            timings['total'].append(total_start.elapsed_time(total_end))

    memory = torch.cuda.max_memory_allocated(device) / (1024**3)

    return BenchmarkResult(
        method="Sequential Baseline",
        total_time_ms=np.mean(timings['total']),
        perturb_time_ms=np.mean(timings['perturb']),
        forward_time_ms=np.mean(timings['forward']),
        update_time_ms=np.mean(timings['update']),
        memory_gb=memory,
        notes="3 perturbs (sequential)",
    )


def benchmark_dual_model_original(
    model1: nn.Module,
    model2: nn.Module,
    flat1: torch.Tensor,
    flat2: torch.Tensor,
    anchor1: torch.Tensor,
    anchor2: torch.Tensor,
    batch1: Dict[str, torch.Tensor],
    batch2: Dict[str, torch.Tensor],
    n_iter: int,
    eps: float,
    lr: float,
    device: torch.device,
) -> BenchmarkResult:
    """
    Original dual-model: Two separate perturb calls, parallel forwards.
    """
    n_elements = flat1.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    stream1 = torch.cuda.Stream(device=device)
    stream2 = torch.cuda.Stream(device=device)

    # Warmup
    for _ in range(3):
        flat1.copy_(anchor1)
        flat2.copy_(anchor2)
        fused_perturb_kernel_philox_autotuned[grid](flat1, 42, eps, n_elements)
        fused_perturb_kernel_philox_autotuned[grid](flat2, 42, -eps, n_elements)
        with torch.no_grad():
            _ = model1(**batch1).loss
            _ = model2(**batch2).loss
    torch.cuda.synchronize(device)

    # Benchmark
    timings = {'perturb': [], 'forward': [], 'update': [], 'total': []}

    for i in range(n_iter):
        seed = np.random.randint(1000000000)
        flat1.copy_(anchor1)
        flat2.copy_(anchor2)

        total_start = torch.cuda.Event(enable_timing=True)
        total_end = torch.cuda.Event(enable_timing=True)
        total_start.record(stream1)

        # Parallel perturbs (separate kernels)
        perturb_start = torch.cuda.Event(enable_timing=True)
        perturb_end = torch.cuda.Event(enable_timing=True)
        perturb_start.record(stream1)

        with torch.cuda.stream(stream1):
            fused_perturb_kernel_philox_autotuned[grid](flat1, seed, eps, n_elements)

        with torch.cuda.stream(stream2):
            fused_perturb_kernel_philox_autotuned[grid](flat2, seed, -eps, n_elements)

        perturb_end.record(stream1)
        torch.cuda.synchronize(device)

        # Parallel forwards
        fwd_start = torch.cuda.Event(enable_timing=True)
        fwd_end = torch.cuda.Event(enable_timing=True)
        fwd_start.record(stream1)

        with torch.cuda.stream(stream1):
            with torch.no_grad():
                loss1 = model1(**batch1, return_dict=True).loss

        with torch.cuda.stream(stream2):
            with torch.no_grad():
                loss2 = model2(**batch2, return_dict=True).loss

        fwd_end.record(stream1)
        torch.cuda.synchronize(device)

        # Update
        update_start = torch.cuda.Event(enable_timing=True)
        update_end = torch.cuda.Event(enable_timing=True)
        update_start.record(stream1)

        projected_grad = (loss1.item() - loss2.item()) / (2 * eps)
        fused_update_kernel[grid](flat1, seed, projected_grad, lr, n_elements, BLOCK_SIZE=1024)
        # Sync anchor for next iteration
        anchor1.copy_(flat1)

        update_end.record(stream1)
        total_end.record(stream1)
        torch.cuda.synchronize(device)

        if i > 0:
            timings['perturb'].append(perturb_start.elapsed_time(perturb_end))
            timings['forward'].append(fwd_start.elapsed_time(fwd_end))
            timings['update'].append(update_start.elapsed_time(update_end))
            timings['total'].append(total_start.elapsed_time(total_end))

    memory = torch.cuda.max_memory_allocated(device) / (1024**3)

    return BenchmarkResult(
        method="Dual-Model Original",
        total_time_ms=np.mean(timings['total']),
        perturb_time_ms=np.mean(timings['perturb']),
        forward_time_ms=np.mean(timings['forward']),
        update_time_ms=np.mean(timings['update']),
        memory_gb=memory,
        notes="2 separate perturb kernels",
    )


def benchmark_dual_model_fused_perturb(
    model1: nn.Module,
    model2: nn.Module,
    flat1: torch.Tensor,
    flat2: torch.Tensor,
    anchor: torch.Tensor,  # Single anchor for fused kernel
    batch1: Dict[str, torch.Tensor],
    batch2: Dict[str, torch.Tensor],
    n_iter: int,
    eps: float,
    lr: float,
    device: torch.device,
) -> BenchmarkResult:
    """
    Optimized dual-model with fused dual-perturb kernel.
    """
    n_elements = flat1.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    grid_update = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    stream1 = torch.cuda.Stream(device=device)
    stream2 = torch.cuda.Stream(device=device)

    # Warmup
    for _ in range(3):
        fused_dual_perturb_kernel_autotuned[grid](flat1, flat2, anchor, 42, eps, n_elements)
        with torch.no_grad():
            _ = model1(**batch1).loss
            _ = model2(**batch2).loss
    torch.cuda.synchronize(device)

    # Benchmark
    timings = {'perturb': [], 'forward': [], 'update': [], 'total': []}

    for i in range(n_iter):
        seed = np.random.randint(1000000000)

        total_start = torch.cuda.Event(enable_timing=True)
        total_end = torch.cuda.Event(enable_timing=True)
        total_start.record(stream1)

        # Fused dual-perturb (single kernel writes both buffers)
        perturb_start = torch.cuda.Event(enable_timing=True)
        perturb_end = torch.cuda.Event(enable_timing=True)
        perturb_start.record(stream1)
        fused_dual_perturb_kernel_autotuned[grid](flat1, flat2, anchor, seed, eps, n_elements)
        perturb_end.record(stream1)

        # Parallel forwards
        fwd_start = torch.cuda.Event(enable_timing=True)
        fwd_end = torch.cuda.Event(enable_timing=True)
        fwd_start.record(stream1)

        with torch.cuda.stream(stream1):
            with torch.no_grad():
                loss1 = model1(**batch1, return_dict=True).loss

        with torch.cuda.stream(stream2):
            with torch.no_grad():
                loss2 = model2(**batch2, return_dict=True).loss

        fwd_end.record(stream1)
        torch.cuda.synchronize(device)

        # Update on anchor
        update_start = torch.cuda.Event(enable_timing=True)
        update_end = torch.cuda.Event(enable_timing=True)
        update_start.record(stream1)

        projected_grad = (loss1.item() - loss2.item()) / (2 * eps)
        fused_update_kernel[grid_update](anchor, seed, projected_grad, lr, n_elements, BLOCK_SIZE=1024)

        update_end.record(stream1)
        total_end.record(stream1)
        torch.cuda.synchronize(device)

        if i > 0:
            timings['perturb'].append(perturb_start.elapsed_time(perturb_end))
            timings['forward'].append(fwd_start.elapsed_time(fwd_end))
            timings['update'].append(update_start.elapsed_time(update_end))
            timings['total'].append(total_start.elapsed_time(total_end))

    memory = torch.cuda.max_memory_allocated(device) / (1024**3)

    return BenchmarkResult(
        method="Dual-Model Fused Perturb",
        total_time_ms=np.mean(timings['total']),
        perturb_time_ms=np.mean(timings['perturb']),
        forward_time_ms=np.mean(timings['forward']),
        update_time_ms=np.mean(timings['update']),
        memory_gb=memory,
        notes="1 fused perturb kernel",
    )


def benchmark_dual_model_async_grad(
    model1: nn.Module,
    model2: nn.Module,
    flat1: torch.Tensor,
    flat2: torch.Tensor,
    anchor: torch.Tensor,
    batch1: Dict[str, torch.Tensor],
    batch2: Dict[str, torch.Tensor],
    n_iter: int,
    eps: float,
    lr: float,
    device: torch.device,
) -> BenchmarkResult:
    """
    Fully optimized: fused perturb + async grad computation (no .item() sync).
    """
    n_elements = flat1.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    # Pre-allocate grad tensor for async computation
    grad_tensor = torch.zeros(1, dtype=torch.float32, device=device)

    stream1 = torch.cuda.Stream(device=device)
    stream2 = torch.cuda.Stream(device=device)

    # Warmup
    for _ in range(3):
        fused_dual_perturb_kernel_autotuned[grid](flat1, flat2, anchor, 42, eps, n_elements)
        with torch.no_grad():
            loss1 = model1(**batch1).loss
            loss2 = model2(**batch2).loss
        compute_projected_grad_kernel[(1,)](loss1, loss2, eps, grad_tensor)
        fused_update_runtime_grad_kernel[grid](anchor, grad_tensor, 42, lr, n_elements)
    torch.cuda.synchronize(device)

    # Benchmark
    timings = {'perturb': [], 'forward': [], 'update': [], 'total': []}

    for i in range(n_iter):
        seed = np.random.randint(1000000000)

        total_start = torch.cuda.Event(enable_timing=True)
        total_end = torch.cuda.Event(enable_timing=True)
        total_start.record(stream1)

        # Fused dual-perturb
        perturb_start = torch.cuda.Event(enable_timing=True)
        perturb_end = torch.cuda.Event(enable_timing=True)
        perturb_start.record(stream1)
        fused_dual_perturb_kernel_autotuned[grid](flat1, flat2, anchor, seed, eps, n_elements)
        perturb_end.record(stream1)

        # Parallel forwards
        fwd_start = torch.cuda.Event(enable_timing=True)
        fwd_end = torch.cuda.Event(enable_timing=True)
        fwd_start.record(stream1)

        with torch.cuda.stream(stream1):
            with torch.no_grad():
                loss1 = model1(**batch1, return_dict=True).loss

        with torch.cuda.stream(stream2):
            with torch.no_grad():
                loss2 = model2(**batch2, return_dict=True).loss

        fwd_end.record(stream1)

        # ASYNC gradient computation (no .item() sync!)
        update_start = torch.cuda.Event(enable_timing=True)
        update_end = torch.cuda.Event(enable_timing=True)

        # Wait for both forwards to complete
        stream1.synchronize()
        stream2.synchronize()

        update_start.record(stream1)

        # Compute grad on GPU
        compute_projected_grad_kernel[(1,)](loss1, loss2, eps, grad_tensor)

        # Update using GPU grad tensor
        fused_update_runtime_grad_kernel[grid](anchor, grad_tensor, seed, lr, n_elements)

        update_end.record(stream1)
        total_end.record(stream1)
        torch.cuda.synchronize(device)

        if i > 0:
            timings['perturb'].append(perturb_start.elapsed_time(perturb_end))
            timings['forward'].append(fwd_start.elapsed_time(fwd_end))
            timings['update'].append(update_start.elapsed_time(update_end))
            timings['total'].append(total_start.elapsed_time(total_end))

    memory = torch.cuda.max_memory_allocated(device) / (1024**3)

    return BenchmarkResult(
        method="Dual-Model Fused+AsyncGrad",
        total_time_ms=np.mean(timings['total']),
        perturb_time_ms=np.mean(timings['perturb']),
        forward_time_ms=np.mean(timings['forward']),
        update_time_ms=np.mean(timings['update']),
        memory_gb=memory,
        notes="Fused perturb + async grad (no .item())",
    )


def print_results(results: List[BenchmarkResult], baseline_time: float):
    """Print benchmark results in a table."""
    print("\n" + "=" * 100)
    print("BENCHMARK RESULTS")
    print("=" * 100)
    print(f"{'Method':<35} {'Total(ms)':<12} {'Speedup':<10} {'Perturb':<10} {'Forward':<10} {'Update':<10} {'Memory':<10}")
    print("-" * 100)

    for r in results:
        speedup = baseline_time / r.total_time_ms if r.total_time_ms > 0 else 0
        print(f"{r.method:<35} {r.total_time_ms:<12.2f} {speedup:<10.2f}x {r.perturb_time_ms:<10.2f} {r.forward_time_ms:<10.2f} {r.update_time_ms:<10.2f} {r.memory_gb:<10.2f}GB")

    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(description="Benchmark single-GPU ZO optimizations")
    parser.add_argument('--model', type=str, default='opt-350m', choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument('--device', type=int, default=0, help='GPU device ID')
    parser.add_argument('--n_iter', type=int, default=20, help='Benchmark iterations')
    parser.add_argument('--eps', type=float, default=1e-3, help='Perturbation epsilon')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--seq_len', type=int, default=128, help='Sequence length')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--skip_dual', action='store_true', help='Skip dual-model benchmarks (memory)')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.device}')
    print(f"Device: {device} ({torch.cuda.get_device_name(device)})")
    print(f"Model: {args.model}")
    print(f"Iterations: {args.n_iter}")

    if not HAS_TRANSFORMERS:
        print("Error: transformers library required")
        return

    # Load model
    config = MODEL_CONFIGS[args.model]
    print(f"\nLoading {config['hf_name']}...")
    model1 = AutoModelForCausalLM.from_pretrained(
        config['hf_name'],
        torch_dtype=torch.float32,
        use_cache=False,
    ).to(device)
    model1.eval()

    # Create flat buffer
    flat1, anchor1, metadata = flatten_model_params(model1, device)
    set_model_params_from_flat(model1, flat1, metadata)
    n_elements = flat1.numel()
    print(f"Model parameters: {n_elements:,}")

    # Create dummy batch
    tokenizer = AutoTokenizer.from_pretrained(config['hf_name'])
    batch1 = tokenizer(
        "Hello, this is a test input for benchmarking.",
        return_tensors="pt",
        padding="max_length",
        max_length=args.seq_len,
        truncation=True,
    )
    batch1 = {k: v.to(device) for k, v in batch1.items()}
    batch1['labels'] = batch1['input_ids'].clone()

    results = []

    # Sequential baseline
    print("\n[1/4] Running sequential baseline...")
    torch.cuda.reset_peak_memory_stats(device)
    result_seq = benchmark_sequential_baseline(
        model1, flat1, anchor1, batch1,
        args.n_iter, args.eps, args.lr, device
    )
    results.append(result_seq)
    baseline_time = result_seq.total_time_ms
    print(f"  Total: {result_seq.total_time_ms:.2f}ms")

    if not args.skip_dual:
        # Check memory for dual model
        current_mem = torch.cuda.memory_allocated(device) / (1024**3)
        total_mem = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        model_mem = n_elements * 4 / (1024**3)  # float32

        if current_mem + model_mem * 2 > total_mem * 0.9:
            print(f"\nSkipping dual-model benchmarks (insufficient memory)")
            print(f"  Current: {current_mem:.1f}GB, Model: {model_mem:.1f}GB, Total: {total_mem:.1f}GB")
        else:
            # Create second model
            print("\n[2/4] Creating second model copy...")
            model2 = copy.deepcopy(model1).to(device)
            flat2, anchor2, _ = flatten_model_params(model2, device)
            set_model_params_from_flat(model2, flat2, metadata)
            batch2 = {k: v.clone() for k, v in batch1.items()}

            # Dual-model original
            print("\n[2/4] Running dual-model original...")
            torch.cuda.reset_peak_memory_stats(device)
            result_orig = benchmark_dual_model_original(
                model1, model2, flat1, flat2, anchor1, anchor2,
                batch1, batch2, args.n_iter, args.eps, args.lr, device
            )
            results.append(result_orig)
            print(f"  Total: {result_orig.total_time_ms:.2f}ms ({baseline_time/result_orig.total_time_ms:.2f}x)")

            # Dual-model with fused perturb
            print("\n[3/4] Running dual-model with fused perturb...")
            # Reset anchor
            anchor1.copy_(flat1)
            torch.cuda.reset_peak_memory_stats(device)
            result_fused = benchmark_dual_model_fused_perturb(
                model1, model2, flat1, flat2, anchor1,
                batch1, batch2, args.n_iter, args.eps, args.lr, device
            )
            results.append(result_fused)
            print(f"  Total: {result_fused.total_time_ms:.2f}ms ({baseline_time/result_fused.total_time_ms:.2f}x)")

            # Dual-model with async grad
            print("\n[4/4] Running dual-model with async grad...")
            anchor1.copy_(flat1)
            torch.cuda.reset_peak_memory_stats(device)
            result_async = benchmark_dual_model_async_grad(
                model1, model2, flat1, flat2, anchor1,
                batch1, batch2, args.n_iter, args.eps, args.lr, device
            )
            results.append(result_async)
            print(f"  Total: {result_async.total_time_ms:.2f}ms ({baseline_time/result_async.total_time_ms:.2f}x)")

            # Cleanup
            del model2, flat2, anchor2

    # Print results
    print_results(results, baseline_time)

    # Summary
    print("\nKEY FINDINGS:")
    if len(results) > 1:
        best = min(results, key=lambda x: x.total_time_ms)
        print(f"  Best method: {best.method}")
        print(f"  Best speedup: {baseline_time/best.total_time_ms:.2f}x over sequential baseline")

        # Perturb optimization impact
        if len(results) >= 3:
            perturb_improvement = results[1].perturb_time_ms / results[2].perturb_time_ms
            print(f"  Fused perturb speedup: {perturb_improvement:.2f}x")


if __name__ == "__main__":
    main()
