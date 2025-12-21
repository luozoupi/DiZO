#!/usr/bin/env python3
"""
Compare Triton vs CUDA Fused Perturbation Kernels

This script benchmarks both implementations and compares performance.

Model size presets:
- OPT-350M: 331,196,416 parameters
- OPT-2.7B: 2,700,000,000 parameters (approximate)
- OPT-6.7B: 6,700,000,000 parameters (approximate)
- OPT-13B: 13,000,000,000 parameters (approximate)

Run with:
    python compare_kernels.py --model opt-2.7b
    python compare_kernels.py --model all
"""

import torch
import time
import numpy as np
import argparse

# Model size presets (parameter counts)
MODEL_SIZES = {
    'opt-350m': 331_196_416,
    'opt-2.7b': 2_700_000_000,
    'opt-6.7b': 6_700_000_000,
    'opt-13b': 13_000_000_000,
}

# Import Triton kernels
from triton_fused_perturb import (
    fused_perturb_kernel_v1,
    fused_perturb_kernel_v2,
    fused_perturb_kernel_autotuned,
    fused_perturb_kernel_v6_optimized,
)
import triton

# Try to import CUDA extension
try:
    import fused_perturb_cuda
    HAS_CUDA_EXT = True
    print("[INFO] CUDA extension loaded successfully")
except ImportError:
    HAS_CUDA_EXT = False
    print("[WARNING] CUDA extension not available. Run: python setup.py install")


def benchmark_kernel(name, func, params, seed, alpha, n_elements, n_iter=100, warmup=10):
    """Benchmark a kernel function."""
    try:
        # Warmup
        for _ in range(warmup):
            func(params, seed, alpha, n_elements)
        torch.cuda.synchronize()
        
        # Benchmark
        start = time.time()
        for _ in range(n_iter):
            func(params, seed, alpha, n_elements)
        torch.cuda.synchronize()
        elapsed = (time.time() - start) * 1000 / n_iter
        
        return elapsed
    except RuntimeError as e:
        print(f"    ERROR during benchmark: {e}")
        return float('inf')


def compare_kernels(n_elements=331_196_416, n_iter=100, warmup=10):
    """Compare all available kernels."""
    print("=" * 80)
    print("TRITON vs CUDA FUSED PERTURBATION KERNEL COMPARISON")
    print("=" * 80)
    print(f"Elements: {n_elements:,} ({n_elements * 4 / 1024**2:.1f} MB)")
    print(f"Iterations: {n_iter}, Warmup: {warmup}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print()
    
    # Create test tensor
    params = torch.randn(n_elements, device='cuda', dtype=torch.float32)
    seed = 42
    alpha = 1e-3
    
    results = {}
    
    # Test Triton kernels
    print("--- Triton Kernels ---")
    
    # Handle large tensors with chunking
    MAX_GRID_SIZE = 2_000_000_000  # 2B elements per grid
    n_elements_int64 = int(n_elements)
    
    # V1 Basic
    print("Testing V1 Basic (BLOCK=1024)...")
    if n_elements_int64 <= MAX_GRID_SIZE:
        grid_v1 = lambda meta: (triton.cdiv(n_elements_int64, 1024),)
        def triton_v1(p, s, a, n):
            fused_perturb_kernel_v1[grid_v1](p, s, a, n, BLOCK_SIZE=1024)
    else:
        # Chunked
        chunk_size = MAX_GRID_SIZE
        n_chunks = (n_elements_int64 + chunk_size - 1) // chunk_size
        def triton_v1(p, s, a, n):
            for chunk_idx in range(n_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min(start_idx + chunk_size, n_elements_int64)
                chunk_size_actual = end_idx - start_idx
                chunk_params = p[start_idx:end_idx]
                grid_chunk = lambda meta: (triton.cdiv(chunk_size_actual, 1024),)
                fused_perturb_kernel_v1[grid_chunk](chunk_params, s, a, chunk_size_actual, BLOCK_SIZE=1024)
    results['Triton V1 (1024)'] = benchmark_kernel(
        "Triton V1", triton_v1, params.clone(), seed, alpha, n_elements, n_iter, warmup
    )
    print(f"  Time: {results['Triton V1 (1024)']:.3f} ms")
    
    # V2 Optimized
    print("Testing V2 Optimized (BLOCK=2048)...")
    if n_elements_int64 <= MAX_GRID_SIZE:
        grid_v2 = lambda meta: (triton.cdiv(n_elements_int64, 2048),)
        def triton_v2(p, s, a, n):
            fused_perturb_kernel_v2[grid_v2](p, s, a, n, BLOCK_SIZE=2048)
    else:
        chunk_size = MAX_GRID_SIZE
        n_chunks = (n_elements_int64 + chunk_size - 1) // chunk_size
        def triton_v2(p, s, a, n):
            for chunk_idx in range(n_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min(start_idx + chunk_size, n_elements_int64)
                chunk_size_actual = end_idx - start_idx
                chunk_params = p[start_idx:end_idx]
                grid_chunk = lambda meta: (triton.cdiv(chunk_size_actual, 2048),)
                fused_perturb_kernel_v2[grid_chunk](chunk_params, s, a, chunk_size_actual, BLOCK_SIZE=2048)
    results['Triton V2 (2048)'] = benchmark_kernel(
        "Triton V2", triton_v2, params.clone(), seed, alpha, n_elements, n_iter, warmup
    )
    print(f"  Time: {results['Triton V2 (2048)']:.3f} ms")
    
    # Autotuned
    print("Testing Autotuned...")
    if n_elements_int64 <= MAX_GRID_SIZE:
        grid_auto = lambda meta: (triton.cdiv(n_elements_int64, meta['BLOCK_SIZE']),)
        def triton_auto(p, s, a, n):
            fused_perturb_kernel_autotuned[grid_auto](p, s, a, n)
    else:
        chunk_size = MAX_GRID_SIZE
        n_chunks = (n_elements_int64 + chunk_size - 1) // chunk_size
        def triton_auto(p, s, a, n):
            for chunk_idx in range(n_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min(start_idx + chunk_size, n_elements_int64)
                chunk_size_actual = end_idx - start_idx
                chunk_params = p[start_idx:end_idx]
                grid_chunk = lambda meta: (triton.cdiv(chunk_size_actual, meta['BLOCK_SIZE']),)
                fused_perturb_kernel_autotuned[grid_chunk](chunk_params, s, a, chunk_size_actual)
    results['Triton Autotuned'] = benchmark_kernel(
        "Triton Autotuned", triton_auto, params.clone(), seed, alpha, n_elements, n_iter, warmup + 5
    )
    print(f"  Time: {results['Triton Autotuned']:.3f} ms")
    
    # V6 Optimized
    print("Testing V6 Optimized...")
    if n_elements_int64 <= MAX_GRID_SIZE:
        grid_auto = lambda meta: (triton.cdiv(n_elements_int64, meta['BLOCK_SIZE']),)
        def triton_v6(p, s, a, n):
            fused_perturb_kernel_v6_optimized[grid_auto](p, s, a, n)
    else:
        chunk_size = MAX_GRID_SIZE
        n_chunks = (n_elements_int64 + chunk_size - 1) // chunk_size
        def triton_v6(p, s, a, n):
            for chunk_idx in range(n_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min(start_idx + chunk_size, n_elements_int64)
                chunk_size_actual = end_idx - start_idx
                chunk_params = p[start_idx:end_idx]
                grid_chunk = lambda meta: (triton.cdiv(chunk_size_actual, meta['BLOCK_SIZE']),)
                fused_perturb_kernel_v6_optimized[grid_chunk](chunk_params, s, a, chunk_size_actual)
    results['Triton V6 Optimized'] = benchmark_kernel(
        "Triton V6", triton_v6, params.clone(), seed, alpha, n_elements, n_iter, warmup + 5
    )
    print(f"  Time: {results['Triton V6 Optimized']:.3f} ms")
    
    # Test CUDA extension if available
    if HAS_CUDA_EXT:
        print("\n--- CUDA Extension ---")
        print("Testing CUDA Extension...")
        def cuda_ext(p, s, a, n):
            fused_perturb_cuda.fused_perturb(p, s, a)
        results['CUDA Extension'] = benchmark_kernel(
            "CUDA Extension", cuda_ext, params.clone(), seed, alpha, n_elements, n_iter, warmup
        )
        print(f"  Time: {results['CUDA Extension']:.3f} ms")
    
    # PyTorch baseline
    print("\n--- PyTorch Baseline ---")
    print("Testing PyTorch Baseline...")
    def pytorch_baseline(p, s, a, n):
        torch.manual_seed(s)
        z = torch.empty_like(p)
        z.normal_()
        p.add_(z, alpha=a)
    results['PyTorch Baseline'] = benchmark_kernel(
        "PyTorch Baseline", pytorch_baseline, params.clone(), seed, alpha, n_elements, n_iter, warmup
    )
    print(f"  Time: {results['PyTorch Baseline']:.3f} ms")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    baseline = results.get('PyTorch Baseline', float('inf'))
    sorted_results = sorted(results.items(), key=lambda x: x[1])
    
    print(f"{'Kernel':<30} {'Time (ms)':>12} {'Speedup':>10} {'vs Best':>10}")
    print("-" * 62)
    
    best_time = sorted_results[0][1]
    for name, time_ms in sorted_results:
        speedup_vs_baseline = baseline / time_ms if baseline > 0 else 0
        speedup_vs_best = best_time / time_ms if time_ms > 0 else 0
        marker = " ★" if time_ms == best_time else ""
        print(f"{name:<30} {time_ms:>12.3f} {speedup_vs_baseline:>9.2f}x {speedup_vs_best:>9.2f}x{marker}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare CUDA vs Triton kernels')
    parser.add_argument('--model', type=str, choices=list(MODEL_SIZES.keys()) + ['all', 'custom'],
                        default='opt-350m', help='Model size to benchmark')
    parser.add_argument('--n_elements', type=int, default=None,
                        help='Custom number of elements (only used with --model custom)')
    parser.add_argument('--n_iter', type=int, default=100, help='Number of iterations')
    parser.add_argument('--warmup', type=int, default=10, help='Number of warmup iterations')
    args = parser.parse_args()
    
    if args.model == 'all':
        # Test all model sizes
        for model_name, n_elements in MODEL_SIZES.items():
            print("\n" + "="*80)
            print(f"MODEL: {model_name.upper()} ({n_elements:,} parameters)")
            print("="*80)
            compare_kernels(
                n_elements=n_elements,
                n_iter=args.n_iter,
                warmup=args.warmup,
            )
    elif args.model == 'custom':
        if args.n_elements is None:
            raise ValueError("--n_elements must be provided when --model custom")
        compare_kernels(
            n_elements=args.n_elements,
            n_iter=args.n_iter,
            warmup=args.warmup,
        )
    else:
        compare_kernels(
            n_elements=MODEL_SIZES[args.model],
            n_iter=args.n_iter,
            warmup=args.warmup,
        )

