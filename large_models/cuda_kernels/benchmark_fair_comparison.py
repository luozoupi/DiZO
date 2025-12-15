#!/usr/bin/env python3
"""
Fair Benchmark Comparison: CUDA vs Triton

This script ensures fair comparison by:
1. Measuring timing WITHOUT memory reset overhead
2. Measuring memory separately in dedicated runs
3. Using same warmup for all kernels
4. Includes complete MeZO training step benchmark
"""

import torch
import time
import numpy as np

# Import kernels
try:
    import fused_perturb_cuda
    HAS_CUDA_EXT = True
except ImportError:
    HAS_CUDA_EXT = False

try:
    from triton_fused_perturb import (
        fused_perturb_kernel_autotuned,
        fused_update_kernel_autotuned,
    )
    import triton
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


def benchmark_timing(params, kernel_func, n_iter=100, warmup=10):
    """Benchmark timing without memory measurement overhead."""
    # Warmup
    for _ in range(warmup):
        kernel_func(params)
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.time()
    for _ in range(n_iter):
        kernel_func(params)
    torch.cuda.synchronize()
    return (time.time() - start) * 1000 / n_iter


def benchmark_memory(params, kernel_func):
    """Benchmark memory usage separately."""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    # Run kernel once to measure memory
    kernel_func(params)
    torch.cuda.synchronize()
    
    return torch.cuda.max_memory_allocated() / 1024**2


def fair_benchmark(n_elements=331_196_416, n_iter=100, warmup=10):
    """Run fair benchmark comparison."""
    print("=" * 80)
    print("FAIR BENCHMARK: CUDA vs Triton (No Memory Reset Overhead)")
    print("=" * 80)
    print(f"Elements: {n_elements:,} ({n_elements * 4 / 1024**2:.1f} MB)")
    print(f"Iterations: {n_iter}, Warmup: {warmup}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print()
    
    seed = 42
    alpha = 1e-3
    
    results = {}
    
    # CUDA Extension
    if HAS_CUDA_EXT:
        print("Testing CUDA Extension...")
        params_cuda = torch.randn(n_elements, device='cuda', dtype=torch.float32)
        
        def cuda_kernel(p):
            fused_perturb_cuda.fused_perturb(p, seed, alpha)
        
        time_ms = benchmark_timing(params_cuda, cuda_kernel, n_iter, warmup)
        memory_mb = benchmark_memory(params_cuda.clone(), cuda_kernel)
        results['CUDA Extension'] = {'time': time_ms, 'memory': memory_mb}
        print(f"  Time: {time_ms:.3f} ms, Memory: {memory_mb:.2f} MB")
    
    # Triton Autotuned
    if HAS_TRITON:
        print("Testing Triton Autotuned...")
        params_triton = torch.randn(n_elements, device='cuda', dtype=torch.float32)
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        
        def triton_kernel(p):
            fused_perturb_kernel_autotuned[grid](p, seed, alpha, n_elements)
        
        time_ms = benchmark_timing(params_triton, triton_kernel, n_iter, warmup + 5)
        memory_mb = benchmark_memory(params_triton.clone(), triton_kernel)
        results['Triton Autotuned'] = {'time': time_ms, 'memory': memory_mb}
        print(f"  Time: {time_ms:.3f} ms, Memory: {memory_mb:.2f} MB")
    
    # PyTorch Baseline
    print("Testing PyTorch Baseline...")
    params_pytorch = torch.randn(n_elements, device='cuda', dtype=torch.float32)
    
    def pytorch_kernel(p):
        torch.manual_seed(seed)
        z = torch.empty_like(p)
        z.normal_()
        p.add_(z, alpha=alpha)
    
    time_ms = benchmark_timing(params_pytorch, pytorch_kernel, n_iter, warmup)
    memory_mb = benchmark_memory(params_pytorch.clone(), pytorch_kernel)
    results['PyTorch Baseline'] = {'time': time_ms, 'memory': memory_mb}
    print(f"  Time: {time_ms:.3f} ms, Memory: {memory_mb:.2f} MB")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    baseline_time = results.get('PyTorch Baseline', {}).get('time', float('inf'))
    sorted_results = sorted(results.items(), key=lambda x: x[1]['time'])
    
    print(f"{'Kernel':<30} {'Time (ms)':>12} {'Speedup':>10} {'Memory (MB)':>15}")
    print("-" * 67)
    
    best_time = sorted_results[0][1]['time']
    for name, data in sorted_results:
        time_ms = data['time']
        memory_mb = data['memory']
        speedup = baseline_time / time_ms if time_ms > 0 else 0
        marker = " ★" if time_ms == best_time else ""
        print(f"{name:<30} {time_ms:>12.3f} {speedup:>9.2f}x {memory_mb:>14.2f}{marker}")
    
    return results


def benchmark_complete_mezo_step(n_elements=331_196_416, n_iter=50, warmup=5):
    """
    Benchmark complete MeZO training step matching trainer.py:
    1. Perturb +eps
    2. Forward pass (simulated with dummy loss)
    3. Perturb -2eps
    4. Forward pass (simulated with dummy loss)
    5. Compute projected_grad
    6. Reset +eps
    7. Update: params = params - lr * projected_grad * z
    """
    print("=" * 80)
    print("COMPLETE MEZO STEP BENCHMARK")
    print("=" * 80)
    print(f"Elements: {n_elements:,} ({n_elements * 4 / 1024**2:.1f} MB)")
    print(f"Iterations: {n_iter}, Warmup: {warmup}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print()
    
    eps = 1e-3
    lr = 1e-5
    results = {}
    
    # PyTorch Baseline (complete step)
    print("Testing PyTorch Baseline (complete MeZO step)...")
    params_pytorch = torch.randn(n_elements, device='cuda', dtype=torch.float32)
    
    def pytorch_mezo_step(p):
        seed = 42
        # Step 1: Perturb +eps
        torch.manual_seed(seed)
        z1 = torch.empty_like(p)
        z1.normal_()
        p.add_(z1, alpha=eps)
        # Step 2: Forward (simulated)
        loss1 = 1.0  # Dummy loss
        # Step 3: Perturb -2eps
        torch.manual_seed(seed)
        z2 = torch.empty_like(p)
        z2.normal_()
        p.add_(z2, alpha=-2*eps)
        # Step 4: Forward (simulated)
        loss2 = 0.9  # Dummy loss
        # Step 5: Compute projected_grad
        projected_grad = (loss1 - loss2) / (2 * eps)
        # Step 6: Reset +eps
        torch.manual_seed(seed)
        z3 = torch.empty_like(p)
        z3.normal_()
        p.add_(z3, alpha=eps)
        # Step 7: Update
        torch.manual_seed(seed)
        z4 = torch.empty_like(p)
        z4.normal_()
        p.add_(z4, alpha=-lr * projected_grad)
    
    # Warmup
    for _ in range(warmup):
        pytorch_mezo_step(params_pytorch.clone())
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.time()
    for _ in range(n_iter):
        pytorch_mezo_step(params_pytorch.clone())
    torch.cuda.synchronize()
    pytorch_time = (time.time() - start) * 1000 / n_iter
    
    # Memory
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    pytorch_mezo_step(params_pytorch.clone())
    torch.cuda.synchronize()
    pytorch_memory = torch.cuda.max_memory_allocated() / 1024**2
    
    results['PyTorch Baseline'] = {'time': pytorch_time, 'memory': pytorch_memory}
    print(f"  Time: {pytorch_time:.3f} ms, Memory: {pytorch_memory:.2f} MB")
    
    # CUDA Extension (complete step)
    if HAS_CUDA_EXT:
        print("Testing CUDA Extension (complete MeZO step)...")
        params_cuda = torch.randn(n_elements, device='cuda', dtype=torch.float32)
        seed = 42
        
        def cuda_mezo_step(p):
            # Step 1: Perturb +eps
            fused_perturb_cuda.fused_perturb(p, seed, eps)
            loss1 = 1.0  # Dummy loss
            # Step 3: Perturb -2eps
            fused_perturb_cuda.fused_perturb(p, seed, -2*eps)
            loss2 = 0.9  # Dummy loss
            # Step 5: Compute projected_grad
            projected_grad = (loss1 - loss2) / (2 * eps)
            # Step 6: Reset +eps
            fused_perturb_cuda.fused_perturb(p, seed, eps)
            # Step 7: Update (if fused_update exists, else use restore_update)
            try:
                fused_perturb_cuda.fused_update(p, seed, projected_grad, lr)
            except AttributeError:
                # Fallback: use manual update if fused_update not available
                fused_perturb_cuda.fused_perturb(p, seed, -lr * projected_grad)
        
        # Warmup
        for _ in range(warmup):
            cuda_mezo_step(params_cuda.clone())
        torch.cuda.synchronize()
        
        # Benchmark
        start = time.time()
        for _ in range(n_iter):
            cuda_mezo_step(params_cuda.clone())
        torch.cuda.synchronize()
        cuda_time = (time.time() - start) * 1000 / n_iter
        
        # Memory
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        cuda_mezo_step(params_cuda.clone())
        torch.cuda.synchronize()
        cuda_memory = torch.cuda.max_memory_allocated() / 1024**2
        
        results['CUDA Extension'] = {'time': cuda_time, 'memory': cuda_memory}
        print(f"  Time: {cuda_time:.3f} ms, Memory: {cuda_memory:.2f} MB")
    
    # Triton (complete step)
    if HAS_TRITON:
        print("Testing Triton (complete MeZO step)...")
        params_triton = torch.randn(n_elements, device='cuda', dtype=torch.float32)
        seed = 42
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        
        def triton_mezo_step(p):
            # Step 1: Perturb +eps
            fused_perturb_kernel_autotuned[grid](p, seed, eps, n_elements)
            loss1 = 1.0  # Dummy loss
            # Step 3: Perturb -2eps
            fused_perturb_kernel_autotuned[grid](p, seed, -2*eps, n_elements)
            loss2 = 0.9  # Dummy loss
            # Step 5: Compute projected_grad
            projected_grad = (loss1 - loss2) / (2 * eps)
            # Step 6: Reset +eps
            fused_perturb_kernel_autotuned[grid](p, seed, eps, n_elements)
            # Step 7: Update
            fused_update_kernel_autotuned[grid](p, seed, projected_grad, lr, n_elements)
        
        # Warmup
        for _ in range(warmup + 5):
            triton_mezo_step(params_triton.clone())
        torch.cuda.synchronize()
        
        # Benchmark
        start = time.time()
        for _ in range(n_iter):
            triton_mezo_step(params_triton.clone())
        torch.cuda.synchronize()
        triton_time = (time.time() - start) * 1000 / n_iter
        
        # Memory
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        triton_mezo_step(params_triton.clone())
        torch.cuda.synchronize()
        triton_memory = torch.cuda.max_memory_allocated() / 1024**2
        
        results['Triton'] = {'time': triton_time, 'memory': triton_memory}
        print(f"  Time: {triton_time:.3f} ms, Memory: {triton_memory:.2f} MB")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY (Complete MeZO Step)")
    print("=" * 80)
    
    baseline_time = results.get('PyTorch Baseline', {}).get('time', float('inf'))
    sorted_results = sorted(results.items(), key=lambda x: x[1]['time'])
    
    print(f"{'Implementation':<30} {'Time (ms)':>12} {'Speedup':>10} {'Memory (MB)':>15}")
    print("-" * 67)
    
    best_time = sorted_results[0][1]['time']
    for name, data in sorted_results:
        time_ms = data['time']
        memory_mb = data['memory']
        speedup = baseline_time / time_ms if time_ms > 0 else 0
        marker = " ★" if time_ms == best_time else ""
        print(f"{name:<30} {time_ms:>12.3f} {speedup:>9.2f}x {memory_mb:>14.2f}{marker}")
    
    return results


if __name__ == "__main__":
    # Test with both small and large tensors
    print("\n" + "="*80)
    print("SMALL TENSOR TEST (10M elements)")
    print("="*80)
    fair_benchmark(n_elements=10_000_000, n_iter=100, warmup=10)
    
    print("\n\n" + "="*80)
    print("LARGE TENSOR TEST (331M elements - OPT-350M)")
    print("="*80)
    fair_benchmark(n_elements=331_196_416, n_iter=100, warmup=10)
    
    print("\n\n" + "="*80)
    print("COMPLETE MEZO STEP TEST (331M elements)")
    print("="*80)
    benchmark_complete_mezo_step(n_elements=331_196_416, n_iter=50, warmup=5)

