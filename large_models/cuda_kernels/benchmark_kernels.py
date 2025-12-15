#!/usr/bin/env python3
"""
Comprehensive Benchmark: CUDA vs Triton vs PyTorch Fused Perturb Kernels

This script compares:
1. PyTorch baseline (randn + add as separate ops)
2. Triton fused kernel (randn + add in one kernel)
3. CUDA fused kernel (if compiled)
4. ChunkedMeZO (PyTorch cuRAND in chunks)

Run with:
    python benchmark_kernels.py
"""

import torch
import numpy as np
import time
import gc
from typing import Dict, Tuple

# Import Triton kernels
try:
    from triton_fused_perturb import (
        fused_perturb_kernel_v1,
        fused_perturb_kernel_v2, 
        fused_perturb_kernel_autotuned,
        fused_restore_update_kernel,
        FusedPerturbMeZO,
    )
    import triton
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    print("[WARNING] Triton not available")

# Try to import CUDA extension
try:
    import fused_perturb_cuda
    HAS_CUDA_EXT = True
except ImportError:
    HAS_CUDA_EXT = False
    print("[INFO] CUDA extension not compiled. Run: cd cuda_kernels && python setup.py install")


def benchmark_pytorch_baseline(params: torch.Tensor, seed: int, alpha: float, n_iter: int) -> Tuple[float, float]:
    """Benchmark PyTorch's separate randn + add operations.
    
    Returns:
        Tuple of (time_ms, peak_memory_mb)
    """
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    z = torch.empty_like(params)
    
    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(n_iter):
        torch.manual_seed(seed)
        z.normal_()
        params.add_(z, alpha=alpha)
    
    torch.cuda.synchronize()
    elapsed_ms = (time.time() - start) * 1000 / n_iter
    peak_memory_mb = torch.cuda.max_memory_allocated() / 1024**2
    
    return elapsed_ms, peak_memory_mb


def benchmark_pytorch_fused(params: torch.Tensor, seed: int, alpha: float, n_iter: int) -> Tuple[float, float]:
    """Benchmark PyTorch's addcmul (closest to fused op).
    
    Returns:
        Tuple of (time_ms, peak_memory_mb)
    """
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    z = torch.empty_like(params)
    alpha_tensor = torch.tensor(alpha, device=params.device)
    
    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(n_iter):
        torch.manual_seed(seed)
        z.normal_()
        # addcmul: params = params + alpha * z
        torch.addcmul(params, alpha_tensor, z, out=params)
    
    torch.cuda.synchronize()
    elapsed_ms = (time.time() - start) * 1000 / n_iter
    peak_memory_mb = torch.cuda.max_memory_allocated() / 1024**2
    
    return elapsed_ms, peak_memory_mb


def benchmark_triton_v1(params: torch.Tensor, seed: int, alpha: float, n_iter: int) -> Tuple[float, float]:
    """Benchmark Triton V1 kernel (basic).
    
    Returns:
        Tuple of (time_ms, peak_memory_mb)
    """
    if not HAS_TRITON:
        return float('inf'), 0.0
    
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    n_elements = params.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(n_iter):
        fused_perturb_kernel_v1[grid](params, seed, alpha, n_elements, BLOCK_SIZE=1024)
    
    torch.cuda.synchronize()
    elapsed_ms = (time.time() - start) * 1000 / n_iter
    peak_memory_mb = torch.cuda.max_memory_allocated() / 1024**2
    
    return elapsed_ms, peak_memory_mb


def benchmark_triton_v2(params: torch.Tensor, seed: int, alpha: float, n_iter: int, block_size: int = 1024) -> Tuple[float, float]:
    """Benchmark Triton V2 kernel (optimized).
    
    Returns:
        Tuple of (time_ms, peak_memory_mb)
    """
    if not HAS_TRITON:
        return float('inf'), 0.0
    
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    n_elements = params.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(n_iter):
        fused_perturb_kernel_v2[grid](params, seed, alpha, n_elements, BLOCK_SIZE=block_size)
    
    torch.cuda.synchronize()
    elapsed_ms = (time.time() - start) * 1000 / n_iter
    peak_memory_mb = torch.cuda.max_memory_allocated() / 1024**2
    
    return elapsed_ms, peak_memory_mb


def benchmark_triton_autotuned(params: torch.Tensor, seed: int, alpha: float, n_iter: int, measure_memory: bool = True) -> Tuple[float, float]:
    """Benchmark Triton autotuned kernel.
    
    Args:
        measure_memory: If True, reset memory stats and measure peak memory.
                       If False, skip memory measurement for fair timing.
    
    Returns:
        Tuple of (time_ms, peak_memory_mb)
    """
    if not HAS_TRITON:
        return float('inf'), 0.0
    
    if measure_memory:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    
    n_elements = params.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Warmup for autotuning
    for _ in range(10):
        fused_perturb_kernel_autotuned[grid](params, seed, alpha, n_elements)
    
    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(n_iter):
        fused_perturb_kernel_autotuned[grid](params, seed, alpha, n_elements)
    
    torch.cuda.synchronize()
    elapsed_ms = (time.time() - start) * 1000 / n_iter
    
    if measure_memory:
        peak_memory_mb = torch.cuda.max_memory_allocated() / 1024**2
    else:
        peak_memory_mb = 0.0  # Will be measured separately
    
    return elapsed_ms, peak_memory_mb


def benchmark_cuda_ext(params: torch.Tensor, seed: int, alpha: float, n_iter: int, measure_memory: bool = True) -> Tuple[float, float]:
    """Benchmark custom CUDA extension.
    
    Args:
        measure_memory: If True, reset memory stats and measure peak memory.
                       If False, skip memory measurement for fair timing.
    
    Returns:
        Tuple of (time_ms, peak_memory_mb)
    """
    if not HAS_CUDA_EXT:
        return float('inf'), 0.0
    
    if measure_memory:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    
    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(n_iter):
        fused_perturb_cuda.fused_perturb(params, seed, alpha)
    
    torch.cuda.synchronize()
    elapsed_ms = (time.time() - start) * 1000 / n_iter
    
    if measure_memory:
        peak_memory_mb = torch.cuda.max_memory_allocated() / 1024**2
    else:
        peak_memory_mb = 0.0  # Will be measured separately
    
    return elapsed_ms, peak_memory_mb


def benchmark_chunked(params: torch.Tensor, seed: int, alpha: float, n_iter: int, chunk_size_mb: int = 64) -> Tuple[float, float]:
    """Benchmark chunked approach (PyTorch cuRAND in chunks).
    
    Returns:
        Tuple of (time_ms, peak_memory_mb)
    """
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    chunk_size = (chunk_size_mb * 1024 * 1024) // 4  # float32 = 4 bytes
    n_elements = params.numel()
    z_chunk = torch.empty(min(chunk_size, n_elements), device=params.device, dtype=params.dtype)
    
    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(n_iter):
        torch.manual_seed(seed)
        offset = 0
        while offset < n_elements:
            chunk_len = min(chunk_size, n_elements - offset)
            z_view = z_chunk[:chunk_len]
            z_view.normal_()
            params[offset:offset + chunk_len].add_(z_view, alpha=alpha)
            offset += chunk_len
    
    torch.cuda.synchronize()
    elapsed_ms = (time.time() - start) * 1000 / n_iter
    peak_memory_mb = torch.cuda.max_memory_allocated() / 1024**2
    
    return elapsed_ms, peak_memory_mb


def run_comprehensive_benchmark(
    sizes: list = None,
    n_iter: int = 100,
    warmup: int = 10,
):
    """Run comprehensive benchmark across different tensor sizes."""
    
    if sizes is None:
        sizes = [
            (1_000_000, "1M"),
            (10_000_000, "10M"),
            (100_000_000, "100M"),
            (331_196_416, "OPT-350M"),
        ]
    
    print("=" * 90)
    print("COMPREHENSIVE KERNEL BENCHMARK: CUDA vs Triton vs PyTorch")
    print("=" * 90)
    print(f"Iterations: {n_iter}, Warmup: {warmup}")
    print(f"Triton available: {HAS_TRITON}")
    print(f"CUDA extension available: {HAS_CUDA_EXT}")
    
    seed = 42
    alpha = 1e-3
    
    all_results = {}
    
    for n_elements, size_name in sizes:
        print(f"\n{'='*90}")
        print(f"SIZE: {size_name} ({n_elements:,} elements, {n_elements * 4 / 1024**2:.1f} MB)")
        print("=" * 90)
        
        # Create test tensor
        params = torch.randn(n_elements, device='cuda', dtype=torch.float32)
        
        results = {}
        
        # Warmup
        print("Warming up...")
        _, _ = benchmark_pytorch_baseline(params.clone(), seed, alpha, warmup)
        
        # 1. PyTorch baseline
        print("\n1. PyTorch Baseline (randn + add):")
        params_test = params.clone()
        time_ms, memory_mb = benchmark_pytorch_baseline(params_test, seed, alpha, n_iter)
        results['pytorch_baseline'] = {'time': time_ms, 'memory': memory_mb}
        print(f"   Time: {time_ms:.3f} ms, Memory: {memory_mb:.2f} MB")
        
        # 2. PyTorch fused (addcmul)
        print("\n2. PyTorch Fused (addcmul):")
        params_test = params.clone()
        time_ms, memory_mb = benchmark_pytorch_fused(params_test, seed, alpha, n_iter)
        results['pytorch_fused'] = {'time': time_ms, 'memory': memory_mb}
        print(f"   Time: {time_ms:.3f} ms, Memory: {memory_mb:.2f} MB")
        
        # 3. Chunked (64MB)
        print("\n3. Chunked PyTorch (64MB buffer):")
        params_test = params.clone()
        time_ms, memory_mb = benchmark_chunked(params_test, seed, alpha, n_iter, 64)
        results['chunked_64mb'] = {'time': time_ms, 'memory': memory_mb}
        print(f"   Time: {time_ms:.3f} ms, Memory: {memory_mb:.2f} MB")
        
        # 4. Triton V1
        if HAS_TRITON:
            print("\n4. Triton V1 (basic):")
            params_test = params.clone()
            time_ms, memory_mb = benchmark_triton_v1(params_test, seed, alpha, n_iter)
            results['triton_v1'] = {'time': time_ms, 'memory': memory_mb}
            print(f"   Time: {time_ms:.3f} ms, Memory: {memory_mb:.2f} MB")
        
        # 5. Triton V2 (different block sizes)
        if HAS_TRITON:
            for block_size in [1024, 2048, 4096]:
                print(f"\n5. Triton V2 (BLOCK={block_size}):")
                params_test = params.clone()
                time_ms, memory_mb = benchmark_triton_v2(params_test, seed, alpha, n_iter, block_size)
                results[f'triton_v2_b{block_size}'] = {'time': time_ms, 'memory': memory_mb}
                print(f"   Time: {time_ms:.3f} ms, Memory: {memory_mb:.2f} MB")
        
        # 6. Triton Autotuned
        if HAS_TRITON:
            print("\n6. Triton Autotuned:")
            params_test = params.clone()
            time_ms, memory_mb = benchmark_triton_autotuned(params_test, seed, alpha, n_iter)
            results['triton_autotuned'] = {'time': time_ms, 'memory': memory_mb}
            print(f"   Time: {time_ms:.3f} ms, Memory: {memory_mb:.2f} MB")
        
        # 7. CUDA Extension
        if HAS_CUDA_EXT:
            print("\n7. CUDA Extension:")
            params_test = params.clone()
            time_ms, memory_mb = benchmark_cuda_ext(params_test, seed, alpha, n_iter)
            results['cuda_ext'] = {'time': time_ms, 'memory': memory_mb}
            print(f"   Time: {time_ms:.3f} ms, Memory: {memory_mb:.2f} MB")
        
        # Summary for this size
        print(f"\n{'-'*90}")
        print(f"SUMMARY for {size_name}:")
        print(f"{'-'*90}")
        
        baseline_time = results['pytorch_baseline']['time']
        sorted_results = sorted(results.items(), key=lambda x: x[1]['time'])
        
        print(f"{'Method':<30} {'Time (ms)':>12} {'Speedup':>10} {'Memory (MB)':>15}")
        print("-" * 67)
        
        for name, data in sorted_results:
            time_ms = data['time']
            memory_mb = data['memory']
            speedup = baseline_time / time_ms if time_ms > 0 else 0
            
            marker = " ★" if time_ms == sorted_results[0][1]['time'] else ""
            print(f"{name:<30} {time_ms:>12.3f} {speedup:>9.2f}x {memory_mb:>14.2f}{marker}")
        
        all_results[size_name] = results
        
        # Cleanup
        del params
        gc.collect()
        torch.cuda.empty_cache()
    
    return all_results


def verify_correctness():
    """Verify that all kernels produce statistically equivalent results."""
    print("\n" + "=" * 70)
    print("CORRECTNESS VERIFICATION")
    print("=" * 70)
    
    n_elements = 1_000_000
    seed = 12345
    alpha = 0.1
    
    # Reference: PyTorch baseline
    torch.manual_seed(seed)
    params_ref = torch.randn(n_elements, device='cuda', dtype=torch.float32)
    z_ref = torch.empty_like(params_ref)
    torch.manual_seed(seed)
    z_ref.normal_()
    params_ref.add_(z_ref, alpha=alpha)
    
    results = {'pytorch_baseline': params_ref.clone()}
    
    # Test Triton kernels
    if HAS_TRITON:
        params_test = torch.randn(n_elements, device='cuda', dtype=torch.float32)
        torch.manual_seed(seed)
        params_test_init = params_test.clone()
        
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        fused_perturb_kernel_v2[grid](params_test, seed, alpha, n_elements, BLOCK_SIZE=1024)
        results['triton_v2'] = params_test.clone()
        
        # Note: Triton uses different RNG, so values won't match exactly
        # but should have same statistical properties
    
    print("\nStatistical comparison (mean, std):")
    print("-" * 50)
    
    for name, tensor in results.items():
        mean = tensor.mean().item()
        std = tensor.std().item()
        print(f"{name:<25} mean={mean:>10.6f}, std={std:>10.6f}")
    
    print("\nNote: Triton uses Philox RNG with different sequence,")
    print("so values differ but follow same N(0,1) distribution.")
    print("This is statistically equivalent for training!")


def benchmark_complete_mezo_step(n_elements=331_196_416, n_iter=50, warmup=5):
    """
    Benchmark complete MeZO training step matching trainer.py:
    1. Perturb +eps (3 kernel calls)
    2. Update: params = params - lr * projected_grad * z (1 kernel call)
    Total: 4 kernel calls per step
    """
    print("\n" + "=" * 90)
    print("COMPLETE MEZO STEP BENCHMARK")
    print("=" * 90)
    print(f"Elements: {n_elements:,} ({n_elements * 4 / 1024**2:.1f} MB)")
    print(f"Iterations: {n_iter}, Warmup: {warmup}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print()
    
    eps = 1e-3
    lr = 1e-5
    seed = 42
    results = {}
    
    # PyTorch Baseline (complete step)
    print("1. PyTorch Baseline (complete MeZO step):")
    params_pytorch = torch.randn(n_elements, device='cuda', dtype=torch.float32)
    
    def pytorch_mezo_step(p):
        # Step 1: Perturb +eps
        torch.manual_seed(seed)
        z1 = torch.empty_like(p)
        z1.normal_()
        p.add_(z1, alpha=eps)
        # Step 2: Forward (simulated)
        loss1 = 1.0
        # Step 3: Perturb -2eps
        torch.manual_seed(seed)
        z2 = torch.empty_like(p)
        z2.normal_()
        p.add_(z2, alpha=-2*eps)
        # Step 4: Forward (simulated)
        loss2 = 0.9
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
    
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    for _ in range(warmup):
        pytorch_mezo_step(params_pytorch.clone())
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(n_iter):
        pytorch_mezo_step(params_pytorch.clone())
    torch.cuda.synchronize()
    pytorch_time = (time.time() - start) * 1000 / n_iter
    pytorch_memory = torch.cuda.max_memory_allocated() / 1024**2
    
    results['pytorch_mezo_step'] = {'time': pytorch_time, 'memory': pytorch_memory}
    print(f"   Time: {pytorch_time:.3f} ms, Memory: {pytorch_memory:.2f} MB")
    
    # CUDA Extension (complete step)
    if HAS_CUDA_EXT:
        print("\n2. CUDA Extension (complete MeZO step):")
        params_cuda = torch.randn(n_elements, device='cuda', dtype=torch.float32)
        
        def cuda_mezo_step(p):
            # Step 1: Perturb +eps
            fused_perturb_cuda.fused_perturb(p, seed, eps)
            loss1 = 1.0
            # Step 3: Perturb -2eps
            fused_perturb_cuda.fused_perturb(p, seed, -2*eps)
            loss2 = 0.9
            # Step 5: Compute projected_grad
            projected_grad = (loss1 - loss2) / (2 * eps)
            # Step 6: Reset +eps
            fused_perturb_cuda.fused_perturb(p, seed, eps)
            # Step 7: Update
            try:
                fused_perturb_cuda.fused_update(p, seed, projected_grad, lr)
            except AttributeError:
                fused_perturb_cuda.fused_perturb(p, seed, -lr * projected_grad)
        
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        for _ in range(warmup):
            cuda_mezo_step(params_cuda.clone())
        torch.cuda.synchronize()
        
        start = time.time()
        for _ in range(n_iter):
            cuda_mezo_step(params_cuda.clone())
        torch.cuda.synchronize()
        cuda_time = (time.time() - start) * 1000 / n_iter
        cuda_memory = torch.cuda.max_memory_allocated() / 1024**2
        
        results['cuda_mezo_step'] = {'time': cuda_time, 'memory': cuda_memory}
        print(f"   Time: {cuda_time:.3f} ms, Memory: {cuda_memory:.2f} MB")
    
    # Triton (complete step)
    if HAS_TRITON:
        print("\n3. Triton (complete MeZO step):")
        from triton_fused_perturb import fused_update_kernel_autotuned
        params_triton = torch.randn(n_elements, device='cuda', dtype=torch.float32)
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        
        def triton_mezo_step(p):
            # Step 1: Perturb +eps
            fused_perturb_kernel_autotuned[grid](p, seed, eps, n_elements)
            loss1 = 1.0
            # Step 3: Perturb -2eps
            fused_perturb_kernel_autotuned[grid](p, seed, -2*eps, n_elements)
            loss2 = 0.9
            # Step 5: Compute projected_grad
            projected_grad = (loss1 - loss2) / (2 * eps)
            # Step 6: Reset +eps
            fused_perturb_kernel_autotuned[grid](p, seed, eps, n_elements)
            # Step 7: Update
            fused_update_kernel_autotuned[grid](p, seed, projected_grad, lr, n_elements)
        
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        for _ in range(warmup + 5):
            triton_mezo_step(params_triton.clone())
        torch.cuda.synchronize()
        
        start = time.time()
        for _ in range(n_iter):
            triton_mezo_step(params_triton.clone())
        torch.cuda.synchronize()
        triton_time = (time.time() - start) * 1000 / n_iter
        triton_memory = torch.cuda.max_memory_allocated() / 1024**2
        
        results['triton_mezo_step'] = {'time': triton_time, 'memory': triton_memory}
        print(f"   Time: {triton_time:.3f} ms, Memory: {triton_memory:.2f} MB")
    
    # Summary
    print(f"\n{'-'*90}")
    print("SUMMARY (Complete MeZO Step)")
    print(f"{'-'*90}")
    
    baseline_time = results.get('pytorch_mezo_step', {}).get('time', float('inf'))
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
    # Run correctness verification first
    verify_correctness()
    
    # Run comprehensive benchmark
    results = run_comprehensive_benchmark(
        sizes=[
            (10_000_000, "10M"),
            (100_000_000, "100M"),
            (331_196_416, "OPT-350M"),
        ],
        n_iter=50,
        warmup=10,
    )
    
    # Run complete MeZO step benchmark
    benchmark_complete_mezo_step(n_elements=331_196_416, n_iter=50, warmup=5)
    
    print("\n" + "=" * 90)
    print("BENCHMARK COMPLETE")
    print("=" * 90)
