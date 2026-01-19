#!/usr/bin/env python
"""
Benchmark: CUDA vs Triton Kernels for Single-GPU ZO Optimization

Compares performance of:
1. CUDA vectorized kernels (float4)
2. CUDA basic kernels
3. Triton autotuned kernels
4. Triton basic kernels

Usage:
    python benchmark_cuda_vs_triton.py --n_elements 331196416  # OPT-350M
    python benchmark_cuda_vs_triton.py --n_elements 1315753984  # OPT-1.3B

Author: DiZO Team
Date: 2025-01
"""

import os
import sys
import torch
import numpy as np
import argparse
from typing import Dict

# Add paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'Perturb_wise'))

import triton

# Import CUDA kernels
try:
    import single_gpu_opt_cuda
    HAS_CUDA = True
    print("✓ CUDA kernels loaded successfully")
except ImportError as e:
    print(f"✗ CUDA kernels not available: {e}")
    HAS_CUDA = False

# Import Triton kernels
from single_gpu_optimizations import (
    fused_dual_perturb_kernel_autotuned,
    fused_dual_perturb_kernel,
    compute_projected_grad_kernel,
    fused_update_runtime_grad_kernel,
)
from triton_fused_perturb import fused_perturb_kernel_philox_autotuned


def benchmark_dual_perturb(n_elements: int, device: torch.device, n_iter: int = 100, warmup: int = 10) -> Dict[str, float]:
    """Benchmark dual perturbation kernels."""
    dtype = torch.float32
    eps = 1e-3

    anchor = torch.randn(n_elements, dtype=dtype, device=device)
    params_plus = torch.empty_like(anchor)
    params_minus = torch.empty_like(anchor)

    results = {}

    print(f"\n{'='*70}")
    print(f"DUAL PERTURB BENCHMARK (n_elements={n_elements:,})")
    print(f"{'='*70}")

    # ===================
    # Baseline: 2 separate Triton perturbs
    # ===================
    print("\n[1] Baseline: 2 separate Triton perturbs...")
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    for _ in range(warmup):
        params_plus.copy_(anchor)
        params_minus.copy_(anchor)
        fused_perturb_kernel_philox_autotuned[grid](params_plus, 42, eps, n_elements)
        fused_perturb_kernel_philox_autotuned[grid](params_minus, 42, -eps, n_elements)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for i in range(n_iter):
        params_plus.copy_(anchor)
        params_minus.copy_(anchor)
        fused_perturb_kernel_philox_autotuned[grid](params_plus, 42 + i, eps, n_elements)
        fused_perturb_kernel_philox_autotuned[grid](params_minus, 42 + i, -eps, n_elements)
    end.record()
    torch.cuda.synchronize()
    results['baseline_2x_triton'] = start.elapsed_time(end) / n_iter
    print(f"    Time: {results['baseline_2x_triton']:.3f} ms")

    # ===================
    # Triton Fused Dual-Perturb (autotuned)
    # ===================
    print("\n[2] Triton Fused Dual-Perturb (autotuned)...")
    grid_fused = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    for _ in range(warmup):
        fused_dual_perturb_kernel_autotuned[grid_fused](
            params_plus, params_minus, anchor, 42, eps, n_elements
        )
    torch.cuda.synchronize()

    start.record()
    for i in range(n_iter):
        fused_dual_perturb_kernel_autotuned[grid_fused](
            params_plus, params_minus, anchor, 42 + i, eps, n_elements
        )
    end.record()
    torch.cuda.synchronize()
    results['triton_fused_autotuned'] = start.elapsed_time(end) / n_iter
    print(f"    Time: {results['triton_fused_autotuned']:.3f} ms")

    # ===================
    # CUDA Fused Dual-Perturb (vectorized)
    # ===================
    if HAS_CUDA:
        print("\n[3] CUDA Fused Dual-Perturb (vectorized float4)...")

        for _ in range(warmup):
            single_gpu_opt_cuda.fused_dual_perturb(
                params_plus, params_minus, anchor, 42, eps
            )
        torch.cuda.synchronize()

        start.record()
        for i in range(n_iter):
            single_gpu_opt_cuda.fused_dual_perturb(
                params_plus, params_minus, anchor, 42 + i, eps
            )
        end.record()
        torch.cuda.synchronize()
        results['cuda_fused_vectorized'] = start.elapsed_time(end) / n_iter
        print(f"    Time: {results['cuda_fused_vectorized']:.3f} ms")

        # ===================
        # CUDA Fused Dual-Perturb (basic)
        # ===================
        print("\n[4] CUDA Fused Dual-Perturb (basic)...")

        for _ in range(warmup):
            single_gpu_opt_cuda.fused_dual_perturb_basic(
                params_plus, params_minus, anchor, 42, eps
            )
        torch.cuda.synchronize()

        start.record()
        for i in range(n_iter):
            single_gpu_opt_cuda.fused_dual_perturb_basic(
                params_plus, params_minus, anchor, 42 + i, eps
            )
        end.record()
        torch.cuda.synchronize()
        results['cuda_fused_basic'] = start.elapsed_time(end) / n_iter
        print(f"    Time: {results['cuda_fused_basic']:.3f} ms")

    return results


def benchmark_update_with_grad(n_elements: int, device: torch.device, n_iter: int = 100, warmup: int = 10) -> Dict[str, float]:
    """Benchmark update with runtime gradient kernels."""
    dtype = torch.float32
    lr = 1e-5

    params = torch.randn(n_elements, dtype=dtype, device=device)
    params_backup = params.clone()
    grad = torch.tensor([0.5], dtype=dtype, device=device)

    results = {}

    print(f"\n{'='*70}")
    print(f"UPDATE WITH RUNTIME GRAD BENCHMARK (n_elements={n_elements:,})")
    print(f"{'='*70}")

    # ===================
    # Triton Update (autotuned)
    # ===================
    print("\n[1] Triton Update with Runtime Grad (autotuned)...")
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    for _ in range(warmup):
        params.copy_(params_backup)
        fused_update_runtime_grad_kernel[grid](params, grad, 42, lr, n_elements)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for i in range(n_iter):
        params.copy_(params_backup)
        fused_update_runtime_grad_kernel[grid](params, grad, 42 + i, lr, n_elements)
    end.record()
    torch.cuda.synchronize()
    results['triton_update'] = start.elapsed_time(end) / n_iter
    print(f"    Time: {results['triton_update']:.3f} ms")

    # ===================
    # CUDA Update (vectorized)
    # ===================
    if HAS_CUDA:
        print("\n[2] CUDA Update with Runtime Grad (vectorized)...")

        for _ in range(warmup):
            params.copy_(params_backup)
            single_gpu_opt_cuda.fused_update_runtime_grad(params, grad, 42, lr)
        torch.cuda.synchronize()

        start.record()
        for i in range(n_iter):
            params.copy_(params_backup)
            single_gpu_opt_cuda.fused_update_runtime_grad(params, grad, 42 + i, lr)
        end.record()
        torch.cuda.synchronize()
        results['cuda_update_vectorized'] = start.elapsed_time(end) / n_iter
        print(f"    Time: {results['cuda_update_vectorized']:.3f} ms")

        # ===================
        # CUDA Update (basic)
        # ===================
        print("\n[3] CUDA Update with Runtime Grad (basic)...")

        for _ in range(warmup):
            params.copy_(params_backup)
            single_gpu_opt_cuda.fused_update_runtime_grad_basic(params, grad, 42, lr)
        torch.cuda.synchronize()

        start.record()
        for i in range(n_iter):
            params.copy_(params_backup)
            single_gpu_opt_cuda.fused_update_runtime_grad_basic(params, grad, 42 + i, lr)
        end.record()
        torch.cuda.synchronize()
        results['cuda_update_basic'] = start.elapsed_time(end) / n_iter
        print(f"    Time: {results['cuda_update_basic']:.3f} ms")

    return results


def benchmark_grad_compute(device: torch.device, n_iter: int = 1000) -> Dict[str, float]:
    """Benchmark gradient computation kernel."""
    dtype = torch.float32
    eps = 1e-3

    loss1 = torch.tensor([2.5], dtype=dtype, device=device)
    loss2 = torch.tensor([2.3], dtype=dtype, device=device)
    grad_out = torch.zeros(1, dtype=dtype, device=device)

    results = {}

    print(f"\n{'='*70}")
    print(f"PROJECTED GRAD COMPUTATION BENCHMARK")
    print(f"{'='*70}")

    # ===================
    # Triton Grad Compute
    # ===================
    print("\n[1] Triton compute_projected_grad...")

    for _ in range(100):
        compute_projected_grad_kernel[(1,)](loss1, loss2, eps, grad_out)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iter):
        compute_projected_grad_kernel[(1,)](loss1, loss2, eps, grad_out)
    end.record()
    torch.cuda.synchronize()
    results['triton_grad'] = start.elapsed_time(end) / n_iter
    print(f"    Time: {results['triton_grad']*1000:.3f} µs")

    # ===================
    # CUDA Grad Compute
    # ===================
    if HAS_CUDA:
        print("\n[2] CUDA compute_projected_grad...")

        for _ in range(100):
            single_gpu_opt_cuda.compute_projected_grad(loss1, loss2, eps, grad_out)
        torch.cuda.synchronize()

        start.record()
        for _ in range(n_iter):
            single_gpu_opt_cuda.compute_projected_grad(loss1, loss2, eps, grad_out)
        end.record()
        torch.cuda.synchronize()
        results['cuda_grad'] = start.elapsed_time(end) / n_iter
        print(f"    Time: {results['cuda_grad']*1000:.3f} µs")

    # ===================
    # Python .item() baseline
    # ===================
    print("\n[3] Python .item() baseline...")
    start.record()
    for _ in range(n_iter):
        grad = (loss1.item() - loss2.item()) / (2 * eps)
    end.record()
    torch.cuda.synchronize()
    results['python_item'] = start.elapsed_time(end) / n_iter
    print(f"    Time: {results['python_item']*1000:.3f} µs")

    return results


def verify_correctness(n_elements: int, device: torch.device):
    """Verify CUDA and Triton kernels produce the same results."""
    print(f"\n{'='*70}")
    print(f"CORRECTNESS VERIFICATION")
    print(f"{'='*70}")

    dtype = torch.float32
    eps = 1e-3
    seed = 12345

    anchor = torch.randn(n_elements, dtype=dtype, device=device)

    # Triton results
    triton_plus = torch.empty_like(anchor)
    triton_minus = torch.empty_like(anchor)
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, BLOCK_SIZE),)
    fused_dual_perturb_kernel[grid](
        triton_plus, triton_minus, anchor, seed, eps, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    torch.cuda.synchronize()

    if HAS_CUDA:
        # CUDA results
        cuda_plus = torch.empty_like(anchor)
        cuda_minus = torch.empty_like(anchor)
        single_gpu_opt_cuda.fused_dual_perturb(cuda_plus, cuda_minus, anchor, seed, eps)
        torch.cuda.synchronize()

        # Compare
        plus_match = torch.allclose(triton_plus, cuda_plus, rtol=1e-4, atol=1e-5)
        minus_match = torch.allclose(triton_minus, cuda_minus, rtol=1e-4, atol=1e-5)

        if plus_match and minus_match:
            print("✓ CUDA and Triton dual-perturb results match!")
        else:
            print("✗ CUDA and Triton results DIFFER!")
            if not plus_match:
                diff = (triton_plus - cuda_plus).abs()
                print(f"  Plus: max_diff={diff.max():.2e}, mean_diff={diff.mean():.2e}")
            if not minus_match:
                diff = (triton_minus - cuda_minus).abs()
                print(f"  Minus: max_diff={diff.max():.2e}, mean_diff={diff.mean():.2e}")

        # Verify grad computation
        loss1 = torch.tensor([2.5], dtype=dtype, device=device)
        loss2 = torch.tensor([2.3], dtype=dtype, device=device)

        triton_grad = torch.zeros(1, dtype=dtype, device=device)
        cuda_grad = torch.zeros(1, dtype=dtype, device=device)

        compute_projected_grad_kernel[(1,)](loss1, loss2, eps, triton_grad)
        single_gpu_opt_cuda.compute_projected_grad(loss1, loss2, eps, cuda_grad)
        torch.cuda.synchronize()

        expected = (2.5 - 2.3) / (2 * eps)
        triton_val = triton_grad.item()
        cuda_val = cuda_grad.item()

        print(f"\nGrad computation:")
        print(f"  Expected: {expected:.6f}")
        print(f"  Triton:   {triton_val:.6f}")
        print(f"  CUDA:     {cuda_val:.6f}")

        if abs(triton_val - expected) < 1e-4 and abs(cuda_val - expected) < 1e-4:
            print("✓ Grad computation correct!")
        else:
            print("✗ Grad computation mismatch!")
    else:
        print("Skipping CUDA verification (not available)")


def print_summary(dual_results: Dict, update_results: Dict, grad_results: Dict, baseline_time: float):
    """Print summary comparison table."""
    print(f"\n{'='*80}")
    print("SUMMARY: CUDA vs Triton Performance")
    print(f"{'='*80}")

    print(f"\n{'Kernel':<40} {'Time (ms)':<12} {'Speedup':<10}")
    print("-" * 62)

    # Dual perturb
    print("\n[Dual Perturb]")
    for name, time_ms in dual_results.items():
        speedup = baseline_time / time_ms if time_ms > 0 else 0
        print(f"  {name:<38} {time_ms:<12.3f} {speedup:<10.2f}x")

    # Update
    print("\n[Update with Runtime Grad]")
    for name, time_ms in update_results.items():
        print(f"  {name:<38} {time_ms:<12.3f}")

    # Grad compute
    print("\n[Grad Computation]")
    for name, time_ms in grad_results.items():
        print(f"  {name:<38} {time_ms*1000:<12.3f} µs")

    print(f"\n{'='*80}")

    # Best performers
    if dual_results:
        best_dual = min(dual_results.items(), key=lambda x: x[1])
        print(f"Best Dual-Perturb: {best_dual[0]} ({best_dual[1]:.3f} ms)")
        print(f"  Speedup vs baseline: {baseline_time/best_dual[1]:.2f}x")


def main():
    parser = argparse.ArgumentParser(description="Benchmark CUDA vs Triton kernels")
    parser.add_argument('--device', type=int, default=0, help='GPU device ID')
    parser.add_argument('--n_elements', type=int, default=331_196_416,
                        help='Number of elements (default: OPT-350M)')
    parser.add_argument('--n_iter', type=int, default=100, help='Benchmark iterations')
    parser.add_argument('--verify', action='store_true', help='Run correctness verification')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.device}')
    print(f"Device: {device} ({torch.cuda.get_device_name(device)})")
    print(f"Testing with n_elements = {args.n_elements:,}")

    if args.verify:
        verify_correctness(min(args.n_elements, 10_000_000), device)

    # Benchmarks
    dual_results = benchmark_dual_perturb(args.n_elements, device, args.n_iter)
    update_results = benchmark_update_with_grad(args.n_elements, device, args.n_iter)
    grad_results = benchmark_grad_compute(device)

    # Summary
    baseline_time = dual_results.get('baseline_2x_triton', 1.0)
    print_summary(dual_results, update_results, grad_results, baseline_time)


if __name__ == "__main__":
    main()
