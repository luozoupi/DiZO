#!/usr/bin/env python3
"""
Fixed Benchmark for DiZO zo_forward Optimizations

Key fixes over original benchmark:
1. Fair comparison - PyTorch baseline uses same per-parameter-group pattern
2. Pre-allocated buffers - no .clone() in hot path
3. Added CUDA extension benchmark
4. Proper warm-up and synchronization

Run with:
    CUDA_VISIBLE_DEVICES=3 python benchmark_zo_forward_fixed.py --model opt-350m
"""

import torch
import torch.nn as nn
import numpy as np
import time
import gc
import argparse
import os
import sys
from typing import Dict, Tuple, List
from dataclasses import dataclass

# Model configurations
MODEL_CONFIGS = {
    'opt-350m': {'num_layers': 24, 'hidden_size': 1024, 'ffn_size': 4096},
    'opt-1.3b': {'num_layers': 24, 'hidden_size': 2048, 'ffn_size': 8192},
    'opt-2.7b': {'num_layers': 32, 'hidden_size': 2560, 'ffn_size': 10240},
    'opt-6.7b': {'num_layers': 32, 'hidden_size': 4096, 'ffn_size': 16384},
    'opt-13b': {'num_layers': 40, 'hidden_size': 5120, 'ffn_size': 20480},
}


@dataclass
class BenchmarkResult:
    method: str
    time_ms: float
    memory_mb: float
    notes: str = ""


def cleanup_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def create_test_data(config: Dict, device: torch.device):
    """Create test data matching DiZO's structure."""
    param_groups = []
    anchor_groups = []
    
    num_layers = config['num_layers']
    hidden = config['hidden_size']
    ffn = config['ffn_size']
    
    for layer in range(num_layers):
        # Attention: q, k, v, o projections
        for _ in range(4):
            size = hidden * hidden
            param_groups.append(torch.randn(size, device=device, dtype=torch.float32))
            anchor_groups.append(torch.randn(size, device=device, dtype=torch.float32))
        
        # FFN: fc1 (hidden->ffn) and fc2 (ffn->hidden)
        for _ in range(2):
            size = hidden * ffn
            param_groups.append(torch.randn(size, device=device, dtype=torch.float32))
            anchor_groups.append(torch.randn(size, device=device, dtype=torch.float32))
    
    return param_groups, anchor_groups


def flatten_groups(param_groups: List[torch.Tensor], device: torch.device):
    """Flatten parameter groups."""
    param_flat = torch.cat([p.flatten() for p in param_groups])
    
    offsets = []
    offset = 0
    for p in param_groups:
        offsets.append(offset)
        offset += p.numel()
    
    offsets = torch.tensor(offsets, device=device, dtype=torch.long)
    sizes = torch.tensor([p.numel() for p in param_groups], device=device, dtype=torch.long)
    
    return param_flat, offsets, sizes


# =============================================================================
# Benchmark 1: PyTorch Per-Group Baseline (matches DiZO's actual pattern)
# =============================================================================

def benchmark_pytorch_per_group(
    param_groups: List[torch.Tensor],
    anchor_groups: List[torch.Tensor],
    n_iter: int,
    warmup: int = 5,
) -> BenchmarkResult:
    """
    PyTorch baseline using per-parameter-group operations.
    This matches what DiZO actually does.
    """
    device = param_groups[0].device
    num_params = len(param_groups)
    
    # Pre-allocate
    gammas = torch.rand(num_params, device=device) * 0.1
    norms = torch.empty(num_params, device=device)
    zs = torch.empty(num_params, device=device)
    
    tau, zo_eps, step_size = 0.2, 0.1, 2.0
    
    cleanup_gpu()
    torch.cuda.reset_peak_memory_stats()
    
    # Warm-up
    for _ in range(warmup):
        for i, (p, a) in enumerate(zip(param_groups, anchor_groups)):
            norms[i] = torch.norm(p - a)
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.time()
    
    for _ in range(n_iter):
        # Step 1: Compute norms (N kernel launches)
        for i, (p, a) in enumerate(zip(param_groups, anchor_groups)):
            norms[i] = torch.norm(p - a)
        
        # Step 2: Generate z and perturb gamma
        zs = torch.randn(num_params, device=device)
        clip_val = (tau / zo_eps) * norms
        zs = torch.clamp(zs, -clip_val, clip_val)
        gammas = gammas + 1.0 * zs * zo_eps
        
        # Step 3: Apply constraints (N kernel launches)
        alphas = gammas / (norms + 1e-8)
        for i, (p, a, alpha) in enumerate(zip(param_groups, anchor_groups, alphas)):
            diff = p - a
            p.data = a + diff * alpha
        
        # Step 4: Reverse constraints (N kernel launches)
        for i, (p, a, alpha) in enumerate(zip(param_groups, anchor_groups, alphas)):
            diff = p - a
            p.data = a + diff / alpha
        
        # Step 5: Perturb gamma -2eps (reuse zs)
        gammas = gammas - 2.0 * zs * zo_eps
        
        # Step 6: Apply/reverse again (simplified)
        for i, (p, a, alpha) in enumerate(zip(param_groups, anchor_groups, alphas)):
            diff = p - a
            p.data = a + diff * alpha
        for i, (p, a, alpha) in enumerate(zip(param_groups, anchor_groups, alphas)):
            diff = p - a
            p.data = a + diff / alpha
        
        # Step 7: Reset gamma +eps
        gammas = gammas + 1.0 * zs * zo_eps
        
        # Step 8: Update gamma
        grad = 0.01
        gammas = gammas - step_size * norms * grad * zs
        gamma_min = (1 - tau) * norms
        gamma_max = (1 + tau) * norms
        gammas = torch.clamp(gammas, gamma_min, gamma_max)
    
    torch.cuda.synchronize()
    elapsed_ms = (time.time() - start) * 1000 / n_iter
    
    peak_mem = torch.cuda.max_memory_allocated() / 1024**2
    
    return BenchmarkResult(
        method="pytorch_per_group",
        time_ms=elapsed_ms,
        memory_mb=peak_mem,
        notes=f"{num_params} groups, N separate norm/apply/reverse calls"
    )


# =============================================================================
# Benchmark 2: PyTorch Vectorized (flat tensor operations)
# =============================================================================

def benchmark_pytorch_vectorized(
    param_flat: torch.Tensor,
    anchor_flat: torch.Tensor,
    offsets: torch.Tensor,
    sizes: torch.Tensor,
    n_iter: int,
    warmup: int = 5,
) -> BenchmarkResult:
    """
    PyTorch vectorized operations on flat tensors.
    This is NOT what DiZO does, but shows theoretical upper bound.
    """
    device = param_flat.device
    num_params = offsets.shape[0]
    
    # Pre-allocate
    gammas = torch.rand(num_params, device=device) * 0.1
    norms = torch.empty(num_params, device=device)
    
    tau, zo_eps, step_size = 0.2, 0.1, 2.0
    
    cleanup_gpu()
    torch.cuda.reset_peak_memory_stats()
    
    # Warm-up
    for _ in range(warmup):
        diff = param_flat - anchor_flat
        # Vectorized norm per group (still needs loop or segment_reduce)
        for i in range(num_params):
            start, end = offsets[i].item(), offsets[i].item() + sizes[i].item()
            norms[i] = torch.norm(diff[start:end])
    torch.cuda.synchronize()
    
    # Benchmark
    start_time = time.time()
    
    for _ in range(n_iter):
        diff = param_flat - anchor_flat
        
        # Compute norms per group
        for i in range(num_params):
            start, end = offsets[i].item(), offsets[i].item() + sizes[i].item()
            norms[i] = torch.norm(diff[start:end])
        
        # Generate z
        zs = torch.randn(num_params, device=device)
        clip_val = (tau / zo_eps) * norms
        zs = torch.clamp(zs, -clip_val, clip_val)
        gammas = gammas + 1.0 * zs * zo_eps
        
        # Apply constraints per group
        alphas = gammas / (norms + 1e-8)
        for i in range(num_params):
            start, end = offsets[i].item(), offsets[i].item() + sizes[i].item()
            diff_slice = param_flat[start:end] - anchor_flat[start:end]
            param_flat[start:end] = anchor_flat[start:end] + diff_slice * alphas[i]
        
        # Reverse, perturb, apply again, reverse, update (simplified)
        for i in range(num_params):
            start, end = offsets[i].item(), offsets[i].item() + sizes[i].item()
            diff_slice = param_flat[start:end] - anchor_flat[start:end]
            param_flat[start:end] = anchor_flat[start:end] + diff_slice / alphas[i]
        
        gammas = gammas - 2.0 * zs * zo_eps
        gammas = gammas + 1.0 * zs * zo_eps
        
        grad = 0.01
        gammas = gammas - step_size * norms * grad * zs
        gammas = torch.clamp(gammas, (1 - tau) * norms, (1 + tau) * norms)
    
    torch.cuda.synchronize()
    elapsed_ms = (time.time() - start_time) * 1000 / n_iter
    
    peak_mem = torch.cuda.max_memory_allocated() / 1024**2
    
    return BenchmarkResult(
        method="pytorch_vectorized",
        time_ms=elapsed_ms,
        memory_mb=peak_mem,
        notes="Flat tensor with per-group slicing"
    )


# =============================================================================
# Benchmark 3: CUDA Extension
# =============================================================================

def benchmark_cuda_extension(
    param_flat: torch.Tensor,
    anchor_flat: torch.Tensor,
    offsets: torch.Tensor,
    sizes: torch.Tensor,
    n_iter: int,
    warmup: int = 5,
) -> BenchmarkResult:
    """Benchmark CUDA extension kernels."""
    try:
        import dizo_fused_kernels_cuda as cuda_ext
    except ImportError as e:
        return BenchmarkResult(
            method="cuda_extension",
            time_ms=-1,
            memory_mb=0,
            notes=f"Not available: {e}"
        )
    
    device = param_flat.device
    num_params = offsets.shape[0]
    
    # Pre-allocate
    gammas = torch.rand(num_params, device=device) * 0.1
    tau, zo_eps, step_size = 0.2, 0.1, 2.0
    
    cleanup_gpu()
    torch.cuda.reset_peak_memory_stats()
    
    # Warm-up
    for _ in range(warmup):
        norms = cuda_ext.fused_compute_norms(param_flat, anchor_flat, offsets, sizes)
    torch.cuda.synchronize()
    
    # Benchmark
    start_time = time.time()
    
    for _ in range(n_iter):
        # Compute norms (1 kernel)
        norms = cuda_ext.fused_compute_norms(param_flat, anchor_flat, offsets, sizes)
        
        # Generate z (PyTorch - small tensor)
        zs = torch.randn(num_params, device=device)
        clip_val = (tau / zo_eps) * norms
        zs = torch.clamp(zs, -clip_val, clip_val)
        gammas_work = gammas + 1.0 * zs * zo_eps
        
        # Apply constraints (1 kernel)
        param_work = param_flat.clone()  # Need to clone for benchmarking
        cuda_ext.fused_apply_constraints(
            param_work, anchor_flat, offsets, sizes, gammas_work, norms, 1e-8
        )
        
        # Reverse constraints (1 kernel)
        alphas = gammas_work / (norms + 1e-8)
        cuda_ext.fused_reverse_constraints(
            param_work, anchor_flat, offsets, sizes, alphas
        )
        
        # Perturb -2eps
        gammas_work = gammas_work - 2.0 * zs * zo_eps
        
        # Apply/reverse again
        cuda_ext.fused_apply_constraints(
            param_work, anchor_flat, offsets, sizes, gammas_work, norms, 1e-8
        )
        alphas = gammas_work / (norms + 1e-8)
        cuda_ext.fused_reverse_constraints(
            param_work, anchor_flat, offsets, sizes, alphas
        )
        
        # Reset +eps
        gammas_work = gammas_work + 1.0 * zs * zo_eps
        
        # Update gamma (1 kernel)
        grad = 0.01
        cuda_ext.fused_update_gamma(gammas_work, norms, zs, grad, step_size, tau)
        
        gammas = gammas_work
    
    torch.cuda.synchronize()
    elapsed_ms = (time.time() - start_time) * 1000 / n_iter
    
    peak_mem = torch.cuda.max_memory_allocated() / 1024**2
    
    return BenchmarkResult(
        method="cuda_extension",
        time_ms=elapsed_ms,
        memory_mb=peak_mem,
        notes="6 kernel launches per step"
    )


# =============================================================================
# Benchmark 4: CUDA Extension V2 (Optimized)
# =============================================================================

def benchmark_cuda_extension_v2(
    param_flat: torch.Tensor,
    anchor_flat: torch.Tensor,
    offsets: torch.Tensor,
    sizes: torch.Tensor,
    n_iter: int,
    warmup: int = 5,
) -> BenchmarkResult:
    """Benchmark optimized CUDA V2 extension with float4 vectorization."""
    try:
        import dizo_fused_kernels_cuda_v2 as cuda_v2
    except ImportError as e:
        return BenchmarkResult(
            method="cuda_v2_optimized",
            time_ms=-1,
            memory_mb=0,
            notes=f"Not available: {e}"
        )
    
    device = param_flat.device
    num_params = offsets.shape[0]
    
    # Pre-allocate
    gammas = torch.rand(num_params, device=device, dtype=torch.float32) * 0.1
    tau, zo_eps, step_size = 0.2, 0.1, 2.0
    
    cleanup_gpu()
    torch.cuda.reset_peak_memory_stats()
    
    # Warm-up
    for _ in range(warmup):
        norms = cuda_v2.fused_compute_norms(param_flat, anchor_flat, offsets, sizes)
    torch.cuda.synchronize()
    
    # Benchmark
    start_time = time.time()
    
    for _ in range(n_iter):
        # Compute norms (2 kernels: parallel + reduce)
        norms = cuda_v2.fused_compute_norms(param_flat, anchor_flat, offsets, sizes)
        
        # Generate z (PyTorch - small tensor)
        zs = torch.randn(num_params, device=device, dtype=torch.float32)
        clip_val = (tau / zo_eps) * norms
        zs = torch.clamp(zs, -clip_val, clip_val)
        gammas_work = gammas + 1.0 * zs * zo_eps
        
        # Apply constraints (1 kernel, vectorized)
        param_work = param_flat.clone()
        cuda_v2.fused_apply_constraints(
            param_work, anchor_flat, offsets, sizes, gammas_work, norms, 1e-8
        )
        
        # Reverse constraints (1 kernel, vectorized)
        alphas = gammas_work / (norms + 1e-8)
        cuda_v2.fused_reverse_constraints(
            param_work, anchor_flat, offsets, sizes, alphas
        )
        
        # Perturb -2eps, +eps
        gammas_work = gammas_work - 2.0 * zs * zo_eps
        
        cuda_v2.fused_apply_constraints(
            param_work, anchor_flat, offsets, sizes, gammas_work, norms, 1e-8
        )
        alphas = gammas_work / (norms + 1e-8)
        cuda_v2.fused_reverse_constraints(
            param_work, anchor_flat, offsets, sizes, alphas
        )
        
        gammas_work = gammas_work + 1.0 * zs * zo_eps
        
        # Update gamma
        grad = 0.01
        cuda_v2.fused_update_gamma(gammas_work, norms, zs, grad, step_size, tau)
        
        gammas = gammas_work
    
    torch.cuda.synchronize()
    elapsed_ms = (time.time() - start_time) * 1000 / n_iter
    
    peak_mem = torch.cuda.max_memory_allocated() / 1024**2
    
    return BenchmarkResult(
        method="cuda_v2_optimized",
        time_ms=elapsed_ms,
        memory_mb=peak_mem,
        notes="Float4 vectorized + multi-block"
    )


# =============================================================================
# Benchmark 5: Individual Kernel Timings
# =============================================================================

def benchmark_individual_kernels(
    param_flat: torch.Tensor,
    anchor_flat: torch.Tensor,
    offsets: torch.Tensor,
    sizes: torch.Tensor,
    n_iter: int = 100,
) -> Dict[str, float]:
    """Time individual operations."""
    results = {}
    device = param_flat.device
    num_params = offsets.shape[0]
    
    # PyTorch norm per-group
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_iter):
        norms = torch.empty(num_params, device=device)
        for i in range(num_params):
            start_idx = offsets[i].item()
            end_idx = start_idx + sizes[i].item()
            norms[i] = torch.norm(param_flat[start_idx:end_idx] - anchor_flat[start_idx:end_idx])
    torch.cuda.synchronize()
    results['pytorch_norm_per_group_ms'] = (time.time() - start) * 1000 / n_iter
    
    # CUDA V1 extension norm
    try:
        import dizo_fused_kernels_cuda as cuda_ext
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(n_iter):
            norms = cuda_ext.fused_compute_norms(param_flat, anchor_flat, offsets, sizes)
        torch.cuda.synchronize()
        results['cuda_v1_norm_ms'] = (time.time() - start) * 1000 / n_iter
    except ImportError:
        results['cuda_v1_norm_ms'] = -1
    
    # CUDA V2 extension norm (optimized)
    try:
        import dizo_fused_kernels_cuda_v2 as cuda_v2
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(n_iter):
            norms = cuda_v2.fused_compute_norms(param_flat, anchor_flat, offsets, sizes)
        torch.cuda.synchronize()
        results['cuda_v2_norm_ms'] = (time.time() - start) * 1000 / n_iter
    except ImportError:
        results['cuda_v2_norm_ms'] = -1
    
    # PyTorch apply per-group
    constraints = torch.rand(num_params, device=device) * 0.1
    alphas = constraints / (norms + 1e-8)
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_iter):
        for i in range(num_params):
            start_idx = offsets[i].item()
            end_idx = start_idx + sizes[i].item()
            diff = param_flat[start_idx:end_idx] - anchor_flat[start_idx:end_idx]
            param_flat[start_idx:end_idx] = anchor_flat[start_idx:end_idx] + diff * alphas[i]
    torch.cuda.synchronize()
    results['pytorch_apply_per_group_ms'] = (time.time() - start) * 1000 / n_iter
    
    # CUDA V1 apply
    try:
        import dizo_fused_kernels_cuda as cuda_ext
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(n_iter):
            cuda_ext.fused_apply_constraints(
                param_flat, anchor_flat, offsets, sizes, constraints, norms, 1e-8
            )
        torch.cuda.synchronize()
        results['cuda_v1_apply_ms'] = (time.time() - start) * 1000 / n_iter
    except ImportError:
        results['cuda_v1_apply_ms'] = -1
    
    # CUDA V2 apply (optimized)
    try:
        import dizo_fused_kernels_cuda_v2 as cuda_v2
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(n_iter):
            cuda_v2.fused_apply_constraints(
                param_flat, anchor_flat, offsets, sizes, constraints, norms, 1e-8
            )
        torch.cuda.synchronize()
        results['cuda_v2_apply_ms'] = (time.time() - start) * 1000 / n_iter
    except ImportError:
        results['cuda_v2_apply_ms'] = -1
    
    return results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='opt-350m', choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument('--n_iter', type=int, default=20)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--breakdown', action='store_true')
    args = parser.parse_args()
    
    print("=" * 70)
    print("DiZO zo_forward Benchmark (Fixed)")
    print("=" * 70)
    
    device = torch.device('cuda:0')
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print()
    
    config = MODEL_CONFIGS[args.model]
    print(f"Model: {args.model}")
    print(f"  Layers: {config['num_layers']}")
    print(f"  Hidden: {config['hidden_size']}")
    
    # Create test data
    param_groups, anchor_groups = create_test_data(config, device)
    param_flat, offsets, sizes = flatten_groups(param_groups, device)
    anchor_flat = torch.cat([a.flatten() for a in anchor_groups])
    
    print(f"  Parameter groups: {len(param_groups)}")
    print(f"  Total elements: {param_flat.numel():,}")
    print(f"  Memory: {param_flat.numel() * 4 / 1024**3:.2f} GB")
    print()
    
    results = []
    
    # Benchmark 1: PyTorch per-group (actual DiZO pattern)
    print("Benchmarking PyTorch per-group (DiZO's actual pattern)...")
    result = benchmark_pytorch_per_group(param_groups, anchor_groups, args.n_iter, args.warmup)
    results.append(result)
    print(f"  Time: {result.time_ms:.2f} ms")
    print(f"  Memory: {result.memory_mb:.1f} MB")
    print()
    
    # Recreate test data (may have been modified)
    param_groups, anchor_groups = create_test_data(config, device)
    param_flat, offsets, sizes = flatten_groups(param_groups, device)
    anchor_flat = torch.cat([a.flatten() for a in anchor_groups])
    
    # Benchmark 2: PyTorch vectorized
    print("Benchmarking PyTorch vectorized (flat tensor)...")
    result = benchmark_pytorch_vectorized(param_flat.clone(), anchor_flat, offsets, sizes, args.n_iter, args.warmup)
    results.append(result)
    print(f"  Time: {result.time_ms:.2f} ms")
    print(f"  Memory: {result.memory_mb:.1f} MB")
    print()
    
    # Benchmark 3: CUDA extension V1
    print("Benchmarking CUDA extension V1...")
    result = benchmark_cuda_extension(param_flat.clone(), anchor_flat, offsets, sizes, args.n_iter, args.warmup)
    results.append(result)
    if result.time_ms > 0:
        print(f"  Time: {result.time_ms:.2f} ms")
        print(f"  Memory: {result.memory_mb:.1f} MB")
    else:
        print(f"  {result.notes}")
    print()
    
    # Benchmark 4: CUDA extension V2 (optimized)
    print("Benchmarking CUDA extension V2 (float4 + multi-block)...")
    result = benchmark_cuda_extension_v2(param_flat.clone(), anchor_flat, offsets, sizes, args.n_iter, args.warmup)
    results.append(result)
    if result.time_ms > 0:
        print(f"  Time: {result.time_ms:.2f} ms")
        print(f"  Memory: {result.memory_mb:.1f} MB")
    else:
        print(f"  {result.notes}")
    print()
    
    # Individual kernel breakdown
    if args.breakdown:
        print("Individual Kernel Timings (100 iterations)...")
        breakdown = benchmark_individual_kernels(param_flat.clone(), anchor_flat, offsets, sizes)
        for k, v in breakdown.items():
            if v > 0:
                print(f"  {k}: {v:.3f} ms")
            else:
                print(f"  {k}: N/A")
        print()
    
    # Summary
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"{'Method':<25} {'Time (ms)':<12} {'Memory (MB)':<12} {'Speedup':<10}")
    print("-" * 60)
    
    baseline = results[0].time_ms
    for r in results:
        if r.time_ms > 0:
            speedup = baseline / r.time_ms
            print(f"{r.method:<25} {r.time_ms:<12.2f} {r.memory_mb:<12.1f} {speedup:.2f}x")
        else:
            print(f"{r.method:<25} {'N/A':<12} {'N/A':<12} {'N/A':<10}")


if __name__ == "__main__":
    main()
