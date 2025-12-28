#!/usr/bin/env python3
"""
Comprehensive Benchmark for DiZO zo_forward Optimizations

This script benchmarks:
1. Original DiZO zo_forward (PyTorch per-parameter loops)
2. V1 Fused Kernels (dizo_fused_kernels.py)
3. V2 Fused Kernels (dizo_fused_kernels_v2.py) 
4. Individual kernel operations

Metrics:
- Execution time (ms)
- Memory usage (MB)
- Kernel launch count
- Numerical correctness

Usage:
    python benchmark_zo_forward.py --model opt-350m --n_iter 20
    python benchmark_zo_forward.py --model opt-13b --n_iter 5
    python benchmark_zo_forward.py --breakdown  # Profile individual kernels
"""

import torch
import torch.nn as nn
import numpy as np
import time
import gc
import argparse
import os
import sys
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
from datetime import datetime

# Model size presets (EXACT values from profiled OPT parameter shapes)
# Each layer has 16 params: 4 attn projs (w+b), 2 FFN (w+b), 2 layer norms (w+b)
MODEL_CONFIGS = {
    'opt-350m': {
        'num_layers': 24,
        'hidden_size': 1024,
        'ffn_size': 4096,
        'embed_dim': 512,        # OPT-350m uses word_embed_proj_dim=512
        'has_project': True,     # Has project_in/project_out
        'num_heads': 16,
        'total_params': 331_196_416,
        'num_tensors': 388,      # From profiling
    },
    'opt-1.3b': {
        'num_layers': 24,
        'hidden_size': 2048,
        'ffn_size': 8192,
        'embed_dim': 2048,
        'has_project': False,
        'num_heads': 32,
        'total_params': 1_315_753_984,
        'num_tensors': 386,
    },
    'opt-2.7b': {
        'num_layers': 32,
        'hidden_size': 2560,
        'ffn_size': 10240,
        'embed_dim': 2560,
        'has_project': False,
        'num_heads': 32,
        'total_params': 2_651_596_800,  # From profiled structure
        'num_tensors': 514,
    },
    'opt-6.7b': {
        'num_layers': 32,
        'hidden_size': 4096,
        'ffn_size': 16384,
        'embed_dim': 4096,
        'has_project': False,
        'num_heads': 32,
        'total_params': 6_658_473_984,  # Exact from profiling
        'num_tensors': 516,
    },
    'opt-13b': {
        'num_layers': 40,
        'hidden_size': 5120,
        'ffn_size': 20480,
        'embed_dim': 5120,
        'has_project': False,
        'num_heads': 40,
        'total_params': 13_016_023_040,  # From profiled structure
        'num_tensors': 644,
    },
}


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    method: str
    time_ms: float
    memory_mb: float
    kernel_launches: int = 0
    correctness: bool = True
    notes: str = ""


def get_gpu_memory_mb() -> float:
    """Get current GPU memory allocated in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0.0


def cleanup_gpu():
    """Clean up GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def create_mock_dizo_params(config: Dict, device: torch.device) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[int]]:
    """
    Create mock parameter groups matching REAL OPT model structure.
    
    Based on profiled parameter shapes:
    - Embedding layers (embed_tokens, embed_positions, project_in/out)
    - Each transformer layer has 16 params:
      - 4 attention projections (q,k,v,out) × (weight + bias) = 8
      - 2 FFN layers (fc1, fc2) × (weight + bias) = 4
      - 2 layer norms (self_attn, final) × (weight + bias) = 4
    - Decoder final_layer_norm (weight + bias)
    """
    param_groups = []
    anchor_groups = []
    sizes = []
    
    num_layers = config['num_layers']
    hidden = config['hidden_size']
    ffn = config['ffn_size']
    embed_dim = config.get('embed_dim', hidden)
    has_project = config.get('has_project', embed_dim != hidden)
    vocab_size = 50272  # OPT vocab size
    max_pos = 2050      # OPT max positions
    
    def add_param(size):
        param_groups.append(torch.randn(size, device=device, dtype=torch.float32))
        anchor_groups.append(torch.randn(size, device=device, dtype=torch.float32))
        sizes.append(size)
    
    # === Embedding layers ===
    add_param(vocab_size * embed_dim)  # embed_tokens
    add_param(max_pos * hidden)        # embed_positions
    
    # Decoder final_layer_norm
    add_param(hidden)  # weight
    add_param(hidden)  # bias
    
    # project_in/project_out (OPT-350m only)
    if has_project:
        add_param(hidden * embed_dim)  # project_in
        add_param(embed_dim * hidden)  # project_out
    
    # === Transformer layers (16 params per layer) ===
    for layer in range(num_layers):
        # Self-attention projections (k, v, q, out) - each has weight + bias
        for proj in ['k', 'v', 'q', 'out']:
            add_param(hidden * hidden)  # weight
            add_param(hidden)           # bias
        
        # self_attn_layer_norm
        add_param(hidden)  # weight
        add_param(hidden)  # bias
        
        # FFN layers
        add_param(ffn * hidden)  # fc1.weight
        add_param(ffn)           # fc1.bias
        add_param(hidden * ffn)  # fc2.weight
        add_param(hidden)        # fc2.bias
        
        # final_layer_norm (per layer)
        add_param(hidden)  # weight
        add_param(hidden)  # bias

    
    return param_groups, anchor_groups, sizes


def flatten_param_groups(
    param_groups: List[torch.Tensor], 
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Flatten parameter groups into a single tensor."""
    param_flat = torch.cat([p.flatten() for p in param_groups])
    
    offsets = [0]
    for p in param_groups[:-1]:
        offsets.append(offsets[-1] + p.numel())
    
    offsets_tensor = torch.tensor(offsets, device=device, dtype=torch.long)
    sizes_tensor = torch.tensor([p.numel() for p in param_groups], device=device, dtype=torch.long)
    
    return param_flat, offsets_tensor, sizes_tensor


# =============================================================================
# Benchmark: Original PyTorch (Per-Parameter Loops)
# =============================================================================

def benchmark_pytorch_original(
    param_groups: List[torch.Tensor],
    anchor_groups: List[torch.Tensor],
    n_iter: int,
    tau: float = 0.2,
    zo_eps: float = 0.1,
    step_size: float = 2.0,
) -> BenchmarkResult:
    """
    Benchmark original PyTorch implementation (per-parameter loops).
    
    This mimics DiZO's original zo_forward with individual operations.
    """
    device = param_groups[0].device
    num_params = len(param_groups)
    
    # Initialize gamma constraints
    gammas = [torch.tensor([0.1], device=device) for _ in range(num_params)]
    
    cleanup_gpu()
    torch.cuda.reset_peak_memory_stats()
    mem_before = get_gpu_memory_mb()
    
    kernel_count = 0
    
    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(n_iter):
        # Compute norms (N kernel launches)
        norms = []
        for p, a in zip(param_groups, anchor_groups):
            norms.append(torch.norm(p - a))
            kernel_count += 1
        
        # Generate z values for gamma (N kernel launches)
        zs = []
        for i, n in enumerate(norms):
            z = torch.randn(1, device=device)
            clip_val = (tau / zo_eps) * n
            z = torch.clamp(z, -clip_val, clip_val)
            zs.append(z)
            kernel_count += 3  # randn, clamp (min), clamp (max)
        
        # Perturb gamma +eps (N ops)
        for i in range(num_params):
            gammas[i] = gammas[i] + 1.0 * zs[i] * zo_eps
            kernel_count += 1
        
        # Apply constraints (N kernel launches per param)
        for i, (p, a, g, n) in enumerate(zip(param_groups, anchor_groups, gammas, norms)):
            alpha = g / (n + 1e-8)
            p.data = a + (p - a) * alpha
            kernel_count += 4  # sub, div, sub, mul, add
        
        # Simulate loss computation (forward pass)
        # loss1 = ... (not counted here)
        
        # Reverse constraints (N kernel launches)
        for i, (p, a, g, n) in enumerate(zip(param_groups, anchor_groups, gammas, norms)):
            alpha = g / (n + 1e-8)
            p.data = a + (p - a) / alpha
            kernel_count += 4
        
        # Perturb gamma -2eps
        for i in range(num_params):
            gammas[i] = gammas[i] - 2.0 * zs[i] * zo_eps
            kernel_count += 1
        
        # Apply constraints again, loss2, reverse again, perturb +eps
        # (Same pattern repeated)
        
        # Update gamma
        grad = 0.01  # Mock gradient
        for i in range(num_params):
            gammas[i] = gammas[i] - step_size * norms[i] * grad * zs[i]
            gammas[i] = torch.clamp(gammas[i], (1-tau)*norms[i], (1+tau)*norms[i])
            kernel_count += 4
    
    torch.cuda.synchronize()
    elapsed = (time.time() - start) * 1000 / n_iter
    
    mem_peak = torch.cuda.max_memory_allocated() / 1024**2
    
    return BenchmarkResult(
        method="pytorch_original",
        time_ms=elapsed,
        memory_mb=mem_peak - mem_before,
        kernel_launches=kernel_count // n_iter,
        notes=f"{num_params} param groups"
    )


# =============================================================================
# Benchmark: V1 Fused Kernels
# =============================================================================

def benchmark_fused_v1(
    param_flat: torch.Tensor,
    anchor_flat: torch.Tensor,
    offsets: torch.Tensor,
    sizes: torch.Tensor,
    n_iter: int,
    tau: float = 0.2,
    zo_eps: float = 0.1,
    step_size: float = 2.0,
) -> BenchmarkResult:
    """Benchmark V1 fused kernels."""
    try:
        from dizo_fused_kernels import (
            fused_compute_norms,
            fused_apply_constraints,
            fused_reverse_constraints,
            fused_perturb_gamma,
            fused_update_gamma,
        )
    except ImportError as e:
        return BenchmarkResult(
            method="fused_v1",
            time_ms=-1,
            memory_mb=0,
            correctness=False,
            notes=f"Import error: {e}"
        )
    
    device = param_flat.device
    num_params = offsets.shape[0]
    
    # Initialize
    constraints = torch.rand(num_params, device=device, dtype=torch.float32) * 0.1
    
    cleanup_gpu()
    torch.cuda.reset_peak_memory_stats()
    mem_before = get_gpu_memory_mb()
    
    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(n_iter):
        # Compute norms (1 kernel)
        norms = fused_compute_norms(param_flat.clone(), anchor_flat, offsets, sizes)
        
        # Perturb gamma (PyTorch ops in V1)
        seed = torch.randint(0, 2**31, (1,)).item()
        zs = fused_perturb_gamma(constraints.clone(), norms, seed, 1.0, tau, zo_eps)
        
        # Apply constraints (1 kernel)
        param_work = param_flat.clone()
        fused_apply_constraints(param_work, anchor_flat, offsets, sizes, constraints, norms)
        
        # Reverse constraints (1 kernel)
        alphas = constraints / (norms + 1e-8)
        fused_reverse_constraints(param_work, anchor_flat, offsets, sizes, alphas)
        
        # Perturb gamma -2eps
        fused_perturb_gamma(constraints, norms, seed, -2.0, tau, zo_eps, zs)
        
        # Apply, reverse, perturb +eps again
        fused_apply_constraints(param_work, anchor_flat, offsets, sizes, constraints, norms)
        fused_reverse_constraints(param_work, anchor_flat, offsets, sizes, alphas)
        fused_perturb_gamma(constraints, norms, seed, 1.0, tau, zo_eps, zs)
        
        # Update gamma (1 kernel)
        grad = 0.01
        fused_update_gamma(constraints, norms, zs, grad, step_size, tau)
    
    torch.cuda.synchronize()
    elapsed = (time.time() - start) * 1000 / n_iter
    
    mem_peak = torch.cuda.max_memory_allocated() / 1024**2
    
    return BenchmarkResult(
        method="fused_v1",
        time_ms=elapsed,
        memory_mb=mem_peak - mem_before,
        kernel_launches=8,  # Approximate
        notes="Triton kernels"
    )


# =============================================================================
# Benchmark: V2 Fused Kernels
# =============================================================================

def benchmark_fused_v2(
    param_flat: torch.Tensor,
    anchor_flat: torch.Tensor,
    offsets: torch.Tensor,
    sizes: torch.Tensor,
    n_iter: int,
    tau: float = 0.2,
    zo_eps: float = 0.1,
    step_size: float = 2.0,
) -> BenchmarkResult:
    """Benchmark V2 fused kernels with pre-allocated buffers."""
    try:
        from dizo_fused_kernels_v2 import FusedDiZOKernelsV2
    except ImportError as e:
        return BenchmarkResult(
            method="fused_v2",
            time_ms=-1,
            memory_mb=0,
            correctness=False,
            notes=f"Import error: {e}"
        )
    
    device = param_flat.device
    num_params = offsets.shape[0]
    total_elements = param_flat.numel()
    
    # Initialize with pre-allocated buffers
    kernels = FusedDiZOKernelsV2(num_params, total_elements, device)
    constraints = torch.rand(num_params, device=device, dtype=torch.float32) * 0.1
    
    cleanup_gpu()
    torch.cuda.reset_peak_memory_stats()
    mem_before = get_gpu_memory_mb()
    
    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(n_iter):
        # Compute norms
        norms = kernels.compute_norms(param_flat.clone(), anchor_flat, offsets, sizes)
        
        # Perturb gamma +eps
        zs = kernels.perturb_gamma(constraints.clone(), norms, 1.0, tau, zo_eps, generate_new=True)
        
        # Apply constraints
        param_work = param_flat.clone()
        alphas = kernels.apply_constraints(param_work, anchor_flat, offsets, sizes, constraints, norms)
        
        # Reverse constraints
        kernels.reverse_constraints(param_work, anchor_flat, offsets, sizes, alphas)
        
        # Perturb gamma -2eps
        kernels.perturb_gamma(constraints, norms, -2.0, tau, zo_eps, generate_new=False)
        
        # Apply, reverse, perturb +eps
        kernels.apply_constraints(param_work, anchor_flat, offsets, sizes, constraints, norms)
        kernels.reverse_constraints(param_work, anchor_flat, offsets, sizes, alphas)
        kernels.perturb_gamma(constraints, norms, 1.0, tau, zo_eps, generate_new=False)
        
        # Update gamma
        grad = 0.01
        kernels.update_gamma(constraints, norms, grad, step_size, tau)
    
    torch.cuda.synchronize()
    elapsed = (time.time() - start) * 1000 / n_iter
    
    mem_peak = torch.cuda.max_memory_allocated() / 1024**2
    
    return BenchmarkResult(
        method="fused_v2",
        time_ms=elapsed,
        memory_mb=mem_peak - mem_before,
        kernel_launches=9,  # All fused
        notes="Pre-allocated buffers"
    )


# =============================================================================
# Individual Kernel Breakdown Benchmark
# =============================================================================

def benchmark_kernel_breakdown(
    param_flat: torch.Tensor,
    anchor_flat: torch.Tensor,
    offsets: torch.Tensor,
    sizes: torch.Tensor,
    n_iter: int = 100,
) -> Dict[str, float]:
    """Benchmark individual kernel operations."""
    results = {}
    device = param_flat.device
    num_params = offsets.shape[0]
    
    try:
        from dizo_fused_kernels import (
            fused_compute_norms,
            fused_apply_constraints,
            fused_reverse_constraints,
        )
    except ImportError:
        return {"error": "Could not import fused kernels"}
    
    # Benchmark norm computation
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_iter):
        norms = fused_compute_norms(param_flat, anchor_flat, offsets, sizes)
    torch.cuda.synchronize()
    results['norm_compute_ms'] = (time.time() - start) * 1000 / n_iter
    
    # Benchmark constraint application
    constraints = torch.rand(num_params, device=device) * 0.1
    param_work = param_flat.clone()
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_iter):
        fused_apply_constraints(param_work, anchor_flat, offsets, sizes, constraints, norms)
    torch.cuda.synchronize()
    results['apply_constraints_ms'] = (time.time() - start) * 1000 / n_iter
    
    # Benchmark constraint reversal
    alphas = constraints / (norms + 1e-8)
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_iter):
        fused_reverse_constraints(param_work, anchor_flat, offsets, sizes, alphas)
    torch.cuda.synchronize()
    results['reverse_constraints_ms'] = (time.time() - start) * 1000 / n_iter
    
    return results


# =============================================================================
# Numerical Correctness Test
# =============================================================================

def test_numerical_correctness(
    param_groups: List[torch.Tensor],
    anchor_groups: List[torch.Tensor],
) -> Dict[str, bool]:
    """Test numerical correctness of fused kernels against PyTorch reference."""
    results = {}
    device = param_groups[0].device
    num_params = len(param_groups)
    
    # Flatten
    param_flat = torch.cat([p.flatten() for p in param_groups])
    anchor_flat = torch.cat([a.flatten() for a in anchor_groups])
    
    offsets = [0]
    for p in param_groups[:-1]:
        offsets.append(offsets[-1] + p.numel())
    offsets = torch.tensor(offsets, device=device, dtype=torch.long)
    sizes = torch.tensor([p.numel() for p in param_groups], device=device, dtype=torch.long)
    
    try:
        from dizo_fused_kernels import fused_compute_norms, fused_apply_constraints
    except ImportError:
        return {"import_error": True}
    
    # Test norm computation
    norms_fused = fused_compute_norms(param_flat, anchor_flat, offsets, sizes)
    norms_ref = torch.tensor([
        torch.norm(p - a).item() for p, a in zip(param_groups, anchor_groups)
    ], device=device)
    
    norm_diff = torch.max(torch.abs(norms_fused - norms_ref)).item()
    results['norm_correct'] = norm_diff < 1e-4
    results['norm_max_diff'] = norm_diff
    
    # Test constraint application
    constraints = torch.rand(num_params, device=device) * 0.1
    
    param_fused = param_flat.clone()
    fused_apply_constraints(param_fused, anchor_flat, offsets, sizes, constraints, norms_fused)
    
    # Reference
    param_ref = param_flat.clone()
    for i, (offset, size) in enumerate(zip(offsets.tolist(), sizes.tolist())):
        alpha = constraints[i] / (norms_ref[i] + 1e-8)
        diff = param_ref[offset:offset+size] - anchor_flat[offset:offset+size]
        param_ref[offset:offset+size] = anchor_flat[offset:offset+size] + diff * alpha
    
    apply_diff = torch.max(torch.abs(param_fused - param_ref)).item()
    results['apply_correct'] = apply_diff < 1e-4
    results['apply_max_diff'] = apply_diff
    
    return results


# =============================================================================
# Main Benchmark Runner
# =============================================================================

def run_benchmarks(args):
    """Run all benchmarks."""
    print("=" * 70)
    print("DiZO zo_forward Optimization Benchmark")
    print("=" * 70)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU Memory: {total_mem:.1f} GB")
    print()
    
    config = MODEL_CONFIGS.get(args.model, MODEL_CONFIGS['opt-350m'])
    print(f"Model: {args.model}")
    print(f"  Layers: {config['num_layers']}")
    print(f"  Hidden: {config['hidden_size']}")
    print(f"  Estimated params: {config['total_params']:,}")
    print()
    
    # Create mock parameters
    print("Creating mock parameters...")
    param_groups, anchor_groups, sizes = create_mock_dizo_params(config, device)
    print(f"  Parameter groups: {len(param_groups)}")
    total_elements = sum(sizes)
    print(f"  Total elements: {total_elements:,}")
    print(f"  Memory: {total_elements * 4 / 1024**3:.2f} GB")
    print()
    
    # Flatten for fused kernels
    param_flat, offsets, sizes_tensor = flatten_param_groups(param_groups, device)
    anchor_flat = torch.cat([a.flatten() for a in anchor_groups])
    
    results = []
    
    # Run benchmarks
    if not args.skip_pytorch:
        print("Benchmarking PyTorch original...")
        result = benchmark_pytorch_original(
            param_groups, anchor_groups, args.n_iter,
            tau=args.tau, zo_eps=args.zo_eps, step_size=args.step_size
        )
        results.append(result)
        print(f"  Time: {result.time_ms:.2f} ms")
        print(f"  Memory: {result.memory_mb:.1f} MB")
        print(f"  Kernel launches: ~{result.kernel_launches}")
        print()
    
    print("Benchmarking Fused V1...")
    result = benchmark_fused_v1(
        param_flat.clone(), anchor_flat, offsets, sizes_tensor, args.n_iter,
        tau=args.tau, zo_eps=args.zo_eps, step_size=args.step_size
    )
    results.append(result)
    if result.time_ms > 0:
        print(f"  Time: {result.time_ms:.2f} ms")
        print(f"  Memory: {result.memory_mb:.1f} MB")
        print(f"  Kernel launches: ~{result.kernel_launches}")
    else:
        print(f"  Error: {result.notes}")
    print()
    
    print("Benchmarking Fused V2...")
    result = benchmark_fused_v2(
        param_flat.clone(), anchor_flat, offsets, sizes_tensor, args.n_iter,
        tau=args.tau, zo_eps=args.zo_eps, step_size=args.step_size
    )
    results.append(result)
    if result.time_ms > 0:
        print(f"  Time: {result.time_ms:.2f} ms")
        print(f"  Memory: {result.memory_mb:.1f} MB")
        print(f"  Kernel launches: ~{result.kernel_launches}")
    else:
        print(f"  Error: {result.notes}")
    print()
    
    # Kernel breakdown
    if args.breakdown:
        print("Kernel Breakdown Benchmark...")
        breakdown = benchmark_kernel_breakdown(
            param_flat, anchor_flat, offsets, sizes_tensor, n_iter=100
        )
        for k, v in breakdown.items():
            print(f"  {k}: {v:.3f}")
        print()
    
    # Correctness test
    if not args.skip_correctness:
        print("Numerical Correctness Test...")
        correctness = test_numerical_correctness(param_groups[:10], anchor_groups[:10])
        for k, v in correctness.items():
            if isinstance(v, bool):
                status = "✓" if v else "✗"
                print(f"  {k}: {status}")
            else:
                print(f"  {k}: {v:.2e}")
        print()
    
    # Summary
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"{'Method':<20} {'Time (ms)':<12} {'Memory (MB)':<12} {'Kernels':<10}")
    print("-" * 54)
    
    baseline_time = None
    for r in results:
        if r.time_ms > 0:
            if baseline_time is None:
                baseline_time = r.time_ms
            speedup = baseline_time / r.time_ms if r.time_ms > 0 else 0
            print(f"{r.method:<20} {r.time_ms:<12.2f} {r.memory_mb:<12.1f} {r.kernel_launches:<10} ({speedup:.2f}x)")
        else:
            print(f"{r.method:<20} {'N/A':<12} {'N/A':<12} {'N/A':<10}")
    
    # Save results
    if args.output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(os.path.dirname(__file__), 'benchmark_results', timestamp)
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(output_dir, f'{args.model}_zo_forward.txt')
        with open(output_file, 'w') as f:
            f.write(f"Model: {args.model}\n")
            f.write(f"Iterations: {args.n_iter}\n\n")
            for r in results:
                f.write(f"{r.method}: {r.time_ms:.2f} ms, {r.memory_mb:.1f} MB\n")
        print(f"\nResults saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Benchmark DiZO zo_forward optimizations')
    parser.add_argument('--model', type=str, default='opt-350m',
                        choices=list(MODEL_CONFIGS.keys()),
                        help='Model size preset')
    parser.add_argument('--n_iter', type=int, default=20,
                        help='Number of iterations')
    parser.add_argument('--tau', type=float, default=0.2,
                        help='Clip range')
    parser.add_argument('--zo_eps', type=float, default=0.1,
                        help='Perturbation epsilon')
    parser.add_argument('--step_size', type=float, default=2.0,
                        help='Step size for gamma update')
    parser.add_argument('--breakdown', action='store_true',
                        help='Profile individual kernels')
    parser.add_argument('--skip_pytorch', action='store_true',
                        help='Skip PyTorch baseline')
    parser.add_argument('--skip_correctness', action='store_true',
                        help='Skip correctness test')
    parser.add_argument('--output', action='store_true',
                        help='Save results to file')
    
    args = parser.parse_args()
    run_benchmarks(args)


if __name__ == "__main__":
    main()
