#!/usr/bin/env python3
"""
Integrated Benchmark: Full MeZO Training Step with All Optimizations (V3 - With Profiling)

This benchmark extends V2 with profiling support:
1. PyTorch Profiler: Chrome trace export for each benchmark method
2. nsys/ncu profiling support (via wrapper scripts)

NEW IN V3:
- PyTorch Profiler integration with Chrome trace export for each method
- nsys/ncu profiling support (via wrapper scripts)
- Per-method profiling output directories
- Detailed profiling with record_function markers
- Support for running all methods OR individual method selection
- Memory-efficient parameter flattening for large models

Usage:
    # Standard benchmark - all methods (same as v2)
    python benchmark_full_training_step_v3.py --model opt-350m
    
    # All methods with PyTorch Profiler (each gets own trace)
    python benchmark_full_training_step_v3.py --model opt-350m --pytorch_profile
    
    # Individual method with PyTorch Profiler
    python benchmark_full_training_step_v3.py --model opt-350m --method pytorch_baseline_dizo --pytorch_profile
    
    # Individual method with nsys
    python benchmark_full_training_step_v3.py --model opt-350m --method triton_perturb_dizo --nsys
    
    # Individual method with ncu
    python benchmark_full_training_step_v3.py --model opt-350m --method cuda_full --ncu

Author: DiZO Team
Date: 2026-01-03 (V3 with Profiling)
"""

import torch
import numpy as np
import os
import sys
import argparse
import subprocess
import glob
import gc
import time
from typing import Optional, Tuple, List
from datetime import datetime
from contextlib import contextmanager

# Import v2 benchmark functions - we'll wrap them with profiling
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Import from v2
import importlib.util
v2_spec = importlib.util.spec_from_file_location("benchmark_v2", 
                                                  os.path.join(SCRIPT_DIR, "benchmark_full_training_step_v2.py"))
benchmark_v2 = importlib.util.module_from_spec(v2_spec)
sys.modules['benchmark_v2'] = benchmark_v2
v2_spec.loader.exec_module(benchmark_v2)

# Import benchmark functions and constants
from benchmark_v2 import (
    MODEL_CONFIGS,
    BenchmarkResult,
    TrainingConfig,
    cleanup_gpu,
    create_param_groups,
    flatten_once,
    benchmark_pytorch_baseline,
    # NOTE: We override Triton benchmarks below with fixed versions that use
    # the autotuned kernels instead of slow runtime-seed kernels
    benchmark_cuda_perturb_only,
    benchmark_cuda_perturb_triton_zo,
    benchmark_cuda_full,
    HAS_TRITON_RUNTIME_SEED,
    HAS_TRITON_ZO_V1,
    HAS_TRITON_ZO_V2,
    HAS_CUDA_PERTURB,
    HAS_CUDA_ZO_V5,
    HAS_TRITON_PERTURB,
    dummy_forward,
    cuda_timer,
)

# Import the FAST Triton kernels (autotuned, not runtime-seed)
import triton
import triton.language as tl
try:
    sys.path.insert(0, os.path.join(SCRIPT_DIR, 'Perturb_wise'))
    from triton_fused_perturb import (
        fused_perturb_kernel_autotuned,
        fused_update_kernel_autotuned,
    )
    HAS_TRITON_AUTOTUNED = True
except ImportError:
    HAS_TRITON_AUTOTUNED = False
    print("Warning: Triton autotuned kernels not available")

# Import ZO-Forward V2 kernels
try:
    sys.path.insert(0, os.path.join(SCRIPT_DIR, 'zo_foward_wise'))
    from dizo_fused_kernels_v2 import FusedDiZOKernelsV2
    HAS_TRITON_ZO_V2_DIRECT = True
except ImportError:
    HAS_TRITON_ZO_V2_DIRECT = False

# PyTorch Profiler imports
from torch.profiler import profile, record_function, ProfilerActivity, schedule, tensorboard_trace_handler

# Set GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "5"


# =============================================================================
# Memory-Efficient Parameter Flattening
# =============================================================================

def create_flat_buffers_direct(config: dict, device):
    """
    Create flat buffers directly from config (no param_groups needed).
    
    This is the memory-efficient approach used by benchmark_triton_v2.py:
    - Creates flat tensors directly
    - Fills with random data using .normal_() (no intermediate tensors)
    - Avoids creating param_groups first (saves ~50GB for opt-13b)
    """
    num_layers = config['num_layers']
    hidden = config['hidden_size']
    ffn = config['ffn_size']
    embed_dim = config.get('embed_dim', hidden)
    has_project = config.get('has_project', False)
    vocab_size = 50272
    max_pos = 2050
    
    # Compute sizes and total elements (same logic as create_param_groups)
    sizes_list = []
    total_elements = 0
    
    # Embeddings
    sizes_list.append(vocab_size * embed_dim)
    total_elements += vocab_size * embed_dim
    sizes_list.append(max_pos * hidden)
    total_elements += max_pos * hidden
    sizes_list.append(hidden)  # layer_norm weight
    total_elements += hidden
    sizes_list.append(hidden)  # layer_norm bias
    total_elements += hidden
    
    if has_project:
        sizes_list.append(hidden * embed_dim)
        total_elements += hidden * embed_dim
        sizes_list.append(embed_dim * hidden)
        total_elements += embed_dim * hidden
    
    # Transformer layers
    for _ in range(num_layers):
        for _ in range(4):  # q, k, v, out
            sizes_list.append(hidden * hidden)
            total_elements += hidden * hidden
            sizes_list.append(hidden)
            total_elements += hidden
        sizes_list.append(hidden)  # attn_ln weight
        total_elements += hidden
        sizes_list.append(hidden)  # attn_ln bias
        total_elements += hidden
        sizes_list.append(ffn * hidden)  # fc1
        total_elements += ffn * hidden
        sizes_list.append(ffn)
        total_elements += ffn
        sizes_list.append(hidden * ffn)  # fc2
        total_elements += hidden * ffn
        sizes_list.append(hidden)
        total_elements += hidden
        sizes_list.append(hidden)  # final_ln weight
        total_elements += hidden
        sizes_list.append(hidden)  # final_ln bias
        total_elements += hidden
    
    print(f"Creating flat buffers directly: {len(sizes_list)} params, {total_elements:,} elements")
    print(f"Estimated memory: {total_elements * 4 * 2 / 1024**3:.2f} GB (param + anchor)")
    
    # Allocate flat tensors directly
    param_flat = torch.empty(total_elements, device=device, dtype=torch.float32)
    anchor_flat = torch.empty(total_elements, device=device, dtype=torch.float32)
    
    # Fill with random data directly (no intermediate tensors)
    offset = 0
    offsets_list = []
    param_views = []
    anchor_views = []
    
    for size in sizes_list:
        offsets_list.append(offset)
        # Generate random data directly into flat tensor
        param_flat[offset:offset+size].normal_()
        anchor_flat[offset:offset+size].normal_()
        
        # Create views (for CUDA perturb methods that need them)
        # Note: views are created but shape info is lost - we'll recreate views later if needed
        offset += size
    
    offsets = torch.tensor(offsets_list, device=device, dtype=torch.long)
    sizes = torch.tensor(sizes_list, device=device, dtype=torch.long)
    
    # Create views for CUDA perturb methods (they need param_views/anchor_views)
    # Reconstruct shapes from sizes (simplified - assumes 1D for now)
    param_views = []
    anchor_views = []
    offset = 0
    for size in sizes_list:
        param_views.append(param_flat[offset:offset+size])
        anchor_views.append(anchor_flat[offset:offset+size])
        offset += size
    
    return param_flat, anchor_flat, offsets, sizes, param_views, anchor_views


def flatten_once_memory_efficient(param_groups: List[torch.Tensor], anchor_groups: List[torch.Tensor], device):
    """
    Memory-efficient flattening: allocates flat tensors first, then fills incrementally.
    
    This avoids the memory spike from torch.cat([p.flatten() for p in param_groups])
    which creates all flattened tensors before concatenating.
    
    Based on create_flattened_params from benchmark_triton_v2.py
    """
    # First pass: compute sizes and total elements
    sizes_list = []
    total_elements = 0
    for p in param_groups:
        numel = p.numel()
        sizes_list.append(numel)
        total_elements += numel
    
    # Allocate flat tensors directly (no intermediate tensors)
    param_flat = torch.empty(total_elements, device=device, dtype=torch.float32)
    anchor_flat = torch.empty(total_elements, device=device, dtype=torch.float32)
    
    # Fill incrementally to avoid peak memory from intermediate tensors
    offset = 0
    offsets_list = []
    param_views = []
    anchor_views = []
    
    for p, a in zip(param_groups, anchor_groups):
        numel = p.numel()
        offsets_list.append(offset)
        
        # Copy data directly into flat tensor (memory-efficient)
        # Use view + copy to avoid creating intermediate flattened tensor
        param_flat[offset:offset+numel].copy_(p.view(-1))
        anchor_flat[offset:offset+numel].copy_(a.view(-1))
        
        # Create views into the flat buffer
        param_views.append(param_flat[offset:offset+numel].view(p.shape))
        anchor_views.append(anchor_flat[offset:offset+numel].view(a.shape))
        
        offset += numel
    
    offsets = torch.tensor(offsets_list, device=device, dtype=torch.long)
    sizes = torch.tensor(sizes_list, device=device, dtype=torch.long)
    
    return param_flat, anchor_flat, offsets, sizes, param_views, anchor_views


# =============================================================================
# FIXED Triton Benchmark Functions (using autotuned kernels, not runtime-seed)
# =============================================================================
# The v2 benchmark uses runtime-seed kernels which are 2-3x slower than the
# autotuned kernels. For fair benchmarking, we use constant seed (valid because
# kernel performance doesn't depend on seed value, and real MeZO uses same seed
# for all calls within one step).

def benchmark_triton_perturb_only(
    param_flat: torch.Tensor,
    anchor_flat: torch.Tensor,
    offsets: torch.Tensor,
    sizes: torch.Tensor,
    param_views: List[torch.Tensor],
    anchor_views: List[torch.Tensor],
    cfg: TrainingConfig,
    n_iter: int,
    include_dizo: bool = True,
) -> BenchmarkResult:
    """
    Triton fused perturb kernels + PyTorch ZO-forward.
    Uses AUTOTUNED kernels with constant seed for accurate performance measurement.
    
    For large models (>2B elements), falls back to v2's runtime-seed kernels
    which have int64 support for offsets.
    """
    n_elements = param_flat.numel()
    INT32_MAX = 2_147_483_647
    
    # For large models (>2B elements), fall back to v2 implementation
    if n_elements > INT32_MAX:
        print(f"  [Large model: {n_elements:,} elements > int32 max, using v2 runtime-seed kernels]")
        from benchmark_v2 import benchmark_triton_perturb_only as v2_triton_perturb
        return v2_triton_perturb(param_flat, anchor_flat, offsets, sizes, 
                                  param_views, anchor_views, cfg, n_iter, include_dizo)
    
    if not HAS_TRITON_AUTOTUNED:
        return BenchmarkResult(method="Triton Perturb (N/A)", total_time_ms=-1, memory_mb=0)
    
    device = param_flat.device
    num_params = len(param_views)
    
    # Initialize gamma
    gammas = [torch.tensor([0.1], device=device) for _ in range(num_params)]
    
    # Pre-compute grid
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Constant seed for benchmarking (valid - same seed used within one MeZO step)
    seed = 42
    
    torch.cuda.reset_peak_memory_stats()
    
    perturb_time = 0.0
    zo_time = 0.0
    update_time = 0.0
    
    # Warmup Triton kernels (important for autotuning)
    for _ in range(50):
        fused_perturb_kernel_autotuned[grid](param_flat, seed, cfg.eps, n_elements)
        fused_update_kernel_autotuned[grid](param_flat, seed, 0.001, cfg.lr, n_elements)
    torch.cuda.synchronize()
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    for _ in range(n_iter):
        # === Perturb +eps (FUSED) ===
        fused_perturb_kernel_autotuned[grid](param_flat, seed, cfg.eps, n_elements)
        
        # === Forward 1 ===
        loss1 = dummy_forward()
        
        # === DiZO constraints (PyTorch) ===
        if include_dizo:
            norms = [torch.norm(p - a) for p, a in zip(param_views, anchor_views)]
            for i, (p, a, g, n) in enumerate(zip(param_views, anchor_views, gammas, norms)):
                alpha = g / (n + 1e-8)
                p.data.copy_(a + (p - a) * alpha)
        
        # === Perturb -2eps (FUSED) ===
        fused_perturb_kernel_autotuned[grid](param_flat, seed, -2*cfg.eps, n_elements)
        
        # === Forward 2 ===
        loss2 = dummy_forward()
        
        # === DiZO reverse (PyTorch) ===
        if include_dizo:
            for i, (p, a, g, n) in enumerate(zip(param_views, anchor_views, gammas, norms)):
                alpha = g / (n + 1e-8)
                p.data.copy_(a + (p - a) / alpha)
        
        projected_grad = (loss1 - loss2) / (2 * cfg.eps)
        
        # === Perturb +eps reset (FUSED) ===
        fused_perturb_kernel_autotuned[grid](param_flat, seed, cfg.eps, n_elements)
        
        # === Update (FUSED) ===
        fused_update_kernel_autotuned[grid](param_flat, seed, projected_grad, cfg.lr, n_elements)
        
        # === DiZO gamma update (PyTorch) ===
        if include_dizo:
            zs = [torch.randn(1, device=device) for _ in range(num_params)]
            for i, (g, n, z) in enumerate(zip(gammas, norms, zs)):
                gammas[i] = torch.clamp(g - cfg.step_size * n * projected_grad * z,
                                       (1-cfg.tau)*n, (1+cfg.tau)*n)
    
    torch.cuda.synchronize()
    total = (time.perf_counter() - start) * 1000 / n_iter
    mem = torch.cuda.max_memory_allocated() / 1024**2
    
    return BenchmarkResult(
        method="Triton Perturb" + (" + PyTorch DiZO" if include_dizo else ""),
        total_time_ms=total,
        memory_mb=mem,
        kernel_launches=4 if not include_dizo else 4 + num_params * 9,
    )


def benchmark_triton_v1_full(
    param_flat: torch.Tensor,
    anchor_flat: torch.Tensor,
    offsets: torch.Tensor,
    sizes: torch.Tensor,
    cfg: TrainingConfig,
    n_iter: int,
) -> BenchmarkResult:
    """Triton Perturb + Triton ZO V1 (not optimized - kept for comparison)."""
    # Import from v2 for this legacy benchmark
    from benchmark_v2 import benchmark_triton_v1_full as v2_triton_v1
    return v2_triton_v1(param_flat, anchor_flat, offsets, sizes, cfg, n_iter)


def benchmark_triton_full(
    param_flat: torch.Tensor,
    anchor_flat: torch.Tensor,
    offsets: torch.Tensor,
    sizes: torch.Tensor,
    cfg: TrainingConfig,
    n_iter: int,
) -> BenchmarkResult:
    """
    Fully optimized: Triton Perturb (autotuned) + Triton ZO-Forward V2.
    Uses constant seed for accurate performance measurement.
    
    For large models (>2B elements), falls back to v2's runtime-seed kernels
    which have int64 support for offsets.
    """
    n_elements = param_flat.numel()
    INT32_MAX = 2_147_483_647
    
    # For large models (>2B elements), fall back to v2 implementation
    # because the autotuned kernels use int32 offsets which overflow
    if n_elements > INT32_MAX:
        print(f"  [Large model: {n_elements:,} elements > int32 max, using v2 runtime-seed kernels]")
        from benchmark_v2 import benchmark_triton_full as v2_triton_full
        return v2_triton_full(param_flat, anchor_flat, offsets, sizes, cfg, n_iter)
    
    if not HAS_TRITON_AUTOTUNED or not HAS_TRITON_ZO_V2_DIRECT:
        missing = []
        if not HAS_TRITON_AUTOTUNED:
            missing.append("Triton Perturb Autotuned")
        if not HAS_TRITON_ZO_V2_DIRECT:
            missing.append("Triton ZO V2")
        return BenchmarkResult(
            method=f"Triton Full (N/A: {', '.join(missing)})",
            total_time_ms=-1, memory_mb=0
        )
    
    device = param_flat.device
    num_params = offsets.shape[0]
    
    # Initialize ZO-Forward V2 kernels
    zo_kernels = FusedDiZOKernelsV2(num_params, n_elements, device, offsets, sizes)
    
    # Initialize constraints
    constraints = torch.rand(num_params, device=device) * 0.1
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Constant seed for benchmarking
    seed = 42
    
    cleanup_gpu()
    torch.cuda.reset_peak_memory_stats()
    
    # Warmup (important for autotuning)
    for _ in range(50):
        fused_perturb_kernel_autotuned[grid](param_flat, seed, cfg.eps, n_elements)
        fused_update_kernel_autotuned[grid](param_flat, seed, 0.001, cfg.lr, n_elements)
        norms_warm = zo_kernels.compute_norms(param_flat, anchor_flat, offsets, sizes)
        alphas_warm = zo_kernels.apply_constraints(param_flat, anchor_flat, offsets, sizes, 
                                                    constraints, norms_warm)
        zo_kernels.reverse_constraints(param_flat, anchor_flat, offsets, sizes, alphas_warm)
    torch.cuda.synchronize()
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    for _ in range(n_iter):
        # === Perturb +eps ===
        fused_perturb_kernel_autotuned[grid](param_flat, seed, cfg.eps, n_elements)
        
        loss1 = dummy_forward()
        
        # === ZO-Forward: Compute norms + Apply constraints (FUSED V2) ===
        norms = zo_kernels.compute_norms(param_flat, anchor_flat, offsets, sizes)
        alphas = zo_kernels.apply_constraints(param_flat, anchor_flat, offsets, sizes, 
                                               constraints, norms)
        
        # === Perturb -2eps ===
        fused_perturb_kernel_autotuned[grid](param_flat, seed, -2*cfg.eps, n_elements)
        
        loss2 = dummy_forward()
        
        # === ZO-Forward: Reverse constraints (FUSED V2) ===
        zo_kernels.reverse_constraints(param_flat, anchor_flat, offsets, sizes, alphas)
        
        projected_grad = (loss1 - loss2) / (2 * cfg.eps)
        
        # === Perturb reset ===
        fused_perturb_kernel_autotuned[grid](param_flat, seed, cfg.eps, n_elements)
        
        # === Update ===
        fused_update_kernel_autotuned[grid](param_flat, seed, projected_grad, cfg.lr, n_elements)
        
        # === Gamma update (V2) ===
        zs = zo_kernels.perturb_gamma(constraints, norms, 1.0, cfg.tau, cfg.zo_eps, generate_new=True)
        zo_kernels.update_gamma(constraints, norms, projected_grad, cfg.step_size, cfg.tau)
    
    torch.cuda.synchronize()
    total = (time.perf_counter() - start) * 1000 / n_iter
    mem = torch.cuda.max_memory_allocated() / 1024**2
    
    return BenchmarkResult(
        method="Triton Perturb + Triton ZO V2",
        total_time_ms=total,
        memory_mb=mem,
        kernel_launches=10,
    )


# =============================================================================
# Method Definitions
# =============================================================================

# Define all available benchmark methods
BENCHMARK_METHODS = {
    'pytorch_baseline_mezo': {
        'name': 'PyTorch Baseline (MeZO only)',
        'func': benchmark_pytorch_baseline,
        'args': {'include_dizo': False},
        'needs_flat': False,
    },
    'pytorch_baseline_dizo': {
        'name': 'PyTorch Baseline + DiZO',
        'func': benchmark_pytorch_baseline,
        'args': {'include_dizo': True},
        'needs_flat': False,
    },
    'triton_perturb_mezo': {
        'name': 'Triton Perturb (MeZO only)',
        'func': benchmark_triton_perturb_only,
        'args': {'include_dizo': False},
        'needs_flat': True,
        'requires': ['HAS_TRITON_AUTOTUNED'],
    },
    'triton_perturb_dizo': {
        'name': 'Triton Perturb + PyTorch DiZO',
        'func': benchmark_triton_perturb_only,
        'args': {'include_dizo': True},
        'needs_flat': True,
        'requires': ['HAS_TRITON_AUTOTUNED'],
    },
    'cuda_perturb_mezo': {
        'name': 'CUDA Perturb (MeZO only)',
        'func': benchmark_cuda_perturb_only,
        'args': {'include_dizo': False},
        'needs_flat': True,
        'requires': ['HAS_CUDA_PERTURB'],
    },
    'cuda_perturb_dizo': {
        'name': 'CUDA Perturb + PyTorch DiZO',
        'func': benchmark_cuda_perturb_only,
        'args': {'include_dizo': True},
        'needs_flat': True,
        'requires': ['HAS_CUDA_PERTURB'],
    },
    'triton_v1_full': {
        'name': 'Triton Perturb + Triton ZO V1',
        'func': benchmark_triton_v1_full,
        'args': {},
        'needs_flat': True,
        'requires': ['HAS_TRITON_RUNTIME_SEED', 'HAS_TRITON_ZO_V1'],
    },
    'triton_v2_full': {
        'name': 'Triton Perturb + Triton ZO V2',
        'func': benchmark_triton_full,
        'args': {},
        'needs_flat': True,
        'requires': ['HAS_TRITON_AUTOTUNED', 'HAS_TRITON_ZO_V2_DIRECT'],
    },
    'cuda_triton_zo': {
        'name': 'CUDA Perturb + Triton ZO V2',
        'func': benchmark_cuda_perturb_triton_zo,
        'args': {},
        'needs_flat': True,
        'requires': ['HAS_CUDA_PERTURB', 'HAS_TRITON_ZO_V2'],
    },
    'cuda_full': {
        'name': 'CUDA Perturb + CUDA ZO V5',
        'func': benchmark_cuda_full,
        'args': {},
        'needs_flat': True,
        'requires': ['HAS_CUDA_PERTURB', 'HAS_CUDA_ZO_V5'],
    },
}


# =============================================================================
# PyTorch Profiler Integration
# =============================================================================

def benchmark_with_pytorch_profiler(
    benchmark_func,
    method_name: str,
    output_dir: str,
    model_name: str,
    n_iter: int,
    *args,
    **kwargs
) -> Tuple[BenchmarkResult, Optional[str]]:
    """
    Run benchmark with PyTorch Profiler and export Chrome trace.
    
    Uses a simple profiler context (no schedule) since benchmark functions
    don't call prof.step(). This still captures all operations and exports traces.
    
    Returns:
        (BenchmarkResult, trace_path)
    """
    # Create method-specific output directory
    method_safe = method_name.replace(" ", "_").replace("+", "_plus_").replace("/", "_").replace("(", "").replace(")", "").replace("___", "_").replace("__", "_")
    profiler_dir = os.path.join(output_dir, "profiler_logs", model_name, method_safe)
    os.makedirs(profiler_dir, exist_ok=True)
    
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    
    result = None
    trace_path = None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create descriptive trace filename: model_method_iter_timestamp.pt.trace.json
    trace_filename = f"{model_name}_{method_safe}_iter{n_iter}_{timestamp}.pt.trace.json"
    trace_path = os.path.join(profiler_dir, trace_filename)
    
    # Use simple profiler context (no schedule) - captures everything
    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
        with_modules=True,
    ) as prof:
        # Run benchmark function - it will be profiled automatically
        with record_function(f"benchmark_{method_safe}"):
            result = benchmark_func(*args, **kwargs)
    
    # Export Chrome trace AFTER context manager exits with descriptive name
    try:
        prof.export_chrome_trace(trace_path)
        print(f"  Chrome trace exported: {os.path.basename(trace_path)}")
        print(f"    Full path: {trace_path}")
    except Exception as e:
        print(f"  Warning: Failed to export Chrome trace: {e}")
        trace_path = None
    
    if result and trace_path:
        if hasattr(result, 'trace_path'):
            result.trace_path = trace_path
    
    return result, trace_path


# =============================================================================
# nsys/ncu Profiling Support
# =============================================================================

def create_profiler_wrapper_script(method_key: str, model: str, n_iter: int, output_dir: str, cfg_dict: dict):
    """Create a standalone script that runs a specific benchmark method."""
    import tempfile
    import json
    
    method_info = BENCHMARK_METHODS[method_key]
    method_name = method_info['name']
    
    # Create the wrapper script
    script_content = f'''#!/usr/bin/env python3
"""Profiler wrapper for {method_name}"""
import sys
import os
sys.path.insert(0, r"{SCRIPT_DIR}")

# Import v2 benchmark
import importlib.util
v2_spec = importlib.util.spec_from_file_location("benchmark_v2", 
                                                  r"{os.path.join(SCRIPT_DIR, 'benchmark_full_training_step_v2.py')}")
benchmark_v2 = importlib.util.module_from_spec(v2_spec)
sys.modules['benchmark_v2'] = benchmark_v2
v2_spec.loader.exec_module(benchmark_v2)

from benchmark_v2 import *
import torch
import numpy as np

if __name__ == "__main__":
    # Configuration
    model = "{model}"
    n_iter = {n_iter}
    method_key = "{method_key}"
    
    # Setup
    config = MODEL_CONFIGS[model]
    device = torch.device('cuda')
    param_groups, anchor_groups = create_param_groups(config, device)
    cfg = TrainingConfig(
        eps={cfg_dict.get('eps', 1e-3)},
        lr={cfg_dict.get('lr', 1e-5)},
        tau={cfg_dict.get('tau', 0.2)},
        zo_eps={cfg_dict.get('zo_eps', 0.1)},
        step_size={cfg_dict.get('step_size', 2.0)}
    )
    
    # Prepare parameters based on method
    method_info = {{
        'pytorch_baseline_mezo': {{'needs_flat': False, 'include_dizo': False}},
        'pytorch_baseline_dizo': {{'needs_flat': False, 'include_dizo': True}},
        'triton_perturb_mezo': {{'needs_flat': True, 'include_dizo': False}},
        'triton_perturb_dizo': {{'needs_flat': True, 'include_dizo': True}},
        'cuda_perturb_mezo': {{'needs_flat': True, 'include_dizo': False}},
        'cuda_perturb_dizo': {{'needs_flat': True, 'include_dizo': True}},
    }}
    
    info = method_info.get(method_key, {{'needs_flat': True, 'include_dizo': True}})
    
    if info['needs_flat']:
        param_flat, anchor_flat, offsets, sizes, param_views, anchor_views = flatten_once(
            param_groups, anchor_groups, device)
        
        if 'triton_perturb' in method_key:
            result = benchmark_triton_perturb_only(
                param_flat, anchor_flat, offsets, sizes, param_views, anchor_views,
                cfg, n_iter, include_dizo=info['include_dizo']
            )
        elif 'cuda_perturb' in method_key and 'zo' not in method_key:
            result = benchmark_cuda_perturb_only(
                param_flat, anchor_flat, param_views, anchor_views,
                cfg, n_iter, include_dizo=info['include_dizo']
            )
        elif method_key == 'triton_v1_full':
            result = benchmark_triton_v1_full(param_flat, anchor_flat, offsets, sizes, cfg, n_iter)
        elif method_key == 'triton_v2_full':
            result = benchmark_triton_full(param_flat, anchor_flat, offsets, sizes, cfg, n_iter)
        elif method_key == 'cuda_triton_zo':
            result = benchmark_cuda_perturb_triton_zo(param_flat, anchor_flat, offsets, sizes, cfg, n_iter)
        elif method_key == 'cuda_full':
            result = benchmark_cuda_full(param_flat, anchor_flat, offsets, sizes, cfg, n_iter)
        else:
            print(f"Unknown method: {{method_key}}")
            sys.exit(1)
    else:
        # PyTorch baseline
        result = benchmark_pytorch_baseline(
            param_groups, anchor_groups, cfg, n_iter,
            include_dizo=info['include_dizo']
        )
    
    print(f"Benchmark complete: {{result.total_time_ms:.2f}} ms")
'''
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        script_path = f.name
        f.write(script_content)
    
    os.chmod(script_path, 0o755)
    return script_path


def run_nsys_profile(method_key: str, model: str, n_iter: int, output_dir: str, args_dict: dict, cfg_dict: dict):
    """Run a single benchmark under nsys profiler."""
    method_info = BENCHMARK_METHODS[method_key]
    method_name = method_info['name']
    method_safe = method_name.replace(" ", "_").replace("+", "_plus_").replace("/", "_").replace("(", "").replace(")", "").replace("___", "_").replace("__", "_")
    
    nsys_output = os.path.join(output_dir, "profiler_logs", "nsys", model, 
                               f"{model}_{method_safe}_iter{n_iter}.nsys-rep")
    os.makedirs(os.path.dirname(nsys_output), exist_ok=True)
    
    # Create wrapper script
    script_path = create_profiler_wrapper_script(method_key, model, n_iter, output_dir, cfg_dict)
    
    cmd = [
        'nsys', 'profile',
        '-t', 'cuda,nvtx,osrt,cudnn,cublas',
        '--cuda-memory-usage=true',
        '--stats=true',
        '--force-overwrite=true',
        '-o', nsys_output,
        'python', script_path
    ]
    
    print(f"Running nsys profile for {method_name}...")
    print(f"  Output: {nsys_output}")
    result = subprocess.run(cmd, capture_output=False, text=True)  # Don't capture to see output
    
    # Cleanup
    try:
        os.unlink(script_path)
    except:
        pass
    
    # Check if file was created
    if os.path.exists(nsys_output) and os.path.getsize(nsys_output) > 0:
        print(f"  ✓ nsys profile saved: {nsys_output}")
        print(f"    View with: nsight-sys {nsys_output}")
        return nsys_output
    else:
        print(f"  ✗ Error: nsys profile file not created or is empty")
        if result.returncode != 0:
            print(f"    Exit code: {result.returncode}")
        return None


def run_ncu_profile(method_key: str, model: str, n_iter: int, output_dir: str, args_dict: dict, cfg_dict: dict):
    """Run a single benchmark under ncu profiler."""
    method_info = BENCHMARK_METHODS[method_key]
    method_name = method_info['name']
    method_safe = method_name.replace(" ", "_").replace("+", "_plus_").replace("/", "_").replace("(", "").replace(")", "").replace("___", "_").replace("__", "_")
    
    ncu_output = os.path.join(output_dir, "profiler_logs", "ncu", model,
                              f"{model}_{method_safe}_iter{n_iter}.ncu-rep")
    os.makedirs(os.path.dirname(ncu_output), exist_ok=True)
    
    # Create wrapper script
    script_path = create_profiler_wrapper_script(method_key, model, n_iter, output_dir, cfg_dict)
    
    # ncu command - use a lighter preset for faster profiling
    cmd = [
        'ncu',
        '--set', 'default',  # Use 'default' instead of 'full' for faster profiling
        '--export', ncu_output,
        '--force-overwrite',
        '--target-processes', 'all',  # Profile all processes
        'python', script_path
    ]
    
    print(f"Running ncu profile for {method_name}...")
    print(f"  Output: {ncu_output}")
    print(f"  Note: ncu profiling may take longer and may require GPU access")
    
    # Don't capture output so user can see ncu progress
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    # Cleanup
    try:
        os.unlink(script_path)
    except:
        pass
    
    # Check if file was created (ncu creates .ncu-rep file)
    ncu_file = ncu_output
    if not ncu_file.endswith('.ncu-rep'):
        ncu_file = ncu_output + '.ncu-rep'
    
    if os.path.exists(ncu_file) and os.path.getsize(ncu_file) > 0:
        print(f"  ✓ ncu profile saved: {ncu_file}")
        print(f"    View with: nsight-compute {ncu_file}")
        return ncu_file
    else:
        print(f"  ✗ Error: ncu profile file not created or is empty")
        if result.returncode != 0:
            print(f"    Exit code: {result.returncode}")
        print(f"    Note: ncu may require sudo or proper GPU permissions")
        return None


# =============================================================================
# Benchmark Execution Functions
# =============================================================================

def run_single_benchmark(method_key: str, args, param_groups, anchor_groups, param_flat, anchor_flat,
                         offsets, sizes, param_views, anchor_views, cfg, n_iter, output_dir):
    """Run a single benchmark method with optional profiling."""
    method_info = BENCHMARK_METHODS[method_key]
    method_name = method_info['name']
    benchmark_func = method_info['func']
    method_args = method_info['args']
    needs_flat = method_info['needs_flat']
    
    # Check requirements
    if 'requires' in method_info:
        for req in method_info['requires']:
            if not globals().get(req, False):
                print(f"Skipping {method_name}: {req} not available")
                return None
    
    print(f"Benchmarking: {method_name}...")
    
    # Prepare arguments based on method type
    if needs_flat:
        if method_key in ['cuda_perturb_mezo', 'cuda_perturb_dizo']:
            # CUDA perturb methods need param_flat, anchor_flat, param_views, anchor_views
            call_args = (param_flat, anchor_flat, param_views, anchor_views, cfg, n_iter)
        else:
            # Triton methods need param_flat, anchor_flat, offsets, sizes, ...
            if method_key in ['triton_perturb_mezo', 'triton_perturb_dizo']:
                call_args = (param_flat, anchor_flat, offsets, sizes, param_views, anchor_views, cfg, n_iter)
            else:
                call_args = (param_flat, anchor_flat, offsets, sizes, cfg, n_iter)
    else:
        # PyTorch baseline methods
        # For single benchmark runs, don't clone (saves memory for large models)
        # The benchmark function may modify params in-place, but that's OK for single runs
        call_args = (param_groups, anchor_groups, cfg, n_iter)
    
    # Merge method_args
    all_kwargs = {**method_args}
    
    # Run with profiling if requested
    if args.pytorch_profile:
        r, trace_path = benchmark_with_pytorch_profiler(
            benchmark_func,
            method_name,
            output_dir,
            args.model,
            n_iter,
            *call_args,
            **all_kwargs
        )
    elif args.nsys:
        cfg_dict = {'eps': cfg.eps, 'lr': cfg.lr, 'tau': cfg.tau, 'zo_eps': cfg.zo_eps, 'step_size': cfg.step_size}
        nsys_file = run_nsys_profile(method_key, args.model, n_iter, output_dir, all_kwargs, cfg_dict)
        # Also run benchmark to get timing
        r = benchmark_func(*call_args, **all_kwargs)
    elif args.ncu:
        cfg_dict = {'eps': cfg.eps, 'lr': cfg.lr, 'tau': cfg.tau, 'zo_eps': cfg.zo_eps, 'step_size': cfg.step_size}
        ncu_file = run_ncu_profile(method_key, args.model, n_iter, output_dir, all_kwargs, cfg_dict)
        # Also run benchmark to get timing
        r = benchmark_func(*call_args, **all_kwargs)
    else:
        r = benchmark_func(*call_args, **all_kwargs)
    
    if r and r.total_time_ms > 0:
        print(f"  → {r.total_time_ms:.2f} ms")
    elif r:
        print(f"  → {r.method}")
    
    cleanup_gpu()
    return r


def run_all_benchmarks(args):
    """Run all available benchmarks with optional profiling."""
    print("=" * 90)
    print("INTEGRATED BENCHMARK V3: Full MeZO/DiZO Training Step (WITH PROFILING)")
    print("=" * 90)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}, PyTorch: {torch.__version__}")
    print()
    
    output_dir = os.path.join(SCRIPT_DIR, 'benchmark_results')
    os.makedirs(output_dir, exist_ok=True)
    
    if args.pytorch_profile:
        print("PyTorch Profiler: ENABLED (each method will generate its own trace file)")
        print(f"  Profile iterations: {args.profile_iter}")
        print(f"  Output directory: {output_dir}/profiler_logs/")
        print()
    elif args.nsys:
        print("nsys Profiler: ENABLED")
        print(f"  Output directory: {output_dir}/profiler_logs/nsys/")
        print()
    elif args.ncu:
        print("ncu Profiler: ENABLED")
        print(f"  Output directory: {output_dir}/profiler_logs/ncu/")
        print()
    
    config = MODEL_CONFIGS[args.model]
    print(f"Model: {args.model}")
    print(f"  Layers: {config['num_layers']}, Hidden: {config['hidden_size']}")
    print(f"  Expected params: {config['total_params']:,}")
    print()
    
    device = torch.device('cuda')
    
    # Create parameter groups once (these are reused for PyTorch baselines)
    print("Creating parameter groups...")
    param_groups, anchor_groups = create_param_groups(config, device)
    total_elements = sum(p.numel() for p in param_groups)
    print(f"  Created {len(param_groups)} parameter groups, {total_elements:,} total elements")
    print(f"  Estimated memory: {total_elements * 4 * 2 / 1024**3:.2f} GB (params + anchors)")
    print()
    
    cfg = TrainingConfig(eps=args.eps, lr=args.lr, tau=args.tau, 
                        zo_eps=args.zo_eps, step_size=args.step_size)
    
    results = []
    n_iter = args.profile_iter if (args.pytorch_profile or args.nsys or args.ncu) else args.n_iter
    
    # Warmup
    print(f"Warming up kernels ({args.warmup} iterations)...")
    for _ in range(args.warmup):
        x = torch.randn(256, 256, device='cuda')
        _ = x @ x.T
    cleanup_gpu()
    print()
    
    # Run all benchmarks (similar to v2's run_benchmarks)
    method_order = [
        'pytorch_baseline_mezo',
        'pytorch_baseline_dizo',
        'triton_perturb_mezo',
        'triton_perturb_dizo',
        'cuda_perturb_mezo',
        'cuda_perturb_dizo',
        'triton_v1_full',
        'triton_v2_full',
        'cuda_triton_zo',
        'cuda_full',
    ]
    
    for method_key in method_order:
        if method_key not in BENCHMARK_METHODS:
            continue
        
        # Skip baseline MeZO if requested
        if args.skip_baseline and method_key == 'pytorch_baseline_mezo':
            continue
        
        # Create fresh flat buffers for each benchmark (memory-efficient)
        # Only create when needed, and clean up immediately after
        param_flat = anchor_flat = offsets = sizes = param_views = anchor_views = None
        
        if BENCHMARK_METHODS[method_key]['needs_flat']:
            # Aggressive cleanup before creating flat buffers (free up fragmented memory)
            cleanup_gpu()
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Use memory-efficient flattening
            try:
                pf, af, off, sz, pv, av = flatten_once_memory_efficient(param_groups, anchor_groups, device)
                if method_key in ['cuda_perturb_mezo', 'cuda_perturb_dizo']:
                    param_flat, anchor_flat, param_views, anchor_views = pf, af, pv, av
                    offsets, sizes = None, None  # Not needed for CUDA perturb-only methods
                else:
                    param_flat, anchor_flat, offsets, sizes, param_views, anchor_views = pf, af, off, sz, pv, av
            except torch.cuda.OutOfMemoryError as e:
                print(f"  ✗ Out of memory: {e}")
                print(f"    Skipping {method_key} due to OOM")
                cleanup_gpu()
                gc.collect()
                torch.cuda.empty_cache()
                continue
        
        try:
            r = run_single_benchmark(
                method_key, args, param_groups, anchor_groups,
                param_flat, anchor_flat, offsets, sizes, param_views, anchor_views,
                cfg, n_iter, output_dir
            )
            if r:
                results.append(r)
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Aggressive cleanup after each benchmark
            del param_flat, anchor_flat, offsets, sizes, param_views, anchor_views
            cleanup_gpu()
            gc.collect()
            torch.cuda.empty_cache()
    
    # Print summary
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    for r in results:
        if r and r.total_time_ms > 0:
            print(f"{r.method:<50} {r.total_time_ms:.2f} ms")
            if hasattr(r, 'trace_path') and r.trace_path:
                print(f"  Trace: {os.path.basename(r.trace_path)}")
    
    return results


def run_benchmarks(args):
    """Main entry point - run all or single benchmark based on args."""
    if args.method:
        # Run single benchmark
        if args.method not in BENCHMARK_METHODS:
            print(f"Error: Unknown method '{args.method}'")
            print(f"Available methods: {', '.join(BENCHMARK_METHODS.keys())}")
            return []
        
        # For single method, we still need setup
        config = MODEL_CONFIGS[args.model]
        device = torch.device('cuda')
        
        # For flat-buffer benchmarks, create buffers directly (no param_groups needed)
        # This saves ~50GB for opt-13b
        if BENCHMARK_METHODS[args.method]['needs_flat']:
            print("Creating flat buffers directly (memory-efficient for large models)...")
            pf, af, off, sz, pv, av = create_flat_buffers_direct(config, device)
            if args.method in ['cuda_perturb_mezo', 'cuda_perturb_dizo']:
                param_flat, anchor_flat, param_views, anchor_views = pf, af, pv, av
                offsets, sizes = None, None
            else:
                param_flat, anchor_flat, offsets, sizes, param_views, anchor_views = pf, af, off, sz, pv, av
            param_groups = anchor_groups = None  # Not needed for flat-buffer benchmarks
        else:
            # PyTorch baseline needs param_groups
            param_groups, anchor_groups = create_param_groups(config, device)
            param_flat = anchor_flat = offsets = sizes = param_views = anchor_views = None
        
        cfg = TrainingConfig(eps=args.eps, lr=args.lr, tau=args.tau, 
                            zo_eps=args.zo_eps, step_size=args.step_size)
        n_iter = args.profile_iter if (args.pytorch_profile or args.nsys or args.ncu) else args.n_iter
        output_dir = os.path.join(SCRIPT_DIR, 'benchmark_results')
        
        r = run_single_benchmark(
            args.method, args, param_groups, anchor_groups,
            param_flat, anchor_flat, offsets, sizes, param_views, anchor_views,
            cfg, n_iter, output_dir
        )
        return [r] if r else []
    else:
        # Run all benchmarks
        return run_all_benchmarks(args)


def main():
    parser = argparse.ArgumentParser(description='Full MeZO/DiZO Training Step Benchmark V3 (with Profiling)')
    parser.add_argument('--model', type=str, default='opt-350m', choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument('--method', type=str, default=None, 
                       choices=list(BENCHMARK_METHODS.keys()) + [None],
                       help='Run a single benchmark method (if not specified, runs all)')
    parser.add_argument('--n_iter', type=int, default=20, help='Number of benchmark iterations (without profiling)')
    parser.add_argument('--profile_iter', type=int, default=5, help='Number of iterations when profiling')
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--eps', type=float, default=1e-3)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--tau', type=float, default=0.2)
    parser.add_argument('--zo_eps', type=float, default=0.1)
    parser.add_argument('--step_size', type=float, default=2.0)
    parser.add_argument('--skip_baseline', action='store_true', help='Skip PyTorch Baseline (MeZO only)')
    parser.add_argument('--output', action='store_true')
    parser.add_argument('--pytorch_profile', action='store_true', 
                       help='Enable PyTorch Profiler with Chrome trace export')
    parser.add_argument('--nsys', action='store_true', help='Enable nsys profiling (for single method)')
    parser.add_argument('--ncu', action='store_true', help='Enable ncu profiling (for single method)')
    
    args = parser.parse_args()
    
    # Validation
    if args.nsys and args.ncu:
        print("Error: Cannot use both --nsys and --ncu at the same time")
        return
    
    if (args.nsys or args.ncu) and not args.method:
        print("Error: --nsys and --ncu can only be used with --method (single benchmark)")
        return
    
    run_benchmarks(args)


if __name__ == "__main__":
    main()
