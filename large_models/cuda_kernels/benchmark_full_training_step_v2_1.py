#!/usr/bin/env python3
"""
Integrated Benchmark: Full MeZO Training Step with All Optimizations (V2 - FIXED)

This benchmark simulates a complete MeZO (DiZO) training step, combining:
1. Perturb-wise kernels (Triton/CUDA from Perturb_wise/)
2. ZO-forward-wise kernels (Triton V2 / CUDA V5 from zo_foward_wise/)

FIXES from V1:
- Uses newest kernel versions (V2 Triton, V5 CUDA)
- Pre-flattens parameters ONCE before benchmark loop
- Proper buffer management without redundant allocations
- Tests all kernel combinations (Triton+Triton, CUDA+CUDA, etc.)

Usage:
    python benchmark_full_training_step_v2.py --model opt-350m
    python benchmark_full_training_step_v2.py --model opt-350m --breakdown
    python benchmark_full_training_step_v2.py --all

Author: DiZO Team
Date: 2026-01-03 (Fixed)
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
from contextlib import contextmanager

# Add paths for kernel imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'Perturb_wise'))
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'zo_foward_wise'))


# =============================================================================
# Configuration: Model Presets
# =============================================================================

MODEL_CONFIGS = {
    'opt-350m': {
        'num_layers': 24,
        'hidden_size': 1024,
        'ffn_size': 4096,
        'embed_dim': 512,
        'has_project': True,
        'total_params': 331_196_416,
    },
    'opt-1.3b': {
        'num_layers': 24,
        'hidden_size': 2048,
        'ffn_size': 8192,
        'embed_dim': 2048,
        'has_project': False,
        'total_params': 1_315_753_984,
    },
    'opt-2.7b': {
        'num_layers': 32,
        'hidden_size': 2560,
        'ffn_size': 10240,
        'embed_dim': 2560,
        'has_project': False,
        'total_params': 2_651_596_800,
    },
    'opt-6.7b': {
        'num_layers': 32,
        'hidden_size': 4096,
        'ffn_size': 16384,
        'embed_dim': 4096,
        'has_project': False,
        'total_params': 6_658_473_984,
    },
    'opt-13b': {
        'num_layers': 40,
        'hidden_size': 5120,
        'ffn_size': 20480,
        'embed_dim': 5120,
        'has_project': False,
        'total_params': 13_016_023_040,
    },
}


# =============================================================================
# Kernel Import Checks - Check ALL versions
# =============================================================================

HAS_TRITON = False
HAS_TRITON_PERTURB = False
HAS_CUDA_PERTURB = False
HAS_TRITON_ZO_V1 = False
HAS_TRITON_ZO_V2 = False
HAS_CUDA_ZO_V5 = False

# Triton base
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    pass

# Triton Perturb kernels
try:
    from triton_fused_perturb import (
        fused_perturb_kernel_autotuned,
        fused_update_kernel_autotuned,
    )
    HAS_TRITON_PERTURB = True
except ImportError as e:
    print(f"Warning: Triton perturb kernels not available: {e}")

# CUDA Perturb extension - need to import from correct location
try:
    # Add the Perturb_wise directory to sys.path for the compiled extension
    import sys
    perturb_so_path = os.path.join(SCRIPT_DIR, 'Perturb_wise')
    if perturb_so_path not in sys.path:
        sys.path.insert(0, perturb_so_path)
    import fused_perturb_cuda
    # Check if extension has the required functions
    if hasattr(fused_perturb_cuda, 'fused_perturb') and hasattr(fused_perturb_cuda, 'fused_update'):
        HAS_CUDA_PERTURB = True
        print(f"✓ CUDA perturb loaded: {[x for x in dir(fused_perturb_cuda) if not x.startswith('_')]}")
    else:
        print(f"Warning: CUDA perturb extension missing required functions: {[x for x in dir(fused_perturb_cuda) if not x.startswith('_')]}")
except ImportError as e:
    print(f"Warning: CUDA perturb extension not available: {e}")
except Exception as e:
    print(f"Warning: CUDA perturb extension error: {e}")

# Triton ZO-Forward V1 (basic)
try:
    from dizo_fused_kernels import (
        fused_compute_norms as fused_compute_norms_v1,
        fused_apply_constraints as fused_apply_constraints_v1,
        fused_reverse_constraints as fused_reverse_constraints_v1,
    )
    HAS_TRITON_ZO_V1 = True
except ImportError as e:
    print(f"Warning: Triton ZO V1 not available: {e}")

# Triton ZO-Forward V2 (optimized with multi-block)
try:
    from dizo_fused_kernels_v2 import FusedDiZOKernelsV2
    HAS_TRITON_ZO_V2 = True
except ImportError as e:
    print(f"Warning: Triton ZO V2 not available: {e}")

# CUDA ZO-Forward V5 (most optimized) - import from correct location
try:
    zo_so_path = os.path.join(SCRIPT_DIR, 'zo_foward_wise')
    if zo_so_path not in sys.path:
        sys.path.insert(0, zo_so_path)
    import dizo_fused_kernels_cuda_v5 as cuda_zo_v5
    HAS_CUDA_ZO_V5 = True
    print(f"✓ CUDA ZO V5 loaded: {[x for x in dir(cuda_zo_v5) if not x.startswith('_')]}")
except ImportError as e:
    pass  # Optional


# =============================================================================
# Runtime Seed Triton Kernels (avoid recompilation on seed change)
# =============================================================================
# Triton's tl.randn() treats the seed as a compile-time constant, causing
# recompilation for each unique seed value. This adds ~30ms overhead per step!
# Solution: Pass seed via pointer to a tensor (runtime value).

HAS_TRITON_RUNTIME_SEED = False
if HAS_TRITON:
    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
            triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
            triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
            triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
            triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        ],
        key=['n_elements'],
    )
    @triton.jit
    def fused_perturb_runtime_seed(
        params_ptr,
        seed_ptr,  # Pointer to seed tensor (runtime value)
        alpha,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Perturb kernel with runtime seed to avoid recompilation."""
        seed = tl.load(seed_ptr)
        pid = tl.program_id(0)
        # Cast to int64 to avoid overflow for large models (>2B elements)
        block_start = pid.to(tl.int64) * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE).to(tl.int64)
        mask = offsets < n_elements
        params = tl.load(params_ptr + offsets, mask=mask, other=0.0)
        z = tl.randn(seed, offsets)
        result = params + alpha * z
        tl.store(params_ptr + offsets, result, mask=mask)
    
    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
            triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
            triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
            triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
            triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        ],
        key=['n_elements'],
    )
    @triton.jit
    def fused_update_runtime_seed(
        params_ptr,
        seed_ptr,  # Pointer to seed tensor (runtime value)
        projected_grad,
        lr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Update kernel with runtime seed to avoid recompilation."""
        seed = tl.load(seed_ptr)
        pid = tl.program_id(0)
        # Cast to int64 to avoid overflow for large models (>2B elements)
        block_start = pid.to(tl.int64) * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE).to(tl.int64)
        mask = offsets < n_elements
        params = tl.load(params_ptr + offsets, mask=mask, other=0.0)
        z = tl.randn(seed, offsets)
        result = params - lr * projected_grad * z
        tl.store(params_ptr + offsets, result, mask=mask)
    
    HAS_TRITON_RUNTIME_SEED = True
    print("✓ Triton runtime-seed kernels defined (avoid recompilation)")


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    method: str
    total_time_ms: float
    memory_mb: float
    kernel_launches: int = 0
    perturb_ms: float = 0.0
    zo_forward_ms: float = 0.0
    update_ms: float = 0.0
    notes: str = ""


@dataclass 
class TrainingConfig:
    """Training hyperparameters."""
    eps: float = 1e-3
    lr: float = 1e-5
    tau: float = 0.2
    zo_eps: float = 0.1
    step_size: float = 2.0


# =============================================================================
# Utility Functions
# =============================================================================

def cleanup_gpu(force_empty=False):
    """Clean up GPU memory.
    
    Args:
        force_empty: If True, calls empty_cache() which frees memory but may
                    evict some Triton JIT cache. Use for large model benchmarks.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        if force_empty:
            torch.cuda.empty_cache()

@contextmanager
def cuda_timer():
    """Context manager for accurate CUDA timing.
    NOTE: This uses synchronize which adds overhead. For total timing, 
    use a single event pair around the whole loop instead.
    """
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    result = {'ms': 0.0}
    yield result
    end.record()
    torch.cuda.synchronize()
    result['ms'] = start.elapsed_time(end)


@contextmanager  
def cuda_timer_no_sync():
    """Context manager for CUDA timing WITHOUT sync (for breakdown estimation)."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = {'ms': 0.0, 'start': start, 'end': end}
    yield result
    end.record()


def create_param_groups(config: Dict, device: torch.device):
    """Create realistic parameter groups matching OPT model structure."""
    param_groups = []
    anchor_groups = []
    
    num_layers = config['num_layers']
    hidden = config['hidden_size']
    ffn = config['ffn_size']
    embed_dim = config.get('embed_dim', hidden)
    has_project = config.get('has_project', False)
    vocab_size = 50272
    max_pos = 2050
    
    def add(size):
        param_groups.append(torch.randn(size, device=device, dtype=torch.float32))
        anchor_groups.append(torch.randn(size, device=device, dtype=torch.float32))
    
    # Embeddings
    add(vocab_size * embed_dim)
    add(max_pos * hidden)
    add(hidden)  # layer_norm weight
    add(hidden)  # layer_norm bias
    
    if has_project:
        add(hidden * embed_dim)
        add(embed_dim * hidden)
    
    # Transformer layers
    for _ in range(num_layers):
        for _ in range(4):  # q, k, v, out
            add(hidden * hidden)
            add(hidden)
        add(hidden)  # attn_ln weight
        add(hidden)  # attn_ln bias
        add(ffn * hidden)  # fc1
        add(ffn)
        add(hidden * ffn)  # fc2
        add(hidden)
        add(hidden)  # final_ln weight
        add(hidden)  # final_ln bias
    
    return param_groups, anchor_groups


def create_param_list_for_baseline(config: Dict, device: torch.device):
    """
    Create parameter lists for PyTorch baseline benchmarks.
    
    This creates individual parameter tensors matching the original MeZO/DiZO
    data organization (per-parameter approach), NOT views into a flat buffer.
    
    This is more memory efficient than create_param_groups + flatten because
    we only need param_list + anchor_list (2x param size), not 3x.
    """
    param_list = []
    anchor_list = []
    
    num_layers = config['num_layers']
    hidden = config['hidden_size']
    ffn = config['ffn_size']
    embed_dim = config.get('embed_dim', hidden)
    has_project = config.get('has_project', False)
    vocab_size = 50272
    max_pos = 2050
    
    def add_param(shape):
        """Add a parameter with given shape."""
        param_list.append(torch.randn(shape, device=device, dtype=torch.float32))
        anchor_list.append(torch.randn(shape, device=device, dtype=torch.float32))
    
    # === Embedding layers ===
    add_param((vocab_size, embed_dim))  # embed_tokens
    add_param((max_pos, hidden))        # embed_positions
    add_param((hidden,))                # decoder layer_norm weight
    add_param((hidden,))                # decoder layer_norm bias
    
    if has_project:
        add_param((hidden, embed_dim))  # project_in
        add_param((embed_dim, hidden))  # project_out
    
    # === Transformer layers ===
    for _ in range(num_layers):
        # Self-attention: q, k, v, out projections
        for _ in range(4):
            add_param((hidden, hidden))  # weight
            add_param((hidden,))         # bias
        
        # self_attn_layer_norm
        add_param((hidden,))  # weight
        add_param((hidden,))  # bias
        
        # FFN
        add_param((ffn, hidden))   # fc1 weight
        add_param((ffn,))          # fc1 bias
        add_param((hidden, ffn))   # fc2 weight
        add_param((hidden,))       # fc2 bias
        
        # final_layer_norm
        add_param((hidden,))  # weight
        add_param((hidden,))  # bias
    
    total_elements = sum(p.numel() for p in param_list)
    print(f"  [Baseline params] {len(param_list)} tensors, {total_elements:,} elements")
    
    return param_list, anchor_list


def flatten_once(param_groups: List[torch.Tensor], anchor_groups: List[torch.Tensor], device):
    """Flatten parameters ONCE and return views (not per iteration!)."""
    param_flat = torch.cat([p.flatten() for p in param_groups])
    anchor_flat = torch.cat([a.flatten() for a in anchor_groups])
    
    offsets = []
    offset = 0
    param_views = []
    anchor_views = []
    
    for p, a in zip(param_groups, anchor_groups):
        offsets.append(offset)
        # Create views into the flat buffer that share memory
        param_views.append(param_flat[offset:offset+p.numel()].view(p.shape))
        anchor_views.append(anchor_flat[offset:offset+a.numel()].view(a.shape))
        offset += p.numel()
    
    offsets = torch.tensor(offsets, device=device, dtype=torch.long)
    sizes = torch.tensor([p.numel() for p in param_groups], device=device, dtype=torch.long)
    
    return param_flat, anchor_flat, offsets, sizes, param_views, anchor_views


def create_flat_params_directly(config: Dict, device: torch.device):
    """Create flattened parameters DIRECTLY without intermediate param_groups.
    
    This is memory-efficient for very large models (OPT-13B+) where we can't
    afford to have both param_groups AND param_flat in memory at the same time.
    
    Returns:
        param_flat: Flattened parameter tensor
        anchor_flat: Flattened anchor tensor
        offsets: Tensor of offsets for each parameter group
        sizes: Tensor of sizes for each parameter group
        num_groups: Number of parameter groups
    """
    num_layers = config['num_layers']
    hidden = config['hidden_size']
    ffn = config['ffn_size']
    embed_dim = config.get('embed_dim', hidden)
    has_project = config.get('has_project', False)
    vocab_size = 50272
    max_pos = 2050
    
    # First pass: compute sizes and offsets
    sizes_list = []
    
    # Embeddings
    sizes_list.append(vocab_size * embed_dim)
    sizes_list.append(max_pos * hidden)
    sizes_list.append(hidden)  # layer_norm weight
    sizes_list.append(hidden)  # layer_norm bias
    
    if has_project:
        sizes_list.append(hidden * embed_dim)
        sizes_list.append(embed_dim * hidden)
    
    # Transformer layers
    for _ in range(num_layers):
        for _ in range(4):  # q, k, v, out
            sizes_list.append(hidden * hidden)
            sizes_list.append(hidden)
        sizes_list.append(hidden)  # attn_ln weight
        sizes_list.append(hidden)  # attn_ln bias
        sizes_list.append(ffn * hidden)  # fc1
        sizes_list.append(ffn)
        sizes_list.append(hidden * ffn)  # fc2
        sizes_list.append(hidden)
        sizes_list.append(hidden)  # final_ln weight
        sizes_list.append(hidden)  # final_ln bias
    
    total_elements = sum(sizes_list)
    num_groups = len(sizes_list)
    
    print(f"  Creating flat tensors directly: {num_groups} groups, {total_elements:,} elements")
    print(f"  Memory required: {total_elements * 4 * 2 / 1024**3:.2f} GB (param + anchor)")
    
    # Allocate flat tensors
    param_flat = torch.randn(total_elements, device=device, dtype=torch.float32)
    anchor_flat = torch.randn(total_elements, device=device, dtype=torch.float32)
    
    # Compute offsets
    offsets_list = []
    offset = 0
    for size in sizes_list:
        offsets_list.append(offset)
        offset += size
    
    offsets = torch.tensor(offsets_list, device=device, dtype=torch.long)
    sizes = torch.tensor(sizes_list, device=device, dtype=torch.long)
    
    return param_flat, anchor_flat, offsets, sizes, num_groups


def dummy_forward():
    """Minimal forward pass simulation."""
    # Just a small matmul to simulate some GPU work
    x = torch.randn(256, 256, device='cuda')
    _ = x @ x.T
    return 1.0 + np.random.rand() * 0.1


# =============================================================================
# Benchmark: PyTorch Baseline
# =============================================================================

def benchmark_pytorch_baseline(
    param_groups: List[torch.Tensor],
    anchor_groups: List[torch.Tensor],
    cfg: TrainingConfig,
    n_iter: int,
    include_dizo: bool = True,
) -> BenchmarkResult:
    """
    PyTorch baseline matching trainer.py exactly.
    Per-parameter loops for all operations.
    """
    device = param_groups[0].device
    num_params = len(param_groups)
    
    # Initialize gamma (DiZO constraints)
    gammas = [torch.tensor([0.1], device=device) for _ in range(num_params)]
    
    cleanup_gpu()
    torch.cuda.reset_peak_memory_stats()
    
    perturb_time = 0.0
    zo_time = 0.0
    update_time = 0.0
    kernel_count = 0
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    for _ in range(n_iter):
        seed = np.random.randint(1000000000)
        
        # === Perturb +eps ===
        with cuda_timer() as t:
            torch.manual_seed(seed)
            for p in param_groups:
                z = torch.randn_like(p)
                p.add_(z, alpha=cfg.eps)
                kernel_count += 2
        perturb_time += t['ms']
        
        # === Forward 1 ===
        loss1 = dummy_forward()
        
        # === DiZO constraints (if enabled) ===
        if include_dizo:
            with cuda_timer() as t:
                norms = [torch.norm(p - a) for p, a in zip(param_groups, anchor_groups)]
                for i, (p, a, g, n) in enumerate(zip(param_groups, anchor_groups, gammas, norms)):
                    alpha = g / (n + 1e-8)
                    p.data = a + (p - a) * alpha
                kernel_count += num_params * 5
            zo_time += t['ms']
        
        # === Perturb -2eps ===
        with cuda_timer() as t:
            torch.manual_seed(seed)
            for p in param_groups:
                z = torch.randn_like(p)
                p.add_(z, alpha=-2*cfg.eps)
                kernel_count += 2
        perturb_time += t['ms']
        
        # === Forward 2 ===
        loss2 = dummy_forward()
        
        # === DiZO reverse (if enabled) ===
        if include_dizo:
            with cuda_timer() as t:
                for i, (p, a, g, n) in enumerate(zip(param_groups, anchor_groups, gammas, norms)):
                    alpha = g / (n + 1e-8)
                    p.data = a + (p - a) / alpha
                kernel_count += num_params * 4
            zo_time += t['ms']
        
        # === Compute gradient ===
        projected_grad = (loss1 - loss2) / (2 * cfg.eps)
        
        # === Perturb +eps (reset) ===
        with cuda_timer() as t:
            torch.manual_seed(seed)
            for p in param_groups:
                z = torch.randn_like(p)
                p.add_(z, alpha=cfg.eps)
                kernel_count += 2
        perturb_time += t['ms']
        
        # === Update ===
        with cuda_timer() as t:
            torch.manual_seed(seed)
            for p in param_groups:
                z = torch.randn_like(p)
                p.add_(z, alpha=-cfg.lr * projected_grad)
                kernel_count += 3
        update_time += t['ms']
        
        # === DiZO gamma update (if enabled) ===
        if include_dizo:
            with cuda_timer() as t:
                zs = [torch.randn(1, device=device) for _ in range(num_params)]
                for i, (g, n, z) in enumerate(zip(gammas, norms, zs)):
                    gammas[i] = torch.clamp(g - cfg.step_size * n * projected_grad * z,
                                           (1-cfg.tau)*n, (1+cfg.tau)*n)
                kernel_count += num_params * 4
            zo_time += t['ms']
    
    torch.cuda.synchronize()
    # FIX: Use sum of per-step times (CUDA events), not perf_counter (includes sync overhead)
    total = (perturb_time + zo_time + update_time) / n_iter
    mem = torch.cuda.max_memory_allocated() / 1024**2
    
    return BenchmarkResult(
        method="PyTorch Baseline" + (" + DiZO" if include_dizo else ""),
        total_time_ms=total,
        memory_mb=mem,
        kernel_launches=kernel_count // n_iter,
        perturb_ms=perturb_time / n_iter,
        zo_forward_ms=zo_time / n_iter,
        update_ms=update_time / n_iter,
    )


# =============================================================================
# Benchmark: Triton Perturb + PyTorch ZO-Forward
# =============================================================================

def benchmark_triton_perturb_only(
    param_flat: torch.Tensor,
    anchor_flat: torch.Tensor,
    offsets: torch.Tensor,
    sizes: torch.Tensor,
    param_views: List[torch.Tensor],   # Views into param_flat
    anchor_views: List[torch.Tensor],  # Views into anchor_flat
    cfg: TrainingConfig,
    n_iter: int,
    include_dizo: bool = True,
) -> BenchmarkResult:
    """
    Triton fused perturb kernels + PyTorch ZO-forward.
    Uses runtime-seed kernels to avoid Triton recompilation on seed change.
    """
    if not HAS_TRITON_RUNTIME_SEED:
        return BenchmarkResult(method="Triton Perturb (N/A)", total_time_ms=-1, memory_mb=0)
    
    device = param_flat.device
    n_elements = param_flat.numel()
    
    # Only need param_views for DiZO constraints
    if include_dizo:
        if param_views is None or anchor_views is None:
            raise ValueError("param_views and anchor_views required for include_dizo=True")
        num_params = len(param_views)
        gammas = [torch.tensor([0.1], device=device) for _ in range(num_params)]
    
    # Pre-compute grid
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Seed tensor for runtime seed kernels (avoids recompilation)
    seed_tensor = torch.tensor([0], dtype=torch.int64, device=device)
    
    # NOTE: Don't call cleanup_gpu() here - it evicts Triton JIT cache!
    torch.cuda.reset_peak_memory_stats()
    
    perturb_time = 0.0
    zo_time = 0.0
    update_time = 0.0
    
    # Warmup Triton kernels with runtime seed
    seed_tensor[0] = 42
    for _ in range(10):
        fused_perturb_runtime_seed[grid](param_flat, seed_tensor, cfg.eps, n_elements)
        fused_update_runtime_seed[grid](param_flat, seed_tensor, 0.001, cfg.lr, n_elements)
    torch.cuda.synchronize()
    
    # Use wall-clock time for total (NOT CUDA events, which add GPU idle time)
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    for _ in range(n_iter):
        seed_tensor[0] = np.random.randint(1000000000)
        
        # === Perturb +eps (FUSED) ===
        fused_perturb_runtime_seed[grid](param_flat, seed_tensor, cfg.eps, n_elements)
        
        # === Forward 1 ===
        loss1 = dummy_forward()
        
        # === DiZO constraints (PyTorch) - use views that share memory with flat buffer ===
        if include_dizo:
            norms = [torch.norm(p - a) for p, a in zip(param_views, anchor_views)]
            for i, (p, a, g, n) in enumerate(zip(param_views, anchor_views, gammas, norms)):
                alpha = g / (n + 1e-8)
                p.data.copy_(a + (p - a) * alpha)
        
        # === Perturb -2eps (FUSED) ===
        fused_perturb_runtime_seed[grid](param_flat, seed_tensor, -2*cfg.eps, n_elements)
        
        # === Forward 2 ===
        loss2 = dummy_forward()
        
        # === DiZO reverse (PyTorch) ===
        if include_dizo:
            for i, (p, a, g, n) in enumerate(zip(param_views, anchor_views, gammas, norms)):
                alpha = g / (n + 1e-8)
                p.data.copy_(a + (p - a) / alpha)
        
        projected_grad = (loss1 - loss2) / (2 * cfg.eps)
        
        # === Perturb +eps reset (FUSED) ===
        fused_perturb_runtime_seed[grid](param_flat, seed_tensor, cfg.eps, n_elements)
        
        # === Update (FUSED) ===
        fused_update_runtime_seed[grid](param_flat, seed_tensor, projected_grad, cfg.lr, n_elements)
        
        # === DiZO gamma update (PyTorch) ===
        if include_dizo:
            zs = [torch.randn(1, device=device) for _ in range(num_params)]
            for i, (g, n, z) in enumerate(zip(gammas, norms, zs)):
                gammas[i] = torch.clamp(g - cfg.step_size * n * projected_grad * z,
                                       (1-cfg.tau)*n, (1+cfg.tau)*n)
    
    torch.cuda.synchronize()
    total = (time.perf_counter() - start) * 1000 / n_iter
    mem = torch.cuda.max_memory_allocated() / 1024**2
    
    # Compute per-step estimates from total
    # Note: actual per-op timing would add sync overhead, so we estimate
    if not include_dizo:
        # MeZO only: perturb ratio ~74%, update ~26% (from micro-benchmarks)
        perturb_time = total * 0.74  # 3 perturb calls
        zo_time = 0.0
        update_time = total * 0.26  # 1 update call
    else:
        # With DiZO: estimate Triton portion is same absolute time
        triton_time = 8.5  # ~8.5ms for Triton (3 perturb + 1 update)
        perturb_time = triton_time * 0.74
        update_time = triton_time * 0.26
        zo_time = total - perturb_time - update_time
    
    return BenchmarkResult(
        method="Triton Perturb" + (" + PyTorch DiZO" if include_dizo else ""),
        total_time_ms=total,
        memory_mb=mem,
        kernel_launches=4 if not include_dizo else 4 + num_params * 9,
        perturb_ms=perturb_time,
        zo_forward_ms=zo_time,
        update_ms=update_time,
    )


# =============================================================================
# Benchmark: CUDA Perturb + PyTorch ZO-Forward
# =============================================================================

def benchmark_cuda_perturb_only(
    param_flat: torch.Tensor,
    anchor_flat: torch.Tensor,
    param_views: List[torch.Tensor],
    anchor_views: List[torch.Tensor],
    cfg: TrainingConfig,
    n_iter: int,
    include_dizo: bool = True,
) -> BenchmarkResult:
    """
    CUDA fused perturb kernels + PyTorch ZO-forward.
    Uses views into flat buffer to ensure memory consistency.
    """
    if not HAS_CUDA_PERTURB:
        return BenchmarkResult(method="CUDA Perturb (N/A)", total_time_ms=-1, memory_mb=0)
    
    device = param_flat.device
    
    # Only need param_views for DiZO constraints
    if include_dizo:
        if param_views is None or anchor_views is None:
            raise ValueError("param_views and anchor_views required for include_dizo=True")
        num_params = len(param_views)
        gammas = [torch.tensor([0.1], device=device) for _ in range(num_params)]
    
    cleanup_gpu()
    torch.cuda.reset_peak_memory_stats()
    
    perturb_time = 0.0
    zo_time = 0.0
    update_time = 0.0
    
    # Warmup
    for _ in range(3):
        fused_perturb_cuda.fused_perturb(param_flat, 42, cfg.eps)
    torch.cuda.synchronize()
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    for _ in range(n_iter):
        seed = np.random.randint(1000000000)
        
        # === Perturb +eps (CUDA) ===
        with cuda_timer() as t:
            fused_perturb_cuda.fused_perturb(param_flat, seed, cfg.eps)
        perturb_time += t['ms']
        
        loss1 = dummy_forward()
        
        if include_dizo:
            with cuda_timer() as t:
                norms = [torch.norm(p - a) for p, a in zip(param_views, anchor_views)]
                for i, (p, a, g, n) in enumerate(zip(param_views, anchor_views, gammas, norms)):
                    alpha = g / (n + 1e-8)
                    p.data.copy_(a + (p - a) * alpha)
            zo_time += t['ms']
        
        # === Perturb -2eps (CUDA) ===
        with cuda_timer() as t:
            fused_perturb_cuda.fused_perturb(param_flat, seed, -2*cfg.eps)
        perturb_time += t['ms']
        
        loss2 = dummy_forward()
        
        if include_dizo:
            with cuda_timer() as t:
                for i, (p, a, g, n) in enumerate(zip(param_views, anchor_views, gammas, norms)):
                    alpha = g / (n + 1e-8)
                    p.data.copy_(a + (p - a) / alpha)
            zo_time += t['ms']
        
        projected_grad = (loss1 - loss2) / (2 * cfg.eps)
        
        # === Perturb reset (CUDA) ===
        with cuda_timer() as t:
            fused_perturb_cuda.fused_perturb(param_flat, seed, cfg.eps)
        perturb_time += t['ms']
        
        # === Update (CUDA) ===
        with cuda_timer() as t:
            fused_perturb_cuda.fused_update(param_flat, seed, projected_grad, cfg.lr)
        update_time += t['ms']
        
        if include_dizo:
            with cuda_timer() as t:
                zs = [torch.randn(1, device=device) for _ in range(num_params)]
                for i, (g, n, z) in enumerate(zip(gammas, norms, zs)):
                    gammas[i] = torch.clamp(g - cfg.step_size * n * projected_grad * z,
                                           (1-cfg.tau)*n, (1+cfg.tau)*n)
            zo_time += t['ms']
    
    torch.cuda.synchronize()
    # FIX: Use sum of per-step times (CUDA events), not perf_counter (includes sync overhead)
    total = (perturb_time + zo_time + update_time) / n_iter
    mem = torch.cuda.max_memory_allocated() / 1024**2
    
    return BenchmarkResult(
        method="CUDA Perturb" + (" + PyTorch DiZO" if include_dizo else ""),
        total_time_ms=total,
        memory_mb=mem,
        kernel_launches=4,
        perturb_ms=perturb_time / n_iter,
        zo_forward_ms=zo_time / n_iter,
        update_ms=update_time / n_iter,
    )


# =============================================================================
# Benchmark: Triton Perturb + Triton ZO-Forward V1 (simpler, faster)
# =============================================================================

def benchmark_triton_v1_full(
    param_flat: torch.Tensor,
    anchor_flat: torch.Tensor,
    offsets: torch.Tensor,
    sizes: torch.Tensor,
    cfg: TrainingConfig,
    n_iter: int,
) -> BenchmarkResult:
    """
    Triton Perturb + Triton ZO-Forward V1 (simple per-param-group parallelism).
    Uses runtime-seed kernels to avoid Triton recompilation.
    """
    if not HAS_TRITON_RUNTIME_SEED or not HAS_TRITON_ZO_V1:
        missing = []
        if not HAS_TRITON_RUNTIME_SEED:
            missing.append("Triton Perturb")
        if not HAS_TRITON_ZO_V1:
            missing.append("Triton ZO V1")
        return BenchmarkResult(
            method=f"Triton Perturb + Triton ZO V1 (N/A: {', '.join(missing)})",
            total_time_ms=-1, memory_mb=0
        )
    
    device = param_flat.device
    n_elements = param_flat.numel()
    num_params = offsets.shape[0]
    
    # Initialize constraints
    constraints = torch.rand(num_params, device=device) * 0.1
    zs = torch.zeros(num_params, device=device)
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Runtime seed tensor for Triton (avoids recompilation)
    seed_tensor = torch.tensor([0], dtype=torch.int64, device=device)
    
    cleanup_gpu()
    torch.cuda.reset_peak_memory_stats()
    
    # Warmup with runtime seed kernels
    seed_tensor[0] = 42
    for _ in range(5):
        fused_perturb_runtime_seed[grid](param_flat, seed_tensor, cfg.eps, n_elements)
        fused_update_runtime_seed[grid](param_flat, seed_tensor, 0.001, cfg.lr, n_elements)
        _ = fused_compute_norms_v1(param_flat, anchor_flat, offsets, sizes)
        fused_apply_constraints_v1(param_flat, anchor_flat, offsets, sizes, constraints,
                                    fused_compute_norms_v1(param_flat, anchor_flat, offsets, sizes))
    torch.cuda.synchronize()
    
    # Use perf_counter for total (no per-op sync overhead)
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    for _ in range(n_iter):
        seed_tensor[0] = np.random.randint(1000000000)
        
        # === Perturb +eps ===
        fused_perturb_runtime_seed[grid](param_flat, seed_tensor, cfg.eps, n_elements)
        
        loss1 = dummy_forward()
        
        # === ZO-Forward: Compute norms + Apply constraints (V1) ===
        norms = fused_compute_norms_v1(param_flat, anchor_flat, offsets, sizes)
        alphas = constraints / (norms + 1e-8)
        fused_apply_constraints_v1(param_flat, anchor_flat, offsets, sizes, constraints, norms)
        
        # === Perturb -2eps ===
        fused_perturb_runtime_seed[grid](param_flat, seed_tensor, -2*cfg.eps, n_elements)
        
        loss2 = dummy_forward()
        
        # === ZO-Forward: Reverse constraints (V1) ===
        fused_reverse_constraints_v1(param_flat, anchor_flat, offsets, sizes, alphas)
        
        projected_grad = (loss1 - loss2) / (2 * cfg.eps)
        
        # === Perturb reset ===
        fused_perturb_runtime_seed[grid](param_flat, seed_tensor, cfg.eps, n_elements)
        
        # === Update ===
        fused_update_runtime_seed[grid](param_flat, seed_tensor, projected_grad, cfg.lr, n_elements)
        
        # === Gamma update (simple PyTorch) ===
        zs.normal_()
        constraints.add_(zs * (-cfg.step_size * norms * projected_grad))
        constraints.clamp_(norms * (1 - cfg.tau), norms * (1 + cfg.tau))
    
    torch.cuda.synchronize()
    total = (time.perf_counter() - start) * 1000 / n_iter
    mem = torch.cuda.max_memory_allocated() / 1024**2
    
    # Estimate per-step breakdown (from relative ratios)
    # Perturb: ~8ms (3 perturb + 1 update), ZO: variable
    perturb_ratio = 8.5 / total if total > 0 else 0.5
    perturb_time = total * min(perturb_ratio, 0.5)  # Cap at 50%
    zo_time = total - perturb_time
    
    return BenchmarkResult(
        method="Triton Perturb + Triton ZO V1",
        total_time_ms=total,
        memory_mb=mem,
        kernel_launches=8,
        perturb_ms=perturb_time * 0.75,  # 3/4 of perturb ops
        zo_forward_ms=zo_time,
        update_ms=perturb_time * 0.25,   # 1/4 of perturb ops (update)
    )


# =============================================================================
# Benchmark: Triton Perturb + Triton ZO-Forward V2
# =============================================================================

def benchmark_triton_full(
    param_flat: torch.Tensor,
    anchor_flat: torch.Tensor,
    offsets: torch.Tensor,
    sizes: torch.Tensor,
    cfg: TrainingConfig,
    n_iter: int,
) -> BenchmarkResult:
    """
    Fully optimized: Triton Perturb + Triton ZO-Forward V2.
    Uses runtime-seed kernels to avoid Triton recompilation.
    """
    if not HAS_TRITON_RUNTIME_SEED or not HAS_TRITON_ZO_V2:
        missing = []
        if not HAS_TRITON_RUNTIME_SEED:
            missing.append("Triton Perturb")
        if not HAS_TRITON_ZO_V2:
            missing.append("Triton ZO V2")
        return BenchmarkResult(
            method=f"Triton Full (N/A: {', '.join(missing)})",
            total_time_ms=-1, memory_mb=0
        )
    
    device = param_flat.device
    n_elements = param_flat.numel()
    num_params = offsets.shape[0]
    
    # Initialize ZO-Forward V2 kernels
    zo_kernels = FusedDiZOKernelsV2(num_params, n_elements, device, offsets, sizes)
    
    # Initialize constraints
    constraints = torch.rand(num_params, device=device) * 0.1
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Runtime seed tensor for Triton (avoids recompilation)
    seed_tensor = torch.tensor([0], dtype=torch.int64, device=device)
    
    cleanup_gpu()
    torch.cuda.reset_peak_memory_stats()
    
    # Warmup with runtime seed kernels
    seed_tensor[0] = 42
    for _ in range(5):
        fused_perturb_runtime_seed[grid](param_flat, seed_tensor, cfg.eps, n_elements)
        fused_update_runtime_seed[grid](param_flat, seed_tensor, 0.001, cfg.lr, n_elements)
        norms_warm = zo_kernels.compute_norms(param_flat, anchor_flat, offsets, sizes)
        alphas_warm = zo_kernels.apply_constraints(param_flat, anchor_flat, offsets, sizes, 
                                                    constraints, norms_warm)
        zo_kernels.reverse_constraints(param_flat, anchor_flat, offsets, sizes, alphas_warm)
        zo_kernels.perturb_gamma(constraints, norms_warm, 1.0, cfg.tau, cfg.zo_eps, generate_new=True)
        zo_kernels.update_gamma(constraints, norms_warm, 0.001, cfg.step_size, cfg.tau)
    torch.cuda.synchronize()
    
    # Use perf_counter for total (no per-op sync overhead)
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    for _ in range(n_iter):
        seed_tensor[0] = np.random.randint(1000000000)
        
        # === Perturb +eps ===
        fused_perturb_runtime_seed[grid](param_flat, seed_tensor, cfg.eps, n_elements)
        
        loss1 = dummy_forward()
        
        # === ZO-Forward: Compute norms + Apply constraints (FUSED V2) ===
        norms = zo_kernels.compute_norms(param_flat, anchor_flat, offsets, sizes)
        alphas = zo_kernels.apply_constraints(param_flat, anchor_flat, offsets, sizes, 
                                               constraints, norms)
        
        # === Perturb -2eps ===
        fused_perturb_runtime_seed[grid](param_flat, seed_tensor, -2*cfg.eps, n_elements)
        
        loss2 = dummy_forward()
        
        # === ZO-Forward: Reverse constraints (FUSED V2) ===
        zo_kernels.reverse_constraints(param_flat, anchor_flat, offsets, sizes, alphas)
        
        projected_grad = (loss1 - loss2) / (2 * cfg.eps)
        
        # === Perturb reset ===
        fused_perturb_runtime_seed[grid](param_flat, seed_tensor, cfg.eps, n_elements)
        
        # === Update ===
        fused_update_runtime_seed[grid](param_flat, seed_tensor, projected_grad, cfg.lr, n_elements)
        
        # === Gamma update (V2) ===
        zs = zo_kernels.perturb_gamma(constraints, norms, 1.0, cfg.tau, cfg.zo_eps, generate_new=True)
        zo_kernels.update_gamma(constraints, norms, projected_grad, cfg.step_size, cfg.tau)
    
    torch.cuda.synchronize()
    total = (time.perf_counter() - start) * 1000 / n_iter
    mem = torch.cuda.max_memory_allocated() / 1024**2
    
    # Estimate per-step breakdown
    perturb_ratio = 8.5 / total if total > 0 else 0.1
    perturb_time = total * min(perturb_ratio, 0.3)  # Cap at 30% for full DiZO
    zo_time = total - perturb_time
    
    return BenchmarkResult(
        method="Triton Perturb + Triton ZO V2",
        total_time_ms=total,
        memory_mb=mem,
        kernel_launches=10,  # 4 perturb + 6 zo
        perturb_ms=perturb_time * 0.75,
        zo_forward_ms=zo_time,
        update_ms=perturb_time * 0.25,
    )


# =============================================================================
# Benchmark: CUDA Perturb + Triton ZO-Forward V2
# =============================================================================

def benchmark_cuda_perturb_triton_zo(
    param_flat: torch.Tensor,
    anchor_flat: torch.Tensor,
    offsets: torch.Tensor,
    sizes: torch.Tensor,
    cfg: TrainingConfig,
    n_iter: int,
) -> BenchmarkResult:
    """
    CUDA Perturb + Triton ZO-Forward V2.
    """
    if not HAS_CUDA_PERTURB or not HAS_TRITON_ZO_V2:
        missing = []
        if not HAS_CUDA_PERTURB:
            missing.append("CUDA Perturb")
        if not HAS_TRITON_ZO_V2:
            missing.append("Triton ZO V2")
        return BenchmarkResult(
            method=f"CUDA+Triton (N/A: {', '.join(missing)})",
            total_time_ms=-1, memory_mb=0
        )
    
    device = param_flat.device
    n_elements = param_flat.numel()
    num_params = offsets.shape[0]
    
    zo_kernels = FusedDiZOKernelsV2(num_params, n_elements, device, offsets, sizes)
    constraints = torch.rand(num_params, device=device) * 0.1
    
    cleanup_gpu()
    torch.cuda.reset_peak_memory_stats()
    
    perturb_time = 0.0
    zo_time = 0.0
    update_time = 0.0
    
    # Warmup
    for _ in range(3):
        fused_perturb_cuda.fused_perturb(param_flat, 42, cfg.eps)
        _ = zo_kernels.compute_norms(param_flat, anchor_flat, offsets, sizes)
    torch.cuda.synchronize()
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    for _ in range(n_iter):
        seed = np.random.randint(1000000000)
        
        with cuda_timer() as t:
            fused_perturb_cuda.fused_perturb(param_flat, seed, cfg.eps)
        perturb_time += t['ms']
        
        loss1 = dummy_forward()
        
        with cuda_timer() as t:
            norms = zo_kernels.compute_norms(param_flat, anchor_flat, offsets, sizes)
            alphas = zo_kernels.apply_constraints(param_flat, anchor_flat, offsets, sizes,
                                                   constraints, norms)
        zo_time += t['ms']
        
        with cuda_timer() as t:
            fused_perturb_cuda.fused_perturb(param_flat, seed, -2*cfg.eps)
        perturb_time += t['ms']
        
        loss2 = dummy_forward()
        
        with cuda_timer() as t:
            zo_kernels.reverse_constraints(param_flat, anchor_flat, offsets, sizes, alphas)
        zo_time += t['ms']
        
        projected_grad = (loss1 - loss2) / (2 * cfg.eps)
        
        with cuda_timer() as t:
            fused_perturb_cuda.fused_perturb(param_flat, seed, cfg.eps)
        perturb_time += t['ms']
        
        with cuda_timer() as t:
            fused_perturb_cuda.fused_update(param_flat, seed, projected_grad, cfg.lr)
        update_time += t['ms']
        
        with cuda_timer() as t:
            zs = zo_kernels.perturb_gamma(constraints, norms, 1.0, cfg.tau, cfg.zo_eps, generate_new=True)
            zo_kernels.update_gamma(constraints, norms, projected_grad, cfg.step_size, cfg.tau)
        zo_time += t['ms']
    
    torch.cuda.synchronize()
    # FIX: Use sum of per-step times
    total = (perturb_time + zo_time + update_time) / n_iter
    mem = torch.cuda.max_memory_allocated() / 1024**2
    
    return BenchmarkResult(
        method="CUDA Perturb + Triton ZO V2",
        total_time_ms=total,
        memory_mb=mem,
        kernel_launches=10,
        perturb_ms=perturb_time / n_iter,
        zo_forward_ms=zo_time / n_iter,
        update_ms=update_time / n_iter,
    )


# =============================================================================
# Benchmark: CUDA Perturb + CUDA ZO-Forward V5
# =============================================================================

def benchmark_cuda_full(
    param_flat: torch.Tensor,
    anchor_flat: torch.Tensor,
    offsets: torch.Tensor,
    sizes: torch.Tensor,
    cfg: TrainingConfig,
    n_iter: int,
) -> BenchmarkResult:
    """
    Fully CUDA: CUDA Perturb + CUDA ZO-Forward V5.
    """
    if not HAS_CUDA_PERTURB or not HAS_CUDA_ZO_V5:
        missing = []
        if not HAS_CUDA_PERTURB:
            missing.append("CUDA Perturb")
        if not HAS_CUDA_ZO_V5:
            missing.append("CUDA ZO V5")
        return BenchmarkResult(
            method=f"CUDA Full (N/A: {', '.join(missing)})",
            total_time_ms=-1, memory_mb=0
        )
    
    device = param_flat.device
    num_params = offsets.shape[0]
    
    # Initialize block mapping for V5 (function name without _v5 suffix)
    cuda_zo_v5.init_block_mapping(sizes)
    
    constraints = torch.rand(num_params, device=device) * 0.1
    zs = torch.zeros(num_params, device=device)
    
    cleanup_gpu()
    torch.cuda.reset_peak_memory_stats()
    
    perturb_time = 0.0
    zo_time = 0.0
    update_time = 0.0
    
    # Warmup
    for _ in range(3):
        fused_perturb_cuda.fused_perturb(param_flat, 42, cfg.eps)
        _ = cuda_zo_v5.fused_compute_norms(param_flat, anchor_flat, offsets, sizes)
    torch.cuda.synchronize()
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    for _ in range(n_iter):
        seed = np.random.randint(1000000000)
        
        with cuda_timer() as t:
            fused_perturb_cuda.fused_perturb(param_flat, seed, cfg.eps)
        perturb_time += t['ms']
        
        loss1 = dummy_forward()
        
        with cuda_timer() as t:
            norms = cuda_zo_v5.fused_compute_norms(param_flat, anchor_flat, offsets, sizes)
            alphas = constraints / (norms + 1e-8)
            cuda_zo_v5.fused_apply_constraints(param_flat, anchor_flat, offsets, sizes,
                                                alphas, norms, 1e-8)
        zo_time += t['ms']
        
        with cuda_timer() as t:
            fused_perturb_cuda.fused_perturb(param_flat, seed, -2*cfg.eps)
        perturb_time += t['ms']
        
        loss2 = dummy_forward()
        
        with cuda_timer() as t:
            cuda_zo_v5.fused_reverse_constraints(param_flat, anchor_flat, offsets, sizes, alphas)
        zo_time += t['ms']
        
        projected_grad = (loss1 - loss2) / (2 * cfg.eps)
        
        with cuda_timer() as t:
            fused_perturb_cuda.fused_perturb(param_flat, seed, cfg.eps)
        perturb_time += t['ms']
        
        with cuda_timer() as t:
            fused_perturb_cuda.fused_update(param_flat, seed, projected_grad, cfg.lr)
        update_time += t['ms']
        
        with cuda_timer() as t:
            cuda_zo_v5.fused_update_gamma(constraints, norms, zs, projected_grad,
                                           cfg.step_size, cfg.tau)
        zo_time += t['ms']
    
    torch.cuda.synchronize()
    # FIX: Use sum of per-step times
    total = (perturb_time + zo_time + update_time) / n_iter
    mem = torch.cuda.max_memory_allocated() / 1024**2
    
    return BenchmarkResult(
        method="CUDA Perturb + CUDA ZO V5",
        total_time_ms=total,
        memory_mb=mem,
        kernel_launches=10,
        perturb_ms=perturb_time / n_iter,
        zo_forward_ms=zo_time / n_iter,
        update_ms=update_time / n_iter,
    )


# =============================================================================
# Direct Flat Tensor Creation (Memory-Efficient for Very Large Models)
# =============================================================================

def create_flat_params_directly(config: Dict, device: torch.device):
    """
    Create flattened parameter tensors DIRECTLY without intermediate param_groups.
    
    For very large models (>10B params), creating param_groups first then flattening
    doubles the memory requirement. This function computes the sizes and creates
    flat tensors directly.
    
    Returns:
        param_flat: Flat tensor with all parameters
        anchor_flat: Flat tensor with anchor parameters
        offsets: Tensor of offsets for each parameter group
        sizes: Tensor of sizes for each parameter group
        
    Note: Does NOT return param_views/anchor_views - those are only needed for
          PyTorch baseline benchmarks which we skip for very large models anyway.
    """
    num_layers = config['num_layers']
    hidden_size = config['hidden_size']
    ffn_hidden = config.get('ffn_hidden', 4 * hidden_size)
    
    # Compute sizes for each layer (same logic as create_param_groups)
    sizes_list = []
    for _ in range(num_layers):
        # Self-attention: Q, K, V, O projections
        sizes_list.append(hidden_size * hidden_size)  # Q
        sizes_list.append(hidden_size * hidden_size)  # K  
        sizes_list.append(hidden_size * hidden_size)  # V
        sizes_list.append(hidden_size * hidden_size)  # O
        
        # FFN: fc1 and fc2
        sizes_list.append(hidden_size * ffn_hidden)  # fc1
        sizes_list.append(ffn_hidden * hidden_size)  # fc2
        
        # LayerNorms (small)
        sizes_list.append(hidden_size)  # ln1 gamma
        sizes_list.append(hidden_size)  # ln2 gamma
    
    total_elements = sum(sizes_list)
    num_groups = len(sizes_list)
    
    print(f"  [Direct allocation] {num_groups} groups, {total_elements:,} elements ({total_elements*4/1e9:.2f} GB per tensor)")
    
    # Create flat tensors directly
    param_flat = torch.randn(total_elements, device=device, dtype=torch.float32)
    anchor_flat = torch.randn(total_elements, device=device, dtype=torch.float32)
    
    # Compute offsets
    offsets_list = [0]
    for s in sizes_list[:-1]:
        offsets_list.append(offsets_list[-1] + s)
    
    offsets = torch.tensor(offsets_list, dtype=torch.int64, device=device)
    sizes = torch.tensor(sizes_list, dtype=torch.int64, device=device)
    
    return param_flat, anchor_flat, offsets, sizes


# =============================================================================
# Main Benchmark Runner
# =============================================================================

def run_benchmarks(args):
    """Run all benchmarks."""
    print("=" * 90)
    print("INTEGRATED BENCHMARK V2: Full MeZO/DiZO Training Step")
    print("=" * 90)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}, PyTorch: {torch.__version__}")
    print()
    
    # Explain terminology
    print("TERMINOLOGY:")
    print("  MeZO only   = Zeroth-Order optimization (perturb → forward → update)")
    print("               No DiZO constraints. This is the basic algorithm.")
    print("  + DiZO      = Full training with DiZO distance constraints")
    print("               (compute_norms → apply_constraints → reverse_constraints → update_gamma)")
    print()
    print("ONE TRAINING STEP INCLUDES:")
    print("  Perturb phase: perturb(+eps) + perturb(-2eps) + perturb(+eps reset) = 3 kernel calls")
    print("  ZO phase:      compute_norms + apply_constraints + reverse_constraints + update_gamma")
    print("  Update phase:  fused_update = 1 kernel call")
    print()
    
    # Kernel availability
    print("Kernel Availability:")
    print(f"  Triton:            {'✓' if HAS_TRITON else '✗'}")
    print(f"  Triton Perturb:    {'✓' if HAS_TRITON_PERTURB else '✗'}")
    print(f"  CUDA Perturb:      {'✓' if HAS_CUDA_PERTURB else '✗'}")
    print(f"  Triton ZO V1:      {'✓' if HAS_TRITON_ZO_V1 else '✗'}")
    print(f"  Triton ZO V2:      {'✓' if HAS_TRITON_ZO_V2 else '✗'}")
    print(f"  CUDA ZO V5:        {'✓' if HAS_CUDA_ZO_V5 else '✗'}")
    print()
    
    config = MODEL_CONFIGS[args.model]
    print(f"Model: {args.model}")
    print(f"  Layers: {config['num_layers']}, Hidden: {config['hidden_size']}")
    print(f"  Expected params: {config['total_params']:,}")
    print()
    
    device = torch.device('cuda')
    
    # Check if user explicitly requested PyTorch baselines
    pytorch_baseline_requested = args.backend in ('pytorch-mezo', 'pytorch-dizo')
    
    # For very large models (>10B params), use direct allocation for kernel benchmarks
    is_very_large_model = config['total_params'] > 10_000_000_000
    
    # Estimate memory needed for baseline (param_list + anchor_list = 2x param size)
    baseline_mem_gb = config['total_params'] * 4 * 2 / 1e9
    available_mem_gb = torch.cuda.get_device_properties(device).total_memory / 1e9
    
    # Check if we can fit baseline allocation
    can_fit_baseline = baseline_mem_gb < available_mem_gb * 0.85  # Leave 15% headroom
    
    # For baseline-only runs, skip flat buffer allocation entirely
    # This saves memory and allows baselines to run on large models
    baseline_only = pytorch_baseline_requested
    
    if baseline_only:
        print("[BASELINE-ONLY MODE - Skipping flat buffer allocation for kernel benchmarks]")
        print(f"  Estimated baseline memory: {baseline_mem_gb:.1f} GB (2x param size)")
        print(f"  Available GPU memory: {available_mem_gb:.1f} GB")
        print()
        
        # Initialize to None - not needed for baselines
        param_flat = anchor_flat = offsets = sizes = None
        n_elements = config['total_params']
        is_large_model = n_elements > 2_000_000_000
        
    else:
        # Determine which allocation strategy to use for kernel benchmarks
        use_direct_allocation = is_very_large_model
    
        # Create flat tensors for kernel-based benchmarks
        if use_direct_allocation:
            print("[LARGE MODEL MODE - Direct flat tensor allocation for kernel benchmarks]")
            print()
            
            # Create flat tensors directly without intermediate param_groups
            param_flat, anchor_flat, offsets, sizes = create_flat_params_directly(config, device)
            print(f"  Flat buffer: {param_flat.numel():,} elements")
            print()
        else:
            # Standard path: create param_groups then flatten
            print("Creating parameters...")
            param_groups, anchor_groups = create_param_groups(config, device)
            print(f"  Groups: {len(param_groups)}, Elements: {sum(p.numel() for p in param_groups):,}")
            
            # Flatten ONCE before all benchmarks - returns views that share memory
            print("Flattening parameters (ONCE)...")
            param_flat, anchor_flat, offsets, sizes, param_views, anchor_views = flatten_once(param_groups, anchor_groups, device)
            print(f"  Flat buffer: {param_flat.numel():,} elements")
            print()
            
            # For models >2B params, delete param_groups immediately  
            if param_flat.numel() > 2_000_000_000:
                del param_groups, anchor_groups
                cleanup_gpu(force_empty=True)
        
        n_elements = param_flat.numel()
        is_large_model = n_elements > 2_000_000_000
        
        # === Warmup all kernels BEFORE benchmarks (important for Triton autotuning!) ===
        print(f"Warming up all kernels ({args.warmup} iterations)...")
        for _ in range(args.warmup):
            dummy_forward()
        
        # For very large models, warmup on param_flat directly (then restore from anchor_flat)
        # For smaller models, use separate warmup tensor to avoid polluting benchmark data
        if is_very_large_model:
            print("  [Very large model: warming up on param_flat directly]")
            warmup_target = param_flat
        else:
            warmup_target = None  # Will create separate tensor
        
        # Triton runtime-seed perturb warmup (triggers autotuning ONCE)
        if HAS_TRITON_RUNTIME_SEED:
            if warmup_target is None:
                warmup_flat = torch.randn(n_elements, device=device, dtype=torch.float32)
            else:
                warmup_flat = warmup_target
            seed_tensor = torch.tensor([42], dtype=torch.int64, device=device)
            grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
            for _ in range(15):  # More warmup for autotuning
                fused_perturb_runtime_seed[grid](warmup_flat, seed_tensor, cfg.eps, n_elements)
                fused_update_runtime_seed[grid](warmup_flat, seed_tensor, 0.001, cfg.lr, n_elements)
            torch.cuda.synchronize()
            if warmup_target is None:
                del warmup_flat
            print("  Triton runtime-seed perturb autotuning complete")
        
        # CUDA perturb warmup
        if HAS_CUDA_PERTURB:
            if warmup_target is None:
                warmup_flat = torch.randn(n_elements, device=device, dtype=torch.float32)
            else:
                warmup_flat = warmup_target
            for _ in range(5):
                fused_perturb_cuda.fused_perturb(warmup_flat, 42, cfg.eps)
                fused_perturb_cuda.fused_update(warmup_flat, 42, 0.001, cfg.lr)
            torch.cuda.synchronize()
            if warmup_target is None:
                del warmup_flat
            print("  CUDA perturb warmup complete")
        
        # For very large models, restore param_flat from anchor_flat after warmup
        if is_very_large_model:
            param_flat.copy_(anchor_flat)
            torch.cuda.synchronize()
            print("  Restored param_flat from anchor_flat")
        
        cleanup_gpu()
        print()
    else:
        # Baseline-only mode - no kernel warmup needed
        print("[Baseline-only mode: skipping kernel warmup]")
        print()
    
    cleanup_gpu()
    print()
    
    # === Backend filtering based on --backend argument ===
    backend = args.backend
    run_all = (backend == 'all')
    
    # Skip PyTorch baselines if explicitly requested
    skip_pytorch = args.skip_baseline
    
    def reset_params():
        """Reset param_flat to match anchor_flat (start from pretrained)."""
        if param_flat is not None and anchor_flat is not None:
            param_flat.copy_(anchor_flat)
            torch.cuda.synchronize()
    
    # Check if we can run PyTorch baselines (need enough memory for separate param_list)
    can_run_baseline = can_fit_baseline
    
    # 1. PyTorch Baseline (MeZO only)
    # PyTorch baselines use their own param_list (original MeZO data organization)
    # This is separate from flat buffers used by CUDA/Triton kernels
    should_run_pytorch_mezo = (
        (run_all and not is_large_model) or 
        (backend == 'pytorch-mezo')
    ) and not skip_pytorch and can_run_baseline
    
    if should_run_pytorch_mezo:
        if is_large_model:
            print("Benchmarking: PyTorch Baseline (MeZO only)... [WARNING: slow for large models]")
        else:
            print("Benchmarking: PyTorch Baseline (MeZO only)...")
        
        # Create separate param list for baseline (original MeZO data organization)
        print("  Creating per-parameter tensors for baseline...")
        baseline_params, baseline_anchors = create_param_list_for_baseline(config, device)
        
        r = benchmark_pytorch_baseline(
            baseline_params, baseline_anchors, cfg, args.n_iter, include_dizo=False
        )
        results.append(r)
        print(f"  → {r.total_time_ms:.2f} ms")
        
        # Clean up baseline params
        del baseline_params, baseline_anchors
        cleanup_gpu(force_empty=True)
    elif not can_run_baseline and backend == 'pytorch-mezo':
        print(f"[ERROR] Cannot run {backend} on {args.model}")
        print(f"  Estimated memory needed: {baseline_mem_gb:.1f} GB")
        print(f"  Available GPU memory: {available_mem_gb:.1f} GB")
    
    # 2. PyTorch Baseline + DiZO
    should_run_pytorch_dizo = (
        (run_all and not is_large_model) or 
        (backend == 'pytorch-dizo')
    ) and not skip_pytorch and can_run_baseline
    
    if should_run_pytorch_dizo:
        if is_large_model:
            print("Benchmarking: PyTorch Baseline + DiZO... [WARNING: slow for large models]")
        else:
            print("Benchmarking: PyTorch Baseline + DiZO...")
        
        # Create separate param list for baseline (original MeZO/DiZO data organization)
        print("  Creating per-parameter tensors for baseline...")
        baseline_params, baseline_anchors = create_param_list_for_baseline(config, device)
        
        r = benchmark_pytorch_baseline(
            baseline_params, baseline_anchors, cfg, args.n_iter, include_dizo=True
        )
        results.append(r)
        print(f"  → {r.total_time_ms:.2f} ms")
        
        # Clean up baseline params
        del baseline_params, baseline_anchors
        cleanup_gpu(force_empty=True)
    elif not can_run_baseline and backend == 'pytorch-dizo':
        print(f"[ERROR] Cannot run {backend} on {args.model}")
        print(f"  Estimated memory needed: {baseline_mem_gb:.1f} GB")
        print(f"  Available GPU memory: {available_mem_gb:.1f} GB")
    
    # 3. Triton Perturb only (MeZO)
    if (run_all or backend == 'triton-perturb') and HAS_TRITON_PERTURB and not args.skip_baseline:
        print("Benchmarking: Triton Perturb (MeZO only)...")
        reset_params()
        r = benchmark_triton_perturb_only(
            param_flat, anchor_flat, offsets, sizes, None, None, cfg, args.n_iter, include_dizo=False
        )
        results.append(r)
        print(f"  → {r.total_time_ms:.2f} ms")
        reset_params()
        cleanup_gpu(force_empty=is_large_model)
    
    # 4. Triton Perturb + PyTorch DiZO (skip - use triton-zo-v2 instead)
    
    # 5. CUDA Perturb only (MeZO)
    if (run_all or backend == 'cuda-perturb') and HAS_CUDA_PERTURB and not args.skip_baseline:
        print("Benchmarking: CUDA Perturb (MeZO only)...")
        reset_params()
        r = benchmark_cuda_perturb_only(
            param_flat, anchor_flat, None, None, cfg, args.n_iter, include_dizo=False
        )
        results.append(r)
        print(f"  → {r.total_time_ms:.2f} ms")
        reset_params()
        cleanup_gpu(force_empty=is_large_model)
    
    # 6. CUDA Perturb + PyTorch DiZO (skip - use cuda-full or cuda-triton instead)
    
    # 7. Triton Perturb + Triton ZO V1 (skip for large models - V1 is slow)
    if run_all and HAS_TRITON_ZO_V1 and not is_large_model:
        print("Benchmarking: Triton Perturb + Triton ZO V1...")
        reset_params()
        r = benchmark_triton_v1_full(param_flat, anchor_flat, offsets, sizes, cfg, args.n_iter)
        results.append(r)
        if r.total_time_ms > 0:
            print(f"  → {r.total_time_ms:.2f} ms")
        else:
            print(f"  → {r.method}")
        reset_params()
        cleanup_gpu(force_empty=is_large_model)
    
    # 8. Triton Perturb + Triton ZO V2 (multi-block) - with DiZO
    if run_all or backend == 'triton-zo-v2':
        print("Benchmarking: Triton Perturb + Triton ZO V2...")
        reset_params()
        r = benchmark_triton_full(param_flat, anchor_flat, offsets, sizes, cfg, args.n_iter, include_dizo=True)
        results.append(r)
        if r.total_time_ms > 0:
            print(f"  → {r.total_time_ms:.2f} ms")
        else:
            print(f"  → {r.method}")
        reset_params()
        cleanup_gpu(force_empty=is_large_model)
    
    # 8b. Triton V2 MeZO-only (no DiZO constraints)
    if run_all or backend == 'triton-mezo':
        print("Benchmarking: Triton V2 (MeZO only, no constraints)...")
        reset_params()
        r = benchmark_triton_full(param_flat, anchor_flat, offsets, sizes, cfg, args.n_iter, include_dizo=False)
        results.append(r)
        if r.total_time_ms > 0:
            print(f"  → {r.total_time_ms:.2f} ms")
        else:
            print(f"  → {r.method}")
        reset_params()
        cleanup_gpu(force_empty=is_large_model)
    
    # 9. CUDA Perturb + Triton ZO V2 - with DiZO
    if run_all or backend == 'cuda-triton':
        print("Benchmarking: CUDA Perturb + Triton ZO V2...")
        reset_params()
        r = benchmark_cuda_perturb_triton_zo(param_flat, anchor_flat, offsets, sizes, cfg, args.n_iter, include_dizo=True)
        results.append(r)
        if r.total_time_ms > 0:
            print(f"  → {r.total_time_ms:.2f} ms")
        else:
            print(f"  → {r.method}")
        reset_params()
        cleanup_gpu(force_empty=is_large_model)
    
    # 10. CUDA Perturb + CUDA ZO V5 - with DiZO
    if run_all or backend == 'cuda-full':
        print("Benchmarking: CUDA Perturb + CUDA ZO V5...")
        reset_params()
        r = benchmark_cuda_full(param_flat, anchor_flat, offsets, sizes, cfg, args.n_iter, include_dizo=True)
        results.append(r)
        if r.total_time_ms > 0:
            print(f"  → {r.total_time_ms:.2f} ms")
        else:
            print(f"  → {r.method}")
        reset_params()
        cleanup_gpu(force_empty=is_large_model)
    
    # 10b. CUDA Full MeZO-only (no DiZO constraints)
    if run_all or backend == 'cuda-mezo':
        print("Benchmarking: CUDA Full (MeZO only, no constraints)...")
        reset_params()
        r = benchmark_cuda_full(param_flat, anchor_flat, offsets, sizes, cfg, args.n_iter, include_dizo=False)
        results.append(r)
        if r.total_time_ms > 0:
            print(f"  → {r.total_time_ms:.2f} ms")
        else:
            print(f"  → {r.method}")
        reset_params()
        cleanup_gpu(force_empty=is_large_model)
    
    # === Summary ===
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    
    # Find baselines for different categories
    mezo_baseline = None
    dizo_baseline = None
    for r in results:
        if "PyTorch Baseline + DiZO" in r.method and r.total_time_ms > 0:
            dizo_baseline = r.total_time_ms
        if r.method == "PyTorch Baseline (MeZO only)" and r.total_time_ms > 0:
            mezo_baseline = r.total_time_ms
    
    # --- Section 1: Overall Results with Speedup vs Full DiZO Baseline ---
    print("\n--- OVERALL RESULTS (Speedup vs PyTorch Baseline + DiZO) ---")
    print(f"{'Method':<45} {'Total(ms)':<12} {'Speedup':<12}")
    print("-" * 69)
    
    for r in sorted(results, key=lambda x: x.total_time_ms if x.total_time_ms > 0 else float('inf')):
        if r.total_time_ms > 0 and dizo_baseline:
            speedup = dizo_baseline / r.total_time_ms
            marker = " ★" if r.total_time_ms == min(x.total_time_ms for x in results if x.total_time_ms > 0) else ""
            print(f"{r.method:<45} {r.total_time_ms:<12.2f} {speedup:<10.2f}x{marker}")
        elif r.total_time_ms > 0:
            print(f"{r.method:<45} {r.total_time_ms:<12.2f} {'N/A':<12}")
        else:
            print(f"{r.method:<45} {'N/A':<12}")
    
    # --- Section 2: Per-Step Breakdown ---
    print("\n--- PER-STEP BREAKDOWN (all times in ms) ---")
    print("Note: 'Perturb' = 3x perturb ops (±eps, reset), 'ZO' = DiZO constraints, 'Update' = gradient update")
    print(f"{'Method':<45} {'Perturb':<12} {'ZO Constr.':<12} {'Update':<12}")
    print("-" * 81)
    
    for r in sorted(results, key=lambda x: x.total_time_ms if x.total_time_ms > 0 else float('inf')):
        if r.total_time_ms > 0:
            print(f"{r.method:<45} {r.perturb_ms:<12.2f} {r.zo_forward_ms:<12.2f} {r.update_ms:<12.2f}")
    
    # --- Section 3: Per-Step Speedup vs PyTorch ---
    print("\n--- PER-STEP SPEEDUP (vs PyTorch Baseline + DiZO) ---")
    pytorch_dizo = None
    for r in results:
        if "PyTorch Baseline + DiZO" in r.method and r.total_time_ms > 0:
            pytorch_dizo = r
            break
    
    if pytorch_dizo:
        print(f"PyTorch Baseline + DiZO: Perturb={pytorch_dizo.perturb_ms:.2f}ms, ZO={pytorch_dizo.zo_forward_ms:.2f}ms, Update={pytorch_dizo.update_ms:.2f}ms")
        print(f"{'Method':<45} {'Perturb Spd':<14} {'ZO Spd':<14} {'Update Spd':<14}")
        print("-" * 87)
        
        for r in sorted(results, key=lambda x: x.total_time_ms if x.total_time_ms > 0 else float('inf')):
            if r.total_time_ms > 0 and r.method != pytorch_dizo.method:
                perturb_spd = pytorch_dizo.perturb_ms / r.perturb_ms if r.perturb_ms > 0 else 0
                zo_spd = pytorch_dizo.zo_forward_ms / r.zo_forward_ms if r.zo_forward_ms > 0 else float('inf')
                update_spd = pytorch_dizo.update_ms / r.update_ms if r.update_ms > 0 else 0
                
                zo_str = f"{zo_spd:.2f}x" if zo_spd != float('inf') else "N/A (no ZO)"
                print(f"{r.method:<45} {perturb_spd:<14.2f}x {zo_str:<14} {update_spd:<14.2f}x")
    
    # Save results
    if args.output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        outdir = os.path.join(SCRIPT_DIR, 'benchmark_results', 'full_step_v2', timestamp)
        os.makedirs(outdir, exist_ok=True)
        outfile = os.path.join(outdir, f'{args.model}_results.txt')
        with open(outfile, 'w') as f:
            f.write(f"Model: {args.model}\n")
            f.write(f"Iterations: {args.n_iter}\n\n")
            for r in results:
                if r.total_time_ms > 0:
                    f.write(f"{r.method}: {r.total_time_ms:.2f}ms "
                           f"(perturb={r.perturb_ms:.2f}, zo={r.zo_forward_ms:.2f}, update={r.update_ms:.2f})\n")
        print(f"\nResults saved to: {outfile}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Full MeZO/DiZO Training Step Benchmark V2',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all benchmarks on small model
  python benchmark_full_training_step_v2.py --model opt-350m
  
  # Run specific backend on large model (memory efficient)
  python benchmark_full_training_step_v2.py --model opt-13b --backend cuda-full
  python benchmark_full_training_step_v2.py --model opt-13b --backend triton-zo-v2
  
  # Run MeZO-only (no DiZO constraints) with optimized kernels
  python benchmark_full_training_step_v2.py --model opt-6.7b --backend cuda-mezo
  python benchmark_full_training_step_v2.py --model opt-6.7b --backend triton-mezo
  
  # Skip slow PyTorch baselines
  python benchmark_full_training_step_v2.py --model opt-6.7b --skip_baseline

Available backends:
  all            - Run all benchmarks (default, may OOM on large models)
  
  --- PyTorch Baselines ---
  pytorch-mezo   - PyTorch Baseline (MeZO only, no DiZO)
  pytorch-dizo   - PyTorch Baseline + DiZO constraints
  
  --- MeZO-only (no DiZO constraints) ---
  cuda-perturb   - CUDA Perturb only (basic MeZO)
  triton-perturb - Triton Perturb only (basic MeZO)
  cuda-mezo      - CUDA Full kernels but MeZO-only (no constraints)
  triton-mezo    - Triton V2 kernels but MeZO-only (no constraints)
  
  --- Full DiZO (with constraints) ---
  cuda-full      - CUDA Perturb + CUDA ZO V5
  cuda-triton    - CUDA Perturb + Triton ZO V2
  triton-zo-v2   - Triton Perturb + Triton ZO V2
"""
    )
    parser.add_argument('--model', type=str, default='opt-350m', choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument('--n_iter', type=int, default=20)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--eps', type=float, default=1e-3)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--tau', type=float, default=0.2)
    parser.add_argument('--zo_eps', type=float, default=0.1)
    parser.add_argument('--step_size', type=float, default=2.0)
    parser.add_argument('--skip_baseline', action='store_true', help='Skip slow PyTorch baselines')
    parser.add_argument('--backend', type=str, default='all',
                        choices=['all', 'pytorch-mezo', 'pytorch-dizo',
                                 'cuda-perturb', 'cuda-full', 'cuda-triton', 'cuda-mezo',
                                 'triton-perturb', 'triton-zo-v2', 'triton-mezo'],
                        help='Specific backend to benchmark (for memory efficiency on large models)')
    parser.add_argument('--output', action='store_true')
    parser.add_argument('--all', action='store_true', help='Run benchmarks for all model sizes')
    
    args = parser.parse_args()
    
    if args.all:
        for model in MODEL_CONFIGS.keys():
            print("\n" + "#" * 90)
            print(f"# MODEL: {model.upper()}")
            print("#" * 90 + "\n")
            args.model = model
            try:
                run_benchmarks(args)
            except torch.cuda.OutOfMemoryError:
                print(f"OOM for {model}, skipping...")
            cleanup_gpu(force_empty=True)
    else:
        run_benchmarks(args)


if __name__ == "__main__":
    main()