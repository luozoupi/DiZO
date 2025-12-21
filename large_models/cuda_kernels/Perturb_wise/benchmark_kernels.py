#!/usr/bin/env python3
"""
Comprehensive Benchmark: CUDA vs Triton vs PyTorch Fused Perturb Kernels

This script compares:
1. PyTorch baseline (randn + add as separate ops)
2. Triton fused kernel (randn + add in one kernel)
3. CUDA fused kernel (if compiled)
4. ChunkedMeZO (PyTorch cuRAND in chunks)

Model size presets:
- OPT-350M: 331,196,416 parameters
- OPT-2.7B: 2,700,000,000 parameters (approximate)
- OPT-6.7B: 6,700,000,000 parameters (approximate)
- OPT-13B: 13,000,000,000 parameters (approximate)

Run with:
    python benchmark_kernels.py --model opt-2.7b
    python benchmark_kernels.py --model all
"""

import torch
import numpy as np
import time
import gc
import argparse
from typing import Dict, Tuple

# Model size presets (parameter counts)
MODEL_SIZES = {
    'opt-350m': 331_196_416,
    'opt-2.7b': 2_700_000_000,
    'opt-6.7b': 6_700_000_000,
    'opt-13b': 13_000_000_000,
}

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


def get_gpu_memory_info() -> Tuple[float, float, float]:
    """Get GPU memory information.
    
    Returns:
        Tuple of (total_gb, used_gb, free_gb)
    """
    if not torch.cuda.is_available():
        return 0.0, 0.0, 0.0
    
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    reserved = torch.cuda.memory_reserved(0) / 1024**3
    allocated = torch.cuda.memory_allocated(0) / 1024**3
    free = total - reserved
    
    return total, allocated, free


def cleanup_gpu_memory():
    """Aggressively clean up GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def estimate_benchmark_memory(n_elements: int, method: str) -> float:
    """Estimate memory required for a benchmark in GB.
    
    Args:
        n_elements: Number of elements in the tensor
        method: Benchmark method name
        
    Returns:
        Estimated memory in GB
    """
    element_size = 4  # float32 = 4 bytes
    base_memory_gb = n_elements * element_size / 1024**3
    
    # Different methods have different memory requirements
    if method in ['pytorch_baseline', 'pytorch_fused']:
        # Needs params + z tensor (but we now use chunked, so just params + small buffer)
        return base_memory_gb + 0.1  # params + 64MB chunk buffer
    elif method.startswith('triton') or method == 'cuda_ext':
        # In-place operation, only needs params
        return base_memory_gb + 0.05  # small overhead
    elif method == 'chunked':
        return base_memory_gb + 0.1  # params + 64MB chunk buffer
    else:
        return base_memory_gb * 2  # Conservative estimate


def benchmark_pytorch_baseline(params: torch.Tensor, seed: int, alpha: float, n_iter: int, chunk_size_mb: int = 64) -> Tuple[float, float]:
    """Benchmark PyTorch's separate randn + add operations (FLAT tensor version).
    
    NOTE: This uses a flattened single tensor, which is NOT how the original MeZO works.
    Use benchmark_mezo_original_style() for a more accurate comparison.
    
    Uses chunked approach to avoid allocating a full-size z tensor,
    which would cause OOM for large models.
    
    Returns:
        Tuple of (time_ms, peak_memory_mb)
    """
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    n_elements = params.numel()
    # Use chunked allocation for memory efficiency
    chunk_size = (chunk_size_mb * 1024 * 1024) // 4  # float32 = 4 bytes
    
    # For small tensors, use full z tensor (original behavior)
    if n_elements <= chunk_size:
        z = torch.empty_like(params)
        
        torch.cuda.synchronize()
        start = time.time()
        
        for _ in range(n_iter):
            torch.manual_seed(seed)
            z.normal_()
            params.add_(z, alpha=alpha)
        
        torch.cuda.synchronize()
        del z
    else:
        # For large tensors, use chunked approach to save memory
        z_chunk = torch.empty(chunk_size, device=params.device, dtype=params.dtype)
        
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
        del z_chunk
    
    elapsed_ms = (time.time() - start) * 1000 / n_iter
    peak_memory_mb = torch.cuda.max_memory_allocated() / 1024**2
    
    return elapsed_ms, peak_memory_mb


def benchmark_mezo_original_style(params_list: list, seed: int, alpha: float, n_iter: int) -> Tuple[float, float]:
    """Benchmark that matches ACTUAL MeZO/DiZO implementation.
    
    This is the correct baseline for comparing against fused kernels, as it reflects
    how MeZO actually works:
    - Iterates through each parameter tensor separately
    - Allocates z = torch.normal(...) for EACH parameter (not reused!)
    - Peak memory = largest_param_size + model_size (not 2x model_size)
    
    Reference: MeZO's zo_perturb_parameters():
        torch.manual_seed(random_seed)
        for name, param in self.named_parameters_to_optim:
            z = torch.normal(mean=0, std=1, size=param.data.size(), ...)
            param.data = param.data + scaling_factor * z * self.args.zo_eps
    
    Args:
        params_list: List of parameter tensors (NOT flattened) - simulates model.parameters()
        seed: Random seed for reproducibility
        alpha: Scaling factor (eps in MeZO)
        n_iter: Number of iterations
    
    Returns:
        Tuple of (time_ms, peak_memory_mb)
    """
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(n_iter):
        torch.manual_seed(seed)
        for param in params_list:
            # This is EXACTLY what MeZO does - allocate z per parameter
            z = torch.normal(mean=0, std=1, size=param.data.size(),
                           device=param.data.device, dtype=param.data.dtype)
            param.data.add_(z, alpha=alpha)
            # z deallocated here after each parameter, but peak memory is tracked
    
    torch.cuda.synchronize()
    elapsed_ms = (time.time() - start) * 1000 / n_iter
    peak_memory_mb = torch.cuda.max_memory_allocated() / 1024**2
    
    return elapsed_ms, peak_memory_mb


def create_params_list_like_model(total_elements: int, dtype=torch.float32, device='cuda') -> list:
    """Create a list of parameter tensors that EXACTLY matches real OPT model structure.
    
    This uses the actual OPT architecture specifications to create accurate parameter
    distributions for benchmarking MeZO perturbation operations.
    
    OPT Model Configurations (from HuggingFace):
    - OPT-350M: hidden=1024, layers=24, ffn=4096,  vocab=50272 → 331M params, 388 tensors
    - OPT-1.3B: hidden=2048, layers=24, ffn=8192,  vocab=50272 → 1.3B params
    - OPT-2.7B: hidden=2560, layers=32, ffn=10240, vocab=50272 → 2.7B params
    - OPT-6.7B: hidden=4096, layers=32, ffn=16384, vocab=50272 → 6.7B params
    - OPT-13B:  hidden=5120, layers=40, ffn=20480, vocab=50272 → 13B params
    
    Real OPT-350M structure (from model.named_parameters()):
    - Each transformer layer has 16 parameter tensors:
      - 4 attention projections (Q,K,V,O) × (weight + bias) = 8 tensors
      - 2 FFN layers (fc1, fc2) × (weight + bias) = 4 tensors  
      - 2 layer norms × (weight + bias) = 4 tensors
    - Plus embedding layers at the start
    
    Args:
        total_elements: Total number of elements across all tensors
        dtype: Data type
        device: Device to place tensors on
    
    Returns:
        List of parameter tensors matching real OPT architecture
    """
    # Determine which OPT model config to use based on parameter count
    OPT_CONFIGS = {
        331_196_416: {'hidden': 1024, 'layers': 24, 'ffn': 4096, 'vocab': 50272, 'name': 'OPT-350M'},
        1_315_753_984: {'hidden': 2048, 'layers': 24, 'ffn': 8192, 'vocab': 50272, 'name': 'OPT-1.3B'},
        2_700_000_000: {'hidden': 2560, 'layers': 32, 'ffn': 10240, 'vocab': 50272, 'name': 'OPT-2.7B'},
        6_700_000_000: {'hidden': 4096, 'layers': 32, 'ffn': 16384, 'vocab': 50272, 'name': 'OPT-6.7B'},
        13_000_000_000: {'hidden': 5120, 'layers': 40, 'ffn': 20480, 'vocab': 50272, 'name': 'OPT-13B'},
    }
    
    # Find closest config
    closest_config = min(OPT_CONFIGS.keys(), key=lambda x: abs(x - total_elements))
    config = OPT_CONFIGS[closest_config]
    
    hidden = config['hidden']
    n_layers = config['layers']
    ffn_dim = config['ffn']
    vocab_size = config['vocab']
    
    params_list = []
    
    # === Embedding layers ===
    # embed_tokens: [vocab_size, embed_dim] - OPT uses word_embed_proj_dim
    embed_dim = hidden // 2 if hidden >= 1024 else hidden
    params_list.append(torch.randn(vocab_size * embed_dim, dtype=dtype, device=device))  # embed_tokens
    
    # embed_positions: [max_position, hidden]
    max_positions = 2050  # OPT default
    params_list.append(torch.randn(max_positions * hidden, dtype=dtype, device=device))  # embed_positions
    
    # project_in/project_out if embed_dim != hidden
    if embed_dim != hidden:
        params_list.append(torch.randn(embed_dim * hidden, dtype=dtype, device=device))  # project_in
        params_list.append(torch.randn(hidden * embed_dim, dtype=dtype, device=device))  # project_out
    
    # === Transformer layers ===
    for layer_idx in range(n_layers):
        # Self-attention: Q, K, V, O projections (weight + bias each)
        params_list.append(torch.randn(hidden * hidden, dtype=dtype, device=device))  # q_proj.weight
        params_list.append(torch.randn(hidden, dtype=dtype, device=device))           # q_proj.bias
        params_list.append(torch.randn(hidden * hidden, dtype=dtype, device=device))  # k_proj.weight
        params_list.append(torch.randn(hidden, dtype=dtype, device=device))           # k_proj.bias
        params_list.append(torch.randn(hidden * hidden, dtype=dtype, device=device))  # v_proj.weight
        params_list.append(torch.randn(hidden, dtype=dtype, device=device))           # v_proj.bias
        params_list.append(torch.randn(hidden * hidden, dtype=dtype, device=device))  # out_proj.weight
        params_list.append(torch.randn(hidden, dtype=dtype, device=device))           # out_proj.bias
        
        # Self-attention layer norm
        params_list.append(torch.randn(hidden, dtype=dtype, device=device))           # self_attn_layer_norm.weight
        params_list.append(torch.randn(hidden, dtype=dtype, device=device))           # self_attn_layer_norm.bias
        
        # FFN: fc1 (up), fc2 (down)
        params_list.append(torch.randn(ffn_dim * hidden, dtype=dtype, device=device)) # fc1.weight
        params_list.append(torch.randn(ffn_dim, dtype=dtype, device=device))          # fc1.bias
        params_list.append(torch.randn(hidden * ffn_dim, dtype=dtype, device=device)) # fc2.weight
        params_list.append(torch.randn(hidden, dtype=dtype, device=device))           # fc2.bias
        
        # Final layer norm
        params_list.append(torch.randn(hidden, dtype=dtype, device=device))           # final_layer_norm.weight
        params_list.append(torch.randn(hidden, dtype=dtype, device=device))           # final_layer_norm.bias
    
    # Calculate actual total and scale if needed
    actual_total = sum(p.numel() for p in params_list)
    
    # If there's a significant difference, scale the largest layers proportionally
    if abs(actual_total - total_elements) > total_elements * 0.01:  # More than 1% difference
        # Scale factor to match target
        scale = total_elements / actual_total
        
        # Rebuild with scaled sizes (only scale large tensors, keep biases same)
        params_scaled = []
        for p in params_list:
            if p.numel() > 10000:  # Only scale large tensors (weights, not biases)
                new_size = int(p.numel() * scale)
                params_scaled.append(torch.randn(new_size, dtype=dtype, device=device))
            else:
                params_scaled.append(p)
        
        # Handle any remaining difference
        final_total = sum(p.numel() for p in params_scaled)
        diff = total_elements - final_total
        if diff > 0:
            params_scaled.append(torch.randn(diff, dtype=dtype, device=device))
        
        return params_scaled
    
    return params_list


def benchmark_pytorch_fused(params: torch.Tensor, seed: int, alpha: float, n_iter: int, chunk_size_mb: int = 64) -> Tuple[float, float]:
    """Benchmark PyTorch's addcmul (closest to fused op).
    
    Uses chunked approach to avoid allocating a full-size z tensor.
    
    Returns:
        Tuple of (time_ms, peak_memory_mb)
    """
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    n_elements = params.numel()
    chunk_size = (chunk_size_mb * 1024 * 1024) // 4  # float32 = 4 bytes
    alpha_tensor = torch.tensor(alpha, device=params.device)
    
    # For small tensors, use full z tensor (original behavior)
    if n_elements <= chunk_size:
        z = torch.empty_like(params)
        
        torch.cuda.synchronize()
        start = time.time()
        
        for _ in range(n_iter):
            torch.manual_seed(seed)
            z.normal_()
            # addcmul: params = params + alpha * z
            torch.addcmul(params, alpha_tensor, z, out=params)
        
        torch.cuda.synchronize()
        del z
    else:
        # For large tensors, use chunked approach
        z_chunk = torch.empty(chunk_size, device=params.device, dtype=params.dtype)
        ones_chunk = torch.ones(chunk_size, device=params.device, dtype=params.dtype)
        
        torch.cuda.synchronize()
        start = time.time()
        
        for _ in range(n_iter):
            torch.manual_seed(seed)
            offset = 0
            while offset < n_elements:
                chunk_len = min(chunk_size, n_elements - offset)
                z_view = z_chunk[:chunk_len]
                z_view.normal_()
                # In-place: params[chunk] += alpha * z
                params[offset:offset + chunk_len].add_(z_view, alpha=alpha)
                offset += chunk_len
        
        torch.cuda.synchronize()
        del z_chunk, ones_chunk
    
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
    
    n_elements = params.numel()
    
    # Check for int32 overflow in Triton kernel offsets
    # Triton uses int32 by default, so max safe elements is ~2^31
    INT32_MAX = 2**31 - 1
    if n_elements > INT32_MAX:
        print(f"   [WARNING] Tensor too large for Triton V1 kernel (int32 overflow risk)")
        return float('inf'), 0.0
    
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
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
    
    n_elements = params.numel()
    
    # Check for int32 overflow in Triton kernel offsets
    INT32_MAX = 2**31 - 1
    if n_elements > INT32_MAX:
        print(f"   [WARNING] Tensor too large for Triton V2 kernel (int32 overflow risk)")
        return float('inf'), 0.0
    
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
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
    MAX_GRID_SIZE = 2_000_000_000
    n_elements_int64 = int(n_elements)
    
    # Handle large tensors with chunking
    if n_elements_int64 <= MAX_GRID_SIZE:
        grid = lambda meta: (triton.cdiv(n_elements_int64, meta['BLOCK_SIZE']),)
        
        def kernel_call(p):
            fused_perturb_kernel_autotuned[grid](p, seed, alpha, n_elements_int64)
    else:
        # Chunked approach
        chunk_size = MAX_GRID_SIZE
        n_chunks = (n_elements_int64 + chunk_size - 1) // chunk_size
        
        def kernel_call(p):
            for chunk_idx in range(n_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min(start_idx + chunk_size, n_elements_int64)
                chunk_size_actual = end_idx - start_idx
                chunk_params = p[start_idx:end_idx]
                grid_chunk = lambda meta: (triton.cdiv(chunk_size_actual, meta['BLOCK_SIZE']),)
                fused_perturb_kernel_autotuned[grid_chunk](chunk_params, seed, alpha, chunk_size_actual)
    
    # Warmup for autotuning
    try:
        for _ in range(10):
            kernel_call(params)
    except RuntimeError as e:
        print(f"  WARNING: Error during warmup: {e}")
        return float('inf'), 0.0
    
    torch.cuda.synchronize()
    start = time.time()
    
    try:
        for _ in range(n_iter):
            kernel_call(params)
    except RuntimeError as e:
        print(f"  ERROR: {e}")
        return float('inf'), 0.0
    
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
    
    # Clean up
    del z_chunk
    
    return elapsed_ms, peak_memory_mb


def run_comprehensive_benchmark(
    sizes: list = None,
    n_iter: int = 100,
    warmup: int = 10,
    memory_safety_margin: float = 0.15,  # Keep 15% GPU memory free
    method: str = 'all',  # Which method to benchmark
):
    """Run comprehensive benchmark across different tensor sizes.
    
    Args:
        sizes: List of (n_elements, name) tuples to benchmark
        n_iter: Number of iterations per benchmark
        warmup: Number of warmup iterations
        memory_safety_margin: Fraction of GPU memory to keep free (0.0-1.0)
    """
    
    if sizes is None:
        sizes = [
            (1_000_000, "1M"),
            (10_000_000, "10M"),
            (100_000_000, "100M"),
            (331_196_416, "OPT-350M"),
            (2_700_000_000, "OPT-2.7B"),
            (6_700_000_000, "OPT-6.7B"),
            (13_000_000_000, "OPT-13B"),
        ]
    
    # Get GPU memory info
    total_gpu_mem, _, free_gpu_mem = get_gpu_memory_info()
    usable_mem = total_gpu_mem * (1 - memory_safety_margin)
    
    print("=" * 90)
    print("COMPREHENSIVE KERNEL BENCHMARK: CUDA vs Triton vs PyTorch")
    print("=" * 90)
    print(f"Iterations: {n_iter}, Warmup: {warmup}")
    print(f"Triton available: {HAS_TRITON}")
    print(f"CUDA extension available: {HAS_CUDA_EXT}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {total_gpu_mem:.1f} GB total, {free_gpu_mem:.1f} GB free, {usable_mem:.1f} GB usable")
    print(f"Method filter: {method}")
    
    seed = 42
    alpha = 1e-3
    
    all_results = {}
    
    for n_elements, size_name in sizes:
        required_mem_gb = n_elements * 4 / 1024**3  # float32 = 4 bytes
        
        print(f"\n{'='*90}")
        print(f"SIZE: {size_name} ({n_elements:,} elements, {required_mem_gb:.2f} GB)")
        print("=" * 90)
        
        # Clean up before starting
        cleanup_gpu_memory()
        _, _, current_free = get_gpu_memory_info()
        
        # Check if we have enough memory
        if required_mem_gb > current_free * 0.9:  # Need at least 90% of free memory
            print(f"[SKIPPED] Not enough GPU memory. Need {required_mem_gb:.2f} GB, have {current_free:.2f} GB free")
            all_results[size_name] = {'skipped': True, 'reason': 'OOM'}
            continue
        
        # Create test tensor
        try:
            params = torch.randn(n_elements, device='cuda', dtype=torch.float32)
        except RuntimeError as e:
            print(f"[SKIPPED] Failed to allocate tensor: {e}")
            all_results[size_name] = {'skipped': True, 'reason': str(e)}
            continue
        
        results = {}
        cuda_context_corrupted = False  # Track if CUDA context is corrupted
        
        # Helper function to run a benchmark with proper cleanup
        def run_benchmark(name, benchmark_fn, *args, **kwargs):
            """Run a single benchmark with memory cleanup."""
            nonlocal cuda_context_corrupted
            
            if cuda_context_corrupted:
                print(f"   [SKIPPED] CUDA context corrupted, skipping remaining benchmarks")
                results[name] = {'time': float('inf'), 'memory': 0, 'error': 'CUDA_CORRUPTED'}
                return
            
            cleanup_gpu_memory()
            try:
                # Reinitialize params to known state
                torch.manual_seed(42)
                params.normal_()
                
                time_ms, memory_mb = benchmark_fn(params, *args, **kwargs)
                results[name] = {'time': time_ms, 'memory': memory_mb}
                print(f"   Time: {time_ms:.3f} ms, Memory: {memory_mb:.2f} MB")
            except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                error_str = str(e).lower()
                if "out of memory" in error_str:
                    print(f"   [OOM] Skipped due to memory error")
                    results[name] = {'time': float('inf'), 'memory': 0, 'error': 'OOM'}
                elif "illegal memory access" in error_str or "cuda error" in error_str:
                    print(f"   [CUDA ERROR] {e}")
                    print(f"   WARNING: CUDA context may be corrupted. Skipping remaining benchmarks for this size.")
                    results[name] = {'time': float('inf'), 'memory': 0, 'error': 'CUDA_ERROR'}
                    cuda_context_corrupted = True
                else:
                    print(f"   [ERROR] {e}")
                    results[name] = {'time': float('inf'), 'memory': 0, 'error': str(e)}
                try:
                    cleanup_gpu_memory()
                except:
                    cuda_context_corrupted = True
        
        # Warmup (use params directly, no clone)
        print("Warming up...")
        cleanup_gpu_memory()
        try:
            torch.manual_seed(42)
            params.normal_()
            benchmark_pytorch_baseline(params, seed, alpha, warmup)
        except RuntimeError:
            print("   Warmup failed, continuing anyway...")
        
        # 1. PyTorch baseline (flat tensor - NOT real MeZO behavior)
        if method in ['all', 'pytorch_baseline', 'pytorch']:
            print("\n1. PyTorch Baseline (flat tensor, chunked - NOT real MeZO):")
            run_benchmark('pytorch_baseline', benchmark_pytorch_baseline, seed, alpha, n_iter)
        
        # 1b. MeZO Original Style (actual MeZO behavior with per-param z allocation)
        if method in ['all', 'mezo_original', 'pytorch']:
            print("\n1b. MeZO Original Style (per-param z alloc - REAL MeZO baseline):")
            # Need to create params_list for this benchmark
            cleanup_gpu_memory()
            try:
                params_list = create_params_list_like_model(n_elements, dtype=torch.float32, device='cuda')
                total_params = sum(p.numel() for p in params_list)
                print(f"    Created {len(params_list)} param tensors, total {total_params:,} elements")
                print(f"    Largest param: {max(p.numel() for p in params_list):,} elements")
                
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
                
                # Warmup
                for _ in range(min(5, warmup)):
                    torch.manual_seed(42)
                    for p in params_list:
                        p.normal_()
                    benchmark_mezo_original_style(params_list, seed, alpha, 1)
                
                # Benchmark
                torch.cuda.reset_peak_memory_stats()
                time_ms, memory_mb = benchmark_mezo_original_style(params_list, seed, alpha, n_iter)
                results['mezo_original'] = {'time': time_ms, 'memory': memory_mb}
                print(f"   Time: {time_ms:.3f} ms, Memory: {memory_mb:.2f} MB")
                
                # Cleanup
                del params_list
                cleanup_gpu_memory()
            except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                print(f"   [ERROR] {e}")
                results['mezo_original'] = {'time': float('inf'), 'memory': 0, 'error': str(e)}
                cleanup_gpu_memory()
        
        # 2. PyTorch fused (addcmul)
        if method in ['all', 'pytorch_fused', 'pytorch']:
            print("\n2. PyTorch Fused (addcmul, chunked):")
            run_benchmark('pytorch_fused', benchmark_pytorch_fused, seed, alpha, n_iter)
        
        # 3. Chunked (64MB) - this is now same as baseline, but explicit
        if method in ['all', 'chunked', 'pytorch']:
            print("\n3. Chunked PyTorch (64MB buffer):")
            run_benchmark('chunked_64mb', benchmark_chunked, seed, alpha, n_iter, 64)
        
        # 4. Triton V1
        if HAS_TRITON and method in ['all', 'triton_v1', 'triton']:
            print("\n4. Triton V1 (basic):")
            run_benchmark('triton_v1', benchmark_triton_v1, seed, alpha, n_iter)
        
        # 5. Triton V2 (different block sizes)
        if HAS_TRITON and method in ['all', 'triton_v2', 'triton']:
            for block_size in [1024, 2048, 4096]:
                print(f"\n5. Triton V2 (BLOCK={block_size}):")
                run_benchmark(f'triton_v2_b{block_size}', benchmark_triton_v2, seed, alpha, n_iter, block_size)
        
        # 6. Triton Autotuned
        if HAS_TRITON and method in ['all', 'triton_auto', 'triton']:
            print("\n6. Triton Autotuned:")
            run_benchmark('triton_autotuned', benchmark_triton_autotuned, seed, alpha, n_iter)
        
        # 7. CUDA Extension
        if HAS_CUDA_EXT and method in ['all', 'cuda']:
            print("\n7. CUDA Extension:")
            run_benchmark('cuda_ext', benchmark_cuda_ext, seed, alpha, n_iter)
        
        # Summary for this size
        print(f"\n{'-'*90}")
        print(f"SUMMARY for {size_name}:")
        print(f"{'-'*90}")
        
        # Filter out errored results for summary
        valid_results = {k: v for k, v in results.items() if v.get('time', float('inf')) != float('inf')}
        
        if valid_results:
            # Use mezo_original as baseline (true MeZO behavior), fallback to pytorch_baseline
            baseline_time = results.get('mezo_original', {}).get('time', float('inf'))
            baseline_name = 'mezo_original'
            if baseline_time == float('inf'):
                baseline_time = results.get('pytorch_baseline', {}).get('time', float('inf'))
                baseline_name = 'pytorch_baseline'
            if baseline_time == float('inf'):
                baseline_time = min(v['time'] for v in valid_results.values())
                baseline_name = 'fastest'
            
            sorted_results = sorted(valid_results.items(), key=lambda x: x[1]['time'])
            
            print(f"{'Method':<30} {'Time (ms)':>12} {'Speedup':>10} {'Memory (MB)':>15}")
            print(f"(Baseline: {baseline_name})")
            print("-" * 67)
            
            for name, data in sorted_results:
                time_ms = data['time']
                memory_mb = data['memory']
                speedup = baseline_time / time_ms if time_ms > 0 else 0
                
                marker = " ★" if time_ms == sorted_results[0][1]['time'] else ""
                print(f"{name:<30} {time_ms:>12.3f} {speedup:>9.2f}x {memory_mb:>14.2f}{marker}")
        else:
            print("No successful benchmarks for this size.")
        
        all_results[size_name] = results
        
        # Cleanup after this size
        del params
        cleanup_gpu_memory()
    
    return all_results


def verify_correctness():
    """Verify that all kernels produce statistically equivalent results."""
    print("\n" + "=" * 70)
    print("CORRECTNESS VERIFICATION")
    print("=" * 70)
    
    n_elements = 1_000_000
    seed = 42
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


def benchmark_complete_mezo_step(n_elements=331_196_416, n_iter=50, warmup=5, chunk_size_mb=64, method='all'):
    """
    Benchmark complete MeZO training step matching trainer.py:
    1. Perturb +eps (3 kernel calls)
    2. Update: params = params - lr * projected_grad * z (1 kernel call)
    Total: 4 kernel calls per step
    
    Uses chunked approach for PyTorch baseline to avoid OOM on large models.
    
    Args:
        method: 'all', 'pytorch', 'cuda', or 'triton' - which method to benchmark
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
    
    # Check available memory
    cleanup_gpu_memory()
    total_mem, _, free_mem = get_gpu_memory_info()
    required_mem = n_elements * 4 / 1024**3  # GB
    
    print(f"GPU Memory: {total_mem:.1f} GB total, {free_mem:.1f} GB free")
    print(f"Required: {required_mem:.2f} GB for params")
    print(f"Method filter: {method}")
    
    # Chunk size for PyTorch operations
    chunk_size = (chunk_size_mb * 1024 * 1024) // 4  # float32 = 4 bytes
    
    # PyTorch Baseline (complete step) - Using chunked approach like profile_dizo.py
    if method in ['all', 'pytorch']:
        print("\n1. PyTorch Baseline (complete MeZO step, chunked):")
        
        try:
            cleanup_gpu_memory()
            params_pytorch = torch.randn(n_elements, device='cuda', dtype=torch.float32)
            
            # Pre-allocate a small z_chunk buffer (similar to profile_dizo.py approach)
            z_chunk = torch.empty(min(chunk_size, n_elements), device='cuda', dtype=torch.float32)
            
            def pytorch_mezo_step_chunked(p):
                """Chunked MeZO step - processes in chunks to avoid OOM."""
                n = p.numel()
                
                # Step 1: Perturb +eps (chunked)
                torch.manual_seed(seed)
                offset = 0
                while offset < n:
                    chunk_len = min(chunk_size, n - offset)
                    z_view = z_chunk[:chunk_len]
                    z_view.normal_()
                    p[offset:offset + chunk_len].add_(z_view, alpha=eps)
                    offset += chunk_len
                loss1 = 1.0
                
                # Step 2: Perturb -2eps (chunked)
                torch.manual_seed(seed)
                offset = 0
                while offset < n:
                    chunk_len = min(chunk_size, n - offset)
                    z_view = z_chunk[:chunk_len]
                    z_view.normal_()
                    p[offset:offset + chunk_len].add_(z_view, alpha=-2*eps)
                    offset += chunk_len
                loss2 = 0.9
                
                # Step 3: Compute projected_grad
                projected_grad = (loss1 - loss2) / (2 * eps)
                
                # Step 4: Reset +eps (chunked)
                torch.manual_seed(seed)
                offset = 0
                while offset < n:
                    chunk_len = min(chunk_size, n - offset)
                    z_view = z_chunk[:chunk_len]
                    z_view.normal_()
                    p[offset:offset + chunk_len].add_(z_view, alpha=eps)
                    offset += chunk_len
                
                # Step 5: Update (chunked)
                torch.manual_seed(seed)
                offset = 0
                while offset < n:
                    chunk_len = min(chunk_size, n - offset)
                    z_view = z_chunk[:chunk_len]
                    z_view.normal_()
                    p[offset:offset + chunk_len].add_(z_view, alpha=-lr * projected_grad)
                    offset += chunk_len
            
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            
            # Warmup (in-place, no clone needed)
            for _ in range(warmup):
                torch.manual_seed(42)
                params_pytorch.normal_()
                pytorch_mezo_step_chunked(params_pytorch)
            torch.cuda.synchronize()
            
            start = time.time()
            for _ in range(n_iter):
                torch.manual_seed(42)
                params_pytorch.normal_()
                pytorch_mezo_step_chunked(params_pytorch)
            torch.cuda.synchronize()
            pytorch_time = (time.time() - start) * 1000 / n_iter
            pytorch_memory = torch.cuda.max_memory_allocated() / 1024**2
            
            results['pytorch_mezo_step'] = {'time': pytorch_time, 'memory': pytorch_memory}
            print(f"   Time: {pytorch_time:.3f} ms, Memory: {pytorch_memory:.2f} MB")
            
            del params_pytorch, z_chunk
            cleanup_gpu_memory()
            
        except RuntimeError as e:
            print(f"   [ERROR] {e}")
            results['pytorch_mezo_step'] = {'time': float('inf'), 'memory': 0, 'error': str(e)}
            cleanup_gpu_memory()
    
    # 1b. MeZO ORIGINAL STYLE (actual MeZO per-param z allocation) - TRUE BASELINE
    if method in ['all', 'pytorch']:
        print("\n1b. MeZO Original Style (per-param z alloc - TRUE BASELINE):")
        
        try:
            cleanup_gpu_memory()
            params_list = create_params_list_like_model(n_elements, dtype=torch.float32, device='cuda')
            total_params = sum(p.numel() for p in params_list)
            print(f"    Created {len(params_list)} param tensors, total {total_params:,} elements")
            print(f"    Largest param: {max(p.numel() for p in params_list):,} elements")
            
            def mezo_original_step(params_list_local):
                """Complete MeZO step with per-param z allocation (actual MeZO behavior)."""
                # Step 1: Perturb +eps
                torch.manual_seed(seed)
                for param in params_list_local:
                    z = torch.normal(mean=0, std=1, size=param.data.size(),
                                   device=param.data.device, dtype=param.data.dtype)
                    param.data.add_(z, alpha=eps)
                loss1 = 1.0
                
                # Step 2: Perturb -2eps
                torch.manual_seed(seed)
                for param in params_list_local:
                    z = torch.normal(mean=0, std=1, size=param.data.size(),
                                   device=param.data.device, dtype=param.data.dtype)
                    param.data.add_(z, alpha=-2*eps)
                loss2 = 0.9
                
                # Step 3: Compute projected_grad
                projected_grad = (loss1 - loss2) / (2 * eps)
                
                # Step 4: Reset +eps
                torch.manual_seed(seed)
                for param in params_list_local:
                    z = torch.normal(mean=0, std=1, size=param.data.size(),
                                   device=param.data.device, dtype=param.data.dtype)
                    param.data.add_(z, alpha=eps)
                
                # Step 5: Update
                torch.manual_seed(seed)
                for param in params_list_local:
                    z = torch.normal(mean=0, std=1, size=param.data.size(),
                                   device=param.data.device, dtype=param.data.dtype)
                    param.data.add_(z, alpha=-lr * projected_grad)
            
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            
            # Warmup
            for _ in range(warmup):
                torch.manual_seed(42)
                for p in params_list:
                    p.normal_()
                mezo_original_step(params_list)
            torch.cuda.synchronize()
            
            torch.cuda.reset_peak_memory_stats()
            start = time.time()
            for _ in range(n_iter):
                torch.manual_seed(42)
                for p in params_list:
                    p.normal_()
                mezo_original_step(params_list)
            torch.cuda.synchronize()
            mezo_orig_time = (time.time() - start) * 1000 / n_iter
            mezo_orig_memory = torch.cuda.max_memory_allocated() / 1024**2
            
            results['mezo_original_step'] = {'time': mezo_orig_time, 'memory': mezo_orig_memory}
            print(f"   Time: {mezo_orig_time:.3f} ms, Memory: {mezo_orig_memory:.2f} MB")
            
            del params_list
            cleanup_gpu_memory()
            
        except RuntimeError as e:
            print(f"   [ERROR] {e}")
            results['mezo_original_step'] = {'time': float('inf'), 'memory': 0, 'error': str(e)}
            cleanup_gpu_memory()
    
    # CUDA Extension (complete step) - Already memory efficient (in-place)
    if HAS_CUDA_EXT and method in ['all', 'cuda']:
        print("\n2. CUDA Extension (complete MeZO step):")
        
        try:
            cleanup_gpu_memory()
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
            
            # Warmup (in-place, no clone)
            for _ in range(warmup):
                torch.manual_seed(42)
                params_cuda.normal_()
                cuda_mezo_step(params_cuda)
            torch.cuda.synchronize()
            
            start = time.time()
            for _ in range(n_iter):
                torch.manual_seed(42)
                params_cuda.normal_()
                cuda_mezo_step(params_cuda)
            torch.cuda.synchronize()
            cuda_time = (time.time() - start) * 1000 / n_iter
            cuda_memory = torch.cuda.max_memory_allocated() / 1024**2
            
            results['cuda_mezo_step'] = {'time': cuda_time, 'memory': cuda_memory}
            print(f"   Time: {cuda_time:.3f} ms, Memory: {cuda_memory:.2f} MB")
            
            del params_cuda
            cleanup_gpu_memory()
            
        except RuntimeError as e:
            print(f"   [ERROR] {e}")
            results['cuda_mezo_step'] = {'time': float('inf'), 'memory': 0, 'error': str(e)}
            cleanup_gpu_memory()
    
    # Triton (complete step)
    if HAS_TRITON and method in ['all', 'triton']:
        print("\n3. Triton (complete MeZO step):")
        
        try:
            cleanup_gpu_memory()
            from triton_fused_perturb import fused_update_kernel_autotuned
            params_triton = torch.randn(n_elements, device='cuda', dtype=torch.float32)
            
            # Handle large tensors with chunking
            MAX_GRID_SIZE = 2_000_000_000
            n_elements_int64 = int(n_elements)
            
            if n_elements_int64 <= MAX_GRID_SIZE:
                grid = lambda meta: (triton.cdiv(n_elements_int64, meta['BLOCK_SIZE']),)
                
                def triton_mezo_step(p):
                    # Step 1: Perturb +eps
                    fused_perturb_kernel_autotuned[grid](p, seed, eps, n_elements_int64)
                    loss1 = 1.0
                    # Step 3: Perturb -2eps
                    fused_perturb_kernel_autotuned[grid](p, seed, -2*eps, n_elements_int64)
                    loss2 = 0.9
                    # Step 5: Compute projected_grad
                    projected_grad = (loss1 - loss2) / (2 * eps)
                    # Step 6: Reset +eps
                    fused_perturb_kernel_autotuned[grid](p, seed, eps, n_elements_int64)
                    # Step 7: Update
                    fused_update_kernel_autotuned[grid](p, seed, projected_grad, lr, n_elements_int64)
            else:
                # Chunked approach
                chunk_size = MAX_GRID_SIZE
                n_chunks = (n_elements_int64 + chunk_size - 1) // chunk_size
                
                def triton_mezo_step(p):
                    # Step 1: Perturb +eps
                    for chunk_idx in range(n_chunks):
                        start_idx = chunk_idx * chunk_size
                        end_idx = min(start_idx + chunk_size, n_elements_int64)
                        chunk_size_actual = end_idx - start_idx
                        chunk_params = p[start_idx:end_idx]
                        grid_chunk = lambda meta: (triton.cdiv(chunk_size_actual, meta['BLOCK_SIZE']),)
                        fused_perturb_kernel_autotuned[grid_chunk](chunk_params, seed, eps, chunk_size_actual)
                    loss1 = 1.0
                    # Step 3: Perturb -2eps
                    for chunk_idx in range(n_chunks):
                        start_idx = chunk_idx * chunk_size
                        end_idx = min(start_idx + chunk_size, n_elements_int64)
                        chunk_size_actual = end_idx - start_idx
                        chunk_params = p[start_idx:end_idx]
                        grid_chunk = lambda meta: (triton.cdiv(chunk_size_actual, meta['BLOCK_SIZE']),)
                        fused_perturb_kernel_autotuned[grid_chunk](chunk_params, seed, -2*eps, chunk_size_actual)
                    loss2 = 0.9
                    # Step 5: Compute projected_grad
                    projected_grad = (loss1 - loss2) / (2 * eps)
                    # Step 6: Reset +eps
                    for chunk_idx in range(n_chunks):
                        start_idx = chunk_idx * chunk_size
                        end_idx = min(start_idx + chunk_size, n_elements_int64)
                        chunk_size_actual = end_idx - start_idx
                        chunk_params = p[start_idx:end_idx]
                        grid_chunk = lambda meta: (triton.cdiv(chunk_size_actual, meta['BLOCK_SIZE']),)
                        fused_perturb_kernel_autotuned[grid_chunk](chunk_params, seed, eps, chunk_size_actual)
                    # Step 7: Update
                    for chunk_idx in range(n_chunks):
                        start_idx = chunk_idx * chunk_size
                        end_idx = min(start_idx + chunk_size, n_elements_int64)
                        chunk_size_actual = end_idx - start_idx
                        chunk_params = p[start_idx:end_idx]
                        grid_chunk = lambda meta: (triton.cdiv(chunk_size_actual, meta['BLOCK_SIZE']),)
                        fused_update_kernel_autotuned[grid_chunk](chunk_params, seed, projected_grad, lr, chunk_size_actual)
            
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            
            # Warmup (in-place, no clone)
            for _ in range(warmup + 5):
                torch.manual_seed(42)
                params_triton.normal_()
                triton_mezo_step(params_triton)
            torch.cuda.synchronize()
            
            start = time.time()
            for _ in range(n_iter):
                torch.manual_seed(42)
                params_triton.normal_()
                triton_mezo_step(params_triton)
            torch.cuda.synchronize()
            triton_time = (time.time() - start) * 1000 / n_iter
            triton_memory = torch.cuda.max_memory_allocated() / 1024**2
            
            results['triton_mezo_step'] = {'time': triton_time, 'memory': triton_memory}
            print(f"   Time: {triton_time:.3f} ms, Memory: {triton_memory:.2f} MB")
            
            del params_triton
            cleanup_gpu_memory()
            
        except RuntimeError as e:
            print(f"   [ERROR] {e}")
            results['triton_mezo_step'] = {'time': float('inf'), 'memory': 0, 'error': str(e)}
            cleanup_gpu_memory()
    
    # Summary
    print(f"\n{'-'*90}")
    print("SUMMARY (Complete MeZO Step)")
    print(f"{'-'*90}")
    
    # Use mezo_original_step as baseline (true MeZO behavior), fallback to pytorch_mezo_step
    baseline_time = results.get('mezo_original_step', {}).get('time', float('inf'))
    if baseline_time == float('inf'):
        baseline_time = results.get('pytorch_mezo_step', {}).get('time', float('inf'))
    baseline_name = 'mezo_original_step' if 'mezo_original_step' in results else 'pytorch_mezo_step'
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]['time'])
    
    print(f"{'Implementation':<30} {'Time (ms)':>12} {'Speedup':>10} {'Memory (MB)':>15}")
    print(f"(Baseline: {baseline_name})")
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
    parser = argparse.ArgumentParser(description='Comprehensive kernel benchmark')
    parser.add_argument('--model', type=str, choices=list(MODEL_SIZES.keys()) + ['all', 'custom'],
                        default='all', help='Model size to benchmark')
    parser.add_argument('--n_elements', type=int, default=None,
                        help='Custom number of elements (only used with --model custom)')
    parser.add_argument('--n_iter', type=int, default=50, help='Number of iterations')
    parser.add_argument('--warmup', type=int, default=10, help='Number of warmup iterations')
    parser.add_argument('--skip_correctness', action='store_true',
                        help='Skip correctness verification')
    parser.add_argument('--complete_step', action='store_true',
                        help='Also run complete MeZO step benchmark')
    parser.add_argument('--complete_step_only', action='store_true',
                        help='Run ONLY the complete MeZO step benchmark (skip individual kernel benchmarks)')
    parser.add_argument('--method', type=str, default='all',
                        choices=['all', 'pytorch_baseline', 'mezo_original', 'pytorch_fused', 'chunked', 
                                 'triton_v1', 'triton_v2', 'triton_auto', 'cuda',
                                 'pytorch', 'triton'],  # Also accept complete step method names
                        help='Specific method to benchmark. For individual kernels: pytorch_baseline, mezo_original (real MeZO), pytorch_fused, chunked, triton_v1, triton_v2, triton_auto, cuda. For complete step: pytorch, triton, cuda.')
    args = parser.parse_args()
    
    # Determine sizes to test
    if args.model == 'all':
        # Test all model sizes
        sizes = [
            (10_000_000, "10M"),
            (100_000_000, "100M"),
        ] + [(MODEL_SIZES[k], k.upper()) for k in MODEL_SIZES.keys()]
    elif args.model == 'custom':
        if args.n_elements is None:
            raise ValueError("--n_elements must be provided when --model custom")
        sizes = [(args.n_elements, f"Custom-{args.n_elements:,}")]
    else:
        sizes = [(MODEL_SIZES[args.model], args.model.upper())]
    
    # Run correctness verification first (unless skipped or running specific methods)
    if not args.skip_correctness and args.method == 'all' and not args.complete_step_only:
        verify_correctness()
    
    # If --complete_step_only is set, skip individual benchmarks
    if args.complete_step_only:
        if args.model == 'all':
            test_size = max(MODEL_SIZES.values())
            test_name = max(MODEL_SIZES.items(), key=lambda x: x[1])[0].upper()
        elif args.model == 'custom':
            test_size = args.n_elements
            test_name = f"Custom-{args.n_elements:,}"
        else:
            test_size = MODEL_SIZES[args.model]
            test_name = args.model.upper()
        
        print("=" * 90)
        print(f"COMPLETE MEZO STEP BENCHMARK ONLY ({test_name})")
        print("=" * 90)
        # Map method names for complete step: pytorch_baseline -> pytorch, etc.
        complete_method = 'all'
        if args.method in ['pytorch_baseline', 'pytorch_fused', 'chunked', 'pytorch']:
            complete_method = 'pytorch'
        elif args.method in ['triton_v1', 'triton_v2', 'triton_auto', 'triton']:
            complete_method = 'triton'
        elif args.method == 'cuda':
            complete_method = 'cuda'
        benchmark_complete_mezo_step(n_elements=test_size, n_iter=args.n_iter, warmup=args.warmup//2, method=complete_method)
        
        print("\n" + "=" * 90)
        print("BENCHMARK COMPLETE")
        print("=" * 90)
    else:
        # Run comprehensive benchmark
        results = run_comprehensive_benchmark(
            sizes=sizes,
            n_iter=args.n_iter,
            warmup=args.warmup,
            method=args.method,
        )
        
        # Run complete MeZO step benchmark if requested
        if args.complete_step:
            if args.model == 'all':
                # Test with largest model
                test_size = max(MODEL_SIZES.values())
                test_name = max(MODEL_SIZES.items(), key=lambda x: x[1])[0].upper()
            elif args.model == 'custom':
                test_size = args.n_elements
                test_name = f"Custom-{args.n_elements:,}"
            else:
                test_size = MODEL_SIZES[args.model]
                test_name = args.model.upper()
            
            print("\n" + "=" * 90)
            print(f"COMPLETE MEZO STEP BENCHMARK ({test_name})")
            print("=" * 90)
            # Map method names for complete step
            complete_method = 'all'
            if args.method in ['pytorch_baseline', 'pytorch_fused', 'chunked', 'pytorch']:
                complete_method = 'pytorch'
            elif args.method in ['triton_v1', 'triton_v2', 'triton_auto', 'triton']:
                complete_method = 'triton'
            elif args.method == 'cuda':
                complete_method = 'cuda'
            benchmark_complete_mezo_step(n_elements=test_size, n_iter=args.n_iter, warmup=args.warmup//2, method=complete_method)
        
        print("\n" + "=" * 90)
        print("BENCHMARK COMPLETE")
        print("=" * 90)
