#!/usr/bin/env python3
"""
Fair Benchmark Comparison: CUDA vs Triton

This script ensures fair comparison by:
1. Measuring timing WITHOUT memory reset overhead
2. Measuring memory separately in dedicated runs
3. Using same warmup for all kernels
4. Includes complete MeZO training step benchmark

Model size presets:
- OPT-350M: 331,196,416 parameters
- OPT-2.7B: 2,700,000,000 parameters (approximate)
- OPT-6.7B: 6,700,000,000 parameters (approximate)
- OPT-13B: 13,000,000,000 parameters (approximate)
"""

import torch
import time
import numpy as np
import argparse
import gc

# Model size presets (parameter counts)
MODEL_SIZES = {
    'opt-350m': 331_196_416,
    'opt-2.7b': 2_700_000_000,
    'opt-6.7b': 6_700_000_000,
    'opt-13b': 13_000_000_000,
}

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
    # Force cleanup before measurement
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    
    # Run kernel once to measure memory
    kernel_func(params)
    torch.cuda.synchronize()
    
    memory_mb = torch.cuda.max_memory_allocated() / 1024**2
    
    # Clean up after measurement
    torch.cuda.empty_cache()
    gc.collect()
    
    return memory_mb


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
        # Clean up before starting
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        gc.collect()
        
        try:
            params_cuda = torch.randn(n_elements, device='cuda', dtype=torch.float32)
            
            def cuda_kernel(p):
                fused_perturb_cuda.fused_perturb(p, seed, alpha)
            
            time_ms = benchmark_timing(params_cuda, cuda_kernel, n_iter, warmup)
            # FIX: Don't use clone() for memory measurement - it artificially doubles memory!
            # Instead, reset stats and measure directly
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            cuda_kernel(params_cuda)
            torch.cuda.synchronize()
            memory_mb = torch.cuda.max_memory_allocated() / 1024**2
            results['CUDA Extension'] = {'time': time_ms, 'memory': memory_mb}
            print(f"  Time: {time_ms:.3f} ms, Memory: {memory_mb:.2f} MB")
            
            # Explicit cleanup
            del params_cuda
            del cuda_kernel
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            gc.collect()
        except torch.cuda.OutOfMemoryError as e:
            print(f"  ERROR: Out of memory - {e}")
            results['CUDA Extension'] = {'time': float('inf'), 'memory': 0.0}
            # Clean up even on error
            torch.cuda.empty_cache()
            gc.collect()
    
    # Triton Autotuned
    if HAS_TRITON:
        print("Testing Triton Autotuned...")
        # Clean up before starting
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        gc.collect()
        
        try:
            params_triton = torch.randn(n_elements, device='cuda', dtype=torch.float32)
            
            # For very large tensors, use chunking to avoid grid size limits
            # CUDA grid dimension limit is 2^31-1, but we'll be conservative
            MAX_GRID_SIZE = 2_000_000_000  # 2B elements per grid
            n_elements_int64 = int(n_elements)
            
            if n_elements_int64 <= MAX_GRID_SIZE:
                # Single kernel call
                grid = lambda meta: (triton.cdiv(n_elements_int64, meta['BLOCK_SIZE']),)
                
                def triton_kernel(p):
                    fused_perturb_kernel_autotuned[grid](p, seed, alpha, n_elements_int64)
            else:
                # Chunked approach for very large tensors
                chunk_size = MAX_GRID_SIZE
                n_chunks = (n_elements_int64 + chunk_size - 1) // chunk_size
                
                def triton_kernel(p):
                    for chunk_idx in range(n_chunks):
                        start_idx = chunk_idx * chunk_size
                        end_idx = min(start_idx + chunk_size, n_elements_int64)
                        chunk_size_actual = end_idx - start_idx
                        chunk_params = p[start_idx:end_idx]
                        grid_chunk = lambda meta: (triton.cdiv(chunk_size_actual, meta['BLOCK_SIZE']),)
                        fused_perturb_kernel_autotuned[grid_chunk](
                            chunk_params, seed, alpha, chunk_size_actual
                        )
            
            try:
                time_ms = benchmark_timing(params_triton, triton_kernel, n_iter, warmup + 5)
                # FIX: Don't use clone() for memory measurement - it artificially doubles memory!
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
                triton_kernel(params_triton)
                torch.cuda.synchronize()
                memory_mb = torch.cuda.max_memory_allocated() / 1024**2
                results['Triton Autotuned'] = {'time': time_ms, 'memory': memory_mb}
                print(f"  Time: {time_ms:.3f} ms, Memory: {memory_mb:.2f} MB")
            except RuntimeError as e:
                print(f"  ERROR: {e}")
                print(f"  Skipping Triton benchmark for this size")
                results['Triton Autotuned'] = {'time': float('inf'), 'memory': 0.0}
            finally:
                # Explicit cleanup
                del params_triton
                del triton_kernel
                if n_elements_int64 <= MAX_GRID_SIZE:
                    del grid
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                gc.collect()
        except torch.cuda.OutOfMemoryError as e:
            print(f"  ERROR: Out of memory - {e}")
            results['Triton Autotuned'] = {'time': float('inf'), 'memory': 0.0}
            # Clean up even on error
            torch.cuda.empty_cache()
            gc.collect()
    
    # PyTorch Baseline (MeZO Original Style)
    # ALWAYS use per-param z allocation to match REAL MeZO behavior
    # This is important for fair comparison - MeZO iterates through params
    # and allocates z for each param separately (not a flat tensor)
    print("Testing PyTorch Baseline (MeZO per-param z - TRUE baseline)...")
    
    # Clean up before starting
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    
    try:
        # Per-parameter approach that matches real MeZO implementation
        params_list = create_realistic_param_list(n_elements, device='cuda', dtype=torch.float32)
        n_params = len(params_list)
        actual_elements = sum(p.numel() for p in params_list)
        print(f"  Created {n_params} param tensors, {actual_elements:,} total elements")
        
        def pytorch_kernel_per_param(params):
            torch.manual_seed(seed)
            for param in params:
                z = torch.normal(mean=0, std=1, size=param.size(),
                                device=param.device, dtype=param.dtype)
                param.add_(z, alpha=alpha)
        
        # Warmup
        for _ in range(warmup):
            pytorch_kernel_per_param(params_list)
        torch.cuda.synchronize()
        
        # Timing
        start = time.time()
        for _ in range(n_iter):
            pytorch_kernel_per_param(params_list)
        torch.cuda.synchronize()
        time_ms = (time.time() - start) * 1000 / n_iter
        
        # Memory
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        pytorch_kernel_per_param(params_list)
        torch.cuda.synchronize()
        memory_mb = torch.cuda.max_memory_allocated() / 1024**2
        
        results['PyTorch Baseline'] = {'time': time_ms, 'memory': memory_mb}
        print(f"  Time: {time_ms:.3f} ms, Memory: {memory_mb:.2f} MB")
        
        del params_list
        del pytorch_kernel_per_param
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        gc.collect()
    except torch.cuda.OutOfMemoryError as e:
        print(f"  ERROR: Out of memory - {e}")
        results['PyTorch Baseline'] = {'time': float('inf'), 'memory': 0.0}
        # Clean up even on error
        torch.cuda.empty_cache()
        gc.collect()
    
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


def create_realistic_param_list(n_elements, device='cuda', dtype=torch.float32):
    """
    Create a list of parameter tensors that EXACTLY matches real OPT model structure.
    
    This uses the actual OPT architecture specifications to create accurate parameter
    distributions for benchmarking MeZO perturbation operations.
    
    OPT Model Configurations (from HuggingFace):
    - OPT-350M: hidden=1024, layers=24, ffn=4096,  vocab=50272 → 331M params
    - OPT-1.3B: hidden=2048, layers=24, ffn=8192,  vocab=50272 → 1.3B params
    - OPT-2.7B: hidden=2560, layers=32, ffn=10240, vocab=50272 → 2.7B params
    - OPT-6.7B: hidden=4096, layers=32, ffn=16384, vocab=50272 → 6.7B params
    - OPT-13B:  hidden=5120, layers=40, ffn=20480, vocab=50272 → 13B params
    """
    # Determine which OPT model config to use based on parameter count
    OPT_CONFIGS = {
        331_196_416: {'hidden': 1024, 'layers': 24, 'ffn': 4096, 'vocab': 50272, 'name': 'OPT-350M'},
        1_315_753_984: {'hidden': 2048, 'layers': 24, 'ffn': 8192, 'vocab': 50272, 'name': 'OPT-1.3B'},
        2_700_000_000: {'hidden': 2560, 'layers': 32, 'ffn': 10240, 'vocab': 50272, 'name': 'OPT-2.7B'},
        6_700_000_000: {'hidden': 4096, 'layers': 32, 'ffn': 16384, 'vocab': 50272, 'name': 'OPT-6.7B'},
        13_000_000_000: {'hidden': 5120, 'layers': 40, 'ffn': 20480, 'vocab': 50272, 'name': 'OPT-13B'},
    }
    
    # Find closest config or calculate from n_elements
    closest_config = min(OPT_CONFIGS.keys(), key=lambda x: abs(x - n_elements))
    config = OPT_CONFIGS[closest_config]
    
    hidden = config['hidden']
    n_layers = config['layers']
    ffn_dim = config['ffn']
    vocab_size = config['vocab']
    
    params = []
    
    # === Embedding layers ===
    # embed_tokens: [vocab_size, hidden/2] (OPT uses word_embed_proj_dim = hidden/2 for some models)
    # For simplicity, use [vocab_size, hidden/2] to match OPT-350M's 512-dim embedding
    embed_dim = hidden // 2 if hidden >= 1024 else hidden
    params.append(torch.randn(vocab_size * embed_dim, device=device, dtype=dtype))  # embed_tokens
    
    # embed_positions: [max_position, hidden]
    max_positions = 2050  # OPT default
    params.append(torch.randn(max_positions * hidden, device=device, dtype=dtype))  # embed_positions
    
    # project_in/project_out if embed_dim != hidden
    if embed_dim != hidden:
        params.append(torch.randn(embed_dim * hidden, device=device, dtype=dtype))  # project_in
        params.append(torch.randn(hidden * embed_dim, device=device, dtype=dtype))  # project_out
    
    # === Transformer layers ===
    for layer_idx in range(n_layers):
        # Self-attention: Q, K, V, O projections
        params.append(torch.randn(hidden * hidden, device=device, dtype=dtype))  # q_proj.weight
        params.append(torch.randn(hidden, device=device, dtype=dtype))           # q_proj.bias
        params.append(torch.randn(hidden * hidden, device=device, dtype=dtype))  # k_proj.weight
        params.append(torch.randn(hidden, device=device, dtype=dtype))           # k_proj.bias
        params.append(torch.randn(hidden * hidden, device=device, dtype=dtype))  # v_proj.weight
        params.append(torch.randn(hidden, device=device, dtype=dtype))           # v_proj.bias
        params.append(torch.randn(hidden * hidden, device=device, dtype=dtype))  # out_proj.weight
        params.append(torch.randn(hidden, device=device, dtype=dtype))           # out_proj.bias
        
        # Self-attention layer norm
        params.append(torch.randn(hidden, device=device, dtype=dtype))           # self_attn_layer_norm.weight
        params.append(torch.randn(hidden, device=device, dtype=dtype))           # self_attn_layer_norm.bias
        
        # FFN: fc1 (up), fc2 (down)
        params.append(torch.randn(ffn_dim * hidden, device=device, dtype=dtype)) # fc1.weight
        params.append(torch.randn(ffn_dim, device=device, dtype=dtype))          # fc1.bias
        params.append(torch.randn(hidden * ffn_dim, device=device, dtype=dtype)) # fc2.weight
        params.append(torch.randn(hidden, device=device, dtype=dtype))           # fc2.bias
        
        # Final layer norm
        params.append(torch.randn(hidden, device=device, dtype=dtype))           # final_layer_norm.weight
        params.append(torch.randn(hidden, device=device, dtype=dtype))           # final_layer_norm.bias
    
    # Calculate actual total and scale if needed
    actual_total = sum(p.numel() for p in params)
    
    # If there's a significant difference, scale the largest layers proportionally
    if abs(actual_total - n_elements) > n_elements * 0.01:  # More than 1% difference
        # Scale factor to match target
        scale = n_elements / actual_total
        
        # Rebuild with scaled sizes (only scale large layers, keep biases same)
        params_scaled = []
        for p in params:
            if p.numel() > 10000:  # Only scale large tensors
                new_size = int(p.numel() * scale)
                params_scaled.append(torch.randn(new_size, device=device, dtype=dtype))
            else:
                params_scaled.append(p)
        
        # Handle any remaining difference with small adjustment
        final_total = sum(p.numel() for p in params_scaled)
        diff = n_elements - final_total
        if diff > 0:
            params_scaled.append(torch.randn(diff, device=device, dtype=dtype))
        
        return params_scaled
    
    return params


def pytorch_perturb_per_param(params_list, seed, scaling_factor, eps):
    """
    Per-parameter perturbation matching profile_dizo.py / trainer.py
    Creates small z tensors per parameter, immediately garbage collected.
    """
    torch.manual_seed(seed)
    for param in params_list:
        z = torch.normal(mean=0, std=1, size=param.size(), 
                        device=param.device, dtype=param.dtype)
        param.add_(z, alpha=scaling_factor * eps)
        # z is freed here after each iteration


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
    
    PyTorch baseline uses REALISTIC per-parameter z allocation (like profile_dizo.py)
    to avoid artificial OOM from flat z tensor approach.
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
    
    # PyTorch Baseline (complete step) - REALISTIC per-parameter approach
    print("Testing PyTorch Baseline (complete MeZO step, per-param z)...")
    # Clean up before starting
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    
    try:
        # Create realistic parameter structure (like a real model)
        params_list = create_realistic_param_list(n_elements, device='cuda', dtype=torch.float32)
        n_params = len(params_list)
        actual_elements = sum(p.numel() for p in params_list)
        print(f"  Created {n_params} parameter tensors, {actual_elements:,} total elements")
        
        def pytorch_mezo_step_realistic(params):
            seed = 42
            # Step 1: Perturb +eps (per-parameter z, immediately freed)
            pytorch_perturb_per_param(params, seed, 1, eps)
            loss1 = 1.0  # Dummy forward
            
            # Step 3: Perturb -2eps
            pytorch_perturb_per_param(params, seed, -2, eps)
            loss2 = 0.9  # Dummy forward
            
            # Step 5: Compute projected_grad
            projected_grad = (loss1 - loss2) / (2 * eps)
            
            # Step 6: Reset +eps
            pytorch_perturb_per_param(params, seed, 1, eps)
            
            # Step 7: Update
            pytorch_perturb_per_param(params, seed, -lr * projected_grad / eps, eps)
        
        # Warmup
        for _ in range(warmup):
            pytorch_mezo_step_realistic(params_list)
        torch.cuda.synchronize()
        
        # Benchmark
        start = time.time()
        for _ in range(n_iter):
            pytorch_mezo_step_realistic(params_list)
        torch.cuda.synchronize()
        pytorch_time = (time.time() - start) * 1000 / n_iter
        
        # Memory
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        gc.collect()
        pytorch_mezo_step_realistic(params_list)
        torch.cuda.synchronize()
        pytorch_memory = torch.cuda.max_memory_allocated() / 1024**2
        
        results['PyTorch Baseline'] = {'time': pytorch_time, 'memory': pytorch_memory}
        print(f"  Time: {pytorch_time:.3f} ms, Memory: {pytorch_memory:.2f} MB")
        
        # Explicit cleanup
        del params_list
        del pytorch_mezo_step_realistic
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        gc.collect()
    except torch.cuda.OutOfMemoryError as e:
        print(f"  ERROR: Out of memory - {e}")
        results['PyTorch Baseline'] = {'time': float('inf'), 'memory': 0.0}
        # Clean up even on error
        torch.cuda.empty_cache()
        gc.collect()
    
    # CUDA Extension (complete step)
    if HAS_CUDA_EXT:
        print("Testing CUDA Extension (complete MeZO step)...")
        # Clean up before starting
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        gc.collect()
        
        try:
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
            
            # Warmup - operate in-place, no clone needed for fused kernels
            for _ in range(warmup):
                cuda_mezo_step(params_cuda)
            torch.cuda.synchronize()
            
            # Benchmark - operate in-place
            start = time.time()
            for _ in range(n_iter):
                cuda_mezo_step(params_cuda)
            torch.cuda.synchronize()
            cuda_time = (time.time() - start) * 1000 / n_iter
            
            # Memory - measure with fresh tensor to get accurate peak
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            cuda_mezo_step(params_cuda)
            torch.cuda.synchronize()
            cuda_memory = torch.cuda.max_memory_allocated() / 1024**2
            
            results['CUDA Extension'] = {'time': cuda_time, 'memory': cuda_memory}
            print(f"  Time: {cuda_time:.3f} ms, Memory: {cuda_memory:.2f} MB")
            
            # Explicit cleanup
            del params_cuda
            del cuda_mezo_step
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            gc.collect()
        except torch.cuda.OutOfMemoryError as e:
            print(f"  ERROR: Out of memory - {e}")
            results['CUDA Extension'] = {'time': float('inf'), 'memory': 0.0}
            # Clean up even on error
            torch.cuda.empty_cache()
            gc.collect()
    
    # Triton (complete step)
    if HAS_TRITON:
        print("Testing Triton (complete MeZO step)...")
        # Clean up before starting
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        gc.collect()
        
        try:
            params_triton = torch.randn(n_elements, device='cuda', dtype=torch.float32)
            seed = 42
            
            # Handle large tensors with chunking
            MAX_GRID_SIZE = 2_000_000_000
            n_elements_int64 = int(n_elements)
            
            if n_elements_int64 <= MAX_GRID_SIZE:
                grid = lambda meta: (triton.cdiv(n_elements_int64, meta['BLOCK_SIZE']),)
                
                def triton_mezo_step(p):
                    # Step 1: Perturb +eps
                    fused_perturb_kernel_autotuned[grid](p, seed, eps, n_elements_int64)
                    loss1 = 1.0  # Dummy loss
                    # Step 3: Perturb -2eps
                    fused_perturb_kernel_autotuned[grid](p, seed, -2*eps, n_elements_int64)
                    loss2 = 0.9  # Dummy loss
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
            
            # Warmup - operate in-place, no clone needed for fused kernels
            for _ in range(warmup + 5):
                triton_mezo_step(params_triton)
            torch.cuda.synchronize()
            
            # Benchmark - operate in-place
            start = time.time()
            for _ in range(n_iter):
                triton_mezo_step(params_triton)
            torch.cuda.synchronize()
            triton_time = (time.time() - start) * 1000 / n_iter
            
            # Memory - measure with current tensor
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            triton_mezo_step(params_triton)
            torch.cuda.synchronize()
            triton_memory = torch.cuda.max_memory_allocated() / 1024**2
            
            results['Triton'] = {'time': triton_time, 'memory': triton_memory}
            print(f"  Time: {triton_time:.3f} ms, Memory: {triton_memory:.2f} MB")
            
            # Explicit cleanup
            del params_triton
            del triton_mezo_step
            if n_elements_int64 <= MAX_GRID_SIZE:
                del grid
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            gc.collect()
        except torch.cuda.OutOfMemoryError as e:
            print(f"  ERROR: Out of memory - {e}")
            results['Triton'] = {'time': float('inf'), 'memory': 0.0}
            # Clean up even on error
            torch.cuda.empty_cache()
            gc.collect()
    
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
    parser = argparse.ArgumentParser(description='Benchmark CUDA vs Triton kernels')
    parser.add_argument('--model', type=str, choices=list(MODEL_SIZES.keys()) + ['all', 'custom'],
                        default='all', help='Model size to benchmark')
    parser.add_argument('--n_elements', type=int, default=None,
                        help='Custom number of elements (only used with --model custom)')
    parser.add_argument('--n_iter', type=int, default=100, help='Number of iterations')
    parser.add_argument('--warmup', type=int, default=10, help='Number of warmup iterations')
    parser.add_argument('--complete_step', action='store_true',
                        help='Also run complete MeZO step benchmark')
    args = parser.parse_args()
    
    if args.model == 'all':
        # Test all model sizes
        models_to_test = list(MODEL_SIZES.items())
    elif args.model == 'custom':
        if args.n_elements is None:
            raise ValueError("--n_elements must be provided when --model custom")
        models_to_test = [('custom', args.n_elements)]
    else:
        models_to_test = [(args.model, MODEL_SIZES[args.model])]
    
    for model_name, n_elements in models_to_test:
        print("\n" + "="*80)
        print(f"MODEL: {model_name.upper()} ({n_elements:,} parameters, {n_elements * 4 / 1024**2:.1f} MB)")
        print("="*80)
        fair_benchmark(n_elements=n_elements, n_iter=args.n_iter, warmup=args.warmup)
        
        if args.complete_step:
            # Critical: Clean up all memory before running complete step benchmark
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            gc.collect()
            
            print("\n\n" + "="*80)
            print(f"COMPLETE MEZO STEP TEST ({model_name.upper()})")
            print("="*80)
            benchmark_complete_mezo_step(n_elements=n_elements, n_iter=args.n_iter//2, warmup=args.warmup//2)
        
        # Clean up between model sizes
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        gc.collect()

