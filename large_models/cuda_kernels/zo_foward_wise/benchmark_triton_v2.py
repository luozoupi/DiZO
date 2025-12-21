"""
Benchmark: Optimized Triton V2 Kernels vs PyTorch vs CUDA V2

Compares:
1. PyTorch per-group (baseline)
2. PyTorch per-group with torch.compile
3. Triton V2 single-block (legacy)
4. Triton V2 multi-block (optimized)
5. CUDA V2 (if available)
6. CUDA V5 (if available)
"""

import torch
import time
import argparse
import gc
from typing import Dict, Tuple

# Import Triton kernels
from dizo_fused_kernels_v2 import FusedDiZOKernelsV2

# Try to import CUDA V2
try:
    import dizo_fused_kernels_cuda_v2 as cuda_v2
    CUDA_V2_AVAILABLE = True
except ImportError:
    CUDA_V2_AVAILABLE = False
    print("CUDA V2 not available, skipping CUDA V2 comparison")

# Try to import CUDA V3
try:
    import dizo_fused_kernels_cuda_v3 as cuda_v3
    CUDA_V3_AVAILABLE = True
except ImportError:
    CUDA_V3_AVAILABLE = False

# Try to import CUDA V5
try:
    import dizo_fused_kernels_cuda_v5 as cuda_v5
    CUDA_V5_AVAILABLE = True
except ImportError:
    CUDA_V5_AVAILABLE = False
    print("CUDA V5 not available, skipping CUDA V5 comparison")


def create_model_params(model_name: str, device: torch.device) -> Tuple[list, int]:
    """Create parameter groups matching model architecture."""
    configs = {
        'opt-350m': {'layers': 24, 'hidden': 1024, 'ffn': 4096},
        'opt-1.3b': {'layers': 24, 'hidden': 2048, 'ffn': 8192},
        'opt-2.7b': {'layers': 32, 'hidden': 2560, 'ffn': 10240},
        'opt-6.7b': {'layers': 32, 'hidden': 4096, 'ffn': 16384},
    }
    
    config = configs.get(model_name, configs['opt-350m'])
    layers = config['layers']
    hidden = config['hidden']
    ffn = config['ffn']
    
    param_groups = []
    for _ in range(layers):
        # Attention: q, k, v, o projections
        for _ in range(4):
            param_groups.append(torch.randn(hidden * hidden, device=device, dtype=torch.float32))
        # FFN: fc1, fc2
        for _ in range(2):
            param_groups.append(torch.randn(hidden * ffn, device=device, dtype=torch.float32))
    
    return param_groups, sum(p.numel() for p in param_groups)


def benchmark_pytorch_per_group(
    param_flat: torch.Tensor,
    anchor_flat: torch.Tensor,
    offsets: torch.Tensor,
    sizes: torch.Tensor,
    constraints: torch.Tensor,
    n_iter: int = 10,
    warmup: int = 3,
) -> Dict[str, float]:
    """Benchmark PyTorch per-group operations (baseline)."""
    num_params = len(sizes)
    device = param_flat.device
    
    # Warmup
    for _ in range(warmup):
        norms = torch.empty(num_params, device=device)
        for i in range(num_params):
            s, e = offsets[i].item(), offsets[i].item() + sizes[i].item()
            norms[i] = torch.norm(param_flat[s:e] - anchor_flat[s:e])
    torch.cuda.synchronize()
    
    # Benchmark norm
    start = time.time()
    for _ in range(n_iter):
        norms = torch.empty(num_params, device=device)
        for i in range(num_params):
            s, e = offsets[i].item(), offsets[i].item() + sizes[i].item()
            norms[i] = torch.norm(param_flat[s:e] - anchor_flat[s:e])
    torch.cuda.synchronize()
    norm_time = (time.time() - start) * 1000 / n_iter
    
    # Benchmark apply
    start = time.time()
    for _ in range(n_iter):
        alphas = constraints / (norms + 1e-8)
        for i in range(num_params):
            s, e = offsets[i].item(), offsets[i].item() + sizes[i].item()
            diff = param_flat[s:e] - anchor_flat[s:e]
            param_flat[s:e] = anchor_flat[s:e] + diff * alphas[i]
    torch.cuda.synchronize()
    apply_time = (time.time() - start) * 1000 / n_iter
    
    return {'norm': norm_time, 'apply': apply_time, 'total': norm_time + apply_time}


def benchmark_pytorch_compiled(
    param_flat: torch.Tensor,
    anchor_flat: torch.Tensor,
    offsets: torch.Tensor,
    sizes: torch.Tensor,
    constraints: torch.Tensor,
    n_iter: int = 10,
    warmup: int = 3,
) -> Dict[str, float]:
    """Benchmark PyTorch per-group operations with torch.compile."""
    num_params = len(sizes)
    device = param_flat.device
    
    # Define functions to compile
    def compute_norms_per_group(param_flat, anchor_flat, offsets, sizes):
        norms = torch.empty(num_params, device=device)
        for i in range(num_params):
            s, e = offsets[i].item(), offsets[i].item() + sizes[i].item()
            norms[i] = torch.norm(param_flat[s:e] - anchor_flat[s:e])
        return norms
    
    def apply_constraints_per_group(param_flat, anchor_flat, offsets, sizes, alphas):
        for i in range(num_params):
            s, e = offsets[i].item(), offsets[i].item() + sizes[i].item()
            diff = param_flat[s:e] - anchor_flat[s:e]
            param_flat[s:e] = anchor_flat[s:e] + diff * alphas[i]
        return param_flat
    
    # Compile with torch.compile
    compiled_norms = torch.compile(compute_norms_per_group, mode='reduce-overhead')
    compiled_apply = torch.compile(apply_constraints_per_group, mode='reduce-overhead')
    
    # Extended warmup for torch.compile (needs more iterations to stabilize)
    for _ in range(warmup + 5):
        norms = compiled_norms(param_flat.clone(), anchor_flat, offsets, sizes)
    torch.cuda.synchronize()
    
    # Benchmark norm
    start = time.time()
    for _ in range(n_iter):
        norms = compiled_norms(param_flat.clone(), anchor_flat, offsets, sizes)
    torch.cuda.synchronize()
    norm_time = (time.time() - start) * 1000 / n_iter
    
    # Extended warmup for apply
    alphas = constraints / (norms + 1e-8)
    for _ in range(warmup + 5):
        _ = compiled_apply(param_flat.clone(), anchor_flat, offsets, sizes, alphas)
    torch.cuda.synchronize()
    
    # Benchmark apply
    start = time.time()
    for _ in range(n_iter):
        _ = compiled_apply(param_flat.clone(), anchor_flat, offsets, sizes, alphas)
    torch.cuda.synchronize()
    apply_time = (time.time() - start) * 1000 / n_iter
    
    return {'norm': norm_time, 'apply': apply_time, 'total': norm_time + apply_time}


def benchmark_pytorch_vectorized(
    param_flat: torch.Tensor,
    anchor_flat: torch.Tensor,
    offsets: torch.Tensor,
    sizes: torch.Tensor,
    constraints: torch.Tensor,
    n_iter: int = 10,
    warmup: int = 3,
) -> Dict[str, float]:
    """Benchmark vectorized PyTorch operations using scatter_add.
    
    This version pre-computes segment IDs and uses fully vectorized operations
    (no Python loops in the hot path).
    """
    num_params = len(sizes)
    device = param_flat.device
    total_elements = param_flat.shape[0]
    
    # Pre-compute segment IDs (done once, outside loop)
    segment_ids = torch.zeros(total_elements, device=device, dtype=torch.long)
    offsets_cpu = offsets.cpu()
    sizes_cpu = sizes.cpu()
    for i in range(num_params):
        s = offsets_cpu[i].item()
        e = s + sizes_cpu[i].item()
        segment_ids[s:e] = i
    
    # Pure vectorized functions (no Python loops)
    def compute_norms_vectorized(param_flat, anchor_flat, segment_ids, num_params):
        diff = param_flat - anchor_flat
        sq_diff = diff * diff
        norms_sq = torch.zeros(num_params, device=param_flat.device)
        norms_sq.scatter_add_(0, segment_ids, sq_diff)
        return torch.sqrt(norms_sq)
    
    def apply_constraints_vectorized(param_flat, anchor_flat, segment_ids, alphas):
        diff = param_flat - anchor_flat
        alpha_expanded = alphas[segment_ids]  # Gather alphas by segment
        return anchor_flat + diff * alpha_expanded
    
    # Warmup
    for _ in range(warmup):
        norms = compute_norms_vectorized(param_flat.clone(), anchor_flat, segment_ids, num_params)
    torch.cuda.synchronize()
    
    # Benchmark norm
    start = time.time()
    for _ in range(n_iter):
        norms = compute_norms_vectorized(param_flat.clone(), anchor_flat, segment_ids, num_params)
    torch.cuda.synchronize()
    norm_time = (time.time() - start) * 1000 / n_iter
    
    # Warmup apply
    alphas = constraints / (norms + 1e-8)
    for _ in range(warmup):
        _ = apply_constraints_vectorized(param_flat.clone(), anchor_flat, segment_ids, alphas)
    torch.cuda.synchronize()
    
    # Benchmark apply
    start = time.time()
    for _ in range(n_iter):
        _ = apply_constraints_vectorized(param_flat.clone(), anchor_flat, segment_ids, alphas)
    torch.cuda.synchronize()
    apply_time = (time.time() - start) * 1000 / n_iter
    
    return {'norm': norm_time, 'apply': apply_time, 'total': norm_time + apply_time}


def benchmark_triton_simple(
    kernels: FusedDiZOKernelsV2,
    param_flat: torch.Tensor,
    anchor_flat: torch.Tensor,
    offsets: torch.Tensor,
    sizes: torch.Tensor,
    constraints: torch.Tensor,
    n_iter: int = 10,
    warmup: int = 3,
) -> Dict[str, float]:
    """Benchmark Triton single-block kernels."""
    # Warmup
    for _ in range(warmup):
        norms = kernels.compute_norms_simple(param_flat, anchor_flat, offsets, sizes)
    torch.cuda.synchronize()
    
    # Benchmark norm
    start = time.time()
    for _ in range(n_iter):
        norms = kernels.compute_norms_simple(param_flat, anchor_flat, offsets, sizes)
    torch.cuda.synchronize()
    norm_time = (time.time() - start) * 1000 / n_iter
    
    # Warmup apply
    for _ in range(warmup):
        kernels.apply_constraints_simple(param_flat, anchor_flat, offsets, sizes, constraints, norms)
    torch.cuda.synchronize()
    
    # Benchmark apply
    start = time.time()
    for _ in range(n_iter):
        kernels.apply_constraints_simple(param_flat, anchor_flat, offsets, sizes, constraints, norms)
    torch.cuda.synchronize()
    apply_time = (time.time() - start) * 1000 / n_iter
    
    return {'norm': norm_time, 'apply': apply_time, 'total': norm_time + apply_time}


def benchmark_triton_multiblock(
    kernels: FusedDiZOKernelsV2,
    param_flat: torch.Tensor,
    anchor_flat: torch.Tensor,
    offsets: torch.Tensor,
    sizes: torch.Tensor,
    constraints: torch.Tensor,
    n_iter: int = 10,
    warmup: int = 3,
) -> Dict[str, float]:
    """Benchmark Triton multi-block kernels (optimized)."""
    # Warmup
    for _ in range(warmup):
        norms = kernels.compute_norms(param_flat, anchor_flat, offsets, sizes)
    torch.cuda.synchronize()
    
    # Benchmark norm
    start = time.time()
    for _ in range(n_iter):
        norms = kernels.compute_norms(param_flat, anchor_flat, offsets, sizes)
    torch.cuda.synchronize()
    norm_time = (time.time() - start) * 1000 / n_iter
    
    # Warmup apply
    for _ in range(warmup):
        kernels.apply_constraints(param_flat, anchor_flat, offsets, sizes, constraints, norms)
    torch.cuda.synchronize()
    
    # Benchmark apply
    start = time.time()
    for _ in range(n_iter):
        kernels.apply_constraints(param_flat, anchor_flat, offsets, sizes, constraints, norms)
    torch.cuda.synchronize()
    apply_time = (time.time() - start) * 1000 / n_iter
    
    return {'norm': norm_time, 'apply': apply_time, 'total': norm_time + apply_time}


def benchmark_cuda_v2(
    param_flat: torch.Tensor,
    anchor_flat: torch.Tensor,
    offsets: torch.Tensor,
    sizes: torch.Tensor,
    constraints: torch.Tensor,
    n_iter: int = 10,
    warmup: int = 3,
) -> Dict[str, float]:
    """Benchmark CUDA V2 kernels."""
    if not CUDA_V2_AVAILABLE:
        return None
    
    # Warmup
    for _ in range(warmup):
        norms = cuda_v2.fused_compute_norms(param_flat, anchor_flat, offsets, sizes)
    torch.cuda.synchronize()
    
    # Benchmark norm
    start = time.time()
    for _ in range(n_iter):
        norms = cuda_v2.fused_compute_norms(param_flat, anchor_flat, offsets, sizes)
    torch.cuda.synchronize()
    norm_time = (time.time() - start) * 1000 / n_iter
    
    # Warmup apply
    for _ in range(warmup):
        cuda_v2.fused_apply_constraints(param_flat, anchor_flat, offsets, sizes, constraints, norms, 1e-8)
    torch.cuda.synchronize()
    
    # Benchmark apply
    start = time.time()
    for _ in range(n_iter):
        cuda_v2.fused_apply_constraints(param_flat, anchor_flat, offsets, sizes, constraints, norms, 1e-8)
    torch.cuda.synchronize()
    apply_time = (time.time() - start) * 1000 / n_iter
    
    return {'norm': norm_time, 'apply': apply_time, 'total': norm_time + apply_time}


def benchmark_cuda_v3(
    param_flat: torch.Tensor,
    anchor_flat: torch.Tensor,
    offsets: torch.Tensor,
    sizes: torch.Tensor,
    constraints: torch.Tensor,
    n_iter: int = 10,
    warmup: int = 3,
) -> Dict[str, float]:
    """Benchmark CUDA V3 kernels (atomic multi-block)."""
    if not CUDA_V3_AVAILABLE:
        return None
    
    # Initialize block mapping cache
    cuda_v3.init_block_mapping(sizes)
    
    # Warmup
    for _ in range(warmup):
        norms = cuda_v3.fused_compute_norms(param_flat, anchor_flat, offsets, sizes)
    torch.cuda.synchronize()
    
    # Benchmark norm
    start = time.time()
    for _ in range(n_iter):
        norms = cuda_v3.fused_compute_norms(param_flat, anchor_flat, offsets, sizes)
    torch.cuda.synchronize()
    norm_time = (time.time() - start) * 1000 / n_iter
    
    # Warmup apply
    for _ in range(warmup):
        cuda_v3.fused_apply_constraints(param_flat, anchor_flat, offsets, sizes, constraints, norms, 1e-8)
    torch.cuda.synchronize()
    
    # Benchmark apply
    start = time.time()
    for _ in range(n_iter):
        cuda_v3.fused_apply_constraints(param_flat, anchor_flat, offsets, sizes, constraints, norms, 1e-8)
    torch.cuda.synchronize()
    apply_time = (time.time() - start) * 1000 / n_iter
    
    return {'norm': norm_time, 'apply': apply_time, 'total': norm_time + apply_time}


def benchmark_cuda_v5(
    param_flat: torch.Tensor,
    anchor_flat: torch.Tensor,
    offsets: torch.Tensor,
    sizes: torch.Tensor,
    constraints: torch.Tensor,
    n_iter: int = 10,
    warmup: int = 3,
) -> Dict[str, float]:
    """Benchmark CUDA V5 kernels (optimized multi-block with float4)."""
    if not CUDA_V5_AVAILABLE:
        return None
    
    # Initialize block mapping cache
    cuda_v5.init_block_mapping(sizes)
    
    # Warmup
    for _ in range(warmup):
        norms = cuda_v5.fused_compute_norms(param_flat, anchor_flat, offsets, sizes)
    torch.cuda.synchronize()
    
    # Benchmark norm
    start = time.time()
    for _ in range(n_iter):
        norms = cuda_v5.fused_compute_norms(param_flat, anchor_flat, offsets, sizes)
    torch.cuda.synchronize()
    norm_time = (time.time() - start) * 1000 / n_iter
    
    # Warmup apply
    for _ in range(warmup):
        cuda_v5.fused_apply_constraints(param_flat, anchor_flat, offsets, sizes, constraints, norms, 1e-8)
    torch.cuda.synchronize()
    
    # Benchmark apply
    start = time.time()
    for _ in range(n_iter):
        cuda_v5.fused_apply_constraints(param_flat, anchor_flat, offsets, sizes, constraints, norms, 1e-8)
    torch.cuda.synchronize()
    apply_time = (time.time() - start) * 1000 / n_iter
    
    return {'norm': norm_time, 'apply': apply_time, 'total': norm_time + apply_time}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='opt-350m',
                        choices=['opt-350m', 'opt-1.3b', 'opt-2.7b', 'opt-6.7b'])
    parser.add_argument('--n_iter', type=int, default=10)
    parser.add_argument('--warmup', type=int, default=3)
    args = parser.parse_args()
    
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Model: {args.model}")
    print(f"Iterations: {args.n_iter}")
    print()
    
    # Create parameters
    param_groups, total_elements = create_model_params(args.model, device)
    num_params = len(param_groups)
    
    print(f"Parameter groups: {num_params}")
    print(f"Total elements: {total_elements:,}")
    print(f"Memory: {total_elements * 4 / 1024**3:.2f} GB")
    print()
    
    # Flatten
    param_flat = torch.cat([p.flatten() for p in param_groups])
    anchor_flat = torch.randn_like(param_flat)
    
    offsets = []
    offset = 0
    for p in param_groups:
        offsets.append(offset)
        offset += p.numel()
    offsets = torch.tensor(offsets, device=device, dtype=torch.long)
    sizes = torch.tensor([p.numel() for p in param_groups], device=device, dtype=torch.long)
    constraints = torch.rand(num_params, device=device) * 0.1
    
    # Initialize Triton kernels
    print("Initializing Triton V2 kernels...")
    kernels = FusedDiZOKernelsV2(num_params, total_elements, device, offsets, sizes)
    print()
    
    # Run benchmarks
    print("=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    
    # PyTorch baseline
    print("\n[1] PyTorch per-group (baseline)...")
    pt_results = benchmark_pytorch_per_group(
        param_flat.clone(), anchor_flat, offsets, sizes, constraints,
        args.n_iter, args.warmup
    )
    print(f"    Norm:  {pt_results['norm']:.2f} ms")
    print(f"    Apply: {pt_results['apply']:.2f} ms")
    print(f"    Total: {pt_results['total']:.2f} ms")
    
    # PyTorch with torch.compile (per-group)
    print("\n[2] PyTorch per-group + torch.compile...")
    try:
        pt_compiled = benchmark_pytorch_compiled(
            param_flat.clone(), anchor_flat, offsets, sizes, constraints,
            args.n_iter, args.warmup
        )
        print(f"    Norm:  {pt_compiled['norm']:.2f} ms")
        print(f"    Apply: {pt_compiled['apply']:.2f} ms")
        print(f"    Total: {pt_compiled['total']:.2f} ms")
        print(f"    Speedup vs PyTorch: {pt_results['total'] / pt_compiled['total']:.2f}x")
    except Exception as e:
        print(f"    Failed: {e}")
        pt_compiled = None
    
    # Clear memory before vectorized torch.compile
    torch._dynamo.reset()
    gc.collect()
    torch.cuda.empty_cache()
    
    # PyTorch vectorized (without torch.compile for fair comparison)
    print("\n[3] PyTorch vectorized (scatter_add)...")
    try:
        pt_vec_compiled = benchmark_pytorch_vectorized(
            param_flat.clone(), anchor_flat, offsets, sizes, constraints,
            args.n_iter, args.warmup
        )
        print(f"    Norm:  {pt_vec_compiled['norm']:.2f} ms")
        print(f"    Apply: {pt_vec_compiled['apply']:.2f} ms")
        print(f"    Total: {pt_vec_compiled['total']:.2f} ms")
        print(f"    Speedup vs PyTorch: {pt_results['total'] / pt_vec_compiled['total']:.2f}x")
    except Exception as e:
        print(f"    Failed: {e}")
        pt_vec_compiled = None
    
    # Clear memory before Triton
    gc.collect()
    torch.cuda.empty_cache()
    
    # Triton single-block
    print("\n[4] Triton V2 single-block...")
    triton_simple = benchmark_triton_simple(
        kernels, param_flat.clone(), anchor_flat, offsets, sizes, constraints,
        args.n_iter, args.warmup
    )
    print(f"    Norm:  {triton_simple['norm']:.2f} ms")
    print(f"    Apply: {triton_simple['apply']:.2f} ms")
    print(f"    Total: {triton_simple['total']:.2f} ms")
    print(f"    Speedup vs PyTorch: {pt_results['total'] / triton_simple['total']:.2f}x")
    
    # Triton multi-block
    print("\n[5] Triton V2 multi-block (optimized)...")
    triton_multi = benchmark_triton_multiblock(
        kernels, param_flat.clone(), anchor_flat, offsets, sizes, constraints,
        args.n_iter, args.warmup
    )
    print(f"    Norm:  {triton_multi['norm']:.2f} ms")
    print(f"    Apply: {triton_multi['apply']:.2f} ms")
    print(f"    Total: {triton_multi['total']:.2f} ms")
    print(f"    Speedup vs PyTorch: {pt_results['total'] / triton_multi['total']:.2f}x")
    print(f"    Speedup vs Triton simple: {triton_simple['total'] / triton_multi['total']:.2f}x")
    
    # CUDA V2
    cuda_results = None
    if CUDA_V2_AVAILABLE:
        print("\n[6] CUDA V2...")
        cuda_results = benchmark_cuda_v2(
            param_flat.clone(), anchor_flat, offsets, sizes, constraints,
            args.n_iter, args.warmup
        )
        print(f"    Norm:  {cuda_results['norm']:.2f} ms")
        print(f"    Apply: {cuda_results['apply']:.2f} ms")
        print(f"    Total: {cuda_results['total']:.2f} ms")
        print(f"    Speedup vs PyTorch: {pt_results['total'] / cuda_results['total']:.2f}x")
        print(f"    Triton multi vs CUDA V2: {cuda_results['total'] / triton_multi['total']:.2f}x")
    
    # CUDA V5
    cuda_v5_results = None
    if CUDA_V5_AVAILABLE:
        print("\n[7] CUDA V5 (optimized multi-block)...")
        cuda_v5_results = benchmark_cuda_v5(
            param_flat.clone(), anchor_flat, offsets, sizes, constraints,
            args.n_iter, args.warmup
        )
        print(f"    Norm:  {cuda_v5_results['norm']:.2f} ms")
        print(f"    Apply: {cuda_v5_results['apply']:.2f} ms")
        print(f"    Total: {cuda_v5_results['total']:.2f} ms")
        print(f"    Speedup vs PyTorch: {pt_results['total'] / cuda_v5_results['total']:.2f}x")
        if cuda_results:
            print(f"    Speedup vs CUDA V2: {cuda_results['total'] / cuda_v5_results['total']:.2f}x")
    
    # CUDA V3
    cuda_v3_results = None
    if CUDA_V3_AVAILABLE:
        print("\n[8] CUDA V3 (atomic multi-block)...")
        cuda_v3_results = benchmark_cuda_v3(
            param_flat.clone(), anchor_flat, offsets, sizes, constraints,
            args.n_iter, args.warmup
        )
        print(f"    Norm:  {cuda_v3_results['norm']:.2f} ms")
        print(f"    Apply: {cuda_v3_results['apply']:.2f} ms")
        print(f"    Total: {cuda_v3_results['total']:.2f} ms")
        print(f"    Speedup vs PyTorch: {pt_results['total'] / cuda_v3_results['total']:.2f}x")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Method':<40} {'Norm (ms)':<12} {'Apply (ms)':<12} {'Total (ms)':<12} {'Speedup':<10}")
    print("-" * 86)
    print(f"{'PyTorch per-group':<40} {pt_results['norm']:<12.2f} {pt_results['apply']:<12.2f} {pt_results['total']:<12.2f} {'1.00x':<10}")
    if pt_compiled:
        print(f"{'PyTorch + torch.compile':<40} {pt_compiled['norm']:<12.2f} {pt_compiled['apply']:<12.2f} {pt_compiled['total']:<12.2f} {pt_results['total'] / pt_compiled['total']:.2f}x")
    if pt_vec_compiled:
        print(f"{'PyTorch vectorized (scatter_add)':<40} {pt_vec_compiled['norm']:<12.2f} {pt_vec_compiled['apply']:<12.2f} {pt_vec_compiled['total']:<12.2f} {pt_results['total'] / pt_vec_compiled['total']:.2f}x")
    print(f"{'Triton V2 single-block':<40} {triton_simple['norm']:<12.2f} {triton_simple['apply']:<12.2f} {triton_simple['total']:<12.2f} {pt_results['total'] / triton_simple['total']:.2f}x")
    print(f"{'Triton V2 multi-block':<40} {triton_multi['norm']:<12.2f} {triton_multi['apply']:<12.2f} {triton_multi['total']:<12.2f} {pt_results['total'] / triton_multi['total']:.2f}x")
    if cuda_results:
        print(f"{'CUDA V2':<40} {cuda_results['norm']:<12.2f} {cuda_results['apply']:<12.2f} {cuda_results['total']:<12.2f} {pt_results['total'] / cuda_results['total']:.2f}x")
    if cuda_v5_results:
        print(f"{'CUDA V5 (optimized)':<40} {cuda_v5_results['norm']:<12.2f} {cuda_v5_results['apply']:<12.2f} {cuda_v5_results['total']:<12.2f} {pt_results['total'] / cuda_v5_results['total']:.2f}x")
    if cuda_v3_results:
        print(f"{'CUDA V3 (atomic)':<40} {cuda_v3_results['norm']:<12.2f} {cuda_v3_results['apply']:<12.2f} {cuda_v3_results['total']:<12.2f} {pt_results['total'] / cuda_v3_results['total']:.2f}x")


if __name__ == '__main__':
    main()