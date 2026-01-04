#!/usr/bin/env python3
"""
Diagnostic script to identify constraint bottleneck in DiZO kernels for large models.

This script:
1. Verifies the constraint kernel usage in benchmark_full_training_step_v2.py
2. Compares with individual benchmark scripts
3. Identifies the root cause of slowdown for large models (OPT-13B)

Key issue identified:
- benchmark_full_training_step_v2.py may be using inefficient constraint implementation
- For large models, the constraint operation scales with number of parameters AND parameter sizes
- Block mapping overhead becomes significant for OPT-13B (~644 params, ~6400 blocks)

Usage:
    CUDA_VISIBLE_DEVICES=5 python diagnose_constraint_bottleneck.py --model opt-350m
    CUDA_VISIBLE_DEVICES=5 python diagnose_constraint_bottleneck.py --model opt-13b
"""

import torch
import time
import gc
import argparse
import os
import sys

# Add kernel paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'Perturb_wise'))
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'zo_foward_wise'))

# Model configs
MODEL_CONFIGS = {
    'opt-350m': {
        'num_layers': 24,
        'hidden_size': 1024,
        'ffn_size': 4096,
        'embed_dim': 512,
        'has_project': True,
        'total_params': 331_196_416,
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


def create_param_structure(config, device):
    """Create parameter structure matching OPT model exactly."""
    sizes_list = []
    num_layers = config['num_layers']
    hidden = config['hidden_size']
    ffn = config['ffn_size']
    embed_dim = config.get('embed_dim', hidden)
    has_project = config.get('has_project', False)
    vocab_size = 50272
    max_pos = 2050
    
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
    num_params = len(sizes_list)
    
    # Compute offsets
    offsets_list = []
    offset = 0
    for size in sizes_list:
        offsets_list.append(offset)
        offset += size
    
    # Create tensors
    param_flat = torch.randn(total_elements, device=device, dtype=torch.float32)
    anchor_flat = torch.randn(total_elements, device=device, dtype=torch.float32)
    offsets = torch.tensor(offsets_list, device=device, dtype=torch.long)
    sizes = torch.tensor(sizes_list, device=device, dtype=torch.long)
    constraints = torch.rand(num_params, device=device) * 0.1
    
    return param_flat, anchor_flat, offsets, sizes, constraints, num_params, total_elements


def benchmark_pytorch_baseline(param_flat, anchor_flat, offsets, sizes, constraints, n_iter=10):
    """PyTorch baseline - per-parameter loop (matches trainer.py)."""
    num_params = len(offsets)
    
    # Create views into flat tensors
    param_views = []
    anchor_views = []
    for i in range(num_params):
        start = offsets[i].item()
        end = start + sizes[i].item()
        param_views.append(param_flat[start:end])
        anchor_views.append(anchor_flat[start:end])
    
    gammas = [torch.tensor([0.1], device=param_flat.device) for _ in range(num_params)]
    
    # Warmup
    for _ in range(3):
        norms = [torch.norm(p - a) for p, a in zip(param_views, anchor_views)]
        for i, (p, a, g, n) in enumerate(zip(param_views, anchor_views, gammas, norms)):
            alpha = g / (n + 1e-8)
            p.data.copy_(a + (p - a) * alpha)
    torch.cuda.synchronize()
    
    # Benchmark norm computation
    start = time.time()
    for _ in range(n_iter):
        norms = [torch.norm(p - a) for p, a in zip(param_views, anchor_views)]
    torch.cuda.synchronize()
    norm_time = (time.time() - start) * 1000 / n_iter
    
    # Benchmark apply constraints
    start = time.time()
    for _ in range(n_iter):
        for i, (p, a, g, n) in enumerate(zip(param_views, anchor_views, gammas, norms)):
            alpha = g / (n + 1e-8)
            p.data.copy_(a + (p - a) * alpha)
    torch.cuda.synchronize()
    apply_time = (time.time() - start) * 1000 / n_iter
    
    # Benchmark reverse constraints
    start = time.time()
    for _ in range(n_iter):
        for i, (p, a, g, n) in enumerate(zip(param_views, anchor_views, gammas, norms)):
            alpha = g / (n + 1e-8)
            p.data.copy_(a + (p - a) / alpha)
    torch.cuda.synchronize()
    reverse_time = (time.time() - start) * 1000 / n_iter
    
    return {
        'norm': norm_time,
        'apply': apply_time,
        'reverse': reverse_time,
        'total': norm_time + apply_time + reverse_time,
    }


def benchmark_triton_v2(param_flat, anchor_flat, offsets, sizes, constraints, n_iter=10):
    """Triton V2 multi-block kernels."""
    try:
        from dizo_fused_kernels_v2 import FusedDiZOKernelsV2
    except ImportError as e:
        print(f"Triton V2 not available: {e}")
        return None
    
    num_params = len(offsets)
    total_elements = param_flat.numel()
    device = param_flat.device
    
    # Initialize kernels
    zo_kernels = FusedDiZOKernelsV2(num_params, total_elements, device, offsets, sizes)
    
    # Warmup
    for _ in range(3):
        norms = zo_kernels.compute_norms(param_flat, anchor_flat, offsets, sizes)
        alphas = zo_kernels.apply_constraints(param_flat, anchor_flat, offsets, sizes, constraints, norms)
        zo_kernels.reverse_constraints(param_flat, anchor_flat, offsets, sizes, alphas)
    torch.cuda.synchronize()
    
    # Benchmark norm computation
    start = time.time()
    for _ in range(n_iter):
        norms = zo_kernels.compute_norms(param_flat, anchor_flat, offsets, sizes)
    torch.cuda.synchronize()
    norm_time = (time.time() - start) * 1000 / n_iter
    
    # Benchmark apply constraints
    start = time.time()
    for _ in range(n_iter):
        alphas = zo_kernels.apply_constraints(param_flat, anchor_flat, offsets, sizes, constraints, norms)
    torch.cuda.synchronize()
    apply_time = (time.time() - start) * 1000 / n_iter
    
    # Benchmark reverse constraints
    start = time.time()
    for _ in range(n_iter):
        zo_kernels.reverse_constraints(param_flat, anchor_flat, offsets, sizes, alphas)
    torch.cuda.synchronize()
    reverse_time = (time.time() - start) * 1000 / n_iter
    
    return {
        'norm': norm_time,
        'apply': apply_time,
        'reverse': reverse_time,
        'total': norm_time + apply_time + reverse_time,
        'num_blocks': zo_kernels.num_blocks,
    }


def benchmark_cuda_v5(param_flat, anchor_flat, offsets, sizes, constraints, n_iter=10):
    """CUDA V5 multi-block kernels."""
    try:
        import dizo_fused_kernels_cuda_v5 as cuda_v5
    except ImportError as e:
        print(f"CUDA V5 not available: {e}")
        return None
    
    num_params = len(offsets)
    
    # Initialize block mapping
    cuda_v5.init_block_mapping(sizes)
    
    # Warmup
    for _ in range(3):
        norms = cuda_v5.fused_compute_norms(param_flat, anchor_flat, offsets, sizes)
        alphas = constraints / (norms + 1e-8)
        cuda_v5.fused_apply_constraints(param_flat, anchor_flat, offsets, sizes, alphas, norms, 1e-8)
        cuda_v5.fused_reverse_constraints(param_flat, anchor_flat, offsets, sizes, alphas)
    torch.cuda.synchronize()
    
    # Benchmark norm computation
    start = time.time()
    for _ in range(n_iter):
        norms = cuda_v5.fused_compute_norms(param_flat, anchor_flat, offsets, sizes)
    torch.cuda.synchronize()
    norm_time = (time.time() - start) * 1000 / n_iter
    
    # Benchmark apply constraints
    alphas = constraints / (norms + 1e-8)
    start = time.time()
    for _ in range(n_iter):
        cuda_v5.fused_apply_constraints(param_flat, anchor_flat, offsets, sizes, alphas, norms, 1e-8)
    torch.cuda.synchronize()
    apply_time = (time.time() - start) * 1000 / n_iter
    
    # Benchmark reverse constraints
    start = time.time()
    for _ in range(n_iter):
        cuda_v5.fused_reverse_constraints(param_flat, anchor_flat, offsets, sizes, alphas)
    torch.cuda.synchronize()
    reverse_time = (time.time() - start) * 1000 / n_iter
    
    # Get num_blocks
    num_blocks = cuda_v5.get_num_blocks()
    
    return {
        'norm': norm_time,
        'apply': apply_time,
        'reverse': reverse_time,
        'total': norm_time + apply_time + reverse_time,
        'num_blocks': num_blocks,
    }


def analyze_block_mapping(sizes, BLOCK_SIZE=2048):
    """Analyze block mapping overhead."""
    sizes_list = sizes.tolist()
    num_params = len(sizes_list)
    
    total_blocks = 0
    max_blocks_per_param = 0
    large_params = 0
    
    for size in sizes_list:
        blocks = (size + BLOCK_SIZE - 1) // BLOCK_SIZE
        total_blocks += blocks
        max_blocks_per_param = max(max_blocks_per_param, blocks)
        if blocks > 1:
            large_params += 1
    
    return {
        'num_params': num_params,
        'total_blocks': total_blocks,
        'avg_blocks_per_param': total_blocks / num_params,
        'max_blocks_per_param': max_blocks_per_param,
        'large_params': large_params,
        'block_mapping_size': total_blocks * 2 * 4,  # 2 arrays, 4 bytes each
    }


def main():
    parser = argparse.ArgumentParser(description='Diagnose constraint bottleneck')
    parser.add_argument('--model', type=str, default='opt-350m',
                        choices=['opt-350m', 'opt-2.7b', 'opt-6.7b', 'opt-13b'])
    parser.add_argument('--n_iter', type=int, default=20)
    args = parser.parse_args()
    
    print("=" * 80)
    print(f"CONSTRAINT BOTTLENECK DIAGNOSIS: {args.model.upper()}")
    print("=" * 80)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print()
    
    config = MODEL_CONFIGS[args.model]
    device = torch.device('cuda')
    
    # Create tensors
    print(f"Creating parameter tensors for {args.model}...")
    param_flat, anchor_flat, offsets, sizes, constraints, num_params, total_elements = \
        create_param_structure(config, device)
    
    print(f"  Parameters: {num_params}")
    print(f"  Total elements: {total_elements:,} ({total_elements * 4 / 1e9:.2f} GB)")
    print()
    
    # Analyze block mapping
    print("Block Mapping Analysis:")
    analysis = analyze_block_mapping(sizes)
    print(f"  Total blocks: {analysis['total_blocks']:,}")
    print(f"  Avg blocks per param: {analysis['avg_blocks_per_param']:.2f}")
    print(f"  Max blocks per param: {analysis['max_blocks_per_param']}")
    print(f"  Params with >1 block: {analysis['large_params']}")
    print(f"  Block mapping size: {analysis['block_mapping_size'] / 1024:.2f} KB")
    print()
    
    # Benchmark PyTorch baseline
    print("Benchmarking PyTorch baseline (per-param loop)...")
    gc.collect()
    torch.cuda.empty_cache()
    pytorch_results = benchmark_pytorch_baseline(param_flat, anchor_flat, offsets, sizes, constraints, args.n_iter)
    print(f"  Norm: {pytorch_results['norm']:.3f} ms")
    print(f"  Apply: {pytorch_results['apply']:.3f} ms")
    print(f"  Reverse: {pytorch_results['reverse']:.3f} ms")
    print(f"  Total: {pytorch_results['total']:.3f} ms")
    print()
    
    # Benchmark Triton V2
    print("Benchmarking Triton V2 (multi-block)...")
    gc.collect()
    torch.cuda.empty_cache()
    triton_results = benchmark_triton_v2(param_flat, anchor_flat, offsets, sizes, constraints, args.n_iter)
    if triton_results:
        print(f"  Norm: {triton_results['norm']:.3f} ms")
        print(f"  Apply: {triton_results['apply']:.3f} ms")
        print(f"  Reverse: {triton_results['reverse']:.3f} ms")
        print(f"  Total: {triton_results['total']:.3f} ms")
        print(f"  Num blocks: {triton_results['num_blocks']}")
        print()
    
    # Benchmark CUDA V5
    print("Benchmarking CUDA V5 (multi-block)...")
    gc.collect()
    torch.cuda.empty_cache()
    cuda_results = benchmark_cuda_v5(param_flat, anchor_flat, offsets, sizes, constraints, args.n_iter)
    if cuda_results:
        print(f"  Norm: {cuda_results['norm']:.3f} ms")
        print(f"  Apply: {cuda_results['apply']:.3f} ms")
        print(f"  Reverse: {cuda_results['reverse']:.3f} ms")
        print(f"  Total: {cuda_results['total']:.3f} ms")
        print(f"  Num blocks: {cuda_results['num_blocks']}")
        print()
    
    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Method':<25} {'Norm':>10} {'Apply':>10} {'Reverse':>10} {'Total':>10} {'Speedup':>10}")
    print("-" * 75)
    
    baseline_total = pytorch_results['total']
    print(f"{'PyTorch Baseline':<25} {pytorch_results['norm']:>10.3f} {pytorch_results['apply']:>10.3f} {pytorch_results['reverse']:>10.3f} {pytorch_results['total']:>10.3f} {'1.00x':>10}")
    
    if triton_results:
        speedup = baseline_total / triton_results['total']
        print(f"{'Triton V2 Multi-block':<25} {triton_results['norm']:>10.3f} {triton_results['apply']:>10.3f} {triton_results['reverse']:>10.3f} {triton_results['total']:>10.3f} {speedup:>9.2f}x")
    
    if cuda_results:
        speedup = baseline_total / cuda_results['total']
        print(f"{'CUDA V5 Multi-block':<25} {cuda_results['norm']:>10.3f} {cuda_results['apply']:>10.3f} {cuda_results['reverse']:>10.3f} {cuda_results['total']:>10.3f} {speedup:>9.2f}x")
    
    print()
    print("DIAGNOSIS:")
    
    # Check if kernels are slower than baseline
    if triton_results and triton_results['total'] > baseline_total:
        print("⚠ Triton V2 is SLOWER than PyTorch baseline!")
        print("  Possible causes:")
        print("  1. Block mapping overhead (computed on every norm call)")
        print("  2. Atomic reduction contention for large number of blocks")
        print("  3. Memory bandwidth saturation")
    
    if cuda_results and cuda_results['total'] > baseline_total:
        print("⚠ CUDA V5 is SLOWER than PyTorch baseline!")
        print("  Possible causes:")
        print("  1. Two-phase reduction overhead (partial_sums → final reduction)")
        print("  2. Block mapping not cached properly")
        print("  3. Kernel launch overhead for many blocks")
    
    # Suggest optimizations
    print()
    print("SUGGESTED OPTIMIZATIONS:")
    print("  1. Cache block mapping (only compute once per model)")
    print("  2. Use atomic reduction instead of two-phase for norm")
    print("  3. Fuse norm + apply into single kernel to reduce memory passes")
    print("  4. Use vectorized operations (float4) more aggressively")
    print("  5. Consider stream-based parallelism for independent params")


if __name__ == '__main__':
    main()
