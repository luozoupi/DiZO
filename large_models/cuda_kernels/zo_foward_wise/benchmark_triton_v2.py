"""
Benchmark: Optimized Triton V2 Kernels vs PyTorch vs CUDA V2

Compares:
1. PyTorch per-param (baseline) - matches original trainer.py DiZO implementation
2. PyTorch per-group with torch.compile
3. Triton V2 single-block (legacy)
4. Triton V2 multi-block (optimized)
5. CUDA V2 (if available)
6. CUDA V5 (if available)

Memory optimization:
- Uses lazy parameter generation to avoid OOM for large models
- Supports selecting specific GPU device
"""

import torch
import time
import argparse
import gc
import os
from typing import Dict, Tuple, List, Generator

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


# ============================================================================
# OPT Model Configuration
# ============================================================================
OPT_CONFIGS = {
    'opt-350m': {
        'layers': 24, 'hidden': 1024, 'ffn': 4096, 
        'embed_dim': 512,  # OPT-350m uses word_embed_proj_dim=512
        'vocab': 50272, 'max_pos': 2050,
        'has_project': True,  # Has project_in/project_out
    },
    'opt-1.3b': {
        'layers': 24, 'hidden': 2048, 'ffn': 8192,
        'embed_dim': 2048,  # No projection needed
        'vocab': 50272, 'max_pos': 2050,
        'has_project': False,
    },
    'opt-2.7b': {
        'layers': 32, 'hidden': 2560, 'ffn': 10240,
        'embed_dim': 2560,
        'vocab': 50272, 'max_pos': 2050,
        'has_project': False,
    },
    'opt-6.7b': {
        'layers': 32, 'hidden': 4096, 'ffn': 16384,
        'embed_dim': 4096,
        'vocab': 50272, 'max_pos': 2050,
        'has_project': False,
    },
    'opt-13b': {
        'layers': 40, 'hidden': 5120, 'ffn': 20480,
        'embed_dim': 5120,
        'vocab': 50272, 'max_pos': 2050,
        'has_project': False,
    },
}


def get_param_shapes(model_name: str) -> List[Tuple[str, Tuple[int, ...]]]:
    """Get parameter shapes matching REAL OPT model architecture.
    
    Returns list of (name, shape) tuples matching what model.named_parameters() returns.
    This mirrors the actual OPT model structure from HuggingFace.
    
    Based on actual OPT parameter shapes from profiling:
    - OPT-350m: 388 params, 331M elements (24 layers, hidden=1024, ffn=4096)
    - OPT-2.7b: ~450 params, 2.5B elements (32 layers, hidden=2560, ffn=10240)
    - OPT-6.7b: 516 params, 6.66B elements (32 layers, hidden=4096, ffn=16384)
    - OPT-13b: 644 params, 13B elements (40 layers, hidden=5120, ffn=20480)
    """
    config = OPT_CONFIGS.get(model_name, OPT_CONFIGS['opt-350m'])
    layers = config['layers']
    hidden = config['hidden']
    ffn = config['ffn']
    embed_dim = config['embed_dim']
    vocab = config['vocab']
    max_pos = config['max_pos']
    has_project = config['has_project']
    
    param_shapes = []
    
    # === Embedding layers ===
    # model.model.decoder.embed_tokens.weight
    param_shapes.append(('model.decoder.embed_tokens.weight', (vocab, embed_dim)))
    # model.model.decoder.embed_positions.weight  
    param_shapes.append(('model.decoder.embed_positions.weight', (max_pos, hidden)))
    
    # project_in/project_out for OPT-350m (only when embed_dim != hidden)
    if has_project:
        param_shapes.append(('model.decoder.project_in.weight', (hidden, embed_dim)))
        param_shapes.append(('model.decoder.project_out.weight', (embed_dim, hidden)))
    
    # === Transformer layers ===
    for i in range(layers):
        prefix = f'model.decoder.layers.{i}'
        
        # Self-attention: k, v, q, out projections (weight + bias each)
        # Note: OPT uses separate q, k, v projections, not fused QKV
        param_shapes.append((f'{prefix}.self_attn.k_proj.weight', (hidden, hidden)))
        param_shapes.append((f'{prefix}.self_attn.k_proj.bias', (hidden,)))
        param_shapes.append((f'{prefix}.self_attn.v_proj.weight', (hidden, hidden)))
        param_shapes.append((f'{prefix}.self_attn.v_proj.bias', (hidden,)))
        param_shapes.append((f'{prefix}.self_attn.q_proj.weight', (hidden, hidden)))
        param_shapes.append((f'{prefix}.self_attn.q_proj.bias', (hidden,)))
        param_shapes.append((f'{prefix}.self_attn.out_proj.weight', (hidden, hidden)))
        param_shapes.append((f'{prefix}.self_attn.out_proj.bias', (hidden,)))
        
        # self_attn_layer_norm (weight + bias)
        param_shapes.append((f'{prefix}.self_attn_layer_norm.weight', (hidden,)))
        param_shapes.append((f'{prefix}.self_attn_layer_norm.bias', (hidden,)))
        
        # FFN: fc1, fc2 (weight + bias each)
        param_shapes.append((f'{prefix}.fc1.weight', (ffn, hidden)))
        param_shapes.append((f'{prefix}.fc1.bias', (ffn,)))
        param_shapes.append((f'{prefix}.fc2.weight', (hidden, ffn)))
        param_shapes.append((f'{prefix}.fc2.bias', (hidden,)))
        
        # final_layer_norm (weight + bias)
        param_shapes.append((f'{prefix}.final_layer_norm.weight', (hidden,)))
        param_shapes.append((f'{prefix}.final_layer_norm.bias', (hidden,)))
    
    # Decoder final_layer_norm (weight + bias)
    param_shapes.append(('model.decoder.final_layer_norm.weight', (hidden,)))
    param_shapes.append(('model.decoder.final_layer_norm.bias', (hidden,)))
    
    # LM head (tied with embed_tokens for some models, but we include separately)
    param_shapes.append(('lm_head.weight', (vocab, embed_dim)))
    
    return param_shapes


def create_model_params_lazy(model_name: str, device: torch.device) -> Tuple[List[torch.Tensor], List[str], int]:
    """Create parameter tensors lazily to minimize peak memory usage.
    
    Returns:
        param_list: List of parameter tensors (each with original shape, not flattened)
        param_names: List of parameter names
        total_elements: Total number of elements across all parameters
    """
    param_shapes = get_param_shapes(model_name)
    
    param_list = []
    param_names = []
    total_elements = 0
    
    for name, shape in param_shapes:
        # Create tensor with random data
        param = torch.randn(shape, device=device, dtype=torch.float32)
        param_list.append(param)
        param_names.append(name)
        total_elements += param.numel()
    
    print(f"Created {len(param_list)} parameters with {total_elements:,} total elements")
    print(f"Estimated memory: {total_elements * 4 / 1024**3:.2f} GB (params only)")
    
    return param_list, param_names, total_elements


def create_flattened_params(model_name: str, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Create flattened parameter tensors for fused kernel benchmarks.
    
    This creates the flat representation needed by fused kernels while being
    memory-efficient by generating parameters incrementally.
    
    Returns:
        param_flat: Flattened parameter tensor
        anchor_flat: Flattened anchor tensor
        offsets: Tensor of offsets for each parameter group
        sizes: Tensor of sizes for each parameter group
        num_params: Number of parameter groups
    """
    param_shapes = get_param_shapes(model_name)
    
    # First pass: compute sizes and total elements
    sizes_list = []
    total_elements = 0
    for name, shape in param_shapes:
        numel = 1
        for dim in shape:
            numel *= dim
        sizes_list.append(numel)
        total_elements += numel
    
    print(f"Creating flattened tensors for {len(param_shapes)} params, {total_elements:,} elements")
    print(f"Estimated memory: {total_elements * 4 * 2 / 1024**3:.2f} GB (param + anchor)")
    
    # Allocate flat tensors
    param_flat = torch.empty(total_elements, device=device, dtype=torch.float32)
    anchor_flat = torch.empty(total_elements, device=device, dtype=torch.float32)
    
    # Fill incrementally to avoid peak memory from intermediate tensors
    offset = 0
    offsets_list = []
    for name, shape in param_shapes:
        numel = 1
        for dim in shape:
            numel *= dim
        offsets_list.append(offset)
        
        # Generate random data directly into the flat tensor
        param_flat[offset:offset+numel].normal_()
        anchor_flat[offset:offset+numel].normal_()
        offset += numel
    
    offsets = torch.tensor(offsets_list, device=device, dtype=torch.long)
    sizes = torch.tensor(sizes_list, device=device, dtype=torch.long)
    
    return param_flat, anchor_flat, offsets, sizes, len(param_shapes)



def benchmark_pytorch_per_param(
    param_list: List[torch.Tensor],
    anchor_list: List[torch.Tensor],
    constraints: torch.Tensor,
    n_iter: int = 10,
    warmup: int = 3,
) -> Dict[str, float]:
    """Benchmark PyTorch per-parameter operations - matches original trainer.py DiZO.
    
    This replicates the exact pattern from DiZO.apply_constraints() in trainer.py:
        for (name, new_para), anchor_para in zip(new.named_parameters(), pre_trained.parameters()):
            t = new_para.detach() - anchor_para.detach()
            norms = torch.norm(t)  # L2 norm
            ratio = constraint / (norms + 1e-8)
            v = (new_para.detach() - anchor_para.detach()) * ratio
            temp = v + anchor_para.detach()
            new_para.copy_(temp)
    """
    num_params = len(param_list)
    device = param_list[0].device
    
    # Warmup
    for _ in range(warmup):
        norms = torch.empty(num_params, device=device)
        for i, (param, anchor) in enumerate(zip(param_list, anchor_list)):
            t = param - anchor
            norms[i] = torch.norm(t)
    torch.cuda.synchronize()
    
    # Benchmark norm computation (matches DiZO._project_ratio)
    start = time.time()
    for _ in range(n_iter):
        norms = torch.empty(num_params, device=device)
        for i, (param, anchor) in enumerate(zip(param_list, anchor_list)):
            t = param - anchor
            norms[i] = torch.norm(t)  # L2 norm as in trainer.py
    torch.cuda.synchronize()
    norm_time = (time.time() - start) * 1000 / n_iter
    
    # Benchmark apply (matches DiZO.apply_constraints)
    start = time.time()
    for _ in range(n_iter):
        alphas = constraints / (norms + 1e-8)
        for i, (param, anchor) in enumerate(zip(param_list, anchor_list)):
            diff = param - anchor
            temp = anchor + diff * alphas[i]
            param.copy_(temp)
    torch.cuda.synchronize()
    apply_time = (time.time() - start) * 1000 / n_iter
    
    return {'norm': norm_time, 'apply': apply_time, 'total': norm_time + apply_time}


def benchmark_pytorch_per_group(
    param_flat: torch.Tensor,
    anchor_flat: torch.Tensor,
    offsets: torch.Tensor,
    sizes: torch.Tensor,
    constraints: torch.Tensor,
    n_iter: int = 10,
    warmup: int = 3,
) -> Dict[str, float]:
    """Benchmark PyTorch per-group operations using flattened tensors."""
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


def clear_gpu_memory():
    """Aggressively clear GPU memory."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark DiZO constraint operations: PyTorch vs Triton vs CUDA',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all backends on small model
  python benchmark_triton_v2.py --model opt-350m --backend all
  
  # Run realistic PyTorch baseline (matches trainer.py)
  python benchmark_triton_v2.py --model opt-13b --backend pytorch-real --gpu 4
  
  # Run CUDA V5 kernel on large model with specific GPU
  python benchmark_triton_v2.py --model opt-13b --backend cuda-v5 --gpu 5
  
  # Memory-efficient mode for large models
  python benchmark_triton_v2.py --model opt-13b --backend cuda-v5 --low-memory
"""
    )
    parser.add_argument('--model', type=str, default='opt-350m',
                        choices=['opt-350m', 'opt-1.3b', 'opt-2.7b', 'opt-6.7b', 'opt-13b'])
    parser.add_argument('--n_iter', type=int, default=10)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--backend', type=str, default='all',
                        choices=['all', 'pytorch-real', 'pytorch', 'pytorch-compile', 'pytorch-vec', 
                                 'triton-simple', 'triton-multi', 'cuda-v2', 'cuda-v3', 'cuda-v5'],
                        help='Backend to benchmark. pytorch-real matches trainer.py DiZO exactly.')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID to use (default: 0)')
    parser.add_argument('--low-memory', action='store_true',
                        help='Use memory-efficient mode for large models')
    args = parser.parse_args()
    
    # Set GPU device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    torch.cuda.set_device(0)  # After setting CUDA_VISIBLE_DEVICES, device 0 is the selected GPU
    device = torch.device('cuda:0')
    clear_gpu_memory()
    
    # Print GPU memory info
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    free_mem = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1024**3
    
    print(f"GPU {args.gpu}: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {free_mem:.1f} GB free / {total_mem:.1f} GB total")
    print(f"Model: {args.model}")
    print(f"Backend: {args.backend}")
    print(f"Iterations: {args.n_iter}")
    print(f"Low-memory mode: {args.low_memory}")
    print()
    
    # Get parameter info
    param_shapes = get_param_shapes(args.model)
    num_params = len(param_shapes)
    total_elements = sum(s[0] if len(s) == 1 else s[0] * s[1] for _, s in param_shapes)
    
    print(f"Parameter groups: {num_params}")
    print(f"Total elements: {total_elements:,}")
    print(f"Estimated memory for params + anchors: {total_elements * 4 * 2 / 1024**3:.2f} GB")
    print()
    
    # Track results for summary
    results = {}
    
    # Run benchmarks
    print("=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)
    
    # ========================================================================
    # PyTorch real baseline - matches trainer.py DiZO implementation exactly
    # ========================================================================
    pt_real_results = None
    if args.backend in ['all', 'pytorch-real']:
        print("\n[0] PyTorch per-param (matches trainer.py DiZO)...")
        
        # Create parameter list with original shapes
        param_list, param_names, _ = create_model_params_lazy(args.model, device)
        anchor_list = [torch.randn_like(p) for p in param_list]
        constraints = torch.rand(num_params, device=device) * 0.1
        
        pt_real_results = benchmark_pytorch_per_param(
            param_list, anchor_list, constraints,
            args.n_iter, args.warmup
        )
        print(f"    Norm:  {pt_real_results['norm']:.2f} ms")
        print(f"    Apply: {pt_real_results['apply']:.2f} ms")
        print(f"    Total: {pt_real_results['total']:.2f} ms")
        results['PyTorch per-param (trainer.py)'] = pt_real_results
        
        # Clean up
        del param_list, anchor_list
        clear_gpu_memory()
    
    # ========================================================================
    # For fused kernel benchmarks, we need flattened tensors
    # ========================================================================
    param_flat = None
    anchor_flat = None
    offsets = None
    sizes = None
    kernels = None
    
    # Only create flattened tensors if needed
    needs_flat = args.backend in ['all', 'pytorch', 'pytorch-compile', 'pytorch-vec', 
                                   'triton-simple', 'triton-multi', 'cuda-v2', 'cuda-v3', 'cuda-v5']
    
    # Determine if we need to clone tensors (only when running multiple backends)
    # When running a single backend, we can avoid the clone to save ~50% memory
    needs_clone = args.backend == 'all'
    
    if needs_flat:
        print("\nCreating flattened tensors for fused kernel benchmarks...")
        param_flat, anchor_flat, offsets, sizes, _ = create_flattened_params(args.model, device)
        constraints = torch.rand(num_params, device=device) * 0.1
        
        # Initialize Triton kernels if needed
        if args.backend in ['all', 'triton-simple', 'triton-multi']:
            print("Initializing Triton V2 kernels...")
            kernels = FusedDiZOKernelsV2(num_params, total_elements, device, offsets, sizes)
        print()
    
    # Helper function to get param tensor (clone only if needed)
    def get_param_tensor():
        return param_flat.clone() if needs_clone else param_flat
    
    # PyTorch per-group (flat tensors, for comparison)
    pt_results = None
    if args.backend in ['all', 'pytorch']:
        print("\n[1] PyTorch per-group (flattened tensors)...")
        pt_results = benchmark_pytorch_per_group(
            get_param_tensor(), anchor_flat, offsets, sizes, constraints,
            args.n_iter, args.warmup
        )
        print(f"    Norm:  {pt_results['norm']:.2f} ms")
        print(f"    Apply: {pt_results['apply']:.2f} ms")
        print(f"    Total: {pt_results['total']:.2f} ms")
        results['PyTorch per-group'] = pt_results
        clear_gpu_memory()
    
    # PyTorch with torch.compile (per-group)
    pt_compiled = None
    if args.backend in ['all', 'pytorch-compile']:
        print("\n[2] PyTorch per-group + torch.compile...")
        try:
            pt_compiled = benchmark_pytorch_compiled(
                get_param_tensor(), anchor_flat, offsets, sizes, constraints,
                args.n_iter, args.warmup
            )
            print(f"    Norm:  {pt_compiled['norm']:.2f} ms")
            print(f"    Apply: {pt_compiled['apply']:.2f} ms")
            print(f"    Total: {pt_compiled['total']:.2f} ms")
            if pt_results:
                print(f"    Speedup vs PyTorch: {pt_results['total'] / pt_compiled['total']:.2f}x")
            results['PyTorch + torch.compile'] = pt_compiled
        except Exception as e:
            print(f"    Failed: {e}")
        
        # Clear memory after torch.compile
        torch._dynamo.reset()
        clear_gpu_memory()
    
    # PyTorch vectorized (without torch.compile for fair comparison)
    pt_vec_compiled = None
    if args.backend in ['all', 'pytorch-vec']:
        print("\n[3] PyTorch vectorized (scatter_add)...")
        try:
            pt_vec_compiled = benchmark_pytorch_vectorized(
                get_param_tensor(), anchor_flat, offsets, sizes, constraints,
                args.n_iter, args.warmup
            )
            print(f"    Norm:  {pt_vec_compiled['norm']:.2f} ms")
            print(f"    Apply: {pt_vec_compiled['apply']:.2f} ms")
            print(f"    Total: {pt_vec_compiled['total']:.2f} ms")
            if pt_results:
                print(f"    Speedup vs PyTorch: {pt_results['total'] / pt_vec_compiled['total']:.2f}x")
            results['PyTorch vectorized'] = pt_vec_compiled
        except Exception as e:
            print(f"    Failed: {e}")
        clear_gpu_memory()
    
    # Triton single-block
    triton_simple = None
    if args.backend in ['all', 'triton-simple']:
        print("\n[4] Triton V2 single-block...")
        triton_simple = benchmark_triton_simple(
            kernels, get_param_tensor(), anchor_flat, offsets, sizes, constraints,
            args.n_iter, args.warmup
        )
        print(f"    Norm:  {triton_simple['norm']:.2f} ms")
        print(f"    Apply: {triton_simple['apply']:.2f} ms")
        print(f"    Total: {triton_simple['total']:.2f} ms")
        if pt_real_results:
            print(f"    Speedup vs PyTorch (trainer.py): {pt_real_results['total'] / triton_simple['total']:.2f}x")
        elif pt_results:
            print(f"    Speedup vs PyTorch (flat): {pt_results['total'] / triton_simple['total']:.2f}x")
        results['Triton V2 single-block'] = triton_simple
        clear_gpu_memory()
    
    # Triton multi-block
    triton_multi = None
    if args.backend in ['all', 'triton-multi']:
        print("\n[5] Triton V2 multi-block (optimized)...")
        triton_multi = benchmark_triton_multiblock(
            kernels, get_param_tensor(), anchor_flat, offsets, sizes, constraints,
            args.n_iter, args.warmup
        )
        print(f"    Norm:  {triton_multi['norm']:.2f} ms")
        print(f"    Apply: {triton_multi['apply']:.2f} ms")
        print(f"    Total: {triton_multi['total']:.2f} ms")
        if pt_real_results:
            print(f"    Speedup vs PyTorch (trainer.py): {pt_real_results['total'] / triton_multi['total']:.2f}x")
        elif pt_results:
            print(f"    Speedup vs PyTorch (flat): {pt_results['total'] / triton_multi['total']:.2f}x")
        if triton_simple:
            print(f"    Speedup vs Triton simple: {triton_simple['total'] / triton_multi['total']:.2f}x")
        results['Triton V2 multi-block'] = triton_multi
        clear_gpu_memory()
    
    # CUDA V2
    cuda_results = None
    if CUDA_V2_AVAILABLE and args.backend in ['all', 'cuda-v2']:
        print("\n[6] CUDA V2...")
        cuda_results = benchmark_cuda_v2(
            get_param_tensor(), anchor_flat, offsets, sizes, constraints,
            args.n_iter, args.warmup
        )
        print(f"    Norm:  {cuda_results['norm']:.2f} ms")
        print(f"    Apply: {cuda_results['apply']:.2f} ms")
        print(f"    Total: {cuda_results['total']:.2f} ms")
        if pt_real_results:
            print(f"    Speedup vs PyTorch (trainer.py): {pt_real_results['total'] / cuda_results['total']:.2f}x")
        elif pt_results:
            print(f"    Speedup vs PyTorch (flat): {pt_results['total'] / cuda_results['total']:.2f}x")
        if triton_multi:
            print(f"    Triton multi vs CUDA V2: {cuda_results['total'] / triton_multi['total']:.2f}x")
        results['CUDA V2'] = cuda_results
        clear_gpu_memory()
    
    # CUDA V5
    cuda_v5_results = None
    if CUDA_V5_AVAILABLE and args.backend in ['all', 'cuda-v5']:
        print("\n[7] CUDA V5 (optimized multi-block)...")
        cuda_v5_results = benchmark_cuda_v5(
            param_flat, anchor_flat, offsets, sizes, constraints,
            args.n_iter, args.warmup
        )
        print(f"    Norm:  {cuda_v5_results['norm']:.2f} ms")
        print(f"    Apply: {cuda_v5_results['apply']:.2f} ms")
        print(f"    Total: {cuda_v5_results['total']:.2f} ms")
        if pt_real_results:
            print(f"    Speedup vs PyTorch (trainer.py): {pt_real_results['total'] / cuda_v5_results['total']:.2f}x")
        elif pt_results:
            print(f"    Speedup vs PyTorch (flat): {pt_results['total'] / cuda_v5_results['total']:.2f}x")
        if cuda_results:
            print(f"    Speedup vs CUDA V2: {cuda_results['total'] / cuda_v5_results['total']:.2f}x")
        results['CUDA V5'] = cuda_v5_results
        clear_gpu_memory()
    
    # CUDA V3
    cuda_v3_results = None
    if CUDA_V3_AVAILABLE and args.backend in ['all', 'cuda-v3']:
        print("\n[8] CUDA V3 (atomic multi-block)...")
        cuda_v3_results = benchmark_cuda_v3(
            get_param_tensor(), anchor_flat, offsets, sizes, constraints,
            args.n_iter, args.warmup
        )
        print(f"    Norm:  {cuda_v3_results['norm']:.2f} ms")
        print(f"    Apply: {cuda_v3_results['apply']:.2f} ms")
        print(f"    Total: {cuda_v3_results['total']:.2f} ms")
        if pt_real_results:
            print(f"    Speedup vs PyTorch (trainer.py): {pt_real_results['total'] / cuda_v3_results['total']:.2f}x")
        results['CUDA V3'] = cuda_v3_results
        clear_gpu_memory()
    
    # Summary
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    
    if not results:
        print("No benchmarks were run.")
        return
    
    # Get baseline for speedup calculation - prefer the realistic PyTorch baseline
    if pt_real_results:
        baseline_total = pt_real_results['total']
        baseline_name = 'PyTorch per-param (trainer.py)'
    elif pt_results:
        baseline_total = pt_results['total']
        baseline_name = 'PyTorch per-group'
    else:
        baseline_total = None
        baseline_name = None
    
    if baseline_name:
        print(f"Baseline: {baseline_name}")
        print()
    
    print(f"{'Method':<42} {'Norm (ms)':<12} {'Apply (ms)':<12} {'Total (ms)':<12} {'Speedup':<10}")
    print("-" * 90)
    
    for name, res in results.items():
        if baseline_total and res:
            speedup = f"{baseline_total / res['total']:.2f}x"
        else:
            speedup = "N/A"
        if name == baseline_name:
            speedup = '1.00x (baseline)'
        print(f"{name:<42} {res['norm']:<12.2f} {res['apply']:<12.2f} {res['total']:<12.2f} {speedup:<10}")
    
    print("\n" + "=" * 90)
    print("USAGE TIPS")
    print("=" * 90)
    print("For large models (OPT-13B+), select GPU and use specific backend:")
    print("  python benchmark_triton_v2.py --model opt-13b --backend pytorch-real --gpu 4")
    print("  python benchmark_triton_v2.py --model opt-13b --backend cuda-v5 --gpu 5")
    print("\nTo compare against realistic baseline (matches trainer.py DiZO):")
    print("  python benchmark_triton_v2.py --model opt-6.7b --backend pytorch-real")


if __name__ == '__main__':
    main()