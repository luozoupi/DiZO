#!/usr/bin/env python3
"""
Accurate benchmark for full MeZO training step with detailed timing breakdown.

This script measures ACTUAL timing for each operation rather than using estimates.
"""

import torch
import time
import gc
import argparse
import os
import sys
import numpy as np

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


# Import Triton perturb kernels
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

# Import Triton ZO V2 kernels
try:
    from dizo_fused_kernels_v2 import FusedDiZOKernelsV2
    HAS_TRITON_ZO_V2 = True
except ImportError as e:
    HAS_TRITON_ZO_V2 = False
    print(f"Triton ZO V2 not available: {e}")

# Import CUDA ZO V5 kernels
try:
    import dizo_fused_kernels_cuda_v5 as cuda_zo_v5
    HAS_CUDA_ZO_V5 = True
except ImportError as e:
    HAS_CUDA_ZO_V5 = False
    print(f"CUDA ZO V5 not available: {e}")

# Import CUDA perturb kernels
try:
    import fused_perturb_cuda
    HAS_CUDA_PERTURB = True
except ImportError as e:
    HAS_CUDA_PERTURB = False
    print(f"CUDA Perturb not available: {e}")

# Define Triton runtime-seed kernels
HAS_TRITON_RUNTIME_SEED = False
if HAS_TRITON:
    @triton.autotune(
        configs=[
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
        seed_ptr,
        alpha,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        seed = tl.load(seed_ptr)
        pid = tl.program_id(0)
        block_start = pid.to(tl.int64) * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE).to(tl.int64)
        mask = offsets < n_elements
        params = tl.load(params_ptr + offsets, mask=mask, other=0.0)
        z = tl.randn(seed, offsets)
        result = params + alpha * z
        tl.store(params_ptr + offsets, result, mask=mask)
    
    @triton.autotune(
        configs=[
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
        seed_ptr,
        projected_grad,
        lr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        seed = tl.load(seed_ptr)
        pid = tl.program_id(0)
        block_start = pid.to(tl.int64) * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE).to(tl.int64)
        mask = offsets < n_elements
        params = tl.load(params_ptr + offsets, mask=mask, other=0.0)
        z = tl.randn(seed, offsets)
        result = params - lr * projected_grad * z
        tl.store(params_ptr + offsets, result, mask=mask)
    
    HAS_TRITON_RUNTIME_SEED = True


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


def dummy_forward():
    """Minimal forward pass simulation."""
    x = torch.randn(256, 256, device='cuda')
    _ = x @ x.T
    return 1.0 + np.random.rand() * 0.1


class TimingRecorder:
    """CUDA event-based timing recorder."""
    def __init__(self):
        self.events = {}
        
    def start(self, name):
        if name not in self.events:
            self.events[name] = {'times': [], 'start': None, 'end': None}
        self.events[name]['start'] = torch.cuda.Event(enable_timing=True)
        self.events[name]['end'] = torch.cuda.Event(enable_timing=True)
        self.events[name]['start'].record()
    
    def stop(self, name):
        self.events[name]['end'].record()
        
    def finalize(self):
        torch.cuda.synchronize()
        for name in self.events:
            if self.events[name]['start'] is not None and self.events[name]['end'] is not None:
                elapsed = self.events[name]['start'].elapsed_time(self.events[name]['end'])
                self.events[name]['times'].append(elapsed)
    
    def get_mean(self, name):
        if name in self.events and len(self.events[name]['times']) > 0:
            return np.mean(self.events[name]['times'])
        return 0.0
    
    def get_std(self, name):
        if name in self.events and len(self.events[name]['times']) > 1:
            return np.std(self.events[name]['times'])
        return 0.0


def benchmark_full_step_accurate(
    param_flat, anchor_flat, offsets, sizes, constraints,
    n_iter=10, warmup=3, eps=1e-3, lr=1e-5, tau=0.2, zo_eps=0.1, step_size=2.0,
    use_cuda_perturb=False, use_cuda_zo=False, include_dizo=True
):
    """
    Full MeZO + DiZO training step with accurate per-operation timing.
    
    Args:
        include_dizo: If False, skip DiZO constraint operations (MeZO-only mode).
    """
    device = param_flat.device
    n_elements = param_flat.numel()
    num_params = len(offsets)
    
    # Initialize ZO kernels (only if DiZO is enabled)
    zo_mode = "None"
    zo_kernels = None
    if include_dizo:
        if use_cuda_zo and HAS_CUDA_ZO_V5:
            cuda_zo_v5.init_block_mapping(sizes)
            zo_mode = "CUDA V5"
        elif HAS_TRITON_ZO_V2:
            zo_kernels = FusedDiZOKernelsV2(num_params, n_elements, device, offsets, sizes)
            zo_mode = "Triton V2"
        else:
            raise RuntimeError("No ZO kernels available!")
    
    # Setup perturb kernels
    if use_cuda_perturb and HAS_CUDA_PERTURB:
        perturb_mode = "CUDA"
    elif HAS_TRITON_RUNTIME_SEED:
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        seed_tensor = torch.tensor([0], dtype=torch.int64, device=device)
        perturb_mode = "Triton"
    else:
        raise RuntimeError("No perturb kernels available!")
    
    mode_str = f"perturb: {perturb_mode}"
    if include_dizo:
        mode_str += f", ZO: {zo_mode}"
    else:
        mode_str += " (MeZO only, no DiZO)"
    print(f"Using {mode_str}")
    
    # Warmup
    print(f"Warming up ({warmup} iterations)...")
    for _ in range(warmup):
        seed = 42
        if perturb_mode == "CUDA":
            fused_perturb_cuda.fused_perturb(param_flat, seed, eps)
        else:
            seed_tensor[0] = seed
            fused_perturb_runtime_seed[grid](param_flat, seed_tensor, eps, n_elements)
        
        if include_dizo:
            if zo_mode == "CUDA V5":
                norms = cuda_zo_v5.fused_compute_norms(param_flat, anchor_flat, offsets, sizes)
                alphas = constraints / (norms + 1e-8)
                cuda_zo_v5.fused_apply_constraints(param_flat, anchor_flat, offsets, sizes, alphas, norms, 1e-8)
                cuda_zo_v5.fused_reverse_constraints(param_flat, anchor_flat, offsets, sizes, alphas)
            else:
                norms = zo_kernels.compute_norms(param_flat, anchor_flat, offsets, sizes)
                alphas = zo_kernels.apply_constraints(param_flat, anchor_flat, offsets, sizes, constraints, norms)
                zo_kernels.reverse_constraints(param_flat, anchor_flat, offsets, sizes, alphas)
                zo_kernels.perturb_gamma(constraints, norms, 1.0, tau, zo_eps, generate_new=True)
                zo_kernels.update_gamma(constraints, norms, 0.001, step_size, tau)
        
        if perturb_mode == "CUDA":
            fused_perturb_cuda.fused_update(param_flat, seed, 0.001, lr)
        else:
            fused_update_runtime_seed[grid](param_flat, seed_tensor, 0.001, lr, n_elements)
    
    torch.cuda.synchronize()
    
    # Reset param_flat
    param_flat.copy_(anchor_flat)
    torch.cuda.synchronize()
    
    # Benchmark with detailed timing
    print(f"Benchmarking ({n_iter} iterations)...")
    timings = {
        'perturb1': [],
        'forward1': [],
        'norm': [],
        'apply': [],
        'perturb2': [],
        'forward2': [],
        'reverse': [],
        'perturb3': [],
        'update': [],
        'gamma_perturb': [],
        'gamma_update': [],
        'total': [],
    }
    
    for i in range(n_iter):
        seed = np.random.randint(1000000000)
        if perturb_mode == "Triton":
            seed_tensor[0] = seed
        
        # Total timing
        total_start = torch.cuda.Event(enable_timing=True)
        total_end = torch.cuda.Event(enable_timing=True)
        
        total_start.record()
        
        # === Perturb +eps ===
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        if perturb_mode == "CUDA":
            fused_perturb_cuda.fused_perturb(param_flat, seed, eps)
        else:
            fused_perturb_runtime_seed[grid](param_flat, seed_tensor, eps, n_elements)
        end.record()
        torch.cuda.synchronize()
        timings['perturb1'].append(start.elapsed_time(end))
        
        # === Forward 1 ===
        start.record()
        loss1 = dummy_forward()
        end.record()
        torch.cuda.synchronize()
        timings['forward1'].append(start.elapsed_time(end))
        
        # === DiZO: Compute norms + Apply constraints ===
        if include_dizo:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            if zo_mode == "CUDA V5":
                norms = cuda_zo_v5.fused_compute_norms(param_flat, anchor_flat, offsets, sizes)
            else:
                norms = zo_kernels.compute_norms(param_flat, anchor_flat, offsets, sizes)
            end.record()
            torch.cuda.synchronize()
            timings['norm'].append(start.elapsed_time(end))
            
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            if zo_mode == "CUDA V5":
                alphas = constraints / (norms + 1e-8)
                cuda_zo_v5.fused_apply_constraints(param_flat, anchor_flat, offsets, sizes, alphas, norms, 1e-8)
            else:
                alphas = zo_kernels.apply_constraints(param_flat, anchor_flat, offsets, sizes, constraints, norms)
            end.record()
            torch.cuda.synchronize()
            timings['apply'].append(start.elapsed_time(end))
        else:
            timings['norm'].append(0.0)
            timings['apply'].append(0.0)
        
        # === Perturb -2eps ===
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        if perturb_mode == "CUDA":
            fused_perturb_cuda.fused_perturb(param_flat, seed, -2*eps)
        else:
            fused_perturb_runtime_seed[grid](param_flat, seed_tensor, -2*eps, n_elements)
        end.record()
        torch.cuda.synchronize()
        timings['perturb2'].append(start.elapsed_time(end))
        
        # === Forward 2 ===
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        loss2 = dummy_forward()
        end.record()
        torch.cuda.synchronize()
        timings['forward2'].append(start.elapsed_time(end))
        
        # === DiZO: Reverse constraints ===
        if include_dizo:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            if zo_mode == "CUDA V5":
                cuda_zo_v5.fused_reverse_constraints(param_flat, anchor_flat, offsets, sizes, alphas)
            else:
                zo_kernels.reverse_constraints(param_flat, anchor_flat, offsets, sizes, alphas)
            end.record()
            torch.cuda.synchronize()
            timings['reverse'].append(start.elapsed_time(end))
        else:
            timings['reverse'].append(0.0)
        
        projected_grad = (loss1 - loss2) / (2 * eps)
        
        # === Perturb +eps (reset) ===
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        if perturb_mode == "CUDA":
            fused_perturb_cuda.fused_perturb(param_flat, seed, eps)
        else:
            fused_perturb_runtime_seed[grid](param_flat, seed_tensor, eps, n_elements)
        end.record()
        torch.cuda.synchronize()
        timings['perturb3'].append(start.elapsed_time(end))
        
        # === Update ===
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        if perturb_mode == "CUDA":
            fused_perturb_cuda.fused_update(param_flat, seed, projected_grad, lr)
        else:
            fused_update_runtime_seed[grid](param_flat, seed_tensor, projected_grad, lr, n_elements)
        end.record()
        torch.cuda.synchronize()
        timings['update'].append(start.elapsed_time(end))
        
        # === DiZO: Gamma perturbation + update ===
        if include_dizo:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            if zo_mode == "Triton V2":
                zo_kernels.perturb_gamma(constraints, norms, 1.0, tau, zo_eps, generate_new=True)
            else:
                # CUDA V5 doesn't have gamma perturb kernel, use PyTorch
                zs = torch.randn(num_params, device=device)
            end.record()
            torch.cuda.synchronize()
            timings['gamma_perturb'].append(start.elapsed_time(end))
            
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            if zo_mode == "Triton V2":
                zo_kernels.update_gamma(constraints, norms, projected_grad, step_size, tau)
            else:
                cuda_zo_v5.fused_update_gamma(constraints, norms, zs, projected_grad, step_size, tau)
            end.record()
            torch.cuda.synchronize()
            timings['gamma_update'].append(start.elapsed_time(end))
        else:
            timings['gamma_perturb'].append(0.0)
            timings['gamma_update'].append(0.0)
        
        total_end.record()
        torch.cuda.synchronize()
        timings['total'].append(total_start.elapsed_time(total_end))
    
    # Compute statistics (skip first iteration)
    results = {}
    for key in timings:
        values = timings[key][1:]  # Skip first
        results[key] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
        }
    
    return results


def benchmark_pytorch_baseline(
    param_flat, anchor_flat, offsets, sizes, constraints,
    n_iter=10, warmup=3, eps=1e-3, lr=1e-5, tau=0.2, zo_eps=0.1, step_size=2.0,
    include_dizo=True,
):
    """
    PyTorch baseline matching trainer.py exactly.
    Per-parameter loops for all operations (no custom kernels).
    
    Args:
        include_dizo: If False, skip DiZO constraint operations (MeZO-only mode).
    """
    device = param_flat.device
    n_elements = param_flat.numel()
    num_params = len(offsets)
    
    # Create parameter views
    param_views = []
    anchor_views = []
    for i in range(num_params):
        start = offsets[i].item()
        size = sizes[i].item()
        param_views.append(param_flat[start:start+size])
        anchor_views.append(anchor_flat[start:start+size])
    
    mode_str = "PyTorch baseline (per-parameter loops)"
    if not include_dizo:
        mode_str += " - MeZO only"
    else:
        mode_str += " + DiZO"
    print(f"Using: {mode_str}")
    
    # Warmup
    print(f"Warming up ({warmup} iterations)...")
    for _ in range(warmup):
        seed = 42
        torch.manual_seed(seed)
        
        # Perturb +eps
        for p in param_views:
            z = torch.randn_like(p)
            p.add_(z, alpha=eps)
        
        # Compute norms and apply constraints (DiZO only)
        if include_dizo:
            for i, (p, a, c) in enumerate(zip(param_views, anchor_views, constraints)):
                diff = p - a
                norm = diff.norm()
                if norm > c:
                    alpha = c / (norm + 1e-8)
                    p.copy_(a + alpha * diff)
        
        # Perturb -2eps
        torch.manual_seed(seed)
        for p in param_views:
            z = torch.randn_like(p)
            p.add_(z, alpha=-2*eps)
        
        # Reverse constraints (DiZO only)
        if include_dizo:
            for i, (p, a, c) in enumerate(zip(param_views, anchor_views, constraints)):
                diff = p - a
                norm = diff.norm()
                if norm > c:
                    alpha = c / (norm + 1e-8)
                    p.copy_(a + alpha * diff)
        
        # Update
        torch.manual_seed(seed)
        for p in param_views:
            z = torch.randn_like(p)
            p.add_(z, alpha=-lr * 0.001)
    
    torch.cuda.synchronize()
    
    # Reset param_flat
    param_flat.copy_(anchor_flat)
    torch.cuda.synchronize()
    
    # Benchmark with detailed timing
    print(f"Benchmarking ({n_iter} iterations)...")
    timings = {
        'perturb1': [],
        'forward1': [],
        'norm': [],
        'apply': [],
        'perturb2': [],
        'forward2': [],
        'reverse': [],
        'perturb3': [],
        'update': [],
        'gamma_perturb': [],
        'gamma_update': [],
        'total': [],
    }
    
    for i in range(n_iter):
        seed = np.random.randint(1000000000)
        
        # Total timing
        total_start = torch.cuda.Event(enable_timing=True)
        total_end = torch.cuda.Event(enable_timing=True)
        
        total_start.record()
        
        # === Perturb +eps ===
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        torch.manual_seed(seed)
        for p in param_views:
            z = torch.randn_like(p)
            p.add_(z, alpha=eps)
        end.record()
        torch.cuda.synchronize()
        timings['perturb1'].append(start.elapsed_time(end))
        
        # === Forward 1 ===
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        loss1 = dummy_forward()
        end.record()
        torch.cuda.synchronize()
        timings['forward1'].append(start.elapsed_time(end))
        
        # === DiZO: Compute norms ===
        if include_dizo:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            norms = torch.zeros(num_params, device=device)
            for i_p, (p, a) in enumerate(zip(param_views, anchor_views)):
                norms[i_p] = (p - a).norm()
            end.record()
            torch.cuda.synchronize()
            timings['norm'].append(start.elapsed_time(end))
            
            # === DiZO: Apply constraints ===
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            alphas = torch.zeros(num_params, device=device)
            for i_p, (p, a, c, n) in enumerate(zip(param_views, anchor_views, constraints, norms)):
                if n > c:
                    alpha = c / (n + 1e-8)
                    alphas[i_p] = alpha
                    diff = p - a
                    p.copy_(a + alpha * diff)
                else:
                    alphas[i_p] = 1.0
            end.record()
            torch.cuda.synchronize()
            timings['apply'].append(start.elapsed_time(end))
        else:
            timings['norm'].append(0.0)
            timings['apply'].append(0.0)
        
        # === Perturb -2eps ===
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        torch.manual_seed(seed)
        for p in param_views:
            z = torch.randn_like(p)
            p.add_(z, alpha=-2*eps)
        end.record()
        torch.cuda.synchronize()
        timings['perturb2'].append(start.elapsed_time(end))
        
        # === Forward 2 ===
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        loss2 = dummy_forward()
        end.record()
        torch.cuda.synchronize()
        timings['forward2'].append(start.elapsed_time(end))
        
        # === DiZO: Reverse constraints ===
        if include_dizo:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for i_p, (p, a, alpha) in enumerate(zip(param_views, anchor_views, alphas)):
                if alpha < 1.0:
                    diff = p - a
                    p.copy_(a + diff / alpha)
            end.record()
            torch.cuda.synchronize()
            timings['reverse'].append(start.elapsed_time(end))
        else:
            timings['reverse'].append(0.0)
        
        projected_grad = (loss1 - loss2) / (2 * eps)
        
        # === Perturb +eps (reset) ===
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        torch.manual_seed(seed)
        for p in param_views:
            z = torch.randn_like(p)
            p.add_(z, alpha=eps)
        end.record()
        torch.cuda.synchronize()
        timings['perturb3'].append(start.elapsed_time(end))
        
        # === Update ===
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        torch.manual_seed(seed)
        for p in param_views:
            z = torch.randn_like(p)
            p.add_(z, alpha=-lr * projected_grad)
        end.record()
        torch.cuda.synchronize()
        timings['update'].append(start.elapsed_time(end))
        
        # === DiZO: Gamma perturbation + update ===
        if include_dizo:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            zs = torch.randn(num_params, device=device)
            constraints_perturbed = constraints * torch.exp(tau * zo_eps * zs / norms.clamp(min=1e-8))
            end.record()
            torch.cuda.synchronize()
            timings['gamma_perturb'].append(start.elapsed_time(end))
            
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            constraints.mul_(torch.exp(-step_size * tau * projected_grad * zs / norms.clamp(min=1e-8)))
            end.record()
            torch.cuda.synchronize()
            timings['gamma_update'].append(start.elapsed_time(end))
        else:
            timings['gamma_perturb'].append(0.0)
            timings['gamma_update'].append(0.0)
        
        total_end.record()
        torch.cuda.synchronize()
        timings['total'].append(total_start.elapsed_time(total_end))
    
    # Compute statistics (skip first iteration)
    results = {}
    for key in timings:
        values = timings[key][1:]  # Skip first
        results[key] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
        }
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Accurate full step benchmark with detailed per-operation timing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available backends:
  --- PyTorch Baselines ---
  pytorch-mezo   - PyTorch Baseline (MeZO only, no DiZO constraints)
  pytorch-dizo   - PyTorch Baseline + DiZO constraints
  
  --- MeZO-only (no DiZO constraints) ---
  cuda-mezo      - CUDA Perturb only (MeZO, no constraints)
  triton-mezo    - Triton Perturb only (MeZO, no constraints)
  
  --- Full DiZO (with constraints) ---
  triton-dizo    - Triton Perturb + Triton ZO V2 (default)
  cuda-dizo      - CUDA Perturb + Triton ZO V2
  cuda-full      - CUDA Perturb + CUDA ZO V5

Examples:
  # PyTorch baselines
  python benchmark_full_step_accurate.py --model opt-350m --backend pytorch-mezo
  python benchmark_full_step_accurate.py --model opt-350m --backend pytorch-dizo
  
  # MeZO-only (no DiZO constraints)  
  python benchmark_full_step_accurate.py --model opt-13b --backend cuda-mezo
  python benchmark_full_step_accurate.py --model opt-13b --backend triton-mezo
  
  # Full DiZO (with constraints)
  python benchmark_full_step_accurate.py --model opt-13b --backend cuda-dizo
  python benchmark_full_step_accurate.py --model opt-13b --backend triton-dizo
"""
    )
    parser.add_argument('--model', type=str, default='opt-350m',
                        choices=['opt-350m', 'opt-2.7b', 'opt-6.7b', 'opt-13b'])
    parser.add_argument('--n_iter', type=int, default=15)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--backend', type=str, default='triton-dizo',
                        choices=['pytorch-mezo', 'pytorch-dizo', 
                                 'cuda-mezo', 'triton-mezo',
                                 'triton-dizo', 'cuda-dizo', 'cuda-full'],
                        help='Backend to benchmark')
    # Keep legacy arguments for backwards compatibility
    parser.add_argument('--cuda-perturb', action='store_true', help='(Legacy) Use CUDA perturb')
    parser.add_argument('--cuda-zo', action='store_true', help='(Legacy) Use CUDA ZO V5')
    parser.add_argument('--pytorch', action='store_true', help='(Legacy) Run PyTorch baseline')
    args = parser.parse_args()
    
    # Handle legacy arguments
    if args.pytorch:
        args.backend = 'pytorch-dizo'
    elif args.cuda_perturb and args.cuda_zo:
        args.backend = 'cuda-full'
    elif args.cuda_perturb:
        args.backend = 'cuda-dizo'
    
    # Parse backend into options
    use_pytorch = args.backend.startswith('pytorch')
    use_cuda_perturb = args.backend.startswith('cuda')
    use_cuda_zo = args.backend == 'cuda-full'
    include_dizo = args.backend.endswith('dizo') or args.backend == 'cuda-full'
    
    print("=" * 80)
    print(f"ACCURATE FULL STEP BENCHMARK: {args.model.upper()}")
    print("=" * 80)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Backend: {args.backend}")
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
    
    # Run benchmark
    if use_pytorch:
        results = benchmark_pytorch_baseline(
            param_flat, anchor_flat, offsets, sizes, constraints,
            n_iter=args.n_iter, warmup=args.warmup,
            include_dizo=include_dizo,
        )
    else:
        results = benchmark_full_step_accurate(
            param_flat, anchor_flat, offsets, sizes, constraints,
            n_iter=args.n_iter, warmup=args.warmup,
            use_cuda_perturb=use_cuda_perturb, use_cuda_zo=use_cuda_zo,
            include_dizo=include_dizo,
        )
    
    # Print results
    print()
    print("=" * 80)
    print("DETAILED TIMING BREAKDOWN (ms)")
    print("=" * 80)
    print(f"{'Operation':<20} {'Mean':>12} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-" * 62)
    
    # Compute grouped timings
    perturb_total = results['perturb1']['mean'] + results['perturb2']['mean'] + results['perturb3']['mean']
    zo_total = results['norm']['mean'] + results['apply']['mean'] + results['reverse']['mean']
    gamma_total = results['gamma_perturb']['mean'] + results['gamma_update']['mean']
    update_total = results['update']['mean']
    forward_total = results['forward1']['mean'] + results['forward2']['mean']
    
    for key in ['perturb1', 'forward1', 'norm', 'apply', 'perturb2', 'forward2', 'reverse', 
                'perturb3', 'update', 'gamma_perturb', 'gamma_update', 'total']:
        r = results[key]
        print(f"{key:<20} {r['mean']:>12.3f} {r['std']:>10.3f} {r['min']:>10.3f} {r['max']:>10.3f}")
    
    print()
    print("=" * 80)
    print("GROUPED TIMING SUMMARY (ms)")
    print("=" * 80)
    print(f"Perturb (3 ops):     {perturb_total:>10.3f} ms")
    print(f"ZO Constraints:      {zo_total:>10.3f} ms (norm + apply + reverse)")
    print(f"Gamma Update:        {gamma_total:>10.3f} ms (perturb + update)")
    print(f"Update (MeZO):       {update_total:>10.3f} ms")
    print(f"Forward (dummy):     {forward_total:>10.3f} ms")
    print(f"Total measured:      {results['total']['mean']:>10.3f} ms")
    print()
    
    # Compute expected total
    expected = perturb_total + zo_total + gamma_total + update_total + forward_total
    print(f"Expected (sum):      {expected:>10.3f} ms")
    print(f"Overhead:            {results['total']['mean'] - expected:>10.3f} ms")


if __name__ == '__main__':
    main()
