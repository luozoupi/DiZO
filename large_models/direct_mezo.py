#!/usr/bin/env python3
"""
Optimized MeZO with Direct Parameter Access + Triton Kernels

This version achieves maximum speedup by:
1. Operating directly on model parameter memory (no copy/sync)
2. Using Triton kernels for batched operations
3. Single RNG call for all parameters

Key insight: We can create tensor views into each parameter and pass them
to Triton, avoiding the need to flatten and sync.
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl
import time
import numpy as np
from typing import List, Tuple, Optional


# =============================================================================
# Triton Kernels  
# =============================================================================

@triton.jit
def fused_add_kernel(
    param_ptr,
    z_ptr,
    n_elements,
    alpha,
    BLOCK_SIZE: tl.constexpr,
):
    """param[i] += alpha * z[i]"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    p = tl.load(param_ptr + offsets, mask=mask)
    z = tl.load(z_ptr + offsets, mask=mask)
    result = p + alpha * z
    tl.store(param_ptr + offsets, result, mask=mask)


# =============================================================================
# Direct Parameter MeZO Trainer
# =============================================================================

class DirectMeZOTrainer:
    """
    MeZO trainer that operates directly on model parameters.
    
    This avoids the sync overhead by:
    1. Generating z tensors that match parameter shapes
    2. Calling Triton kernel on each parameter directly
    3. Using batched RNG to generate z values efficiently
    """
    
    def __init__(
        self,
        model: nn.Module,
        eps: float = 1e-3,
        lr: float = 1e-5,
        block_size: int = 1024,
    ):
        self.model = model
        self.eps = eps
        self.lr = lr
        self.block_size = block_size
        
        # Collect trainable parameters
        self.trainable_params: List[nn.Parameter] = []
        self.param_numels: List[int] = []
        self.offsets: List[int] = []
        self.total_numel = 0
        
        offset = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.trainable_params.append(param)
                self.param_numels.append(param.numel())
                self.offsets.append(offset)
                self.total_numel += param.numel()
                offset += param.numel()
        
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        
        print(f"[DirectMeZO] {len(self.trainable_params)} params, {self.total_numel:,} elements")
        
        # Pre-allocate flat z buffer and views for each parameter
        self.z_flat = torch.empty(self.total_numel, device=self.device, dtype=self.dtype)
        self.z_views: List[torch.Tensor] = []
        
        for i, param in enumerate(self.trainable_params):
            start = self.offsets[i]
            end = start + self.param_numels[i]
            z_view = self.z_flat[start:end].view(param.shape)
            self.z_views.append(z_view)
        
        # Timing
        self.timing = {
            'rng': [],
            'perturb': [],
            'forward': [],
            'update': [],
            'total': [],
        }
    
    def generate_perturbation(self, seed: Optional[int] = None):
        """Generate random perturbation using single RNG call"""
        t0 = time.perf_counter()
        
        if seed is not None:
            torch.manual_seed(seed)
        self.z_flat.normal_()  # Single RNG call!
        
        self.timing['rng'].append((time.perf_counter() - t0) * 1000)
    
    def _perturb_triton(self, alpha: float):
        """Apply perturbation using Triton kernels directly on parameters"""
        for param, z_view in zip(self.trainable_params, self.z_views):
            n = param.numel()
            grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
            fused_add_kernel[grid](
                param.data.view(-1),  # Flat view of parameter
                z_view.view(-1),       # Flat view of z
                n,
                alpha,
                BLOCK_SIZE=self.block_size,
            )
    
    def _perturb_pytorch(self, alpha: float):
        """Apply perturbation using PyTorch (for comparison)"""
        for param, z_view in zip(self.trainable_params, self.z_views):
            param.data.add_(z_view, alpha=alpha)
    
    def zo_forward(self, batch) -> torch.Tensor:
        """Forward pass returning loss"""
        outputs = self.model(**batch)
        return outputs.loss if hasattr(outputs, 'loss') else outputs[0]
    
    def zo_step(self, batch, use_triton: bool = True) -> Tuple[torch.Tensor, float]:
        """
        Single ZO gradient estimation step.
        
        Args:
            use_triton: If True, use Triton kernels; otherwise PyTorch
        """
        t0 = time.perf_counter()
        
        seed = np.random.randint(0, 2**31)
        
        # Generate perturbation (single batched RNG)
        self.generate_perturbation(seed)
        
        # Perturb +eps
        perturb_t0 = time.perf_counter()
        if use_triton:
            self._perturb_triton(self.eps)
        else:
            self._perturb_pytorch(self.eps)
        perturb_time = time.perf_counter() - perturb_t0
        
        # Forward pass 1
        forward_t0 = time.perf_counter()
        with torch.no_grad():
            loss1 = self.zo_forward(batch)
        
        # Perturb -2eps
        if use_triton:
            self._perturb_triton(-2 * self.eps)
        else:
            self._perturb_pytorch(-2 * self.eps)
        
        # Forward pass 2
        with torch.no_grad():
            loss2 = self.zo_forward(batch)
        self.timing['forward'].append((time.perf_counter() - forward_t0) * 1000)
        
        # Reset: +eps
        if use_triton:
            self._perturb_triton(self.eps)
        else:
            self._perturb_pytorch(self.eps)
        
        self.timing['perturb'].append(perturb_time * 1000)
        
        # Compute gradient
        projected_grad = (loss1.item() - loss2.item()) / (2 * self.eps)
        
        # Update
        update_t0 = time.perf_counter()
        scale = -self.lr * projected_grad
        if use_triton:
            self._perturb_triton(scale / self.eps * self.eps)  # Reuse z
        else:
            for param, z_view in zip(self.trainable_params, self.z_views):
                param.data.add_(z_view, alpha=-self.lr * projected_grad)
        self.timing['update'].append((time.perf_counter() - update_t0) * 1000)
        
        self.timing['total'].append((time.perf_counter() - t0) * 1000)
        
        return (loss1 + loss2) / 2, projected_grad
    
    def print_timing(self, name: str = "DirectMeZO"):
        """Print timing summary"""
        print(f"\n{'='*60}")
        print(f"{name} TIMING")
        print("="*60)
        
        for key, times in self.timing.items():
            if times:
                avg = np.mean(times)
                std = np.std(times) if len(times) > 1 else 0
                print(f"  {key:12s}: {avg:8.3f} ± {std:6.3f} ms")


# =============================================================================
# Baseline
# =============================================================================

class BaselineMeZO:
    """Original baseline for comparison"""
    
    def __init__(self, model, eps=1e-3, lr=1e-5):
        self.model = model
        self.eps = eps
        self.lr = lr
        self.trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.timing = {'total': []}
    
    def zo_step(self, batch):
        t0 = time.perf_counter()
        seed = np.random.randint(0, 2**31)
        
        # Per-parameter RNG and perturbation
        torch.manual_seed(seed)
        for p in self.trainable_params:
            z = torch.randn_like(p)
            p.data.add_(z, alpha=self.eps)
        
        with torch.no_grad():
            loss1 = self.model(**batch).loss
        
        torch.manual_seed(seed)
        for p in self.trainable_params:
            z = torch.randn_like(p)
            p.data.add_(z, alpha=-2*self.eps)
        
        with torch.no_grad():
            loss2 = self.model(**batch).loss
        
        torch.manual_seed(seed)
        for p in self.trainable_params:
            z = torch.randn_like(p)
            p.data.add_(z, alpha=self.eps)
        
        grad = (loss1.item() - loss2.item()) / (2 * self.eps)
        
        torch.manual_seed(seed)
        for p in self.trainable_params:
            z = torch.randn_like(p)
            p.data.add_(z, alpha=-self.lr * grad)
        
        self.timing['total'].append((time.perf_counter() - t0) * 1000)
        return (loss1 + loss2) / 2, grad


# =============================================================================
# Benchmark
# =============================================================================

def benchmark(model_name="facebook/opt-350m", num_steps=20, warmup_steps=5):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print("="*70)
    print("DIRECT MEZO BENCHMARK")
    print("="*70)
    
    device = "cuda"
    
    # Load model
    print(f"\nLoading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()
    
    # Create batch
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    batch = tokenizer(["Test sentence"] * 4, return_tensors="pt", 
                      padding="max_length", max_length=128, truncation=True)
    batch = {k: v.to(device) for k, v in batch.items()}
    batch['labels'] = batch['input_ids'].clone()
    
    results = {}
    
    # =========================================================================
    # Baseline
    # =========================================================================
    print("\n" + "-"*70)
    print("1. BASELINE (per-parameter RNG)")
    print("-"*70)
    
    baseline = BaselineMeZO(model)
    for _ in range(warmup_steps):
        baseline.zo_step(batch)
    baseline.timing = {'total': []}
    
    torch.cuda.synchronize()
    for i in range(num_steps):
        loss, _ = baseline.zo_step(batch)
        if i % 10 == 0:
            print(f"  Step {i}: loss={loss.item():.4f}")
    torch.cuda.synchronize()
    
    results['baseline'] = np.mean(baseline.timing['total'])
    print(f"\n  Average: {results['baseline']:.2f} ms/step")
    
    # =========================================================================
    # Direct PyTorch (batched RNG, no Triton)
    # =========================================================================
    del model
    torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()
    
    print("\n" + "-"*70)
    print("2. DIRECT PYTORCH (batched RNG, PyTorch ops)")
    print("-"*70)
    
    direct_pt = DirectMeZOTrainer(model, eps=1e-3, lr=1e-5)
    for _ in range(warmup_steps):
        direct_pt.zo_step(batch, use_triton=False)
    direct_pt.timing = {k: [] for k in direct_pt.timing}
    
    torch.cuda.synchronize()
    for i in range(num_steps):
        loss, _ = direct_pt.zo_step(batch, use_triton=False)
        if i % 10 == 0:
            print(f"  Step {i}: loss={loss.item():.4f}")
    torch.cuda.synchronize()
    
    direct_pt.print_timing("DIRECT PYTORCH")
    results['direct_pytorch'] = np.mean(direct_pt.timing['total'])
    
    # =========================================================================
    # Direct Triton
    # =========================================================================
    del model
    torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()
    
    print("\n" + "-"*70)
    print("3. DIRECT TRITON (batched RNG, Triton kernels)")
    print("-"*70)
    
    direct_triton = DirectMeZOTrainer(model, eps=1e-3, lr=1e-5)
    for _ in range(warmup_steps):
        direct_triton.zo_step(batch, use_triton=True)
    direct_triton.timing = {k: [] for k in direct_triton.timing}
    
    torch.cuda.synchronize()
    for i in range(num_steps):
        loss, _ = direct_triton.zo_step(batch, use_triton=True)
        if i % 10 == 0:
            print(f"  Step {i}: loss={loss.item():.4f}")
    torch.cuda.synchronize()
    
    direct_triton.print_timing("DIRECT TRITON")
    results['direct_triton'] = np.mean(direct_triton.timing['total'])
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print(f"\n  {'Method':<30} {'Time (ms)':<12} {'Speedup':<10}")
    print(f"  {'-'*50}")
    
    for name, time_ms in results.items():
        speedup = results['baseline'] / time_ms
        print(f"  {name:<30} {time_ms:>8.2f}     {speedup:>6.2f}x")
    
    print("\n" + "="*70)
    
    # Analysis
    print("\nANALYSIS:")
    print(f"  - Batched RNG saves: {results['baseline'] - results['direct_pytorch']:.2f} ms/step")
    print(f"  - Triton vs PyTorch: {results['direct_pytorch'] / results['direct_triton']:.2f}x")
    
    # Breakdown for Triton version
    rng_time = np.mean(direct_triton.timing['rng'])
    perturb_time = np.mean(direct_triton.timing['perturb'])
    forward_time = np.mean(direct_triton.timing['forward'])
    update_time = np.mean(direct_triton.timing['update'])
    total_time = results['direct_triton']
    
    print(f"\n  Direct Triton Breakdown:")
    print(f"    RNG:      {rng_time:.2f} ms ({rng_time/total_time*100:.1f}%)")
    print(f"    Perturb:  {perturb_time:.2f} ms ({perturb_time/total_time*100:.1f}%)")
    print(f"    Forward:  {forward_time:.2f} ms ({forward_time/total_time*100:.1f}%)")
    print(f"    Update:   {update_time:.2f} ms ({update_time/total_time*100:.1f}%)")
    
    print(f"\n  Remaining optimization potential:")
    print(f"    Forward pass dominates ({forward_time/total_time*100:.0f}%)")
    print(f"    Consider: torch.compile on model, Flash Attention, etc.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="facebook/opt-350m")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()
    
    benchmark(args.model, args.steps, args.warmup)
