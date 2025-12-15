#!/usr/bin/env python3
"""
Fused MeZO Operations using Triton Kernels

This is the optimized implementation that achieves maximum speedup by:
1. Batched RNG - Single randn() call for all parameters  
2. Fused Triton kernels - Single kernel launch for perturbation/update
3. Minimal Python overhead

Key optimization: Operating on a FLAT parameter buffer allows single-kernel operations.
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
def perturb_add_kernel(
    params_ptr,
    z_ptr,
    n_elements,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """params[i] += eps * z[i]"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    params = tl.load(params_ptr + offsets, mask=mask)
    z = tl.load(z_ptr + offsets, mask=mask)
    result = params + eps * z
    tl.store(params_ptr + offsets, result, mask=mask)


@triton.jit
def perturb_sub_kernel(
    params_ptr,
    z_ptr,
    n_elements,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """params[i] -= 2*eps * z[i]"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    params = tl.load(params_ptr + offsets, mask=mask)
    z = tl.load(z_ptr + offsets, mask=mask)
    result = params - 2.0 * eps * z
    tl.store(params_ptr + offsets, result, mask=mask)


@triton.jit
def update_kernel(
    params_ptr,
    z_ptr,
    n_elements,
    scale,
    BLOCK_SIZE: tl.constexpr,
):
    """params[i] -= scale * z[i]"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    params = tl.load(params_ptr + offsets, mask=mask)
    z = tl.load(z_ptr + offsets, mask=mask)
    result = params - scale * z
    tl.store(params_ptr + offsets, result, mask=mask)


# =============================================================================
# MeZO Trainer with Fused Kernels
# =============================================================================

class FusedMeZOTrainer:
    """
    MeZO Trainer with fused Triton kernels.
    
    Architecture:
    - Maintains a flat buffer that mirrors all trainable parameters
    - Perturbation/update operate on flat buffer with single kernel
    - Syncs buffer back to model parameters for forward pass
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
        self.param_shapes: List[torch.Size] = []
        self.param_numels: List[int] = []
        self.offsets: List[int] = []
        self.total_numel = 0
        
        offset = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.trainable_params.append(param)
                self.param_shapes.append(param.shape)
                self.param_numels.append(param.numel())
                self.offsets.append(offset)
                self.total_numel += param.numel()
                offset += param.numel()
        
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        
        print(f"[FusedMeZO] {len(self.trainable_params)} params, {self.total_numel:,} elements")
        
        # Allocate flat buffers
        self.params_flat = torch.empty(self.total_numel, device=self.device, dtype=self.dtype)
        self.z_flat = torch.empty(self.total_numel, device=self.device, dtype=self.dtype)
        
        # Initial sync from model
        self._sync_from_model()
        
        # Timing
        self.timing = {
            'rng': [],
            'perturb': [],
            'sync': [],
            'forward': [],
            'update': [],
            'total': [],
        }
    
    def _sync_from_model(self):
        """Copy model parameters to flat buffer"""
        for i, param in enumerate(self.trainable_params):
            start = self.offsets[i]
            end = start + self.param_numels[i]
            self.params_flat[start:end].copy_(param.data.view(-1))
    
    def _sync_to_model(self):
        """Copy flat buffer to model parameters"""
        for i, param in enumerate(self.trainable_params):
            start = self.offsets[i]
            end = start + self.param_numels[i]
            param.data.copy_(self.params_flat[start:end].view(self.param_shapes[i]))
    
    def generate_perturbation(self, seed: Optional[int] = None):
        """Generate random perturbation - single RNG call"""
        t0 = time.perf_counter()
        
        if seed is not None:
            torch.manual_seed(seed)
        self.z_flat.normal_()
        
        self.timing['rng'].append((time.perf_counter() - t0) * 1000)
    
    def _perturb_add(self):
        """Apply positive perturbation using Triton kernel"""
        grid = lambda meta: (triton.cdiv(self.total_numel, meta['BLOCK_SIZE']),)
        perturb_add_kernel[grid](
            self.params_flat,
            self.z_flat,
            self.total_numel,
            self.eps,
            BLOCK_SIZE=self.block_size,
        )
    
    def _perturb_sub(self):
        """Apply negative perturbation using Triton kernel"""
        grid = lambda meta: (triton.cdiv(self.total_numel, meta['BLOCK_SIZE']),)
        perturb_sub_kernel[grid](
            self.params_flat,
            self.z_flat,
            self.total_numel,
            self.eps,
            BLOCK_SIZE=self.block_size,
        )
    
    def _update(self, scale: float):
        """Apply parameter update using Triton kernel"""
        grid = lambda meta: (triton.cdiv(self.total_numel, meta['BLOCK_SIZE']),)
        update_kernel[grid](
            self.params_flat,
            self.z_flat,
            self.total_numel,
            scale,
            BLOCK_SIZE=self.block_size,
        )
    
    def zo_forward(self, batch) -> torch.Tensor:
        """Forward pass returning loss"""
        outputs = self.model(**batch)
        return outputs.loss if hasattr(outputs, 'loss') else outputs[0]
    
    def zo_step(self, batch) -> Tuple[torch.Tensor, float]:
        """
        Single ZO gradient estimation step with fused Triton kernels.
        """
        t0 = time.perf_counter()
        
        seed = np.random.randint(0, 2**31)
        
        # Generate perturbation (single RNG call)
        self.generate_perturbation(seed)
        
        # Perturb +eps (single Triton kernel)
        perturb_t0 = time.perf_counter()
        self._perturb_add()
        self.timing['perturb'].append((time.perf_counter() - perturb_t0) * 1000)
        
        # Sync to model and forward pass 1
        sync_t0 = time.perf_counter()
        self._sync_to_model()
        self.timing['sync'].append((time.perf_counter() - sync_t0) * 1000)
        
        forward_t0 = time.perf_counter()
        with torch.no_grad():
            loss1 = self.zo_forward(batch)
        
        # Perturb -2eps (single Triton kernel)
        perturb_t0 = time.perf_counter()
        self._perturb_sub()
        self.timing['perturb'].append((time.perf_counter() - perturb_t0) * 1000)
        
        # Sync to model and forward pass 2
        sync_t0 = time.perf_counter()
        self._sync_to_model()
        self.timing['sync'].append((time.perf_counter() - sync_t0) * 1000)
        
        with torch.no_grad():
            loss2 = self.zo_forward(batch)
        
        self.timing['forward'].append((time.perf_counter() - forward_t0) * 1000)
        
        # Reset: perturb +eps
        perturb_t0 = time.perf_counter()
        self._perturb_add()
        self.timing['perturb'].append((time.perf_counter() - perturb_t0) * 1000)
        
        # Compute gradient estimate
        projected_grad = (loss1.item() - loss2.item()) / (2 * self.eps)
        
        # Update (single Triton kernel)
        update_t0 = time.perf_counter()
        scale = self.lr * projected_grad
        self._update(scale)
        self.timing['update'].append((time.perf_counter() - update_t0) * 1000)
        
        # Final sync
        sync_t0 = time.perf_counter()
        self._sync_to_model()
        self.timing['sync'].append((time.perf_counter() - sync_t0) * 1000)
        
        self.timing['total'].append((time.perf_counter() - t0) * 1000)
        
        return (loss1 + loss2) / 2, projected_grad
    
    def print_timing(self):
        """Print timing summary"""
        print("\n" + "="*60)
        print("FUSED MEZO (TRITON) TIMING")
        print("="*60)
        
        for key, times in self.timing.items():
            if times:
                avg = np.mean(times)
                std = np.std(times) if len(times) > 1 else 0
                if key == 'perturb':
                    # perturb is called 3x per step
                    print(f"  {key:12s}: {avg:8.3f} ± {std:6.3f} ms  (per call, 3x/step)")
                elif key == 'sync':
                    # sync is called 3x per step
                    print(f"  {key:12s}: {avg:8.3f} ± {std:6.3f} ms  (per call, 3x/step)")
                else:
                    print(f"  {key:12s}: {avg:8.3f} ± {std:6.3f} ms")
        
        # Compute total kernel time vs sync overhead
        if self.timing['perturb'] and self.timing['sync']:
            avg_perturb = np.mean(self.timing['perturb'])
            avg_sync = np.mean(self.timing['sync'])
            avg_update = np.mean(self.timing['update']) if self.timing['update'] else 0
            
            kernel_time = avg_perturb * 3 + avg_update
            sync_time = avg_sync * 3
            
            print(f"\n  Breakdown per step:")
            print(f"    Triton kernels: {kernel_time:.3f} ms")
            print(f"    Sync overhead:  {sync_time:.3f} ms")


# =============================================================================
# Baseline for comparison
# =============================================================================

class BaselineMeZOTrainer:
    """Original baseline MeZO"""
    
    def __init__(self, model: nn.Module, eps: float = 1e-3, lr: float = 1e-5):
        self.model = model
        self.eps = eps
        self.lr = lr
        self.trainable_params = [p for p in model.parameters() if p.requires_grad]
        print(f"[BaselineMeZO] {len(self.trainable_params)} params")
        
        self.timing = {'total': []}
    
    def zo_perturb_parameters(self, seed: int, scaling_factor: float = 1.0):
        torch.manual_seed(seed)
        for param in self.trainable_params:
            z = torch.randn_like(param)
            param.data.add_(z, alpha=scaling_factor * self.eps)
    
    def zo_forward(self, batch):
        outputs = self.model(**batch)
        return outputs.loss if hasattr(outputs, 'loss') else outputs[0]
    
    def zo_step(self, batch):
        t0 = time.perf_counter()
        seed = np.random.randint(0, 2**31)
        
        self.zo_perturb_parameters(seed, scaling_factor=1.0)
        
        with torch.no_grad():
            loss1 = self.zo_forward(batch)
        
        self.zo_perturb_parameters(seed, scaling_factor=-2.0)
        
        with torch.no_grad():
            loss2 = self.zo_forward(batch)
        
        self.zo_perturb_parameters(seed, scaling_factor=1.0)
        
        projected_grad = (loss1.item() - loss2.item()) / (2 * self.eps)
        
        torch.manual_seed(seed)
        for param in self.trainable_params:
            z = torch.randn_like(param)
            param.data.add_(z, alpha=-self.lr * projected_grad)
        
        self.timing['total'].append((time.perf_counter() - t0) * 1000)
        
        return (loss1 + loss2) / 2, projected_grad


# =============================================================================
# Benchmark
# =============================================================================

def benchmark(
    model_name: str = "facebook/opt-350m",
    num_steps: int = 20,
    warmup_steps: int = 5,
    batch_size: int = 4,
    seq_length: int = 128,
):
    """Comprehensive benchmark"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print("="*70)
    print("FUSED TRITON MEZO BENCHMARK")
    print("="*70)
    print(f"Model: {model_name}")
    print(f"Steps: {num_steps}, Warmup: {warmup_steps}")
    print(f"Batch: {batch_size} x {seq_length}")
    
    device = "cuda"
    
    # Load model
    print(f"\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    
    # Create batch
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    texts = ["This is a test sentence for benchmarking."] * batch_size
    batch = tokenizer(texts, return_tensors="pt", padding="max_length",
                      max_length=seq_length, truncation=True)
    batch = {k: v.to(device) for k, v in batch.items()}
    batch['labels'] = batch['input_ids'].clone()
    
    # =========================================================================
    # Baseline
    # =========================================================================
    print("\n" + "-"*70)
    print("BASELINE MeZO (per-parameter loops)")
    print("-"*70)
    
    baseline = BaselineMeZOTrainer(model, eps=1e-3, lr=1e-5)
    
    # Warmup
    for _ in range(warmup_steps):
        baseline.zo_step(batch)
    baseline.timing = {'total': []}
    
    # Benchmark
    torch.cuda.synchronize()
    for step in range(num_steps):
        loss, _ = baseline.zo_step(batch)
        if step % 10 == 0:
            print(f"  Step {step}: loss={loss.item():.4f}")
    torch.cuda.synchronize()
    
    baseline_time = np.mean(baseline.timing['total'])
    print(f"\n  Average step time: {baseline_time:.2f} ms")
    
    # =========================================================================
    # Fused Triton
    # =========================================================================
    del model
    torch.cuda.empty_cache()
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    
    print("\n" + "-"*70)
    print("FUSED MeZO (Triton kernels)")
    print("-"*70)
    
    fused = FusedMeZOTrainer(model, eps=1e-3, lr=1e-5)
    
    # Warmup
    for _ in range(warmup_steps):
        fused.zo_step(batch)
    fused.timing = {k: [] for k in fused.timing}
    
    # Benchmark
    torch.cuda.synchronize()
    for step in range(num_steps):
        loss, _ = fused.zo_step(batch)
        if step % 10 == 0:
            print(f"  Step {step}: loss={loss.item():.4f}")
    torch.cuda.synchronize()
    
    fused.print_timing()
    fused_time = np.mean(fused.timing['total'])
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*70)
    print("SPEEDUP SUMMARY")
    print("="*70)
    
    speedup = baseline_time / fused_time
    
    print(f"\n  Baseline:  {baseline_time:.2f} ms/step")
    print(f"  Fused:     {fused_time:.2f} ms/step")
    print(f"  Speedup:   {speedup:.2f}x")
    
    # Analyze bottleneck
    sync_overhead = np.mean(fused.timing['sync']) * 3  # 3 syncs per step
    kernel_time = np.mean(fused.timing['perturb']) * 3 + np.mean(fused.timing['update'])
    forward_time = np.mean(fused.timing['forward'])
    
    print(f"\n  Fused breakdown:")
    print(f"    Triton kernels: {kernel_time:.2f} ms ({kernel_time/fused_time*100:.1f}%)")
    print(f"    Sync overhead:  {sync_overhead:.2f} ms ({sync_overhead/fused_time*100:.1f}%)")
    print(f"    Forward pass:   {forward_time:.2f} ms ({forward_time/fused_time*100:.1f}%)")
    
    print(f"\n  Insight: Sync overhead ({sync_overhead:.1f}ms) is significant.")
    print(f"  Further optimization: Avoid sync by operating directly on param.data")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="facebook/opt-350m")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-length", type=int, default=128)
    args = parser.parse_args()
    
    benchmark(
        model_name=args.model,
        num_steps=args.steps,
        warmup_steps=args.warmup,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
    )
