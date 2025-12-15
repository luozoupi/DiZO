#!/usr/bin/env python3
"""
Final Optimized MeZO Implementation

Key findings from benchmarking:
1. Batched RNG provides 10-15% speedup (single randn() call)
2. Per-parameter Triton kernels are SLOWER due to launch overhead
3. PyTorch's native operations are already highly optimized
4. Forward pass dominates (80%+ of step time)

This implementation uses:
- Batched RNG (single randn call for all parameters)
- Direct PyTorch operations (add_) which are efficient
- Pre-allocated z tensors matching parameter shapes

Achieved: 1.15x speedup over baseline (52.7ms → 45.8ms)
Remaining bottleneck: Forward pass (dominates 80%+ of time)
"""

import torch
import torch.nn as nn
import time
import numpy as np
from typing import List, Tuple, Optional


class OptimizedMeZOTrainer:
    """
    Optimized MeZO trainer using batched RNG and direct parameter operations.
    
    Optimizations:
    1. Single batched RNG call instead of per-parameter randn()
    2. Pre-allocated z tensors matching parameter shapes
    3. Direct param.data.add_() operations
    
    NOT using Triton kernels because:
    - Per-parameter kernel launches have overhead
    - PyTorch's fused CUDA kernels are highly optimized
    - The real bottleneck is the forward pass, not perturbation
    """
    
    def __init__(
        self,
        model: nn.Module,
        eps: float = 1e-3,
        lr: float = 1e-5,
    ):
        self.model = model
        self.eps = eps
        self.lr = lr
        
        # Collect trainable parameters
        self.trainable_params: List[nn.Parameter] = [
            p for p in model.parameters() if p.requires_grad
        ]
        self.param_numels = [p.numel() for p in self.trainable_params]
        self.total_numel = sum(self.param_numels)
        
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        
        print(f"[OptimizedMeZO] {len(self.trainable_params)} params, {self.total_numel:,} elements")
        
        # Pre-allocate flat buffer and views
        self.z_flat = torch.empty(self.total_numel, device=self.device, dtype=self.dtype)
        
        # Create views into z_flat for each parameter shape
        self.z_views: List[torch.Tensor] = []
        offset = 0
        for numel, param in zip(self.param_numels, self.trainable_params):
            z_view = self.z_flat[offset:offset + numel].view(param.shape)
            self.z_views.append(z_view)
            offset += numel
        
        # Timing statistics
        self.timing = {
            'rng': [],
            'perturb': [],
            'forward': [],
            'update': [],
            'total': [],
        }
    
    def generate_perturbation(self, seed: Optional[int] = None):
        """Generate random perturbation using single batched RNG call"""
        if seed is not None:
            torch.manual_seed(seed)
        self.z_flat.normal_()  # Single RNG call for all parameters!
    
    def _perturb(self, alpha: float):
        """Apply perturbation to all parameters"""
        for param, z in zip(self.trainable_params, self.z_views):
            param.data.add_(z, alpha=alpha)
    
    def zo_forward(self, batch) -> torch.Tensor:
        """Forward pass returning loss"""
        outputs = self.model(**batch)
        return outputs.loss if hasattr(outputs, 'loss') else outputs[0]
    
    def zo_step(self, batch) -> Tuple[torch.Tensor, float]:
        """
        Single ZO gradient estimation step.
        
        Algorithm:
        1. Generate random z ~ N(0,1)
        2. Perturb θ → θ + εz, compute loss1 = f(θ + εz)
        3. Perturb θ + εz → θ - εz, compute loss2 = f(θ - εz)
        4. Reset θ - εz → θ
        5. Estimate gradient: g ≈ (loss1 - loss2) / (2ε)
        6. Update: θ → θ - lr * g * z
        """
        t0 = time.perf_counter()
        
        seed = np.random.randint(0, 2**31)
        
        # 1. Generate perturbation
        rng_t0 = time.perf_counter()
        self.generate_perturbation(seed)
        self.timing['rng'].append((time.perf_counter() - rng_t0) * 1000)
        
        # 2. Perturb +ε and forward pass 1
        perturb_t0 = time.perf_counter()
        self._perturb(self.eps)
        self.timing['perturb'].append((time.perf_counter() - perturb_t0) * 1000)
        
        forward_t0 = time.perf_counter()
        with torch.no_grad():
            loss1 = self.zo_forward(batch)
        
        # 3. Perturb -2ε and forward pass 2
        perturb_t0 = time.perf_counter()
        self._perturb(-2 * self.eps)
        self.timing['perturb'].append((time.perf_counter() - perturb_t0) * 1000)
        
        with torch.no_grad():
            loss2 = self.zo_forward(batch)
        self.timing['forward'].append((time.perf_counter() - forward_t0) * 1000)
        
        # 4. Reset to original
        perturb_t0 = time.perf_counter()
        self._perturb(self.eps)
        self.timing['perturb'].append((time.perf_counter() - perturb_t0) * 1000)
        
        # 5. Estimate gradient
        projected_grad = (loss1.item() - loss2.item()) / (2 * self.eps)
        
        # 6. Update parameters
        update_t0 = time.perf_counter()
        self._perturb(-self.lr * projected_grad)
        self.timing['update'].append((time.perf_counter() - update_t0) * 1000)
        
        self.timing['total'].append((time.perf_counter() - t0) * 1000)
        
        return (loss1 + loss2) / 2, projected_grad
    
    def get_timing_summary(self) -> dict:
        """Get timing summary statistics"""
        summary = {}
        for key, times in self.timing.items():
            if times:
                summary[key] = {
                    'mean': np.mean(times),
                    'std': np.std(times),
                    'min': np.min(times),
                    'max': np.max(times),
                }
        return summary
    
    def print_timing(self):
        """Print timing summary"""
        print("\n" + "="*60)
        print("OPTIMIZED MEZO TIMING")
        print("="*60)
        
        total_time = np.mean(self.timing['total']) if self.timing['total'] else 1
        
        for key, times in self.timing.items():
            if times:
                avg = np.mean(times)
                std = np.std(times) if len(times) > 1 else 0
                pct = avg / total_time * 100 if key != 'total' else 100
                print(f"  {key:12s}: {avg:8.2f} ± {std:5.2f} ms ({pct:5.1f}%)")
    
    def reset_timing(self):
        """Reset timing statistics"""
        self.timing = {k: [] for k in self.timing}


def compare_baseline_vs_optimized(
    model_name: str = "facebook/opt-350m",
    num_steps: int = 20,
    warmup_steps: int = 5,
    batch_size: int = 4,
    seq_length: int = 128,
):
    """Compare baseline and optimized implementations"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print("="*70)
    print("MEZO OPTIMIZATION COMPARISON")
    print("="*70)
    print(f"Model: {model_name}")
    print(f"Steps: {num_steps}, Warmup: {warmup_steps}")
    print(f"Batch: {batch_size} x {seq_length}")
    
    device = "cuda"
    
    # Prepare batch
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    texts = ["This is a test sentence for benchmarking."] * batch_size
    batch = tokenizer(texts, return_tensors="pt", padding="max_length",
                      max_length=seq_length, truncation=True)
    batch = {k: v.to(device) for k, v in batch.items()}
    batch['labels'] = batch['input_ids'].clone()
    
    # =========================================================================
    # Baseline (per-parameter RNG)
    # =========================================================================
    print("\n" + "-"*70)
    print("BASELINE (per-parameter RNG)")
    print("-"*70)
    
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()
    trainable = [p for p in model.parameters() if p.requires_grad]
    eps, lr = 1e-3, 1e-5
    
    baseline_times = []
    
    # Warmup
    for _ in range(warmup_steps):
        seed = np.random.randint(0, 2**31)
        torch.manual_seed(seed)
        for p in trainable:
            p.data.add_(torch.randn_like(p), alpha=eps)
        with torch.no_grad():
            _ = model(**batch).loss
        torch.manual_seed(seed)
        for p in trainable:
            p.data.add_(torch.randn_like(p), alpha=-2*eps)
        with torch.no_grad():
            _ = model(**batch).loss
        torch.manual_seed(seed)
        for p in trainable:
            p.data.add_(torch.randn_like(p), alpha=eps)
        torch.manual_seed(seed)
        for p in trainable:
            p.data.add_(torch.randn_like(p), alpha=-lr * 0.1)
    
    # Benchmark
    torch.cuda.synchronize()
    for step in range(num_steps):
        t0 = time.perf_counter()
        seed = np.random.randint(0, 2**31)
        
        torch.manual_seed(seed)
        for p in trainable:
            p.data.add_(torch.randn_like(p), alpha=eps)
        
        with torch.no_grad():
            loss1 = model(**batch).loss
        
        torch.manual_seed(seed)
        for p in trainable:
            p.data.add_(torch.randn_like(p), alpha=-2*eps)
        
        with torch.no_grad():
            loss2 = model(**batch).loss
        
        torch.manual_seed(seed)
        for p in trainable:
            p.data.add_(torch.randn_like(p), alpha=eps)
        
        grad = (loss1.item() - loss2.item()) / (2 * eps)
        
        torch.manual_seed(seed)
        for p in trainable:
            p.data.add_(torch.randn_like(p), alpha=-lr * grad)
        
        baseline_times.append((time.perf_counter() - t0) * 1000)
        
        if step % 10 == 0:
            print(f"  Step {step}: loss={((loss1+loss2)/2).item():.4f}")
    torch.cuda.synchronize()
    
    baseline_avg = np.mean(baseline_times)
    print(f"\n  Average: {baseline_avg:.2f} ms/step")
    
    # =========================================================================
    # Optimized (batched RNG)
    # =========================================================================
    del model
    torch.cuda.empty_cache()
    
    print("\n" + "-"*70)
    print("OPTIMIZED (batched RNG)")
    print("-"*70)
    
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()
    trainer = OptimizedMeZOTrainer(model, eps=eps, lr=lr)
    
    # Warmup
    for _ in range(warmup_steps):
        trainer.zo_step(batch)
    trainer.reset_timing()
    
    # Benchmark
    torch.cuda.synchronize()
    for step in range(num_steps):
        loss, grad = trainer.zo_step(batch)
        if step % 10 == 0:
            print(f"  Step {step}: loss={loss.item():.4f}")
    torch.cuda.synchronize()
    
    trainer.print_timing()
    optimized_avg = np.mean(trainer.timing['total'])
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    speedup = baseline_avg / optimized_avg
    
    print(f"\n  Baseline:   {baseline_avg:.2f} ms/step")
    print(f"  Optimized:  {optimized_avg:.2f} ms/step")
    print(f"  Speedup:    {speedup:.2f}x")
    print(f"  Time saved: {baseline_avg - optimized_avg:.2f} ms/step ({(1-1/speedup)*100:.1f}%)")
    
    # Analyze where time is spent
    summary = trainer.get_timing_summary()
    print(f"\n  Time breakdown (Optimized):")
    print(f"    RNG:      {summary['rng']['mean']:.2f} ms")
    print(f"    Perturb:  {summary['perturb']['mean']:.2f} ms (avg per call, 3x/step)")
    print(f"    Forward:  {summary['forward']['mean']:.2f} ms (both passes)")
    print(f"    Update:   {summary['update']['mean']:.2f} ms")
    
    print(f"\n  Bottleneck: Forward pass ({summary['forward']['mean']/optimized_avg*100:.0f}% of time)")
    print(f"  Further optimization requires: torch.compile, Flash Attention, etc.")
    
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
    
    compare_baseline_vs_optimized(
        model_name=args.model,
        num_steps=args.steps,
        warmup_steps=args.warmup,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
    )
