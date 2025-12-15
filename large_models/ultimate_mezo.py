#!/usr/bin/env python3
"""
Ultimate MeZO: Combining All Optimizations

Combines:
1. Fully Fused Flat Buffer - Single kernel for all 331M params
2. Async Pipelining - Overlap RNG generation with forward pass
3. Double Buffering - Prepare next Z while using current
4. Stream Overlap - Multiple CUDA streams for parallel execution

The idea:
- While GPU is doing forward pass 1, CPU/another stream prepares Z for next step
- While GPU is doing forward pass 2, we've already started generating next Z
- Perturbation is single-kernel on flat buffer (no iteration overhead)

Timeline visualization:
  Step N:   [RNG_N]──[Perturb+ε]──[Forward1]──[Perturb-2ε]──[Forward2]──[Update]
  Step N+1:                        [RNG_N+1 async]──────────────────────[use Z_N+1]

With double buffer:
  Buffer A: [Generate Z]...[Use for step N]...
  Buffer B:               [Generate Z async]...[Use for step N+1]...
"""

import torch
import torch.nn as nn
import time
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class UltimateMeZOConfig:
    eps: float = 1e-3
    lr: float = 1e-5
    use_double_buffer: bool = True
    use_async_rng: bool = True
    prefetch_batches: bool = False  # For data loading overlap


class UltimateMeZO:
    """
    Ultimate optimized MeZO combining all techniques:
    
    1. Flat parameter buffer (single-kernel perturbation)
    2. Async RNG generation (overlap with compute)
    3. Double buffering (swap between two Z buffers)
    4. CUDA stream pipelining
    """
    
    def __init__(self, model: nn.Module, config: Optional[UltimateMeZOConfig] = None):
        self.config = config or UltimateMeZOConfig()
        self.model = model
        
        # =====================================================================
        # 1. Setup flat parameter buffer (FULLY FUSED)
        # =====================================================================
        self.param_info = []  # (name, shape, numel, offset)
        total_numel = 0
        
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.param_info.append((name, p.shape, p.numel(), total_numel))
                total_numel += p.numel()
        
        self.total_numel = total_numel
        self.n_params = len(self.param_info)
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        
        # THE flat parameter buffer - all params in one contiguous tensor
        self.flat_params = torch.empty(total_numel, device=self.device, dtype=self.dtype)
        
        # Copy original params and create views
        self.param_views = {}
        for name, shape, numel, offset in self.param_info:
            orig_param = dict(model.named_parameters())[name]
            self.flat_params[offset:offset + numel].copy_(orig_param.data.flatten())
            self.param_views[name] = self.flat_params[offset:offset + numel].view(shape)
        
        # Replace model params with views
        self._replace_params_with_views()
        
        # =====================================================================
        # 2. Setup double buffering for Z vectors
        # =====================================================================
        if self.config.use_double_buffer:
            self.z_buffers = [
                torch.empty(total_numel, device=self.device, dtype=self.dtype),
                torch.empty(total_numel, device=self.device, dtype=self.dtype)
            ]
            self.current_buffer_idx = 0
            self.next_buffer_ready = False
        else:
            self.z_flat = torch.empty(total_numel, device=self.device, dtype=self.dtype)
        
        # =====================================================================
        # 3. Setup CUDA streams for async operations
        # =====================================================================
        if self.config.use_async_rng:
            self.compute_stream = torch.cuda.current_stream()
            self.rng_stream = torch.cuda.Stream()
            self.rng_done_event = torch.cuda.Event()
            self.compute_done_event = torch.cuda.Event()
        
        # Timing stats
        self.timing = {'rng': [], 'perturb': [], 'forward': [], 'update': [], 'total': []}
        self.step_count = 0
        
        print(f"[UltimateMeZO] {self.n_params} params → 1 flat buffer ({total_numel:,} elements)")
        print(f"[UltimateMeZO] Double buffering: {self.config.use_double_buffer}")
        print(f"[UltimateMeZO] Async RNG: {self.config.use_async_rng}")
        print(f"[UltimateMeZO] Kernels per perturb: 1 (vs {self.n_params * 2} baseline)")
    
    def _replace_params_with_views(self):
        """Replace model parameters with views into flat buffer"""
        for name, view in self.param_views.items():
            parts = name.split('.')
            module = self.model
            for part in parts[:-1]:
                module = getattr(module, part)
            setattr(module, parts[-1], nn.Parameter(view, requires_grad=True))
    
    @property
    def z_current(self) -> torch.Tensor:
        """Get current Z buffer"""
        if self.config.use_double_buffer:
            return self.z_buffers[self.current_buffer_idx]
        return self.z_flat
    
    @property
    def z_next(self) -> torch.Tensor:
        """Get next Z buffer (for prefetching)"""
        if self.config.use_double_buffer:
            return self.z_buffers[1 - self.current_buffer_idx]
        return self.z_flat
    
    def _generate_z_sync(self, seed: int, buffer: torch.Tensor):
        """Synchronous Z generation"""
        torch.manual_seed(seed)
        buffer.normal_()
    
    def _generate_z_async(self, seed: int, buffer: torch.Tensor):
        """Async Z generation on separate stream"""
        with torch.cuda.stream(self.rng_stream):
            torch.manual_seed(seed)
            buffer.normal_()
            self.rng_done_event.record()
    
    def _wait_for_z(self):
        """Wait for async Z generation to complete"""
        if self.config.use_async_rng:
            self.rng_done_event.synchronize()
    
    def _swap_buffers(self):
        """Swap double buffers"""
        if self.config.use_double_buffer:
            self.current_buffer_idx = 1 - self.current_buffer_idx
    
    def step(self, batch: Dict) -> Tuple[float, float]:
        """
        Single MeZO step with all optimizations.
        
        Kernel operations:
        1. z.normal_() - 1 kernel (async, overlapped with previous forward)
        2. flat_params.add_(z, eps) - 1 kernel
        3. Forward pass 1 - N kernels (model-dependent)
        4. flat_params.add_(z, -2*eps) - 1 kernel  
        5. Forward pass 2 - N kernels
        6. flat_params.add_(z, eps - lr*grad) - 1 kernel (fused restore+update)
        
        Total perturbation kernels: 3 (vs ~2300+ baseline)
        """
        t_start = time.perf_counter()
        
        seed = np.random.randint(0, 2**31)
        
        # =====================================================================
        # RNG Generation (async if enabled)
        # =====================================================================
        t_rng_start = time.perf_counter()
        
        if self.step_count == 0 or not self.config.use_double_buffer:
            # First step or no double buffer: must generate synchronously
            if self.config.use_async_rng:
                self._generate_z_async(seed, self.z_current)
                self._wait_for_z()
            else:
                self._generate_z_sync(seed, self.z_current)
        else:
            # Use pre-generated Z from previous step's async generation
            self._wait_for_z()  # Make sure it's ready
        
        t_rng_end = time.perf_counter()
        
        z = self.z_current
        
        # =====================================================================
        # Perturb +ε (SINGLE KERNEL for all 331M params!)
        # =====================================================================
        t_perturb_start = time.perf_counter()
        self.flat_params.add_(z, alpha=self.config.eps)
        
        # =====================================================================
        # Forward Pass 1
        # =====================================================================
        t_forward_start = time.perf_counter()
        
        # Start generating NEXT Z while forward pass runs (async pipeline)
        if self.config.use_double_buffer and self.config.use_async_rng:
            next_seed = np.random.randint(0, 2**31)
            self._generate_z_async(next_seed, self.z_next)
        
        with torch.no_grad():
            loss1 = self.model(**batch).loss
        
        t_forward1_end = time.perf_counter()
        
        # =====================================================================
        # Perturb -2ε (SINGLE KERNEL)
        # =====================================================================
        self.flat_params.add_(z, alpha=-2 * self.config.eps)
        t_perturb2_end = time.perf_counter()
        
        # =====================================================================
        # Forward Pass 2
        # =====================================================================
        with torch.no_grad():
            loss2 = self.model(**batch).loss
        
        t_forward2_end = time.perf_counter()
        
        # =====================================================================
        # Restore + Update (FUSED into SINGLE KERNEL!)
        # =====================================================================
        t_update_start = time.perf_counter()
        
        loss1_val = loss1.item()
        loss2_val = loss2.item()
        projected_grad = (loss1_val - loss2_val) / (2 * self.config.eps)
        
        # Fused: restore (+ε) and update (-lr*grad) in one add_
        # θ = θ - εz + εz - lr*grad*z = θ - lr*grad*z... wait, we need to restore first
        # Current state: θ - εz
        # Restore to θ: add εz
        # Update: subtract lr*grad*z
        # Combined: add (ε - lr*grad) * z
        self.flat_params.add_(z, alpha=self.config.eps - self.config.lr * projected_grad)
        
        t_update_end = time.perf_counter()
        
        # Swap buffers for next iteration
        if self.config.use_double_buffer:
            self._swap_buffers()
        
        t_end = time.perf_counter()
        
        # Record timing
        self.timing['rng'].append((t_rng_end - t_rng_start) * 1000)
        self.timing['perturb'].append(
            ((t_forward_start - t_perturb_start) + (t_perturb2_end - t_forward1_end)) * 1000
        )
        self.timing['forward'].append(
            ((t_forward1_end - t_forward_start) + (t_forward2_end - t_perturb2_end)) * 1000
        )
        self.timing['update'].append((t_update_end - t_update_start) * 1000)
        self.timing['total'].append((t_end - t_start) * 1000)
        
        self.step_count += 1
        
        return (loss1_val + loss2_val) / 2, projected_grad
    
    def get_timing_summary(self) -> str:
        lines = ["\n" + "=" * 60, "ULTIMATE MEZO TIMING", "=" * 60]
        total = np.mean(self.timing['total']) if self.timing['total'] else 1
        for key in ['rng', 'perturb', 'forward', 'update', 'total']:
            if self.timing[key]:
                mean, std = np.mean(self.timing[key]), np.std(self.timing[key])
                pct = mean / total * 100 if key != 'total' else 100
                lines.append(f"  {key:12s}: {mean:8.3f} ± {std:.3f} ms ({pct:5.1f}%)")
        return "\n".join(lines)


# =============================================================================
# Baseline for comparison
# =============================================================================

def baseline_mezo_step(model, params, batch, eps=1e-3, lr=1e-5):
    """Original per-parameter MeZO"""
    seed = np.random.randint(0, 2**31)
    
    # Perturb +ε (388 randn + 388 add_ = 776 kernels)
    torch.manual_seed(seed)
    for p in params:
        z = torch.randn_like(p)
        p.data.add_(z, alpha=eps)
    
    with torch.no_grad():
        loss1 = model(**batch).loss
    
    # Perturb -2ε (776 more kernels)
    torch.manual_seed(seed)
    for p in params:
        z = torch.randn_like(p)
        p.data.add_(z, alpha=-2*eps)
    
    with torch.no_grad():
        loss2 = model(**batch).loss
    
    # Restore + update (776 more kernels)
    projected_grad = (loss1.item() - loss2.item()) / (2 * eps)
    torch.manual_seed(seed)
    for p in params:
        z = torch.randn_like(p)
        p.data.add_(z, alpha=eps - lr * projected_grad)
    
    return (loss1.item() + loss2.item()) / 2


# =============================================================================
# Benchmark
# =============================================================================

def benchmark(model_name: str = "facebook/opt-350m", n_steps: int = 30, warmup: int = 5):
    from transformers import AutoModelForCausalLM
    
    print("=" * 70)
    print("ULTIMATE MEZO BENCHMARK: ALL OPTIMIZATIONS COMBINED")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Steps: {n_steps}, Warmup: {warmup}")
    
    batch_size, seq_len = 4, 128
    batch = {
        'input_ids': torch.randint(0, 1000, (batch_size, seq_len), device='cuda'),
        'attention_mask': torch.ones(batch_size, seq_len, device='cuda', dtype=torch.long),
        'labels': torch.randint(0, 1000, (batch_size, seq_len), device='cuda'),
    }
    
    results = {}
    
    # =========================================================================
    # 1. Baseline (per-parameter, no optimizations)
    # =========================================================================
    print("\n" + "-" * 70)
    print("1. BASELINE (per-parameter RNG, no optimization)")
    print("-" * 70)
    
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).cuda()
    model.eval()
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"   {len(params)} params, {sum(p.numel() for p in params):,} elements")
    
    times = []
    for i in range(n_steps + warmup):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        loss = baseline_mezo_step(model, params, batch)
        torch.cuda.synchronize()
        if i >= warmup:
            times.append((time.perf_counter() - t0) * 1000)
            if i == warmup:
                print(f"   Step 0: loss={loss:.4f}")
    
    baseline_time = np.mean(times)
    baseline_std = np.std(times)
    results['baseline'] = baseline_time
    print(f"   Average: {baseline_time:.2f} ± {baseline_std:.2f} ms/step")
    
    del model
    torch.cuda.empty_cache()
    
    # =========================================================================
    # 2. Flat buffer only (no async)
    # =========================================================================
    print("\n" + "-" * 70)
    print("2. FLAT BUFFER ONLY (single-kernel perturb, no async)")
    print("-" * 70)
    
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).cuda()
    model.eval()
    
    trainer = UltimateMeZO(model, UltimateMeZOConfig(
        use_double_buffer=False,
        use_async_rng=False
    ))
    
    for i in range(n_steps + warmup):
        torch.cuda.synchronize()
        loss, _ = trainer.step(batch)
        torch.cuda.synchronize()
        if i == warmup:
            print(f"   Step 0: loss={loss:.4f}")
    
    trainer.timing = {k: v[warmup:] for k, v in trainer.timing.items()}
    flat_time = np.mean(trainer.timing['total'])
    results['flat_only'] = flat_time
    print(trainer.get_timing_summary())
    
    del model, trainer
    torch.cuda.empty_cache()
    
    # =========================================================================
    # 3. Flat buffer + async RNG
    # =========================================================================
    print("\n" + "-" * 70)
    print("3. FLAT BUFFER + ASYNC RNG (stream overlap)")
    print("-" * 70)
    
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).cuda()
    model.eval()
    
    trainer = UltimateMeZO(model, UltimateMeZOConfig(
        use_double_buffer=False,
        use_async_rng=True
    ))
    
    for i in range(n_steps + warmup):
        torch.cuda.synchronize()
        loss, _ = trainer.step(batch)
        torch.cuda.synchronize()
        if i == warmup:
            print(f"   Step 0: loss={loss:.4f}")
    
    trainer.timing = {k: v[warmup:] for k, v in trainer.timing.items()}
    async_time = np.mean(trainer.timing['total'])
    results['flat_async'] = async_time
    print(trainer.get_timing_summary())
    
    del model, trainer
    torch.cuda.empty_cache()
    
    # =========================================================================
    # 4. ULTIMATE: Flat buffer + async RNG + double buffer
    # =========================================================================
    print("\n" + "-" * 70)
    print("4. ULTIMATE (flat buffer + async RNG + double buffer)")
    print("-" * 70)
    
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).cuda()
    model.eval()
    
    trainer = UltimateMeZO(model, UltimateMeZOConfig(
        use_double_buffer=True,
        use_async_rng=True
    ))
    
    for i in range(n_steps + warmup):
        torch.cuda.synchronize()
        loss, _ = trainer.step(batch)
        torch.cuda.synchronize()
        if i == warmup:
            print(f"   Step 0: loss={loss:.4f}")
    
    trainer.timing = {k: v[warmup:] for k, v in trainer.timing.items()}
    ultimate_time = np.mean(trainer.timing['total'])
    results['ultimate'] = ultimate_time
    print(trainer.get_timing_summary())
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print(f"\n  {'Method':<45} {'Time (ms)':>12} {'Speedup':>10}")
    print("  " + "-" * 67)
    
    best = min(results.values())
    for method, t in results.items():
        speedup = baseline_time / t
        marker = " ★" if t == best else ""
        print(f"  {method:<45} {t:>12.2f} {speedup:>9.2f}x{marker}")
    
    print("\n" + "-" * 70)
    print("OPTIMIZATION BREAKDOWN:")
    print("-" * 70)
    print(f"  Baseline → Flat buffer:      {baseline_time/results['flat_only']:.2f}x")
    print(f"  Flat buffer → + Async:       {results['flat_only']/results['flat_async']:.2f}x")
    print(f"  + Async → + Double buffer:   {results['flat_async']/results['ultimate']:.2f}x")
    print(f"  Total improvement:           {baseline_time/results['ultimate']:.2f}x")
    
    print("\n" + "-" * 70)
    print("KERNEL LAUNCH REDUCTION:")
    print("-" * 70)
    n_params = 388
    print(f"  Baseline per perturb:   {n_params} randn + {n_params} add_ = {n_params*2} kernels")
    print(f"  Optimized per perturb:  1 normal_ + 1 add_ = 2 kernels")
    print(f"  Per step (3 perturbs):  {n_params*2*3} → 6 kernels ({n_params*2*3/6:.0f}x reduction)")
    
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="facebook/opt-350m")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()
    
    benchmark(args.model, args.steps, args.warmup)
