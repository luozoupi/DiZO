#!/usr/bin/env python3
"""
Profile Individual Contributions of Each Optimization

Tests each optimization INDIVIDUALLY to measure its exact contribution:
1. Baseline (per-parameter RNG)
2. Batched RNG only
3. Flat buffer only
4. Async/Double buffer only
5. Triton kernels only
6. CUDA kernels only
7. torch.compile only
8. Combined: Flat buffer + Async
9. Combined: Flat buffer + Triton
10. Combined: All (except torch.compile)
11. Combined: All (with torch.compile)

Each test measures:
- Wall time per step
- Kernel launch count (via profiler)
- Time breakdown (RNG, perturb, forward, update)
"""

import torch
import torch.nn as nn
import time
import numpy as np
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
import gc

# Triton import
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    print("[Warning] Triton not available")


# =============================================================================
# Triton Kernels
# =============================================================================

if HAS_TRITON:
    @triton.jit
    def perturb_kernel(
        params_ptr, z_ptr, n_elements,
        alpha,
        BLOCK_SIZE: tl.constexpr
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        params = tl.load(params_ptr + offsets, mask=mask)
        z = tl.load(z_ptr + offsets, mask=mask)
        result = params + alpha * z
        tl.store(params_ptr + offsets, result, mask=mask)
    
    def triton_perturb(params: torch.Tensor, z: torch.Tensor, alpha: float):
        n = params.numel()
        BLOCK_SIZE = 1024
        grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
        perturb_kernel[grid](params, z, n, alpha, BLOCK_SIZE=BLOCK_SIZE)


# =============================================================================
# Test Configurations
# =============================================================================

@dataclass
class TestConfig:
    name: str
    use_flat_buffer: bool = False
    use_batched_rng: bool = False
    use_async_rng: bool = False
    use_double_buffer: bool = False
    use_triton: bool = False
    use_torch_compile: bool = False


CONFIGS = [
    TestConfig("1. baseline"),
    TestConfig("2. batched_rng", use_batched_rng=True),
    TestConfig("3. flat_buffer", use_flat_buffer=True),
    TestConfig("4. async_double_buf", use_async_rng=True, use_double_buffer=True),
    TestConfig("5. triton_kernels", use_triton=True),
    TestConfig("6. flat+batched", use_flat_buffer=True, use_batched_rng=True),
    TestConfig("7. flat+async", use_flat_buffer=True, use_async_rng=True, use_double_buffer=True),
    TestConfig("8. flat+triton", use_flat_buffer=True, use_triton=True),
    TestConfig("9. all_no_compile", use_flat_buffer=True, use_async_rng=True, use_double_buffer=True, use_triton=True),
    TestConfig("10. torch_compile", use_torch_compile=True),
    TestConfig("11. flat+compile", use_flat_buffer=True, use_torch_compile=True),
    TestConfig("12. all_with_compile", use_flat_buffer=True, use_async_rng=True, use_double_buffer=True, use_torch_compile=True),
]


# =============================================================================
# MeZO Trainer with Configurable Optimizations
# =============================================================================

class ConfigurableMeZO:
    """MeZO trainer with individually toggleable optimizations"""
    
    def __init__(self, model: nn.Module, config: TestConfig, eps: float = 1e-3, lr: float = 1e-5):
        self.config = config
        self.eps = eps
        self.lr = lr
        self.model = model
        
        # Get parameters
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.n_params = len(self.params)
        self.total_numel = sum(p.numel() for p in self.params)
        self.device = self.params[0].device
        self.dtype = self.params[0].dtype
        
        # Setup based on config
        if config.use_flat_buffer:
            self._setup_flat_buffer()
        
        if config.use_async_rng or config.use_double_buffer:
            self._setup_async()
        
        # Timing
        self.timing = {'rng': [], 'perturb': [], 'forward': [], 'update': [], 'total': []}
    
    def _setup_flat_buffer(self):
        """Setup flat parameter buffer"""
        self.flat_params = torch.empty(self.total_numel, device=self.device, dtype=self.dtype)
        self.z_flat = torch.empty(self.total_numel, device=self.device, dtype=self.dtype)
        
        # Copy params and create views
        self.param_views = []
        offset = 0
        param_info = []
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                param_info.append((name, p.shape, p.numel(), offset))
                self.flat_params[offset:offset + p.numel()].copy_(p.data.flatten())
                self.param_views.append(self.flat_params[offset:offset + p.numel()].view(p.shape))
                offset += p.numel()
        
        # Replace model params with views
        for (name, shape, numel, off), view in zip(param_info, self.param_views):
            parts = name.split('.')
            mod = self.model
            for part in parts[:-1]:
                mod = getattr(mod, part)
            setattr(mod, parts[-1], nn.Parameter(view, requires_grad=True))
        
        # Update params list to point to views
        self.params = [p for p in self.model.parameters() if p.requires_grad]
    
    def _setup_async(self):
        """Setup async RNG and double buffering"""
        if self.config.use_double_buffer:
            if self.config.use_flat_buffer:
                self.z_buffers = [
                    torch.empty(self.total_numel, device=self.device, dtype=self.dtype),
                    torch.empty(self.total_numel, device=self.device, dtype=self.dtype)
                ]
            else:
                # Per-param double buffer (less efficient but for testing)
                self.z_buffers = [
                    [torch.empty_like(p) for p in self.params],
                    [torch.empty_like(p) for p in self.params]
                ]
            self.buf_idx = 0
        
        if self.config.use_async_rng:
            self.rng_stream = torch.cuda.Stream()
            self.rng_event = torch.cuda.Event()
    
    def _generate_rng(self, seed: int):
        """Generate random perturbation"""
        torch.manual_seed(seed)
        
        if self.config.use_flat_buffer:
            if self.config.use_double_buffer:
                buf = self.z_buffers[self.buf_idx]
            else:
                buf = self.z_flat
            
            if self.config.use_async_rng:
                with torch.cuda.stream(self.rng_stream):
                    buf.normal_()
                    self.rng_event.record()
            else:
                buf.normal_()
        else:
            # Per-parameter RNG
            if self.config.use_batched_rng:
                # Still per-param but pre-allocate
                if self.config.use_double_buffer:
                    for z in self.z_buffers[self.buf_idx]:
                        z.normal_()
                else:
                    if not hasattr(self, 'z_list'):
                        self.z_list = [torch.empty_like(p) for p in self.params]
                    for z in self.z_list:
                        z.normal_()
            # else: generate on-the-fly in perturb
    
    def _wait_rng(self):
        """Wait for async RNG"""
        if self.config.use_async_rng:
            self.rng_event.synchronize()
    
    def _get_z(self, idx: int = None):
        """Get perturbation vector(s)"""
        if self.config.use_flat_buffer:
            if self.config.use_double_buffer:
                return self.z_buffers[self.buf_idx]
            return self.z_flat
        else:
            if self.config.use_double_buffer:
                return self.z_buffers[self.buf_idx]
            elif self.config.use_batched_rng:
                return self.z_list
            return None  # Generate on-the-fly
    
    def _perturb(self, alpha: float, seed: int = None):
        """Apply perturbation"""
        if self.config.use_flat_buffer:
            z = self._get_z()
            if self.config.use_triton and HAS_TRITON:
                triton_perturb(self.flat_params, z, alpha)
            else:
                self.flat_params.add_(z, alpha=alpha)
        else:
            z_list = self._get_z()
            if z_list is not None:
                # Pre-generated Z
                for p, z in zip(self.params, z_list):
                    p.data.add_(z, alpha=alpha)
            else:
                # On-the-fly generation (baseline)
                if seed is not None:
                    torch.manual_seed(seed)
                for p in self.params:
                    z = torch.randn_like(p)
                    p.data.add_(z, alpha=alpha)
    
    def _swap_buffers(self):
        """Swap double buffers"""
        if self.config.use_double_buffer:
            self.buf_idx = 1 - self.buf_idx
    
    def step(self, batch: Dict) -> float:
        """Single MeZO step"""
        t_start = time.perf_counter()
        
        seed = np.random.randint(0, 2**31)
        
        # RNG generation
        t_rng = time.perf_counter()
        if self.config.use_batched_rng or self.config.use_flat_buffer:
            self._generate_rng(seed)
            if self.config.use_async_rng:
                self._wait_rng()
        t_rng_end = time.perf_counter()
        
        # Perturb +ε
        t_perturb = time.perf_counter()
        if self.config.use_batched_rng or self.config.use_flat_buffer:
            self._perturb(self.eps)
        else:
            self._perturb(self.eps, seed)
        
        # Forward 1
        t_forward = time.perf_counter()
        
        # Start next Z generation (async overlap)
        if self.config.use_async_rng and self.config.use_double_buffer:
            next_seed = np.random.randint(0, 2**31)
            self._swap_buffers()
            with torch.cuda.stream(self.rng_stream):
                torch.manual_seed(next_seed)
                if self.config.use_flat_buffer:
                    self.z_buffers[self.buf_idx].normal_()
                else:
                    for z in self.z_buffers[self.buf_idx]:
                        z.normal_()
                self.rng_event.record()
            self._swap_buffers()  # Swap back for this step
        
        with torch.no_grad():
            loss1 = self.model(**batch).loss
        
        # Perturb -2ε
        if self.config.use_batched_rng or self.config.use_flat_buffer:
            self._perturb(-2 * self.eps)
        else:
            self._perturb(-2 * self.eps, seed)
        t_perturb_end = time.perf_counter()
        
        # Forward 2
        with torch.no_grad():
            loss2 = self.model(**batch).loss
        t_forward_end = time.perf_counter()
        
        # Update
        t_update = time.perf_counter()
        projected_grad = (loss1.item() - loss2.item()) / (2 * self.eps)
        
        # Restore + update (fused)
        if self.config.use_batched_rng or self.config.use_flat_buffer:
            self._perturb(self.eps - self.lr * projected_grad)
        else:
            self._perturb(self.eps - self.lr * projected_grad, seed)
        t_update_end = time.perf_counter()
        
        # Swap buffers for next iteration
        if self.config.use_double_buffer:
            self._swap_buffers()
        
        t_end = time.perf_counter()
        
        # Record timing
        self.timing['rng'].append((t_rng_end - t_rng) * 1000)
        self.timing['perturb'].append(((t_forward - t_perturb) + (t_perturb_end - t_forward) * 0.5) * 1000)
        self.timing['forward'].append((t_forward_end - t_forward) * 1000)
        self.timing['update'].append((t_update_end - t_update) * 1000)
        self.timing['total'].append((t_end - t_start) * 1000)
        
        return (loss1.item() + loss2.item()) / 2


def count_cuda_kernels(func, *args, **kwargs):
    """Count CUDA kernel launches using profiler"""
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False,
    ) as prof:
        result = func(*args, **kwargs)
    
    # Count actual kernel launches (not just unique kernels)
    kernel_count = 0
    for event in prof.key_averages():
        if event.device_type == torch.autograd.DeviceType.CUDA:
            kernel_count += event.count
    
    return result, kernel_count


def run_single_test(
    config: TestConfig,
    model_name: str,
    batch: Dict,
    n_steps: int,
    warmup: int
) -> Dict:
    """Run a single test configuration"""
    from transformers import AutoModelForCausalLM
    
    # Load fresh model
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    model = model.cuda()
    model.eval()
    
    # Apply torch.compile if needed
    if config.use_torch_compile:
        model = torch.compile(model, mode='default', dynamic=True)
    
    # Create trainer
    trainer = ConfigurableMeZO(model, config)
    
    # Warmup
    for _ in range(warmup):
        trainer.step(batch)
    
    # Clear timing from warmup
    trainer.timing = {'rng': [], 'perturb': [], 'forward': [], 'update': [], 'total': []}
    
    # Count kernels on one step
    def one_step():
        return trainer.step(batch)
    
    _, kernel_count = count_cuda_kernels(one_step)
    
    # Run timed steps
    for _ in range(n_steps):
        torch.cuda.synchronize()
        trainer.step(batch)
        torch.cuda.synchronize()
    
    # Collect results
    result = {
        'name': config.name,
        'time_ms': np.mean(trainer.timing['total']),
        'time_std': np.std(trainer.timing['total']),
        'kernels': kernel_count,
        'breakdown': {
            'rng': np.mean(trainer.timing['rng']),
            'perturb': np.mean(trainer.timing['perturb']),
            'forward': np.mean(trainer.timing['forward']),
            'update': np.mean(trainer.timing['update']),
        }
    }
    
    # Cleanup
    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()
    
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="facebook/opt-350m")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=128)
    args = parser.parse_args()
    
    # Enable TF32
    torch.set_float32_matmul_precision('high')
    
    print("=" * 80)
    print("INDIVIDUAL CONTRIBUTION PROFILING")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Steps: {args.steps}, Warmup: {args.warmup}")
    print(f"Batch: {args.batch_size} x {args.seq_len}")
    print(f"Triton available: {HAS_TRITON}")
    print("=" * 80)
    
    # Create batch
    batch = {
        'input_ids': torch.randint(0, 1000, (args.batch_size, args.seq_len), device='cuda'),
        'attention_mask': torch.ones(args.batch_size, args.seq_len, device='cuda', dtype=torch.long),
        'labels': torch.randint(0, 1000, (args.batch_size, args.seq_len), device='cuda'),
    }
    
    results = []
    
    for config in CONFIGS:
        # Skip triton tests if not available
        if config.use_triton and not HAS_TRITON:
            print(f"\n[SKIP] {config.name} (Triton not available)")
            continue
        
        print(f"\n{'─' * 80}")
        print(f"Testing: {config.name}")
        print(f"  flat_buffer={config.use_flat_buffer}, batched_rng={config.use_batched_rng}")
        print(f"  async={config.use_async_rng}, double_buf={config.use_double_buffer}")
        print(f"  triton={config.use_triton}, compile={config.use_torch_compile}")
        print(f"{'─' * 80}")
        
        try:
            result = run_single_test(config, args.model, batch, args.steps, args.warmup)
            results.append(result)
            
            print(f"  Time: {result['time_ms']:.2f} ± {result['time_std']:.2f} ms/step")
            print(f"  Kernels: {result['kernels']}")
            print(f"  Breakdown: rng={result['breakdown']['rng']:.2f}ms, "
                  f"perturb={result['breakdown']['perturb']:.2f}ms, "
                  f"forward={result['breakdown']['forward']:.2f}ms, "
                  f"update={result['breakdown']['update']:.2f}ms")
        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback
            traceback.print_exc()
    
    # Summary table
    if results:
        baseline_time = results[0]['time_ms']
        baseline_kernels = results[0]['kernels']
        
        print("\n" + "=" * 80)
        print("SUMMARY: INDIVIDUAL CONTRIBUTIONS")
        print("=" * 80)
        print(f"\n{'Configuration':<35} {'Time (ms)':>12} {'Speedup':>10} {'Kernels':>10} {'K.Reduce':>10}")
        print("─" * 77)
        
        for r in results:
            speedup = baseline_time / r['time_ms']
            k_reduce = baseline_kernels / r['kernels'] if r['kernels'] > 0 else 0
            best_time = " ★" if r['time_ms'] == min(x['time_ms'] for x in results) else ""
            print(f"{r['name']:<35} {r['time_ms']:>12.2f} {speedup:>9.2f}x {r['kernels']:>10} {k_reduce:>9.1f}x{best_time}")
        
        # Contribution analysis
        print("\n" + "=" * 80)
        print("CONTRIBUTION ANALYSIS (vs baseline)")
        print("=" * 80)
        
        baseline = results[0]['time_ms']
        contributions = []
        
        for r in results[1:]:
            saved = baseline - r['time_ms']
            pct = saved / baseline * 100
            contributions.append((r['name'], saved, pct))
        
        contributions.sort(key=lambda x: -x[1])
        
        print(f"\n{'Optimization':<35} {'Time Saved':>12} {'% of Baseline':>15}")
        print("─" * 62)
        for name, saved, pct in contributions:
            print(f"{name:<35} {saved:>12.2f} ms {pct:>14.1f}%")
        
        # Best combination without torch.compile
        no_compile = [r for r in results if 'compile' not in r['name'].lower()]
        if no_compile:
            best_no_compile = min(no_compile, key=lambda x: x['time_ms'])
            print(f"\nBest without torch.compile: {best_no_compile['name']}")
            print(f"  Time: {best_no_compile['time_ms']:.2f} ms ({baseline/best_no_compile['time_ms']:.2f}x speedup)")
        
        # Best overall
        best = min(results, key=lambda x: x['time_ms'])
        print(f"\nBest overall: {best['name']}")
        print(f"  Time: {best['time_ms']:.2f} ms ({baseline/best['time_ms']:.2f}x speedup)")
        
        print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
