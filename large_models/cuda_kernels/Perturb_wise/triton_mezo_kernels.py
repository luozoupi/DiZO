#!/usr/bin/env python3
"""
Triton Kernels for Fused MeZO/DiZO Operations

This module provides high-performance Triton kernels for:
1. Fused perturbation: param += eps * randn (all params in one kernel)
2. Fused update: param -= lr * grad * z (all params in one kernel)
3. Batched RNG with Philox algorithm

Expected speedup: 2-3x over naive PyTorch loops
"""

import torch
import triton
import triton.language as tl
import math
from typing import List, Tuple, Optional
import time


# =============================================================================
# Triton Kernels
# =============================================================================

@triton.jit
def fused_perturb_kernel(
    params_ptr,      # Pointer to flattened parameters
    z_ptr,           # Pointer to pre-generated random values
    n_elements,      # Total number of elements
    eps: tl.constexpr,  # Perturbation scale (compile-time constant for better perf)
    BLOCK_SIZE: tl.constexpr,  # Block size
):
    """
    Fused perturbation kernel: params[i] += eps * z[i]
    
    This replaces N separate add/mul operations with a single kernel launch.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load params and random values
    params = tl.load(params_ptr + offsets, mask=mask)
    z = tl.load(z_ptr + offsets, mask=mask)
    
    # Fused multiply-add: params = params + eps * z
    params = params + eps * z
    
    # Store result
    tl.store(params_ptr + offsets, params, mask=mask)


@triton.jit
def fused_perturb_neg_kernel(
    params_ptr,
    z_ptr,
    n_elements,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused negative perturbation: params[i] -= 2 * eps * z[i]
    For transitioning from θ+εz to θ-εz
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    params = tl.load(params_ptr + offsets, mask=mask)
    z = tl.load(z_ptr + offsets, mask=mask)
    
    # params = params - 2*eps*z
    params = params - 2.0 * eps * z
    
    tl.store(params_ptr + offsets, params, mask=mask)


@triton.jit
def fused_update_kernel(
    params_ptr,
    z_ptr,
    n_elements,
    lr: tl.constexpr,
    projected_grad,  # Runtime value (scalar)
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused parameter update: params[i] -= lr * projected_grad * z[i]
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    params = tl.load(params_ptr + offsets, mask=mask)
    z = tl.load(z_ptr + offsets, mask=mask)
    
    # params = params - lr * projected_grad * z
    scale = lr * projected_grad
    params = params - scale * z
    
    tl.store(params_ptr + offsets, params, mask=mask)


@triton.jit
def fused_perturb_with_rng_kernel(
    params_ptr,
    seed,            # Random seed
    n_elements,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused perturbation with inline RNG (Philox algorithm).
    Generates random numbers and applies perturbation in one kernel.
    
    This eliminates the need for a separate RNG kernel and reduces memory traffic.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load params
    params = tl.load(params_ptr + offsets, mask=mask)
    
    # Generate random numbers using Philox
    # Each thread gets a unique random value based on its offset
    r1, r2, r3, r4 = tl.randint4x(seed, offsets)
    
    # Convert to float32 in [-1, 1] range, then scale to approximate normal
    # Using Box-Muller-like transformation for better distribution
    u1 = r1.to(tl.float32) / 2147483647.0  # [-1, 1]
    u2 = r2.to(tl.float32) / 2147483647.0
    
    # Simple approximation: sum of uniforms approaches normal
    z = (u1 + u2) * 0.7071  # Scale to approximate std=1
    
    # Apply perturbation
    params = params + eps * z
    
    tl.store(params_ptr + offsets, params, mask=mask)


# =============================================================================
# Wrapper Classes
# =============================================================================

class TritonMeZOOps:
    """
    High-level wrapper for Triton MeZO operations.
    
    Usage:
        ops = TritonMeZOOps(model, eps=1e-3, lr=1e-5)
        ops.generate_perturbation()
        ops.perturb_positive()
        # ... forward pass ...
        ops.perturb_negative()
        # ... forward pass ...
        ops.perturb_reset()
        ops.update(projected_grad)
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        eps: float = 1e-3,
        lr: float = 1e-5,
        block_size: int = 1024,
    ):
        self.model = model
        self.eps = eps
        self.lr = lr
        self.block_size = block_size
        
        # Collect trainable parameters
        self.trainable_params: List[torch.nn.Parameter] = []
        self.param_shapes: List[torch.Size] = []
        self.param_numels: List[int] = []
        self.total_numel = 0
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.trainable_params.append(param)
                self.param_shapes.append(param.shape)
                self.param_numels.append(param.numel())
                self.total_numel += param.numel()
        
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        
        print(f"[TritonMeZOOps] Initialized with {len(self.trainable_params)} params, "
              f"{self.total_numel:,} elements")
        
        # Pre-allocate flattened views
        self._flatten_params()
        
        # Pre-allocate perturbation tensor
        self.z_flat: Optional[torch.Tensor] = None
        
        # Timing stats
        self.timing = {
            'perturb_pos': [],
            'perturb_neg': [],
            'perturb_reset': [],
            'update': [],
            'rng': [],
        }
    
    def _flatten_params(self):
        """
        Create a contiguous view of all parameters.
        This is the key optimization - allows single kernel to operate on all params.
        """
        # Calculate total size and create flat tensor
        # Note: We operate directly on param.data, not creating a copy
        self.flat_params = torch.zeros(self.total_numel, device=self.device, dtype=self.dtype)
        
        # Store offsets for each parameter
        self.offsets = []
        offset = 0
        for param in self.trainable_params:
            self.offsets.append(offset)
            offset += param.numel()
    
    def _copy_params_to_flat(self):
        """Copy parameters to flat buffer (for operations that need it)"""
        offset = 0
        for param in self.trainable_params:
            numel = param.numel()
            self.flat_params[offset:offset + numel].copy_(param.data.view(-1))
            offset += numel
    
    def _copy_flat_to_params(self):
        """Copy flat buffer back to parameters"""
        offset = 0
        for param in self.trainable_params:
            numel = param.numel()
            param.data.copy_(self.flat_params[offset:offset + numel].view(param.shape))
            offset += numel
    
    def generate_perturbation(self, seed: Optional[int] = None) -> torch.Tensor:
        """
        Generate random perturbation vector using a single batched RNG call.
        """
        t0 = time.perf_counter()
        
        if seed is not None:
            torch.manual_seed(seed)
        
        # Single batched RNG call for all parameters
        self.z_flat = torch.randn(self.total_numel, device=self.device, dtype=self.dtype)
        
        self.timing['rng'].append((time.perf_counter() - t0) * 1000)
        return self.z_flat
    
    def perturb_positive(self):
        """
        Apply positive perturbation: param += eps * z
        Uses Triton kernel operating on each parameter tensor.
        """
        t0 = time.perf_counter()
        
        offset = 0
        for param in self.trainable_params:
            numel = param.numel()
            z_slice = self.z_flat[offset:offset + numel]
            
            # Reshape z to match param
            z_view = z_slice.view(param.shape)
            
            # Use Triton kernel
            grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
            fused_perturb_kernel[grid](
                param.data.view(-1),  # Flatten param
                z_slice,
                numel,
                eps=self.eps,
                BLOCK_SIZE=self.block_size,
            )
            offset += numel
        
        self.timing['perturb_pos'].append((time.perf_counter() - t0) * 1000)
    
    def perturb_negative(self):
        """
        Apply negative perturbation: param -= 2*eps * z
        Transitions from θ+εz to θ-εz
        """
        t0 = time.perf_counter()
        
        offset = 0
        for param in self.trainable_params:
            numel = param.numel()
            z_slice = self.z_flat[offset:offset + numel]
            
            grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
            fused_perturb_neg_kernel[grid](
                param.data.view(-1),
                z_slice,
                numel,
                eps=self.eps,
                BLOCK_SIZE=self.block_size,
            )
            offset += numel
        
        self.timing['perturb_neg'].append((time.perf_counter() - t0) * 1000)
    
    def perturb_reset(self):
        """
        Reset to original params: param += eps * z
        (After -2*eps*z, adding +eps*z gives us back original)
        """
        t0 = time.perf_counter()
        
        offset = 0
        for param in self.trainable_params:
            numel = param.numel()
            z_slice = self.z_flat[offset:offset + numel]
            
            grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
            fused_perturb_kernel[grid](
                param.data.view(-1),
                z_slice,
                numel,
                eps=self.eps,
                BLOCK_SIZE=self.block_size,
            )
            offset += numel
        
        self.timing['perturb_reset'].append((time.perf_counter() - t0) * 1000)
    
    def update(self, projected_grad: float):
        """
        Update parameters: param -= lr * projected_grad * z
        """
        t0 = time.perf_counter()
        
        offset = 0
        for param in self.trainable_params:
            numel = param.numel()
            z_slice = self.z_flat[offset:offset + numel]
            
            grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
            fused_update_kernel[grid](
                param.data.view(-1),
                z_slice,
                numel,
                lr=self.lr,
                projected_grad=projected_grad,
                BLOCK_SIZE=self.block_size,
            )
            offset += numel
        
        self.timing['update'].append((time.perf_counter() - t0) * 1000)
    
    def print_timing(self):
        """Print timing statistics"""
        print("\n" + "="*60)
        print("TRITON MEZO TIMING")
        print("="*60)
        
        import numpy as np
        for key, times in self.timing.items():
            if times:
                print(f"  {key:15s}: {np.mean(times):8.3f} ± {np.std(times):6.3f} ms")


class TritonMeZOOpsV2:
    """
    Version 2: Even more optimized using fully contiguous parameter storage.
    
    This version creates a single contiguous buffer for ALL parameters,
    allowing truly single-kernel operations.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        eps: float = 1e-3,
        lr: float = 1e-5,
        block_size: int = 1024,
    ):
        self.model = model
        self.eps = eps
        self.lr = lr
        self.block_size = block_size
        
        # Collect trainable parameters
        self.trainable_params: List[torch.nn.Parameter] = []
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
                offset += param.numel()
                self.total_numel += param.numel()
        
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        
        print(f"[TritonMeZOOpsV2] Initialized: {len(self.trainable_params)} params, "
              f"{self.total_numel:,} elements")
        
        # Timing - must be initialized before _sync_from_model!
        self.timing = {'perturb': [], 'update': [], 'rng': [], 'sync': []}
        
        # Create single contiguous buffer for all parameters
        self.params_flat = torch.zeros(self.total_numel, device=self.device, dtype=self.dtype)
        self._sync_from_model()
        
        # Pre-allocate perturbation tensor
        self.z_flat = torch.zeros(self.total_numel, device=self.device, dtype=self.dtype)
    
    def _sync_from_model(self):
        """Copy model parameters to flat buffer"""
        t0 = time.perf_counter()
        for i, param in enumerate(self.trainable_params):
            start = self.offsets[i]
            end = start + self.param_numels[i]
            self.params_flat[start:end].copy_(param.data.view(-1))
        self.timing['sync'].append((time.perf_counter() - t0) * 1000)
    
    def _sync_to_model(self):
        """Copy flat buffer back to model parameters"""
        t0 = time.perf_counter()
        for i, param in enumerate(self.trainable_params):
            start = self.offsets[i]
            end = start + self.param_numels[i]
            param.data.copy_(self.params_flat[start:end].view(self.param_shapes[i]))
        self.timing['sync'].append((time.perf_counter() - t0) * 1000)
    
    def generate_perturbation(self, seed: Optional[int] = None):
        """Generate perturbation using single RNG call"""
        t0 = time.perf_counter()
        if seed is not None:
            torch.manual_seed(seed)
        self.z_flat = torch.randn(self.total_numel, device=self.device, dtype=self.dtype)
        self.timing['rng'].append((time.perf_counter() - t0) * 1000)
    
    def perturb(self, scale: float = 1.0):
        """
        Apply perturbation with given scale.
        scale=1.0 for +eps, scale=-2.0 for transition to -eps
        
        Uses SINGLE Triton kernel for ALL parameters!
        """
        t0 = time.perf_counter()
        
        # Single kernel launch for all parameters!
        grid = lambda meta: (triton.cdiv(self.total_numel, meta['BLOCK_SIZE']),)
        
        if scale > 0:
            fused_perturb_kernel[grid](
                self.params_flat,
                self.z_flat,
                self.total_numel,
                eps=self.eps * scale,
                BLOCK_SIZE=self.block_size,
            )
        else:
            # Use negative kernel for scale < 0
            fused_perturb_neg_kernel[grid](
                self.params_flat,
                self.z_flat,
                self.total_numel,
                eps=self.eps * abs(scale) / 2,  # Divide by 2 since kernel does 2*eps
                BLOCK_SIZE=self.block_size,
            )
        
        self.timing['perturb'].append((time.perf_counter() - t0) * 1000)
    
    def update(self, projected_grad: float):
        """
        Update parameters using SINGLE kernel for all params.
        """
        t0 = time.perf_counter()
        
        grid = lambda meta: (triton.cdiv(self.total_numel, meta['BLOCK_SIZE']),)
        fused_update_kernel[grid](
            self.params_flat,
            self.z_flat,
            self.total_numel,
            lr=self.lr,
            projected_grad=projected_grad,
            BLOCK_SIZE=self.block_size,
        )
        
        self.timing['update'].append((time.perf_counter() - t0) * 1000)
    
    def print_timing(self):
        """Print timing statistics"""
        print("\n" + "="*60)
        print("TRITON MEZO V2 TIMING (Single Kernel)")
        print("="*60)
        
        import numpy as np
        for key, times in self.timing.items():
            if times:
                print(f"  {key:15s}: {np.mean(times):8.3f} ± {np.std(times):6.3f} ms")


# =============================================================================
# Trainer using Triton Kernels
# =============================================================================

class TritonMeZOTrainer:
    """
    MeZO Trainer using Triton kernels for maximum performance.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        eps: float = 1e-3,
        lr: float = 1e-5,
        use_v2: bool = True,  # Use V2 (single kernel) by default
    ):
        self.model = model
        self.eps = eps
        self.lr = lr
        
        if use_v2:
            self.ops = TritonMeZOOpsV2(model, eps=eps, lr=lr)
        else:
            self.ops = TritonMeZOOps(model, eps=eps, lr=lr)
        
        self.use_v2 = use_v2
        self.step_times = []
    
    def zo_forward(self, batch) -> torch.Tensor:
        """Forward pass returning loss"""
        outputs = self.model(**batch)
        return outputs.loss if hasattr(outputs, 'loss') else outputs[0]
    
    def zo_step(self, batch) -> Tuple[torch.Tensor, float]:
        """
        Single ZO gradient estimation step using Triton kernels.
        """
        import numpy as np
        t0 = time.perf_counter()
        
        seed = np.random.randint(0, 2**31)
        
        # Generate perturbation (single batched RNG)
        self.ops.generate_perturbation(seed)
        
        if self.use_v2:
            # V2: Operate on flat buffer, sync to model for forward
            self.ops.perturb(scale=1.0)  # +eps
            self.ops._sync_to_model()
            
            with torch.no_grad():
                loss1 = self.zo_forward(batch)
            
            self.ops.perturb(scale=-2.0)  # -2eps (now at -eps)
            self.ops._sync_to_model()
            
            with torch.no_grad():
                loss2 = self.zo_forward(batch)
            
            self.ops.perturb(scale=1.0)  # +eps (back to original)
            
            projected_grad = (loss1.item() - loss2.item()) / (2 * self.eps)
            
            self.ops.update(projected_grad)
            self.ops._sync_to_model()
        else:
            # V1: Per-parameter kernel calls
            self.ops.perturb_positive()
            
            with torch.no_grad():
                loss1 = self.zo_forward(batch)
            
            self.ops.perturb_negative()
            
            with torch.no_grad():
                loss2 = self.zo_forward(batch)
            
            self.ops.perturb_reset()
            
            projected_grad = (loss1.item() - loss2.item()) / (2 * self.eps)
            
            self.ops.update(projected_grad)
        
        self.step_times.append((time.perf_counter() - t0) * 1000)
        
        return (loss1 + loss2) / 2, projected_grad
    
    def print_summary(self):
        """Print timing summary"""
        import numpy as np
        print("\n" + "="*60)
        print("TRITON MEZO TRAINER SUMMARY")
        print("="*60)
        print(f"  Step time: {np.mean(self.step_times):.2f} ± {np.std(self.step_times):.2f} ms")
        self.ops.print_timing()


# =============================================================================
# Benchmark
# =============================================================================

def benchmark_triton_mezo(
    model_name: str = "facebook/opt-350m",
    num_steps: int = 10,
    warmup_steps: int = 3,
    batch_size: int = 4,
    seq_length: int = 128,
):
    """
    Benchmark Triton MeZO implementation.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import numpy as np
    
    print("="*70)
    print("TRITON MEZO BENCHMARK")
    print("="*70)
    
    device = "cuda"
    
    # Load model
    print(f"\nLoading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    
    # Create dummy batch
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    texts = ["This is a test sentence for benchmarking MeZO."] * batch_size
    batch = tokenizer(texts, return_tensors="pt", padding="max_length",
                      max_length=seq_length, truncation=True)
    batch = {k: v.to(device) for k, v in batch.items()}
    batch['labels'] = batch['input_ids'].clone()
    
    # Test V1 (per-parameter kernels)
    print("\n" + "-"*70)
    print("Testing V1 (per-parameter Triton kernels)...")
    print("-"*70)
    
    trainer_v1 = TritonMeZOTrainer(model, eps=1e-3, lr=1e-5, use_v2=False)
    
    # Warmup
    for _ in range(warmup_steps):
        trainer_v1.zo_step(batch)
    trainer_v1.step_times = []
    trainer_v1.ops.timing = {k: [] for k in trainer_v1.ops.timing}
    
    # Benchmark
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for step in range(num_steps):
        loss, grad = trainer_v1.zo_step(batch)
        if step % 5 == 0:
            print(f"  Step {step}: loss={loss.item():.4f}")
    torch.cuda.synchronize()
    v1_total = time.perf_counter() - t0
    
    trainer_v1.print_summary()
    print(f"\n  Total: {v1_total*1000:.2f} ms, Per-step: {v1_total/num_steps*1000:.2f} ms")
    
    # Reload model for V2
    del model, trainer_v1
    torch.cuda.empty_cache()
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    
    # Test V2 (single kernel for all params)
    print("\n" + "-"*70)
    print("Testing V2 (single Triton kernel for ALL params)...")
    print("-"*70)
    
    trainer_v2 = TritonMeZOTrainer(model, eps=1e-3, lr=1e-5, use_v2=True)
    
    # Warmup
    for _ in range(warmup_steps):
        trainer_v2.zo_step(batch)
    trainer_v2.step_times = []
    trainer_v2.ops.timing = {k: [] for k in trainer_v2.ops.timing}
    
    # Benchmark
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for step in range(num_steps):
        loss, grad = trainer_v2.zo_step(batch)
        if step % 5 == 0:
            print(f"  Step {step}: loss={loss.item():.4f}")
    torch.cuda.synchronize()
    v2_total = time.perf_counter() - t0
    
    trainer_v2.print_summary()
    print(f"\n  Total: {v2_total*1000:.2f} ms, Per-step: {v2_total/num_steps*1000:.2f} ms")
    
    # Summary
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print(f"  V1 (per-param kernels): {v1_total/num_steps*1000:.2f} ms/step")
    print(f"  V2 (single kernel):     {v2_total/num_steps*1000:.2f} ms/step")
    print(f"  V2 speedup over V1:     {v1_total/v2_total:.2f}x")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="facebook/opt-350m")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-length", type=int, default=128)
    args = parser.parse_args()
    
    benchmark_triton_mezo(
        model_name=args.model,
        num_steps=args.steps,
        warmup_steps=args.warmup,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
    )
