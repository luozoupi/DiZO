"""
Single-GPU Parallelism Optimizations for ZO (Zeroth-Order) Optimization

This module implements advanced optimizations to improve single-GPU ZO step performance:

1. Fused Dual-Perturb Kernel: Generate z once, write both +eps*z and -eps*z
2. Async Gradient Computation: Compute (loss1-loss2)/(2*eps) on GPU
3. CUDA Graph Capture: Capture entire ZO step as a graph
4. Reduced Synchronization: Event-based timing without explicit syncs

Performance Targets:
- Current: 1.21x speedup (Single-GPU Dual-Model)
- Target: 1.4-1.5x speedup with these optimizations

Author: DiZO Team
Date: 2025-01
"""

import torch
import triton
import triton.language as tl
import numpy as np
import time
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass


# =============================================================================
# Fused Dual-Perturb Kernel
# =============================================================================
# Key optimization: Generate random z once, write to TWO buffers simultaneously
# - params_plus = anchor + eps * z
# - params_minus = anchor - eps * z
# Benefits:
# - Single Philox RNG call instead of two
# - Read anchor once, write twice (better memory bandwidth)
# - Eliminates kernel launch overhead for second perturb
# =============================================================================

@triton.jit
def fused_dual_perturb_kernel(
    params_plus_ptr,    # Output: θ + εz
    params_minus_ptr,   # Output: θ - εz
    anchor_ptr,         # Input: θ₀ (original parameters)
    seed,               # Random seed
    eps,                # Perturbation epsilon
    n_elements,         # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused dual perturbation kernel.

    Generates random z and computes both:
    - params_plus = anchor + eps * z
    - params_minus = anchor - eps * z

    in a single kernel, halving the RNG cost and improving memory efficiency.
    """
    # Philox constants
    M0 = 0xD2511F53
    M1 = 0xCD9E8D57
    W0 = 0x9E3779B9
    W1 = 0xBB67AE85
    TWO_PI = 6.283185307179586

    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load anchor parameters (read once)
    anchor = tl.load(anchor_ptr + offsets, mask=mask, other=0.0)

    # Inline Philox RNG
    c0 = offsets.to(tl.uint32)
    c1 = tl.zeros_like(c0)
    c2 = tl.zeros_like(c0)
    c3 = tl.zeros_like(c0)
    k0 = tl.full(c0.shape, seed, dtype=tl.uint32)
    k1 = tl.zeros_like(k0)

    # 10 rounds of Philox
    for _ in range(10):
        prod0 = c0.to(tl.uint64) * M0
        hi0 = (prod0 >> 32).to(tl.uint32)
        lo0 = (prod0 & 0xFFFFFFFF).to(tl.uint32)
        prod1 = c2.to(tl.uint64) * M1
        hi1 = (prod1 >> 32).to(tl.uint32)
        lo1 = (prod1 & 0xFFFFFFFF).to(tl.uint32)
        c0, c1, c2, c3 = hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0
        k0 = k0 + W0
        k1 = k1 + W1

    # Box-Muller transform
    u1 = (c0.to(tl.float32) + 0.5) * (1.0 / 4294967296.0)
    u2 = (c1.to(tl.float32) + 0.5) * (1.0 / 4294967296.0)
    r = tl.sqrt(-2.0 * tl.log(u1))
    z = r * tl.cos(TWO_PI * u2)

    # Compute both perturbations
    eps_z = eps * z
    result_plus = anchor + eps_z
    result_minus = anchor - eps_z

    # Store both results (write twice from single read)
    tl.store(params_plus_ptr + offsets, result_plus, mask=mask)
    tl.store(params_minus_ptr + offsets, result_minus, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_dual_perturb_kernel_autotuned(
    params_plus_ptr,
    params_minus_ptr,
    anchor_ptr,
    seed,
    eps,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Auto-tuned version of fused dual perturbation kernel."""
    # Philox constants
    M0 = 0xD2511F53
    M1 = 0xCD9E8D57
    W0 = 0x9E3779B9
    W1 = 0xBB67AE85
    TWO_PI = 6.283185307179586

    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load anchor (single read)
    anchor = tl.load(anchor_ptr + offsets, mask=mask, other=0.0)

    # Inline Philox
    c0 = offsets.to(tl.uint32)
    c1 = tl.zeros_like(c0)
    c2 = tl.zeros_like(c0)
    c3 = tl.zeros_like(c0)
    k0 = tl.full(c0.shape, seed, dtype=tl.uint32)
    k1 = tl.zeros_like(k0)

    for _ in range(10):
        prod0 = c0.to(tl.uint64) * M0
        hi0 = (prod0 >> 32).to(tl.uint32)
        lo0 = (prod0 & 0xFFFFFFFF).to(tl.uint32)
        prod1 = c2.to(tl.uint64) * M1
        hi1 = (prod1 >> 32).to(tl.uint32)
        lo1 = (prod1 & 0xFFFFFFFF).to(tl.uint32)
        c0, c1, c2, c3 = hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0
        k0 = k0 + W0
        k1 = k1 + W1

    # Box-Muller
    u1 = (c0.to(tl.float32) + 0.5) * (1.0 / 4294967296.0)
    u2 = (c1.to(tl.float32) + 0.5) * (1.0 / 4294967296.0)
    r = tl.sqrt(-2.0 * tl.log(u1))
    z = r * tl.cos(TWO_PI * u2)

    # Dual perturbation
    eps_z = eps * z
    tl.store(params_plus_ptr + offsets, anchor + eps_z, mask=mask)
    tl.store(params_minus_ptr + offsets, anchor - eps_z, mask=mask)


# =============================================================================
# Fused Dual-Perturb 2x Kernel (Even more optimized)
# =============================================================================
# Uses 2 normal values from single Philox call (4 outputs -> 2 normals)

@triton.jit
def fast_randn_2_inline(seed, offsets):
    """Generate 2 normal values from single Philox call (inline version)."""
    M0 = 0xD2511F53
    M1 = 0xCD9E8D57
    W0 = 0x9E3779B9
    W1 = 0xBB67AE85
    TWO_PI = 6.283185307179586

    c0 = offsets.to(tl.uint32)
    c1 = tl.zeros_like(c0)
    c2 = tl.zeros_like(c0)
    c3 = tl.zeros_like(c0)
    k0 = tl.full(c0.shape, seed, dtype=tl.uint32)
    k1 = tl.zeros_like(k0)

    for _ in range(10):
        prod0 = c0.to(tl.uint64) * M0
        hi0 = (prod0 >> 32).to(tl.uint32)
        lo0 = (prod0 & 0xFFFFFFFF).to(tl.uint32)
        prod1 = c2.to(tl.uint64) * M1
        hi1 = (prod1 >> 32).to(tl.uint32)
        lo1 = (prod1 & 0xFFFFFFFF).to(tl.uint32)
        c0, c1, c2, c3 = hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0
        k0 = k0 + W0
        k1 = k1 + W1

    # 4 outputs -> 2 normals via Box-Muller
    u1 = (c0.to(tl.float32) + 0.5) * (1.0 / 4294967296.0)
    u2 = (c1.to(tl.float32) + 0.5) * (1.0 / 4294967296.0)
    u3 = (c2.to(tl.float32) + 0.5) * (1.0 / 4294967296.0)
    u4 = (c3.to(tl.float32) + 0.5) * (1.0 / 4294967296.0)

    r1 = tl.sqrt(-2.0 * tl.log(u1))
    r2 = tl.sqrt(-2.0 * tl.log(u3))
    z0 = r1 * tl.cos(TWO_PI * u2)
    z1 = r2 * tl.cos(TWO_PI * u4)

    return z0, z1


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_dual_perturb_2x_kernel(
    params_plus_ptr,
    params_minus_ptr,
    anchor_ptr,
    seed,
    eps,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Highly optimized dual-perturb kernel using 2x Philox efficiency.

    Processes 2*BLOCK_SIZE elements per program by generating 2 normals
    per Philox call.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE * 2

    # First half
    offsets_a = block_start + tl.arange(0, BLOCK_SIZE)
    mask_a = offsets_a < n_elements

    # Second half
    offsets_b = block_start + BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask_b = offsets_b < n_elements

    # Load anchor values
    anchor_a = tl.load(anchor_ptr + offsets_a, mask=mask_a, other=0.0)
    anchor_b = tl.load(anchor_ptr + offsets_b, mask=mask_b, other=0.0)

    # Generate 2 normals from single Philox (use offsets_a as base)
    z_a, z_b = fast_randn_2_inline(seed, offsets_a)

    # Compute perturbations
    eps_z_a = eps * z_a
    eps_z_b = eps * z_b

    # Store both directions for both halves
    tl.store(params_plus_ptr + offsets_a, anchor_a + eps_z_a, mask=mask_a)
    tl.store(params_plus_ptr + offsets_b, anchor_b + eps_z_b, mask=mask_b)
    tl.store(params_minus_ptr + offsets_a, anchor_a - eps_z_a, mask=mask_a)
    tl.store(params_minus_ptr + offsets_b, anchor_b - eps_z_b, mask=mask_b)


# =============================================================================
# Async Gradient Computation Kernel
# =============================================================================
# Computes projected_grad = (loss1 - loss2) / (2 * eps) on GPU
# Avoids CPU-GPU sync from .item() call

@triton.jit
def compute_projected_grad_kernel(
    loss1_ptr,       # Scalar tensor: loss from +eps perturbation
    loss2_ptr,       # Scalar tensor: loss from -eps perturbation
    eps,             # Perturbation epsilon
    grad_ptr,        # Output: projected gradient
    BLOCK: tl.constexpr = 1,
):
    """Compute projected gradient on GPU to avoid sync."""
    loss1 = tl.load(loss1_ptr)
    loss2 = tl.load(loss2_ptr)
    grad = (loss1 - loss2) / (2.0 * eps)
    tl.store(grad_ptr, grad)


# =============================================================================
# Fused Update with Runtime Grad Kernel
# =============================================================================
# Update parameters using gradient stored on GPU (no .item() needed)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_update_runtime_grad_kernel(
    params_ptr,      # Parameters to update (in-place)
    grad_ptr,        # Pointer to projected gradient (on GPU)
    seed,            # Random seed
    lr,              # Learning rate
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Update kernel that reads gradient from GPU tensor.

    Formula: params = params - lr * grad * z

    where grad is stored as a GPU scalar tensor, avoiding .item() sync.
    """
    M0 = 0xD2511F53
    M1 = 0xCD9E8D57
    W0 = 0x9E3779B9
    W1 = 0xBB67AE85
    TWO_PI = 6.283185307179586

    # Load gradient from GPU (scalar broadcast)
    grad = tl.load(grad_ptr)

    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    params = tl.load(params_ptr + offsets, mask=mask, other=0.0)

    # Philox RNG
    c0 = offsets.to(tl.uint32)
    c1 = tl.zeros_like(c0)
    c2 = tl.zeros_like(c0)
    c3 = tl.zeros_like(c0)
    k0 = tl.full(c0.shape, seed, dtype=tl.uint32)
    k1 = tl.zeros_like(k0)

    for _ in range(10):
        prod0 = c0.to(tl.uint64) * M0
        hi0 = (prod0 >> 32).to(tl.uint32)
        lo0 = (prod0 & 0xFFFFFFFF).to(tl.uint32)
        prod1 = c2.to(tl.uint64) * M1
        hi1 = (prod1 >> 32).to(tl.uint32)
        lo1 = (prod1 & 0xFFFFFFFF).to(tl.uint32)
        c0, c1, c2, c3 = hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0
        k0 = k0 + W0
        k1 = k1 + W1

    u1 = (c0.to(tl.float32) + 0.5) * (1.0 / 4294967296.0)
    u2 = (c1.to(tl.float32) + 0.5) * (1.0 / 4294967296.0)
    r = tl.sqrt(-2.0 * tl.log(u1))
    z = r * tl.cos(TWO_PI * u2)

    # Update: params = params - lr * grad * z
    result = params - lr * grad * z
    tl.store(params_ptr + offsets, result, mask=mask)


# =============================================================================
# Python Wrapper Classes
# =============================================================================

class FusedDualPerturbOps:
    """
    Wrapper class for fused dual-perturb operations.

    Provides convenient interface for single-GPU parallel ZO optimization.
    """

    def __init__(self, n_elements: int, device: torch.device, dtype: torch.dtype = torch.float32):
        self.n_elements = n_elements
        self.device = device
        self.dtype = dtype

        # Pre-compute grid
        self.grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        self.grid_2x = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE'] * 2),)

        # Pre-allocate scalar tensors for async gradient computation
        self.grad_tensor = torch.zeros(1, dtype=dtype, device=device)

    def dual_perturb(
        self,
        params_plus: torch.Tensor,
        params_minus: torch.Tensor,
        anchor: torch.Tensor,
        seed: int,
        eps: float,
    ):
        """
        Apply dual perturbation: generate z once, compute both +eps*z and -eps*z.

        Args:
            params_plus: Output buffer for θ + εz
            params_minus: Output buffer for θ - εz
            anchor: Input anchor parameters θ₀
            seed: Random seed
            eps: Perturbation epsilon
        """
        fused_dual_perturb_kernel_autotuned[self.grid](
            params_plus, params_minus, anchor,
            seed, eps, self.n_elements,
        )

    def dual_perturb_2x(
        self,
        params_plus: torch.Tensor,
        params_minus: torch.Tensor,
        anchor: torch.Tensor,
        seed: int,
        eps: float,
    ):
        """2x optimized version using efficient Philox output utilization."""
        fused_dual_perturb_2x_kernel[self.grid_2x](
            params_plus, params_minus, anchor,
            seed, eps, self.n_elements,
        )

    def compute_grad_async(
        self,
        loss1: torch.Tensor,
        loss2: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        """
        Compute projected gradient on GPU (async, no .item() sync).

        Returns:
            Tensor containing projected_grad = (loss1 - loss2) / (2 * eps)
        """
        compute_projected_grad_kernel[(1,)](
            loss1, loss2, eps, self.grad_tensor,
        )
        return self.grad_tensor

    def update_with_grad_tensor(
        self,
        params: torch.Tensor,
        grad_tensor: torch.Tensor,
        seed: int,
        lr: float,
    ):
        """
        Update parameters using gradient stored on GPU.

        Args:
            params: Parameters to update (in-place)
            grad_tensor: Projected gradient (GPU tensor)
            seed: Random seed (same as used for perturbation)
            lr: Learning rate
        """
        fused_update_runtime_grad_kernel[self.grid](
            params, grad_tensor, seed, lr, self.n_elements,
        )


# =============================================================================
# CUDA Graph Wrapper for ZO Step
# =============================================================================

class CUDAGraphZOStep:
    """
    Captures ZO step as a CUDA graph for minimal kernel launch overhead.

    Usage:
        graph_zo = CUDAGraphZOStep(model1, model2, ...)
        graph_zo.capture(sample_batch)

        for batch in batches:
            loss = graph_zo.replay(batch, seed)
    """

    def __init__(
        self,
        n_elements: int,
        device: torch.device,
        eps: float = 1e-3,
        lr: float = 1e-5,
    ):
        self.n_elements = n_elements
        self.device = device
        self.eps = eps
        self.lr = lr

        self.graph = None
        self.captured = False

        # Placeholders for captured tensors
        self.seed_tensor = torch.zeros(1, dtype=torch.int64, device=device)
        self.grad_tensor = torch.zeros(1, dtype=torch.float32, device=device)

        # Ops wrapper
        self.ops = FusedDualPerturbOps(n_elements, device)

    def capture(
        self,
        params_plus: torch.Tensor,
        params_minus: torch.Tensor,
        anchor: torch.Tensor,
        forward_fn1,  # Callable returning loss tensor
        forward_fn2,  # Callable returning loss tensor
        warmup_iters: int = 3,
    ):
        """
        Capture ZO step as CUDA graph.

        Args:
            params_plus: Buffer for +eps perturbation
            params_minus: Buffer for -eps perturbation
            anchor: Anchor parameters
            forward_fn1: Forward function for model 1 (returns loss tensor)
            forward_fn2: Forward function for model 2 (returns loss tensor)
        """
        # Warmup
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())

        with torch.cuda.stream(s):
            for _ in range(warmup_iters):
                self.ops.dual_perturb(
                    params_plus, params_minus, anchor,
                    int(self.seed_tensor.item()), self.eps
                )
                loss1 = forward_fn1()
                loss2 = forward_fn2()
                self.ops.compute_grad_async(loss1, loss2, self.eps)
                self.ops.update_with_grad_tensor(
                    anchor, self.grad_tensor,
                    int(self.seed_tensor.item()), self.lr
                )

        torch.cuda.current_stream().wait_stream(s)

        # Capture graph
        self.graph = torch.cuda.CUDAGraph()

        with torch.cuda.graph(self.graph):
            self.ops.dual_perturb(
                params_plus, params_minus, anchor,
                int(self.seed_tensor.item()), self.eps
            )
            self._loss1_placeholder = forward_fn1()
            self._loss2_placeholder = forward_fn2()
            self.ops.compute_grad_async(
                self._loss1_placeholder,
                self._loss2_placeholder,
                self.eps
            )
            self.ops.update_with_grad_tensor(
                anchor, self.grad_tensor,
                int(self.seed_tensor.item()), self.lr
            )

        self.captured = True
        print(f"[CUDAGraph] ZO step captured successfully")

    def replay(self, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Replay captured graph with new seed.

        Args:
            seed: New random seed

        Returns:
            Tuple of (loss1, loss2) tensors
        """
        if not self.captured:
            raise RuntimeError("Graph not captured. Call capture() first.")

        self.seed_tensor.fill_(seed)
        self.graph.replay()

        return self._loss1_placeholder, self._loss2_placeholder


# =============================================================================
# Benchmark Utilities
# =============================================================================

def benchmark_dual_perturb_kernels(
    n_elements: int,
    device: torch.device,
    n_iter: int = 100,
    warmup: int = 10,
) -> Dict[str, float]:
    """
    Benchmark different dual-perturb kernel implementations.

    Returns dict with timing for each method.
    """
    dtype = torch.float32
    eps = 1e-3

    # Allocate buffers
    anchor = torch.randn(n_elements, dtype=dtype, device=device)
    params_plus = torch.empty_like(anchor)
    params_minus = torch.empty_like(anchor)

    results = {}

    # Baseline: Two separate perturb calls
    print("Benchmarking baseline (2 separate perturbs)...")

    # Import existing kernel for comparison
    import sys
    sys.path.insert(0, '/home/luo00466/DiZO_old/DiZO/large_models/cuda_kernels/Perturb_wise')
    try:
        from triton_fused_perturb import fused_perturb_kernel_philox_autotuned
        has_baseline = True
    except ImportError:
        has_baseline = False
        print("  Warning: Could not import baseline kernel")

    if has_baseline:
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

        # Warmup
        for _ in range(warmup):
            params_plus.copy_(anchor)
            params_minus.copy_(anchor)
            fused_perturb_kernel_philox_autotuned[grid](params_plus, 42, eps, n_elements)
            fused_perturb_kernel_philox_autotuned[grid](params_minus, 42, -eps, n_elements)
        torch.cuda.synchronize()

        # Benchmark
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for i in range(n_iter):
            seed = 42 + i
            params_plus.copy_(anchor)
            params_minus.copy_(anchor)
            fused_perturb_kernel_philox_autotuned[grid](params_plus, seed, eps, n_elements)
            fused_perturb_kernel_philox_autotuned[grid](params_minus, seed, -eps, n_elements)
        end.record()
        torch.cuda.synchronize()

        results['baseline_2x_perturb'] = start.elapsed_time(end) / n_iter

    # Fused dual-perturb (autotuned)
    print("Benchmarking fused dual-perturb (autotuned)...")
    grid_fused = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    # Warmup
    for _ in range(warmup):
        fused_dual_perturb_kernel_autotuned[grid_fused](
            params_plus, params_minus, anchor, 42, eps, n_elements
        )
    torch.cuda.synchronize()

    # Benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for i in range(n_iter):
        seed = 42 + i
        fused_dual_perturb_kernel_autotuned[grid_fused](
            params_plus, params_minus, anchor, seed, eps, n_elements
        )
    end.record()
    torch.cuda.synchronize()

    results['fused_dual_perturb'] = start.elapsed_time(end) / n_iter

    # Fused dual-perturb 2x
    print("Benchmarking fused dual-perturb 2x...")
    grid_2x = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE'] * 2),)

    # Warmup
    for _ in range(warmup):
        fused_dual_perturb_2x_kernel[grid_2x](
            params_plus, params_minus, anchor, 42, eps, n_elements
        )
    torch.cuda.synchronize()

    # Benchmark
    start.record()
    for i in range(n_iter):
        seed = 42 + i
        fused_dual_perturb_2x_kernel[grid_2x](
            params_plus, params_minus, anchor, seed, eps, n_elements
        )
    end.record()
    torch.cuda.synchronize()

    results['fused_dual_perturb_2x'] = start.elapsed_time(end) / n_iter

    return results


def verify_dual_perturb_correctness(n_elements: int = 1000000, device: torch.device = None):
    """Verify fused dual-perturb produces correct results."""
    if device is None:
        device = torch.device('cuda:0')

    dtype = torch.float32
    eps = 1e-3
    seed = 12345

    anchor = torch.randn(n_elements, dtype=dtype, device=device)

    # Reference: using existing single perturb (non-autotuned for determinism)
    # Note: The baseline kernel does params = params + alpha*z (in-place ADD)
    # Our fused kernel does params_out = anchor + eps*z (write from anchor)
    import sys
    sys.path.insert(0, '/home/luo00466/DiZO_old/DiZO/large_models/cuda_kernels/Perturb_wise')
    from triton_fused_perturb import fused_perturb_kernel_philox

    # Use fixed block size for deterministic comparison
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, BLOCK_SIZE),)

    # Reference: start from anchor, apply +eps and -eps perturbations
    ref_plus = anchor.clone()
    ref_minus = anchor.clone()
    fused_perturb_kernel_philox[grid](ref_plus, seed, eps, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    fused_perturb_kernel_philox[grid](ref_minus, seed, -eps, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    torch.cuda.synchronize()

    # Test fused kernel (non-autotuned) with same block size
    test_plus = torch.empty_like(anchor)
    test_minus = torch.empty_like(anchor)

    grid_fused = lambda meta: (triton.cdiv(n_elements, BLOCK_SIZE),)
    fused_dual_perturb_kernel[grid_fused](
        test_plus, test_minus, anchor, seed, eps, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    torch.cuda.synchronize()

    # Check results
    plus_match = torch.allclose(ref_plus, test_plus, rtol=1e-5, atol=1e-6)
    minus_match = torch.allclose(ref_minus, test_minus, rtol=1e-5, atol=1e-6)

    # Debug: show actual values
    print(f"  anchor[0:5]: {anchor[0:5].tolist()}")
    print(f"  ref_plus[0:5]: {ref_plus[0:5].tolist()}")
    print(f"  test_plus[0:5]: {test_plus[0:5].tolist()}")
    print(f"  ref_minus[0:5]: {ref_minus[0:5].tolist()}")
    print(f"  test_minus[0:5]: {test_minus[0:5].tolist()}")

    # Show implied z values
    z_ref = (ref_plus - anchor) / eps
    z_test = (test_plus - anchor) / eps
    print(f"  z_ref[0:5]: {z_ref[0:5].tolist()}")
    print(f"  z_test[0:5]: {z_test[0:5].tolist()}")

    if plus_match and minus_match:
        print(f"[PASS] Fused dual-perturb correctness verified (n={n_elements})")
    else:
        print(f"[FAIL] Fused dual-perturb mismatch!")
        if not plus_match:
            diff = (ref_plus - test_plus).abs()
            print(f"  Plus: max_diff={diff.max():.2e}, mean_diff={diff.mean():.2e}")
        if not minus_match:
            diff = (ref_minus - test_minus).abs()
            print(f"  Minus: max_diff={diff.max():.2e}, mean_diff={diff.mean():.2e}")

    return plus_match and minus_match


# =============================================================================
# Main Test
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test single-GPU optimization kernels")
    parser.add_argument('--device', type=int, default=0, help='GPU device ID')
    parser.add_argument('--n_elements', type=int, default=331_196_416,
                        help='Number of elements (default: OPT-350M)')
    parser.add_argument('--n_iter', type=int, default=100, help='Benchmark iterations')
    parser.add_argument('--verify', action='store_true', help='Run correctness verification')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.device}')
    print(f"Using device: {device} ({torch.cuda.get_device_name(device)})")
    print(f"Testing with n_elements = {args.n_elements:,}")

    if args.verify:
        print("\n=== Correctness Verification ===")
        verify_dual_perturb_correctness(args.n_elements, device)

    print("\n=== Performance Benchmark ===")
    results = benchmark_dual_perturb_kernels(
        args.n_elements, device,
        n_iter=args.n_iter,
    )

    print("\n=== Results ===")
    baseline_time = results.get('baseline_2x_perturb', None)
    for name, time_ms in results.items():
        speedup = ""
        if baseline_time and name != 'baseline_2x_perturb':
            speedup = f" ({baseline_time/time_ms:.2f}x vs baseline)"
        print(f"  {name}: {time_ms:.3f} ms{speedup}")
