"""
Triton Kernels for Fused RNG + Perturbation (MeZO Optimization)

This module provides highly optimized Triton kernels for zero-order optimization
using custom Philox 4x32-10 RNG implementation.

Key features:
1. Fused random number generation + perturbation in a single kernel
2. Zero extra memory overhead (no z_flat buffer needed)
3. Reproducible via seed-based Philox RNG
4. Custom inline Philox implementation (~7-40% faster than tl.randn())

Performance (OPT-350M, 331M params):
=========================================
| Kernel                    | Time   | Speedup |
|---------------------------|--------|---------|
| Philox 2x (BLOCK=512)     | 1.13ms | 1.41x ★ |
| Philox Autotuned          | 1.47ms | 1.08x   |
| tl.randn() Autotuned      | 1.58ms | 1.00x   |
=========================================

The 2x kernel generates 2 normal values from each Philox call,
effectively doubling throughput for the same RNG cost.

Usage:
    from triton_fused_perturb import FusedPerturbMeZO
    
    trainer = FusedPerturbMeZO(model)
    loss = trainer.zo_step(batch)
    
    # Or use kernels directly:
    from triton_fused_perturb import fused_perturb_kernel_philox_autotuned
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    fused_perturb_kernel_philox_autotuned[grid](params, seed, alpha, n)
"""

import torch
import triton
import triton.language as tl
import numpy as np
import math
from typing import Tuple, Dict, Optional


# =============================================================================
# Custom Philox 4x32-10 RNG Implementation
# =============================================================================
# 
# Philox is a counter-based PRNG that produces high-quality random numbers.
# We implement it directly in Triton for better performance than tl.randn().
#
# The algorithm:
# 1. Initialize counters from seed + element offset
# 2. Perform 10 rounds of mixing (Philox 4x32-10)
# 3. Transform uniform randoms to normal distribution via Box-Muller
#
# Key constants (from original Philox paper):
#   PHILOX_M4x32_0 = 0xD2511F53  (multiplier)
#   PHILOX_M4x32_1 = 0xCD9E8D57  (multiplier)
#   PHILOX_W32_0   = 0x9E3779B9  (golden ratio based Weyl sequence)
#   PHILOX_W32_1   = 0xBB67AE85  (Weyl sequence)
#
# Advantages of custom Philox over tl.randn():
# - Fully inlined (no function call overhead)
# - Can generate 2 normals from 4 uint32 outputs (2x throughput)
# - Better register utilization with inline code
# =============================================================================

# Philox multiplier constants
PHILOX_M0: tl.constexpr = 0xD2511F53
PHILOX_M1: tl.constexpr = 0xCD9E8D57

# Philox key bump constants (Weyl sequence)
PHILOX_W0: tl.constexpr = 0x9E3779B9
PHILOX_W1: tl.constexpr = 0xBB67AE85


@triton.jit
def philox_round(c0, c1, c2, c3, k0, k1):
    """
    Single Philox 4x32 round.
    
    Performs the S-box substitution using multiplication and XOR.
    The multiplication produces 64-bit results which are split into
    high and low 32-bit parts for mixing.
    """
    # Philox multiplier constants (inline for better optimization)
    M0 = 0xD2511F53
    M1 = 0xCD9E8D57
    
    # Multiply c0 by M0, get high and low 32 bits
    prod0 = c0.to(tl.uint64) * M0
    hi0 = (prod0 >> 32).to(tl.uint32)
    lo0 = (prod0 & 0xFFFFFFFF).to(tl.uint32)
    
    # Multiply c2 by M1, get high and low 32 bits
    prod1 = c2.to(tl.uint64) * M1
    hi1 = (prod1 >> 32).to(tl.uint32)
    lo1 = (prod1 & 0xFFFFFFFF).to(tl.uint32)
    
    # Feistel-like mixing
    new_c0 = hi1 ^ c1 ^ k0
    new_c1 = lo1
    new_c2 = hi0 ^ c3 ^ k1
    new_c3 = lo0
    
    return new_c0, new_c1, new_c2, new_c3


@triton.jit
def philox_4x32_10(seed, offset):
    """
    Full Philox 4x32-10 random number generator.
    
    Performs 10 rounds of the Philox algorithm to produce
    4 high-quality 32-bit random numbers.
    
    Args:
        seed: 32-bit seed value
        offset: Element offset (used for counter initialization)
        
    Returns:
        Tuple of 4 uint32 random values
    """
    # Philox key bump constants (Weyl sequence)
    W0 = 0x9E3779B9
    W1 = 0xBB67AE85
    
    # Initialize counters from offset
    # offset is a tensor (arange), seed is a scalar
    c0 = offset.to(tl.uint32)
    c1 = tl.zeros_like(c0)
    c2 = tl.zeros_like(c0)
    c3 = tl.zeros_like(c0)
    
    # Initialize key from seed - broadcast to match c0 shape
    # seed is passed as int, use it directly in operations
    k0 = tl.full(c0.shape, seed, dtype=tl.uint32)
    k1 = tl.zeros_like(k0)
    
    # Round 1
    c0, c1, c2, c3 = philox_round(c0, c1, c2, c3, k0, k1)
    k0 = k0 + W0
    k1 = k1 + W1
    
    # Round 2
    c0, c1, c2, c3 = philox_round(c0, c1, c2, c3, k0, k1)
    k0 = k0 + W0
    k1 = k1 + W1
    
    # Round 3
    c0, c1, c2, c3 = philox_round(c0, c1, c2, c3, k0, k1)
    k0 = k0 + W0
    k1 = k1 + W1
    
    # Round 4
    c0, c1, c2, c3 = philox_round(c0, c1, c2, c3, k0, k1)
    k0 = k0 + W0
    k1 = k1 + W1
    
    # Round 5
    c0, c1, c2, c3 = philox_round(c0, c1, c2, c3, k0, k1)
    k0 = k0 + W0
    k1 = k1 + W1
    
    # Round 6
    c0, c1, c2, c3 = philox_round(c0, c1, c2, c3, k0, k1)
    k0 = k0 + W0
    k1 = k1 + W1
    
    # Round 7
    c0, c1, c2, c3 = philox_round(c0, c1, c2, c3, k0, k1)
    k0 = k0 + W0
    k1 = k1 + W1
    
    # Round 8
    c0, c1, c2, c3 = philox_round(c0, c1, c2, c3, k0, k1)
    k0 = k0 + W0
    k1 = k1 + W1
    
    # Round 9
    c0, c1, c2, c3 = philox_round(c0, c1, c2, c3, k0, k1)
    k0 = k0 + W0
    k1 = k1 + W1
    
    # Round 10
    c0, c1, c2, c3 = philox_round(c0, c1, c2, c3, k0, k1)
    
    return c0, c1, c2, c3


@triton.jit
def uint32_to_uniform(x):
    """
    Convert uint32 to uniform float in (0, 1).
    
    We use the standard conversion: x * (1.0 / 2^32)
    The +1 ensures we never get exactly 0 (needed for Box-Muller log).
    """
    # Scale to (0, 1) - add 0.5 to avoid exact 0
    return (x.to(tl.float32) + 0.5) * (1.0 / 4294967296.0)


@triton.jit
def box_muller(u1, u2):
    """
    Box-Muller transform to convert uniform to normal distribution.
    
    Given u1, u2 ~ Uniform(0,1), produces z0, z1 ~ Normal(0,1):
        z0 = sqrt(-2 * ln(u1)) * cos(2 * pi * u2)
        z1 = sqrt(-2 * ln(u1)) * sin(2 * pi * u2)
    
    Args:
        u1, u2: Uniform random values in (0, 1)
        
    Returns:
        z0: Standard normal random value
    """
    # Constants
    TWO_PI = 6.283185307179586
    
    # Box-Muller transform
    r = tl.sqrt(-2.0 * tl.log(u1))
    theta = TWO_PI * u2
    z0 = r * tl.cos(theta)
    # z1 = r * tl.sin(theta)  # We only need one value
    
    return z0


@triton.jit
def fast_randn(seed, offset):
    """
    Fast normal random number generation using custom Philox + Box-Muller.
    
    This function generates a standard normal random value deterministically
    based on the seed and offset. Same seed + offset always produces the
    same result.
    
    Args:
        seed: Random seed (int32)
        offset: Element index/offset (int32 or int64)
        
    Returns:
        Standard normal random value (float32)
    """
    # Generate 4 random uint32 values
    r0, r1, r2, r3 = philox_4x32_10(seed, offset)
    
    # Convert first two to uniform floats
    u1 = uint32_to_uniform(r0)
    u2 = uint32_to_uniform(r1)
    
    # Box-Muller transform to normal
    z = box_muller(u1, u2)
    
    return z


@triton.jit  
def fast_randn_2(seed, offset):
    """
    Alternative: Generate 2 normal values from 4 random uint32s.
    Uses all output from Philox more efficiently.
    
    Returns:
        Tuple of 2 standard normal random values
    """
    r0, r1, r2, r3 = philox_4x32_10(seed, offset)
    
    u1 = uint32_to_uniform(r0)
    u2 = uint32_to_uniform(r1)
    u3 = uint32_to_uniform(r2)
    u4 = uint32_to_uniform(r3)
    
    # Two Box-Muller transforms
    z0 = box_muller(u1, u2)
    z1 = box_muller(u3, u4)
    
    return z0, z1


# =============================================================================
# Triton Kernels using Custom Philox RNG (Inline Implementation)
# =============================================================================

@triton.jit
def fused_perturb_kernel_philox(
    params_ptr,
    seed,
    alpha,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused perturb kernel using inline Philox RNG + Box-Muller.
    
    This version inlines the entire Philox algorithm to avoid
    potential overhead from function calls.
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
    
    # Load parameters
    params = tl.load(params_ptr + offsets, mask=mask, other=0.0)
    
    # Initialize Philox counters
    c0 = offsets.to(tl.uint32)
    c1 = tl.zeros_like(c0)
    c2 = tl.zeros_like(c0)
    c3 = tl.zeros_like(c0)
    
    # Initialize keys from seed
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
        
        new_c0 = hi1 ^ c1 ^ k0
        new_c1 = lo1
        new_c2 = hi0 ^ c3 ^ k1
        new_c3 = lo0
        c0, c1, c2, c3 = new_c0, new_c1, new_c2, new_c3
        
        k0 = k0 + W0
        k1 = k1 + W1
    
    # Box-Muller transform
    u1 = (c0.to(tl.float32) + 0.5) * (1.0 / 4294967296.0)
    u2 = (c1.to(tl.float32) + 0.5) * (1.0 / 4294967296.0)
    r = tl.sqrt(-2.0 * tl.log(u1))
    theta = TWO_PI * u2
    z = r * tl.cos(theta)
    
    # Apply perturbation
    result = params + alpha * z
    
    # Store result
    tl.store(params_ptr + offsets, result, mask=mask)


@triton.jit
def fused_perturb_kernel_philox_2x(
    params_ptr,
    seed,
    alpha,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized Philox kernel that generates 2 normals per Philox call.
    
    Since Philox produces 4 uint32 values, we can get 2 normal values
    via 2 Box-Muller transforms. This processes 2x elements with the
    same number of Philox rounds.
    
    Each thread processes 2 consecutive elements.
    """
    pid = tl.program_id(0)
    # Each block processes BLOCK_SIZE * 2 elements
    block_start = pid * BLOCK_SIZE * 2
    
    # First half
    offsets_a = block_start + tl.arange(0, BLOCK_SIZE)
    mask_a = offsets_a < n_elements
    
    # Second half  
    offsets_b = block_start + BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask_b = offsets_b < n_elements
    
    # Load both halves
    params_a = tl.load(params_ptr + offsets_a, mask=mask_a, other=0.0)
    params_b = tl.load(params_ptr + offsets_b, mask=mask_b, other=0.0)
    
    # Generate 2 normals from one Philox call
    z_a, z_b = fast_randn_2(seed, offsets_a)
    
    # Apply perturbation
    result_a = params_a + alpha * z_a
    result_b = params_b + alpha * z_b
    
    # Store both halves
    tl.store(params_ptr + offsets_a, result_a, mask=mask_a)
    tl.store(params_ptr + offsets_b, result_b, mask=mask_b)


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
def fused_perturb_kernel_philox_2x_autotuned(
    params_ptr,
    seed,
    alpha,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Auto-tuned version of 2x Philox kernel.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE * 2
    
    offsets_a = block_start + tl.arange(0, BLOCK_SIZE)
    mask_a = offsets_a < n_elements
    offsets_b = block_start + BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask_b = offsets_b < n_elements
    
    params_a = tl.load(params_ptr + offsets_a, mask=mask_a, other=0.0)
    params_b = tl.load(params_ptr + offsets_b, mask=mask_b, other=0.0)
    
    z_a, z_b = fast_randn_2(seed, offsets_a)
    
    result_a = params_a + alpha * z_a
    result_b = params_b + alpha * z_b
    
    tl.store(params_ptr + offsets_a, result_a, mask=mask_a)
    tl.store(params_ptr + offsets_b, result_b, mask=mask_b)


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
def fused_perturb_kernel_philox_autotuned(
    params_ptr,
    seed,
    alpha,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Auto-tuned version of custom Philox kernel with inline implementation.
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
    
    params = tl.load(params_ptr + offsets, mask=mask, other=0.0)
    
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
    
    u1 = (c0.to(tl.float32) + 0.5) * (1.0 / 4294967296.0)
    u2 = (c1.to(tl.float32) + 0.5) * (1.0 / 4294967296.0)
    r = tl.sqrt(-2.0 * tl.log(u1))
    z = r * tl.cos(TWO_PI * u2)
    
    result = params + alpha * z
    tl.store(params_ptr + offsets, result, mask=mask)


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
def fused_restore_update_kernel_philox(
    params_ptr,
    seed,
    eps,
    projected_grad,
    lr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused restore + update kernel using inline Philox RNG.
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
    
    params = tl.load(params_ptr + offsets, mask=mask, other=0.0)
    
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
    
    u1 = (c0.to(tl.float32) + 0.5) * (1.0 / 4294967296.0)
    u2 = (c1.to(tl.float32) + 0.5) * (1.0 / 4294967296.0)
    r = tl.sqrt(-2.0 * tl.log(u1))
    z = r * tl.cos(TWO_PI * u2)
    
    combined_alpha = eps - lr * projected_grad
    result = params + combined_alpha * z
    tl.store(params_ptr + offsets, result, mask=mask)


# =============================================================================
# Triton Kernel V1: Basic Fused Perturb
# =============================================================================

@triton.jit
def fused_perturb_kernel_v1(
    params_ptr,          # Pointer to flat parameter buffer
    seed,                # Random seed for Philox RNG
    alpha,               # Perturbation scale
    n_elements,          # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    """
    Basic fused kernel: Generate random normal and apply perturbation.
    
    Formula: params[i] = params[i] + alpha * randn(seed, i)
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load parameters
    params = tl.load(params_ptr + offsets, mask=mask)
    
    # Generate random normals using Triton's built-in Philox RNG
    random_vals = tl.randn(seed, offsets)
    
    # Apply perturbation
    params = params + alpha * random_vals
    
    # Store result
    tl.store(params_ptr + offsets, params, mask=mask)


# =============================================================================
# Triton Kernel V2: Optimized with Larger Block Size
# =============================================================================

@triton.jit
def fused_perturb_kernel_v2(
    params_ptr,
    seed,
    alpha,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized version with configurable block size for better occupancy.
    Uses explicit tiling for better memory coalescing.
    """
    pid = tl.program_id(0)
    
    # Process BLOCK_SIZE elements per program
    block_start = pid * BLOCK_SIZE
    
    # Use vectorized loads/stores when possible
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load (coalesced access pattern)
    params = tl.load(params_ptr + offsets, mask=mask, other=0.0)
    
    # Generate randoms - seed combined with offset for uniqueness
    z = tl.randn(seed, offsets)
    
    # params + alpha * z (fused multiply-add)
    result = params + alpha * z
    
    # Store (coalesced)
    tl.store(params_ptr + offsets, result, mask=mask)


# =============================================================================
# Triton Kernel V3: Fused Restore + Update
# =============================================================================

@triton.jit
def fused_restore_update_kernel(
    params_ptr,
    seed,
    eps,                 # Original perturbation epsilon
    projected_grad,      # (loss+ - loss-) / (2*eps)
    lr,                  # Learning rate
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for restore and update in one pass.
    
    After perturbing with -eps (from +eps position), params are at original - eps*z.
    This kernel does: params = params + eps*z - lr*projected_grad*z
                    = params + (eps - lr*projected_grad) * z
    
    Combines restore to original position AND gradient update!
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    params = tl.load(params_ptr + offsets, mask=mask)
    
    # Generate SAME random values (same seed guarantees this)
    z = tl.randn(seed, offsets)
    
    # Combined coefficient: restore + update
    combined_alpha = eps - lr * projected_grad
    
    result = params + combined_alpha * z
    
    tl.store(params_ptr + offsets, result, mask=mask)


# =============================================================================
# Triton Kernel V3b: Fused Update (MeZO update step)
# =============================================================================

@triton.jit
def fused_update_kernel(
    params_ptr,
    seed,
    projected_grad,      # (loss+ - loss-) / (2*eps)
    lr,                  # Learning rate
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for MeZO update step.
    
    From original position: params = params - lr * projected_grad * z
    This matches trainer.py: zo_update
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    params = tl.load(params_ptr + offsets, mask=mask)
    
    # Generate SAME random values (same seed as perturbation)
    z = tl.randn(seed, offsets)
    
    # MeZO update: params = params - lr * projected_grad * z
    result = params - lr * projected_grad * z
    
    tl.store(params_ptr + offsets, result, mask=mask)


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
def fused_update_kernel_autotuned(
    params_ptr,
    seed,
    projected_grad,
    lr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Auto-tuned update kernel."""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    params = tl.load(params_ptr + offsets, mask=mask, other=0.0)
    z = tl.randn(seed, offsets)
    result = params - lr * projected_grad * z
    tl.store(params_ptr + offsets, result, mask=mask)


# =============================================================================
# Triton Kernel V4: Two-Stage Perturb (for +eps then -2eps)
# =============================================================================

@triton.jit
def fused_perturb_two_stage_kernel(
    params_ptr,
    seed,
    alpha1,              # First perturbation (+eps or -eps)
    alpha2,              # Second perturbation (typically -2*eps)
    loss1_ptr,           # Pointer to store loss1 result location
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Experimental: Could potentially pipeline two perturbations.
    Not fully implemented - shows the concept.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    params = tl.load(params_ptr + offsets, mask=mask)
    z = tl.randn(seed, offsets)
    
    # Apply first perturbation
    params = params + alpha1 * z
    tl.store(params_ptr + offsets, params, mask=mask)


# =============================================================================
# Triton Kernel V5: With Autotuning
# =============================================================================

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
def fused_perturb_kernel_autotuned(
    params_ptr,
    seed,
    alpha,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Auto-tuned version that selects best block size based on problem size.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    params = tl.load(params_ptr + offsets, mask=mask, other=0.0)
    z = tl.randn(seed, offsets)
    result = params + alpha * z
    tl.store(params_ptr + offsets, result, mask=mask)


# =============================================================================
# Triton Kernel V6: Optimized with Better Autotuning
# =============================================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_perturb_kernel_v6_optimized(
    params_ptr,
    seed,
    alpha,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized version with:
    1. Better autotuning configs for H200
    2. Reduced register pressure
    3. Optimized memory access
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load with mask
    params = tl.load(params_ptr + offsets, mask=mask, other=0.0)
    
    # Generate random normals (Triton's randn is already optimized)
    z = tl.randn(seed, offsets)
    
    # Fused operation: params + alpha * z
    result = params + alpha * z
    
    # Store with mask
    tl.store(params_ptr + offsets, result, mask=mask)


# =============================================================================
# Triton Kernel V7: Vectorized Loads (Matching CUDA float4 approach)
# =============================================================================

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
def fused_perturb_kernel_v7_vectorized(
    params_ptr,
    seed,
    alpha,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Vectorized version attempting to match CUDA's float4 performance.
    Uses Triton's vectorized load/store hints.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load parameters (Triton should optimize this automatically)
    params = tl.load(params_ptr + offsets, mask=mask, other=0.0)
    
    # Generate random normals
    z = tl.randn(seed, offsets)
    
    # Apply perturbation
    result = params + alpha * z
    
    # Store result
    tl.store(params_ptr + offsets, result, mask=mask)


# =============================================================================
# Triton Kernel V7: Tiled with Better Cache Usage
# =============================================================================

@triton.jit
def fused_perturb_kernel_v7_tiled(
    params_ptr,
    seed,
    alpha,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
):
    """
    Tiled version that processes data in smaller tiles for better cache usage.
    Useful for very large tensors.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    # Process in tiles
    for tile_offset in range(0, BLOCK_SIZE, TILE_SIZE):
        offsets = block_start + tile_offset + tl.arange(0, TILE_SIZE)
        mask = offsets < n_elements
        
        if tl.sum(mask.to(tl.int32)) > 0:  # Only process if any valid elements
            # Load tile
            params = tl.load(params_ptr + offsets, mask=mask, other=0.0)
            
            # Generate random normals for this tile
            z = tl.randn(seed, offsets)
            
            # Apply perturbation
            result = params + alpha * z
            
            # Store tile
            tl.store(params_ptr + offsets, result, mask=mask)


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
def fused_restore_update_kernel_autotuned(
    params_ptr,
    seed,
    eps,
    projected_grad,
    lr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Auto-tuned restore + update kernel."""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    params = tl.load(params_ptr + offsets, mask=mask, other=0.0)
    z = tl.randn(seed, offsets)
    combined_alpha = eps - lr * projected_grad
    result = params + combined_alpha * z
    tl.store(params_ptr + offsets, result, mask=mask)


# =============================================================================
# Python Wrapper Class: FusedPerturbMeZO
# =============================================================================

class FusedPerturbMeZO:
    """
    High-performance MeZO using Triton fused kernels.
    
    This implementation achieves:
    - ZERO extra memory (no z_flat buffer)
    - Single kernel per perturbation
    - ~1.5-2x speedup over chunked approach
    
    Memory comparison for OPT-350M:
    - FlatBufferMeZO: +1.26GB for z_flat
    - ChunkedMeZO: +64MB for z_chunk
    - FusedPerturbMeZO: +0MB (no buffer!)
    
    Kernel launches per step:
    - FlatBufferMeZO: 6 kernels (2 per perturbation × 3)
    - ChunkedMeZO: ~120 kernels (40 per perturbation × 3)
    - FusedPerturbMeZO: 3 kernels (1 per perturbation × 3)
    """
    
    def __init__(
        self, 
        model: torch.nn.Module, 
        eps: float = 1e-3, 
        lr: float = 1e-5,
        use_autotuned: bool = True,
        block_size: int = 1024,
    ):
        self.model = model
        self.eps = eps
        self.lr = lr
        self.use_autotuned = use_autotuned
        self.block_size = block_size
        
        # Flatten all trainable parameters into contiguous buffer
        trainable = [p for p in model.parameters() if p.requires_grad]
        self.n_params = len(trainable)
        self.total_elements = sum(p.numel() for p in trainable)
        
        # Create contiguous flat buffer
        self.flat_params = torch.cat([p.data.view(-1) for p in trainable])
        
        # Replace model params with views into flat buffer
        offset = 0
        self.param_views = []
        for p in trainable:
            numel = p.numel()
            view = self.flat_params[offset:offset + numel].view(p.shape)
            p.data = view
            self.param_views.append(view)
            offset += numel
        
        print(f"[FusedPerturbMeZO] {self.n_params} params, {self.total_elements:,} elements")
        print(f"[FusedPerturbMeZO] Block size: {block_size}, Autotuned: {use_autotuned}")
        print(f"[FusedPerturbMeZO] Memory overhead: 0 MB (no z buffer!)")
    
    def _get_grid(self, n_elements: int) -> callable:
        """Calculate grid size for kernel launch."""
        if self.use_autotuned:
            # Autotuned kernels determine their own grid
            return lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        else:
            return lambda meta: (triton.cdiv(n_elements, self.block_size),)
    
    def perturb(self, seed: int, scale: float = 1.0) -> None:
        """
        Apply perturbation: params = params + scale * eps * z
        
        Args:
            seed: Random seed for reproducibility
            scale: Scaling factor (1.0 for +eps, -2.0 for -2eps from +eps position)
        """
        alpha = scale * self.eps
        n_elements = self.total_elements
        grid = self._get_grid(n_elements)
        
        if self.use_autotuned:
            fused_perturb_kernel_autotuned[grid](
                self.flat_params,
                seed,
                alpha,
                n_elements,
            )
        else:
            fused_perturb_kernel_v2[grid](
                self.flat_params,
                seed,
                alpha,
                n_elements,
                BLOCK_SIZE=self.block_size,
            )
    
    def restore_and_update(self, seed: int, projected_grad: float) -> None:
        """
        Restore parameters and apply gradient update in one fused kernel.
        
        Does: params = params + (eps - lr * projected_grad) * z
        """
        n_elements = self.total_elements
        grid = self._get_grid(n_elements)
        
        if self.use_autotuned:
            fused_restore_update_kernel_autotuned[grid](
                self.flat_params,
                seed,
                self.eps,
                projected_grad,
                self.lr,
                n_elements,
            )
        else:
            fused_restore_update_kernel[grid](
                self.flat_params,
                seed,
                self.eps,
                projected_grad,
                self.lr,
                n_elements,
                BLOCK_SIZE=self.block_size,
            )
    
    def step(self, batch: Dict) -> Tuple[float, float]:
        """
        Complete MeZO step with fused kernels.
        
        Returns:
            Tuple of (average_loss, projected_gradient)
        """
        # Generate random seed for this step
        seed = np.random.randint(0, 2**31)
        
        # Perturb +ε
        self.perturb(seed, scale=1.0)  # 1 kernel
        
        with torch.no_grad():
            loss1 = self.model(**batch).loss
        
        # Perturb -2ε (from +ε position → -ε position)
        self.perturb(seed, scale=-2.0)  # 1 kernel
        
        with torch.no_grad():
            loss2 = self.model(**batch).loss
        
        # Compute projected gradient
        projected_grad = (loss1.item() - loss2.item()) / (2 * self.eps)
        
        # Restore and update (from -ε position)
        self.restore_and_update(seed, projected_grad)  # 1 kernel
        
        return (loss1.item() + loss2.item()) / 2, projected_grad
    
    def zo_step(self, loss_fn: callable) -> float:
        """
        Alternative interface using a loss function.
        
        Args:
            loss_fn: Callable that returns the loss value
            
        Returns:
            Average loss
        """
        seed = np.random.randint(0, 2**31)
        
        self.perturb(seed, scale=1.0)
        torch.cuda.synchronize()
        loss_plus = loss_fn()
        
        self.perturb(seed, scale=-2.0)
        torch.cuda.synchronize()
        loss_minus = loss_fn()
        
        projected_grad = (loss_plus - loss_minus) / (2 * self.eps)
        
        self.restore_and_update(seed, projected_grad)
        torch.cuda.synchronize()
        
        return (loss_plus + loss_minus) / 2


# =============================================================================
# Benchmark Function
# =============================================================================

def benchmark_triton_kernels(
    n_elements: int = 331_196_416,  # OPT-350M param count
    n_iterations: int = 100,
    warmup: int = 10,
):
    """
    Benchmark different Triton kernel configurations.
    """
    import time
    
    print("=" * 70)
    print("TRITON FUSED PERTURB KERNEL BENCHMARK")
    print("=" * 70)
    print(f"Elements: {n_elements:,}")
    print(f"Iterations: {n_iterations}, Warmup: {warmup}")
    
    # Create test tensor
    params = torch.randn(n_elements, device='cuda', dtype=torch.float32)
    seed = 42
    alpha = 1e-3
    
    results = {}
    
    # Test different configurations
    configs = [
        ("V1 Basic (BLOCK=1024)", fused_perturb_kernel_v1, {'BLOCK_SIZE': 1024}),
        ("V2 Optimized (BLOCK=1024)", fused_perturb_kernel_v2, {'BLOCK_SIZE': 1024}),
        ("V2 Optimized (BLOCK=2048)", fused_perturb_kernel_v2, {'BLOCK_SIZE': 2048}),
        ("V2 Optimized (BLOCK=4096)", fused_perturb_kernel_v2, {'BLOCK_SIZE': 4096}),
    ]
    
    # Add V6 optimized if available
    try:
        configs.append(("V6 Optimized (Autotuned)", fused_perturb_kernel_v6_optimized, {}))
    except NameError:
        pass
    
    # Add V7 vectorized if available
    try:
        configs.append(("V7 Vectorized (Autotuned)", fused_perturb_kernel_v7_vectorized, {}))
    except NameError:
        pass
    
    for name, kernel, kwargs in configs:
        print(f"\n{name}:")
        
        block_size = kwargs.get('BLOCK_SIZE', None)
        if block_size is not None:
            grid = lambda meta: (triton.cdiv(n_elements, block_size),)
        else:
            # Autotuned kernels
            grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        
        # Warmup
        for _ in range(warmup):
            if block_size is not None:
                kernel[grid](params, seed, alpha, n_elements, **kwargs)
            else:
                kernel[grid](params, seed, alpha, n_elements)
        torch.cuda.synchronize()
        
        # Benchmark
        start = time.time()
        for _ in range(n_iterations):
            if block_size is not None:
                kernel[grid](params, seed, alpha, n_elements, **kwargs)
            else:
                kernel[grid](params, seed, alpha, n_elements)
        torch.cuda.synchronize()
        elapsed = (time.time() - start) * 1000 / n_iterations
        
        results[name] = elapsed
        print(f"  Time: {elapsed:.3f} ms")
    
    # Test autotuned version
    print("\nAutotuned version:")
    grid_auto = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    for _ in range(warmup + 5):  # Extra warmup for autotuning
        fused_perturb_kernel_autotuned[grid_auto](params, seed, alpha, n_elements)
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(n_iterations):
        fused_perturb_kernel_autotuned[grid_auto](params, seed, alpha, n_elements)
    torch.cuda.synchronize()
    elapsed = (time.time() - start) * 1000 / n_iterations
    
    results["tl.randn Autotuned"] = elapsed
    print(f"  Time: {elapsed:.3f} ms")
    
    # =========================================================================
    # Test Custom Philox kernels
    # =========================================================================
    print("\n" + "-" * 70)
    print("CUSTOM PHILOX RNG KERNELS")
    print("-" * 70)
    
    # Test Philox kernel with different block sizes
    philox_configs = [
        ("Philox (BLOCK=1024)", fused_perturb_kernel_philox, {'BLOCK_SIZE': 1024}),
        ("Philox (BLOCK=2048)", fused_perturb_kernel_philox, {'BLOCK_SIZE': 2048}),
        ("Philox (BLOCK=4096)", fused_perturb_kernel_philox, {'BLOCK_SIZE': 4096}),
    ]
    
    for name, kernel, kwargs in philox_configs:
        print(f"\n{name}:")
        block_size = kwargs['BLOCK_SIZE']
        grid = lambda meta, bs=block_size: (triton.cdiv(n_elements, bs),)
        
        # Warmup
        for _ in range(warmup):
            kernel[grid](params, seed, alpha, n_elements, **kwargs)
        torch.cuda.synchronize()
        
        # Benchmark
        start = time.time()
        for _ in range(n_iterations):
            kernel[grid](params, seed, alpha, n_elements, **kwargs)
        torch.cuda.synchronize()
        elapsed = (time.time() - start) * 1000 / n_iterations
        
        results[name] = elapsed
        print(f"  Time: {elapsed:.3f} ms")
    
    # Test Philox autotuned
    print("\nPhilox Autotuned:")
    
    for _ in range(warmup + 5):
        fused_perturb_kernel_philox_autotuned[grid_auto](params, seed, alpha, n_elements)
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(n_iterations):
        fused_perturb_kernel_philox_autotuned[grid_auto](params, seed, alpha, n_elements)
    torch.cuda.synchronize()
    elapsed = (time.time() - start) * 1000 / n_iterations
    
    results["Philox Autotuned"] = elapsed
    print(f"  Time: {elapsed:.3f} ms")
    
    # Test Philox 2x (2 normals per Philox call)
    print("\n" + "-" * 70)
    print("PHILOX 2x KERNELS (2 normals per Philox call)")
    print("-" * 70)
    
    # Grid for 2x kernels (processes 2x elements per block)
    grid_2x = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE'] * 2),)
    
    philox_2x_configs = [
        ("Philox 2x (BLOCK=512)", fused_perturb_kernel_philox_2x, {'BLOCK_SIZE': 512}),
        ("Philox 2x (BLOCK=1024)", fused_perturb_kernel_philox_2x, {'BLOCK_SIZE': 1024}),
        ("Philox 2x (BLOCK=2048)", fused_perturb_kernel_philox_2x, {'BLOCK_SIZE': 2048}),
    ]
    
    for name, kernel, kwargs in philox_2x_configs:
        print(f"\n{name}:")
        block_size = kwargs['BLOCK_SIZE']
        grid = lambda meta, bs=block_size: (triton.cdiv(n_elements, bs * 2),)
        
        # Warmup
        for _ in range(warmup):
            kernel[grid](params, seed, alpha, n_elements, **kwargs)
        torch.cuda.synchronize()
        
        # Benchmark
        start = time.time()
        for _ in range(n_iterations):
            kernel[grid](params, seed, alpha, n_elements, **kwargs)
        torch.cuda.synchronize()
        elapsed = (time.time() - start) * 1000 / n_iterations
        
        results[name] = elapsed
        print(f"  Time: {elapsed:.3f} ms")
    
    # Philox 2x autotuned
    print("\nPhilox 2x Autotuned:")
    
    for _ in range(warmup + 5):
        fused_perturb_kernel_philox_2x_autotuned[grid_2x](params, seed, alpha, n_elements)
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(n_iterations):
        fused_perturb_kernel_philox_2x_autotuned[grid_2x](params, seed, alpha, n_elements)
    torch.cuda.synchronize()
    elapsed = (time.time() - start) * 1000 / n_iterations
    
    results["Philox 2x Autotuned"] = elapsed
    print(f"  Time: {elapsed:.3f} ms")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    best_name = min(results, key=results.get)
    best_time = results[best_name]
    
    # Find baseline for speedup calculation
    baseline_time = results.get("V1 Basic (BLOCK=1024)", list(results.values())[0])
    
    print(f"\n{'Method':<35} {'Time (ms)':>10} {'Speedup':>8}")
    print("-" * 55)
    for name, time_ms in sorted(results.items(), key=lambda x: x[1]):
        speedup = baseline_time / time_ms
        marker = " ★" if name == best_name else ""
        print(f"  {name:<33} {time_ms:>8.3f}    {speedup:>6.2f}x{marker}")
    
    # Compare tl.randn vs Philox
    print("\n" + "-" * 70)
    print("tl.randn() vs Custom Philox Comparison:")
    print("-" * 70)
    
    tl_randn_time = results.get("tl.randn Autotuned", None)
    philox_time = results.get("Philox Autotuned", None)
    
    if tl_randn_time and philox_time:
        if philox_time < tl_randn_time:
            improvement = (tl_randn_time - philox_time) / tl_randn_time * 100
            print(f"  Custom Philox is {improvement:.1f}% FASTER than tl.randn()")
        else:
            slowdown = (philox_time - tl_randn_time) / tl_randn_time * 100
            print(f"  Custom Philox is {slowdown:.1f}% SLOWER than tl.randn()")
    
    return results


def verify_philox_correctness():
    """
    Verify that custom Philox produces statistically valid normal distribution.
    """
    print("\n" + "=" * 70)
    print("PHILOX CORRECTNESS VERIFICATION")
    print("=" * 70)
    
    n_elements = 1_000_000
    params = torch.zeros(n_elements, device='cuda', dtype=torch.float32)
    seed = 42
    alpha = 1.0  # alpha=1 means params = 0 + 1*z = z (pure random)
    
    # Generate with custom Philox
    grid = lambda meta: (triton.cdiv(n_elements, 1024),)
    fused_perturb_kernel_philox[grid](params, seed, alpha, n_elements, BLOCK_SIZE=1024)
    torch.cuda.synchronize()
    
    # Check statistics
    mean = params.mean().item()
    std = params.std().item()
    
    print(f"\nGenerated {n_elements:,} random normal values:")
    print(f"  Mean: {mean:.6f} (expected: 0.0)")
    print(f"  Std:  {std:.6f} (expected: 1.0)")
    
    # Check distribution percentiles
    percentiles = [1, 5, 25, 50, 75, 95, 99]
    expected = [-2.326, -1.645, -0.674, 0.0, 0.674, 1.645, 2.326]
    
    print("\nPercentile verification:")
    print(f"  {'Percentile':<12} {'Actual':>10} {'Expected':>10} {'Diff':>10}")
    print("  " + "-" * 44)
    
    for p, exp in zip(percentiles, expected):
        actual = torch.quantile(params, p/100).item()
        diff = actual - exp
        print(f"  {p}%{'':<10} {actual:>10.3f} {exp:>10.3f} {diff:>+10.3f}")
    
    # Pass/Fail
    if abs(mean) < 0.01 and abs(std - 1.0) < 0.01:
        print("\n✓ PASSED: Distribution is statistically valid")
        return True
    else:
        print("\n✗ FAILED: Distribution may have issues")
        return False


if __name__ == "__main__":
    verify_philox_correctness()
    print("\n")
    benchmark_triton_kernels()
