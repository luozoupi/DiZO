"""
Fused CUDA/Triton Kernels for DiZO zo_forward Optimization

This module provides optimized kernels for:
1. Batch norm computation across parameter groups
2. Fused constraint application (compute alpha + apply projection)
3. Fused constraint reversal
4. Fused gamma perturbation with Philox RNG
5. Fused gamma update with clipping

All kernels operate on flattened parameter tensors to minimize kernel launches.
"""

import torch
import triton
import triton.language as tl
import math
from typing import Tuple, List, Optional

# Reuse Philox implementation from Perturb_wise
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../Perturb_wise'))
try:
    from triton_fused_perturb import (
        PHILOX_M0, PHILOX_M1, PHILOX_W0, PHILOX_W1,
        philox_round, philox_generate, philox_next_normal
    )
except ImportError:
    # Fallback: inline Philox implementation
    PHILOX_M0: tl.constexpr = 0xD2511F53
    PHILOX_M1: tl.constexpr = 0xCD9E8D57
    PHILOX_W0: tl.constexpr = 0x9E3779B9
    PHILOX_W1: tl.constexpr = 0xBB67AE85


# =============================================================================
# Kernel 1: Fused Norm Computation
# =============================================================================

@triton.jit
def fused_norm_kernel(
    param_flat_ptr,
    anchor_flat_ptr,
    offsets_ptr,
    sizes_ptr,
    norms_out_ptr,
    num_params: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute L2 norms for multiple parameter groups in parallel.
    
    Each program processes one parameter group.
    """
    pid = tl.program_id(0)
    
    if pid >= num_params:
        return
    
    # Load offset and size for this parameter group
    offset = tl.load(offsets_ptr + pid)
    size = tl.load(sizes_ptr + pid)
    
    # Compute norm using block-wise reduction
    acc = 0.0
    for base in range(0, size, BLOCK_SIZE):
        idx = offset + base + tl.arange(0, BLOCK_SIZE)
        mask = (base + tl.arange(0, BLOCK_SIZE)) < size
        
        param_val = tl.load(param_flat_ptr + idx, mask=mask, other=0.0)
        anchor_val = tl.load(anchor_flat_ptr + idx, mask=mask, other=0.0)
        
        diff = param_val - anchor_val
        # Apply mask before summing (multiply by mask to zero out invalid elements)
        sq_diff = diff * diff
        masked_sq_diff = tl.where(mask, sq_diff, 0.0)
        acc += tl.sum(masked_sq_diff)
    
    # Write result
    norm = tl.sqrt(acc + 1e-8)  # Add epsilon for numerical stability
    tl.store(norms_out_ptr + pid, norm)


def fused_compute_norms(
    param_flat: torch.Tensor,
    anchor_flat: torch.Tensor,
    offsets: torch.Tensor,
    sizes: torch.Tensor,
) -> torch.Tensor:
    """
    Compute L2 norms for multiple parameter groups.
    
    Args:
        param_flat: Flattened parameters [total_elements]
        anchor_flat: Flattened anchors [total_elements]
        offsets: Starting offsets for each param group [num_params]
        sizes: Sizes of each param group [num_params]
    
    Returns:
        norms: L2 norms for each parameter group [num_params]
    """
    num_params = offsets.shape[0]
    norms = torch.empty(num_params, device=param_flat.device, dtype=param_flat.dtype)
    
    BLOCK_SIZE = 1024
    grid = lambda meta: (num_params,)
    
    fused_norm_kernel[grid](
        param_flat,
        anchor_flat,
        offsets,
        sizes,
        norms,
        num_params,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return norms


# =============================================================================
# Kernel 2: Fused Constraint Application
# =============================================================================

@triton.jit
def fused_apply_constraints_kernel(
    param_flat_ptr,
    anchor_flat_ptr,
    offsets_ptr,
    sizes_ptr,
    constraints_ptr,  # gamma values
    norms_ptr,  # pre-computed norms
    num_params: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Apply constraints to parameters in a fused kernel.
    
    Operation: param = anchor + (param - anchor) * alpha
    where alpha = constraint / (norm + eps)
    """
    pid = tl.program_id(0)
    
    if pid >= num_params:
        return
    
    # Load constraint and norm for this parameter group
    constraint = tl.load(constraints_ptr + pid)
    norm = tl.load(norms_ptr + pid)
    
    # Compute alpha
    alpha = constraint / (norm + eps)
    
    # Load offset and size
    offset = tl.load(offsets_ptr + pid)
    size = tl.load(sizes_ptr + pid)
    
    # Apply constraint
    for i in range(0, size, BLOCK_SIZE):
        idx = offset + i + tl.arange(0, BLOCK_SIZE)
        mask = (i + tl.arange(0, BLOCK_SIZE)) < size
        
        param_val = tl.load(param_flat_ptr + idx, mask=mask, other=0.0)
        anchor_val = tl.load(anchor_flat_ptr + idx, mask=mask, other=0.0)
        
        # Project: param = anchor + (param - anchor) * alpha
        diff = param_val - anchor_val
        new_val = anchor_val + diff * alpha
        
        tl.store(param_flat_ptr + idx, new_val, mask=mask)


def fused_apply_constraints(
    param_flat: torch.Tensor,
    anchor_flat: torch.Tensor,
    offsets: torch.Tensor,
    sizes: torch.Tensor,
    constraints: torch.Tensor,
    norms: torch.Tensor,
    eps: float = 1e-8,
) -> None:
    """
    Apply constraints to flattened parameters.
    
    Args:
        param_flat: Flattened parameters [total_elements] (modified in-place)
        anchor_flat: Flattened anchors [total_elements]
        offsets: Starting offsets [num_params]
        sizes: Sizes [num_params]
        constraints: Constraint values (gamma) [num_params]
        norms: Pre-computed norms [num_params]
        eps: Epsilon for numerical stability
    """
    num_params = offsets.shape[0]
    BLOCK_SIZE = 1024
    grid = lambda meta: (num_params,)
    
    fused_apply_constraints_kernel[grid](
        param_flat,
        anchor_flat,
        offsets,
        sizes,
        constraints,
        norms,
        num_params,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )


# =============================================================================
# Kernel 3: Fused Constraint Reversal
# =============================================================================

@triton.jit
def fused_reverse_constraints_kernel(
    param_flat_ptr,
    anchor_flat_ptr,
    offsets_ptr,
    sizes_ptr,
    alphas_ptr,  # alpha values from apply_constraints
    num_params: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Reverse constraint application.
    
    Operation: param = anchor + (param - anchor) / alpha
    """
    pid = tl.program_id(0)
    
    if pid >= num_params:
        return
    
    # Load alpha
    alpha = tl.load(alphas_ptr + pid)
    
    # Load offset and size
    offset = tl.load(offsets_ptr + pid)
    size = tl.load(sizes_ptr + pid)
    
    # Reverse constraint
    for i in range(0, size, BLOCK_SIZE):
        idx = offset + i + tl.arange(0, BLOCK_SIZE)
        mask = (i + tl.arange(0, BLOCK_SIZE)) < size
        
        param_val = tl.load(param_flat_ptr + idx, mask=mask, other=0.0)
        anchor_val = tl.load(anchor_flat_ptr + idx, mask=mask, other=0.0)
        
        # Reverse: param = anchor + (param - anchor) / alpha
        diff = param_val - anchor_val
        new_val = anchor_val + diff / alpha
        
        tl.store(param_flat_ptr + idx, new_val, mask=mask)


def fused_reverse_constraints(
    param_flat: torch.Tensor,
    anchor_flat: torch.Tensor,
    offsets: torch.Tensor,
    sizes: torch.Tensor,
    alphas: torch.Tensor,
) -> None:
    """
    Reverse constraint application.
    
    Args:
        param_flat: Flattened parameters [total_elements] (modified in-place)
        anchor_flat: Flattened anchors [total_elements]
        offsets: Starting offsets [num_params]
        sizes: Sizes [num_params]
        alphas: Alpha values from apply_constraints [num_params]
    """
    num_params = offsets.shape[0]
    BLOCK_SIZE = 1024
    grid = lambda meta: (num_params,)
    
    fused_reverse_constraints_kernel[grid](
        param_flat,
        anchor_flat,
        offsets,
        sizes,
        alphas,
        num_params,
        BLOCK_SIZE=BLOCK_SIZE,
    )


# =============================================================================
# Kernel 4: Fused Gamma Perturbation
# =============================================================================

# For gamma perturbation, we use PyTorch's normal generation
# since it's only a small number of scalars (one per constraint)
# The overhead of a Triton kernel for small arrays isn't worth it


# Gamma perturbation uses PyTorch operations (see fused_perturb_gamma function)
# since num_params is small (< 1000 typically)


def fused_perturb_gamma(
    gamma: torch.Tensor,
    ts: torch.Tensor,
    seed: int,
    delta: float,
    tau: float,
    zo_eps: float,
    zs: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Perturb gamma constraints.
    
    For small arrays (num_params is typically < 1000), PyTorch operations
    are efficient enough. We use PyTorch's normal generation.
    
    Args:
        gamma: Constraint values [num_params] (modified in-place)
        ts: Norm values [num_params]
        seed: Random seed
        delta: Perturbation direction (+1, -2, +1)
        tau: Clip range
        zo_eps: Perturbation epsilon
        zs: Precomputed z values [num_params] (optional)
    
    Returns:
        zs: Generated or reused z values [num_params]
    """
    if zs is None:
        # Generate random z values
        torch.manual_seed(seed)
        zs = torch.normal(0, 1, size=gamma.shape, device=gamma.device, dtype=gamma.dtype)
        
        # Clip z based on tau and ts
        clip_min = (-tau / zo_eps) * ts
        clip_max = (tau / zo_eps) * ts
        zs = torch.clamp(zs, clip_min, clip_max)
    
    # Update gamma
    gamma.data = gamma.data + delta * zs * zo_eps
    
    return zs


# =============================================================================
# Kernel 5: Fused Gamma Update
# =============================================================================

@triton.jit
def fused_update_gamma_kernel(
    gamma_ptr,
    ts_ptr,
    zs_ptr,
    grad: tl.constexpr,
    step_size: tl.constexpr,
    tau: tl.constexpr,
    num_params: tl.constexpr,
):
    """
    Update gamma with gradient and clip.
    
    Operation: gamma = clip(gamma - step_size * ts * grad * z, 
                           (1-tau)*ts, (1+tau)*ts)
    """
    pid = tl.program_id(0)
    
    if pid >= num_params:
        return
    
    gamma = tl.load(gamma_ptr + pid)
    t = tl.load(ts_ptr + pid)
    z = tl.load(zs_ptr + pid)
    
    # Compute update
    gamma_new = gamma - step_size * t * grad * z
    
    # Clip
    gamma_min = (1.0 - tau) * t
    gamma_max = (1.0 + tau) * t
    gamma_clipped = tl.maximum(tl.minimum(gamma_new, gamma_max), gamma_min)
    
    tl.store(gamma_ptr + pid, gamma_clipped)


def fused_update_gamma(
    gamma: torch.Tensor,
    ts: torch.Tensor,
    zs: torch.Tensor,
    grad: float,
    step_size: float,
    tau: float,
) -> None:
    """
    Update gamma constraints with gradient.
    
    Args:
        gamma: Constraint values [num_params] (modified in-place)
        ts: Norm values [num_params]
        zs: Z values from perturbation [num_params]
        grad: Gradient estimate (scalar)
        step_size: Step size
        tau: Clip range
    """
    num_params = gamma.shape[0]
    grid = lambda meta: (num_params,)
    
    fused_update_gamma_kernel[grid](
        gamma,
        ts,
        zs,
        grad,
        step_size,
        tau,
        num_params,
    )

