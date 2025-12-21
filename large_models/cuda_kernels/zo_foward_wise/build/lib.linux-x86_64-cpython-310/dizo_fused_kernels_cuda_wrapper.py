"""
Python wrapper for CUDA kernels.

This module provides a Python interface to the CUDA kernels,
with automatic fallback to Triton kernels if CUDA extension is not available.
"""

import torch

try:
    import dizo_fused_kernels_cuda
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    print("Warning: CUDA kernels not available, falling back to Triton kernels")


def fused_compute_norms_cuda(
    param_flat: torch.Tensor,
    anchor_flat: torch.Tensor,
    offsets: torch.Tensor,
    sizes: torch.Tensor,
) -> torch.Tensor:
    """
    Compute L2 norms using CUDA kernel (with Triton fallback).
    
    Args:
        param_flat: Flattened parameters [total_elements]
        anchor_flat: Flattened anchors [total_elements]
        offsets: Starting offsets [num_params]
        sizes: Sizes [num_params]
    
    Returns:
        norms: L2 norms [num_params]
    """
    if CUDA_AVAILABLE:
        try:
            return dizo_fused_kernels_cuda.fused_compute_norms(
                param_flat, anchor_flat, offsets, sizes
            )
        except Exception as e:
            print(f"CUDA kernel failed, falling back to Triton: {e}")
            # Fall through to Triton
    
    # Fallback to Triton
    from .dizo_fused_kernels import fused_compute_norms
    return fused_compute_norms(param_flat, anchor_flat, offsets, sizes)


def fused_apply_constraints_cuda(
    param_flat: torch.Tensor,
    anchor_flat: torch.Tensor,
    offsets: torch.Tensor,
    sizes: torch.Tensor,
    constraints: torch.Tensor,
    norms: torch.Tensor,
    eps: float = 1e-8,
) -> None:
    """
    Apply constraints using CUDA kernel (with Triton fallback).
    """
    if CUDA_AVAILABLE:
        try:
            dizo_fused_kernels_cuda.fused_apply_constraints(
                param_flat, anchor_flat, offsets, sizes, constraints, norms, eps
            )
            return
        except Exception as e:
            print(f"CUDA kernel failed, falling back to Triton: {e}")
    
    # Fallback to Triton
    from .dizo_fused_kernels import fused_apply_constraints
    fused_apply_constraints(param_flat, anchor_flat, offsets, sizes, constraints, norms, eps)


def fused_reverse_constraints_cuda(
    param_flat: torch.Tensor,
    anchor_flat: torch.Tensor,
    offsets: torch.Tensor,
    sizes: torch.Tensor,
    alphas: torch.Tensor,
) -> None:
    """
    Reverse constraints using CUDA kernel (with Triton fallback).
    """
    if CUDA_AVAILABLE:
        try:
            dizo_fused_kernels_cuda.fused_reverse_constraints(
                param_flat, anchor_flat, offsets, sizes, alphas
            )
            return
        except Exception as e:
            print(f"CUDA kernel failed, falling back to Triton: {e}")
    
    # Fallback to Triton
    from .dizo_fused_kernels import fused_reverse_constraints
    fused_reverse_constraints(param_flat, anchor_flat, offsets, sizes, alphas)


def fused_update_gamma_cuda(
    gamma: torch.Tensor,
    ts: torch.Tensor,
    zs: torch.Tensor,
    grad: float,
    step_size: float,
    tau: float,
) -> None:
    """
    Update gamma using CUDA kernel (with Triton fallback).
    """
    if CUDA_AVAILABLE:
        try:
            dizo_fused_kernels_cuda.fused_update_gamma(
                gamma, ts, zs, grad, step_size, tau
            )
            return
        except Exception as e:
            print(f"CUDA kernel failed, falling back to Triton: {e}")
    
    # Fallback to Triton
    from .dizo_fused_kernels import fused_update_gamma
    fused_update_gamma(gamma, ts, zs, grad, step_size, tau)
