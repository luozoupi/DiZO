"""
KerZOO Fused Triton Kernels
============================

Fused Triton kernels for KerZOO's unique operations over MeZO:

1. Interpolation + scaled perturbation (zo_perturb_parameters, judge > 0)
   param = c_param/beta_k + (1-1/beta_k)*param + eps*k*z

2. Kernel-weighted gradient accumulation (zo_update inner loop)
   grad_buf += projected_grad[i] * z * kernel_function(k, 1)
   where kernel_function(r, t) = 15*r*(5 - 7*(r/t)^2)

3. Clipped parameter update with interpolation (zo_update outer loop)
   c_param -= lr * clip(avg_grad)
   param = c_param/beta_k + (1-1/beta_k)*param

KerZOO differs from standard MeZO by:
- A polynomial kernel function K(r) = 15r(5 - 7r^2) for gradient weighting
- Maintaining a copy of parameters with beta_k interpolation
- 3-sample gradient estimation instead of 1
- Per-tensor random scalar k in addition to per-element z

RNG uses inline Philox 4x32-10 (deterministic for a given seed and offset).
All kernels use int64 indexing for models with >2 billion elements.
"""

from __future__ import annotations
from typing import Optional

import torch
import triton
import triton.language as tl


# ============================================================
# Per-tensor perturb kernel (scalar alpha)
# ============================================================

@triton.jit
def _kerzoo_perturb_kernel(
    param_ptr,
    c_param_ptr,
    seed,
    alpha,          # = eps * k * scaling_factor (scalar, same for all elements)
    inv_beta_k,     # = 1.0 / beta_k
    comp_beta_k,    # = 1.0 - 1.0 / beta_k
    base_offset,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused perturb (judge > 0): p = c*inv_beta + comp_beta*p + alpha*z"""
    M0: tl.constexpr = 0xD2511F53
    M1: tl.constexpr = 0xCD9E8D57
    W0: tl.constexpr = 0x9E3779B9
    W1: tl.constexpr = 0xBB67AE85
    TWO_PI: tl.constexpr = 6.283185307179586

    n_elements_i64 = n_elements.to(tl.int64)
    pid = tl.program_id(0).to(tl.int64)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = offsets < n_elements_i64

    rng_offsets = offsets + base_offset

    p = tl.load(param_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    c = tl.load(c_param_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # Philox 4x32-10
    c0 = rng_offsets.to(tl.uint32)
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

    result = c * inv_beta_k + comp_beta_k * p + alpha * z
    tl.store(param_ptr + offsets, result, mask=mask)


# ============================================================
# Flat buffer perturb kernel (per-element alpha from buffer)
# ============================================================

@triton.jit
def _kerzoo_perturb_flat_kernel(
    param_ptr,
    c_param_ptr,
    alpha_ptr,      # per-element alpha buffer (pre-expanded: alpha[j] = eps*k_for_param_of_j)
    seed,
    inv_beta_k,
    comp_beta_k,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Flat buffer perturb: p = c*inv_beta + comp_beta*p + alpha[i]*z"""
    M0: tl.constexpr = 0xD2511F53
    M1: tl.constexpr = 0xCD9E8D57
    W0: tl.constexpr = 0x9E3779B9
    W1: tl.constexpr = 0xBB67AE85
    TWO_PI: tl.constexpr = 6.283185307179586

    n_elements_i64 = n_elements.to(tl.int64)
    pid = tl.program_id(0).to(tl.int64)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = offsets < n_elements_i64

    p = tl.load(param_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    c = tl.load(c_param_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    alpha = tl.load(alpha_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # Philox 4x32-10 (offsets as counter — no base_offset for flat buffer)
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

    result = c * inv_beta_k + comp_beta_k * p + alpha * z
    tl.store(param_ptr + offsets, result, mask=mask)


# ============================================================
# Per-tensor gradient accumulation kernel (scalar weight)
# ============================================================

@triton.jit
def _kerzoo_accum_kernel(
    grad_buf_ptr,
    seed,
    weight,         # scalar: projected_grad[i] * kernel_function(k, 1)
    base_offset,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Accumulate: grad_buf += weight * z"""
    M0: tl.constexpr = 0xD2511F53
    M1: tl.constexpr = 0xCD9E8D57
    W0: tl.constexpr = 0x9E3779B9
    W1: tl.constexpr = 0xBB67AE85
    TWO_PI: tl.constexpr = 6.283185307179586

    n_elements_i64 = n_elements.to(tl.int64)
    pid = tl.program_id(0).to(tl.int64)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = offsets < n_elements_i64

    rng_offsets = offsets + base_offset

    g = tl.load(grad_buf_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    c0 = rng_offsets.to(tl.uint32)
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

    g = g + weight * z
    tl.store(grad_buf_ptr + offsets, g, mask=mask)


# ============================================================
# Flat buffer gradient accumulation kernel (per-element weight)
# ============================================================

@triton.jit
def _kerzoo_accum_flat_kernel(
    grad_buf_ptr,
    weight_ptr,     # per-element weight buffer (pre-expanded)
    seed,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Flat buffer accumulate: grad_buf += weight[i] * z"""
    M0: tl.constexpr = 0xD2511F53
    M1: tl.constexpr = 0xCD9E8D57
    W0: tl.constexpr = 0x9E3779B9
    W1: tl.constexpr = 0xBB67AE85
    TWO_PI: tl.constexpr = 6.283185307179586

    n_elements_i64 = n_elements.to(tl.int64)
    pid = tl.program_id(0).to(tl.int64)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = offsets < n_elements_i64

    g = tl.load(grad_buf_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

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

    g = g + w * z
    tl.store(grad_buf_ptr + offsets, g, mask=mask)


# ============================================================
# Parameter update kernel (shared between per-tensor and flat)
# ============================================================

@triton.jit
def _kerzoo_update_param_kernel(
    param_ptr,
    c_param_ptr,
    grad_buf_ptr,
    lr_clip_div3,   # = lr * min(1, threshold/norm) / 3
    inv_beta_k,
    comp_beta_k,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused update: avg_grad=g*factor; c-=avg_grad; p=c*inv_beta+comp_beta*p"""
    n_elements_i64 = n_elements.to(tl.int64)
    pid = tl.program_id(0).to(tl.int64)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = offsets < n_elements_i64

    p = tl.load(param_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    c = tl.load(c_param_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(grad_buf_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # avg_grad = g / 3 * clip_factor, combined into lr_clip_div3
    scaled_grad = g * lr_clip_div3
    c_new = c - scaled_grad
    p_new = c_new * inv_beta_k + comp_beta_k * p

    tl.store(c_param_ptr + offsets, c_new, mask=mask)
    tl.store(param_ptr + offsets, p_new, mask=mask)


# ============================================================
# Python wrapper functions
# ============================================================

def _grid(n_elements: int, block_size: int) -> tuple:
    return (triton.cdiv(n_elements, block_size),)


def kerzoo_fused_perturb(
    param: torch.Tensor,
    c_param: torch.Tensor,
    seed: int,
    alpha: float,
    inv_beta_k: float,
    comp_beta_k: float,
    base_offset: int = 0,
    block_size: int = 1024,
):
    """Per-tensor fused perturb: p = c*inv_beta + comp_beta*p + alpha*z"""
    n = param.numel()
    _kerzoo_perturb_kernel[_grid(n, block_size)](
        param, c_param, seed, alpha, inv_beta_k, comp_beta_k,
        base_offset, n, BLOCK_SIZE=block_size,
    )


def kerzoo_fused_perturb_flat(
    param: torch.Tensor,
    c_param: torch.Tensor,
    alpha_buf: torch.Tensor,
    seed: int,
    inv_beta_k: float,
    comp_beta_k: float,
    block_size: int = 1024,
):
    """Flat buffer fused perturb: p = c*inv_beta + comp_beta*p + alpha[i]*z"""
    n = param.numel()
    _kerzoo_perturb_flat_kernel[_grid(n, block_size)](
        param, c_param, alpha_buf, seed, inv_beta_k, comp_beta_k,
        n, BLOCK_SIZE=block_size,
    )


def kerzoo_fused_accum(
    grad_buf: torch.Tensor,
    seed: int,
    weight: float,
    base_offset: int = 0,
    block_size: int = 1024,
):
    """Per-tensor fused accum: g += weight * z"""
    n = grad_buf.numel()
    _kerzoo_accum_kernel[_grid(n, block_size)](
        grad_buf, seed, weight, base_offset, n, BLOCK_SIZE=block_size,
    )


def kerzoo_fused_accum_flat(
    grad_buf: torch.Tensor,
    weight_buf: torch.Tensor,
    seed: int,
    block_size: int = 1024,
):
    """Flat buffer fused accum: g += weight[i] * z"""
    n = grad_buf.numel()
    _kerzoo_accum_flat_kernel[_grid(n, block_size)](
        grad_buf, weight_buf, seed, n, BLOCK_SIZE=block_size,
    )


def kerzoo_fused_update_param(
    param: torch.Tensor,
    c_param: torch.Tensor,
    grad_buf: torch.Tensor,
    lr_clip_div3: float,
    inv_beta_k: float,
    comp_beta_k: float,
    block_size: int = 1024,
):
    """Fused param update: c -= lr*clip*g/3; p = c*inv_beta + comp_beta*p"""
    n = param.numel()
    _kerzoo_update_param_kernel[_grid(n, block_size)](
        param, c_param, grad_buf,
        lr_clip_div3, inv_beta_k, comp_beta_k,
        n, BLOCK_SIZE=block_size,
    )
