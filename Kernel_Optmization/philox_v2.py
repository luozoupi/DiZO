"""
Optimized Philox RNG Kernels for ZO Optimizer Perturbation
==========================================================

New versions for benchmarking — do NOT replace existing kernels.

Optimizations targeting NCU-identified bottlenecks:
A. PTX mul.hi.u32 / mul.lo.u32 — eliminates uint64 cast that causes FP64
   pipeline stalls (87.6% FP64 utilization at large scale)
B. Reduced Philox rounds (7 vs 10) — sufficient for ML training (JAX/XLA
   uses Philox-4x32-7), saves 30% of Philox compute
C. 2x normals per Philox call — uses all 4 uint32 outputs via dual
   Box-Muller, doubles throughput per invocation
D. tl.randn() baseline — Triton's optimized builtin (46 regs vs 55 regs)
E. Wider thread configs — num_warps=8 (256 threads) for better occupancy

Register pressure comparison (NCU measured, BLOCK_SIZE=1024):
  Original Philox (uint64 path) : 55 regs/thread → 75% theoretical occ
  ZO2 tl.randn (Triton builtin) : 46 regs/thread → 83% theoretical occ
  ZO2 CUDA (native __umulhi)     : 18 regs/thread → 100% theoretical occ
  Target for v2 PTX mulhi        : <40 regs/thread → >83% theoretical occ
"""

import torch
import triton
import triton.language as tl


# ============================================================
# PTX Inline Assembly Helpers (32-bit integer multiply)
# ============================================================

@triton.jit
def _mul_lo_u32(a, b):
    """Low 32 bits of unsigned 32x32 multiply via PTX."""
    return tl.inline_asm_elementwise(
        "mul.lo.u32 $0, $1, $2;",
        "=r,r,r", [a, b],
        dtype=tl.uint32, is_pure=True, pack=1,
    )


@triton.jit
def _mul_hi_u32(a, b):
    """High 32 bits of unsigned 32x32 multiply via PTX.
    Replaces: prod = a.to(uint64) * b; hi = (prod >> 32).to(uint32)
    Avoids FP64 pipeline entirely."""
    return tl.inline_asm_elementwise(
        "mul.hi.u32 $0, $1, $2;",
        "=r,r,r", [a, b],
        dtype=tl.uint32, is_pure=True, pack=1,
    )


# ============================================================
# Philox Round using 32-bit multiply only (no uint64 / FP64)
# ============================================================

@triton.jit
def _philox_round_32(c0, c1, c2, c3, k0, k1, M0, M1):
    """Single Philox 4x32 round using PTX mul.hi/lo (pure uint32)."""
    lo0 = _mul_lo_u32(c0, M0)
    hi0 = _mul_hi_u32(c0, M0)
    lo1 = _mul_lo_u32(c2, M1)
    hi1 = _mul_hi_u32(c2, M1)
    return hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0


# ============================================================
# Box-Muller Transform
# ============================================================

@triton.jit
def _box_muller_1(c0, c1):
    """Convert 2 uint32 → 1 normal via Box-Muller."""
    TWO_PI = 6.283185307179586
    u1 = (c0.to(tl.float32) + 0.5) * (1.0 / 4294967296.0)
    u2 = (c1.to(tl.float32) + 0.5) * (1.0 / 4294967296.0)
    r = tl.sqrt(-2.0 * tl.log(u1))
    return r * tl.cos(TWO_PI * u2)


@triton.jit
def _box_muller_2(c0, c1, c2, c3):
    """Convert 4 uint32 → 2 normals via dual Box-Muller."""
    TWO_PI = 6.283185307179586
    u1 = (c0.to(tl.float32) + 0.5) * (1.0 / 4294967296.0)
    u2 = (c1.to(tl.float32) + 0.5) * (1.0 / 4294967296.0)
    u3 = (c2.to(tl.float32) + 0.5) * (1.0 / 4294967296.0)
    u4 = (c3.to(tl.float32) + 0.5) * (1.0 / 4294967296.0)
    r1 = tl.sqrt(-2.0 * tl.log(u1))
    r2 = tl.sqrt(-2.0 * tl.log(u3))
    z0 = r1 * tl.cos(TWO_PI * u2)
    z1 = r2 * tl.cos(TWO_PI * u4)
    return z0, z1


# ============================================================
# Philox state init + N rounds (shared by all kernels)
# ============================================================

@triton.jit
def _philox_init(offsets, seed):
    """Initialize Philox 4x32 state from offsets and seed."""
    c0 = offsets.to(tl.uint32)
    c1 = tl.zeros_like(c0)
    c2 = tl.zeros_like(c0)
    c3 = tl.zeros_like(c0)
    k0 = tl.full(c0.shape, seed, dtype=tl.uint32)
    k1 = tl.zeros_like(k0)
    M0 = tl.full(c0.shape, 0xD2511F53, dtype=tl.uint32)
    M1 = tl.full(c0.shape, 0xCD9E8D57, dtype=tl.uint32)
    return c0, c1, c2, c3, k0, k1, M0, M1


# ============================================================
# Kernel V2A: PTX mulhi, 10 rounds (just fix FP64)
# ============================================================

@triton.jit
def fused_perturb_v2a_kernel(
    params_ptr, seed, alpha, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Philox-4x32-10 with PTX mulhi. No FP64 pipeline usage."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    params = tl.load(params_ptr + offsets, mask=mask, other=0.0)

    c0, c1, c2, c3, k0, k1, M0, M1 = _philox_init(offsets, seed)
    for _ in range(10):
        c0, c1, c2, c3 = _philox_round_32(c0, c1, c2, c3, k0, k1, M0, M1)
        k0 = k0 + 0x9E3779B9
        k1 = k1 + 0xBB67AE85

    z = _box_muller_1(c0, c1)
    tl.store(params_ptr + offsets, params + alpha * z, mask=mask)


# ============================================================
# Kernel V2B: PTX mulhi, 7 rounds (reduced compute)
# ============================================================

@triton.jit
def fused_perturb_v2b_kernel(
    params_ptr, seed, alpha, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Philox-4x32-7 with PTX mulhi. 30% less Philox compute."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    params = tl.load(params_ptr + offsets, mask=mask, other=0.0)

    c0, c1, c2, c3, k0, k1, M0, M1 = _philox_init(offsets, seed)
    for _ in range(7):
        c0, c1, c2, c3 = _philox_round_32(c0, c1, c2, c3, k0, k1, M0, M1)
        k0 = k0 + 0x9E3779B9
        k1 = k1 + 0xBB67AE85

    z = _box_muller_1(c0, c1)
    tl.store(params_ptr + offsets, params + alpha * z, mask=mask)


# ============================================================
# Kernel V2C: PTX mulhi, 7 rounds, 2x normals per Philox call
# ============================================================

@triton.jit
def fused_perturb_v2c_kernel(
    params_ptr, seed, alpha, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Philox-7, PTX mulhi, 2x normals. Each block handles 2*BLOCK_SIZE elems."""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE * 2

    offsets_a = block_start + tl.arange(0, BLOCK_SIZE)
    offsets_b = block_start + BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask_a = offsets_a < n_elements
    mask_b = offsets_b < n_elements

    params_a = tl.load(params_ptr + offsets_a, mask=mask_a, other=0.0)
    params_b = tl.load(params_ptr + offsets_b, mask=mask_b, other=0.0)

    # Philox uses offsets_a as counter → produces 4 uint32 → 2 normals
    c0, c1, c2, c3, k0, k1, M0, M1 = _philox_init(offsets_a, seed)
    for _ in range(7):
        c0, c1, c2, c3 = _philox_round_32(c0, c1, c2, c3, k0, k1, M0, M1)
        k0 = k0 + 0x9E3779B9
        k1 = k1 + 0xBB67AE85

    z_a, z_b = _box_muller_2(c0, c1, c2, c3)

    tl.store(params_ptr + offsets_a, params_a + alpha * z_a, mask=mask_a)
    tl.store(params_ptr + offsets_b, params_b + alpha * z_b, mask=mask_b)


# ============================================================
# Kernel V2D: tl.randn() — Triton builtin (reference)
# ============================================================

@triton.jit
def fused_perturb_v2d_kernel(
    params_ptr, seed, alpha, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Using Triton's built-in tl.randn (optimized Philox)."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    params = tl.load(params_ptr + offsets, mask=mask, other=0.0)
    z = tl.randn(seed, offsets)
    tl.store(params_ptr + offsets, params + alpha * z, mask=mask)


# ============================================================
# Kernel V2E: Autotuned (best of all PTX mulhi configs)
# ============================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_perturb_v2e_autotuned_kernel(
    params_ptr, seed, alpha, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Autotuned: PTX mulhi, 7 rounds, various block/warp configs."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    params = tl.load(params_ptr + offsets, mask=mask, other=0.0)

    c0, c1, c2, c3, k0, k1, M0, M1 = _philox_init(offsets, seed)
    for _ in range(7):
        c0, c1, c2, c3 = _philox_round_32(c0, c1, c2, c3, k0, k1, M0, M1)
        k0 = k0 + 0x9E3779B9
        k1 = k1 + 0xBB67AE85

    z = _box_muller_1(c0, c1)
    tl.store(params_ptr + offsets, params + alpha * z, mask=mask)


# ============================================================
# Kernel V2F: Autotuned tl.randn (for fair comparison)
# ============================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_perturb_v2f_autotuned_kernel(
    params_ptr, seed, alpha, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Autotuned tl.randn with various block/warp configs."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    params = tl.load(params_ptr + offsets, mask=mask, other=0.0)
    z = tl.randn(seed, offsets)
    tl.store(params_ptr + offsets, params + alpha * z, mask=mask)


# ============================================================
# Update kernels: params = params - lr * grad * z
# grad is provided as a GPU scalar tensor pointer
# ============================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_update_v2e_autotuned_kernel(
    params_ptr, grad_ptr, seed, lr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Autotuned update kernel using PTX mulhi Philox-7r path."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    params = tl.load(params_ptr + offsets, mask=mask, other=0.0)
    grad = tl.load(grad_ptr)

    c0, c1, c2, c3, k0, k1, M0, M1 = _philox_init(offsets, seed)
    for _ in range(7):
        c0, c1, c2, c3 = _philox_round_32(c0, c1, c2, c3, k0, k1, M0, M1)
        k0 = k0 + 0x9E3779B9
        k1 = k1 + 0xBB67AE85

    z = _box_muller_1(c0, c1)
    tl.store(params_ptr + offsets, params - lr * grad * z, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_update_v2f_autotuned_kernel(
    params_ptr, grad_ptr, seed, lr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Autotuned update kernel using tl.randn path."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    params = tl.load(params_ptr + offsets, mask=mask, other=0.0)
    grad = tl.load(grad_ptr)
    z = tl.randn(seed, offsets)
    tl.store(params_ptr + offsets, params - lr * grad * z, mask=mask)


# ============================================================
# Python Wrapper Functions (consistent interface)
# ============================================================

def perturb_v2a(params: torch.Tensor, seed: int, alpha: float,
                block_size: int = 1024, num_warps: int = 4):
    """V2A: PTX mulhi, 10 rounds."""
    n = params.numel()
    grid = (triton.cdiv(n, block_size),)
    fused_perturb_v2a_kernel[grid](
        params, seed, alpha, n,
        BLOCK_SIZE=block_size, num_warps=num_warps,
    )


def perturb_v2b(params: torch.Tensor, seed: int, alpha: float,
                block_size: int = 1024, num_warps: int = 4):
    """V2B: PTX mulhi, 7 rounds."""
    n = params.numel()
    grid = (triton.cdiv(n, block_size),)
    fused_perturb_v2b_kernel[grid](
        params, seed, alpha, n,
        BLOCK_SIZE=block_size, num_warps=num_warps,
    )


def perturb_v2c(params: torch.Tensor, seed: int, alpha: float,
                block_size: int = 512, num_warps: int = 4):
    """V2C: PTX mulhi, 7 rounds, 2x normals."""
    n = params.numel()
    grid = (triton.cdiv(n, block_size * 2),)
    fused_perturb_v2c_kernel[grid](
        params, seed, alpha, n,
        BLOCK_SIZE=block_size, num_warps=num_warps,
    )


def perturb_v2d(params: torch.Tensor, seed: int, alpha: float,
                block_size: int = 1024, num_warps: int = 4):
    """V2D: tl.randn (Triton builtin)."""
    n = params.numel()
    grid = (triton.cdiv(n, block_size),)
    fused_perturb_v2d_kernel[grid](
        params, seed, alpha, n,
        BLOCK_SIZE=block_size, num_warps=num_warps,
    )


def perturb_v2e(params: torch.Tensor, seed: int, alpha: float):
    """V2E: Autotuned PTX mulhi, 7 rounds."""
    n = params.numel()
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    fused_perturb_v2e_autotuned_kernel[grid](params, seed, alpha, n)


def perturb_v2f(params: torch.Tensor, seed: int, alpha: float):
    """V2F: Autotuned tl.randn."""
    n = params.numel()
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    fused_perturb_v2f_autotuned_kernel[grid](params, seed, alpha, n)


def update_v2e(params: torch.Tensor, grad: torch.Tensor, seed: int, lr: float):
    """V2E update: Autotuned PTX mulhi, 7 rounds, grad read from GPU scalar tensor."""
    n = params.numel()
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    fused_update_v2e_autotuned_kernel[grid](params, grad, seed, lr, n)


def update_v2f(params: torch.Tensor, grad: torch.Tensor, seed: int, lr: float):
    """V2F update: Autotuned tl.randn, grad read from GPU scalar tensor."""
    n = params.numel()
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    fused_update_v2f_autotuned_kernel[grid](params, grad, seed, lr, n)
