#!/usr/bin/env python3
"""
Dispatch Overhead Benchmark for ZO Methods: HiZOO, DiZO, ZO2

Following the same methodology as benchmark_bubble.py (MeZO), this script
benchmarks per-tensor dispatch overhead for each ZO method's unique operations:

  --method hizoo:  Hessian-scaled perturbation + fused update-restore
  --method dizo:   DiZO distance constraint operations (apply + reverse)
  --method zo2:    Dual-perturbation pattern (theta+eps*z, theta-eps*z)

For each method, compares:
  (A) Per-tensor PyTorch baseline (what the trainer actually does)
  (B) Flat buffer PyTorch equivalent (eliminates loop dispatch overhead)
  (C) Flat buffer Triton fused kernel (eliminates dispatch + fuses operations)

Usage:
    python benchmark_bubble_methods.py --method hizoo
    python benchmark_bubble_methods.py --method dizo
    python benchmark_bubble_methods.py --method zo2
    python benchmark_bubble_methods.py --method hizoo --models opt-350m opt-2.7b
    python benchmark_bubble_methods.py --method hizoo --dtype fp32
"""

import os
import re
import sys
import csv
import gc
import time
import argparse
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Triton (for per-tensor DiZO fused kernels)
# ---------------------------------------------------------------------------

HAS_TRITON_DIRECT = False
try:
    import triton
    import triton.language as tl
    HAS_TRITON_DIRECT = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Kernel availability
# ---------------------------------------------------------------------------

HAS_HIZOO_TRITON = False
HAS_DIZO_TRITON_V2 = False
HAS_ZO2_TRITON = False

# HiZOO Triton kernels
try:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "HiZOO" / "cuda_kernels"))
    from hizoo_fused_ops import hizoo_fused_perturb, hizoo_fused_update
    HAS_HIZOO_TRITON = True
except ImportError as e:
    print(f"Warning: HiZOO Triton kernels not available: {e}")

# DiZO Triton V2 kernels
try:
    zo_path = str(Path(__file__).resolve().parent.parent
                  / "DiZO" / "large_models" / "cuda_kernels" / "zo_foward_wise")
    sys.path.insert(0, zo_path)
    from dizo_fused_kernels_v2 import FusedDiZOKernelsV2
    HAS_DIZO_TRITON_V2 = True
except ImportError as e:
    print(f"Warning: DiZO Triton V2 kernels not available: {e}")

# ZO2 Triton kernels
try:
    zo2_path = str(Path(__file__).resolve().parent.parent
                   / "zo2" / "cuda_kernels" / "block_wise")
    sys.path.insert(0, zo2_path)
    from triton_kernels import (
        zo2_fused_dual_perturb,
        zo2_fused_dual_perturb_and_update,
        zo2_fused_block_update,
    )
    HAS_ZO2_TRITON = True
except ImportError as e:
    print(f"Warning: ZO2 Triton kernels not available: {e}")

# ZO2 CUDA kernels (JIT compiled)
HAS_ZO2_CUDA = False
_zo2_cuda = None
try:
    import torch.utils.cpp_extension as _ext
    _ext._check_cuda_version = lambda *a, **k: None  # CUDA 13.0 vs PyTorch 12.8
    _zo2_cuda_src = str(Path(__file__).resolve().parent.parent
                        / "zo2" / "cuda_kernels" / "block_wise" / "zo2_cuda_kernels.cu")
    if Path(_zo2_cuda_src).exists():
        _zo2_cuda = _ext.load(
            name='zo2_cuda_kernels_bench',
            sources=[_zo2_cuda_src],
            extra_cuda_cflags=['-O3', '-use_fast_math', '--expt-relaxed-constexpr',
                               '-gencode', 'arch=compute_90,code=sm_90',
                               '-gencode', 'arch=compute_120,code=sm_120'],
            extra_cflags=['-O3'],
            verbose=False,
        )
        HAS_ZO2_CUDA = True
except Exception as e:
    print(f"Warning: ZO2 CUDA kernels not available: {e}")

# KerZOO Triton kernels (local file)
HAS_KERZOO_TRITON = False
try:
    from kerzoo_fused_ops import (
        kerzoo_fused_perturb,
        kerzoo_fused_perturb_flat,
        kerzoo_fused_accum,
        kerzoo_fused_accum_flat,
        kerzoo_fused_update_param,
    )
    HAS_KERZOO_TRITON = True
except ImportError as e:
    print(f"Warning: KerZOO Triton kernels not available: {e}")

# ---------------------------------------------------------------------------
# Parameter shapes
# ---------------------------------------------------------------------------

SHAPES_DIR = (Path(__file__).resolve().parent.parent
              / "DiZO" / "large_models" / "cuda_kernels")

MODEL_SHAPE_FILES = {
    "opt-350m":  SHAPES_DIR / "opt-350m_parameter_shapes.txt",
    "opt-2.7b":  SHAPES_DIR / "opt-2_7b_parameter_shapes.txt",
    "opt-6.7b":  SHAPES_DIR / "opt-6_7b_parameter_shapes.txt",
    "opt-13b":   SHAPES_DIR / "opt-13b_parameter_shapes.txt",
}


def parse_shapes_file(path):
    shapes = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("=") or line.startswith("-"):
                continue
            if line.startswith("No.") or line.startswith("TOTAL"):
                continue
            if line.startswith("Model:") or line.startswith("Total trainable"):
                continue
            if line.startswith("PARAMETER"):
                continue
            m = re.match(r"^\d+\s+(\S+)\s+\(([^)]+)\)\s+", line)
            if m:
                name = m.group(1)
                shape_str = m.group(2)
                dims = tuple(
                    int(d.strip().replace(",", ""))
                    for d in shape_str.split(",")
                    if d.strip()
                )
                shapes.append((name, dims))
    return shapes


# ---------------------------------------------------------------------------
# Timing infrastructure
# ---------------------------------------------------------------------------

def _timed_run(fn, n_warmup, n_repeat):
    for i in range(n_warmup):
        fn(i)
    torch.cuda.synchronize()

    gpu_times = []
    wall_times = []

    for i in range(n_repeat):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        start_event.record()

        fn(n_warmup + i)

        end_event.record()
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        gpu_times.append(start_event.elapsed_time(end_event))
        wall_times.append((t1 - t0) * 1000)

    return {"gpu_ms": gpu_times, "wall_ms": wall_times}


def _free_gpu():
    """Force garbage collection and free CUDA cache."""
    gc.collect()
    torch.cuda.empty_cache()


def _safe_bench(bench_fn, label, *args, **kwargs):
    """Run a benchmark function with OOM protection."""
    try:
        result = bench_fn(*args, **kwargs)
        return result
    except torch.cuda.OutOfMemoryError:
        print(f"    [SKIP] OOM during {label}")
        _free_gpu()
        return None


# ===================================================================
# HiZOO benchmark functions
# ===================================================================

def bench_hizoo_perturb_per_tensor(shapes, device, seed, n_warmup, n_repeat, dtype=torch.float32):
    """Per-tensor HiZOO perturb: loop with randn + rsqrt(h) scaling."""
    params = [torch.randn(s, device=device, dtype=dtype) for _, s in shapes]
    hessians = [torch.ones_like(p) + 0.1 * torch.rand_like(p) for p in params]
    torch.cuda.synchronize()

    def fn(rep):
        torch.manual_seed(seed + rep)
        for p, h in zip(params, hessians):
            z = torch.randn_like(p)
            p.add_(1e-3 * z * torch.rsqrt(h))

    return _timed_run(fn, n_warmup, n_repeat)


def bench_hizoo_perturb_flat_pytorch(shapes, device, seed, n_warmup, n_repeat, dtype=torch.float32):
    """Flat buffer HiZOO perturb: PyTorch ops on concatenated tensor."""
    total_n = sum(int(np.prod(s)) for _, s in shapes)
    buf = torch.randn(total_n, device=device, dtype=dtype)
    h_buf = torch.ones(total_n, device=device, dtype=dtype) + 0.1
    z = torch.empty_like(buf)
    torch.cuda.synchronize()

    def fn(rep):
        torch.manual_seed(seed + rep)
        z.normal_()
        # in-place to minimize temporaries
        tmp = torch.rsqrt(h_buf)
        tmp.mul_(z).mul_(1e-3)
        buf.add_(tmp)

    return _timed_run(fn, n_warmup, n_repeat)


def bench_hizoo_perturb_flat_fused(shapes, device, seed, n_warmup, n_repeat, dtype=torch.float32):
    """Flat buffer HiZOO perturb: Triton fused kernel (1 launch)."""
    if not HAS_HIZOO_TRITON:
        return None
    total_n = sum(int(np.prod(s)) for _, s in shapes)
    buf = torch.randn(total_n, device=device, dtype=dtype)
    h_buf = torch.ones(total_n, device=device, dtype=dtype) + 0.1
    torch.cuda.synchronize()

    def fn(rep):
        hizoo_fused_perturb(buf, h_buf, seed + rep, 1e-3, 1.0)

    return _timed_run(fn, n_warmup, n_repeat)


def bench_hizoo_perturb_per_tensor_fused(shapes, device, seed, n_warmup, n_repeat, dtype=torch.float32):
    """Per-tensor HiZOO perturb: Triton fused kernel called per parameter."""
    if not HAS_HIZOO_TRITON:
        return None
    params = [torch.randn(s, device=device, dtype=dtype) for _, s in shapes]
    hessians = [torch.ones_like(p) + 0.1 * torch.rand_like(p) for p in params]
    # Cumulative offsets for deterministic Philox RNG
    cum_offsets = []
    off = 0
    for p in params:
        cum_offsets.append(off)
        off += p.numel()
    torch.cuda.synchronize()

    def fn(rep):
        for p, h, base_off in zip(params, hessians, cum_offsets):
            hizoo_fused_perturb(p.view(-1), h.view(-1), seed + rep, 1e-3, 1.0,
                                base_offset=base_off)

    return _timed_run(fn, n_warmup, n_repeat)


def bench_hizoo_update_per_tensor(shapes, device, seed, n_warmup, n_repeat, dtype=torch.float32):
    """Per-tensor HiZOO update-restore: complex loop per param."""
    params = [torch.randn(s, device=device, dtype=dtype) for _, s in shapes]
    hessians = [torch.ones_like(p) + 0.1 * torch.rand_like(p) for p in params]
    eps = 1e-3
    lr = 1e-5
    wd = 0.01
    smooth = 0.1
    l1, l2, l0 = 1.5, 1.3, 1.4
    torch.cuda.synchronize()

    def fn(rep):
        torch.manual_seed(seed + rep)
        for p, h in zip(params, hessians):
            z = torch.randn_like(p)
            theta = p + eps * z * torch.rsqrt(h)
            curv = abs(l1 + l2 - 2 * l0) * smooth / (2 * eps * eps)
            h_new = (1 - smooth) * h + curv * h * z * z
            grad_scale = (l1 - l2) / (2 * eps)
            grad = grad_scale * z * torch.rsqrt(h_new)
            p.data.copy_(theta - lr * (grad + wd * theta))
            h.data.copy_(h_new)

    return _timed_run(fn, n_warmup, n_repeat)


def bench_hizoo_update_flat_pytorch(shapes, device, seed, n_warmup, n_repeat, dtype=torch.float32):
    """Flat buffer HiZOO update-restore: PyTorch ops on flat tensor."""
    total_n = sum(int(np.prod(s)) for _, s in shapes)
    buf = torch.randn(total_n, device=device, dtype=dtype)
    h_buf = torch.ones(total_n, device=device, dtype=dtype) + 0.1
    z = torch.empty_like(buf)
    tmp = torch.empty_like(buf)
    eps = 1e-3
    lr = 1e-5
    wd = 0.01
    smooth = 0.1
    l1, l2, l0 = 1.5, 1.3, 1.4
    torch.cuda.synchronize()

    def fn(rep):
        torch.manual_seed(seed + rep)
        z.normal_()
        # theta = buf + eps * z * rsqrt(h)
        torch.rsqrt(h_buf, out=tmp)
        tmp.mul_(z).mul_(eps)
        theta = buf + tmp  # need separate copy for final formula
        # h_new = (1-s)*h + curv*h*z*z
        curv = abs(l1 + l2 - 2 * l0) * smooth / (2 * eps * eps)
        torch.mul(z, z, out=tmp)
        tmp.mul_(h_buf).mul_(curv)
        h_new = (1 - smooth) * h_buf + tmp
        # grad = scale * z * rsqrt(h_new)
        grad_scale = (l1 - l2) / (2 * eps)
        torch.rsqrt(h_new, out=tmp)
        tmp.mul_(z).mul_(grad_scale)
        # p = theta - lr*(grad + wd*theta)
        buf.copy_(theta - lr * (tmp + wd * theta))
        h_buf.copy_(h_new)

    return _timed_run(fn, n_warmup, n_repeat)


def bench_hizoo_update_flat_fused(shapes, device, seed, n_warmup, n_repeat, dtype=torch.float32):
    """Flat buffer HiZOO update: Triton fused kernel (1 launch)."""
    if not HAS_HIZOO_TRITON:
        return None
    total_n = sum(int(np.prod(s)) for _, s in shapes)
    buf = torch.randn(total_n, device=device, dtype=dtype)
    h_buf = torch.ones(total_n, device=device, dtype=dtype) + 0.1
    loss1 = torch.tensor([1.5], device=device, dtype=dtype)
    loss2 = torch.tensor([1.3], device=device, dtype=dtype)
    loss0 = torch.tensor([1.4], device=device, dtype=dtype)
    torch.cuda.synchronize()

    def fn(rep):
        hizoo_fused_update(buf, h_buf, loss1, loss2, loss0,
                           seed + rep, 1e-3, 1e-5, 0.01, 0.1)

    return _timed_run(fn, n_warmup, n_repeat)


def bench_hizoo_update_per_tensor_fused(shapes, device, seed, n_warmup, n_repeat, dtype=torch.float32):
    """Per-tensor HiZOO update: Triton fused kernel called per parameter."""
    if not HAS_HIZOO_TRITON:
        return None
    params = [torch.randn(s, device=device, dtype=dtype) for _, s in shapes]
    hessians = [torch.ones_like(p) + 0.1 * torch.rand_like(p) for p in params]
    loss1 = torch.tensor([1.5], device=device, dtype=dtype)
    loss2 = torch.tensor([1.3], device=device, dtype=dtype)
    loss0 = torch.tensor([1.4], device=device, dtype=dtype)
    cum_offsets = []
    off = 0
    for p in params:
        cum_offsets.append(off)
        off += p.numel()
    torch.cuda.synchronize()

    def fn(rep):
        for p, h, base_off in zip(params, hessians, cum_offsets):
            hizoo_fused_update(p.view(-1), h.view(-1), loss1, loss2, loss0,
                               seed + rep, 1e-3, 1e-5, 0.01, 0.1,
                               base_offset=base_off)

    return _timed_run(fn, n_warmup, n_repeat)


# ===================================================================
# Per-tensor DiZO Triton kernels (for approach D benchmarking)
# ===================================================================

if HAS_TRITON_DIRECT:
    @triton.jit
    def _dizo_norm_per_tensor_kernel(
        param_ptr, anchor_ptr, out_ptr, n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Multi-block norm: atomic_add partial sum-of-squared-diffs."""
        n_elements_i64 = n_elements.to(tl.int64)
        pid = tl.program_id(0).to(tl.int64)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
        mask = offsets < n_elements_i64
        p = tl.load(param_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        a = tl.load(anchor_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        diff = p - a
        partial = tl.sum(diff * diff)
        tl.atomic_add(out_ptr, partial)

    @triton.jit
    def _dizo_apply_per_tensor_kernel(
        param_ptr, anchor_ptr, alpha_ptr, n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Elementwise: p = a + (p - a) * alpha.  Pass 1/alpha for reverse."""
        n_elements_i64 = n_elements.to(tl.int64)
        pid = tl.program_id(0).to(tl.int64)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
        mask = offsets < n_elements_i64
        p = tl.load(param_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        a = tl.load(anchor_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        alpha = tl.load(alpha_ptr).to(tl.float32)
        result = a + (p - a) * alpha
        tl.store(param_ptr + offsets, result, mask=mask)


# ===================================================================
# DiZO benchmark functions
# ===================================================================

def bench_dizo_constraints_per_tensor(shapes, device, seed, n_warmup, n_repeat, dtype=torch.float32):
    """Per-tensor DiZO constraints: norms + apply + reverse."""
    params = [torch.randn(s, device=device, dtype=dtype) for _, s in shapes]
    anchors = [torch.randn_like(p) for p in params]
    gammas = [torch.tensor([0.1], device=device, dtype=dtype) for _ in params]
    torch.cuda.synchronize()

    def fn(rep):
        # Apply constraints
        norms = [torch.norm(p - a) for p, a in zip(params, anchors)]
        for p, a, g, n in zip(params, anchors, gammas, norms):
            alpha = g / (n + 1e-8)
            p.data.copy_(a + (p - a) * alpha)
        # Reverse constraints
        for p, a, g, n in zip(params, anchors, gammas, norms):
            alpha = g / (n + 1e-8)
            p.data.copy_(a + (p - a) / alpha)

    return _timed_run(fn, n_warmup, n_repeat)


def bench_dizo_constraints_flat_pytorch(shapes, device, seed, n_warmup, n_repeat, dtype=torch.float32):
    """Flat buffer DiZO constraints: PyTorch ops on flat tensor."""
    sizes_list = [int(np.prod(s)) for _, s in shapes]
    total_n = sum(sizes_list)
    param_flat = torch.randn(total_n, device=device, dtype=dtype)
    anchor_flat = torch.randn(total_n, device=device, dtype=dtype)
    gammas = torch.full((len(shapes),), 0.1, device=device, dtype=dtype)
    torch.cuda.synchronize()

    offsets_list = []
    off = 0
    for sz in sizes_list:
        offsets_list.append(off)
        off += sz

    def fn(rep):
        # Apply constraints (per-param-group on flat buffer)
        norms = torch.zeros(len(sizes_list), device=device)
        for i, (o, sz) in enumerate(zip(offsets_list, sizes_list)):
            diff = param_flat[o:o+sz] - anchor_flat[o:o+sz]
            norms[i] = torch.norm(diff)
        for i, (o, sz) in enumerate(zip(offsets_list, sizes_list)):
            alpha = gammas[i] / (norms[i] + 1e-8)
            diff = param_flat[o:o+sz] - anchor_flat[o:o+sz]
            param_flat[o:o+sz] = anchor_flat[o:o+sz] + diff * alpha
        # Reverse constraints
        for i, (o, sz) in enumerate(zip(offsets_list, sizes_list)):
            alpha = gammas[i] / (norms[i] + 1e-8)
            diff = param_flat[o:o+sz] - anchor_flat[o:o+sz]
            param_flat[o:o+sz] = anchor_flat[o:o+sz] + diff / alpha

    return _timed_run(fn, n_warmup, n_repeat)


def bench_dizo_constraints_flat_fused(shapes, device, seed, n_warmup, n_repeat, dtype=torch.float32):
    """Flat buffer DiZO constraints: Triton V2 fused (few launches)."""
    if not HAS_DIZO_TRITON_V2:
        return None
    sizes_list = [int(np.prod(s)) for _, s in shapes]
    total_n = sum(sizes_list)
    param_flat = torch.randn(total_n, device=device, dtype=dtype)
    anchor_flat = torch.randn(total_n, device=device, dtype=dtype)
    num_params = len(shapes)
    constraints = torch.full((num_params,), 0.1, device=device, dtype=dtype)

    offsets_list = []
    off = 0
    for sz in sizes_list:
        offsets_list.append(off)
        off += sz
    offsets = torch.tensor(offsets_list, device=device, dtype=torch.long)
    sizes = torch.tensor(sizes_list, device=device, dtype=torch.long)

    zo_kernels = FusedDiZOKernelsV2(num_params, total_n, device, offsets, sizes)
    torch.cuda.synchronize()

    def fn(rep):
        norms = zo_kernels.compute_norms(param_flat, anchor_flat, offsets, sizes)
        alphas = zo_kernels.apply_constraints(
            param_flat, anchor_flat, offsets, sizes, constraints, norms)
        zo_kernels.reverse_constraints(
            param_flat, anchor_flat, offsets, sizes, alphas)

    return _timed_run(fn, n_warmup, n_repeat)


def bench_dizo_constraints_per_tensor_fused(shapes, device, seed, n_warmup, n_repeat, dtype=torch.float32):
    """Per-tensor DiZO constraints: Triton fused kernel per parameter."""
    if not HAS_TRITON_DIRECT:
        return None
    params = [torch.randn(s, device=device, dtype=dtype).contiguous().view(-1) for _, s in shapes]
    anchors = [torch.randn_like(p) for p in params]
    gammas_t = [torch.tensor([0.1], device=device, dtype=torch.float32) for _ in shapes]
    norm_out = torch.zeros(1, device=device, dtype=torch.float32)
    alpha_t = torch.empty(1, device=device, dtype=torch.float32)
    BLOCK_SIZE = 1024
    torch.cuda.synchronize()

    def fn(rep):
        # Compute per-tensor norms
        norms = []
        for p, a in zip(params, anchors):
            norm_out.zero_()
            n = p.numel()
            grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
            _dizo_norm_per_tensor_kernel[grid](p, a, norm_out, n, BLOCK_SIZE=BLOCK_SIZE)
            norms.append(torch.sqrt(norm_out.clone()))
        # Apply constraints: p = a + (p-a) * alpha, alpha = gamma/(norm+eps)
        for p, a, g, norm_val in zip(params, anchors, gammas_t, norms):
            torch.div(g, norm_val + 1e-8, out=alpha_t)
            n = p.numel()
            grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
            _dizo_apply_per_tensor_kernel[grid](p, a, alpha_t, n, BLOCK_SIZE=BLOCK_SIZE)
        # Reverse: p = a + (p-a) * (1/alpha) = a + (p-a) * (norm+eps)/gamma
        for p, a, g, norm_val in zip(params, anchors, gammas_t, norms):
            torch.div(norm_val + 1e-8, g, out=alpha_t)
            n = p.numel()
            grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
            _dizo_apply_per_tensor_kernel[grid](p, a, alpha_t, n, BLOCK_SIZE=BLOCK_SIZE)

    return _timed_run(fn, n_warmup, n_repeat)


# ===================================================================
# ZO2 benchmark functions
# ===================================================================

def bench_zo2_dual_perturb_per_tensor(shapes, device, seed, n_warmup, n_repeat, dtype=torch.float32):
    """Per-tensor ZO2 dual perturb: loop generating z, writing +/- copies."""
    params = [torch.randn(s, device=device, dtype=dtype) for _, s in shapes]
    plus = [torch.empty_like(p) for p in params]
    minus = [torch.empty_like(p) for p in params]
    eps = 1e-3
    torch.cuda.synchronize()

    def fn(rep):
        torch.manual_seed(seed + rep)
        for p, pp, pm in zip(params, plus, minus):
            z = torch.randn_like(p)
            eps_z = eps * z
            pp.copy_(p + eps_z)
            pm.copy_(p - eps_z)

    return _timed_run(fn, n_warmup, n_repeat)


def bench_zo2_dual_perturb_flat_pytorch(shapes, device, seed, n_warmup, n_repeat, dtype=torch.float32):
    """Flat buffer ZO2 dual perturb: PyTorch on flat tensor."""
    total_n = sum(int(np.prod(s)) for _, s in shapes)
    buf = torch.randn(total_n, device=device, dtype=dtype)
    plus = torch.empty_like(buf)
    minus = torch.empty_like(buf)
    eps = 1e-3
    torch.cuda.synchronize()

    def fn(rep):
        torch.manual_seed(seed + rep)
        z = torch.randn_like(buf)
        eps_z = eps * z
        torch.add(buf, eps_z, out=plus)
        torch.sub(buf, eps_z, out=minus)

    return _timed_run(fn, n_warmup, n_repeat)


def bench_zo2_dual_perturb_flat_fused(shapes, device, seed, n_warmup, n_repeat, dtype=torch.float32):
    """Flat buffer ZO2 dual perturb: Triton fused (1 read + 2 writes)."""
    if not HAS_ZO2_TRITON:
        return None
    total_n = sum(int(np.prod(s)) for _, s in shapes)
    buf = torch.randn(total_n, device=device, dtype=dtype)
    plus = torch.empty_like(buf)
    minus = torch.empty_like(buf)
    eps = 1e-3
    torch.cuda.synchronize()

    def fn(rep):
        zo2_fused_dual_perturb(buf, plus, minus, seed + rep, eps)

    return _timed_run(fn, n_warmup, n_repeat)


def bench_zo2_dual_perturb_per_tensor_fused(shapes, device, seed, n_warmup, n_repeat, dtype=torch.float32):
    """Per-tensor ZO2 dual perturb: Triton fused kernel per parameter."""
    if not HAS_ZO2_TRITON:
        return None
    params = [torch.randn(s, device=device, dtype=dtype).contiguous().view(-1) for _, s in shapes]
    plus = [torch.empty_like(p) for p in params]
    minus = [torch.empty_like(p) for p in params]
    eps = 1e-3
    torch.cuda.synchronize()

    def fn(rep):
        for i, (p, pl, mi) in enumerate(zip(params, plus, minus)):
            zo2_fused_dual_perturb(p, pl, mi, seed + rep + i, eps)

    return _timed_run(fn, n_warmup, n_repeat)


# ===================================================================
# ZO2 Dual-forward benchmark functions (update + dual perturb combined)
# ===================================================================

def bench_zo2_dual_forward_per_tensor(shapes, device, seed, n_warmup, n_repeat, dtype=torch.float32):
    """Per-tensor ZO2 dual-forward: PyTorch update + randn + dual perturb loop."""
    params = [torch.randn(s, device=device, dtype=dtype) for _, s in shapes]
    plus = [torch.empty_like(p) for p in params]
    minus = [torch.empty_like(p) for p in params]
    eps = 1e-3
    lr = 1e-5
    weight_decay = 0.01
    grad_val = 0.5     # simulated projected gradient
    torch.cuda.synchronize()

    def fn(rep):
        torch.manual_seed(seed + rep)
        for p, pp, pm in zip(params, plus, minus):
            # Update: p -= lr * (grad * z_prev + wd * p)
            # Use same seed as "previous step" for z_prev
            z_prev = torch.randn_like(p)
            p.data.sub_(lr * (grad_val * z_prev + weight_decay * p))
            # Dual perturb: generate new z
            z_new = torch.randn_like(p)
            eps_z = eps * z_new
            pp.copy_(p + eps_z)
            pm.copy_(p - eps_z)

    return _timed_run(fn, n_warmup, n_repeat)


def bench_zo2_dual_forward_per_tensor_fused(shapes, device, seed, n_warmup, n_repeat, dtype=torch.float32):
    """Per-tensor ZO2 dual-forward: Triton combined update+dual_perturb per param."""
    if not HAS_ZO2_TRITON:
        return None
    params = [torch.randn(s, device=device, dtype=dtype).contiguous().view(-1) for _, s in shapes]
    plus = [torch.empty_like(p) for p in params]
    minus = [torch.empty_like(p) for p in params]
    eps = 1e-3
    lr = 1e-5
    weight_decay = 0.01
    grad_val = 0.5
    torch.cuda.synchronize()

    def fn(rep):
        for i, (p, pl, mi) in enumerate(zip(params, plus, minus)):
            zo2_fused_dual_perturb_and_update(
                p, pl, mi,
                grad=grad_val, lr=lr, weight_decay=weight_decay,
                seed=seed + rep + i, last_seed=seed + rep + i - 1,
                eps=eps, do_update=True,
            )

    return _timed_run(fn, n_warmup, n_repeat)


def bench_zo2_dual_forward_flat_pytorch(shapes, device, seed, n_warmup, n_repeat, dtype=torch.float32):
    """Flat buffer ZO2 dual-forward: PyTorch update + dual perturb on flat."""
    total_n = sum(int(np.prod(s)) for _, s in shapes)
    buf = torch.randn(total_n, device=device, dtype=dtype)
    plus = torch.empty_like(buf)
    minus = torch.empty_like(buf)
    z_prev = torch.empty_like(buf)
    z_new = torch.empty_like(buf)
    tmp = torch.empty_like(buf)
    eps = 1e-3
    lr = 1e-5
    weight_decay = 0.01
    grad_val = 0.5
    torch.cuda.synchronize()

    def fn(rep):
        # Update: buf -= lr * (grad * z_prev + wd * buf)
        torch.manual_seed(seed + rep - 1)
        z_prev.normal_()
        torch.mul(z_prev, grad_val, out=tmp)
        tmp.add_(buf, alpha=weight_decay)
        buf.sub_(tmp, alpha=lr)
        # Dual perturb: generate new z, write plus/minus
        torch.manual_seed(seed + rep)
        z_new.normal_()
        torch.mul(z_new, eps, out=tmp)
        torch.add(buf, tmp, out=plus)
        torch.sub(buf, tmp, out=minus)

    return _timed_run(fn, n_warmup, n_repeat)


def bench_zo2_dual_forward_flat_fused(shapes, device, seed, n_warmup, n_repeat, dtype=torch.float32):
    """Flat buffer ZO2 dual-forward: Triton combined update+dual_perturb (1 launch)."""
    if not HAS_ZO2_TRITON:
        return None
    total_n = sum(int(np.prod(s)) for _, s in shapes)
    buf = torch.randn(total_n, device=device, dtype=dtype)
    plus = torch.empty_like(buf)
    minus = torch.empty_like(buf)
    eps = 1e-3
    lr = 1e-5
    weight_decay = 0.01
    grad_val = 0.5
    torch.cuda.synchronize()

    def fn(rep):
        zo2_fused_dual_perturb_and_update(
            buf, plus, minus,
            grad=grad_val, lr=lr, weight_decay=weight_decay,
            seed=seed + rep, last_seed=seed + rep - 1,
            eps=eps, do_update=True,
        )

    return _timed_run(fn, n_warmup, n_repeat)


# ===================================================================
# ZO2 CUDA benchmark functions
# ===================================================================

def bench_zo2_dual_perturb_flat_cuda(shapes, device, seed, n_warmup, n_repeat, dtype=torch.float32):
    """Flat buffer ZO2 dual perturb: CUDA kernel (1 launch)."""
    if not HAS_ZO2_CUDA:
        return None
    total_n = sum(int(np.prod(s)) for _, s in shapes)
    buf = torch.randn(total_n, device=device, dtype=dtype)
    plus = torch.empty_like(buf)
    minus = torch.empty_like(buf)
    eps = 1e-3
    torch.cuda.synchronize()

    def fn(rep):
        _zo2_cuda.fused_dual_perturb(buf, plus, minus, seed + rep, eps)

    return _timed_run(fn, n_warmup, n_repeat)


def bench_zo2_dual_perturb_per_tensor_cuda(shapes, device, seed, n_warmup, n_repeat, dtype=torch.float32):
    """Per-tensor ZO2 dual perturb: CUDA kernel per parameter."""
    if not HAS_ZO2_CUDA:
        return None
    params = [torch.randn(s, device=device, dtype=dtype).contiguous().view(-1) for _, s in shapes]
    plus = [torch.empty_like(p) for p in params]
    minus = [torch.empty_like(p) for p in params]
    eps = 1e-3
    torch.cuda.synchronize()

    def fn(rep):
        for i, (p, pl, mi) in enumerate(zip(params, plus, minus)):
            _zo2_cuda.fused_dual_perturb(p, pl, mi, seed + rep + i, eps)

    return _timed_run(fn, n_warmup, n_repeat)


def bench_zo2_dual_forward_flat_cuda(shapes, device, seed, n_warmup, n_repeat, dtype=torch.float32):
    """Flat buffer ZO2 dual-forward: CUDA combined update+dual_perturb (1 launch)."""
    if not HAS_ZO2_CUDA:
        return None
    total_n = sum(int(np.prod(s)) for _, s in shapes)
    buf = torch.randn(total_n, device=device, dtype=dtype)
    plus = torch.empty_like(buf)
    minus = torch.empty_like(buf)
    eps = 1e-3
    lr = 1e-5
    weight_decay = 0.01
    grad_val = 0.5
    torch.cuda.synchronize()

    def fn(rep):
        _zo2_cuda.fused_update_and_dual_perturb(
            buf, plus, minus, seed + rep, seed + rep - 1,
            grad_val, lr, weight_decay, eps, True)

    return _timed_run(fn, n_warmup, n_repeat)


def bench_zo2_dual_forward_per_tensor_cuda(shapes, device, seed, n_warmup, n_repeat, dtype=torch.float32):
    """Per-tensor ZO2 dual-forward: CUDA combined update+dual_perturb per param."""
    if not HAS_ZO2_CUDA:
        return None
    params = [torch.randn(s, device=device, dtype=dtype).contiguous().view(-1) for _, s in shapes]
    plus = [torch.empty_like(p) for p in params]
    minus = [torch.empty_like(p) for p in params]
    eps = 1e-3
    lr = 1e-5
    weight_decay = 0.01
    grad_val = 0.5
    torch.cuda.synchronize()

    def fn(rep):
        for i, (p, pl, mi) in enumerate(zip(params, plus, minus)):
            _zo2_cuda.fused_update_and_dual_perturb(
                p, pl, mi, seed + rep + i, seed + rep + i - 1,
                grad_val, lr, weight_decay, eps, True)

    return _timed_run(fn, n_warmup, n_repeat)


# ===================================================================
# KerZOO helper
# ===================================================================

def _kernel_function(r, t=1.0):
    """KerZOO kernel function: K(r, t) = 15r(5 - 7(r/t)^2)."""
    return (15 * r) * (5 - 7 * (r / t) ** 2)


# ===================================================================
# KerZOO benchmark functions — Perturb (judge > 0)
# ===================================================================

def bench_kerzoo_perturb_per_tensor(shapes, device, seed, n_warmup, n_repeat, dtype=torch.float32):
    """Per-tensor KerZOO perturb: interpolation + randn + scaled add per param."""
    params = [torch.randn(s, device=device, dtype=dtype) for _, s in shapes]
    c_params = [torch.randn_like(p) for p in params]
    eps = 1e-3
    beta_k = 100.0   # simulate mid-training
    decay = 0.5      # max(1 - step/4000, 0.0001) at some mid point
    inv_beta = 1.0 / beta_k
    comp_beta = 1.0 - inv_beta
    torch.cuda.synchronize()

    def fn(rep):
        torch.manual_seed(seed + rep)
        for p, c in zip(params, c_params):
            z = torch.randn_like(p)
            k = (2 * torch.rand(1, device=device).item() - 1) * decay
            p.data.copy_(c * inv_beta + comp_beta * p + eps * k * z)

    return _timed_run(fn, n_warmup, n_repeat)


def bench_kerzoo_perturb_per_tensor_fused(shapes, device, seed, n_warmup, n_repeat, dtype=torch.float32):
    """Per-tensor KerZOO perturb: Triton fused kernel per parameter."""
    if not HAS_KERZOO_TRITON:
        return None
    params = [torch.randn(s, device=device, dtype=dtype).contiguous().view(-1) for _, s in shapes]
    c_params = [torch.randn_like(p) for p in params]
    eps = 1e-3
    beta_k = 100.0
    decay = 0.5
    inv_beta = 1.0 / beta_k
    comp_beta = 1.0 - inv_beta
    cum_offsets = []
    off = 0
    for p in params:
        cum_offsets.append(off)
        off += p.numel()
    torch.cuda.synchronize()

    def fn(rep):
        rng = np.random.RandomState(seed + rep)
        k_vals = (2 * rng.rand(len(params)) - 1) * decay
        for p, c, base_off, k in zip(params, c_params, cum_offsets, k_vals):
            alpha = eps * float(k)
            kerzoo_fused_perturb(p, c, seed + rep, alpha, inv_beta, comp_beta,
                                 base_offset=base_off)

    return _timed_run(fn, n_warmup, n_repeat)


def bench_kerzoo_perturb_flat_pytorch(shapes, device, seed, n_warmup, n_repeat, dtype=torch.float32):
    """Flat buffer KerZOO perturb: pre-expand k, then flat PyTorch ops."""
    sizes_list = [int(np.prod(s)) for _, s in shapes]
    total_n = sum(sizes_list)
    param_flat = torch.randn(total_n, device=device, dtype=dtype)
    c_flat = torch.randn(total_n, device=device, dtype=dtype)
    alpha_buf = torch.empty(total_n, device=device, dtype=dtype)
    z = torch.empty(total_n, device=device, dtype=dtype)
    tmp = torch.empty(total_n, device=device, dtype=dtype)
    eps = 1e-3
    beta_k = 100.0
    decay = 0.5
    inv_beta = 1.0 / beta_k
    comp_beta = 1.0 - inv_beta
    offsets_list = []
    off = 0
    for sz in sizes_list:
        offsets_list.append(off)
        off += sz
    torch.cuda.synchronize()

    def fn(rep):
        rng = np.random.RandomState(seed + rep)
        k_vals = (2 * rng.rand(len(sizes_list)) - 1) * decay
        for i, (o, sz) in enumerate(zip(offsets_list, sizes_list)):
            alpha_buf[o:o+sz] = eps * k_vals[i]
        torch.manual_seed(seed + rep)
        z.normal_()
        # param = c * inv_beta + comp_beta * param + alpha_buf * z
        torch.mul(c_flat, inv_beta, out=tmp)
        tmp.add_(param_flat, alpha=comp_beta)
        tmp.addcmul_(alpha_buf, z)
        param_flat.copy_(tmp)

    return _timed_run(fn, n_warmup, n_repeat)


def bench_kerzoo_perturb_flat_fused(shapes, device, seed, n_warmup, n_repeat, dtype=torch.float32):
    """Flat buffer KerZOO perturb: Triton fused (1 kernel launch)."""
    if not HAS_KERZOO_TRITON:
        return None
    sizes_list = [int(np.prod(s)) for _, s in shapes]
    total_n = sum(sizes_list)
    param_flat = torch.randn(total_n, device=device, dtype=dtype)
    c_flat = torch.randn(total_n, device=device, dtype=dtype)
    alpha_buf = torch.empty(total_n, device=device, dtype=dtype)
    eps = 1e-3
    beta_k = 100.0
    decay = 0.5
    inv_beta = 1.0 / beta_k
    comp_beta = 1.0 - inv_beta
    offsets_list = []
    off = 0
    for sz in sizes_list:
        offsets_list.append(off)
        off += sz
    torch.cuda.synchronize()

    def fn(rep):
        rng = np.random.RandomState(seed + rep)
        k_vals = (2 * rng.rand(len(sizes_list)) - 1) * decay
        for i, (o, sz) in enumerate(zip(offsets_list, sizes_list)):
            alpha_buf[o:o+sz] = eps * k_vals[i]
        kerzoo_fused_perturb_flat(param_flat, c_flat, alpha_buf, seed + rep,
                                  inv_beta, comp_beta)

    return _timed_run(fn, n_warmup, n_repeat)


# ===================================================================
# KerZOO benchmark functions — Update (grad accum + param update)
# ===================================================================

def bench_kerzoo_update_per_tensor(shapes, device, seed, n_warmup, n_repeat, dtype=torch.float32):
    """Per-tensor KerZOO update: 3-sample kernel-weighted accum + clip + interpolate."""
    params = [torch.randn(s, device=device, dtype=dtype) for _, s in shapes]
    c_params = [torch.randn_like(p) for p in params]
    grad_bufs = [torch.zeros_like(p) for p in params]
    projected_grads = [0.5, 0.3, 0.4]
    lr = 1e-5
    beta_k = 100.0
    decay = 0.5
    clip_thresh = 400000.0
    inv_beta = 1.0 / beta_k
    comp_beta = 1.0 - inv_beta
    torch.cuda.synchronize()

    def fn(rep):
        for g in grad_bufs:
            g.zero_()
        # Gradient accumulation (3 samples x N params)
        for si in range(3):
            torch.manual_seed(seed + rep - si)
            for j, p in enumerate(params):
                z = torch.randn_like(p)
                k = (2 * torch.rand(1, device=device).item() - 1) * decay
                kernel_val = _kernel_function(k)
                grad_bufs[j].add_(z, alpha=projected_grads[si] * kernel_val)
        # Parameter update (N params)
        for p, c, g in zip(params, c_params, grad_bufs):
            avg_grad = g / 3
            norm_val = torch.norm(avg_grad, p=2)
            clip = torch.clamp(clip_thresh / (norm_val + 1e-8), max=1.0)
            avg_grad.mul_(clip)
            c.data.sub_(avg_grad, alpha=lr)
            p.data.copy_(c * inv_beta + comp_beta * p)

    return _timed_run(fn, n_warmup, n_repeat)


def bench_kerzoo_update_per_tensor_fused(shapes, device, seed, n_warmup, n_repeat, dtype=torch.float32):
    """Per-tensor KerZOO update: Triton fused accum + update per parameter."""
    if not HAS_KERZOO_TRITON:
        return None
    params = [torch.randn(s, device=device, dtype=dtype).contiguous().view(-1) for _, s in shapes]
    c_params = [torch.randn_like(p) for p in params]
    grad_bufs = [torch.zeros_like(p) for p in params]
    projected_grads = [0.5, 0.3, 0.4]
    lr = 1e-5
    beta_k = 100.0
    decay = 0.5
    clip_thresh = 400000.0
    inv_beta = 1.0 / beta_k
    comp_beta = 1.0 - inv_beta
    cum_offsets = []
    off = 0
    for p in params:
        cum_offsets.append(off)
        off += p.numel()
    torch.cuda.synchronize()

    def fn(rep):
        for g in grad_bufs:
            g.zero_()
        # Fused gradient accumulation (3 samples x N Triton launches)
        for si in range(3):
            rng = np.random.RandomState(seed + rep - si)
            k_vals = (2 * rng.rand(len(params)) - 1) * decay
            for j, (g, base_off) in enumerate(zip(grad_bufs, cum_offsets)):
                kernel_val = _kernel_function(k_vals[j])
                weight = float(projected_grads[si] * kernel_val)
                kerzoo_fused_accum(g, seed + rep - si, weight, base_offset=base_off)
        # Compute norms (batch), then fused update
        norms = [torch.norm(g / 3, p=2) for g in grad_bufs]
        torch.cuda.synchronize()
        clips = [min(1.0, clip_thresh / (n.item() + 1e-8)) for n in norms]
        for p, c, g, clip_val in zip(params, c_params, grad_bufs, clips):
            lr_clip_div3 = lr * clip_val / 3
            kerzoo_fused_update_param(p, c, g, lr_clip_div3, inv_beta, comp_beta)

    return _timed_run(fn, n_warmup, n_repeat)


def bench_kerzoo_update_flat_pytorch(shapes, device, seed, n_warmup, n_repeat, dtype=torch.float32):
    """Flat buffer KerZOO update: pre-expand weights + flat PyTorch ops."""
    sizes_list = [int(np.prod(s)) for _, s in shapes]
    total_n = sum(sizes_list)
    param_flat = torch.randn(total_n, device=device, dtype=dtype)
    c_flat = torch.randn(total_n, device=device, dtype=dtype)
    grad_flat = torch.zeros(total_n, device=device, dtype=dtype)
    weight_buf = torch.empty(total_n, device=device, dtype=dtype)
    z = torch.empty(total_n, device=device, dtype=dtype)
    tmp = torch.empty(total_n, device=device, dtype=dtype)
    projected_grads = [0.5, 0.3, 0.4]
    lr = 1e-5
    beta_k = 100.0
    decay = 0.5
    clip_thresh = 400000.0
    inv_beta = 1.0 / beta_k
    comp_beta = 1.0 - inv_beta
    offsets_list = []
    off = 0
    for sz in sizes_list:
        offsets_list.append(off)
        off += sz
    torch.cuda.synchronize()

    def fn(rep):
        grad_flat.zero_()
        for si in range(3):
            rng = np.random.RandomState(seed + rep - si)
            k_vals = (2 * rng.rand(len(sizes_list)) - 1) * decay
            for i, (o, sz) in enumerate(zip(offsets_list, sizes_list)):
                kernel_val = _kernel_function(k_vals[i])
                weight_buf[o:o+sz] = projected_grads[si] * kernel_val
            torch.manual_seed(seed + rep - si)
            z.normal_()
            grad_flat.addcmul_(weight_buf, z)
        # Clip (global norm)
        torch.div(grad_flat, 3.0, out=tmp)
        norm_val = torch.norm(tmp, p=2)
        clip = min(1.0, clip_thresh / (norm_val.item() + 1e-8))
        lr_clip_div3 = lr * clip / 3
        # c -= lr_clip_div3 * grad; p = c*inv_beta + comp_beta*p
        torch.mul(grad_flat, lr_clip_div3, out=tmp)
        c_flat.sub_(tmp)
        torch.mul(c_flat, inv_beta, out=tmp)
        tmp.add_(param_flat, alpha=comp_beta)
        param_flat.copy_(tmp)

    return _timed_run(fn, n_warmup, n_repeat)


def bench_kerzoo_update_flat_fused(shapes, device, seed, n_warmup, n_repeat, dtype=torch.float32):
    """Flat buffer KerZOO update: Triton fused accum (3 launches) + update (1 launch)."""
    if not HAS_KERZOO_TRITON:
        return None
    sizes_list = [int(np.prod(s)) for _, s in shapes]
    total_n = sum(sizes_list)
    param_flat = torch.randn(total_n, device=device, dtype=dtype)
    c_flat = torch.randn(total_n, device=device, dtype=dtype)
    grad_flat = torch.zeros(total_n, device=device, dtype=dtype)
    weight_buf = torch.empty(total_n, device=device, dtype=dtype)
    projected_grads = [0.5, 0.3, 0.4]
    lr = 1e-5
    beta_k = 100.0
    decay = 0.5
    clip_thresh = 400000.0
    inv_beta = 1.0 / beta_k
    comp_beta = 1.0 - inv_beta
    offsets_list = []
    off = 0
    for sz in sizes_list:
        offsets_list.append(off)
        off += sz
    torch.cuda.synchronize()

    def fn(rep):
        grad_flat.zero_()
        for si in range(3):
            rng = np.random.RandomState(seed + rep - si)
            k_vals = (2 * rng.rand(len(sizes_list)) - 1) * decay
            for i, (o, sz) in enumerate(zip(offsets_list, sizes_list)):
                kernel_val = _kernel_function(k_vals[i])
                weight_buf[o:o+sz] = projected_grads[si] * kernel_val
            kerzoo_fused_accum_flat(grad_flat, weight_buf, seed + rep - si)
        # Norm + clip (1 PyTorch call)
        norm_val = torch.norm(grad_flat / 3, p=2)
        clip = min(1.0, clip_thresh / (norm_val.item() + 1e-8))
        lr_clip_div3 = lr * clip / 3
        # Fused update (1 Triton launch)
        kerzoo_fused_update_param(param_flat, c_flat, grad_flat,
                                  lr_clip_div3, inv_beta, comp_beta)

    return _timed_run(fn, n_warmup, n_repeat)


# ===================================================================
# Output
# ===================================================================

def print_results(method, model_name, n_params, total_n, results_dict,
                  dtype_str="fp32", bytes_per_elem=4):
    print(f"\n{'=' * 90}")
    print(f"  {model_name}  ({n_params} params, "
          f"{total_n:,} elements, "
          f"{total_n * bytes_per_elem / 1e6:.1f} MB {dtype_str})  [{method.upper()}]")
    print(f"{'=' * 90}")

    header = (f"  {'#':<5s}{'Approach':<48s} "
              f"{'Stream (ms)':>12s} {'Wall (ms)':>10s} {'Speedup':>8s}")
    print(header)
    print(f"  {'-' * 83}")

    keys = list(results_dict.keys())
    baseline_key = keys[0]
    baseline_stream = np.median(results_dict[baseline_key]["gpu_ms"])

    for key in keys:
        data = results_dict[key]
        stream = np.median(data["gpu_ms"])
        wall = np.median(data["wall_ms"])
        speedup = baseline_stream / stream if stream > 0 else 0
        tag = data.get("tag", key)
        label = data.get("label", key)
        print(f"  ({tag}) {label:<46s} {stream:>12.2f} {wall:>10.2f} "
              f"{speedup:>7.2f}x")

    # Decomposition (tag-based for flexible A/B/C/D combinations)
    tag_to_stream = {}
    for key in keys:
        data = results_dict[key]
        tag = data.get("tag", "?")
        tag_to_stream[tag] = np.median(data["gpu_ms"])

    A = tag_to_stream.get("A")
    B = tag_to_stream.get("B")
    C = tag_to_stream.get("C")
    D = tag_to_stream.get("D")
    E = tag_to_stream.get("E")
    F = tag_to_stream.get("F")

    print()
    if A is not None and (B or C or D or E or F):
        print(f"  Decomposition:")
        if D is not None:
            fusion_pt = A - D
            print(f"    Fusion benefit per-tensor (A-D):     "
                  f"{fusion_pt:>8.2f} ms  "
                  f"({100 * fusion_pt / A:.1f}% of A)")
        if F is not None:
            fusion_cuda_pt = A - F
            print(f"    CUDA fusion per-tensor (A-F):        "
                  f"{fusion_cuda_pt:>8.2f} ms  "
                  f"({100 * fusion_cuda_pt / A:.1f}% of A)")
        if D is not None and F is not None:
            triton_vs_cuda_pt = D - F
            print(f"    Triton vs CUDA per-tensor (D-F):     "
                  f"{triton_vs_cuda_pt:>8.2f} ms  "
                  f"({'CUDA faster' if triton_vs_cuda_pt > 0 else 'Triton faster'})")
        if B is not None:
            flat_impact = A - B
            print(f"    Flat buffer impact (A-B):             "
                  f"{flat_impact:>8.2f} ms  "
                  f"({100 * flat_impact / A:.1f}% of A)")
        if D is not None and C is not None:
            consol = D - C
            print(f"    Consolidation benefit (D-C):          "
                  f"{consol:>8.2f} ms  "
                  f"({100 * consol / A:.1f}% of A)")
        if B is not None and C is not None:
            fusion_flat = B - C
            print(f"    Fusion benefit flat buf (B-C):        "
                  f"{fusion_flat:>8.2f} ms  "
                  f"({100 * fusion_flat / A:.1f}% of A)")
        if C is not None:
            total = A - C
            print(f"    Total Triton saving (A-C):            "
                  f"{total:>8.2f} ms  "
                  f"({100 * total / A:.1f}% of A)")
        if E is not None:
            total_cuda = A - E
            print(f"    Total CUDA saving (A-E):              "
                  f"{total_cuda:>8.2f} ms  "
                  f"({100 * total_cuda / A:.1f}% of A)")
        if C is not None and E is not None:
            triton_vs_cuda_flat = C - E
            print(f"    Triton vs CUDA flat (C-E):            "
                  f"{triton_vs_cuda_flat:>8.2f} ms  "
                  f"({'CUDA faster' if triton_vs_cuda_flat > 0 else 'Triton faster'})")

    print()
    print(f"  Kernel launches:")
    print(f"    (A) Per-tensor PyTorch:  ~{n_params} loops x multiple ops each")
    if D is not None:
        print(f"    (D) Per-tensor Triton:   ~{n_params} loops x 1 fused launch each")
    if F is not None:
        print(f"    (F) Per-tensor CUDA:     ~{n_params} loops x 1 fused launch each")
    print(f"    (B) Flat buf PyTorch:    few launches (on contiguous buffer)")
    print(f"    (C) Flat buf Triton:     1-3 launches total")
    if E is not None:
        print(f"    (E) Flat buf CUDA:       1 launch total")


def export_csv(method, all_results, output_dir, dtype_str="fp32"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"benchmark_bubble_{method}_{dtype_str}_{timestamp}.csv"
    fpath = os.path.join(output_dir, fname)
    gpu_name = torch.cuda.get_device_name(0)

    with open(fpath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["# GPU", gpu_name])
        writer.writerow(["# Date", datetime.now().isoformat()])
        writer.writerow(["# Method", method])
        writer.writerow(["# Dtype", dtype_str])
        writer.writerow([])
        writer.writerow([
            "model", "n_params", "total_elements", "operation",
            "approach_tag", "approach_label",
            "stream_median_ms", "stream_std_ms",
            "wall_median_ms", "wall_std_ms", "speedup_vs_baseline",
        ])

        for model_name, mdata in all_results.items():
            rd = mdata["results"]
            keys = list(rd.keys())
            for op_key, op_data in rd.items():
                baseline_key = list(op_data.keys())[0]
                baseline_stream = np.median(op_data[baseline_key]["gpu_ms"])

                for app_key, data in op_data.items():
                    stream = np.median(data["gpu_ms"])
                    speedup = baseline_stream / stream if stream > 0 else 0
                    writer.writerow([
                        model_name, mdata["n_params"], mdata["total_n"],
                        op_key, data.get("tag", app_key), data.get("label", app_key),
                        f"{stream:.3f}", f"{np.std(data['gpu_ms']):.3f}",
                        f"{np.median(data['wall_ms']):.3f}",
                        f"{np.std(data['wall_ms']):.3f}",
                        f"{speedup:.2f}",
                    ])
    return fpath


def generate_latex(method, all_results, output_dir, dtype_str="fp32", bytes_per_elem=4):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"benchmark_bubble_{method}_{dtype_str}_{timestamp}.tex"
    fpath = os.path.join(output_dir, fname)
    gpu_name = torch.cuda.get_device_name(0)

    lines = []
    L = lines.append
    L(r"% Auto-generated by benchmark_bubble_methods.py")
    L(r"% GPU: " + gpu_name)
    L(r"% Method: " + method)
    L(r"% Dtype: " + dtype_str)
    L("")
    L(r"\begin{table}[htbp]")
    L(r"\centering")
    method_label = {"hizoo": "HiZOO", "dizo": "DiZO", "zo2": "ZO2", "kerzoo": "KerZOO"}[method]
    L(r"\caption{Per-tensor dispatch overhead for " + method_label +
      r" operations. Tested on " + gpu_name.replace("_", r"\_") + r".}")
    L(r"\label{tab:dispatch_" + method + "}")
    L(r"\small")
    L(r"\begin{tabular}{lllrrr}")
    L(r"\toprule")
    L(r"\textbf{Model} & \textbf{Operation} & \textbf{Approach} "
      r"& \textbf{Stream (ms)} & \textbf{Speedup} \\")
    L(r"\midrule")

    first_model = True
    for model_name, mdata in all_results.items():
        if not first_model:
            L(r"\addlinespace[4pt]")
        first_model = False
        model_esc = model_name.replace("_", r"\_")

        for op_key, op_data in mdata["results"].items():
            keys = list(op_data.keys())
            baseline_stream = np.median(op_data[keys[0]]["gpu_ms"])

            for i, (app_key, data) in enumerate(op_data.items()):
                stream = np.median(data["gpu_ms"])
                speedup = baseline_stream / stream if stream > 0 else 0
                label = data.get("label", app_key).replace("_", r"\_")
                tag = data.get("tag", "?")
                spd = f"{speedup:.1f}$\\times$"

                mcell = ""
                if i == 0:
                    total_rows = sum(len(od) for od in mdata["results"].values())
                    mcell = r"\multirow{" + str(total_rows) + "}{*}{" + model_esc + "}"

                is_fastest = (i == len(keys) - 1)
                if is_fastest:
                    L(f"{mcell} & {op_key} & \\textbf{{({tag}) {label}}} "
                      f"& \\textbf{{{stream:.1f}}} & \\textbf{{{spd}}} \\\\")
                else:
                    L(f"{mcell} & {op_key} & ({tag}) {label} "
                      f"& {stream:.1f} & {spd} \\\\")

    L(r"\bottomrule")
    L(r"\end{tabular}")
    L(r"\end{table}")

    with open(fpath, "w") as f:
        f.write("\n".join(lines) + "\n")
    return fpath


# ===================================================================
# Main
# ===================================================================

def run_hizoo(shapes, model_name, n_params, total_n, device, args):
    results = {}
    dtype = args.torch_dtype

    # --- Perturb benchmarks ---
    print(f"  [perturb] (A) Per-tensor HiZOO perturb ...")
    rA = _safe_bench(bench_hizoo_perturb_per_tensor, "per-tensor perturb",
                     shapes, device, args.seed, args.warmup, args.repeat, dtype)
    if rA is None:
        print(f"    Cannot proceed without baseline, skipping model")
        return results
    rA["tag"] = "A"
    rA["label"] = "Per-tensor HiZOO perturb (baseline)"
    _free_gpu()

    print(f"  [perturb] (B) Flat buf PyTorch HiZOO perturb ...")
    rB = _safe_bench(bench_hizoo_perturb_flat_pytorch, "flat pytorch perturb",
                     shapes, device, args.seed, args.warmup, args.repeat, dtype)
    _free_gpu()

    rC = None
    if HAS_HIZOO_TRITON:
        print(f"  [perturb] (C) Flat buf Triton fused HiZOO perturb ...")
        rC = _safe_bench(bench_hizoo_perturb_flat_fused, "flat fused perturb",
                         shapes, device, args.seed, args.warmup, args.repeat, dtype)
        if rC is not None:
            rC["tag"] = "C"
            rC["label"] = "Flat buf + Triton fused (1 launch)"
    _free_gpu()

    rD = None
    if HAS_HIZOO_TRITON:
        print(f"  [perturb] (D) Per-tensor Triton fused HiZOO perturb ...")
        rD = _safe_bench(bench_hizoo_perturb_per_tensor_fused, "per-tensor fused perturb",
                         shapes, device, args.seed, args.warmup, args.repeat, dtype)
        if rD is not None:
            rD["tag"] = "D"
            rD["label"] = "Per-tensor + Triton fused (N launches)"
    _free_gpu()

    perturb_results = {"per_tensor": rA}
    if rD is not None:
        perturb_results["per_tensor_fused"] = rD
    if rB is not None:
        rB["tag"] = "B"
        rB["label"] = "Flat buf + PyTorch (randn+rsqrt+add)"
        perturb_results["flat_pytorch"] = rB
    if rC is not None:
        perturb_results["flat_fused"] = rC
    results["perturb"] = perturb_results

    print_results("hizoo-perturb", model_name, n_params, total_n, perturb_results,
                  args.dtype, args.bytes_per_elem)
    _free_gpu()

    # --- Update-restore benchmarks ---
    print(f"\n  [update-restore] (A) Per-tensor HiZOO update-restore ...")
    rA2 = _safe_bench(bench_hizoo_update_per_tensor, "per-tensor update",
                      shapes, device, args.seed, args.warmup, args.repeat, dtype)
    if rA2 is None:
        return results
    rA2["tag"] = "A"
    rA2["label"] = "Per-tensor HiZOO update-restore (baseline)"
    _free_gpu()

    print(f"  [update-restore] (B) Flat buf PyTorch update-restore ...")
    rB2 = _safe_bench(bench_hizoo_update_flat_pytorch, "flat pytorch update",
                      shapes, device, args.seed, args.warmup, args.repeat, dtype)
    _free_gpu()

    rC2 = None
    if HAS_HIZOO_TRITON:
        print(f"  [update-restore] (C) Flat buf Triton fused update ...")
        rC2 = _safe_bench(bench_hizoo_update_flat_fused, "flat fused update",
                          shapes, device, args.seed, args.warmup, args.repeat, dtype)
        if rC2 is not None:
            rC2["tag"] = "C"
            rC2["label"] = "Flat buf + Triton fused (1 launch)"
    _free_gpu()

    rD2 = None
    if HAS_HIZOO_TRITON:
        print(f"  [update-restore] (D) Per-tensor Triton fused update ...")
        rD2 = _safe_bench(bench_hizoo_update_per_tensor_fused, "per-tensor fused update",
                          shapes, device, args.seed, args.warmup, args.repeat, dtype)
        if rD2 is not None:
            rD2["tag"] = "D"
            rD2["label"] = "Per-tensor + Triton fused (N launches)"
    _free_gpu()

    update_results = {"per_tensor": rA2}
    if rD2 is not None:
        update_results["per_tensor_fused"] = rD2
    if rB2 is not None:
        rB2["tag"] = "B"
        rB2["label"] = "Flat buf + PyTorch (multiple ops)"
        update_results["flat_pytorch"] = rB2
    if rC2 is not None:
        update_results["flat_fused"] = rC2
    results["update_restore"] = update_results

    print_results("hizoo-update", model_name, n_params, total_n, update_results,
                  args.dtype, args.bytes_per_elem)
    return results


def run_dizo(shapes, model_name, n_params, total_n, device, args):
    results = {}
    dtype = args.torch_dtype

    print(f"  [constraints] (A) Per-tensor DiZO constraints ...")
    rA = _safe_bench(bench_dizo_constraints_per_tensor, "per-tensor constraints",
                     shapes, device, args.seed, args.warmup, args.repeat, dtype)
    if rA is None:
        print(f"    Cannot proceed without baseline, skipping model")
        return results
    rA["tag"] = "A"
    rA["label"] = "Per-tensor constraints (apply+reverse)"
    _free_gpu()

    print(f"  [constraints] (B) Flat buf PyTorch constraints ...")
    rB = _safe_bench(bench_dizo_constraints_flat_pytorch, "flat pytorch constraints",
                     shapes, device, args.seed, args.warmup, args.repeat, dtype)
    _free_gpu()

    rC = None
    if HAS_DIZO_TRITON_V2:
        print(f"  [constraints] (C) Flat buf Triton V2 fused constraints ...")
        rC = _safe_bench(bench_dizo_constraints_flat_fused, "flat fused constraints",
                         shapes, device, args.seed, args.warmup, args.repeat, dtype)
        if rC is not None:
            rC["tag"] = "C"
            rC["label"] = "Flat buf + Triton V2 fused (3 launches)"
    _free_gpu()

    rD = None
    if HAS_TRITON_DIRECT:
        print(f"  [constraints] (D) Per-tensor Triton fused constraints ...")
        rD = _safe_bench(bench_dizo_constraints_per_tensor_fused, "per-tensor fused constraints",
                         shapes, device, args.seed, args.warmup, args.repeat, dtype)
        if rD is not None:
            rD["tag"] = "D"
            rD["label"] = "Per-tensor + Triton fused (3N launches)"
    _free_gpu()

    constraint_results = {"per_tensor": rA}
    if rD is not None:
        constraint_results["per_tensor_fused"] = rD
    if rB is not None:
        rB["tag"] = "B"
        rB["label"] = "Flat buf + PyTorch (loop w/ slices)"
        constraint_results["flat_pytorch"] = rB
    if rC is not None:
        constraint_results["flat_fused"] = rC
    results["constraints"] = constraint_results

    print_results("dizo-constraints", model_name, n_params, total_n,
                  constraint_results, args.dtype, args.bytes_per_elem)
    return results


def run_zo2(shapes, model_name, n_params, total_n, device, args):
    results = {}
    dtype = args.torch_dtype

    # --- Dual perturb benchmarks (A/D/B/C) ---
    print(f"  [dual-perturb] (A) Per-tensor dual perturb ...")
    rA = _safe_bench(bench_zo2_dual_perturb_per_tensor, "per-tensor dual perturb",
                     shapes, device, args.seed, args.warmup, args.repeat, dtype)
    if rA is None:
        print(f"    Cannot proceed without baseline, skipping model")
        return results
    rA["tag"] = "A"
    rA["label"] = "Per-tensor dual perturb (baseline)"
    _free_gpu()

    rD = None
    if HAS_ZO2_TRITON:
        print(f"  [dual-perturb] (D) Per-tensor Triton fused dual perturb ...")
        rD = _safe_bench(bench_zo2_dual_perturb_per_tensor_fused, "per-tensor fused dual perturb",
                         shapes, device, args.seed, args.warmup, args.repeat, dtype)
        if rD is not None:
            rD["tag"] = "D"
            rD["label"] = "Per-tensor + Triton fused (N launches)"
    _free_gpu()

    print(f"  [dual-perturb] (B) Flat buf PyTorch dual perturb ...")
    rB = _safe_bench(bench_zo2_dual_perturb_flat_pytorch, "flat pytorch dual perturb",
                     shapes, device, args.seed, args.warmup, args.repeat, dtype)
    _free_gpu()

    rC = None
    if HAS_ZO2_TRITON:
        print(f"  [dual-perturb] (C) Flat buf Triton fused dual perturb ...")
        rC = _safe_bench(bench_zo2_dual_perturb_flat_fused, "flat fused dual perturb",
                         shapes, device, args.seed, args.warmup, args.repeat, dtype)
        if rC is not None:
            rC["tag"] = "C"
            rC["label"] = "Flat buf + Triton fused (1 launch)"
    _free_gpu()

    rF = None
    if HAS_ZO2_CUDA:
        print(f"  [dual-perturb] (F) Per-tensor CUDA fused dual perturb ...")
        rF = _safe_bench(bench_zo2_dual_perturb_per_tensor_cuda, "per-tensor CUDA dual perturb",
                         shapes, device, args.seed, args.warmup, args.repeat, dtype)
        if rF is not None:
            rF["tag"] = "F"
            rF["label"] = "Per-tensor + CUDA fused (N launches)"
    _free_gpu()

    rE = None
    if HAS_ZO2_CUDA:
        print(f"  [dual-perturb] (E) Flat buf CUDA fused dual perturb ...")
        rE = _safe_bench(bench_zo2_dual_perturb_flat_cuda, "flat CUDA dual perturb",
                         shapes, device, args.seed, args.warmup, args.repeat, dtype)
        if rE is not None:
            rE["tag"] = "E"
            rE["label"] = "Flat buf + CUDA fused (1 launch)"
    _free_gpu()

    dual_results = {"per_tensor": rA}
    if rD is not None:
        dual_results["per_tensor_fused"] = rD
    if rF is not None:
        dual_results["per_tensor_cuda"] = rF
    if rB is not None:
        rB["tag"] = "B"
        rB["label"] = "Flat buf + PyTorch (randn + add/sub)"
        dual_results["flat_pytorch"] = rB
    if rC is not None:
        dual_results["flat_fused"] = rC
    if rE is not None:
        dual_results["flat_cuda"] = rE
    results["dual_perturb"] = dual_results

    print_results("zo2-dual-perturb", model_name, n_params, total_n,
                  dual_results, args.dtype, args.bytes_per_elem)
    _free_gpu()

    # --- Dual-forward benchmarks (update + dual perturb combined) ---
    print(f"\n  [dual-forward] (A) Per-tensor dual-forward ...")
    rA2 = _safe_bench(bench_zo2_dual_forward_per_tensor, "per-tensor dual-forward",
                      shapes, device, args.seed, args.warmup, args.repeat, dtype)
    if rA2 is None:
        return results
    rA2["tag"] = "A"
    rA2["label"] = "Per-tensor dual-forward (baseline)"
    _free_gpu()

    rD2 = None
    if HAS_ZO2_TRITON:
        print(f"  [dual-forward] (D) Per-tensor Triton combined dual-forward ...")
        rD2 = _safe_bench(bench_zo2_dual_forward_per_tensor_fused, "per-tensor fused dual-forward",
                          shapes, device, args.seed, args.warmup, args.repeat, dtype)
        if rD2 is not None:
            rD2["tag"] = "D"
            rD2["label"] = "Per-tensor + Triton combined (N launches)"
    _free_gpu()

    print(f"  [dual-forward] (B) Flat buf PyTorch dual-forward ...")
    rB2 = _safe_bench(bench_zo2_dual_forward_flat_pytorch, "flat pytorch dual-forward",
                      shapes, device, args.seed, args.warmup, args.repeat, dtype)
    _free_gpu()

    rC2 = None
    if HAS_ZO2_TRITON:
        print(f"  [dual-forward] (C) Flat buf Triton combined dual-forward ...")
        rC2 = _safe_bench(bench_zo2_dual_forward_flat_fused, "flat fused dual-forward",
                          shapes, device, args.seed, args.warmup, args.repeat, dtype)
        if rC2 is not None:
            rC2["tag"] = "C"
            rC2["label"] = "Flat buf + Triton combined (1 launch)"
    _free_gpu()

    rF2 = None
    if HAS_ZO2_CUDA:
        print(f"  [dual-forward] (F) Per-tensor CUDA combined dual-forward ...")
        rF2 = _safe_bench(bench_zo2_dual_forward_per_tensor_cuda, "per-tensor CUDA dual-forward",
                          shapes, device, args.seed, args.warmup, args.repeat, dtype)
        if rF2 is not None:
            rF2["tag"] = "F"
            rF2["label"] = "Per-tensor + CUDA combined (N launches)"
    _free_gpu()

    rE2 = None
    if HAS_ZO2_CUDA:
        print(f"  [dual-forward] (E) Flat buf CUDA combined dual-forward ...")
        rE2 = _safe_bench(bench_zo2_dual_forward_flat_cuda, "flat CUDA dual-forward",
                          shapes, device, args.seed, args.warmup, args.repeat, dtype)
        if rE2 is not None:
            rE2["tag"] = "E"
            rE2["label"] = "Flat buf + CUDA combined (1 launch)"
    _free_gpu()

    fwd_results = {"per_tensor": rA2}
    if rD2 is not None:
        fwd_results["per_tensor_fused"] = rD2
    if rF2 is not None:
        fwd_results["per_tensor_cuda"] = rF2
    if rB2 is not None:
        rB2["tag"] = "B"
        rB2["label"] = "Flat buf + PyTorch (update + randn + add/sub)"
        fwd_results["flat_pytorch"] = rB2
    if rC2 is not None:
        fwd_results["flat_fused"] = rC2
    if rE2 is not None:
        fwd_results["flat_cuda"] = rE2
    results["dual_forward"] = fwd_results

    print_results("zo2-dual-forward", model_name, n_params, total_n,
                  fwd_results, args.dtype, args.bytes_per_elem)
    return results


def run_kerzoo(shapes, model_name, n_params, total_n, device, args):
    results = {}
    dtype = args.torch_dtype

    # --- Perturb benchmarks ---
    print(f"  [perturb] (A) Per-tensor KerZOO perturb ...")
    rA = _safe_bench(bench_kerzoo_perturb_per_tensor, "per-tensor perturb",
                     shapes, device, args.seed, args.warmup, args.repeat, dtype)
    if rA is None:
        print(f"    Cannot proceed without baseline, skipping model")
        return results
    rA["tag"] = "A"
    rA["label"] = "Per-tensor KerZOO perturb (baseline)"
    _free_gpu()

    rD = None
    if HAS_KERZOO_TRITON:
        print(f"  [perturb] (D) Per-tensor Triton fused KerZOO perturb ...")
        rD = _safe_bench(bench_kerzoo_perturb_per_tensor_fused, "per-tensor fused perturb",
                         shapes, device, args.seed, args.warmup, args.repeat, dtype)
        if rD is not None:
            rD["tag"] = "D"
            rD["label"] = "Per-tensor + Triton fused (N launches)"
    _free_gpu()

    print(f"  [perturb] (B) Flat buf PyTorch KerZOO perturb ...")
    rB = _safe_bench(bench_kerzoo_perturb_flat_pytorch, "flat pytorch perturb",
                     shapes, device, args.seed, args.warmup, args.repeat, dtype)
    _free_gpu()

    rC = None
    if HAS_KERZOO_TRITON:
        print(f"  [perturb] (C) Flat buf Triton fused KerZOO perturb ...")
        rC = _safe_bench(bench_kerzoo_perturb_flat_fused, "flat fused perturb",
                         shapes, device, args.seed, args.warmup, args.repeat, dtype)
        if rC is not None:
            rC["tag"] = "C"
            rC["label"] = "Flat buf + Triton fused (1 launch)"
    _free_gpu()

    perturb_results = {"per_tensor": rA}
    if rD is not None:
        perturb_results["per_tensor_fused"] = rD
    if rB is not None:
        rB["tag"] = "B"
        rB["label"] = "Flat buf + PyTorch (expand k + flat ops)"
        perturb_results["flat_pytorch"] = rB
    if rC is not None:
        perturb_results["flat_fused"] = rC
    results["perturb"] = perturb_results

    print_results("kerzoo-perturb", model_name, n_params, total_n, perturb_results,
                  args.dtype, args.bytes_per_elem)
    _free_gpu()

    # --- Update benchmarks ---
    print(f"\n  [update] (A) Per-tensor KerZOO update ...")
    rA2 = _safe_bench(bench_kerzoo_update_per_tensor, "per-tensor update",
                      shapes, device, args.seed, args.warmup, args.repeat, dtype)
    if rA2 is None:
        return results
    rA2["tag"] = "A"
    rA2["label"] = "Per-tensor KerZOO update (baseline)"
    _free_gpu()

    rD2 = None
    if HAS_KERZOO_TRITON:
        print(f"  [update] (D) Per-tensor Triton fused KerZOO update ...")
        rD2 = _safe_bench(bench_kerzoo_update_per_tensor_fused, "per-tensor fused update",
                          shapes, device, args.seed, args.warmup, args.repeat, dtype)
        if rD2 is not None:
            rD2["tag"] = "D"
            rD2["label"] = "Per-tensor + Triton fused (3N+N launches)"
    _free_gpu()

    print(f"  [update] (B) Flat buf PyTorch KerZOO update ...")
    rB2 = _safe_bench(bench_kerzoo_update_flat_pytorch, "flat pytorch update",
                      shapes, device, args.seed, args.warmup, args.repeat, dtype)
    _free_gpu()

    rC2 = None
    if HAS_KERZOO_TRITON:
        print(f"  [update] (C) Flat buf Triton fused KerZOO update ...")
        rC2 = _safe_bench(bench_kerzoo_update_flat_fused, "flat fused update",
                          shapes, device, args.seed, args.warmup, args.repeat, dtype)
        if rC2 is not None:
            rC2["tag"] = "C"
            rC2["label"] = "Flat buf + Triton fused (3+1 launches)"
    _free_gpu()

    update_results = {"per_tensor": rA2}
    if rD2 is not None:
        update_results["per_tensor_fused"] = rD2
    if rB2 is not None:
        rB2["tag"] = "B"
        rB2["label"] = "Flat buf + PyTorch (expand + flat ops)"
        update_results["flat_pytorch"] = rB2
    if rC2 is not None:
        update_results["flat_fused"] = rC2
    results["update"] = update_results

    print_results("kerzoo-update", model_name, n_params, total_n, update_results,
                  args.dtype, args.bytes_per_elem)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Dispatch overhead benchmark for HiZOO, DiZO, ZO2, KerZOO")
    parser.add_argument("--method", required=True, choices=["hizoo", "dizo", "zo2", "kerzoo"],
                        help="ZO method to benchmark")
    parser.add_argument("--models", nargs="+", default=list(MODEL_SHAPE_FILES.keys()),
                        choices=list(MODEL_SHAPE_FILES.keys()),
                        help="Models to benchmark")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16",
                        help="Data type for tensors (fp16 halves memory)")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    # Map string to torch dtype
    args.torch_dtype = {"fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    args.bytes_per_elem = 2 if args.dtype == "fp16" else 4

    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Method: {args.method}")
    print(f"Dtype: {args.dtype}")
    print(f"HiZOO Triton: {'available' if HAS_HIZOO_TRITON else 'N/A'}")
    print(f"DiZO Triton V2: {'available' if HAS_DIZO_TRITON_V2 else 'N/A'}")
    print(f"ZO2 Triton: {'available' if HAS_ZO2_TRITON else 'N/A'}")
    print(f"ZO2 CUDA: {'available' if HAS_ZO2_CUDA else 'N/A'}")
    print(f"KerZOO Triton: {'available' if HAS_KERZOO_TRITON else 'N/A'}")

    runner = {"hizoo": run_hizoo, "dizo": run_dizo, "zo2": run_zo2,
              "kerzoo": run_kerzoo}[args.method]
    all_results = {}

    for model_name in args.models:
        shape_file = MODEL_SHAPE_FILES[model_name]
        if not shape_file.exists():
            print(f"Skipping {model_name}: shape file not found")
            continue

        shapes = parse_shapes_file(shape_file)
        n_params = len(shapes)
        total_n = sum(int(np.prod(s)) for _, s in shapes)

        print(f"\n>>> {model_name}: {n_params} parameters, "
              f"{total_n:,} elements, "
              f"{total_n * args.bytes_per_elem / 1e6:.1f} MB {args.dtype}")

        rd = runner(shapes, model_name, n_params, total_n, device, args)
        all_results[model_name] = {
            "n_params": n_params,
            "total_n": total_n,
            "results": rd,
        }
        torch.cuda.empty_cache()

    # Export
    if all_results:
        out_dir = args.output_dir or str(SCRIPT_DIR)
        os.makedirs(out_dir, exist_ok=True)
        csv_path = export_csv(args.method, all_results, out_dir, args.dtype)
        print(f"\nCSV saved to: {csv_path}")
        tex_path = generate_latex(args.method, all_results, out_dir, args.dtype, args.bytes_per_elem)
        print(f"LaTeX saved to: {tex_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
