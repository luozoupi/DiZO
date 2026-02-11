#!/usr/bin/env python3
"""
Unified Benchmark: All ZO Perturbation Kernel Variants
======================================================

Compares original kernels (DiZO/HiZOO/ZO2) against optimized v2 variants
on OPT model parameter dimensions using CUDA event timing.

Usage:
    CUDA_VISIBLE_DEVICES=0 conda run -n py310 python benchmark_all.py
    CUDA_VISIBLE_DEVICES=0 conda run -n py310 python benchmark_all.py --dims small
    CUDA_VISIBLE_DEVICES=0 conda run -n py310 python benchmark_all.py --dims large --repeat 100
"""

import argparse
import csv
import math
import os
import sys
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import triton

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Import original kernels
# ---------------------------------------------------------------------------

# DiZO / MeZO Philox
try:
    from DiZO.large_models.cuda_kernels.Perturb_wise.triton_fused_perturb import (
        fused_perturb_kernel_philox as orig_philox_kernel,
        fused_perturb_kernel_philox_2x as orig_philox_2x_kernel,
    )
    HAS_DIZO = True
except ImportError as e:
    HAS_DIZO = False
    print(f"[skip] DiZO kernels: {e}")

# HiZOO
try:
    from HiZOO.cuda_kernels.hizoo_fused_ops import (
        hizoo_randn_kernel as hizoo_randn_kern,
    )
    HAS_HIZOO = True
except ImportError as e:
    HAS_HIZOO = False
    print(f"[skip] HiZOO kernels: {e}")

# ZO2 Triton
try:
    from zo2.cuda_kernels.block_wise.triton_kernels import (
        _zo2_fused_block_perturb_kernel as zo2_triton_kern,
    )
    HAS_ZO2_TRITON = True
except ImportError as e:
    HAS_ZO2_TRITON = False
    print(f"[skip] ZO2 Triton kernels: {e}")

# ZO2 CUDA
try:
    from zo2.cuda_kernels.cuda_wrapper import fused_block_perturb as zo2_cuda_perturb
    HAS_ZO2_CUDA = True
except ImportError as e:
    HAS_ZO2_CUDA = False
    print(f"[skip] ZO2 CUDA kernels: {e}")

# New optimized kernels (Triton)
try:
    from kernel_optimization.philox_v2 import (
        fused_perturb_v2a_kernel,
        fused_perturb_v2b_kernel,
        fused_perturb_v2c_kernel,
        fused_perturb_v2d_kernel,
        perturb_v2e,
        perturb_v2f,
    )
    HAS_V2 = True
except Exception as e:
    HAS_V2 = False
    print(f"[skip] V2 Triton kernels: {e}")

# New optimized kernels (CUDA)
HAS_V2_CUDA = False
try:
    import philox_v2_cuda_ext as _v2_cuda
    HAS_V2_CUDA = True
except ImportError:
    try:
        from torch.utils.cpp_extension import load as _jit_load
        _v2_cuda_src = Path(__file__).resolve().parent / "philox_v2_cuda.cu"
        if _v2_cuda_src.exists():
            # Detect GPU arch for JIT compilation
            _sm_flag = "-arch=sm_120"
            try:
                _cap = torch.cuda.get_device_capability(0)
                _sm = f"{_cap[0]}{_cap[1]}0" if _cap[1] >= 10 else f"{_cap[0]}{_cap[1]}"
                _sm_flag = f"-gencode=arch=compute_{_sm},code=sm_{_sm}"
            except Exception:
                pass
            print(f"[JIT] Compiling CUDA kernels with {_sm_flag} ...")
            _v2_cuda = _jit_load(
                name="philox_v2_cuda_ext",
                sources=[str(_v2_cuda_src)],
                extra_cuda_cflags=["-O3", _sm_flag, "--use_fast_math",
                                   "-lineinfo", "--ptxas-options=-v"],
                verbose=True,
            )
            HAS_V2_CUDA = True
    except Exception as e:
        print(f"[skip] V2 CUDA kernels: {e}")


# ---------------------------------------------------------------------------
# Dimension sets
# ---------------------------------------------------------------------------

DIMS_SMALL = OrderedDict([
    ("bias_1k",    1_024),
    ("attn_350m",  1_048_576),
    ("ffn_350m",   4_194_304),
])

DIMS_MEDIUM = OrderedDict([
    ("bias_1k",    1_024),
    ("attn_350m",  1_048_576),
    ("ffn_350m",   4_194_304),
    ("embed_350m", 25_739_264),
    ("attn_2.7b",  6_553_600),
    ("ffn_2.7b",   26_214_400),
])

DIMS_LARGE = OrderedDict([
    ("bias_1k",      1_024),
    ("attn_350m",    1_048_576),
    ("ffn_350m",     4_194_304),
    ("embed_350m",  25_739_264),
    ("attn_2.7b",    6_553_600),
    ("ffn_2.7b",    26_214_400),
    ("attn_6.7b",   16_777_216),
    ("ffn_6.7b",    67_108_864),
    ("embed_6.7b", 205_914_112),
    ("attn_13b",    26_214_400),
    ("ffn_13b",    104_857_600),
    ("embed_13b",  257_392_640),
])


# ---------------------------------------------------------------------------
# Kernel registry: name → (launch_fn, description)
#   launch_fn(buf, seed) — operates in-place on buf
# ---------------------------------------------------------------------------

def build_kernel_registry(device: torch.device) -> OrderedDict:
    """Build registry of all available kernels."""
    BS = 1024  # default BLOCK_SIZE
    reg = OrderedDict()

    # --- PyTorch baseline ---
    def _pytorch(buf, seed):
        torch.manual_seed(seed)
        buf.copy_(torch.randn_like(buf))
    reg["PyTorch randn"] = (_pytorch, "torch.randn baseline")

    # --- Original DiZO Philox (as profiled by NCU) ---
    if HAS_DIZO:
        def _orig_philox_4w(buf, seed):
            n = buf.numel()
            grid = (triton.cdiv(n, BS),)
            orig_philox_kernel[grid](buf, seed, 1.0, n, BLOCK_SIZE=BS, num_warps=4)
        reg["Orig Philox 4w"] = (_orig_philox_4w,
            "Original fused_perturb_kernel_philox, BS=1024, 4 warps")

        def _orig_philox_8w(buf, seed):
            n = buf.numel()
            grid = (triton.cdiv(n, BS),)
            orig_philox_kernel[grid](buf, seed, 1.0, n, BLOCK_SIZE=BS, num_warps=8)
        reg["Orig Philox 8w"] = (_orig_philox_8w,
            "Original Philox with 8 warps (256 threads)")

        def _orig_2x(buf, seed):
            n = buf.numel()
            grid = (triton.cdiv(n, 512 * 2),)
            orig_philox_2x_kernel[grid](buf, seed, 1.0, n, BLOCK_SIZE=512,
                                        num_warps=4)
        reg["Orig Philox 2x"] = (_orig_2x,
            "Original 2x kernel (2 normals/Philox)")

    # --- HiZOO randn ---
    if HAS_HIZOO:
        def _hizoo_randn(buf, seed):
            n = buf.numel()
            grid = (triton.cdiv(n, BS),)
            hizoo_randn_kern[grid](buf, seed, 0, n, BLOCK_SIZE=BS, num_warps=4)
        reg["HiZOO randn"] = (_hizoo_randn, "HiZOO Philox randn kernel")

    # --- ZO2 Triton ---
    if HAS_ZO2_TRITON:
        def _zo2_triton(buf, seed):
            n = buf.numel()
            grid = (triton.cdiv(n, BS),)
            zo2_triton_kern[grid](buf, seed, 1.0, 1.0, n, BLOCK_SIZE=BS,
                                  num_warps=4)
        reg["ZO2 Triton"] = (_zo2_triton, "ZO2 tl.randn Triton kernel")

    # --- ZO2 CUDA ---
    if HAS_ZO2_CUDA:
        def _zo2_cuda(buf, seed):
            zo2_cuda_perturb(buf, seed, 1.0, 1.0)
        reg["ZO2 CUDA"] = (_zo2_cuda, "ZO2 CUDA extension (__umulhi)")

    # --- V2 optimized ---
    if HAS_V2:
        # V2A: mulhi 10 rounds, 4 warps
        def _v2a_4w(buf, seed):
            n = buf.numel()
            grid = (triton.cdiv(n, BS),)
            fused_perturb_v2a_kernel[grid](buf, seed, 1.0, n,
                                           BLOCK_SIZE=BS, num_warps=4)
        reg["V2A mulhi-10r 4w"] = (_v2a_4w,
            "PTX mulhi, 10 rounds, 4 warps")

        # V2A: mulhi 10 rounds, 8 warps
        def _v2a_8w(buf, seed):
            n = buf.numel()
            grid = (triton.cdiv(n, BS),)
            fused_perturb_v2a_kernel[grid](buf, seed, 1.0, n,
                                           BLOCK_SIZE=BS, num_warps=8)
        reg["V2A mulhi-10r 8w"] = (_v2a_8w,
            "PTX mulhi, 10 rounds, 8 warps")

        # V2B: mulhi 7 rounds, 4 warps
        def _v2b_4w(buf, seed):
            n = buf.numel()
            grid = (triton.cdiv(n, BS),)
            fused_perturb_v2b_kernel[grid](buf, seed, 1.0, n,
                                           BLOCK_SIZE=BS, num_warps=4)
        reg["V2B mulhi-7r 4w"] = (_v2b_4w,
            "PTX mulhi, 7 rounds, 4 warps")

        # V2B: mulhi 7 rounds, 8 warps
        def _v2b_8w(buf, seed):
            n = buf.numel()
            grid = (triton.cdiv(n, BS),)
            fused_perturb_v2b_kernel[grid](buf, seed, 1.0, n,
                                           BLOCK_SIZE=BS, num_warps=8)
        reg["V2B mulhi-7r 8w"] = (_v2b_8w,
            "PTX mulhi, 7 rounds, 8 warps")

        # V2C: mulhi 7r, 2x normals, 4 warps
        def _v2c_4w(buf, seed):
            n = buf.numel()
            grid = (triton.cdiv(n, 512 * 2),)
            fused_perturb_v2c_kernel[grid](buf, seed, 1.0, n,
                                           BLOCK_SIZE=512, num_warps=4)
        reg["V2C 2x-7r 4w"] = (_v2c_4w,
            "PTX mulhi, 7 rounds, 2x normals, 4 warps")

        # V2C: mulhi 7r, 2x normals, 8 warps
        def _v2c_8w(buf, seed):
            n = buf.numel()
            grid = (triton.cdiv(n, 512 * 2),)
            fused_perturb_v2c_kernel[grid](buf, seed, 1.0, n,
                                           BLOCK_SIZE=512, num_warps=8)
        reg["V2C 2x-7r 8w"] = (_v2c_8w,
            "PTX mulhi, 7 rounds, 2x normals, 8 warps")

        # V2D: tl.randn, 4 warps
        def _v2d_4w(buf, seed):
            n = buf.numel()
            grid = (triton.cdiv(n, BS),)
            fused_perturb_v2d_kernel[grid](buf, seed, 1.0, n,
                                           BLOCK_SIZE=BS, num_warps=4)
        reg["V2D tl.randn 4w"] = (_v2d_4w,
            "tl.randn, 4 warps")

        # V2D: tl.randn, 8 warps
        def _v2d_8w(buf, seed):
            n = buf.numel()
            grid = (triton.cdiv(n, BS),)
            fused_perturb_v2d_kernel[grid](buf, seed, 1.0, n,
                                           BLOCK_SIZE=BS, num_warps=8)
        reg["V2D tl.randn 8w"] = (_v2d_8w,
            "tl.randn, 8 warps")

        # V2E: autotuned mulhi 7r
        def _v2e(buf, seed):
            perturb_v2e(buf, seed, 1.0)
        reg["V2E auto mulhi-7r"] = (_v2e,
            "Autotuned PTX mulhi, 7 rounds")

        # V2F: autotuned tl.randn
        def _v2f(buf, seed):
            perturb_v2f(buf, seed, 1.0)
        reg["V2F auto tl.randn"] = (_v2f,
            "Autotuned tl.randn")

    # --- V2 CUDA optimized ---
    if HAS_V2_CUDA:
        def _cuda_scalar_10r(buf, seed):
            _v2_cuda.perturb_scalar_10r(buf, seed, 1.0)
        reg["CUDA scalar-10r"] = (_cuda_scalar_10r,
            "CUDA __umulhi scalar, 10 rounds")

        def _cuda_scalar_7r(buf, seed):
            _v2_cuda.perturb_scalar_7r(buf, seed, 1.0)
        reg["CUDA scalar-7r"] = (_cuda_scalar_7r,
            "CUDA __umulhi scalar, 7 rounds")

        def _cuda_vec4_10r(buf, seed):
            _v2_cuda.perturb_vec4_10r(buf, seed, 1.0)
        reg["CUDA vec4-10r"] = (_cuda_vec4_10r,
            "CUDA vec4+sincosf, 10 rounds, 4x normals")

        def _cuda_vec4_7r(buf, seed):
            _v2_cuda.perturb_vec4_7r(buf, seed, 1.0)
        reg["CUDA vec4-7r"] = (_cuda_vec4_7r,
            "CUDA vec4+sincosf, 7 rounds, 4x normals (fastest)")

    return reg


# ---------------------------------------------------------------------------
# Correctness check: verify N(0,1) distribution
# ---------------------------------------------------------------------------

def check_distribution(buf: torch.Tensor, name: str) -> bool:
    """Quick check: mean ≈ 0, std ≈ 1."""
    n = buf.numel()
    mean = buf.mean().item()
    std = buf.std().item()
    se_mean = 1.0 / math.sqrt(n)
    se_std = 1.0 / math.sqrt(2.0 * n)
    z_mean = abs(mean) / se_mean
    z_std = abs(std - 1.0) / se_std
    ok = z_mean < 4.0 and z_std < 4.0
    status = "OK" if ok else "FAIL"
    print(f"  [{status}] {name}: mean={mean:.5f} std={std:.5f} "
          f"(z_mean={z_mean:.1f} z_std={z_std:.1f})")
    return ok


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def benchmark_kernel(
    kernel_fn: Callable,
    n_elements: int,
    device: torch.device,
    seed: int = 42,
    n_warmup: int = 5,
    n_repeat: int = 50,
) -> Dict[str, float]:
    """Benchmark a single kernel, return timing stats."""
    buf = torch.zeros(n_elements, device=device, dtype=torch.float32)

    # Warmup (also triggers Triton JIT compilation)
    for i in range(n_warmup):
        buf.zero_()
        try:
            kernel_fn(buf, seed + i)
        except Exception as e:
            return {"error": str(e)}
    torch.cuda.synchronize()

    # Timed runs
    gpu_times = []
    for rep in range(n_repeat):
        buf.zero_()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        kernel_fn(buf, seed + rep)
        end.record()
        torch.cuda.synchronize()
        gpu_times.append(start.elapsed_time(end) * 1e3)  # ms → us

    med_us = float(np.median(gpu_times))
    std_us = float(np.std(gpu_times))
    mem_bytes = n_elements * 4  # float32: read + write = 2 * 4 bytes, but
    # kernel only writes (RNG generates in-place), so effective = 1 store
    # Actually: load + store for fused perturb = 2 * n * 4 bytes
    bw_gbs = (2 * mem_bytes / 1e9) / (med_us / 1e6) if med_us > 0 else 0.0

    return {
        "median_us": med_us,
        "std_us": std_us,
        "bw_gbs": bw_gbs,
    }


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def export_csv(
    results: Dict[str, Dict[str, Dict[str, float]]],
    dims: OrderedDict,
    gpu_name: str,
    output_dir: str,
) -> str:
    """Write benchmark results to a CSV file.  Returns the file path."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"benchmark_results_{timestamp}.csv"
    fpath = os.path.join(output_dir, fname)

    sorted_dims = sorted(dims.items(), key=lambda kv: kv[1])
    # Collect all kernel names preserving order
    all_kernels: List[str] = []
    for dim_label, _ in sorted_dims:
        if dim_label in results:
            for k in results[dim_label]:
                if k not in all_kernels:
                    all_kernels.append(k)

    with open(fpath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["# GPU", gpu_name])
        writer.writerow(["# Date", datetime.now().isoformat()])
        writer.writerow([])
        writer.writerow([
            "dimension", "n_elements",
            "kernel", "median_us", "std_us", "bw_gbs",
            "speedup_vs_pytorch", "speedup_vs_orig_philox",
        ])
        for dim_label, n_elements in sorted_dims:
            if dim_label not in results:
                continue
            dim_res = results[dim_label]
            pt_us = dim_res.get("PyTorch randn", {}).get("median_us", None)
            orig_us = dim_res.get("Orig Philox 4w", {}).get("median_us", None)
            for kname in all_kernels:
                if kname not in dim_res:
                    continue
                s = dim_res[kname]
                spd_pt = (pt_us / s["median_us"]
                          if pt_us and s["median_us"] > 0 else "")
                spd_orig = (orig_us / s["median_us"]
                            if orig_us and s["median_us"] > 0 else "")
                writer.writerow([
                    dim_label, n_elements,
                    kname,
                    f"{s['median_us']:.2f}",
                    f"{s['std_us']:.2f}",
                    f"{s['bw_gbs']:.1f}",
                    f"{spd_pt:.2f}" if isinstance(spd_pt, float) else "",
                    f"{spd_orig:.2f}" if isinstance(spd_orig, float) else "",
                ])
    return fpath


# ---------------------------------------------------------------------------
# LaTeX table generation
# ---------------------------------------------------------------------------

# Classify kernels into groups for table presentation
_KERNEL_GROUPS: List[Tuple[str, List[str]]] = [
    ("Baseline", [
        "PyTorch randn",
    ]),
    ("Original Triton", [
        "Orig Philox 4w",
        "Orig Philox 8w",
        "Orig Philox 2x",
        "HiZOO randn",
        "ZO2 Triton",
        "ZO2 CUDA",
    ]),
    ("Custom Triton (ours)", [
        "V2A mulhi-10r 4w",
        "V2A mulhi-10r 8w",
        "V2B mulhi-7r 4w",
        "V2B mulhi-7r 8w",
        "V2C 2x-7r 4w",
        "V2C 2x-7r 8w",
        "V2D tl.randn 4w",
        "V2D tl.randn 8w",
        "V2E auto mulhi-7r",
        "V2F auto tl.randn",
    ]),
    ("Custom CUDA (ours)", [
        "CUDA scalar-10r",
        "CUDA scalar-7r",
        "CUDA vec4-10r",
        "CUDA vec4-7r",
    ]),
]


def _latex_escape(s: str) -> str:
    """Escape characters that are special in LaTeX."""
    return s.replace("_", r"\_").replace("#", r"\#").replace("&", r"\&")


def _fmt_dim(n: int) -> str:
    """Format element count for display (e.g., 1.0M, 25.7M)."""
    if n >= 1_000_000:
        return f"{n / 1e6:.1f}M"
    if n >= 1_000:
        return f"{n / 1e3:.0f}K"
    return str(n)


def generate_latex(
    results: Dict[str, Dict[str, Dict[str, float]]],
    dims: OrderedDict,
    gpu_name: str,
    output_dir: str,
) -> str:
    """Generate a publication-ready LaTeX table.  Returns the file path."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"benchmark_table_{timestamp}.tex"
    fpath = os.path.join(output_dir, fname)

    sorted_dims = sorted(dims.items(), key=lambda kv: kv[1])
    # Only dims with results
    sorted_dims = [(d, n) for d, n in sorted_dims if d in results]

    # Determine which kernels actually ran
    present_kernels: List[str] = []
    for dim_label, _ in sorted_dims:
        for k in results[dim_label]:
            if k not in present_kernels:
                present_kernels.append(k)

    # For each dimension find the fastest custom kernel for bolding
    def _best_custom(dim_label: str) -> Optional[float]:
        best = float("inf")
        for kn in present_kernels:
            if kn in ("PyTorch randn",):
                continue
            val = results[dim_label].get(kn, {}).get("median_us", float("inf"))
            if val < best:
                best = val
        return best if best < float("inf") else None

    n_dims = len(sorted_dims)

    lines: List[str] = []
    L = lines.append

    # ---- Table 1: Median GPU time (us) + speedup vs PyTorch ----
    L(r"% Auto-generated by benchmark_all.py -- do not edit by hand")
    L(r"% GPU: " + gpu_name)
    L(r"% Date: " + datetime.now().isoformat())
    L("")

    # Column spec: group label | kernel name | one col per dimension
    col_spec = "ll" + "r" * n_dims
    L(r"\begin{table}[htbp]")
    L(r"\centering")
    L(r"\caption{Kernel execution time (median, $\mu$s) and speedup vs "
      r"\texttt{torch.randn} across OPT model layer dimensions on "
      + _latex_escape(gpu_name) + r".}")
    L(r"\label{tab:kernel_benchmark}")
    if n_dims > 5:
        L(r"\small")
    if n_dims > 8:
        L(r"\scriptsize")
    L(r"\begin{tabular}{" + col_spec + "}")
    L(r"\toprule")

    # Header row 1: dimension names
    hdr = r"\textbf{Group} & \textbf{Kernel}"
    for dim_label, n in sorted_dims:
        hdr += r" & \textbf{" + _latex_escape(dim_label) + "}"
    hdr += r" \\"
    L(hdr)

    # Header row 2: element counts
    sub_hdr = " & "
    for _, n in sorted_dims:
        sub_hdr += r" & " + _fmt_dim(n)
    sub_hdr += r" \\"
    L(sub_hdr)
    L(r"\midrule")

    # Body: iterate groups
    first_group = True
    for group_name, group_kernels in _KERNEL_GROUPS:
        # Only include kernels that actually ran
        active = [k for k in group_kernels if k in present_kernels]
        if not active:
            continue
        if not first_group:
            L(r"\addlinespace[3pt]")
        first_group = False

        for idx, kname in enumerate(active):
            # Group label only on first row (multirow)
            if idx == 0:
                grp_cell = (r"\multirow{" + str(len(active)) + "}{*}{"
                            + _latex_escape(group_name) + "}")
            else:
                grp_cell = ""

            cells = grp_cell + " & " + _latex_escape(kname)
            for dim_label, _ in sorted_dims:
                s = results[dim_label].get(kname, {})
                med = s.get("median_us", None)
                if med is None:
                    cells += r" & --"
                    continue
                pt_us = results[dim_label].get(
                    "PyTorch randn", {}).get("median_us", None)
                spd = pt_us / med if pt_us and med > 0 else None

                best = _best_custom(dim_label)
                is_best = (best is not None
                           and abs(med - best) < 0.01
                           and kname != "PyTorch randn")

                time_str = f"{med:.1f}"
                if spd is not None and kname != "PyTorch randn":
                    entry = (f"{time_str}"
                             r"\,{\scriptsize(" + f"{spd:.1f}" + r"$\times$)}")
                else:
                    entry = time_str
                if is_best:
                    entry = r"\textbf{" + entry + "}"
                cells += " & " + entry
            cells += r" \\"
            L(cells)

    L(r"\bottomrule")
    L(r"\end{tabular}")
    L(r"\end{table}")
    L("")

    # ---- Table 2: Bandwidth summary at largest dimension ----
    if sorted_dims:
        largest_label, largest_n = sorted_dims[-1]
        if largest_label in results:
            L(r"\begin{table}[htbp]")
            L(r"\centering")
            L(r"\caption{Effective read+write bandwidth (GB/s) at "
              + _latex_escape(largest_label)
              + r" ($n=" + f"{largest_n:,}" + r"$).}")
            L(r"\label{tab:kernel_bandwidth}")
            L(r"\begin{tabular}{llrr}")
            L(r"\toprule")
            L(r"\textbf{Group} & \textbf{Kernel} "
              r"& \textbf{BW (GB/s)} & \textbf{Speedup} \\")
            L(r"\midrule")

            pt_us = results[largest_label].get(
                "PyTorch randn", {}).get("median_us", None)

            first_group = True
            for group_name, group_kernels in _KERNEL_GROUPS:
                active = [k for k in group_kernels
                          if k in results[largest_label]]
                if not active:
                    continue
                if not first_group:
                    L(r"\addlinespace[3pt]")
                first_group = False
                for idx, kname in enumerate(active):
                    s = results[largest_label][kname]
                    bw = s.get("bw_gbs", 0.0)
                    med = s.get("median_us", 0.0)
                    spd_str = ""
                    if pt_us and med > 0 and kname != "PyTorch randn":
                        spd_str = f"{pt_us / med:.2f}$\\times$"
                    grp = (_latex_escape(group_name) if idx == 0 else "")
                    L(f"  {grp} & {_latex_escape(kname)} "
                      f"& {bw:.1f} & {spd_str} \\\\")

            L(r"\bottomrule")
            L(r"\end{tabular}")
            L(r"\end{table}")

    with open(fpath, "w") as f:
        f.write("\n".join(lines) + "\n")
    return fpath


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Unified ZO perturbation kernel benchmark")
    parser.add_argument("--dims", choices=["small", "medium", "large"],
                        default="medium")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verify", action="store_true",
                        help="Run distribution correctness check")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory for CSV/LaTeX output files "
                             "(default: same directory as this script)")
    args = parser.parse_args()

    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Triton version: {triton.__version__}")
    print()

    dims = {"small": DIMS_SMALL, "medium": DIMS_MEDIUM,
            "large": DIMS_LARGE}[args.dims]

    registry = build_kernel_registry(device)
    print(f"Registered {len(registry)} kernels:")
    for name, (_, desc) in registry.items():
        print(f"  {name:25s}  {desc}")
    print()

    # --- Optional correctness check ---
    if args.verify:
        print("=" * 70)
        print("Distribution Correctness Check (n=1M)")
        print("=" * 70)
        n_check = 1_000_000
        for name, (fn, _) in registry.items():
            buf = torch.zeros(n_check, device=device, dtype=torch.float32)
            try:
                fn(buf, args.seed)
                torch.cuda.synchronize()
                check_distribution(buf, name)
            except Exception as e:
                print(f"  [ERR] {name}: {e}")
            del buf
        torch.cuda.empty_cache()
        print()

    # --- Benchmark ---
    print("=" * 70)
    print("Performance Benchmark")
    print("=" * 70)

    # results[dim_label][kernel_name] = {"median_us": ..., ...}
    results: Dict[str, Dict[str, Dict[str, float]]] = {}

    sorted_dims = sorted(dims.items(), key=lambda kv: kv[1])

    for dim_label, n_elements in sorted_dims:
        mem_mb = n_elements * 4 / 1e6
        free_mb = torch.cuda.mem_get_info(device)[0] / 1e6
        if mem_mb > free_mb * 0.8:
            print(f"\n--- {dim_label} (n={n_elements:,}) SKIPPED "
                  f"({mem_mb:.0f}MB > {free_mb:.0f}MB free) ---")
            continue

        print(f"\n--- {dim_label} (n={n_elements:,}, {mem_mb:.1f} MB) ---")
        dim_results = {}

        for name, (fn, _) in registry.items():
            stats = benchmark_kernel(
                fn, n_elements, device, args.seed,
                args.warmup, args.repeat,
            )
            if "error" in stats:
                print(f"  {name:25s}  ERROR: {stats['error']}")
                continue
            dim_results[name] = stats
            print(f"  {name:25s}  gpu {stats['median_us']:>8.1f} us "
                  f"(±{stats['std_us']:>5.1f})  "
                  f"BW {stats['bw_gbs']:>7.1f} GB/s")

        # Speedups vs PyTorch
        pt_us = dim_results.get("PyTorch randn", {}).get("median_us")
        if pt_us and pt_us > 0:
            print(f"  {'--- speedup vs PyTorch ---':37s}")
            for name, stats in dim_results.items():
                if name == "PyTorch randn":
                    continue
                spd = pt_us / stats["median_us"]
                print(f"  {name:25s}  {spd:>8.2f}x")

        # Speedups vs Original Philox
        orig_us = dim_results.get("Orig Philox 4w", {}).get("median_us")
        if orig_us and orig_us > 0:
            print(f"  {'--- speedup vs Orig Philox ---':37s}")
            for name, stats in dim_results.items():
                if name in ("PyTorch randn", "Orig Philox 4w"):
                    continue
                spd = orig_us / stats["median_us"]
                print(f"  {name:25s}  {spd:>8.2f}x")

        results[dim_label] = dim_results
        torch.cuda.empty_cache()

    # --- Summary table ---
    print("\n" + "=" * 70)
    print("SUMMARY TABLE (median GPU time in us)")
    print("=" * 70)

    k_names = list(registry.keys())
    # Header
    hdr = f"{'Dimension':15s} {'n':>12s}"
    for name in k_names:
        short = name[:12]
        hdr += f" {short:>12s}"
    hdr += f" {'Best spdup':>10s}"
    print(hdr)
    print("-" * len(hdr))

    for dim_label, n_elements in sorted_dims:
        if dim_label not in results:
            continue
        row = f"{dim_label:15s} {n_elements:>12,}"
        pt_us = results[dim_label].get("PyTorch randn", {}).get("median_us",
                                                                  float("inf"))
        best_spd = 0.0
        for name in k_names:
            stats = results[dim_label].get(name, {})
            val = stats.get("median_us", float("nan"))
            row += f" {val:>12.1f}"
            if name != "PyTorch randn" and val > 0 and not math.isnan(val):
                spd = pt_us / val
                best_spd = max(best_spd, spd)
        if best_spd > 0:
            row += f" {best_spd:>9.2f}x"
        else:
            row += f" {'N/A':>10s}"
        print(row)

    # --- Bandwidth summary at largest dim ---
    if sorted_dims and sorted_dims[-1][0] in results:
        largest = sorted_dims[-1][0]
        print(f"\nEffective Read+Write Bandwidth at {largest}:")
        for name, stats in results[largest].items():
            bw = stats.get("bw_gbs", 0.0)
            print(f"  {name:25s} {bw:>8.1f} GB/s")

    # --- Export CSV and LaTeX ---
    if results:
        out_dir = args.output_dir or str(Path(__file__).resolve().parent)
        os.makedirs(out_dir, exist_ok=True)
        gpu_name = torch.cuda.get_device_name(0)

        csv_path = export_csv(results, dims, gpu_name, out_dir)
        print(f"\nCSV results saved to:  {csv_path}")

        tex_path = generate_latex(results, dims, gpu_name, out_dir)
        print(f"LaTeX table saved to:  {tex_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
