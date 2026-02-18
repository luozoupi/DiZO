#!/usr/bin/env python3
"""
CPU-GPU Dispatch Overhead Benchmark
====================================

Quantifies the per-tensor dispatch overhead in zeroth-order perturbation
without loading any model.  Parameter shapes are read from text files.

The original trainer.py loops over every parameter (388-644 tensors),
calling torch.normal() + add for each one.  This creates:
  - Memory allocation overhead (one temp tensor per parameter)
  - Many small kernel launches that underutilize GPU bandwidth
  - Python loop + PyTorch dispatcher overhead between launches

This script decomposes the overhead by measuring five approaches:
  A. Per-tensor RNG only   — loop calling torch.normal() per tensor (no add)
  B. Per-tensor add only   — pre-allocated z, loop calling add_() per tensor
  C. Per-tensor full       — RNG + add per tensor (the trainer.py baseline)
  D. Flat buf + PyTorch    — single torch.normal() + single add_() on flat buf
  E. Flat buf + fused      — single custom CUDA/Triton kernel on flat buf

Comparing C vs D (same total work, different dispatch pattern) isolates the
per-tensor dispatch overhead.  Comparing A vs D quantifies the RNG dispatch
overhead specifically.

No model loading required.

Usage:
    CUDA_VISIBLE_DEVICES=0 python benchmark_bubble.py
    CUDA_VISIBLE_DEVICES=0 python benchmark_bubble.py --models opt-350m opt-13b
    CUDA_VISIBLE_DEVICES=0 python benchmark_bubble.py --repeat 30
"""

import argparse
import csv
import os
import re
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

# Import fused kernels
HAS_V2_CUDA = False
try:
    import philox_v2_cuda_ext as _v2_cuda
    HAS_V2_CUDA = True
except ImportError:
    try:
        from torch.utils.cpp_extension import load as _jit_load
        _v2_cuda_src = SCRIPT_DIR / "philox_v2_cuda.cu"
        if _v2_cuda_src.exists():
            _cap = torch.cuda.get_device_capability(0)
            _sm = f"{_cap[0]}{_cap[1]}0" if _cap[1] >= 10 else f"{_cap[0]}{_cap[1]}"
            _sm_flag = f"-gencode=arch=compute_{_sm},code=sm_{_sm}"
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
        print(f"[skip] V2 CUDA: {e}")

HAS_V2_TRITON = False
try:
    from kernel_optimization.philox_v2 import perturb_v2e
    HAS_V2_TRITON = True
except Exception as e:
    print(f"[skip] V2 Triton: {e}")


# ---------------------------------------------------------------------------
# Parse parameter shapes from text files
# ---------------------------------------------------------------------------

SHAPE_DIR = REPO_ROOT / "DiZO" / "large_models" / "cuda_kernels"

SHAPE_FILES = OrderedDict([
    ("opt-350m", SHAPE_DIR / "opt-350m_parameter_shapes.txt"),
    ("opt-2.7b", SHAPE_DIR / "opt-2_7b_parameter_shapes.txt"),
    ("opt-6.7b", SHAPE_DIR / "opt-6_7b_parameter_shapes.txt"),
    ("opt-13b",  SHAPE_DIR / "opt-13b_parameter_shapes.txt"),
])


def parse_shapes(filepath: Path) -> List[Tuple[str, Tuple[int, ...]]]:
    """Parse parameter shapes from a shape text file."""
    shapes = []
    with open(filepath, "r") as f:
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
            m = re.match(
                r"^\d+\s+(\S+)\s+\(([^)]+)\)\s+", line,
            )
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
# Generic timing harness
# ---------------------------------------------------------------------------

def _timed_run(fn, n_warmup: int, n_repeat: int) -> Dict[str, List[float]]:
    """Run fn() repeatedly, capturing wall-clock and GPU stream time."""
    wall_times = []
    gpu_times = []

    for rep in range(-n_warmup, n_repeat):
        torch.cuda.synchronize()
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)

        start_ev.record()
        t0 = time.perf_counter()

        fn(rep if rep >= 0 else 0)

        end_ev.record()
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        if rep >= 0:
            wall_times.append((t1 - t0) * 1e3)
            gpu_times.append(start_ev.elapsed_time(end_ev))

    return {"wall_ms": wall_times, "gpu_ms": gpu_times}


# ---------------------------------------------------------------------------
# Benchmark approaches
# ---------------------------------------------------------------------------

def bench_per_tensor_rng_only(
    shapes, device, dtype, seed, n_warmup, n_repeat,
) -> Dict[str, List[float]]:
    """(A) Per-tensor RNG only: loop calling torch.normal() per tensor, no add.

    Isolates the cost of per-tensor RNG dispatch + temp allocation.
    """
    # Pre-allocate destination tensors
    tensors = [torch.randn(shape, device=device, dtype=dtype)
               for _, shape in shapes]
    torch.cuda.synchronize()

    def fn(rep):
        torch.manual_seed(seed + rep)
        for param in tensors:
            z = torch.normal(mean=0, std=1, size=param.size(),
                             device=param.device, dtype=param.dtype)

    return _timed_run(fn, n_warmup, n_repeat)


def bench_per_tensor_add_only(
    shapes, device, dtype, seed, n_warmup, n_repeat,
) -> Dict[str, List[float]]:
    """(B) Per-tensor add only: pre-allocated z, loop calling add_() per tensor.

    Isolates the cost of per-tensor perturbation dispatch.
    """
    tensors = [torch.randn(shape, device=device, dtype=dtype)
               for _, shape in shapes]
    # Pre-generate z for each tensor (reused every rep)
    z_list = [torch.randn(shape, device=device, dtype=dtype)
              for _, shape in shapes]
    torch.cuda.synchronize()

    def fn(rep):
        for param, z in zip(tensors, z_list):
            param.add_(z, alpha=1e-3)

    return _timed_run(fn, n_warmup, n_repeat)


def bench_per_tensor_full(
    shapes, device, dtype, seed, n_warmup, n_repeat,
) -> Dict[str, List[float]]:
    """(C) Per-tensor full: torch.normal() + add per tensor.

    This is the exact trainer.py zo_perturb_parameters pattern.
    """
    tensors = [torch.randn(shape, device=device, dtype=dtype)
               for _, shape in shapes]
    torch.cuda.synchronize()

    def fn(rep):
        torch.manual_seed(seed + rep)
        for param in tensors:
            z = torch.normal(mean=0, std=1, size=param.size(),
                             device=param.device, dtype=param.dtype)
            param.data = param.data + 1.0 * z * 1e-3

    return _timed_run(fn, n_warmup, n_repeat)


def bench_flat_pytorch(
    shapes, device, dtype, seed, n_warmup, n_repeat,
) -> Dict[str, List[float]]:
    """(D) Flat buffer + PyTorch RNG: single normal() + single add_().

    Same total work as (C) but 2 kernel launches instead of O(n_params).
    """
    total_n = sum(int(np.prod(s)) for _, s in shapes)
    buf = torch.randn(total_n, device=device, dtype=dtype)
    torch.cuda.synchronize()

    def fn(rep):
        torch.manual_seed(seed + rep)
        z = torch.normal(mean=0, std=1, size=buf.size(),
                         device=buf.device, dtype=buf.dtype)
        buf.add_(z, alpha=1e-3)

    return _timed_run(fn, n_warmup, n_repeat)


def bench_flat_pytorch_rng_only(
    shapes, device, dtype, seed, n_warmup, n_repeat,
) -> Dict[str, List[float]]:
    """(D') Flat buffer + PyTorch in-place normal_(): RNG only, no add.

    Pure RNG baseline for flat buffer — single kernel launch, write-only.
    Compare with (A) to isolate per-tensor RNG dispatch overhead.
    """
    total_n = sum(int(np.prod(s)) for _, s in shapes)
    buf = torch.empty(total_n, device=device, dtype=dtype)
    torch.cuda.synchronize()

    def fn(rep):
        torch.manual_seed(seed + rep)
        buf.normal_()

    return _timed_run(fn, n_warmup, n_repeat)


def bench_fused_kernel(
    shapes, device, dtype, seed, n_warmup, n_repeat,
) -> Dict[str, List[float]]:
    """(E) Flat buffer + fused custom kernel: single kernel launch.

    Combines RNG + perturbation in one kernel.
    """
    total_n = sum(int(np.prod(s)) for _, s in shapes)
    buf = torch.randn(total_n, device=device, dtype=torch.float32)
    torch.cuda.synchronize()

    if HAS_V2_CUDA:
        def fused_fn(b, s):
            _v2_cuda.perturb_vec4_7r(b, s, 1e-3)
        kernel_name = "CUDA vec4-7r"
    elif HAS_V2_TRITON:
        def fused_fn(b, s):
            perturb_v2e(b, s, 1e-3)
        kernel_name = "Triton V2E"
    else:
        return None

    def fn(rep):
        fused_fn(buf, seed + rep)

    result = _timed_run(fn, n_warmup, n_repeat)
    result["kernel"] = kernel_name
    return result


def bench_per_tensor_cuda_fused(
    shapes, device, dtype, seed, n_warmup, n_repeat,
) -> Dict[str, List[float]]:
    """(F) Per-tensor loop + CUDA fused kernel per tensor.

    Still loops over each parameter, but uses the fused CUDA kernel
    (1 launch per param, 0 temp allocations) instead of PyTorch's
    torch.normal() + add (3 launches + 1 alloc per param).
    Isolates the fused-kernel benefit at per-tensor granularity.
    """
    if not HAS_V2_CUDA:
        return None
    # CUDA fused kernels require fp32
    tensors = [torch.randn(shape, device=device, dtype=torch.float32)
               for _, shape in shapes]
    torch.cuda.synchronize()

    def fn(rep):
        for i, param in enumerate(tensors):
            _v2_cuda.perturb_vec4_7r(param.view(-1), seed + rep + i, 1e-3)

    result = _timed_run(fn, n_warmup, n_repeat)
    result["kernel"] = "CUDA vec4-7r"
    return result


def bench_per_tensor_triton_fused(
    shapes, device, dtype, seed, n_warmup, n_repeat,
) -> Dict[str, List[float]]:
    """(G) Per-tensor loop + Triton fused kernel per tensor.

    Still loops over each parameter, but uses the fused Triton kernel
    (1 launch per param, 0 temp allocations) instead of PyTorch's
    torch.normal() + add (3 launches + 1 alloc per param).
    """
    if not HAS_V2_TRITON:
        return None
    # Triton fused kernels require fp32
    tensors = [torch.randn(shape, device=device, dtype=torch.float32)
               for _, shape in shapes]
    torch.cuda.synchronize()

    def fn(rep):
        for i, param in enumerate(tensors):
            perturb_v2e(param.view(-1), seed + rep + i, 1e-3)

    result = _timed_run(fn, n_warmup, n_repeat)
    result["kernel"] = "Triton V2E"
    return result


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_results(model_name, n_params, total_n, results_dict, mode="perturb"):
    """Print decomposed results for one model."""
    mode_label = "RNG + perturbation" if mode == "perturb" else "RNG only"
    print(f"\n{'=' * 90}")
    print(f"  {model_name}  ({n_params} params, "
          f"{total_n:,} elements, "
          f"{total_n * 2 / 1e6:.1f} MB fp16)  [{mode_label}]")
    print(f"{'=' * 90}")

    header = (f"  {'#':<4s}{'Approach':<42s} "
              f"{'Stream (ms)':>12s} {'Wall (ms)':>10s} {'Speedup':>8s}")
    print(header)
    print(f"  {'-' * 76}")

    if mode == "perturb":
        baseline_stream = np.median(results_dict["C_full"]["gpu_ms"])

        labels = [
            ("A", "Per-tensor RNG only (torch.normal loop)", "A_rng_only"),
            ("B", "Per-tensor add only (add_ loop)", "B_add_only"),
            ("C", "Per-tensor full (trainer.py baseline)", "C_full"),
        ]
        if results_dict.get("F_pt_cuda") is not None:
            kname = results_dict["F_pt_cuda"].get("kernel", "CUDA")
            labels.append(("F", f"Per-tensor + {kname} (fused loop)", "F_pt_cuda"))
        if results_dict.get("G_pt_triton") is not None:
            kname = results_dict["G_pt_triton"].get("kernel", "Triton")
            labels.append(("G", f"Per-tensor + {kname} (fused loop)", "G_pt_triton"))
        labels.append(("D", "Flat buf + PyTorch RNG (2 launches)", "D_flat_pytorch"))
        if results_dict.get("E_fused") is not None:
            kname = results_dict["E_fused"].get("kernel", "fused")
            labels.append(("E", f"Flat buf + {kname} (1 launch)", "E_fused"))
    else:  # rng mode
        baseline_stream = np.median(results_dict["A_rng_only"]["gpu_ms"])

        labels = [
            ("A", "Per-tensor RNG only (baseline)", "A_rng_only"),
        ]
        if results_dict.get("F_pt_cuda") is not None:
            kname = results_dict["F_pt_cuda"].get("kernel", "CUDA")
            labels.append(("F", f"Per-tensor + {kname} (fused*)", "F_pt_cuda"))
        if results_dict.get("G_pt_triton") is not None:
            kname = results_dict["G_pt_triton"].get("kernel", "Triton")
            labels.append(("G", f"Per-tensor + {kname} (fused*)", "G_pt_triton"))
        labels.append(("D'", "Flat buf + PyTorch RNG (normal_)", "D_flat_rng"))
        if results_dict.get("E_fused") is not None:
            kname = results_dict["E_fused"].get("kernel", "fused")
            labels.append(("E", f"Flat buf + {kname} (fused*)", "E_fused"))

    for tag, label, key in labels:
        data = results_dict[key]
        stream = np.median(data["gpu_ms"])
        wall = np.median(data["wall_ms"])
        speedup = baseline_stream / stream if stream > 0 else 0
        print(f"  ({tag}) {label:<40s} {stream:>12.2f} {wall:>10.2f} "
              f"{speedup:>7.2f}x")

    # Decomposition analysis
    print()
    if mode == "perturb":
        A_stream = np.median(results_dict["A_rng_only"]["gpu_ms"])
        B_stream = np.median(results_dict["B_add_only"]["gpu_ms"])
        C_stream = np.median(results_dict["C_full"]["gpu_ms"])
        D_stream = np.median(results_dict["D_flat_pytorch"]["gpu_ms"])

        dispatch_overhead = C_stream - D_stream
        rng_dispatch_overhead = A_stream - D_stream

        print(f"  Overhead decomposition (vs flat buffer baseline D):")
        print(f"    Total per-tensor dispatch overhead (C-D):   "
              f"{dispatch_overhead:>8.2f} ms  "
              f"({100 * dispatch_overhead / C_stream:.1f}% of baseline)")
        print(f"    RNG dispatch contribution (A-D):            "
              f"{rng_dispatch_overhead:>8.2f} ms  "
              f"({100 * rng_dispatch_overhead / C_stream:.1f}% of baseline)")
        print(f"    Per-tensor add-only loop time (B):          "
              f"{B_stream:>8.2f} ms")
        print(f"    Flat buf RNG+add time (D):                  "
              f"{D_stream:>8.2f} ms")

        if results_dict.get("E_fused") is not None:
            E_stream = np.median(results_dict["E_fused"]["gpu_ms"])
            rng_impl_saving = D_stream - E_stream
            total_saving = C_stream - E_stream
            print(f"    Fused kernel saving over flat PT (D-E):     "
                  f"{rng_impl_saving:>8.2f} ms  "
                  f"({100 * rng_impl_saving / C_stream:.1f}% of baseline)")
            print(f"    Total end-to-end saving (C-E):              "
                  f"{total_saving:>8.2f} ms  "
                  f"({100 * total_saving / C_stream:.1f}% of baseline)")

        # Per-tensor fused analysis
        for key, label in [("F_pt_cuda", "CUDA fused"),
                           ("G_pt_triton", "Triton fused")]:
            if results_dict.get(key) is not None:
                F_stream = np.median(results_dict[key]["gpu_ms"])
                fused_vs_pt = C_stream - F_stream
                loop_overhead = F_stream - E_stream if results_dict.get("E_fused") else F_stream - D_stream
                ref_label = "E" if results_dict.get("E_fused") else "D"
                print(f"    Per-tensor {label} saving vs baseline (C-{key[0].upper()}): "
                      f"{fused_vs_pt:>6.2f} ms  "
                      f"({100 * fused_vs_pt / C_stream:.1f}% of baseline)")
                print(f"    Remaining loop overhead ({key[0].upper()}-{ref_label}):        "
                      f"{loop_overhead:>6.2f} ms  "
                      f"({100 * loop_overhead / C_stream:.1f}% of baseline)")

        # Kernel launch counts
        print()
        print(f"  Kernel launches:")
        print(f"    Per-tensor RNG:   ~{n_params} launches (torch.normal)")
        print(f"    Per-tensor add:   ~{n_params} launches (add_)")
        print(f"    Per-tensor alloc: ~{n_params} temp tensors (cudaMalloc/pool)")
        print(f"    Per-tensor full:  ~{n_params * 3}+ launches total")
        print(f"    Per-tensor fused: ~{n_params} launches (1 per param, 0 allocs)")
        print(f"    Flat buf PyTorch: 2 launches (normal + add)")
        print(f"    Fused kernel:     1 launch")
    else:  # rng mode
        A_stream = np.median(results_dict["A_rng_only"]["gpu_ms"])
        D_stream = np.median(results_dict["D_flat_rng"]["gpu_ms"])

        dispatch_overhead = A_stream - D_stream

        print(f"  RNG dispatch overhead decomposition (vs flat buffer D'):")
        print(f"    Per-tensor RNG dispatch overhead (A-D'):    "
              f"{dispatch_overhead:>8.2f} ms  "
              f"({100 * dispatch_overhead / A_stream:.1f}% of baseline A)")
        print(f"    Per-tensor RNG loop time (A):               "
              f"{A_stream:>8.2f} ms")
        print(f"    Flat buf RNG time (D'):                     "
              f"{D_stream:>8.2f} ms")

        if results_dict.get("E_fused") is not None:
            E_stream = np.median(results_dict["E_fused"]["gpu_ms"])
            print(f"    Flat buf fused kernel time (E):             "
                  f"{E_stream:>8.2f} ms  "
                  f"(includes inherent load+add)")
            fused_vs_pt = D_stream - E_stream
            print(f"    Fused vs PyTorch flat RNG (D'-E):           "
                  f"{fused_vs_pt:>8.2f} ms  "
                  f"({'faster' if fused_vs_pt > 0 else 'slower'})")

        # Per-tensor fused analysis
        for key, label in [("F_pt_cuda", "CUDA fused"),
                           ("G_pt_triton", "Triton fused")]:
            if results_dict.get(key) is not None:
                F_stream = np.median(results_dict[key]["gpu_ms"])
                fused_vs_pt = A_stream - F_stream
                print(f"    Per-tensor {label} vs baseline (A-{key[0].upper()}):  "
                      f"{fused_vs_pt:>6.2f} ms  "
                      f"({100 * fused_vs_pt / A_stream:.1f}% of baseline A)")

        print()
        print(f"  * Fused kernels inherently include perturbation (load+add+store).")
        print(f"    They cannot isolate pure RNG, but the add on zero/existing data")
        print(f"    is negligible vs RNG generation cost at large tensor sizes.")
        print()
        print(f"  Kernel launches:")
        print(f"    Per-tensor RNG:   ~{n_params} launches (torch.normal)")
        print(f"    Per-tensor alloc: ~{n_params} temp tensors")
        print(f"    Per-tensor fused: ~{n_params} launches (1 per param, 0 allocs)")
        print(f"    Flat buf PyTorch: 1 launch (normal_)")
        print(f"    Fused kernel:     1 launch")


def export_csv(all_results, output_dir, mode="perturb"):
    """Write results to CSV."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"benchmark_bubble_{mode}_{timestamp}.csv"
    fpath = os.path.join(output_dir, fname)
    gpu_name = torch.cuda.get_device_name(0)

    approach_keys = [
        ("A_rng_only", "per_tensor_rng_only"),
        ("B_add_only", "per_tensor_add_only"),
        ("C_full", "per_tensor_full"),
        ("F_pt_cuda", "per_tensor_cuda_fused"),
        ("G_pt_triton", "per_tensor_triton_fused"),
        ("D_flat_pytorch", "flat_pytorch"),
        ("D_flat_rng", "flat_pytorch_rng_only"),
        ("E_fused", "fused_kernel"),
    ]

    with open(fpath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["# GPU", gpu_name])
        writer.writerow(["# Date", datetime.now().isoformat()])
        writer.writerow(["# Mode", mode])
        writer.writerow([])
        writer.writerow([
            "model", "n_params", "total_elements",
            "approach",
            "stream_median_ms", "stream_std_ms",
            "wall_median_ms", "wall_std_ms",
            "speedup_vs_baseline",
        ])

        for model_name, mdata in all_results.items():
            n_params = mdata["n_params"]
            total_n = mdata["total_n"]
            rd = mdata["results"]
            if mode == "perturb":
                baseline_stream = np.median(rd["C_full"]["gpu_ms"])
            else:
                baseline_stream = np.median(rd["A_rng_only"]["gpu_ms"])

            for key, csv_name in approach_keys:
                if key not in rd or rd[key] is None:
                    continue
                data = rd[key]
                stream = np.median(data["gpu_ms"])
                stream_std = np.std(data["gpu_ms"])
                wall = np.median(data["wall_ms"])
                wall_std = np.std(data["wall_ms"])
                speedup = baseline_stream / stream if stream > 0 else 0

                writer.writerow([
                    model_name, n_params, total_n,
                    csv_name,
                    f"{stream:.3f}", f"{stream_std:.3f}",
                    f"{wall:.3f}", f"{wall_std:.3f}",
                    f"{speedup:.2f}",
                ])

    return fpath


def generate_latex(all_results, output_dir, mode="perturb"):
    """Generate LaTeX table."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"benchmark_bubble_{mode}_{timestamp}.tex"
    fpath = os.path.join(output_dir, fname)
    gpu_name = torch.cuda.get_device_name(0)

    lines = []
    L = lines.append

    L(r"% Auto-generated by benchmark_bubble.py")
    L(r"% GPU: " + gpu_name)
    L(r"% Date: " + datetime.now().isoformat())
    L(r"% Mode: " + mode)
    L("")
    L(r"\begin{table}[htbp]")
    L(r"\centering")
    if mode == "perturb":
        L(r"\caption{Per-tensor dispatch overhead in ZO perturbation. "
          r"Stream time is GPU-side elapsed time (includes inter-kernel idle gaps). "
          r"Comparing (C) trainer.py baseline vs (D) flat buffer isolates the "
          r"dispatch overhead; (A) and (B) decompose RNG vs add contributions. "
          r"Tested on " + gpu_name.replace("_", r"\_") + r".}")
        L(r"\label{tab:dispatch_overhead_perturb}")
    else:
        L(r"\caption{Per-tensor RNG dispatch overhead (RNG only, no perturbation). "
          r"Baseline (A) is the trainer.py pattern of per-tensor \texttt{torch.normal()} "
          r"calls; (D') is a single flat \texttt{normal\_()} call. "
          r"Fused kernels (*) inherently include load+add+store. "
          r"Tested on " + gpu_name.replace("_", r"\_") + r".}")
        L(r"\label{tab:dispatch_overhead_rng}")
    L(r"\small")
    L(r"\begin{tabular}{llrrr}")
    L(r"\toprule")
    L(r"\textbf{Model} & \textbf{Approach} "
      r"& \textbf{Stream (ms)} & \textbf{Overhead (ms)} "
      r"& \textbf{Speedup} \\")
    L(r"\midrule")

    first_model = True
    for model_name, mdata in all_results.items():
        if not first_model:
            L(r"\addlinespace[4pt]")
        first_model = False

        rd = mdata["results"]
        if mode == "perturb":
            baseline_stream = np.median(rd["C_full"]["gpu_ms"])
            ref_stream = np.median(rd["D_flat_pytorch"]["gpu_ms"])

            approach_labels = [
                ("A", "Per-tensor RNG only", "A_rng_only"),
                ("B", "Per-tensor add only", "B_add_only"),
                ("C", "Per-tensor full (baseline)", "C_full"),
            ]
            if rd.get("F_pt_cuda") is not None:
                kname = rd["F_pt_cuda"].get("kernel", "CUDA")
                approach_labels.append(("F", f"Per-tensor + {kname}", "F_pt_cuda"))
            if rd.get("G_pt_triton") is not None:
                kname = rd["G_pt_triton"].get("kernel", "Triton")
                approach_labels.append(("G", f"Per-tensor + {kname}", "G_pt_triton"))
            approach_labels.append(("D", "Flat buf + PyTorch", "D_flat_pytorch"))
            if rd.get("E_fused") is not None:
                kname = rd["E_fused"].get("kernel", "fused")
                approach_labels.append(("E", f"Flat buf + {kname}", "E_fused"))

            fastest_tag = "E" if rd.get("E_fused") is not None else "D"
        else:  # rng mode
            baseline_stream = np.median(rd["A_rng_only"]["gpu_ms"])
            ref_stream = np.median(rd["D_flat_rng"]["gpu_ms"])

            approach_labels = [
                ("A", "Per-tensor RNG only (baseline)", "A_rng_only"),
            ]
            if rd.get("F_pt_cuda") is not None:
                kname = rd["F_pt_cuda"].get("kernel", "CUDA")
                approach_labels.append(("F", f"Per-tensor + {kname}*", "F_pt_cuda"))
            if rd.get("G_pt_triton") is not None:
                kname = rd["G_pt_triton"].get("kernel", "Triton")
                approach_labels.append(("G", f"Per-tensor + {kname}*", "G_pt_triton"))
            approach_labels.append(("D'", "Flat buf + PyTorch RNG", "D_flat_rng"))
            if rd.get("E_fused") is not None:
                kname = rd["E_fused"].get("kernel", "fused")
                approach_labels.append(("E", f"Flat buf + {kname}*", "E_fused"))

            fastest_tag = "E" if rd.get("E_fused") is not None else "D'"

        n_rows = len(approach_labels)
        model_esc = model_name.replace("_", r"\_")

        for idx, (tag, label, key) in enumerate(approach_labels):
            data = rd[key]
            stream = np.median(data["gpu_ms"])
            overhead = stream - ref_stream
            speedup = baseline_stream / stream if stream > 0 else 0

            if idx == 0:
                model_cell = (r"\multirow{" + str(n_rows) + "}{*}{"
                              + model_esc + "}")
            else:
                model_cell = ""

            label_esc = label.replace("_", r"\_")
            spd_str = f"{speedup:.1f}$\\times$"
            oh_str = f"+{overhead:.1f}" if overhead > 0.05 else "---"

            # Bold the fastest
            is_fastest = tag == fastest_tag
            if is_fastest:
                entry = (f"{model_cell} & \\textbf{{({tag}) {label_esc}}} "
                         f"& \\textbf{{{stream:.1f}}} "
                         f"& \\textbf{{{oh_str}}} "
                         f"& \\textbf{{{spd_str}}} \\\\")
            else:
                entry = (f"{model_cell} & ({tag}) {label_esc} "
                         f"& {stream:.1f} & {oh_str} "
                         f"& {spd_str} \\\\")
            L(entry)

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
        description="CPU-GPU dispatch overhead benchmark for ZO perturbation")
    parser.add_argument("--models", nargs="+",
                        default=list(SHAPE_FILES.keys()),
                        choices=list(SHAPE_FILES.keys()),
                        help="Which model shapes to benchmark")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16",
                        help="Parameter dtype (default: fp16, matching trainer)")
    parser.add_argument("--mode", choices=["perturb", "rng"], default="perturb",
                        help="Benchmark mode: 'perturb' = RNG+perturbation "
                             "(default), 'rng' = RNG-only dispatch overhead")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda")
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Dtype:  {dtype}")
    print(f"Mode:   {args.mode} ({'RNG + perturbation' if args.mode == 'perturb' else 'RNG only'})")
    print(f"Fused kernel available: ", end="")
    if HAS_V2_CUDA:
        print("CUDA vec4-7r")
    elif HAS_V2_TRITON:
        print("Triton V2E")
    else:
        print("None (only PyTorch baselines)")
    print()

    all_results = OrderedDict()

    for model_name in args.models:
        shape_file = SHAPE_FILES[model_name]
        if not shape_file.exists():
            print(f"[skip] {model_name}: shape file not found at {shape_file}")
            continue

        shapes = parse_shapes(shape_file)
        n_params = len(shapes)
        total_n = sum(int(np.prod(s)) for _, s in shapes)

        print(f"\n>>> {model_name}: {n_params} parameters, "
              f"{total_n:,} elements, "
              f"{total_n * (2 if dtype == torch.float16 else 4) / 1e6:.1f} MB")

        rd = {}

        print(f"  (A) Per-tensor RNG only ...")
        rd["A_rng_only"] = bench_per_tensor_rng_only(
            shapes, device, dtype, args.seed, args.warmup, args.repeat)

        if args.mode == "perturb":
            print(f"  (B) Per-tensor add only ...")
            rd["B_add_only"] = bench_per_tensor_add_only(
                shapes, device, dtype, args.seed, args.warmup, args.repeat)

            print(f"  (C) Per-tensor full (trainer.py) ...")
            rd["C_full"] = bench_per_tensor_full(
                shapes, device, dtype, args.seed, args.warmup, args.repeat)

            print(f"  (D) Flat buf + PyTorch RNG+add ...")
            rd["D_flat_pytorch"] = bench_flat_pytorch(
                shapes, device, dtype, args.seed, args.warmup, args.repeat)
        else:  # rng mode
            print(f"  (D') Flat buf + PyTorch RNG only ...")
            rd["D_flat_rng"] = bench_flat_pytorch_rng_only(
                shapes, device, dtype, args.seed, args.warmup, args.repeat)

        if HAS_V2_CUDA or HAS_V2_TRITON:
            print(f"  (E) Flat buf + fused kernel ...")
            rd["E_fused"] = bench_fused_kernel(
                shapes, device, torch.float32, args.seed,
                args.warmup, args.repeat)

        if HAS_V2_CUDA:
            print(f"  (F) Per-tensor + CUDA fused ...")
            rd["F_pt_cuda"] = bench_per_tensor_cuda_fused(
                shapes, device, dtype, args.seed, args.warmup, args.repeat)

        if HAS_V2_TRITON:
            print(f"  (G) Per-tensor + Triton fused ...")
            rd["G_pt_triton"] = bench_per_tensor_triton_fused(
                shapes, device, dtype, args.seed, args.warmup, args.repeat)

        all_results[model_name] = {
            "n_params": n_params,
            "total_n": total_n,
            "results": rd,
        }

        print_results(model_name, n_params, total_n, rd, args.mode)
        torch.cuda.empty_cache()

    # Export
    if all_results:
        out_dir = args.output_dir or str(SCRIPT_DIR)
        os.makedirs(out_dir, exist_ok=True)

        csv_path = export_csv(all_results, out_dir, args.mode)
        print(f"\nCSV saved to: {csv_path}")

        tex_path = generate_latex(all_results, out_dir, args.mode)
        print(f"LaTeX saved to: {tex_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
