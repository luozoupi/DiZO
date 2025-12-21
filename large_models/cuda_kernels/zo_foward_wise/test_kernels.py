"""
Test script for DiZO fused kernels.

Tests numerical correctness and measures performance.
"""

import torch
import time
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from param_utils import ParameterFlattener
from dizo_fused_kernels import (
    fused_compute_norms,
    fused_apply_constraints,
    fused_reverse_constraints,
    fused_perturb_gamma,
    fused_update_gamma,
)


def test_norm_computation():
    """Test fused norm computation."""
    print("Testing fused norm computation...")
    
    # Use the first available CUDA device (respects CUDA_VISIBLE_DEVICES)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32
    
    # Create test data
    num_params = 10
    sizes = torch.randint(1000, 10000, (num_params,), device=device)
    total_size = sizes.sum().item()
    
    param_flat = torch.randn(total_size, device=device, dtype=dtype)
    anchor_flat = torch.randn(total_size, device=device, dtype=dtype)
    
    offsets = torch.zeros(num_params, device=device, dtype=torch.long)
    offsets[1:] = torch.cumsum(sizes[:-1], dim=0)
    
    # Compute norms with fused kernel
    norms_fused = fused_compute_norms(param_flat, anchor_flat, offsets, sizes)
    
    # Compute norms with PyTorch (reference)
    norms_ref = []
    for i in range(num_params):
        start = offsets[i].item()
        end = start + sizes[i].item()
        diff = param_flat[start:end] - anchor_flat[start:end]
        norms_ref.append(torch.norm(diff).item())
    norms_ref = torch.tensor(norms_ref, device=device)
    
    # Compare
    max_diff = torch.max(torch.abs(norms_fused - norms_ref)).item()
    print(f"  Max difference: {max_diff:.2e}")
    assert max_diff < 1e-4, f"Norm computation failed: {max_diff}"
    print("  ✓ Passed")
    
    # Benchmark
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        norms_fused = fused_compute_norms(param_flat, anchor_flat, offsets, sizes)
    torch.cuda.synchronize()
    elapsed = (time.time() - start) / 100 * 1000
    print(f"  Time: {elapsed:.3f} ms per call")


def test_constraint_application():
    """Test fused constraint application."""
    print("\nTesting fused constraint application...")
    
    # Use the first available CUDA device (respects CUDA_VISIBLE_DEVICES)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32
    
    # Create test data
    num_params = 10
    sizes = torch.randint(1000, 10000, (num_params,), device=device)
    total_size = sizes.sum().item()
    
    param_flat = torch.randn(total_size, device=device, dtype=dtype)
    anchor_flat = torch.randn(total_size, device=device, dtype=dtype)
    param_flat_ref = param_flat.clone()
    
    offsets = torch.zeros(num_params, device=device, dtype=torch.long)
    offsets[1:] = torch.cumsum(sizes[:-1], dim=0)
    
    # Compute norms
    norms = fused_compute_norms(param_flat, anchor_flat, offsets, sizes)
    
    # Create constraints
    constraints = torch.rand(num_params, device=device, dtype=dtype) * 0.1
    
    # Apply constraints with fused kernel
    fused_apply_constraints(
        param_flat,
        anchor_flat,
        offsets,
        sizes,
        constraints,
        norms,
        eps=1e-8,
    )
    
    # Apply constraints with PyTorch (reference)
    for i in range(num_params):
        start = offsets[i].item()
        end = start + sizes[i].item()
        diff = param_flat_ref[start:end] - anchor_flat[start:end]
        norm = torch.norm(diff)
        alpha = constraints[i] / (norm + 1e-8)
        param_flat_ref[start:end] = anchor_flat[start:end] + diff * alpha
    
    # Compare
    max_diff = torch.max(torch.abs(param_flat - param_flat_ref)).item()
    print(f"  Max difference: {max_diff:.2e}")
    assert max_diff < 1e-4, f"Constraint application failed: {max_diff}"
    print("  ✓ Passed")


def test_constraint_reversal():
    """Test fused constraint reversal."""
    print("\nTesting fused constraint reversal...")
    
    # Use the first available CUDA device (respects CUDA_VISIBLE_DEVICES)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32
    
    # Create test data
    num_params = 10
    sizes = torch.randint(1000, 10000, (num_params,), device=device)
    total_size = sizes.sum().item()
    
    param_flat = torch.randn(total_size, device=device, dtype=dtype)
    anchor_flat = torch.randn(total_size, device=device, dtype=dtype)
    param_flat_ref = param_flat.clone()
    
    offsets = torch.zeros(num_params, device=device, dtype=torch.long)
    offsets[1:] = torch.cumsum(sizes[:-1], dim=0)
    
    # Compute norms and alphas
    norms = fused_compute_norms(param_flat, anchor_flat, offsets, sizes)
    constraints = torch.rand(num_params, device=device, dtype=dtype) * 0.1
    alphas = constraints / (norms + 1e-8)
    
    # Apply constraints first
    fused_apply_constraints(
        param_flat,
        anchor_flat,
        offsets,
        sizes,
        constraints,
        norms,
        eps=1e-8,
    )
    param_flat_ref_applied = param_flat.clone()
    
    # Reverse with fused kernel
    fused_reverse_constraints(
        param_flat,
        anchor_flat,
        offsets,
        sizes,
        alphas,
    )
    
    # Reverse with PyTorch (reference)
    for i in range(num_params):
        start = offsets[i].item()
        end = start + sizes[i].item()
        diff = param_flat_ref_applied[start:end] - anchor_flat[start:end]
        param_flat_ref[start:end] = anchor_flat[start:end] + diff / alphas[i]
    
    # Compare
    max_diff = torch.max(torch.abs(param_flat - param_flat_ref)).item()
    print(f"  Max difference: {max_diff:.2e}")
    assert max_diff < 1e-4, f"Constraint reversal failed: {max_diff}"
    print("  ✓ Passed")


def test_gamma_operations():
    """Test gamma perturbation and update."""
    print("\nTesting gamma operations...")
    
    # Use the first available CUDA device (respects CUDA_VISIBLE_DEVICES)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32
    
    num_params = 100
    gamma = torch.rand(num_params, device=device, dtype=dtype) * 0.1
    ts = torch.rand(num_params, device=device, dtype=dtype) * 0.1
    seed = 42
    
    # Test perturbation
    zs = fused_perturb_gamma(
        gamma.clone(),
        ts,
        seed,
        delta=1.0,
        tau=0.2,
        zo_eps=0.1,
        zs=None,
    )
    
    assert zs.shape == gamma.shape, "zs shape mismatch"
    print("  ✓ Gamma perturbation passed")
    
    # Test update
    gamma_update = gamma.clone()
    grad = 0.01
    step_size = 2.0
    tau = 0.2
    
    fused_update_gamma(
        gamma_update,
        ts,
        zs,
        grad,
        step_size,
        tau,
    )
    
    # Check clipping
    gamma_min = (1.0 - tau) * ts
    gamma_max = (1.0 + tau) * ts
    assert torch.all(gamma_update >= gamma_min - 1e-6), "Gamma below minimum"
    assert torch.all(gamma_update <= gamma_max + 1e-6), "Gamma above maximum"
    print("  ✓ Gamma update passed")


def main():
    """Run all tests."""
    print("=" * 60)
    print("DiZO Fused Kernels Test Suite")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping tests")
        return
    
    # Print device info
    device = torch.device('cuda:0')
    print(f"Using device: {device}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    print()
    
    try:
        test_norm_computation()
        test_constraint_application()
        test_constraint_reversal()
        test_gamma_operations()
        
        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

