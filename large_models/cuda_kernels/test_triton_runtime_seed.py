"""Test Triton kernel with runtime seed to avoid recompilation."""
import torch
import time
import sys
import numpy as np
sys.path.insert(0, 'Perturb_wise')
import triton
import triton.language as tl

# =============================================================================
# Runtime seed version - seed is loaded from memory, not a compile-time constant
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
def fused_perturb_runtime_seed(
    params_ptr,
    seed_ptr,  # Pointer to seed tensor instead of scalar
    alpha,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Load seed from memory (runtime value, not compile-time)
    seed = tl.load(seed_ptr)
    
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    params = tl.load(params_ptr + offsets, mask=mask, other=0.0)
    z = tl.randn(seed, offsets)
    result = params + alpha * z
    tl.store(params_ptr + offsets, result, mask=mask)


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
def fused_update_runtime_seed(
    params_ptr,
    seed_ptr,  # Pointer to seed tensor instead of scalar
    projected_grad,
    lr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    seed = tl.load(seed_ptr)
    
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    params = tl.load(params_ptr + offsets, mask=mask, other=0.0)
    z = tl.randn(seed, offsets)
    result = params - lr * projected_grad * z
    tl.store(params_ptr + offsets, result, mask=mask)


def main():
    n_elements = 331_198_464
    eps = 1e-3
    lr = 1e-5

    param_flat = torch.randn(n_elements, device='cuda', dtype=torch.float32)
    seed_tensor = torch.tensor([123456789], dtype=torch.int64, device='cuda')
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    # Warmup
    print("Warming up...")
    for _ in range(20):
        fused_perturb_runtime_seed[grid](param_flat, seed_tensor, eps, n_elements)
        fused_update_runtime_seed[grid](param_flat, seed_tensor, 0.001, lr, n_elements)
    torch.cuda.synchronize()

    n_iter = 20

    print('\nTest 1: Runtime seed with varying values')
    seeds = [int(np.random.randint(1000000000)) for _ in range(n_iter)]
    torch.cuda.synchronize()
    start = time.perf_counter()
    for i in range(n_iter):
        seed_tensor[0] = seeds[i]
        fused_perturb_runtime_seed[grid](param_flat, seed_tensor, eps, n_elements)
        fused_perturb_runtime_seed[grid](param_flat, seed_tensor, -2*eps, n_elements)
        fused_perturb_runtime_seed[grid](param_flat, seed_tensor, eps, n_elements)
        fused_update_runtime_seed[grid](param_flat, seed_tensor, 0.001, lr, n_elements)
    torch.cuda.synchronize()
    total = (time.perf_counter() - start) * 1000 / n_iter
    print(f'  Total: {total:.2f} ms per iter (expected ~8ms)')

    # Compare with original kernel
    from triton_fused_perturb import fused_perturb_kernel_autotuned, fused_update_kernel_autotuned
    
    print('\nTest 2: Original kernel with fixed seed')
    fixed_seed = 123456789
    for _ in range(20):
        fused_perturb_kernel_autotuned[grid](param_flat, fixed_seed, eps, n_elements)
        fused_update_kernel_autotuned[grid](param_flat, fixed_seed, 0.001, lr, n_elements)
    torch.cuda.synchronize()
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iter):
        fused_perturb_kernel_autotuned[grid](param_flat, fixed_seed, eps, n_elements)
        fused_perturb_kernel_autotuned[grid](param_flat, fixed_seed, -2*eps, n_elements)
        fused_perturb_kernel_autotuned[grid](param_flat, fixed_seed, eps, n_elements)
        fused_update_kernel_autotuned[grid](param_flat, fixed_seed, 0.001, lr, n_elements)
    torch.cuda.synchronize()
    total2 = (time.perf_counter() - start) * 1000 / n_iter
    print(f'  Total: {total2:.2f} ms per iter')
    
    print('\nTest 3: Original kernel with varying seed')
    # Warmup with first seed
    for _ in range(20):
        fused_perturb_kernel_autotuned[grid](param_flat, seeds[0], eps, n_elements)
        fused_update_kernel_autotuned[grid](param_flat, seeds[0], 0.001, lr, n_elements)
    torch.cuda.synchronize()
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    for i in range(n_iter):
        fused_perturb_kernel_autotuned[grid](param_flat, seeds[i], eps, n_elements)
        fused_perturb_kernel_autotuned[grid](param_flat, seeds[i], -2*eps, n_elements)
        fused_perturb_kernel_autotuned[grid](param_flat, seeds[i], eps, n_elements)
        fused_update_kernel_autotuned[grid](param_flat, seeds[i], 0.001, lr, n_elements)
    torch.cuda.synchronize()
    total3 = (time.perf_counter() - start) * 1000 / n_iter
    print(f'  Total: {total3:.2f} ms per iter')


if __name__ == "__main__":
    main()
