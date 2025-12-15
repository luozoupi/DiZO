"""
Debug test for custom Philox implementation
"""
import torch
import triton
import triton.language as tl


@triton.jit
def test_tl_randn_kernel(out_ptr, seed, n_elements, BLOCK_SIZE: tl.constexpr):
    """Test kernel using tl.randn"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    z = tl.randn(seed, offsets)
    
    tl.store(out_ptr + offsets, z, mask=mask)


@triton.jit
def test_custom_philox_kernel(out_ptr, seed, n_elements, BLOCK_SIZE: tl.constexpr):
    """Test kernel using custom Philox RNG"""
    # Constants
    M0 = 0xD2511F53
    M1 = 0xCD9E8D57
    W0 = 0x9E3779B9
    W1 = 0xBB67AE85
    
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Initialize counters
    c0 = offsets.to(tl.uint32)
    c1 = tl.zeros_like(c0)
    c2 = tl.zeros_like(c0)
    c3 = tl.zeros_like(c0)
    
    # Initialize keys - broadcast seed
    k0 = tl.full(c0.shape, seed, dtype=tl.uint32)
    k1 = tl.zeros_like(k0)
    
    # 10 rounds of Philox
    for _ in range(10):
        # Philox round
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
        
        # Bump key
        k0 = k0 + W0
        k1 = k1 + W1
    
    # Convert to uniform then to normal via Box-Muller
    u1 = (c0.to(tl.float32) + 0.5) * (1.0 / 4294967296.0)
    u2 = (c1.to(tl.float32) + 0.5) * (1.0 / 4294967296.0)
    
    # Box-Muller transform
    TWO_PI = 6.283185307179586
    r = tl.sqrt(-2.0 * tl.log(u1))
    theta = TWO_PI * u2
    z = r * tl.cos(theta)
    
    tl.store(out_ptr + offsets, z, mask=mask)


def main():
    n = 1_000_000
    seed = 42
    
    print("=" * 60)
    print("PHILOX RNG DEBUG TEST")
    print("=" * 60)
    
    # Test tl.randn
    out1 = torch.zeros(n, device='cuda', dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(n, 1024),)
    test_tl_randn_kernel[grid](out1, seed, n, BLOCK_SIZE=1024)
    torch.cuda.synchronize()
    
    print("\ntl.randn output:")
    print(f"  Mean: {out1.mean().item():.6f}")
    print(f"  Std: {out1.std().item():.6f}")
    print(f"  Min: {out1.min().item():.4f}")
    print(f"  Max: {out1.max().item():.4f}")
    print(f"  Sample: {out1[:10].tolist()}")
    
    # Test custom Philox
    out2 = torch.zeros(n, device='cuda', dtype=torch.float32)
    test_custom_philox_kernel[grid](out2, seed, n, BLOCK_SIZE=1024)
    torch.cuda.synchronize()
    
    print("\nCustom Philox output:")
    print(f"  Mean: {out2.mean().item():.6f}")
    print(f"  Std: {out2.std().item():.6f}")
    print(f"  Min: {out2.min().item():.4f}")
    print(f"  Max: {out2.max().item():.4f}")
    print(f"  Sample: {out2[:10].tolist()}")
    
    # Check if values are all zeros
    nonzero1 = (out1 != 0).sum().item()
    nonzero2 = (out2 != 0).sum().item()
    print(f"\nNon-zero count:")
    print(f"  tl.randn: {nonzero1:,} / {n:,}")
    print(f"  Custom Philox: {nonzero2:,} / {n:,}")


if __name__ == "__main__":
    main()
