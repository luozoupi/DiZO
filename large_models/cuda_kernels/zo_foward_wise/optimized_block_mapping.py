#!/usr/bin/env python3
"""
Optimized FusedDiZOKernelsV2 with GPU-based block mapping computation.

The original _setup_block_mapping uses Python loops which is very slow for large models:
- OPT-13B: 6.3M blocks, 1.1 seconds on CPU

This version computes block mapping on GPU using vectorized operations:
- OPT-13B: 6.3M blocks, ~5 ms on GPU
"""

import torch
import triton
import triton.language as tl
import time
from typing import Tuple, List, Optional


def compute_block_mapping_gpu(sizes: torch.Tensor, BLOCK_SIZE: int = 2048) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Compute block mapping entirely on GPU using vectorized operations.
    
    Args:
        sizes: Tensor of parameter sizes [num_params]
        BLOCK_SIZE: Number of elements per block
        
    Returns:
        block_to_param: Tensor mapping each block to its parameter index [num_blocks]
        block_start: Tensor of start indices within each parameter [num_blocks]
        num_blocks: Total number of blocks
    """
    device = sizes.device
    num_params = sizes.shape[0]
    
    # Compute number of blocks per parameter
    blocks_per_param = (sizes + BLOCK_SIZE - 1) // BLOCK_SIZE  # [num_params]
    
    # Total number of blocks
    num_blocks = int(blocks_per_param.sum().item())
    
    # Compute cumulative sum for offsets
    param_block_offsets = torch.zeros(num_params + 1, device=device, dtype=torch.int64)
    param_block_offsets[1:] = torch.cumsum(blocks_per_param, dim=0)
    
    # Use repeat_interleave to expand param indices
    # This is much faster than Python loops
    block_to_param = torch.repeat_interleave(
        torch.arange(num_params, device=device, dtype=torch.int32),
        blocks_per_param.int()
    )
    
    # Compute block_start indices
    # For each block, compute: block_idx_within_param * BLOCK_SIZE
    # First compute which block index this is within its parameter
    block_indices = torch.arange(num_blocks, device=device, dtype=torch.int64)
    param_starts = param_block_offsets[block_to_param.long()]  # Start block idx for each param
    block_idx_within_param = block_indices - param_starts
    block_start = block_idx_within_param * BLOCK_SIZE
    
    return block_to_param, block_start, num_blocks


def compute_block_mapping_gpu_v2(sizes: torch.Tensor, BLOCK_SIZE: int = 2048) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Optimized V2: Avoid repeat_interleave by using searchsorted.
    
    The key insight: given a cumulative sum of blocks, we can use searchsorted
    to find which parameter each block belongs to.
    """
    device = sizes.device
    num_params = sizes.shape[0]
    
    # Compute number of blocks per parameter
    blocks_per_param = (sizes + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Total number of blocks
    num_blocks = int(blocks_per_param.sum().item())
    
    # Cumulative sum of blocks per param
    cumsum = torch.zeros(num_params + 1, device=device, dtype=torch.int64)
    cumsum[1:] = torch.cumsum(blocks_per_param, dim=0)
    
    # Use searchsorted to find param index for each block
    # block_to_param[i] = max j such that cumsum[j] <= i
    # searchsorted with right=True gives us j+1 where cumsum[j] <= i < cumsum[j+1]
    block_indices = torch.arange(num_blocks, device=device, dtype=torch.int64)
    block_to_param = torch.searchsorted(cumsum[1:], block_indices, right=True).int()
    
    # Compute block_start
    param_starts = cumsum[block_to_param.long()]
    block_idx_within_param = block_indices - param_starts
    block_start = block_idx_within_param * BLOCK_SIZE
    
    return block_to_param, block_start, num_blocks


class FusedDiZOKernelsV2Optimized:
    """
    Optimized fused kernels for DiZO zo_forward with GPU-based block mapping.
    
    Key optimizations over original:
    1. Block mapping computed on GPU (100x faster for large models)
    2. Cached block mapping (no recomputation)
    3. Pre-allocated buffers
    """
    
    def __init__(self, num_params: int, total_elements: int, device: torch.device,
                 offsets: torch.Tensor = None, sizes: torch.Tensor = None):
        self.num_params = num_params
        self.total_elements = total_elements
        self.device = device
        
        # Pre-allocate buffers
        self.partial_sums = torch.zeros(num_params, device=device, dtype=torch.float32)
        self.norms = torch.empty(num_params, device=device, dtype=torch.float32)
        self.alphas = torch.empty(num_params, device=device, dtype=torch.float32)
        self.zs = torch.empty(num_params, device=device, dtype=torch.float32)
        
        # Pre-allocate scalar tensors for runtime values
        self._seed_tensor = torch.zeros(1, device=device, dtype=torch.int64)
        self._delta_tensor = torch.zeros(1, device=device, dtype=torch.float32)
        self._grad_tensor = torch.zeros(1, device=device, dtype=torch.float32)
        
        self._seed = 0
        
        # Block mapping
        self.block_to_param = None
        self.block_start = None
        self.num_blocks = 0
        self.BLOCK_SIZE = 2048
        
        if offsets is not None and sizes is not None:
            self._setup_block_mapping_gpu(sizes)
    
    def _setup_block_mapping_gpu(self, sizes: torch.Tensor):
        """
        GPU-accelerated block mapping computation.
        100x faster than CPU-based approach for large models.
        """
        start = time.time()
        
        self.block_to_param, self.block_start, self.num_blocks = \
            compute_block_mapping_gpu_v2(sizes, self.BLOCK_SIZE)
        
        elapsed = (time.time() - start) * 1000
        print(f"GPU block mapping V2: {self.num_params} params -> {self.num_blocks} blocks ({elapsed:.2f} ms)")


def benchmark_block_mapping():
    """Compare CPU vs GPU block mapping computation."""
    import sys
    sys.path.insert(0, '.')
    from dizo_fused_kernels_v2 import FusedDiZOKernelsV2
    
    device = torch.device('cuda')
    
    # OPT-13B-like configuration
    num_params = 644
    total_elements = 12_853_473_280
    
    # Create realistic sizes (mixture of large and small)
    sizes_list = []
    hidden = 5120
    ffn = 20480
    vocab_size = 50272
    
    # Embeddings
    sizes_list.append(vocab_size * hidden)
    sizes_list.append(2050 * hidden)
    
    # Layers (40 layers)
    for _ in range(40):
        for _ in range(4):  # q,k,v,o
            sizes_list.append(hidden * hidden)
            sizes_list.append(hidden)
        sizes_list.extend([hidden, hidden])  # attn ln
        sizes_list.append(ffn * hidden)
        sizes_list.append(ffn)
        sizes_list.append(hidden * ffn)
        sizes_list.append(hidden)
        sizes_list.extend([hidden, hidden])  # final ln
    
    sizes = torch.tensor(sizes_list, device=device, dtype=torch.int64)
    offsets_list = [0]
    for s in sizes_list[:-1]:
        offsets_list.append(offsets_list[-1] + s)
    offsets = torch.tensor(offsets_list, device=device, dtype=torch.int64)
    
    print(f"Parameters: {len(sizes_list)}")
    print(f"Total elements: {sum(sizes_list):,}")
    print()
    
    # Benchmark CPU version
    print("CPU block mapping (original):")
    start = time.time()
    zo_cpu = FusedDiZOKernelsV2(len(sizes_list), sum(sizes_list), device, offsets, sizes)
    cpu_time = (time.time() - start) * 1000
    print(f"  Total time: {cpu_time:.2f} ms")
    print()
    
    # Benchmark GPU V1 version
    print("GPU block mapping V1 (repeat_interleave):")
    torch.cuda.synchronize()
    start = time.time()
    block_to_param_v1, block_start_v1, num_blocks_v1 = compute_block_mapping_gpu(sizes, 2048)
    torch.cuda.synchronize()
    gpu_v1_time = (time.time() - start) * 1000
    print(f"  Time: {gpu_v1_time:.2f} ms")
    print()
    
    # Benchmark GPU V2 version
    print("GPU block mapping V2 (searchsorted):")
    torch.cuda.synchronize()
    start = time.time()
    block_to_param_v2, block_start_v2, num_blocks_v2 = compute_block_mapping_gpu_v2(sizes, 2048)
    torch.cuda.synchronize()
    gpu_v2_time = (time.time() - start) * 1000
    print(f"  Time: {gpu_v2_time:.2f} ms")
    print()
    
    # Verify correctness
    print("Verifying correctness...")
    assert zo_cpu.num_blocks == num_blocks_v1 == num_blocks_v2, f"Num blocks mismatch"
    assert torch.equal(zo_cpu.block_to_param, block_to_param_v1), "V1 block_to_param mismatch"
    assert torch.equal(zo_cpu.block_to_param, block_to_param_v2), "V2 block_to_param mismatch"
    assert torch.equal(zo_cpu.block_start, block_start_v1), "V1 block_start mismatch"
    assert torch.equal(zo_cpu.block_start, block_start_v2), "V2 block_start mismatch"
    print("  ✓ All checks passed!")
    print()
    
    print(f"Speedup V1: {cpu_time / gpu_v1_time:.1f}x")
    print(f"Speedup V2: {cpu_time / gpu_v2_time:.1f}x")


if __name__ == '__main__':
    benchmark_block_mapping()
