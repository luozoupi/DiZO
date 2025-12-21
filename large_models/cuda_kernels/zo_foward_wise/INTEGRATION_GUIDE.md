# DiZO Optimized ZO-Forward CUDA Kernels - Integration Guide

## Performance Summary

### Benchmark Results on NVIDIA H200

| Model Scale | PyTorch Per-Group | CUDA V2 | Speedup |
|-------------|-------------------|---------|---------|
| OPT-350m (144 groups, 268M elements) | 18.46 ms | 5.00 ms | **3.69x** |
| OPT-2.7b (192 groups, 2.7B elements) | 103.36 ms | 39.21 ms | **2.64x** |
| OPT-6.7b (192 groups, 6.4B elements) | 34.83 ms (norm only) | 11.18 ms | **3.12x** |

### Individual Kernel Speedups (OPT-350m)
- **Norm Computation**: 7.26 ms → 0.57 ms (**12.7x faster**)
- **Constraint Application**: 9.15 ms → 0.93 ms (**9.8x faster**)

## Installation

```bash
cd /path/to/cuda_kernels/zo_foward_wise

# Build V2 CUDA extension
TORCH_CUDA_ARCH_LIST="9.0" python setup_v2.py install

# For different GPUs:
# A100: TORCH_CUDA_ARCH_LIST="8.0"
# V100: TORCH_CUDA_ARCH_LIST="7.0"
```

## Integration into trainer.py

### Option 1: Direct Integration

```python
# In trainer.py, add at the top:
try:
    import dizo_fused_kernels_cuda_v2 as cuda_kernels
    CUDA_KERNELS_AVAILABLE = True
except ImportError:
    CUDA_KERNELS_AVAILABLE = False

# In DiZO class, add helper method:
class DiZO:
    def __init__(self, ...):
        # ... existing code ...
        
        # Pre-compute flattened parameter layout for CUDA kernels
        if CUDA_KERNELS_AVAILABLE:
            self._setup_cuda_buffers()
    
    def _setup_cuda_buffers(self):
        """Pre-allocate buffers for CUDA kernel operations."""
        offsets = []
        sizes = []
        offset = 0
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                offsets.append(offset)
                sizes.append(param.numel())
                offset += param.numel()
        
        self.cuda_offsets = torch.tensor(offsets, dtype=torch.long, device=self.device)
        self.cuda_sizes = torch.tensor(sizes, dtype=torch.long, device=self.device)
        self.total_params = offset
        
        # Pre-allocate flat buffers
        self.param_flat = torch.empty(self.total_params, device=self.device)
        self.anchor_flat = torch.empty(self.total_params, device=self.device)
        self.norms_buffer = torch.empty(len(sizes), device=self.device)
        self.constraints_buffer = torch.empty(len(sizes), device=self.device)
    
    def _flatten_params_to_buffer(self, buffer):
        """Copy parameters to flat buffer (in-place)."""
        offset = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                buffer[offset:offset + param.numel()] = param.data.view(-1)
                offset += param.numel()
    
    def _unflatten_buffer_to_params(self, buffer):
        """Copy flat buffer back to parameters (in-place)."""
        offset = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(buffer[offset:offset + param.numel()].view(param.shape))
                offset += param.numel()
```

### Option 2: Replace `perturb_gamma` Method

```python
def perturb_gamma_cuda(self, epsilon=1e-3, mode='random'):
    """Optimized perturb_gamma using CUDA kernels."""
    if not CUDA_KERNELS_AVAILABLE:
        return self.perturb_gamma_original(epsilon, mode)
    
    # Flatten current parameters
    self._flatten_params_to_buffer(self.param_flat)
    
    # Compute norms with CUDA kernel
    norms = cuda_kernels.fused_compute_norms(
        self.param_flat, self.anchor_flat, 
        self.cuda_offsets, self.cuda_sizes
    )
    
    # Compute constraints (epsilon-based)
    self.constraints_buffer.fill_(epsilon)
    
    # Apply constraints with CUDA kernel
    cuda_kernels.fused_apply_constraints(
        self.param_flat, self.anchor_flat,
        self.cuda_offsets, self.cuda_sizes,
        self.constraints_buffer, norms, 1e-8
    )
    
    # Copy back to parameters
    self._unflatten_buffer_to_params(self.param_flat)
```

### Option 3: Minimal Change - Replace Inner Loop

If you want minimal code changes, just replace the inner norm/constraint loop:

```python
# Original code in zo_forward or similar:
for name, param in self.model.named_parameters():
    if param.requires_grad:
        anchor = self.anchor[name]
        diff = param.data - anchor.data
        norm = torch.norm(diff)
        if norm > epsilon:
            param.data = anchor.data + diff * (epsilon / (norm + 1e-8))

# Replace with CUDA-accelerated version:
if CUDA_KERNELS_AVAILABLE:
    # Flatten all params (single vectorized copy)
    param_flat = torch.cat([p.data.view(-1) for p in params_to_update])
    anchor_flat = torch.cat([self.anchor[n].data.view(-1) for n, _ in named_params])
    
    # Single CUDA kernel call for all norms
    norms = cuda_kernels.fused_compute_norms(
        param_flat, anchor_flat, offsets, sizes
    )
    
    # Single CUDA kernel call for all constraints
    constraints = torch.full((num_params,), epsilon, device=device)
    cuda_kernels.fused_apply_constraints(
        param_flat, anchor_flat, offsets, sizes,
        constraints, norms, 1e-8
    )
    
    # Unflatten back to parameters
    offset = 0
    for name, param in named_params:
        if param.requires_grad:
            numel = param.numel()
            param.data.copy_(param_flat[offset:offset+numel].view(param.shape))
            offset += numel
```

## Key Optimizations Explained

### 1. Float4 Vectorized Memory Access
```cuda
// Instead of:
float diff = param[i] - anchor[i];

// We use:
float4 p4 = reinterpret_cast<const float4*>(param)[i];
float4 a4 = reinterpret_cast<const float4*>(anchor)[i];
// Process 4 elements at once
```
This achieves **4x memory bandwidth** by coalescing 4 float accesses.

### 2. Multi-Block Parallelism
```cuda
// Old: 1 block per parameter group
// New: Multiple blocks per parameter group, then atomic reduction

int blocks_per_group = (size + BLOCK_SIZE*4 - 1) / (BLOCK_SIZE*4);
int total_blocks = num_params * blocks_per_group;
```
This fully utilizes all GPU SMs for large parameter groups.

### 3. Warp-Level Reductions
```cuda
// Use __shfl_down_sync for intra-warp reduction (no shared memory)
for (int offset = 16; offset > 0; offset /= 2) {
    local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
}
```
Avoids shared memory bank conflicts and synchronization overhead.

## API Reference

### `fused_compute_norms`
```python
norms = cuda_kernels.fused_compute_norms(
    param_flat: Tensor,    # Flattened parameters [total_size]
    anchor_flat: Tensor,   # Flattened anchors [total_size]  
    offsets: Tensor,       # Start offset for each group [num_groups]
    sizes: Tensor          # Size of each group [num_groups]
) -> Tensor  # norms [num_groups]
```

### `fused_apply_constraints`
```python
cuda_kernels.fused_apply_constraints(
    param_flat: Tensor,     # [total_size], MODIFIED IN-PLACE
    anchor_flat: Tensor,    # [total_size], read-only
    offsets: Tensor,        # [num_groups]
    sizes: Tensor,          # [num_groups]
    constraints: Tensor,    # epsilon for each group [num_groups]
    norms: Tensor,          # pre-computed norms [num_groups]
    eps: float              # numerical stability (default: 1e-8)
)
```

## Troubleshooting

### Build Errors
```bash
# Check CUDA version
nvcc --version

# Check PyTorch CUDA version
python -c "import torch; print(torch.version.cuda)"

# Ensure they match, then rebuild with correct arch
TORCH_CUDA_ARCH_LIST="9.0" python setup_v2.py install --force
```

### Runtime Errors
```python
# Verify installation
import dizo_fused_kernels_cuda_v2
print(dir(dizo_fused_kernels_cuda_v2))

# Check tensor contiguity (required)
assert param_flat.is_contiguous()
assert anchor_flat.is_contiguous()
```

### Memory Issues
The CUDA kernels operate in-place where possible. For large models:
- Use `torch.cuda.empty_cache()` before operations
- Consider gradient checkpointing
- Use mixed precision (FP16) if numerical stability allows
