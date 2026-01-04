#!/usr/bin/env python3
"""
Optimized MeZO with Custom CUDA Kernels and Async Execution

Key optimizations to reduce CPU latency:
1. Custom CUDA kernels (via torch.cuda.jit / load_inline)
2. CUDA streams for async execution
3. CPU-GPU overlap using multiple streams
4. Double buffering for perturbation vectors
5. Gradient checkpointing integration

The main insight: CPU latency comes from:
- Python loop overhead
- Kernel launch latency
- CPU-GPU synchronization points

Solutions:
- Fuse operations into single kernels
- Use multiple CUDA streams to pipeline operations
- Overlap CPU work (RNG prep) with GPU work (forward pass)
- Minimize sync points
"""

import torch
import torch.nn as nn
import torch.utils.cpp_extension
import time
import numpy as np
from typing import List, Tuple, Optional, Dict
import threading
from queue import Queue


# =============================================================================
# CUDA Kernel Loading (using torch's JIT compilation)
# =============================================================================

CUDA_SOURCE = """
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

// Fused perturbation kernel
__global__ void perturb_add_kernel(
    float* __restrict__ params,
    const float* __restrict__ z,
    const int64_t n_elements,
    const float alpha
) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        params[idx] = fmaf(alpha, z[idx], params[idx]);
    }
}

// Vectorized version (4x throughput)
__global__ void perturb_add_vec4_kernel(
    float4* __restrict__ params,
    const float4* __restrict__ z,
    const int64_t n_vec4,
    const float alpha
) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_vec4) {
        float4 p = params[idx];
        float4 z_val = z[idx];
        p.x = fmaf(alpha, z_val.x, p.x);
        p.y = fmaf(alpha, z_val.y, p.y);
        p.z = fmaf(alpha, z_val.z, p.z);
        p.w = fmaf(alpha, z_val.w, p.w);
        params[idx] = p;
    }
}

// Kernel with inline RNG (eliminates separate RNG kernel)
__global__ void perturb_with_rng_kernel(
    float* __restrict__ params,
    const int64_t n_elements,
    const float eps,
    const uint64_t seed
) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        curandState_t state;
        curand_init(seed, idx, 0, &state);
        float z = curand_normal(&state);
        params[idx] = fmaf(eps, z, params[idx]);
    }
}

// Half-precision kernel
__global__ void perturb_add_half_kernel(
    at::Half* __restrict__ params,
    const at::Half* __restrict__ z,
    const int64_t n_elements,
    const float alpha
) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        float p = __half2float(params[idx]);
        float z_val = __half2float(z[idx]);
        params[idx] = __float2half(fmaf(alpha, z_val, p));
    }
}

// Host functions
void perturb_add_cuda(
    torch::Tensor params,
    torch::Tensor z,
    float alpha
) {
    const int64_t n = params.numel();
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;
    
    auto stream = at::cuda::getCurrentCUDAStream();
    
    if (params.dtype() == torch::kFloat32) {
        // Use vectorized kernel if aligned
        if (n % 4 == 0 && reinterpret_cast<uintptr_t>(params.data_ptr()) % 16 == 0) {
            perturb_add_vec4_kernel<<<(n/4 + block_size - 1) / block_size, block_size, 0, stream>>>(
                reinterpret_cast<float4*>(params.data_ptr<float>()),
                reinterpret_cast<const float4*>(z.data_ptr<float>()),
                n / 4,
                alpha
            );
        } else {
            perturb_add_kernel<<<grid_size, block_size, 0, stream>>>(
                params.data_ptr<float>(),
                z.data_ptr<float>(),
                n,
                alpha
            );
        }
    } else if (params.dtype() == torch::kFloat16) {
        perturb_add_half_kernel<<<grid_size, block_size, 0, stream>>>(
            params.data_ptr<at::Half>(),
            z.data_ptr<at::Half>(),
            n,
            alpha
        );
    }
}

void perturb_with_rng_cuda(
    torch::Tensor params,
    float eps,
    int64_t seed
) {
    const int64_t n = params.numel();
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;
    
    auto stream = at::cuda::getCurrentCUDAStream();
    
    perturb_with_rng_kernel<<<grid_size, block_size, 0, stream>>>(
        params.data_ptr<float>(),
        n,
        eps,
        seed
    );
}

// On-stream perturbation with specified stream
void perturb_add_on_stream(
    torch::Tensor params,
    torch::Tensor z,
    float alpha,
    int64_t stream_ptr
) {
    const int64_t n = params.numel();
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;
    
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    
    perturb_add_kernel<<<grid_size, block_size, 0, stream>>>(
        params.data_ptr<float>(),
        z.data_ptr<float>(),
        n,
        alpha
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("perturb_add", &perturb_add_cuda, "Fused perturbation add");
    m.def("perturb_with_rng", &perturb_with_rng_cuda, "Perturbation with inline RNG");
    m.def("perturb_add_on_stream", &perturb_add_on_stream, "Perturbation on specific stream");
}
"""


def load_cuda_kernels():
    """Load custom CUDA kernels using torch's JIT compilation"""
    try:
        from torch.utils.cpp_extension import load_inline
        
        cuda_module = load_inline(
            name='fused_mezo_cuda_v2',
            cpp_sources='',
            cuda_sources=CUDA_SOURCE,
            extra_cuda_cflags=['-O3', '--use_fast_math'],
            verbose=False,
            with_cuda=True
        )
        return cuda_module
    except Exception as e:
        print(f"[Warning] Failed to load CUDA kernels: {e}")
        print("[Warning] Falling back to PyTorch operations")
        return None


# =============================================================================
# Async Execution Utilities
# =============================================================================

class CUDAStreamPool:
    """Pool of CUDA streams for async execution"""
    
    def __init__(self, n_streams: int = 4):
        self.streams = [torch.cuda.Stream() for _ in range(n_streams)]
        self.n_streams = n_streams
        self.current_idx = 0
    
    def get_stream(self) -> torch.cuda.Stream:
        stream = self.streams[self.current_idx]
        self.current_idx = (self.current_idx + 1) % self.n_streams
        return stream
    
    def synchronize_all(self):
        for stream in self.streams:
            stream.synchronize()


class DoubleBuffer:
    """
    Double buffering for perturbation vectors.
    
    While one buffer is being used for perturbation, 
    the other can be prepared (RNG generation) asynchronously.
    """
    
    def __init__(self, size: int, device: torch.device, dtype: torch.dtype):
        self.buffers = [
            torch.empty(size, device=device, dtype=dtype),
            torch.empty(size, device=device, dtype=dtype),
        ]
        self.current_idx = 0
        self.streams = [torch.cuda.Stream(), torch.cuda.Stream()]
        self.seeds = [0, 0]
        self.ready_events = [torch.cuda.Event(), torch.cuda.Event()]
    
    def get_current(self) -> torch.Tensor:
        return self.buffers[self.current_idx]
    
    def get_next(self) -> torch.Tensor:
        return self.buffers[1 - self.current_idx]
    
    def swap(self):
        self.current_idx = 1 - self.current_idx
    
    def prepare_next_async(self, seed: int):
        """Prepare the next buffer asynchronously"""
        next_idx = 1 - self.current_idx
        self.seeds[next_idx] = seed
        
        with torch.cuda.stream(self.streams[next_idx]):
            torch.manual_seed(seed)
            self.buffers[next_idx].normal_()
            self.ready_events[next_idx].record()
    
    def wait_for_current(self):
        """Wait for current buffer to be ready"""
        self.ready_events[self.current_idx].synchronize()


# =============================================================================
# Async MeZO Trainer
# =============================================================================

class AsyncMeZOTrainer:
    """
    MeZO trainer with async execution to hide CPU latency.
    
    Key techniques:
    1. CUDA streams for overlapping operations
    2. Double buffering for z vectors
    3. Async RNG preparation
    4. Pipelined forward passes
    """
    
    def __init__(
        self,
        model: nn.Module,
        eps: float = 1e-3,
        lr: float = 1e-5,
        use_cuda_kernels: bool = True,
        n_streams: int = 4,
        use_double_buffer: bool = True,
        use_gradient_checkpointing: bool = False,
    ):
        self.model = model
        self.eps = eps
        self.lr = lr
        
        # Try to load custom CUDA kernels
        self.cuda_module = load_cuda_kernels() if use_cuda_kernels else None
        self.use_cuda_kernels = self.cuda_module is not None
        
        # Collect trainable parameters
        self.trainable_params: List[nn.Parameter] = [
            p for p in model.parameters() if p.requires_grad
        ]
        self.param_numels = [p.numel() for p in self.trainable_params]
        self.total_numel = sum(self.param_numels)
        
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        
        print(f"[AsyncMeZO] {len(self.trainable_params)} params, {self.total_numel:,} elements")
        print(f"[AsyncMeZO] CUDA kernels: {'enabled' if self.use_cuda_kernels else 'disabled'}")
        
        # Setup double buffering
        self.use_double_buffer = use_double_buffer
        if use_double_buffer:
            self.z_buffer = DoubleBuffer(self.total_numel, self.device, self.dtype)
            print("[AsyncMeZO] Double buffering: enabled")
        else:
            self.z_flat = torch.empty(self.total_numel, device=self.device, dtype=self.dtype)
        
        # Create z views for each parameter
        self.offsets = []
        offset = 0
        for numel in self.param_numels:
            self.offsets.append(offset)
            offset += numel
        
        # CUDA streams for async execution
        self.stream_pool = CUDAStreamPool(n_streams)
        self.compute_stream = torch.cuda.Stream()
        self.rng_stream = torch.cuda.Stream()
        
        # Gradient checkpointing
        self.use_gradient_checkpointing = use_gradient_checkpointing
        if use_gradient_checkpointing:
            self._enable_gradient_checkpointing()
        
        # Timing
        self.timing = {
            'rng': [], 'perturb': [], 'forward': [], 
            'update': [], 'total': [], 'overhead': []
        }
        
        # Pre-prepare first z buffer
        if use_double_buffer:
            seed = np.random.randint(0, 2**31)
            self.z_buffer.prepare_next_async(seed)
            self.z_buffer.swap()
            self._current_seed = seed
    
    def _enable_gradient_checkpointing(self):
        """Enable gradient checkpointing on supported models"""
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            print("[AsyncMeZO] Gradient checkpointing: enabled")
        else:
            print("[AsyncMeZO] Gradient checkpointing: not supported by this model")
    
    def _get_z_views(self, z_flat: torch.Tensor) -> List[torch.Tensor]:
        """Get views into z_flat for each parameter"""
        views = []
        for i, (numel, param) in enumerate(zip(self.param_numels, self.trainable_params)):
            start = self.offsets[i]
            views.append(z_flat[start:start + numel].view(param.shape))
        return views
    
    def _perturb_cuda(self, z_flat: torch.Tensor, alpha: float):
        """Apply perturbation using custom CUDA kernel"""
        if self.use_cuda_kernels:
            z_views = self._get_z_views(z_flat)
            for param, z in zip(self.trainable_params, z_views):
                self.cuda_module.perturb_add(param.data.view(-1), z.view(-1), alpha)
        else:
            self._perturb_pytorch(z_flat, alpha)
    
    def _perturb_pytorch(self, z_flat: torch.Tensor, alpha: float):
        """Fallback: Apply perturbation using PyTorch"""
        z_views = self._get_z_views(z_flat)
        for param, z in zip(self.trainable_params, z_views):
            param.data.add_(z, alpha=alpha)
    
    def _perturb_async(self, z_flat: torch.Tensor, alpha: float, stream: torch.cuda.Stream):
        """Apply perturbation on a specific CUDA stream"""
        with torch.cuda.stream(stream):
            self._perturb_pytorch(z_flat, alpha)
    
    def zo_forward(self, batch) -> torch.Tensor:
        """Forward pass returning loss"""
        outputs = self.model(**batch)
        return outputs.loss if hasattr(outputs, 'loss') else outputs[0]
    
    def zo_step_async(self, batch) -> Tuple[torch.Tensor, float]:
        """
        ZO step with async execution to hide CPU latency.
        
        Timeline:
        [CPU] Generate seed for next iteration
        [GPU Stream 1] Current perturbation + forward 1
        [GPU Stream 2] Prepare next z buffer (async)
        [GPU Stream 1] Negative perturbation + forward 2
        [CPU] Compute gradient estimate (overlapped with GPU)
        [GPU Stream 1] Update + reset
        """
        t0 = time.perf_counter()
        
        # Get current z buffer
        if self.use_double_buffer:
            self.z_buffer.wait_for_current()
            z_flat = self.z_buffer.get_current()
            current_seed = self._current_seed
        else:
            current_seed = np.random.randint(0, 2**31)
            rng_t0 = time.perf_counter()
            torch.manual_seed(current_seed)
            self.z_flat.normal_()
            z_flat = self.z_flat
            self.timing['rng'].append((time.perf_counter() - rng_t0) * 1000)
        
        # Start preparing next z buffer asynchronously
        if self.use_double_buffer:
            next_seed = np.random.randint(0, 2**31)
            self.z_buffer.prepare_next_async(next_seed)
        
        # Perturb +eps
        perturb_t0 = time.perf_counter()
        with torch.cuda.stream(self.compute_stream):
            self._perturb_cuda(z_flat, self.eps)
        self.timing['perturb'].append((time.perf_counter() - perturb_t0) * 1000)
        
        # Forward pass 1
        forward_t0 = time.perf_counter()
        self.compute_stream.synchronize()  # Ensure perturbation is complete
        with torch.no_grad():
            loss1 = self.zo_forward(batch)
        
        # Perturb -2eps (on compute stream)
        perturb_t0 = time.perf_counter()
        with torch.cuda.stream(self.compute_stream):
            self._perturb_cuda(z_flat, -2 * self.eps)
        self.timing['perturb'].append((time.perf_counter() - perturb_t0) * 1000)
        
        # Forward pass 2
        self.compute_stream.synchronize()
        with torch.no_grad():
            loss2 = self.zo_forward(batch)
        self.timing['forward'].append((time.perf_counter() - forward_t0) * 1000)
        
        # Reset: +eps
        perturb_t0 = time.perf_counter()
        with torch.cuda.stream(self.compute_stream):
            self._perturb_cuda(z_flat, self.eps)
        self.timing['perturb'].append((time.perf_counter() - perturb_t0) * 1000)
        
        # Compute gradient estimate (CPU work, can overlap with GPU)
        projected_grad = (loss1.item() - loss2.item()) / (2 * self.eps)
        
        # Update parameters
        update_t0 = time.perf_counter()
        with torch.cuda.stream(self.compute_stream):
            self._perturb_cuda(z_flat, -self.lr * projected_grad)
        self.timing['update'].append((time.perf_counter() - update_t0) * 1000)
        
        # Swap buffers for next iteration
        if self.use_double_buffer:
            self.z_buffer.swap()
            self._current_seed = next_seed
        
        self.compute_stream.synchronize()
        
        self.timing['total'].append((time.perf_counter() - t0) * 1000)
        
        return (loss1 + loss2) / 2, projected_grad
    
    def zo_step_pipelined(self, batch) -> Tuple[torch.Tensor, float]:
        """
        Fully pipelined ZO step with maximum overlap.
        
        Uses separate streams for:
        - RNG generation
        - Perturbation
        - Forward pass
        
        This achieves maximum CPU-GPU overlap.
        """
        t0 = time.perf_counter()
        
        seed = np.random.randint(0, 2**31)
        
        # Stream setup
        rng_stream = self.rng_stream
        compute_stream = self.compute_stream
        
        # Generate z on RNG stream
        rng_t0 = time.perf_counter()
        with torch.cuda.stream(rng_stream):
            torch.manual_seed(seed)
            if self.use_double_buffer:
                z_flat = self.z_buffer.get_current()
            else:
                z_flat = self.z_flat
            z_flat.normal_()
            rng_done = torch.cuda.Event()
            rng_done.record()
        self.timing['rng'].append((time.perf_counter() - rng_t0) * 1000)
        
        # Wait for RNG, then perturb +eps
        compute_stream.wait_event(rng_done)
        
        perturb_t0 = time.perf_counter()
        with torch.cuda.stream(compute_stream):
            self._perturb_pytorch(z_flat, self.eps)
        
        compute_stream.synchronize()
        self.timing['perturb'].append((time.perf_counter() - perturb_t0) * 1000)
        
        # Forward passes
        forward_t0 = time.perf_counter()
        with torch.no_grad():
            loss1 = self.zo_forward(batch)
        
        # Perturb -2eps
        perturb_t0 = time.perf_counter()
        self._perturb_pytorch(z_flat, -2 * self.eps)
        self.timing['perturb'].append((time.perf_counter() - perturb_t0) * 1000)
        
        with torch.no_grad():
            loss2 = self.zo_forward(batch)
        self.timing['forward'].append((time.perf_counter() - forward_t0) * 1000)
        
        # Reset and update
        perturb_t0 = time.perf_counter()
        self._perturb_pytorch(z_flat, self.eps)
        self.timing['perturb'].append((time.perf_counter() - perturb_t0) * 1000)
        
        projected_grad = (loss1.item() - loss2.item()) / (2 * self.eps)
        
        update_t0 = time.perf_counter()
        self._perturb_pytorch(z_flat, -self.lr * projected_grad)
        self.timing['update'].append((time.perf_counter() - update_t0) * 1000)
        
        self.timing['total'].append((time.perf_counter() - t0) * 1000)
        
        return (loss1 + loss2) / 2, projected_grad
    
    def print_timing(self, name: str = "AsyncMeZO"):
        """Print timing summary"""
        print(f"\n{'='*60}")
        print(f"{name} TIMING")
        print("="*60)
        
        total = np.mean(self.timing['total']) if self.timing['total'] else 1
        
        for key, times in self.timing.items():
            if times:
                avg = np.mean(times)
                std = np.std(times) if len(times) > 1 else 0
                pct = avg / total * 100 if key != 'total' else 100
                if key == 'perturb':
                    print(f"  {key:12s}: {avg:8.3f} ± {std:5.3f} ms ({pct:5.1f}%) [per call]")
                else:
                    print(f"  {key:12s}: {avg:8.3f} ± {std:5.3f} ms ({pct:5.1f}%)")
    
    def reset_timing(self):
        self.timing = {k: [] for k in self.timing}


# =============================================================================
# Gradient Checkpointing Wrapper
# =============================================================================

class GradientCheckpointedModel(nn.Module):
    """
    Wrapper that applies gradient checkpointing to reduce memory.
    
    For MeZO, this helps with larger batch sizes.
    """
    
    def __init__(self, model: nn.Module, checkpoint_every: int = 2):
        super().__init__()
        self.model = model
        self.checkpoint_every = checkpoint_every
        
        # Enable gradient checkpointing if supported
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
    
    def forward(self, **kwargs):
        return self.model(**kwargs)


# =============================================================================
# Benchmark
# =============================================================================

def benchmark(
    model_name: str = "facebook/opt-350m",
    num_steps: int = 20,
    warmup_steps: int = 5,
    batch_size: int = 4,
    seq_length: int = 128,
):
    """Comprehensive async benchmark"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print("="*70)
    print("ASYNC MEZO BENCHMARK")
    print("="*70)
    print(f"Model: {model_name}")
    print(f"Steps: {num_steps}, Warmup: {warmup_steps}")
    print(f"Batch: {batch_size} x {seq_length}")
    
    device = "cuda"
    results = {}
    
    # Prepare batch
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    texts = ["Test sentence for benchmarking."] * batch_size
    batch = tokenizer(texts, return_tensors="pt", padding="max_length",
                      max_length=seq_length, truncation=True)
    batch = {k: v.to(device) for k, v in batch.items()}
    batch['labels'] = batch['input_ids'].clone()
    
    # =========================================================================
    # 1. Baseline
    # =========================================================================
    print("\n" + "-"*70)
    print("1. BASELINE (per-parameter RNG)")
    print("-"*70)
    
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()
    trainable = [p for p in model.parameters() if p.requires_grad]
    eps, lr = 1e-3, 1e-5
    baseline_times = []
    
    # Warmup + benchmark
    for step in range(-warmup_steps, num_steps):
        t0 = time.perf_counter()
        seed = np.random.randint(0, 2**31)
        
        torch.manual_seed(seed)
        for p in trainable:
            p.data.add_(torch.randn_like(p), alpha=eps)
        
        with torch.no_grad():
            loss1 = model(**batch).loss
        
        torch.manual_seed(seed)
        for p in trainable:
            p.data.add_(torch.randn_like(p), alpha=-2*eps)
        
        with torch.no_grad():
            loss2 = model(**batch).loss
        
        torch.manual_seed(seed)
        for p in trainable:
            p.data.add_(torch.randn_like(p), alpha=eps)
        
        grad = (loss1.item() - loss2.item()) / (2 * eps)
        
        torch.manual_seed(seed)
        for p in trainable:
            p.data.add_(torch.randn_like(p), alpha=-lr * grad)
        
        if step >= 0:
            baseline_times.append((time.perf_counter() - t0) * 1000)
            if step % 10 == 0:
                print(f"  Step {step}: loss={((loss1+loss2)/2).item():.4f}")
    
    results['baseline'] = np.mean(baseline_times)
    print(f"\n  Average: {results['baseline']:.2f} ms/step")
    
    del model
    torch.cuda.empty_cache()
    
    # =========================================================================
    # 2. Async MeZO (CUDA kernels + double buffer)
    # =========================================================================
    print("\n" + "-"*70)
    print("2. ASYNC MEZO (CUDA kernels + double buffer)")
    print("-"*70)
    
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()
    trainer = AsyncMeZOTrainer(
        model, eps=eps, lr=lr,
        use_cuda_kernels=True,
        use_double_buffer=True
    )
    
    # Warmup
    for _ in range(warmup_steps):
        trainer.zo_step_async(batch)
    trainer.reset_timing()
    
    # Benchmark
    torch.cuda.synchronize()
    for step in range(num_steps):
        loss, _ = trainer.zo_step_async(batch)
        if step % 10 == 0:
            print(f"  Step {step}: loss={loss.item():.4f}")
    torch.cuda.synchronize()
    
    trainer.print_timing("ASYNC MEZO")
    results['async_cuda'] = np.mean(trainer.timing['total'])
    
    del model, trainer
    torch.cuda.empty_cache()
    
    # =========================================================================
    # 3. Pipelined MeZO
    # =========================================================================
    print("\n" + "-"*70)
    print("3. PIPELINED MEZO (stream overlap)")
    print("-"*70)
    
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()
    trainer = AsyncMeZOTrainer(
        model, eps=eps, lr=lr,
        use_cuda_kernels=False,  # PyTorch ops for comparison
        use_double_buffer=True
    )
    
    # Warmup
    for _ in range(warmup_steps):
        trainer.zo_step_pipelined(batch)
    trainer.reset_timing()
    
    # Benchmark
    torch.cuda.synchronize()
    for step in range(num_steps):
        loss, _ = trainer.zo_step_pipelined(batch)
        if step % 10 == 0:
            print(f"  Step {step}: loss={loss.item():.4f}")
    torch.cuda.synchronize()
    
    trainer.print_timing("PIPELINED MEZO")
    results['pipelined'] = np.mean(trainer.timing['total'])
    
    del model, trainer
    torch.cuda.empty_cache()
    
    # =========================================================================
    # 4. With Gradient Checkpointing
    # =========================================================================
    print("\n" + "-"*70)
    print("4. WITH GRADIENT CHECKPOINTING")
    print("-"*70)
    
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()
    trainer = AsyncMeZOTrainer(
        model, eps=eps, lr=lr,
        use_cuda_kernels=False,
        use_double_buffer=True,
        use_gradient_checkpointing=True
    )
    
    # Warmup
    for _ in range(warmup_steps):
        trainer.zo_step_pipelined(batch)
    trainer.reset_timing()
    
    # Benchmark
    torch.cuda.synchronize()
    for step in range(num_steps):
        loss, _ = trainer.zo_step_pipelined(batch)
        if step % 10 == 0:
            print(f"  Step {step}: loss={loss.item():.4f}")
    torch.cuda.synchronize()
    
    trainer.print_timing("GRAD CHECKPOINT")
    results['grad_ckpt'] = np.mean(trainer.timing['total'])
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print(f"\n  {'Method':<35} {'Time (ms)':<12} {'Speedup':<10}")
    print(f"  {'-'*55}")
    
    baseline = results['baseline']
    for name, time_ms in results.items():
        speedup = baseline / time_ms
        marker = " ★" if speedup == max(baseline/t for t in results.values()) else ""
        print(f"  {name:<35} {time_ms:>8.2f}     {speedup:>6.2f}x{marker}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="facebook/opt-350m")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-length", type=int, default=128)
    args = parser.parse_args()
    
    benchmark(
        model_name=args.model,
        num_steps=args.steps,
        warmup_steps=args.warmup,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
    )
