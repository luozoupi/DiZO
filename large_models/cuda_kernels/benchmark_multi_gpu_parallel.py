#!/usr/bin/env python
"""
Multi-GPU Parallel MeZO Benchmark

This script benchmarks the multi-GPU parallel approach where:
- Main GPU (device 0): Handles θ₀ + εz → forward1 → loss1
- Side GPU (device 1): Handles θ₀ - εz → forward2 → loss2

Both forward passes run in parallel, potentially achieving ~2x speedup.

Usage:
    # Basic run (uses GPU 0 and 1)
    python benchmark_multi_gpu_parallel.py --model opt-350m
    
    # Use specific GPUs
    python benchmark_multi_gpu_parallel.py --model opt-350m --main_gpu 1 --side_gpu 2
    
    # Compare with single-GPU baseline
    python benchmark_multi_gpu_parallel.py --model opt-350m --compare_baseline
    
    # Debug mode with detailed timing
    python benchmark_multi_gpu_parallel.py --model opt-350m --debug

Author: DiZO Team
Date: 2026-01-XX
"""

import os
import sys

# Print debug info if environment variable is set
if os.environ.get('DEBUG_PYTHON'):
    print(f"DEBUG: Python executable: {sys.executable}", file=sys.stderr)
    print(f"DEBUG: Python version: {sys.version}", file=sys.stderr)
    print(f"DEBUG: Python path: {sys.path[:3]}", file=sys.stderr)

# Check for required modules before importing
try:
    import torch
    import torch.nn as nn
except ImportError as e:
    print(f"Error: PyTorch not found: {e}", file=sys.stderr)
    print(f"Python executable: {sys.executable}", file=sys.stderr)
    print(f"Python version: {sys.version}", file=sys.stderr)
    print(f"CONDA_DEFAULT_ENV: {os.environ.get('CONDA_DEFAULT_ENV', 'not set')}", file=sys.stderr)
    print(f"CONDA_PREFIX: {os.environ.get('CONDA_PREFIX', 'not set')}", file=sys.stderr)
    print("\nTrying to debug Python path...", file=sys.stderr)
    print(f"Python paths (first 5): {sys.path[:5]}", file=sys.stderr)
    print("\nPlease install PyTorch:", file=sys.stderr)
    print("  conda install pytorch torchvision torchaudio pytorch-cuda -c pytorch -c nvidia", file=sys.stderr)
    print("  or", file=sys.stderr)
    print("  pip install torch torchvision torchaudio", file=sys.stderr)
    sys.exit(1)

import numpy as np
import time
import gc
import argparse
from typing import Dict, Tuple, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from contextlib import contextmanager
import copy

# Try to import transformers for real model loading
HAS_TRANSFORMERS = False
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    HAS_TRANSFORMERS = True
    print("✓ Transformers library loaded")
except ImportError:
    print("Note: Transformers not available, will use synthetic benchmarks only")

# Simple PyTorch perturb helper (used across modes)
def pytorch_perturb_(tensor: torch.Tensor, seed: int, alpha: float):
    torch.manual_seed(seed)
    z = torch.randn_like(tensor)
    tensor.add_(z, alpha=alpha)

# Add paths for kernel imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'Perturb_wise'))
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'zo_foward_wise'))

# Import model configs - define locally to avoid import issues
MODEL_CONFIGS = {
    'opt-350m': {
        'hf_name': 'facebook/opt-350m',
        'num_layers': 24,
        'hidden_size': 1024,
        'ffn_size': 4096,
        'embed_dim': 512,
        'has_project': True,
        'total_params': 331_196_416,
    },
    'opt-1.3b': {
        'hf_name': 'facebook/opt-1.3b',
        'num_layers': 24,
        'hidden_size': 2048,
        'ffn_size': 8192,
        'embed_dim': 2048,
        'has_project': False,
        'total_params': 1_315_753_984,
    },
    'opt-2.7b': {
        'hf_name': 'facebook/opt-2.7b',
        'num_layers': 32,
        'hidden_size': 2560,
        'ffn_size': 10240,
        'embed_dim': 2560,
        'has_project': False,
        'total_params': 2_651_596_800,
    },
    'opt-6.7b': {
        'hf_name': 'facebook/opt-6.7b',
        'num_layers': 32,
        'hidden_size': 4096,
        'ffn_size': 16384,
        'embed_dim': 4096,
        'has_project': False,
        'total_params': 6_658_473_984,
    },
    'opt-13b': {
        'hf_name': 'facebook/opt-13b',
        'num_layers': 40,
        'hidden_size': 5120,
        'ffn_size': 20480,
        'embed_dim': 5120,
        'has_project': False,
        'total_params': 13_016_023_040,
    },
}


# =============================================================================
# Real Model Loading and Flat Buffer Management
# =============================================================================

@dataclass
class ParamMetadata:
    """Metadata for a single parameter in the flat buffer."""
    name: str
    shape: Tuple[int, ...]
    offset: int
    numel: int
    dtype: torch.dtype


@dataclass
class FlatBufferManager:
    """
    Manages flat buffer ↔ model parameter mapping.
    
    This class handles:
    1. Flattening model parameters into a contiguous buffer
    2. Creating views that map back to original parameter shapes
    3. Updating model parameters to use views into the flat buffer
    """
    flat_buffer: torch.Tensor
    anchor_buffer: torch.Tensor
    param_metadata: List[ParamMetadata]
    param_views: List[torch.Tensor] = field(default_factory=list)
    anchor_views: List[torch.Tensor] = field(default_factory=list)
    device: torch.device = None
    total_numel: int = 0
    
    def __post_init__(self):
        if self.device is None:
            self.device = self.flat_buffer.device


def load_opt_model(
    model_name: str,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    use_cache: bool = False,
) -> Tuple[Any, Any]:
    """
    Load an OPT model from HuggingFace.
    
    Args:
        model_name: Short name (e.g., 'opt-350m') or full HF name
        device: Target device
        dtype: Model dtype (float32 or float16)
        use_cache: Whether to use KV cache (disable for training)
    
    Returns:
        Tuple of (model, tokenizer)
    """
    if not HAS_TRANSFORMERS:
        raise RuntimeError("Transformers library not available. Install with: pip install transformers")
    
    # Get HuggingFace name
    if model_name in MODEL_CONFIGS:
        hf_name = MODEL_CONFIGS[model_name]['hf_name']
    else:
        hf_name = model_name
    
    print(f"Loading model: {hf_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(hf_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        hf_name,
        torch_dtype=dtype,
        device_map=None,  # We'll move manually
        use_cache=use_cache,
    )
    model = model.to(device)
    model.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model dtype: {next(model.parameters()).dtype}")
    print(f"  Model device: {next(model.parameters()).device}")
    
    return model, tokenizer


def flatten_model_params(
    model: nn.Module,
    device: torch.device,
    trainable_only: bool = True,
) -> FlatBufferManager:
    """
    Flatten model parameters into a contiguous buffer.
    
    Args:
        model: PyTorch model
        device: Target device
        trainable_only: Only include trainable parameters
    
    Returns:
        FlatBufferManager with flat buffer and metadata
    """
    # Collect parameter metadata
    param_metadata = []
    total_numel = 0
    
    for name, param in model.named_parameters():
        if trainable_only and not param.requires_grad:
            continue
        
        meta = ParamMetadata(
            name=name,
            shape=tuple(param.shape),
            offset=total_numel,
            numel=param.numel(),
            dtype=param.dtype,
        )
        param_metadata.append(meta)
        total_numel += param.numel()
    
    print(f"  Flattening {len(param_metadata)} parameters ({total_numel:,} elements)")
    
    # Create flat buffer
    dtype = param_metadata[0].dtype if param_metadata else torch.float32
    flat_buffer = torch.empty(total_numel, device=device, dtype=dtype)
    anchor_buffer = torch.empty(total_numel, device=device, dtype=dtype)
    
    # Copy parameter data to flat buffer
    param_idx = 0
    for name, param in model.named_parameters():
        if trainable_only and not param.requires_grad:
            continue
        
        meta = param_metadata[param_idx]
        flat_buffer[meta.offset:meta.offset + meta.numel].copy_(param.data.view(-1))
        anchor_buffer[meta.offset:meta.offset + meta.numel].copy_(param.data.view(-1))
        param_idx += 1
    
    # Create views
    param_views = []
    anchor_views = []
    for meta in param_metadata:
        param_views.append(flat_buffer[meta.offset:meta.offset + meta.numel].view(meta.shape))
        anchor_views.append(anchor_buffer[meta.offset:meta.offset + meta.numel].view(meta.shape))
    
    return FlatBufferManager(
        flat_buffer=flat_buffer,
        anchor_buffer=anchor_buffer,
        param_metadata=param_metadata,
        param_views=param_views,
        anchor_views=anchor_views,
        device=device,
        total_numel=total_numel,
    )


def set_model_params_from_flat(
    model: nn.Module,
    flat_buffer: torch.Tensor,
    param_metadata: List[ParamMetadata],
    trainable_only: bool = True,
):
    """
    Update model parameters to use views into the flat buffer.
    
    This function modifies model.param.data to point to views of the flat buffer,
    so that perturbations to the flat buffer are automatically reflected in the model.
    
    Args:
        model: PyTorch model
        flat_buffer: The flat buffer containing parameter data
        param_metadata: List of ParamMetadata describing each parameter
        trainable_only: Only update trainable parameters
    """
    param_idx = 0
    for name, param in model.named_parameters():
        if trainable_only and not param.requires_grad:
            continue
        
        meta = param_metadata[param_idx]
        # Create a view into the flat buffer with the correct shape
        view = flat_buffer[meta.offset:meta.offset + meta.numel].view(meta.shape)
        # Update param.data to point to this view
        param.data = view
        param_idx += 1


def restore_model_params_from_anchor(
    model: nn.Module,
    anchor_buffer: torch.Tensor,
    param_metadata: List[ParamMetadata],
    trainable_only: bool = True,
):
    """
    Restore model parameters from anchor buffer.
    
    Args:
        model: PyTorch model
        anchor_buffer: The anchor buffer containing original parameter data
        param_metadata: List of ParamMetadata
        trainable_only: Only restore trainable parameters
    """
    set_model_params_from_flat(model, anchor_buffer, param_metadata, trainable_only)


def create_dummy_batch(
    tokenizer: Any,
    batch_size: int = 1,
    seq_len: int = 128,
    device: torch.device = None,
) -> Dict[str, torch.Tensor]:
    """
    Create a dummy batch for forward pass.
    
    Args:
        tokenizer: HuggingFace tokenizer
        batch_size: Batch size
        seq_len: Sequence length
        device: Target device
    
    Returns:
        Dictionary with input_ids and labels
    """
    # Create dummy input
    dummy_text = "The quick brown fox jumps over the lazy dog. " * 10
    encoded = tokenizer(
        [dummy_text] * batch_size,
        max_length=seq_len,
        padding='max_length',
        truncation=True,
        return_tensors='pt',
    )
    
    input_ids = encoded['input_ids']
    attention_mask = encoded.get('attention_mask', None)
    
    if device is not None:
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
    
    # Labels are same as input_ids for causal LM
    labels = input_ids.clone()
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
    }


def real_forward(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    device: torch.device = None,
) -> torch.Tensor:
    """
    Perform a real forward pass on the model.
    
    Args:
        model: PyTorch model
        batch: Input batch dictionary
        device: Target device (for moving batch if needed)
    
    Returns:
        Loss tensor
    """
    model.eval()
    
    # Move batch to device if needed
    if device is not None:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    
    with torch.no_grad():
        outputs = model(**batch, return_dict=True)
        loss = outputs.loss
    
    return loss.item() if loss is not None else 0.0

@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    eps: float = 1e-3
    lr: float = 1e-5
    tau: float = 0.2
    zo_eps: float = 0.1
    step_size: float = 2.0

# Global model and batch storage for benchmark functions
# These are set by the main() function and used by forward functions
_GLOBAL_MODEL_MAIN = None
_GLOBAL_MODEL_SIDE = None
_GLOBAL_BATCH_MAIN = None
_GLOBAL_BATCH_SIDE = None
_GLOBAL_FLAT_MANAGER_MAIN = None
_GLOBAL_FLAT_MANAGER_SIDE = None
_USE_REAL_MODEL = False


def dummy_forward(device=None, return_tensor=False):
    """
    Forward pass - either real model or minimal simulation.
    
    If _USE_REAL_MODEL is True and models are loaded, performs real forward.
    Otherwise, performs a simple matrix multiply simulation.
    
    Args:
        device: Target device
        return_tensor: If True, return loss as tensor (for async). If False, return scalar.
                      Use return_tensor=True for parallel execution, then call .item() after sync.
    """
    global _GLOBAL_MODEL_MAIN, _GLOBAL_MODEL_SIDE, _GLOBAL_BATCH_MAIN, _GLOBAL_BATCH_SIDE
    global _USE_REAL_MODEL
    
    if device is None:
        device = torch.cuda.current_device()
    if isinstance(device, torch.device):
        device_idx = device.index if device.index is not None else 0
    else:
        device_idx = device
    
    # Use real model if available
    if _USE_REAL_MODEL:
        if device_idx == 0 and _GLOBAL_MODEL_MAIN is not None and _GLOBAL_BATCH_MAIN is not None:
            model = _GLOBAL_MODEL_MAIN
            batch = _GLOBAL_BATCH_MAIN
            with torch.no_grad():
                outputs = model(**batch, return_dict=True)
                loss = outputs.loss
            # Return tensor for async execution, or scalar if requested
            if return_tensor:
                return loss if loss is not None else torch.tensor(0.0, device=f'cuda:{device_idx}')
            return loss.item() if loss is not None else 0.0
        elif device_idx == 1 and _GLOBAL_MODEL_SIDE is not None and _GLOBAL_BATCH_SIDE is not None:
            model = _GLOBAL_MODEL_SIDE
            batch = _GLOBAL_BATCH_SIDE
            with torch.no_grad():
                outputs = model(**batch, return_dict=True)
                loss = outputs.loss
            if return_tensor:
                return loss if loss is not None else torch.tensor(0.0, device=f'cuda:{device_idx}')
            return loss.item() if loss is not None else 0.0
    
    # Fallback: simple matrix multiply simulation
    device_str = f'cuda:{device_idx}'
    x = torch.randn(256, 256, device=device_str)
    y = x @ x.T
    if return_tensor:
        return torch.tensor(1.0 + np.random.rand() * 0.1, device=device_str)
    return 1.0 + np.random.rand() * 0.1


def setup_real_models(
    model_name: str,
    device_main: torch.device,
    device_side: torch.device,
    batch_size: int = 1,
    seq_len: int = 128,
    dtype: torch.dtype = None,
    auto_dtype: bool = True,
) -> Tuple[FlatBufferManager, FlatBufferManager, Any, Any]:
    """
    Set up real OPT models on both GPUs with flat buffer management.
    
    This function:
    1. Loads the model on main GPU
    2. Copies the model to side GPU
    3. Creates flat buffers for both
    4. Sets up global state for forward passes
    
    Args:
        model_name: Model name (e.g., 'opt-350m', 'opt-1.3b', 'opt-2.7b')
        device_main: Main GPU device
        device_side: Side GPU device
        batch_size: Batch size for dummy input
        seq_len: Sequence length for dummy input
        dtype: Data type for model (torch.float32, torch.float16, torch.bfloat16)
               If None and auto_dtype=True, will auto-select based on model size and GPU memory.
        auto_dtype: If True and dtype is None, auto-select dtype based on model size.
    
    Returns:
        Tuple of (flat_manager_main, flat_manager_side, model_main, model_side)
    """
    global _GLOBAL_MODEL_MAIN, _GLOBAL_MODEL_SIDE, _GLOBAL_BATCH_MAIN, _GLOBAL_BATCH_SIDE
    global _GLOBAL_FLAT_MANAGER_MAIN, _GLOBAL_FLAT_MANAGER_SIDE, _USE_REAL_MODEL
    
    if not HAS_TRANSFORMERS:
        raise RuntimeError("Transformers not available. Install with: pip install transformers")
    
    # Auto-select dtype if not specified
    if dtype is None and auto_dtype:
        # Get available GPU memory
        torch.cuda.empty_cache()
        gc.collect()
        free_mem = torch.cuda.get_device_properties(device_main).total_memory / (1024**3)
        dtype = get_recommended_dtype(model_name, free_mem * 0.9)  # Use 90% of total memory
        if dtype != torch.float32:
            print(f"Auto-selected dtype: {dtype} (for memory efficiency with {model_name})")
    elif dtype is None:
        dtype = torch.float32
    
    print(f"\nLoading real OPT model: {model_name}")
    print(f"  Main GPU: {device_main}")
    print(f"  Side GPU: {device_side}")
    print(f"  Data type: {dtype}")
    
    # Load model on main GPU
    print("\n[Main GPU] Loading model...")
    model_main, tokenizer = load_opt_model(model_name, device_main, dtype=dtype)
    
    # Create flat buffer for main model
    print("\n[Main GPU] Creating flat buffer...")
    flat_manager_main = flatten_model_params(model_main, device_main)
    
    # Set model params to use flat buffer views
    set_model_params_from_flat(model_main, flat_manager_main.flat_buffer, flat_manager_main.param_metadata)
    
    # Copy model to side GPU
    print("\n[Side GPU] Copying model...")
    model_side = copy.deepcopy(model_main)
    model_side = model_side.to(device_side)
    
    # Create flat buffer for side model
    print("\n[Side GPU] Creating flat buffer...")
    flat_manager_side = flatten_model_params(model_side, device_side)
    
    # Set side model params to use flat buffer views
    set_model_params_from_flat(model_side, flat_manager_side.flat_buffer, flat_manager_side.param_metadata)
    
    # Create dummy batches
    print("\nCreating dummy batches...")
    batch_main = create_dummy_batch(tokenizer, batch_size, seq_len, device_main)
    batch_side = create_dummy_batch(tokenizer, batch_size, seq_len, device_side)
    
    # Verify forward passes work
    print("\nVerifying forward passes...")
    with torch.no_grad():
        loss_main = model_main(**batch_main, return_dict=True).loss
        loss_side = model_side(**batch_side, return_dict=True).loss
    print(f"  Main GPU loss: {loss_main.item():.4f}")
    print(f"  Side GPU loss: {loss_side.item():.4f}")
    
    # Set global state
    _GLOBAL_MODEL_MAIN = model_main
    _GLOBAL_MODEL_SIDE = model_side
    _GLOBAL_BATCH_MAIN = batch_main
    _GLOBAL_BATCH_SIDE = batch_side
    _GLOBAL_FLAT_MANAGER_MAIN = flat_manager_main
    _GLOBAL_FLAT_MANAGER_SIDE = flat_manager_side
    _USE_REAL_MODEL = True
    
    print("\n✓ Real models set up successfully")
    
    return flat_manager_main, flat_manager_side, model_main, model_side

# Import kernels
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    print("Warning: Triton not available")

CUDA_PERTURB_TESTED_OK = False
try:
    import fused_perturb_cuda
    HAS_CUDA_PERTURB_IMPORTED = True
    
    # Test the kernel with a small tensor to check for compatibility
    try:
        test_tensor = torch.randn(1024, device='cuda:0', dtype=torch.float32)
        fused_perturb_cuda.fused_perturb(test_tensor, 42, 0.001)
        torch.cuda.synchronize()
        CUDA_PERTURB_TESTED_OK = True
        print("✓ CUDA perturb kernels loaded and tested OK")
        del test_tensor
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"⚠ CUDA perturb kernel loaded but test FAILED: {e}")
        print("  This often happens when the kernel was compiled with a different CUDA version")
        print("  Will use PyTorch fallback for safety")
except ImportError as e:
    HAS_CUDA_PERTURB_IMPORTED = False
    print(f"Warning: CUDA perturb not available: {e}")

try:
    from dizo_fused_kernels_v2 import FusedDiZOKernelsV2
    HAS_TRITON_ZO_V2_IMPORTED = True
except ImportError:
    HAS_TRITON_ZO_V2_IMPORTED = False

try:
    import dizo_fused_kernels_cuda_v5 as cuda_zo_v5
    HAS_CUDA_ZO_V5_IMPORTED = True
except ImportError:
    HAS_CUDA_ZO_V5_IMPORTED = False

# Define Triton runtime-seed kernels if available
HAS_TRITON_RUNTIME_SEED = False
fused_perturb_runtime_seed = None
fused_update_runtime_seed = None

if HAS_TRITON:
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
        seed_ptr,
        alpha,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        seed = tl.load(seed_ptr)
        pid = tl.program_id(0)
        block_start = pid.to(tl.int64) * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE).to(tl.int64)
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
        seed_ptr,
        projected_grad,
        lr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        seed = tl.load(seed_ptr)
        pid = tl.program_id(0)
        block_start = pid.to(tl.int64) * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE).to(tl.int64)
        mask = offsets < n_elements
        params = tl.load(params_ptr + offsets, mask=mask, other=0.0)
        z = tl.randn(seed, offsets)
        result = params - lr * projected_grad * z
        tl.store(params_ptr + offsets, result, mask=mask)
    
    HAS_TRITON_RUNTIME_SEED = True
    print("✓ Triton runtime-seed kernels defined")


@dataclass
class MultiGPUBenchmarkResult:
    """Container for multi-GPU benchmark results."""
    method: str
    total_time_ms: float
    memory_main_mb: float
    memory_side_mb: float
    model_sync_ms: float
    perturb_main_ms: float
    perturb_side_ms: float
    forward_main_ms: float
    forward_side_ms: float
    loss_sync_ms: float
    update_ms: float
    speedup_vs_sequential: float = 0.0
    notes: str = ""


def create_flat_buffers_for_multi_gpu(config: Dict, device_main: torch.device, device_side: torch.device):
    """
    Create flat buffers on both GPUs for multi-GPU parallel execution.
    
    Returns:
        Tuple of (param_flat_main, anchor_flat_main, param_flat_side, anchor_flat_side,
                 offsets, sizes, param_views_main, anchor_views_main, 
                 param_views_side, anchor_views_side)
    """
    num_layers = config['num_layers']
    hidden = config['hidden_size']
    ffn = config['ffn_size']
    embed_dim = config.get('embed_dim', hidden)
    has_project = config.get('has_project', False)
    vocab_size = 50272
    max_pos = 2050
    
    # Compute sizes
    sizes_list = []
    sizes_list.append(vocab_size * embed_dim)
    sizes_list.append(max_pos * hidden)
    sizes_list.append(hidden)
    sizes_list.append(hidden)
    
    if has_project:
        sizes_list.append(hidden * embed_dim)
        sizes_list.append(embed_dim * hidden)
    
    for _ in range(num_layers):
        for _ in range(4):
            sizes_list.append(hidden * hidden)
            sizes_list.append(hidden)
        sizes_list.append(hidden)
        sizes_list.append(hidden)
        sizes_list.append(ffn * hidden)
        sizes_list.append(ffn)
        sizes_list.append(hidden * ffn)
        sizes_list.append(hidden)
        sizes_list.append(hidden)
        sizes_list.append(hidden)
    
    total_elements = sum(sizes_list)
    num_params = len(sizes_list)
    
    # Compute offsets
    offsets_list = []
    offset = 0
    for size in sizes_list:
        offsets_list.append(offset)
        offset += size
    
    offsets = torch.tensor(offsets_list, device=device_main, dtype=torch.long)
    sizes = torch.tensor(sizes_list, device=device_main, dtype=torch.long)
    
    # Create buffers on both GPUs
    param_flat_main = torch.randn(total_elements, device=device_main, dtype=torch.float32)
    anchor_flat_main = torch.randn(total_elements, device=device_main, dtype=torch.float32)
    
    param_flat_side = torch.randn(total_elements, device=device_side, dtype=torch.float32)
    anchor_flat_side = torch.randn(total_elements, device=device_side, dtype=torch.float32)
    
    # Create views
    param_views_main = []
    anchor_views_main = []
    param_views_side = []
    anchor_views_side = []
    
    for i, size in enumerate(sizes_list):
        offset = offsets_list[i]
        param_views_main.append(param_flat_main[offset:offset+size])
        anchor_views_main.append(anchor_flat_main[offset:offset+size])
        param_views_side.append(param_flat_side[offset:offset+size])
        anchor_views_side.append(anchor_flat_side[offset:offset+size])
    
    print(f"  Created flat buffers: {total_elements:,} elements")
    print(f"  Main GPU memory: {param_flat_main.numel() * 4 * 2 / 1024**2:.2f} MB")
    print(f"  Side GPU memory: {param_flat_side.numel() * 4 * 2 / 1024**2:.2f} MB")
    
    return (param_flat_main, anchor_flat_main, param_flat_side, anchor_flat_side,
            offsets, sizes, param_views_main, anchor_views_main,
            param_views_side, anchor_views_side)


def sync_models_multi_gpu(
    param_flat_main: torch.Tensor,
    anchor_flat_main: torch.Tensor,
    param_flat_side: torch.Tensor,
    anchor_flat_side: torch.Tensor,
    device_main: torch.device,
    device_side: torch.device,
) -> float:
    """
    Synchronize model parameters from main GPU to side GPU.
    
    Returns:
        Sync time in milliseconds
    """
    # Events must be on the same device for elapsed_time
    # Use time.perf_counter for cross-device timing instead
    torch.cuda.synchronize(device_main)
    torch.cuda.synchronize(device_side)
    
    start_time = time.perf_counter()
    
    # Copy main GPU buffers to side GPU
    param_flat_side.copy_(param_flat_main, non_blocking=True)
    anchor_flat_side.copy_(anchor_flat_main, non_blocking=True)
    
    torch.cuda.synchronize(device_main)
    torch.cuda.synchronize(device_side)
    
    end_time = time.perf_counter()
    
    return (end_time - start_time) * 1000  # Convert to ms


def check_p2p_access(device_main: torch.device, device_side: torch.device) -> bool:
    """Check if P2P (peer-to-peer) memory access is available between two GPUs."""
    try:
        main_idx = device_main.index if device_main.index is not None else 0
        side_idx = device_side.index if device_side.index is not None else 1
        can_access = torch.cuda.can_device_access_peer(main_idx, side_idx)
        return can_access
    except Exception:
        return False


def enable_p2p_access(device_main: torch.device, device_side: torch.device) -> bool:
    """Enable P2P access between two GPUs if available."""
    try:
        main_idx = device_main.index if device_main.index is not None else 0
        side_idx = device_side.index if device_side.index is not None else 1
        
        if torch.cuda.can_device_access_peer(main_idx, side_idx):
            # Enable P2P access in both directions
            with torch.cuda.device(device_main):
                torch.cuda.set_device(device_main)
            with torch.cuda.device(device_side):
                torch.cuda.set_device(device_side)
            return True
    except Exception:
        pass
    return False


def sync_models_p2p_async(
    param_flat_main: torch.Tensor,
    param_flat_side: torch.Tensor,
    stream_sync: torch.cuda.Stream,
) -> None:
    """
    Async P2P sync: copies main params to side GPU on a separate stream.
    Call stream_sync.synchronize() when you need the sync to complete.
    """
    with torch.cuda.stream(stream_sync):
        param_flat_side.copy_(param_flat_main, non_blocking=True)


def generate_z_on_both_gpus(
    seed: int,
    total_numel: int,
    device_main: torch.device,
    device_side: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate the same random vector z on both GPUs.
    
    Returns:
        Tuple of (z_main, z_side)
    """
    # Generate on CPU first to ensure same values
    torch.manual_seed(seed)
    z_cpu = torch.randn(total_numel, dtype=torch.float32)
    
    # Copy to both GPUs
    z_main = z_cpu.to(device_main, dtype=dtype, non_blocking=True)
    z_side = z_cpu.to(device_side, dtype=dtype, non_blocking=True)
    
    return z_main, z_side


def get_recommended_batch_size(model_name: str, seq_len: int, dtype: torch.dtype, available_memory_gb: float = 48.0) -> int:
    """
    Get recommended batch size based on model size and available GPU memory.
    
    Args:
        model_name: Name of the model (e.g., 'opt-350m', 'opt-1.3b')
        seq_len: Sequence length
        dtype: Data type (torch.float32, torch.float16, torch.bfloat16)
        available_memory_gb: Available GPU memory in GB
    
    Returns:
        Recommended batch size
    """
    # Rough memory estimates (model + activations + buffers)
    bytes_per_param = 4 if dtype == torch.float32 else 2
    
    model_params = {
        'opt-350m': 331_196_416,
        'opt-1.3b': 1_315_753_984,
        'opt-2.7b': 2_651_596_800,
        'opt-6.7b': 6_658_473_984,
        'opt-13b': 13_016_023_040,
    }
    
    num_params = model_params.get(model_name, 331_196_416)
    model_memory_gb = (num_params * bytes_per_param) / (1024**3)
    
    # For MeZO we need: model + anchor + perturbed buffer + activations
    # Multi-GPU doubles this (one copy per GPU)
    base_memory_gb = model_memory_gb * 3  # model + 2 buffers
    
    # Activation memory scales with batch_size * seq_len * hidden_dim
    # Rough estimate: 0.5GB per batch element for OPT models at seq_len=1024
    activation_per_batch_gb = 0.3 * (seq_len / 1024)
    
    available_for_batch = available_memory_gb - base_memory_gb - 2.0  # 2GB headroom
    
    if available_for_batch <= 0:
        return 1
    
    recommended = int(available_for_batch / activation_per_batch_gb)
    return max(1, min(recommended, 32))  # Cap at 32


def get_recommended_dtype(model_name: str, available_memory_gb: float = 48.0) -> torch.dtype:
    """
    Get recommended dtype based on model size.
    
    Large models (>2B params) benefit from fp16/bf16 to fit in memory.
    """
    model_params = {
        'opt-350m': 331_196_416,
        'opt-1.3b': 1_315_753_984,
        'opt-2.7b': 2_651_596_800,
        'opt-6.7b': 6_658_473_984,
        'opt-13b': 13_016_023_040,
    }
    
    num_params = model_params.get(model_name, 331_196_416)
    model_memory_fp32_gb = (num_params * 4) / (1024**3)
    
    # Need ~3x model size for MeZO (model + 2 buffers), times 2 for multi-GPU
    required_memory_gb = model_memory_fp32_gb * 6
    
    if required_memory_gb > available_memory_gb:
        # Check if bf16 is supported
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        else:
            return torch.float16
    
    return torch.float32


def benchmark_multi_gpu_parallel(
    param_flat_main: torch.Tensor,
    anchor_flat_main: torch.Tensor,
    param_flat_side: torch.Tensor,
    anchor_flat_side: torch.Tensor,
    offsets: torch.Tensor,
    sizes: torch.Tensor,
    cfg: TrainingConfig,
    n_iter: int,
    device_main: torch.device,
    device_side: torch.device,
    use_cuda_perturb: bool = True,
    use_pytorch_perturb: bool = False,
    include_dizo: bool = False,
    debug: bool = False,
) -> MultiGPUBenchmarkResult:
    """
    Benchmark multi-GPU parallel MeZO step.
    
    Args:
        use_cuda_perturb: Use CUDA kernels for perturbation (faster than Triton)
        include_dizo: Include DiZO constraint operations (not yet implemented for multi-GPU)
        debug: Enable detailed timing breakdown
    """
    if include_dizo:
        print("Warning: DiZO constraints not yet implemented for multi-GPU. Running MeZO-only.")
        include_dizo = False
    
    n_elements = param_flat_main.numel()
    dtype = param_flat_main.dtype
    
    # Create CUDA streams
    stream_main = torch.cuda.Stream(device=device_main)
    stream_side = torch.cuda.Stream(device=device_side)
    
    # Helper to perturb tensors in PyTorch (fallback / safe mode)
    def pytorch_perturb_(tensor: torch.Tensor, seed: int, alpha: float):
        torch.manual_seed(seed)
        z = torch.randn_like(tensor)
        tensor.add_(z, alpha=alpha)

    # Setup perturb kernels
    if use_pytorch_perturb:
        perturb_mode = "PYTORCH"
    elif use_cuda_perturb and CUDA_PERTURB_TESTED_OK:
        perturb_mode = "CUDA"
    elif HAS_TRITON_RUNTIME_SEED:
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        seed_tensor_main = torch.tensor([0], dtype=torch.int64, device=device_main)
        seed_tensor_side = torch.tensor([0], dtype=torch.int64, device=device_side)
        perturb_mode = "Triton"
    else:
        raise RuntimeError("No perturb kernels available!")
    
    print(f"Using {perturb_mode} perturb kernels")
    
    # Warmup
    print(f"Warming up ({5} iterations)...")
    for _ in range(5):
        seed = 42
        # NOTE: z is generated internally by the fused perturb kernels from seed
        # No need to call generate_z_on_both_gpus - it was adding ~80ms overhead per call!
        
        if perturb_mode == "PYTORCH":
            pytorch_perturb_(param_flat_main, seed, cfg.eps)
            pytorch_perturb_(param_flat_side, seed, -cfg.eps)
        elif perturb_mode == "CUDA":
            with torch.cuda.device(device_main):
                fused_perturb_cuda.fused_perturb(param_flat_main, seed, cfg.eps)
            with torch.cuda.device(device_side):
                fused_perturb_cuda.fused_perturb(param_flat_side, seed, -cfg.eps)
        else:
            seed_tensor_main[0] = seed
            seed_tensor_side[0] = seed
            with torch.cuda.device(device_main):
                fused_perturb_runtime_seed[grid](param_flat_main, seed_tensor_main, cfg.eps, n_elements)
            with torch.cuda.device(device_side):
                fused_perturb_runtime_seed[grid](param_flat_side, seed_tensor_side, -cfg.eps, n_elements)
    
    torch.cuda.synchronize(device_main)
    torch.cuda.synchronize(device_side)
    
    # Reset buffers
    param_flat_main.copy_(anchor_flat_main)
    param_flat_side.copy_(anchor_flat_side)
    torch.cuda.synchronize(device_main)
    torch.cuda.synchronize(device_side)
    
    # Benchmark
    print(f"Benchmarking ({n_iter} iterations)...")
    
    timings = {
        'model_sync': [],
        'perturb_main': [],
        'perturb_side': [],
        'forward_main': [],
        'forward_side': [],
        'loss_sync': [],
        'update': [],
        'total': [],
    }
    
    for i in range(n_iter):
        seed = np.random.randint(1000000000)
        
        # Total timing
        total_start = torch.cuda.Event(enable_timing=True)
        total_end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.device(device_main):
            total_start.record(stream_main)
        
        # === 1. Synchronize models (main → side) ===
        sync_time = sync_models_multi_gpu(
            param_flat_main, anchor_flat_main,
            param_flat_side, anchor_flat_side,
            device_main, device_side
        )
        timings['model_sync'].append(sync_time)
        
        # NOTE: z generation is handled by the fused perturb kernels from seed
        # No need to explicitly generate z here - it was causing ~80ms overhead per iteration!
        
        # === 2. Parallel perturbations ===
        # Main GPU: θ₀ + εz
        start_main = torch.cuda.Event(enable_timing=True)
        end_main = torch.cuda.Event(enable_timing=True)
        start_main.record(stream_main)
        
        with torch.cuda.device(device_main):
            with torch.cuda.stream(stream_main):
                if perturb_mode == "PYTORCH":
                    pytorch_perturb_(param_flat_main, seed, cfg.eps)
                elif perturb_mode == "CUDA":
                    fused_perturb_cuda.fused_perturb(param_flat_main, seed, cfg.eps)
                else:
                    seed_tensor_main[0] = seed
                    fused_perturb_runtime_seed[grid](param_flat_main, seed_tensor_main, cfg.eps, n_elements)
        
        # Side GPU: θ₀ - εz (parallel!)
        start_side = torch.cuda.Event(enable_timing=True)
        end_side = torch.cuda.Event(enable_timing=True)
        start_side.record(stream_side)
        
        with torch.cuda.device(device_side):
            with torch.cuda.stream(stream_side):
                if perturb_mode == "PYTORCH":
                    pytorch_perturb_(param_flat_side, seed, -cfg.eps)
                elif perturb_mode == "CUDA":
                    fused_perturb_cuda.fused_perturb(param_flat_side, seed, -cfg.eps)
                else:
                    seed_tensor_side[0] = seed
                    fused_perturb_runtime_seed[grid](param_flat_side, seed_tensor_side, -cfg.eps, n_elements)
        
        end_main.record(stream_main)
        end_side.record(stream_side)
        torch.cuda.synchronize(device_main)
        torch.cuda.synchronize(device_side)
        
        timings['perturb_main'].append(start_main.elapsed_time(end_main))
        timings['perturb_side'].append(start_side.elapsed_time(end_side))
        
        # === 4. Parallel forward passes ===
        # CRITICAL: Use return_tensor=True to avoid .item() sync that destroys parallelism!
        # Main GPU: forward1
        start_main = torch.cuda.Event(enable_timing=True)
        end_main = torch.cuda.Event(enable_timing=True)
        start_main.record(stream_main)
        
        with torch.cuda.device(device_main):
            with torch.cuda.stream(stream_main):
                loss1_tensor = dummy_forward(device_main, return_tensor=True)
        
        # Side GPU: forward2 (parallel!)
        start_side = torch.cuda.Event(enable_timing=True)
        end_side = torch.cuda.Event(enable_timing=True)
        start_side.record(stream_side)
        
        with torch.cuda.device(device_side):
            with torch.cuda.stream(stream_side):
                loss2_tensor = dummy_forward(device_side, return_tensor=True)
        
        end_main.record(stream_main)
        end_side.record(stream_side)
        
        # Now wait for both GPUs to complete (truly parallel execution happened above!)
        torch.cuda.synchronize(device_main)
        torch.cuda.synchronize(device_side)
        
        timings['forward_main'].append(start_main.elapsed_time(end_main))
        timings['forward_side'].append(start_side.elapsed_time(end_side))
        
        # === 5. Transfer loss2 to main GPU and convert to scalars ===
        # Now it's safe to call .item() since both forwards are complete
        sync_start = time.perf_counter()
        
        loss1 = loss1_tensor.item()
        loss2 = loss2_tensor.item()
        
        sync_end = time.perf_counter()
        timings['loss_sync'].append((sync_end - sync_start) * 1000)
        
        # === 6. Compute gradient estimate ===
        projected_grad = ((loss1 - loss2) / (2 * cfg.eps))
        
        # === 7. Update on main GPU ===
        start_update = torch.cuda.Event(enable_timing=True)
        end_update = torch.cuda.Event(enable_timing=True)
        start_update.record(stream_main)
        
        with torch.cuda.device(device_main):
            if perturb_mode == "PYTORCH":
                pytorch_perturb_(param_flat_main, seed, -cfg.lr * projected_grad)
            elif perturb_mode == "CUDA":
                fused_perturb_cuda.fused_update(param_flat_main, seed, float(projected_grad), cfg.lr)
            else:
                seed_tensor_main[0] = seed
                fused_update_runtime_seed[grid](param_flat_main, seed_tensor_main, float(projected_grad), cfg.lr, n_elements)
        
        end_update.record(stream_main)
        torch.cuda.synchronize(device_main)
        timings['update'].append(start_update.elapsed_time(end_update))
        
        total_end.record(stream_main)
        torch.cuda.synchronize(device_main)
        timings['total'].append(total_start.elapsed_time(total_end))
    
    # Compute statistics (skip first iteration)
    results = {}
    for key in timings:
        values = timings[key][1:] if len(timings[key]) > 1 else timings[key]
        results[key] = {
            'mean': np.mean(values),
            'std': np.std(values) if len(values) > 1 else 0.0,
        }
    
    # Get memory stats
    memory_main = torch.cuda.max_memory_allocated(device_main) / 1024**2
    memory_side = torch.cuda.max_memory_allocated(device_side) / 1024**2
    
    return MultiGPUBenchmarkResult(
        method=f"Multi-GPU Parallel ({perturb_mode})",
        total_time_ms=results['total']['mean'],
        memory_main_mb=memory_main,
        memory_side_mb=memory_side,
        model_sync_ms=results['model_sync']['mean'],
        perturb_main_ms=results['perturb_main']['mean'],
        perturb_side_ms=results['perturb_side']['mean'],
        forward_main_ms=results['forward_main']['mean'],
        forward_side_ms=results['forward_side']['mean'],
        loss_sync_ms=results['loss_sync']['mean'],
        update_ms=results['update']['mean'],
        notes=f"Main GPU: {device_main}, Side GPU: {device_side}",
    )


def benchmark_multi_gpu_optimized(
    param_flat_main: torch.Tensor,
    anchor_flat_main: torch.Tensor,
    param_flat_side: torch.Tensor,
    anchor_flat_side: torch.Tensor,
    offsets: torch.Tensor,
    sizes: torch.Tensor,
    cfg: TrainingConfig,
    n_iter: int,
    device_main: torch.device,
    device_side: torch.device,
    use_cuda_perturb: bool = True,
    use_pytorch_perturb: bool = False,
    debug: bool = False,
) -> MultiGPUBenchmarkResult:
    """
    OPTIMIZED multi-GPU parallel MeZO with overlapped model sync.
    
    Key optimizations:
    1. Model sync is overlapped with perturb on main GPU
    2. Uses P2P transfers if available
    3. Async operations to maximize parallelism
    
    Timeline:
        Main GPU: [perturb +ε] -------- [forward1] ----- [update]
        Side GPU: [sync+perturb -ε] --- [forward2] ---
        
    The sync to side GPU runs in parallel with main GPU's perturb!
    """
    n_elements = param_flat_main.numel()
    dtype = param_flat_main.dtype
    
    # Create CUDA streams
    stream_main = torch.cuda.Stream(device=device_main)
    stream_side = torch.cuda.Stream(device=device_side)
    stream_sync = torch.cuda.Stream(device=device_side)  # Dedicated sync stream
    
    # Check P2P capability
    p2p_enabled = check_p2p_access(device_main, device_side)
    if p2p_enabled:
        enable_p2p_access(device_main, device_side)
    
    # Helper for perturb
    def pytorch_perturb_(tensor: torch.Tensor, seed: int, alpha: float):
        torch.manual_seed(seed)
        z = torch.randn_like(tensor)
        tensor.add_(z, alpha=alpha)

    # Setup perturb kernels
    if use_pytorch_perturb:
        perturb_mode = "PYTORCH"
    elif use_cuda_perturb and CUDA_PERTURB_TESTED_OK:
        perturb_mode = "CUDA"
    elif HAS_TRITON_RUNTIME_SEED:
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        seed_tensor_main = torch.tensor([0], dtype=torch.int64, device=device_main)
        seed_tensor_side = torch.tensor([0], dtype=torch.int64, device=device_side)
        perturb_mode = "Triton"
    else:
        raise RuntimeError("No perturb kernels available!")
    
    print(f"Using {perturb_mode} perturb kernels (P2P: {'enabled' if p2p_enabled else 'disabled'})")
    
    # Warmup
    print(f"Warming up ({5} iterations)...")
    for _ in range(5):
        seed = 42
        if perturb_mode == "PYTORCH":
            pytorch_perturb_(param_flat_main, seed, cfg.eps)
            pytorch_perturb_(param_flat_side, seed, -cfg.eps)
        elif perturb_mode == "CUDA":
            with torch.cuda.device(device_main):
                fused_perturb_cuda.fused_perturb(param_flat_main, seed, cfg.eps)
            with torch.cuda.device(device_side):
                fused_perturb_cuda.fused_perturb(param_flat_side, seed, -cfg.eps)
        else:
            seed_tensor_main[0] = seed
            seed_tensor_side[0] = seed
            with torch.cuda.device(device_main):
                fused_perturb_runtime_seed[grid](param_flat_main, seed_tensor_main, cfg.eps, n_elements)
            with torch.cuda.device(device_side):
                fused_perturb_runtime_seed[grid](param_flat_side, seed_tensor_side, -cfg.eps, n_elements)
    
    torch.cuda.synchronize(device_main)
    torch.cuda.synchronize(device_side)
    
    # Reset buffers
    param_flat_main.copy_(anchor_flat_main)
    param_flat_side.copy_(anchor_flat_side)
    torch.cuda.synchronize(device_main)
    torch.cuda.synchronize(device_side)
    
    # Benchmark
    print(f"Benchmarking ({n_iter} iterations) with overlapped sync...")
    
    timings = {
        'model_sync': [],
        'perturb_main': [],
        'perturb_side': [],
        'forward_main': [],
        'forward_side': [],
        'loss_sync': [],
        'update': [],
        'total': [],
    }
    
    for i in range(n_iter):
        seed = np.random.randint(1000000000)
        
        # Total timing
        total_start = time.perf_counter()
        
        # === OPTIMIZED: Overlap model sync with main GPU perturb ===
        # Start async sync to side GPU
        sync_start = time.perf_counter()
        with torch.cuda.stream(stream_sync):
            param_flat_side.copy_(param_flat_main, non_blocking=True)
        
        # Main GPU: perturb +εz (runs in parallel with sync!)
        start_main = torch.cuda.Event(enable_timing=True)
        end_main = torch.cuda.Event(enable_timing=True)
        start_main.record(stream_main)
        
        with torch.cuda.device(device_main):
            with torch.cuda.stream(stream_main):
                if perturb_mode == "PYTORCH":
                    pytorch_perturb_(param_flat_main, seed, cfg.eps)
                elif perturb_mode == "CUDA":
                    fused_perturb_cuda.fused_perturb(param_flat_main, seed, cfg.eps)
                else:
                    seed_tensor_main[0] = seed
                    fused_perturb_runtime_seed[grid](param_flat_main, seed_tensor_main, cfg.eps, n_elements)
        
        end_main.record(stream_main)
        
        # Wait for sync to complete before perturbing side GPU
        stream_sync.synchronize()
        sync_end = time.perf_counter()
        timings['model_sync'].append((sync_end - sync_start) * 1000)
        
        # Side GPU: perturb -εz
        start_side = torch.cuda.Event(enable_timing=True)
        end_side = torch.cuda.Event(enable_timing=True)
        start_side.record(stream_side)
        
        with torch.cuda.device(device_side):
            with torch.cuda.stream(stream_side):
                if perturb_mode == "PYTORCH":
                    pytorch_perturb_(param_flat_side, seed, -cfg.eps)
                elif perturb_mode == "CUDA":
                    fused_perturb_cuda.fused_perturb(param_flat_side, seed, -cfg.eps)
                else:
                    seed_tensor_side[0] = seed
                    fused_perturb_runtime_seed[grid](param_flat_side, seed_tensor_side, -cfg.eps, n_elements)
        
        end_side.record(stream_side)
        torch.cuda.synchronize(device_main)
        torch.cuda.synchronize(device_side)
        
        timings['perturb_main'].append(start_main.elapsed_time(end_main))
        timings['perturb_side'].append(start_side.elapsed_time(end_side))
        
        # === Parallel forward passes ===
        start_main = torch.cuda.Event(enable_timing=True)
        end_main = torch.cuda.Event(enable_timing=True)
        start_main.record(stream_main)
        
        with torch.cuda.device(device_main):
            with torch.cuda.stream(stream_main):
                loss1_tensor = dummy_forward(device_main, return_tensor=True)
        
        start_side = torch.cuda.Event(enable_timing=True)
        end_side = torch.cuda.Event(enable_timing=True)
        start_side.record(stream_side)
        
        with torch.cuda.device(device_side):
            with torch.cuda.stream(stream_side):
                loss2_tensor = dummy_forward(device_side, return_tensor=True)
        
        end_main.record(stream_main)
        end_side.record(stream_side)
        torch.cuda.synchronize(device_main)
        torch.cuda.synchronize(device_side)
        
        timings['forward_main'].append(start_main.elapsed_time(end_main))
        timings['forward_side'].append(start_side.elapsed_time(end_side))
        
        # === Loss sync and gradient computation ===
        loss_sync_start = time.perf_counter()
        loss1 = loss1_tensor.item()
        loss2 = loss2_tensor.item()
        loss_sync_end = time.perf_counter()
        timings['loss_sync'].append((loss_sync_end - loss_sync_start) * 1000)
        
        projected_grad = ((loss1 - loss2) / (2 * cfg.eps))
        
        # === Update on main GPU ===
        start_update = torch.cuda.Event(enable_timing=True)
        end_update = torch.cuda.Event(enable_timing=True)
        start_update.record(stream_main)
        
        with torch.cuda.device(device_main):
            if perturb_mode == "PYTORCH":
                pytorch_perturb_(param_flat_main, seed, -cfg.lr * float(projected_grad))
            elif perturb_mode == "CUDA":
                fused_perturb_cuda.fused_update(param_flat_main, seed, float(projected_grad), cfg.lr)
            else:
                seed_tensor_main[0] = seed
                fused_update_runtime_seed[grid](param_flat_main, seed_tensor_main, float(projected_grad), cfg.lr, n_elements)
        
        end_update.record(stream_main)
        torch.cuda.synchronize(device_main)
        timings['update'].append(start_update.elapsed_time(end_update))
        
        total_end = time.perf_counter()
        timings['total'].append((total_end - total_start) * 1000)
    
    # Compute statistics
    results = {}
    for key in timings:
        values = timings[key][1:] if len(timings[key]) > 1 else timings[key]
        results[key] = {
            'mean': np.mean(values),
            'std': np.std(values) if len(values) > 1 else 0.0,
        }
    
    memory_main = torch.cuda.max_memory_allocated(device_main) / 1024**2
    memory_side = torch.cuda.max_memory_allocated(device_side) / 1024**2
    
    return MultiGPUBenchmarkResult(
        method=f"Multi-GPU Optimized ({perturb_mode})" + (" +P2P" if p2p_enabled else ""),
        total_time_ms=results['total']['mean'],
        memory_main_mb=memory_main,
        memory_side_mb=memory_side,
        model_sync_ms=results['model_sync']['mean'],
        perturb_main_ms=results['perturb_main']['mean'],
        perturb_side_ms=results['perturb_side']['mean'],
        forward_main_ms=results['forward_main']['mean'],
        forward_side_ms=results['forward_side']['mean'],
        loss_sync_ms=results['loss_sync']['mean'],
        update_ms=results['update']['mean'],
        notes=f"Main GPU: {device_main}, Side GPU: {device_side}, Overlapped sync" + (" +P2P" if p2p_enabled else ""),
    )


def benchmark_single_gpu_parallel(
    param_flat: torch.Tensor,
    anchor_flat: torch.Tensor,
    offsets: torch.Tensor,
    sizes: torch.Tensor,
    cfg: TrainingConfig,
    n_iter: int,
    device: torch.device,
    use_cuda_perturb: bool = True,
    use_pytorch_perturb: bool = False,
) -> MultiGPUBenchmarkResult:
    """
    Single-GPU parallel approach using two buffers and CUDA streams.
    
    Strategy:
    - Buffer 1: θ₀ + εz → forward1 (stream1)
    - Buffer 2: θ₀ - εz → forward2 (stream2)
    - Both forwards dispatched to different streams
    
    NOTE: On a single GPU with a single model, true parallel forward execution is limited
    because the model's forward pass uses the same parameters. This benchmark measures
    the potential overlap from stream-level concurrency (memory ops, etc).
    For true parallel execution, use multi-GPU mode with two model copies.
    
    When using real models, this swaps the model's param buffer between the two perturbed
    versions for each forward, so the forwards run sequentially with different params.
    """
    global _GLOBAL_FLAT_MANAGER_MAIN, _USE_REAL_MODEL, _GLOBAL_MODEL_MAIN
    
    n_elements = param_flat.numel()
    dtype = param_flat.dtype
    
    # For real models, we need to swap the model's flat buffer between the two perturbed versions
    use_real_model_swap = _USE_REAL_MODEL and _GLOBAL_FLAT_MANAGER_MAIN is not None
    
    # Create two buffers for parallel execution (or real model buffer swapping)
    param_buffer_plus = torch.empty_like(param_flat)
    param_buffer_minus = torch.empty_like(param_flat)
    
    # Create CUDA streams
    stream1 = torch.cuda.Stream(device=device)
    stream2 = torch.cuda.Stream(device=device)
    
    # Setup perturb kernels
    if use_pytorch_perturb:
        perturb_mode = "PYTORCH"
    elif use_cuda_perturb and CUDA_PERTURB_TESTED_OK:
        perturb_mode = "CUDA"
    elif HAS_TRITON_RUNTIME_SEED:
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        seed_tensor = torch.tensor([0], dtype=torch.int64, device=device)
        perturb_mode = "Triton"
    else:
        raise RuntimeError("No perturb kernels available!")
    
    # Warmup
    for _ in range(5):
        seed = 42
        param_buffer_plus.copy_(anchor_flat)
        param_buffer_minus.copy_(anchor_flat)
        
        if perturb_mode == "PYTORCH":
            pytorch_perturb_(param_buffer_plus, seed, cfg.eps)
            pytorch_perturb_(param_buffer_minus, seed, -cfg.eps)
        elif perturb_mode == "CUDA":
            fused_perturb_cuda.fused_perturb(param_buffer_plus, seed, cfg.eps)
            fused_perturb_cuda.fused_perturb(param_buffer_minus, seed, -cfg.eps)
        else:
            seed_tensor[0] = seed
            fused_perturb_runtime_seed[grid](param_buffer_plus, seed_tensor, cfg.eps, n_elements)
            seed_tensor[0] = seed
            fused_perturb_runtime_seed[grid](param_buffer_minus, seed_tensor, -cfg.eps, n_elements)
    
    torch.cuda.synchronize(device)
    param_flat.copy_(anchor_flat)
    torch.cuda.synchronize(device)
    
    # Benchmark
    timings = {
        'buffer_copy': [],
        'perturb_plus': [],
        'perturb_minus': [],
        'forward_plus': [],
        'forward_minus': [],
        'update': [],
        'total': [],
    }
    
    for i in range(n_iter):
        seed = np.random.randint(1000000000)
        
        total_start = torch.cuda.Event(enable_timing=True)
        total_end = torch.cuda.Event(enable_timing=True)
        total_start.record(stream1)
        
        # Copy base to both buffers
        start_copy = torch.cuda.Event(enable_timing=True)
        end_copy = torch.cuda.Event(enable_timing=True)
        start_copy.record(stream1)
        
        with torch.cuda.stream(stream1):
            param_buffer_plus.copy_(anchor_flat, non_blocking=True)
        
        with torch.cuda.stream(stream2):
            param_buffer_minus.copy_(anchor_flat, non_blocking=True)
        
        torch.cuda.synchronize(device)
        end_copy.record(stream1)
        torch.cuda.synchronize(device)
        timings['buffer_copy'].append(start_copy.elapsed_time(end_copy))
        
        # Parallel perturbations
        start_plus = torch.cuda.Event(enable_timing=True)
        end_plus = torch.cuda.Event(enable_timing=True)
        start_plus.record(stream1)
        
        with torch.cuda.device(device):
            with torch.cuda.stream(stream1):
                if perturb_mode == "PYTORCH":
                    pytorch_perturb_(param_buffer_plus, seed, cfg.eps)
                elif perturb_mode == "CUDA":
                    fused_perturb_cuda.fused_perturb(param_buffer_plus, seed, cfg.eps)
                else:
                    seed_tensor[0] = seed
                    fused_perturb_runtime_seed[grid](param_buffer_plus, seed_tensor, cfg.eps, n_elements)
        
        start_minus = torch.cuda.Event(enable_timing=True)
        end_minus = torch.cuda.Event(enable_timing=True)
        start_minus.record(stream2)
        
        with torch.cuda.device(device):
            with torch.cuda.stream(stream2):
                if perturb_mode == "PYTORCH":
                    pytorch_perturb_(param_buffer_minus, seed, -cfg.eps)
                elif perturb_mode == "CUDA":
                    fused_perturb_cuda.fused_perturb(param_buffer_minus, seed, -cfg.eps)
                else:
                    seed_tensor[0] = seed
                    fused_perturb_runtime_seed[grid](param_buffer_minus, seed_tensor, -cfg.eps, n_elements)
        
        end_plus.record(stream1)
        end_minus.record(stream2)
        torch.cuda.synchronize(device)
        timings['perturb_plus'].append(start_plus.elapsed_time(end_plus))
        timings['perturb_minus'].append(start_minus.elapsed_time(end_minus))
        
        # Forward passes
        # For real models: swap model's flat buffer to use each perturbed version
        # For synthetic: dispatch both forwards to different streams
        start_plus = torch.cuda.Event(enable_timing=True)
        end_plus = torch.cuda.Event(enable_timing=True)
        
        if use_real_model_swap:
            # Real model: copy perturbed buffer to model's flat buffer, run forward
            start_plus.record(stream1)
            with torch.cuda.device(device):
                with torch.cuda.stream(stream1):
                    param_flat.copy_(param_buffer_plus)
                    loss1_tensor = dummy_forward(device, return_tensor=True)
            end_plus.record(stream1)
            torch.cuda.synchronize(device)
            timings['forward_plus'].append(start_plus.elapsed_time(end_plus))
            
            # Forward 2: swap to minus buffer and run
            start_minus = torch.cuda.Event(enable_timing=True)
            end_minus = torch.cuda.Event(enable_timing=True)
            start_minus.record(stream1)
            with torch.cuda.device(device):
                with torch.cuda.stream(stream1):
                    param_flat.copy_(param_buffer_minus)
                    loss2_tensor = dummy_forward(device, return_tensor=True)
            end_minus.record(stream1)
            torch.cuda.synchronize(device)
            timings['forward_minus'].append(start_minus.elapsed_time(end_minus))
        else:
            # Synthetic mode: dispatch both forwards (without model param swap)
            start_plus.record(stream1)
            with torch.cuda.device(device):
                with torch.cuda.stream(stream1):
                    loss1_tensor = dummy_forward(device, return_tensor=True)
            
            start_minus = torch.cuda.Event(enable_timing=True)
            end_minus = torch.cuda.Event(enable_timing=True)
            start_minus.record(stream2)
            
            with torch.cuda.device(device):
                with torch.cuda.stream(stream2):
                    loss2_tensor = dummy_forward(device, return_tensor=True)
            
            end_plus.record(stream1)
            end_minus.record(stream2)
            torch.cuda.synchronize(device)
            timings['forward_plus'].append(start_plus.elapsed_time(end_plus))
            timings['forward_minus'].append(start_minus.elapsed_time(end_minus))
        
        # Convert to scalars after sync
        loss1 = loss1_tensor.item()
        loss2 = loss2_tensor.item()
        
        # Compute gradient
        projected_grad = (loss1 - loss2) / (2 * cfg.eps)
        
        # Update on main buffer
        start_update = torch.cuda.Event(enable_timing=True)
        end_update = torch.cuda.Event(enable_timing=True)
        start_update.record(stream1)
        
        with torch.cuda.device(device):
            if perturb_mode == "PYTORCH":
                pytorch_perturb_(param_flat, seed, -cfg.lr * projected_grad)
            elif perturb_mode == "CUDA":
                fused_perturb_cuda.fused_update(param_flat, seed, projected_grad, cfg.lr)
            else:
                seed_tensor[0] = seed
                fused_update_runtime_seed[grid](param_flat, seed_tensor, projected_grad, cfg.lr, n_elements)
        
        end_update.record(stream1)
        torch.cuda.synchronize(device)
        timings['update'].append(start_update.elapsed_time(end_update))
        
        total_end.record(stream1)
        torch.cuda.synchronize(device)
        timings['total'].append(total_start.elapsed_time(total_end))
    
    # Compute statistics
    results = {}
    for key in timings:
        values = timings[key][1:] if len(timings[key]) > 1 else timings[key]
        results[key] = {
            'mean': np.mean(values),
            'std': np.std(values) if len(values) > 1 else 0.0,
        }
    
    memory = torch.cuda.max_memory_allocated(device) / 1024**2
    
    perturb_total = results['perturb_plus']['mean'] + results['perturb_minus']['mean']
    forward_total = max(results['forward_plus']['mean'], results['forward_minus']['mean'])
    
    return MultiGPUBenchmarkResult(
        method=f"Single-GPU Parallel ({perturb_mode})",
        total_time_ms=results['total']['mean'],
        memory_main_mb=memory,
        memory_side_mb=0.0,
        model_sync_ms=results['buffer_copy']['mean'],
        perturb_main_ms=perturb_total,
        perturb_side_ms=0.0,
        forward_main_ms=forward_total,
        forward_side_ms=0.0,
        loss_sync_ms=0.0,
        update_ms=results['update']['mean'],
        notes=f"Single GPU: {device}, 2 buffers, 2 streams",
    )


def benchmark_single_gpu_pipelined(
    param_flat: torch.Tensor,
    anchor_flat: torch.Tensor,
    offsets: torch.Tensor,
    sizes: torch.Tensor,
    cfg: TrainingConfig,
    n_iter: int,
    device: torch.device,
    use_cuda_perturb: bool = True,
    use_pytorch_perturb: bool = False,
) -> MultiGPUBenchmarkResult:
    """
    Single-GPU pipelined approach using one buffer with overlapped preparation.
    
    Strategy:
    - Prepare θ₀ + εz → forward1 (stream1)
    - While forward1 runs, prepare θ₀ - εz in temp buffer (stream2)
    - After forward1 completes, swap temp → main buffer → forward2
    
    NOTE: When using real models, this benchmark uses the model's flat buffer directly
    to ensure forward passes use the perturbed parameters.
    """
    global _GLOBAL_FLAT_MANAGER_MAIN, _USE_REAL_MODEL
    
    n_elements = param_flat.numel()
    dtype = param_flat.dtype
    
    # For real models, use the model's flat buffer directly
    # For synthetic, create our own buffers
    if _USE_REAL_MODEL and _GLOBAL_FLAT_MANAGER_MAIN is not None:
        # Use the model's flat buffer - forward passes will see perturbed params
        flat_perturbed = param_flat  # This IS the model's param buffer
        temp_buffer = torch.empty_like(param_flat)  # For preparing second perturbation
    else:
        # Synthetic mode - create separate buffers
        flat_perturbed = torch.empty_like(param_flat)
        temp_buffer = torch.empty_like(param_flat)
    
    # Create CUDA streams
    stream1 = torch.cuda.Stream(device=device)
    stream2 = torch.cuda.Stream(device=device)
    
    # Setup perturb kernels
    if use_pytorch_perturb:
        perturb_mode = "PYTORCH"
    elif use_cuda_perturb and CUDA_PERTURB_TESTED_OK:
        perturb_mode = "CUDA"
    elif HAS_TRITON_RUNTIME_SEED:
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        seed_tensor = torch.tensor([0], dtype=torch.int64, device=device)
        perturb_mode = "Triton"
    else:
        raise RuntimeError("No perturb kernels available!")
    
    # Warmup
    for _ in range(5):
        seed = 42
        flat_perturbed.copy_(anchor_flat)
        temp_buffer.copy_(anchor_flat)
        
        if perturb_mode == "CUDA":
            fused_perturb_cuda.fused_perturb(flat_perturbed, seed, cfg.eps)
            fused_perturb_cuda.fused_perturb(temp_buffer, seed, -cfg.eps)
        elif perturb_mode == "PYTORCH":
            pytorch_perturb_(flat_perturbed, seed, cfg.eps)
            pytorch_perturb_(temp_buffer, seed, -cfg.eps)
        else:
            seed_tensor[0] = seed
            fused_perturb_runtime_seed[grid](flat_perturbed, seed_tensor, cfg.eps, n_elements)
            seed_tensor[0] = seed
            fused_perturb_runtime_seed[grid](temp_buffer, seed_tensor, -cfg.eps, n_elements)
    
    torch.cuda.synchronize(device)
    param_flat.copy_(anchor_flat)
    torch.cuda.synchronize(device)
    
    # Benchmark
    timings = {
        'perturb1': [],
        'forward1': [],
        'perturb2_prep': [],
        'buffer_swap': [],
        'forward2': [],
        'update': [],
        'total': [],
    }
    
    for i in range(n_iter):
        seed = np.random.randint(1000000000)
        
        total_start = torch.cuda.Event(enable_timing=True)
        total_end = torch.cuda.Event(enable_timing=True)
        total_start.record(stream1)
        
        # === Phase 1: Prepare and run θ₀ + εz forward ===
        start_perturb1 = torch.cuda.Event(enable_timing=True)
        end_perturb1 = torch.cuda.Event(enable_timing=True)
        start_perturb1.record(stream1)
        
        with torch.cuda.device(device):
            with torch.cuda.stream(stream1):
                flat_perturbed.copy_(anchor_flat, non_blocking=True)
                if perturb_mode == "CUDA":
                    fused_perturb_cuda.fused_perturb(flat_perturbed, seed, cfg.eps)
                elif perturb_mode == "PYTORCH":
                    pytorch_perturb_(flat_perturbed, seed, cfg.eps)
                else:
                    seed_tensor[0] = seed
                    fused_perturb_runtime_seed[grid](flat_perturbed, seed_tensor, cfg.eps, n_elements)
        
        end_perturb1.record(stream1)
        torch.cuda.synchronize(device)
        timings['perturb1'].append(start_perturb1.elapsed_time(end_perturb1))
        
        # Forward1 - use return_tensor=True to avoid .item() sync that destroys pipelining!
        start_fwd1 = torch.cuda.Event(enable_timing=True)
        end_fwd1 = torch.cuda.Event(enable_timing=True)
        start_fwd1.record(stream1)
        
        with torch.cuda.device(device):
            with torch.cuda.stream(stream1):
                loss1_tensor = dummy_forward(device, return_tensor=True)
        
        # === Phase 2: Prepare θ₀ - εz in parallel (stream2) ===
        # This ACTUALLY overlaps with forward1's computation now!
        start_perturb2 = torch.cuda.Event(enable_timing=True)
        end_perturb2 = torch.cuda.Event(enable_timing=True)
        start_perturb2.record(stream2)
        
        with torch.cuda.device(device):
            with torch.cuda.stream(stream2):
                temp_buffer.copy_(anchor_flat, non_blocking=True)
                if perturb_mode == "CUDA":
                    fused_perturb_cuda.fused_perturb(temp_buffer, seed, -cfg.eps)
                elif perturb_mode == "PYTORCH":
                    pytorch_perturb_(temp_buffer, seed, -cfg.eps)
                else:
                    seed_tensor[0] = seed
                    fused_perturb_runtime_seed[grid](temp_buffer, seed_tensor, -cfg.eps, n_elements)
        
        end_fwd1.record(stream1)
        end_perturb2.record(stream2)
        torch.cuda.synchronize(device)
        timings['forward1'].append(start_fwd1.elapsed_time(end_fwd1))
        timings['perturb2_prep'].append(start_perturb2.elapsed_time(end_perturb2))
        
        # === Phase 3: Swap buffers and run forward2 ===
        start_swap = torch.cuda.Event(enable_timing=True)
        end_swap = torch.cuda.Event(enable_timing=True)
        start_swap.record(stream1)
        
        # Swap temp_buffer → flat_perturbed
        flat_perturbed.copy_(temp_buffer)
        
        end_swap.record(stream1)
        torch.cuda.synchronize(device)
        timings['buffer_swap'].append(start_swap.elapsed_time(end_swap))
        
        # Forward2 - also use return_tensor=True for consistency
        # IMPORTANT: Must use stream1 context for timing to be accurate!
        start_fwd2 = torch.cuda.Event(enable_timing=True)
        end_fwd2 = torch.cuda.Event(enable_timing=True)
        start_fwd2.record(stream1)
        
        with torch.cuda.device(device):
            with torch.cuda.stream(stream1):
                loss2_tensor = dummy_forward(device, return_tensor=True)
        
        end_fwd2.record(stream1)
        torch.cuda.synchronize(device)
        timings['forward2'].append(start_fwd2.elapsed_time(end_fwd2))
        
        # Now convert to scalars after sync
        loss1 = loss1_tensor.item()
        loss2 = loss2_tensor.item()
        
        # Compute gradient
        projected_grad = (loss1 - loss2) / (2 * cfg.eps)
        
        # Update
        start_update = torch.cuda.Event(enable_timing=True)
        end_update = torch.cuda.Event(enable_timing=True)
        start_update.record(stream1)
        
        with torch.cuda.device(device):
            if perturb_mode == "CUDA":
                fused_perturb_cuda.fused_update(param_flat, seed, projected_grad, cfg.lr)
            elif perturb_mode == "PYTORCH":
                pytorch_perturb_(param_flat, seed, -cfg.lr * projected_grad)
            else:
                seed_tensor[0] = seed
                fused_update_runtime_seed[grid](param_flat, seed_tensor, projected_grad, cfg.lr, n_elements)
        
        end_update.record(stream1)
        torch.cuda.synchronize(device)
        timings['update'].append(start_update.elapsed_time(end_update))
        
        total_end.record(stream1)
        torch.cuda.synchronize(device)
        timings['total'].append(total_start.elapsed_time(total_end))
    
    # Compute statistics
    results = {}
    for key in timings:
        values = timings[key][1:] if len(timings[key]) > 1 else timings[key]
        results[key] = {
            'mean': np.mean(values),
            'std': np.std(values) if len(values) > 1 else 0.0,
        }
    
    memory = torch.cuda.max_memory_allocated(device) / 1024**2
    
    forward_total = results['forward1']['mean'] + results['forward2']['mean']
    # Overlapped time: min(forward1, perturb2_prep)
    overlapped_time = min(results['forward1']['mean'], results['perturb2_prep']['mean'])
    
    return MultiGPUBenchmarkResult(
        method=f"Single-GPU Pipelined ({perturb_mode})",
        total_time_ms=results['total']['mean'],
        memory_main_mb=memory,
        memory_side_mb=0.0,
        model_sync_ms=0.0,
        perturb_main_ms=results['perturb1']['mean'] + results['perturb2_prep']['mean'],
        perturb_side_ms=0.0,
        forward_main_ms=forward_total - overlapped_time,  # Effective forward time (with overlap)
        forward_side_ms=0.0,
        loss_sync_ms=0.0,
        update_ms=results['update']['mean'],
        notes=f"Single GPU: {device}, 1 buffer + 1 temp, overlapped prep (saved {overlapped_time:.2f}ms)",
    )


def benchmark_single_gpu_dual_model(
    param_flat: torch.Tensor,
    anchor_flat: torch.Tensor,
    offsets: torch.Tensor,
    sizes: torch.Tensor,
    cfg: TrainingConfig,
    n_iter: int,
    device: torch.device,
    use_cuda_perturb: bool = True,
    use_pytorch_perturb: bool = False,
) -> Optional[MultiGPUBenchmarkResult]:
    """
    Single-GPU with TWO MODEL COPIES for true parallel forward passes.
    
    Strategy:
    - Keep two copies of the model on the same GPU
    - Model1: θ₀ + εz → forward1 (stream1)
    - Model2: θ₀ - εz → forward2 (stream2)
    - Both forwards run TRULY in parallel on same GPU!
    
    Trade-off: 2x memory for ~1.3-1.5x speedup (limited by GPU compute bandwidth)
    
    NOTE: This only works when real models are loaded because we need two model instances.
    """
    global _GLOBAL_MODEL_MAIN, _GLOBAL_MODEL_SIDE, _GLOBAL_BATCH_MAIN, _USE_REAL_MODEL
    global _GLOBAL_FLAT_MANAGER_MAIN
    
    if not _USE_REAL_MODEL or _GLOBAL_MODEL_MAIN is None:
        print("  [Dual-Model] Skipping - requires real model mode with --use_real_model")
        return None
    
    # Check if we have enough memory for a second model copy
    torch.cuda.reset_peak_memory_stats(device)
    current_mem = torch.cuda.memory_allocated(device) / (1024**3)
    total_mem = torch.cuda.get_device_properties(device).total_memory / (1024**3)
    model_mem = param_flat.numel() * param_flat.element_size() / (1024**3)
    
    if current_mem + model_mem * 2 > total_mem * 0.9:  # Need 2x model + buffers
        print(f"  [Dual-Model] Skipping - insufficient GPU memory")
        print(f"    Current: {current_mem:.1f}GB, Model: {model_mem:.1f}GB, Total: {total_mem:.1f}GB")
        return None
    
    print("  [Dual-Model] Creating second model copy on same GPU...")
    
    n_elements = param_flat.numel()
    dtype = param_flat.dtype
    
    # Create second model copy
    try:
        model2 = copy.deepcopy(_GLOBAL_MODEL_MAIN)
        model2 = model2.to(device)
        
        # Create flat buffer for second model
        flat_manager2 = flatten_model_params(model2, device)
        set_model_params_from_flat(model2, flat_manager2.flat_buffer, flat_manager2.param_metadata)
        
        param_flat2 = flat_manager2.flat_buffer
        anchor_flat2 = flat_manager2.anchor_buffer
    except Exception as e:
        print(f"  [Dual-Model] Failed to create second model: {e}")
        return None
    
    print(f"  [Dual-Model] Second model created. Memory now: {torch.cuda.memory_allocated(device)/(1024**3):.2f}GB")
    
    # Create CUDA streams
    stream1 = torch.cuda.Stream(device=device)
    stream2 = torch.cuda.Stream(device=device)
    
    # Create batches for both models
    batch1 = _GLOBAL_BATCH_MAIN
    batch2 = {k: v.clone() for k, v in _GLOBAL_BATCH_MAIN.items()}  # Clone batch
    
    # Setup perturb kernels
    if use_pytorch_perturb:
        perturb_mode = "PYTORCH"
    elif use_cuda_perturb and CUDA_PERTURB_TESTED_OK:
        perturb_mode = "CUDA"
    elif HAS_TRITON_RUNTIME_SEED:
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        seed_tensor = torch.tensor([0], dtype=torch.int64, device=device)
        perturb_mode = "Triton"
    else:
        del model2, flat_manager2
        torch.cuda.empty_cache()
        raise RuntimeError("No perturb kernels available!")
    
    # Helper for forward on specific model
    def forward_model1():
        with torch.no_grad():
            return _GLOBAL_MODEL_MAIN(**batch1, return_dict=True).loss
    
    def forward_model2():
        with torch.no_grad():
            return model2(**batch2, return_dict=True).loss
    
    # Warmup
    print(f"  [Dual-Model] Warming up...")
    for _ in range(3):
        seed = 42
        param_flat.copy_(anchor_flat)
        param_flat2.copy_(anchor_flat2)
        if perturb_mode == "CUDA":
            fused_perturb_cuda.fused_perturb(param_flat, seed, cfg.eps)
            fused_perturb_cuda.fused_perturb(param_flat2, seed, -cfg.eps)
        elif perturb_mode == "PYTORCH":
            pytorch_perturb_(param_flat, seed, cfg.eps)
            pytorch_perturb_(param_flat2, seed, -cfg.eps)
        else:
            seed_tensor[0] = seed
            fused_perturb_runtime_seed[grid](param_flat, seed_tensor, cfg.eps, n_elements)
            fused_perturb_runtime_seed[grid](param_flat2, seed_tensor, -cfg.eps, n_elements)
        forward_model1()
        forward_model2()
    
    torch.cuda.synchronize(device)
    param_flat.copy_(anchor_flat)
    param_flat2.copy_(anchor_flat)
    torch.cuda.synchronize(device)
    
    # Benchmark
    print(f"  [Dual-Model] Benchmarking ({n_iter} iterations)...")
    timings = {
        'perturb1': [],
        'perturb2': [],
        'forward1': [],
        'forward2': [],
        'update': [],
        'total': [],
    }
    
    for i in range(n_iter):
        seed = np.random.randint(1000000000)
        
        total_start = torch.cuda.Event(enable_timing=True)
        total_end = torch.cuda.Event(enable_timing=True)
        total_start.record(stream1)
        
        # Reset both models to anchor params
        param_flat.copy_(anchor_flat)
        param_flat2.copy_(anchor_flat2)
        
        # === PARALLEL PERTURBATIONS ===
        start1 = torch.cuda.Event(enable_timing=True)
        end1 = torch.cuda.Event(enable_timing=True)
        start2 = torch.cuda.Event(enable_timing=True)
        end2 = torch.cuda.Event(enable_timing=True)
        
        start1.record(stream1)
        with torch.cuda.stream(stream1):
            if perturb_mode == "CUDA":
                fused_perturb_cuda.fused_perturb(param_flat, seed, cfg.eps)
            elif perturb_mode == "PYTORCH":
                pytorch_perturb_(param_flat, seed, cfg.eps)
            else:
                seed_tensor[0] = seed
                fused_perturb_runtime_seed[grid](param_flat, seed_tensor, cfg.eps, n_elements)
        end1.record(stream1)
        
        start2.record(stream2)
        with torch.cuda.stream(stream2):
            if perturb_mode == "CUDA":
                fused_perturb_cuda.fused_perturb(param_flat2, seed, -cfg.eps)
            elif perturb_mode == "PYTORCH":
                pytorch_perturb_(param_flat2, seed, -cfg.eps)
            else:
                seed_tensor[0] = seed
                fused_perturb_runtime_seed[grid](param_flat2, seed_tensor, -cfg.eps, n_elements)
        end2.record(stream2)
        
        torch.cuda.synchronize(device)
        timings['perturb1'].append(start1.elapsed_time(end1))
        timings['perturb2'].append(start2.elapsed_time(end2))
        
        # === TRUE PARALLEL FORWARDS ===
        start1 = torch.cuda.Event(enable_timing=True)
        end1 = torch.cuda.Event(enable_timing=True)
        start2 = torch.cuda.Event(enable_timing=True)
        end2 = torch.cuda.Event(enable_timing=True)
        
        start1.record(stream1)
        with torch.cuda.stream(stream1):
            loss1 = forward_model1()
        
        start2.record(stream2)
        with torch.cuda.stream(stream2):
            loss2 = forward_model2()
        
        end1.record(stream1)
        end2.record(stream2)
        torch.cuda.synchronize(device)
        
        timings['forward1'].append(start1.elapsed_time(end1))
        timings['forward2'].append(start2.elapsed_time(end2))
        
        # Compute gradient
        projected_grad = ((loss1.item() - loss2.item()) / (2 * cfg.eps))
        
        # Update on main model's buffer
        start_update = torch.cuda.Event(enable_timing=True)
        end_update = torch.cuda.Event(enable_timing=True)
        start_update.record(stream1)
        
        if perturb_mode == "CUDA":
            fused_perturb_cuda.fused_update(param_flat, seed, float(projected_grad), cfg.lr)
        elif perturb_mode == "PYTORCH":
            pytorch_perturb_(param_flat, seed, -cfg.lr * float(projected_grad))
        else:
            seed_tensor[0] = seed
            fused_update_runtime_seed[grid](param_flat, seed_tensor, float(projected_grad), cfg.lr, n_elements)
        
        end_update.record(stream1)
        torch.cuda.synchronize(device)
        timings['update'].append(start_update.elapsed_time(end_update))
        
        total_end.record(stream1)
        torch.cuda.synchronize(device)
        timings['total'].append(total_start.elapsed_time(total_end))
    
    # Cleanup second model
    del model2, flat_manager2, param_flat2, anchor_flat2
    torch.cuda.empty_cache()
    
    # Compute statistics
    results = {}
    for key in timings:
        values = timings[key][1:] if len(timings[key]) > 1 else timings[key]
        results[key] = {
            'mean': np.mean(values),
            'std': np.std(values) if len(values) > 1 else 0.0,
        }
    
    memory = torch.cuda.max_memory_allocated(device) / 1024**2
    
    forward_max = max(results['forward1']['mean'], results['forward2']['mean'])
    
    return MultiGPUBenchmarkResult(
        method=f"Single-GPU Dual-Model ({perturb_mode})",
        total_time_ms=results['total']['mean'],
        memory_main_mb=memory,
        memory_side_mb=0.0,
        model_sync_ms=0.0,  # No sync needed - both models on same GPU
        perturb_main_ms=max(results['perturb1']['mean'], results['perturb2']['mean']),
        perturb_side_ms=0.0,
        forward_main_ms=forward_max,  # Parallel forwards, report max
        forward_side_ms=0.0,
        loss_sync_ms=0.0,
        update_ms=results['update']['mean'],
        notes=f"Single GPU: {device}, 2 model copies, TRUE parallel forward",
    )


def benchmark_single_gpu_compiled(
    param_flat: torch.Tensor,
    anchor_flat: torch.Tensor,
    offsets: torch.Tensor,
    sizes: torch.Tensor,
    cfg: TrainingConfig,
    n_iter: int,
    device: torch.device,
    use_cuda_perturb: bool = True,
    use_pytorch_perturb: bool = False,
) -> Optional[MultiGPUBenchmarkResult]:
    """
    Single-GPU with torch.compile() for optimized forward passes.
    
    Strategy:
    - Use torch.compile() to JIT-compile the model forward
    - Reduces kernel launch overhead and enables kernel fusion
    - Best for models that run many iterations
    
    NOTE: Compilation has startup overhead, so speedup is seen over many iterations.
    """
    global _GLOBAL_MODEL_MAIN, _GLOBAL_BATCH_MAIN, _USE_REAL_MODEL
    global _GLOBAL_FLAT_MANAGER_MAIN
    
    if not _USE_REAL_MODEL or _GLOBAL_MODEL_MAIN is None:
        print("  [Compiled] Skipping - requires real model mode with --use_real_model")
        return None
    
    # Check if torch.compile is available (PyTorch 2.0+)
    if not hasattr(torch, 'compile'):
        print("  [Compiled] Skipping - torch.compile not available (requires PyTorch 2.0+)")
        return None
    
    n_elements = param_flat.numel()
    dtype = param_flat.dtype
    
    print("  [Compiled] Compiling model forward pass...")
    
    # Create a compiled version of forward
    @torch.no_grad()
    def forward_fn(model, batch):
        return model(**batch, return_dict=True).loss
    
    try:
        # Compile with reduce-overhead mode for inference
        compiled_forward = torch.compile(forward_fn, mode="reduce-overhead", fullgraph=False)
        
        # Warmup compilation
        print("  [Compiled] Warming up compiled model...")
        for _ in range(3):
            _ = compiled_forward(_GLOBAL_MODEL_MAIN, _GLOBAL_BATCH_MAIN)
        torch.cuda.synchronize(device)
    except Exception as e:
        print(f"  [Compiled] Compilation failed: {e}")
        return None
    
    # Setup perturb kernels
    if use_pytorch_perturb:
        perturb_mode = "PYTORCH"
    elif use_cuda_perturb and CUDA_PERTURB_TESTED_OK:
        perturb_mode = "CUDA"
    elif HAS_TRITON_RUNTIME_SEED:
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        seed_tensor = torch.tensor([0], dtype=torch.int64, device=device)
        perturb_mode = "Triton"
    else:
        raise RuntimeError("No perturb kernels available!")
    
    print(f"  [Compiled] Using {perturb_mode} perturb kernels")
    
    # Reset
    param_flat.copy_(anchor_flat)
    torch.cuda.synchronize(device)
    
    stream = torch.cuda.Stream(device=device)
    
    # Benchmark
    print(f"  [Compiled] Benchmarking ({n_iter} iterations)...")
    timings = {
        'perturb1': [],
        'forward1': [],
        'perturb2': [],
        'forward2': [],
        'perturb3': [],
        'update': [],
        'total': [],
    }
    
    for i in range(n_iter):
        seed = np.random.randint(1000000000)
        
        total_start = torch.cuda.Event(enable_timing=True)
        total_end = torch.cuda.Event(enable_timing=True)
        total_start.record(stream)
        
        # Perturb +eps
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        with torch.cuda.stream(stream):
            if perturb_mode == "CUDA":
                fused_perturb_cuda.fused_perturb(param_flat, seed, cfg.eps)
            elif perturb_mode == "PYTORCH":
                pytorch_perturb_(param_flat, seed, cfg.eps)
            else:
                seed_tensor[0] = seed
                fused_perturb_runtime_seed[grid](param_flat, seed_tensor, cfg.eps, n_elements)
        end.record(stream)
        torch.cuda.synchronize(device)
        timings['perturb1'].append(start.elapsed_time(end))
        
        # Forward 1 (compiled)
        start.record(stream)
        with torch.cuda.stream(stream):
            loss1 = compiled_forward(_GLOBAL_MODEL_MAIN, _GLOBAL_BATCH_MAIN)
        end.record(stream)
        torch.cuda.synchronize(device)
        timings['forward1'].append(start.elapsed_time(end))
        
        # Perturb -2eps
        start.record(stream)
        with torch.cuda.stream(stream):
            if perturb_mode == "CUDA":
                fused_perturb_cuda.fused_perturb(param_flat, seed, -2*cfg.eps)
            elif perturb_mode == "PYTORCH":
                pytorch_perturb_(param_flat, seed, -2*cfg.eps)
            else:
                fused_perturb_runtime_seed[grid](param_flat, seed_tensor, -2*cfg.eps, n_elements)
        end.record(stream)
        torch.cuda.synchronize(device)
        timings['perturb2'].append(start.elapsed_time(end))
        
        # Forward 2 (compiled)
        start.record(stream)
        with torch.cuda.stream(stream):
            loss2 = compiled_forward(_GLOBAL_MODEL_MAIN, _GLOBAL_BATCH_MAIN)
        end.record(stream)
        torch.cuda.synchronize(device)
        timings['forward2'].append(start.elapsed_time(end))
        
        # Compute gradient
        projected_grad = ((loss1.item() - loss2.item()) / (2 * cfg.eps))
        
        # Perturb +eps (restore)
        start.record(stream)
        with torch.cuda.stream(stream):
            if perturb_mode == "CUDA":
                fused_perturb_cuda.fused_perturb(param_flat, seed, cfg.eps)
            elif perturb_mode == "PYTORCH":
                pytorch_perturb_(param_flat, seed, cfg.eps)
            else:
                fused_perturb_runtime_seed[grid](param_flat, seed_tensor, cfg.eps, n_elements)
        end.record(stream)
        torch.cuda.synchronize(device)
        timings['perturb3'].append(start.elapsed_time(end))
        
        # Update
        start.record(stream)
        with torch.cuda.stream(stream):
            if perturb_mode == "CUDA":
                fused_perturb_cuda.fused_update(param_flat, seed, float(projected_grad), cfg.lr)
            elif perturb_mode == "PYTORCH":
                pytorch_perturb_(param_flat, seed, -cfg.lr * float(projected_grad))
            else:
                seed_tensor[0] = seed
                fused_update_runtime_seed[grid](param_flat, seed_tensor, float(projected_grad), cfg.lr, n_elements)
        end.record(stream)
        torch.cuda.synchronize(device)
        timings['update'].append(start.elapsed_time(end))
        
        total_end.record(stream)
        torch.cuda.synchronize(device)
        timings['total'].append(total_start.elapsed_time(total_end))
    
    # Compute statistics
    results = {}
    for key in timings:
        values = timings[key][1:] if len(timings[key]) > 1 else timings[key]
        results[key] = {
            'mean': np.mean(values),
            'std': np.std(values) if len(values) > 1 else 0.0,
        }
    
    memory = torch.cuda.max_memory_allocated(device) / 1024**2
    
    forward_total = results['forward1']['mean'] + results['forward2']['mean']
    perturb_total = results['perturb1']['mean'] + results['perturb2']['mean'] + results['perturb3']['mean']
    
    return MultiGPUBenchmarkResult(
        method=f"Single-GPU Compiled ({perturb_mode})",
        total_time_ms=results['total']['mean'],
        memory_main_mb=memory,
        memory_side_mb=0.0,
        model_sync_ms=0.0,
        perturb_main_ms=perturb_total,
        perturb_side_ms=0.0,
        forward_main_ms=forward_total,
        forward_side_ms=0.0,
        loss_sync_ms=0.0,
        update_ms=results['update']['mean'],
        notes=f"Single GPU: {device}, torch.compile() optimized",
    )


def benchmark_sequential_baseline(
    param_flat: torch.Tensor,
    anchor_flat: torch.Tensor,
    offsets: torch.Tensor,
    sizes: torch.Tensor,
    cfg: TrainingConfig,
    n_iter: int,
    device: torch.device,
    use_cuda_perturb: bool = True,
    use_pytorch_perturb: bool = False,
) -> MultiGPUBenchmarkResult:
    """
    Sequential baseline for comparison (single GPU, sequential forwards).
    """
    n_elements = param_flat.numel()
    dtype = param_flat.dtype
    
    # Setup perturb kernels
    if use_pytorch_perturb:
        perturb_mode = "PYTORCH"
    elif use_cuda_perturb and CUDA_PERTURB_TESTED_OK:
        perturb_mode = "CUDA"
    elif HAS_TRITON_RUNTIME_SEED:
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        seed_tensor = torch.tensor([0], dtype=torch.int64, device=device)
        perturb_mode = "Triton"
    else:
        raise RuntimeError("No perturb kernels available!")
    
    # Warmup
    for _ in range(5):
        seed = 42
        if perturb_mode == "CUDA":
            fused_perturb_cuda.fused_perturb(param_flat, seed, cfg.eps)
        elif perturb_mode == "PYTORCH":
            pytorch_perturb_(param_flat, seed, cfg.eps)
        else:
            seed_tensor[0] = seed
            fused_perturb_runtime_seed[grid](param_flat, seed_tensor, cfg.eps, n_elements)
    
    torch.cuda.synchronize(device)
    param_flat.copy_(anchor_flat)
    torch.cuda.synchronize(device)
    
    # Create a stream for timing
    stream = torch.cuda.Stream(device=device)
    
    # Benchmark
    timings = {
        'perturb1': [],
        'forward1': [],
        'perturb2': [],
        'forward2': [],
        'perturb3': [],
        'update': [],
        'total': [],
    }
    
    for i in range(n_iter):
        seed = np.random.randint(1000000000)
        
        total_start = torch.cuda.Event(enable_timing=True)
        total_end = torch.cuda.Event(enable_timing=True)
        total_start.record(stream)
        
        # Perturb +eps
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        with torch.cuda.device(device):
            with torch.cuda.stream(stream):
                if perturb_mode == "CUDA":
                    fused_perturb_cuda.fused_perturb(param_flat, seed, cfg.eps)
                elif perturb_mode == "PYTORCH":
                    pytorch_perturb_(param_flat, seed, cfg.eps)
                else:
                    seed_tensor[0] = seed
                    fused_perturb_runtime_seed[grid](param_flat, seed_tensor, cfg.eps, n_elements)
        end.record(stream)
        torch.cuda.synchronize(device)
        timings['perturb1'].append(start.elapsed_time(end))
        
        # Forward 1
        start.record(stream)
        with torch.cuda.device(device):
            with torch.cuda.stream(stream):
                loss1 = dummy_forward(device)
        end.record(stream)
        torch.cuda.synchronize(device)
        timings['forward1'].append(start.elapsed_time(end))
        
        # Perturb -2eps
        start.record(stream)
        with torch.cuda.device(device):
            with torch.cuda.stream(stream):
                if perturb_mode == "CUDA":
                    fused_perturb_cuda.fused_perturb(param_flat, seed, -2*cfg.eps)
                elif perturb_mode == "PYTORCH":
                    pytorch_perturb_(param_flat, seed, -2 * cfg.eps)
                else:
                    fused_perturb_runtime_seed[grid](param_flat, seed_tensor, -2*cfg.eps, n_elements)
        end.record(stream)
        torch.cuda.synchronize(device)
        timings['perturb2'].append(start.elapsed_time(end))
        
        # Forward 2
        start.record(stream)
        with torch.cuda.device(device):
            with torch.cuda.stream(stream):
                loss2 = dummy_forward(device)
        end.record(stream)
        torch.cuda.synchronize(device)
        timings['forward2'].append(start.elapsed_time(end))
        
        projected_grad = (loss1 - loss2) / (2 * cfg.eps)
        
        # Perturb reset
        start.record(stream)
        with torch.cuda.device(device):
            with torch.cuda.stream(stream):
                if perturb_mode == "CUDA":
                    fused_perturb_cuda.fused_perturb(param_flat, seed, cfg.eps)
                elif perturb_mode == "PYTORCH":
                    pytorch_perturb_(param_flat, seed, cfg.eps)
                else:
                    fused_perturb_runtime_seed[grid](param_flat, seed_tensor, cfg.eps, n_elements)
        end.record(stream)
        torch.cuda.synchronize(device)
        timings['perturb3'].append(start.elapsed_time(end))
        
        # Update
        start.record(stream)
        with torch.cuda.device(device):
            with torch.cuda.stream(stream):
                if perturb_mode == "CUDA":
                    fused_perturb_cuda.fused_update(param_flat, seed, projected_grad, cfg.lr)
                elif perturb_mode == "PYTORCH":
                    pytorch_perturb_(param_flat, seed, -cfg.lr * projected_grad)
                else:
                    fused_update_runtime_seed[grid](param_flat, seed_tensor, projected_grad, cfg.lr, n_elements)
        end.record(stream)
        torch.cuda.synchronize(device)
        timings['update'].append(start.elapsed_time(end))
        
        total_end.record(stream)
        torch.cuda.synchronize(device)
        timings['total'].append(total_start.elapsed_time(total_end))
    
    # Compute statistics
    results = {}
    for key in timings:
        values = timings[key][1:] if len(timings[key]) > 1 else timings[key]
        results[key] = {
            'mean': np.mean(values),
            'std': np.std(values) if len(values) > 1 else 0.0,
        }
    
    memory = torch.cuda.max_memory_allocated(device) / 1024**2
    
    perturb_total = results['perturb1']['mean'] + results['perturb2']['mean'] + results['perturb3']['mean']
    forward_total = results['forward1']['mean'] + results['forward2']['mean']
    
    return MultiGPUBenchmarkResult(
        method=f"Sequential Baseline ({perturb_mode})",
        total_time_ms=results['total']['mean'],
        memory_main_mb=memory,
        memory_side_mb=0.0,
        model_sync_ms=0.0,
        perturb_main_ms=perturb_total,
        perturb_side_ms=0.0,
        forward_main_ms=forward_total,
        forward_side_ms=0.0,
        loss_sync_ms=0.0,
        update_ms=results['update']['mean'],
        notes=f"Single GPU: {device}",
    )


def main():
    parser = argparse.ArgumentParser(
        description='Multi-GPU Parallel MeZO Benchmark',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic run with synthetic data (uses GPU 0 and 1)
  python benchmark_multi_gpu_parallel.py --model opt-350m
  
  # Use REAL OPT model for actual forward passes
  python benchmark_multi_gpu_parallel.py --model opt-350m --use_real_model
  
  # Use specific GPUs (for debugging)
  python benchmark_multi_gpu_parallel.py --model opt-350m --main_gpu 1 --side_gpu 2
  
  # Compare with sequential baseline
  python benchmark_multi_gpu_parallel.py --model opt-350m --compare_baseline --use_real_model
  
  # Debug mode with detailed timing
  python benchmark_multi_gpu_parallel.py --model opt-350m --debug --use_real_model
        """
    )
    parser.add_argument('--model', type=str, default='opt-350m', choices=list(MODEL_CONFIGS.keys()),
                        help='Model name: opt-350m, opt-1.3b, opt-2.7b, opt-6.7b, opt-13b')
    parser.add_argument('--main_gpu', type=int, default=0, help='Main GPU device ID')
    parser.add_argument('--side_gpu', type=int, default=1, help='Side GPU device ID')
    parser.add_argument('--n_iter', type=int, default=20, help='Number of iterations')
    parser.add_argument('--eps', type=float, default=1e-3, help='Perturbation scale')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--compare_baseline', action='store_true', help='Compare with sequential baseline')
    parser.add_argument('--compare_all', action='store_true', help='Compare all methods (multi-GPU, single-GPU parallel, pipelined, sequential)')
    parser.add_argument('--compare_optimized', action='store_true', help='Include optimized multi-GPU benchmark with overlapped sync')
    parser.add_argument('--compare_single_gpu_opt', action='store_true', 
                        help='Include single-GPU optimizations: dual-model (2 copies), torch.compile')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--use_triton', action='store_true', help='Use Triton kernels instead of CUDA')
    parser.add_argument('--use_pytorch_perturb', action='store_true', help='Force PyTorch perturb/update (safe mode)')
    parser.add_argument('--use_real_model', action='store_true', help='Use real OPT model for forward passes (requires transformers)')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size for real model forward (auto if not specified)')
    parser.add_argument('--seq_len', type=int, default=512, help='Sequence length for real model forward (default: 512)')
    parser.add_argument('--dtype', type=str, default='auto', choices=['auto', 'fp32', 'fp16', 'bf16'],
                        help='Data type for model: auto (based on model size), fp32, fp16, bf16')
    args = parser.parse_args()
    
    # Convert dtype string to torch dtype
    dtype_map = {'fp32': torch.float32, 'fp16': torch.float16, 'bf16': torch.bfloat16, 'auto': None}
    args.dtype_torch = dtype_map[args.dtype]
    
    # Check GPU availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    
    if torch.cuda.device_count() < 2:
        raise RuntimeError(f"Need at least 2 GPUs, found {torch.cuda.device_count()}")
    
    if args.main_gpu >= torch.cuda.device_count() or args.side_gpu >= torch.cuda.device_count():
        raise RuntimeError(f"Invalid GPU IDs. Available: 0-{torch.cuda.device_count()-1}")
    
    if args.main_gpu == args.side_gpu:
        raise RuntimeError("Main GPU and side GPU must be different")
    
    device_main = torch.device(f'cuda:{args.main_gpu}')
    device_side = torch.device(f'cuda:{args.side_gpu}')
    
    # Get GPU memory info for auto-config
    torch.cuda.empty_cache()
    gc.collect()
    gpu_mem_main = torch.cuda.get_device_properties(device_main).total_memory / (1024**3)
    
    # Auto-configure batch size if not specified
    if args.batch_size is None:
        args.batch_size = get_recommended_batch_size(args.model, args.seq_len, 
                                                      args.dtype_torch or torch.float32, gpu_mem_main)
        print(f"Auto-selected batch size: {args.batch_size} (based on model size and GPU memory)")
    
    print("=" * 90)
    print("MULTI-GPU PARALLEL MEZO BENCHMARK")
    print("=" * 90)
    print(f"Model: {args.model}")
    print(f"Main GPU: {args.main_gpu} ({torch.cuda.get_device_name(args.main_gpu)}, {gpu_mem_main:.1f}GB)")
    print(f"Side GPU: {args.side_gpu} ({torch.cuda.get_device_name(args.side_gpu)})")
    print(f"Iterations: {args.n_iter}")
    print(f"Use Real Model: {args.use_real_model}")
    if args.use_real_model:
        print(f"Batch Size: {args.batch_size}")
        print(f"Sequence Length: {args.seq_len}")
        print(f"Data Type: {args.dtype} {'(auto-selected)' if args.dtype == 'auto' else ''}")
    print()
    
    config = MODEL_CONFIGS[args.model]
    cfg = TrainingConfig(eps=args.eps, lr=args.lr)
    
    # Create flat buffers - either from real model or synthetic
    if args.use_real_model:
        if not HAS_TRANSFORMERS:
            raise RuntimeError("--use_real_model requires transformers library. Install with: pip install transformers")
        
        print("Setting up REAL OPT models with flat buffers...")
        flat_manager_main, flat_manager_side, model_main, model_side = setup_real_models(
            args.model, device_main, device_side, 
            batch_size=args.batch_size, 
            seq_len=args.seq_len,
            dtype=args.dtype_torch,
            auto_dtype=(args.dtype == 'auto'),
        )
        
        # Extract flat buffers from managers
        param_flat_main = flat_manager_main.flat_buffer
        anchor_flat_main = flat_manager_main.anchor_buffer
        param_flat_side = flat_manager_side.flat_buffer
        anchor_flat_side = flat_manager_side.anchor_buffer
        
        # Create dummy offsets and sizes tensors (for compatibility with existing benchmark functions)
        offsets = torch.tensor([m.offset for m in flat_manager_main.param_metadata], device=device_main, dtype=torch.long)
        sizes = torch.tensor([m.numel for m in flat_manager_main.param_metadata], device=device_main, dtype=torch.long)
        
        print(f"\n✓ Real model flat buffers ready")
        print(f"  Main buffer: {param_flat_main.numel():,} elements ({param_flat_main.numel() * 4 / 1024**2:.2f} MB)")
        print(f"  Side buffer: {param_flat_side.numel():,} elements ({param_flat_side.numel() * 4 / 1024**2:.2f} MB)")
    else:
        print("Creating SYNTHETIC flat buffers on both GPUs...")
        (param_flat_main, anchor_flat_main, param_flat_side, anchor_flat_side,
         offsets, sizes, param_views_main, anchor_views_main,
         param_views_side, anchor_views_side) = create_flat_buffers_for_multi_gpu(
            config, device_main, device_side
        )
    print()
    
    # Reset memory stats
    torch.cuda.reset_peak_memory_stats(device_main)
    torch.cuda.reset_peak_memory_stats(device_side)
    
    # Auto-enable PyTorch mode if CUDA perturb test failed
    if not args.use_pytorch_perturb and not args.use_triton:
        if HAS_CUDA_PERTURB_IMPORTED and not CUDA_PERTURB_TESTED_OK:
            print("⚠ CUDA perturb kernel test failed - automatically using PyTorch fallback")
            args.use_pytorch_perturb = True
        elif not HAS_CUDA_PERTURB_IMPORTED and not HAS_TRITON_RUNTIME_SEED:
            print("⚠ No optimized kernels available - using PyTorch fallback")
            args.use_pytorch_perturb = True
    
    results = []
    
    # Run multi-GPU benchmark
    print("Running multi-GPU parallel benchmark...")
    result_multi = benchmark_multi_gpu_parallel(
        param_flat_main, anchor_flat_main,
        param_flat_side, anchor_flat_side,
        offsets, sizes,
        cfg, args.n_iter,
        device_main, device_side,
        use_cuda_perturb=(not args.use_triton and not args.use_pytorch_perturb),
        use_pytorch_perturb=args.use_pytorch_perturb,
        include_dizo=False,
        debug=args.debug,
    )
    results.append(result_multi)
    
    # Run OPTIMIZED multi-GPU benchmark with overlapped sync
    if args.compare_optimized or args.compare_all:
        print("\nRunning OPTIMIZED multi-GPU benchmark (overlapped sync)...")
        # Reset buffers to anchor state
        param_flat_main.copy_(anchor_flat_main)
        param_flat_side.copy_(anchor_flat_side)
        torch.cuda.synchronize(device_main)
        torch.cuda.synchronize(device_side)
        
        result_optimized = benchmark_multi_gpu_optimized(
            param_flat_main, anchor_flat_main,
            param_flat_side, anchor_flat_side,
            offsets, sizes,
            cfg, args.n_iter,
            device_main, device_side,
            use_cuda_perturb=(not args.use_triton and not args.use_pytorch_perturb),
            use_pytorch_perturb=args.use_pytorch_perturb,
            debug=args.debug,
        )
        results.append(result_optimized)
    
    # Run single-GPU parallel benchmark
    if args.compare_all or args.compare_baseline:
        print("\nRunning single-GPU parallel benchmark (2 buffers, 2 streams)...")
        torch.cuda.reset_peak_memory_stats(device_main)
        
        result_single_parallel = benchmark_single_gpu_parallel(
            param_flat_main, anchor_flat_main,
            offsets, sizes,
            cfg, args.n_iter,
            device_main,
            use_cuda_perturb=(not args.use_triton and not args.use_pytorch_perturb),
            use_pytorch_perturb=args.use_pytorch_perturb,
        )
        results.append(result_single_parallel)
    
    # Run single-GPU pipelined benchmark
    if args.compare_all:
        print("\nRunning single-GPU pipelined benchmark (1 buffer + temp, overlapped prep)...")
        torch.cuda.reset_peak_memory_stats(device_main)
        
        result_pipelined = benchmark_single_gpu_pipelined(
            param_flat_main, anchor_flat_main,
            offsets, sizes,
            cfg, args.n_iter,
            device_main,
            use_cuda_perturb=(not args.use_triton and not args.use_pytorch_perturb),
            use_pytorch_perturb=args.use_pytorch_perturb,
        )
        results.append(result_pipelined)
    
    # Run single-GPU dual-model benchmark (TRUE parallel with 2 model copies)
    if args.compare_single_gpu_opt and args.use_real_model:
        print("\nRunning single-GPU DUAL-MODEL benchmark (2 model copies, true parallel)...")
        # Reset main buffer
        param_flat_main.copy_(anchor_flat_main)
        torch.cuda.synchronize(device_main)
        torch.cuda.reset_peak_memory_stats(device_main)
        
        result_dual = benchmark_single_gpu_dual_model(
            param_flat_main, anchor_flat_main,
            offsets, sizes,
            cfg, args.n_iter,
            device_main,
            use_cuda_perturb=(not args.use_triton and not args.use_pytorch_perturb),
            use_pytorch_perturb=args.use_pytorch_perturb,
        )
        if result_dual is not None:
            results.append(result_dual)
    
    # Run single-GPU compiled benchmark (torch.compile optimized)
    if args.compare_single_gpu_opt and args.use_real_model:
        print("\nRunning single-GPU COMPILED benchmark (torch.compile optimized)...")
        # Reset main buffer
        param_flat_main.copy_(anchor_flat_main)
        torch.cuda.synchronize(device_main)
        torch.cuda.reset_peak_memory_stats(device_main)
        
        result_compiled = benchmark_single_gpu_compiled(
            param_flat_main, anchor_flat_main,
            offsets, sizes,
            cfg, args.n_iter,
            device_main,
            use_cuda_perturb=(not args.use_triton and not args.use_pytorch_perturb),
            use_pytorch_perturb=args.use_pytorch_perturb,
        )
        if result_compiled is not None:
            results.append(result_compiled)
    
    # Run sequential baseline for comparison
    if args.compare_baseline or args.compare_all:
        print("\nRunning sequential baseline for comparison...")
        torch.cuda.reset_peak_memory_stats(device_main)
        
        result_seq = benchmark_sequential_baseline(
            param_flat_main, anchor_flat_main,
            offsets, sizes,
            cfg, args.n_iter,
            device_main,
            use_cuda_perturb=(not args.use_triton and not args.use_pytorch_perturb),
            use_pytorch_perturb=args.use_pytorch_perturb,
        )
        results.append(result_seq)
        
        # Calculate speedups vs sequential
        seq_time = result_seq.total_time_ms
        for r in results:
            if r.total_time_ms > 0 and r != result_seq:
                r.speedup_vs_sequential = seq_time / r.total_time_ms
    
    # Print results
    print("\n" + "=" * 90)
    print("RESULTS")
    print("=" * 90)
    
    for r in results:
        print(f"\n{r.method}:")
        print(f"  Total time:        {r.total_time_ms:.2f} ms")
        print(f"  Memory (main):     {r.memory_main_mb:.2f} MB")
        if r.memory_side_mb > 0:
            print(f"  Memory (side):     {r.memory_side_mb:.2f} MB")
        print(f"  Model sync:        {r.model_sync_ms:.2f} ms")
        print(f"  Perturb (main):    {r.perturb_main_ms:.2f} ms")
        if r.perturb_side_ms > 0:
            print(f"  Perturb (side):    {r.perturb_side_ms:.2f} ms")
        print(f"  Forward (main):    {r.forward_main_ms:.2f} ms")
        if r.forward_side_ms > 0:
            print(f"  Forward (side):    {r.forward_side_ms:.2f} ms")
        print(f"  Loss sync:         {r.loss_sync_ms:.2f} ms")
        print(f"  Update:            {r.update_ms:.2f} ms")
        if r.speedup_vs_sequential > 0:
            print(f"  Speedup vs seq:    {r.speedup_vs_sequential:.2f}x")
        if r.notes:
            print(f"  Notes:             {r.notes}")
    
    # Find sequential baseline for comparison
    seq_result = None
    for r in results:
        if "Sequential" in r.method:
            seq_result = r
            break
    
    if seq_result:
        print("\n" + "=" * 90)
        print("SPEEDUP ANALYSIS (vs Sequential Baseline)")
        print("=" * 90)
        seq_time = seq_result.total_time_ms
        print(f"Sequential baseline: {seq_time:.2f} ms")
        print()
        print(f"{'Method':<40} {'Time (ms)':<15} {'Speedup':<15} {'Efficiency':<15}")
        print("-" * 85)
        
        for r in results:
            if r != seq_result and r.speedup_vs_sequential > 0:
                efficiency = (r.speedup_vs_sequential / 2.0) * 100 if "Multi-GPU" in r.method or "Parallel" in r.method else 0
                eff_str = f"{efficiency:.1f}%" if efficiency > 0 else "N/A"
                print(f"{r.method:<40} {r.total_time_ms:<15.2f} {r.speedup_vs_sequential:<15.2f}x {eff_str:<15}")
        
        # Detailed breakdown for multi-GPU
        multi_result = None
        for r in results:
            if "Multi-GPU" in r.method:
                multi_result = r
                break
        
        if multi_result:
            print("\n" + "-" * 90)
            print("MULTI-GPU BREAKDOWN:")
            print("-" * 90)
            seq_forward = seq_result.forward_main_ms
            multi_forward = max(multi_result.forward_main_ms, multi_result.forward_side_ms)
            forward_speedup = seq_forward / multi_forward if multi_forward > 0 else 0
            
            print(f"Sequential forward time:  {seq_forward:.2f} ms")
            print(f"Multi-GPU forward time:   {multi_forward:.2f} ms (max of both GPUs)")
            print(f"Forward pass speedup:     {forward_speedup:.2f}x")
            print(f"Model sync overhead:      {multi_result.model_sync_ms:.2f} ms")
            print(f"Loss sync overhead:       {multi_result.loss_sync_ms:.2f} ms")
            print(f"Total overhead:           {multi_result.model_sync_ms + multi_result.loss_sync_ms:.2f} ms")
        
        # Compare single-GPU approaches
        single_parallel = None
        single_pipelined = None
        for r in results:
            if "Single-GPU Parallel" in r.method:
                single_parallel = r
            elif "Single-GPU Pipelined" in r.method:
                single_pipelined = r
        
        if single_parallel and single_pipelined:
            print("\n" + "-" * 90)
            print("SINGLE-GPU APPROACHES COMPARISON:")
            print("-" * 90)
            print(f"Parallel (2 buffers):    {single_parallel.total_time_ms:.2f} ms")
            print(f"Pipelined (1 buffer):     {single_pipelined.total_time_ms:.2f} ms")
            print(f"Memory overhead (parallel): {single_parallel.memory_main_mb:.2f} MB (2x buffers)")
            print(f"Memory overhead (pipelined): {single_pipelined.memory_main_mb:.2f} MB (1 buffer + temp)")
            if single_parallel.speedup_vs_sequential > 0:
                print(f"Parallel speedup:         {single_parallel.speedup_vs_sequential:.2f}x")
            if single_pipelined.speedup_vs_sequential > 0:
                print(f"Pipelined speedup:       {single_pipelined.speedup_vs_sequential:.2f}x")


if __name__ == "__main__":
    main()
