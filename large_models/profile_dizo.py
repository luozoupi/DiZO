"""
DiZO Profiling Script - Profile GPU/CPU activities during zeroth-order finetuning

This script profiles the DiZO training process using:
1. PyTorch Profiler - for detailed GPU/CPU activity timeline
2. Generates Chrome trace files for visualization

Usage:
    python profile_dizo.py --model_name facebook/opt-350m --task_name SST2 --num_steps 5
    
To visualize:
    - Open Chrome, go to chrome://tracing, and load the generated .json file
    - Or use TensorBoard: tensorboard --logdir=./profiler_logs
"""

import logging
import os
import sys
import argparse
import time
import random
import copy
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch._dynamo
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader
from torch.profiler import profile, record_function, ProfilerActivity, schedule, tensorboard_trace_handler

from transformers import (
    AutoConfig, AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, DataCollatorForTokenClassification
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import local modules
import tasks
from tasks import get_task
from utils import encode_prompt, count_time
from dataset import FewShotDataset
from lora import LoRA

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class ProfilerArguments:
    """Arguments for profiling"""
    model_name: str = "facebook/opt-350m"  # Smaller model for quick profiling
    task_name: str = "SST2"
    num_steps: int = 5  # Number of training steps to profile
    batch_size: int = 4
    max_length: int = 512
    zo_eps: float = 1e-3
    learning_rate: float = 1e-7
    load_float16: bool = True
    output_dir: str = "./profiler_logs"
    profile_memory: bool = True
    with_stack: bool = True  # Include Python call stack in profile
    torch_compile: bool = False  # Use torch.compile() on model
    compile_mode: str = "default"  # torch.compile mode: default, reduce-overhead, max-autotune
    compile_warmup_steps: int = 3  # Warmup steps to complete torch.compile before profiling
    memory_timeline: bool = False  # Record detailed memory allocation timeline
    export_memory_snapshot: bool = False  # Export memory snapshot for visualization
    # LoRA options
    use_lora: bool = False  # Use LoRA for parameter-efficient training
    lora_r: int = 8  # LoRA rank
    lora_alpha: int = 16  # LoRA alpha (scaling factor)
    # Flat buffer equivalence experiment options
    flat_buffer: bool = False  # Use flat buffer method for ZO perturbation/update
    equivalence_test: bool = False  # Run baseline vs flat buffer comparison
    equivalence_steps: int = 50  # Number of steps for equivalence test
    plot_loss_curve: bool = True  # Plot loss curves during equivalence test
    flat_buffer_optimized: bool = False  # Use optimized single-RNG flat buffer (not equiv to baseline)
    
    def get_model_short_name(self) -> str:
        """Extract short model name for output naming"""
        # facebook/opt-350m -> opt-350m
        # facebook/opt-2.7b -> opt-2.7b
        name = self.model_name.split("/")[-1]
        return name.lower().replace(".", "_")
    
    def get_output_subdir(self) -> str:
        """Get model-specific output subdirectory"""
        model_name = self.get_model_short_name()
        compile_suffix = "_compiled" if self.torch_compile else ""
        lora_suffix = f"_lora_r{self.lora_r}" if self.use_lora else ""
        flat_suffix = "_flatbuf" if self.flat_buffer else ""
        equiv_suffix = "_equiv" if self.equivalence_test else ""
        return os.path.join(self.output_dir, f"profile_mezo_{model_name}{lora_suffix}{compile_suffix}{flat_suffix}{equiv_suffix}")
    
    def get_trace_filename(self) -> str:
        """Get trace filename with model name and compile status"""
        model_name = self.get_model_short_name()
        compile_suffix = "_compiled" if self.torch_compile else ""
        lora_suffix = f"_lora_r{self.lora_r}" if self.use_lora else ""
        return f"mezo_{model_name}{lora_suffix}{compile_suffix}.pt.trace.json"


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SimpleDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def forward_wrap_with_option_len(model, input_ids=None, labels=None, option_len=None, 
                                  num_options=None, return_dict=None, **kwargs):
    """Forward pass with option length handling"""
    with torch.no_grad():
        outputs = model.forward(input_ids=input_ids, **kwargs)
    
    if labels is None:
        return outputs
    
    logits = outputs.logits
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = torch.clone(input_ids)[..., 1:].contiguous()
    shift_labels[shift_labels == model.config.pad_token_id] = -100
    
    if option_len is not None:
        for _i, _len in enumerate(option_len):
            shift_labels[_i, :-_len] = -100
    
    loss_fct = CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(shift_logits.view(-1, model.config.vocab_size), shift_labels.view(-1))
    
    return type('Output', (), {'loss': loss, 'logits': logits})()


class MeZOTrainer:
    """Simplified MeZO trainer for profiling"""
    
    def __init__(self, model, tokenizer, args: ProfilerArguments, dataloader):
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.dataloader = dataloader
        self.data_iter = iter(dataloader)
        self.device = next(model.parameters()).device
        
        # Collect trainable parameters
        self.named_parameters_to_optim = [
            (name, param) for name, param in model.named_parameters() if param.requires_grad
        ]
        logger.info(f"Number of trainable parameters: {sum(p.numel() for _, p in self.named_parameters_to_optim)}")
    
    def get_parameter_shapes(self) -> list:
        """
        Get shapes/dimensions of all parameters to be perturbed and updated.
        Returns a list of dicts with parameter info.
        """
        param_info = []
        total_params = 0
        total_memory_bytes = 0
        
        for name, param in self.named_parameters_to_optim:
            shape = tuple(param.shape)
            numel = param.numel()
            dtype = param.dtype
            # Calculate memory in bytes (fp16=2, fp32=4, bf16=2)
            bytes_per_elem = 2 if dtype in [torch.float16, torch.bfloat16] else 4
            memory_bytes = numel * bytes_per_elem
            
            param_info.append({
                'name': name,
                'shape': shape,
                'numel': numel,
                'dtype': str(dtype),
                'memory_mb': memory_bytes / (1024 * 1024),
                'ndim': len(shape)
            })
            total_params += numel
            total_memory_bytes += memory_bytes
        
        # Add summary at the end
        param_info.append({
            'name': '--- TOTAL ---',
            'shape': None,
            'numel': total_params,
            'dtype': 'N/A',
            'memory_mb': total_memory_bytes / (1024 * 1024),
            'ndim': None
        })
        
        return param_info
    
    def save_parameter_shapes(self, output_path: str):
        """
        Save parameter shapes to a text file.
        """
        param_info = self.get_parameter_shapes()
        
        with open(output_path, 'w') as f:
            f.write("=" * 120 + "\n")
            f.write("PARAMETER SHAPES FOR ZO PERTURBATION AND UPDATE\n")
            f.write("=" * 120 + "\n\n")
            f.write(f"Model: {self.args.model_name}\n")
            f.write(f"Total trainable parameters: {len(self.named_parameters_to_optim)}\n\n")
            
            # Header
            f.write(f"{'No.':<6} {'Parameter Name':<60} {'Shape':<25} {'Elements':>15} {'Memory (MB)':>12} {'Dtype':<15}\n")
            f.write("-" * 120 + "\n")
            
            for i, info in enumerate(param_info[:-1]):  # Exclude summary row
                shape_str = str(info['shape'])
                f.write(f"{i+1:<6} {info['name']:<60} {shape_str:<25} {info['numel']:>15,} {info['memory_mb']:>12.4f} {info['dtype']:<15}\n")
            
            # Summary
            f.write("-" * 120 + "\n")
            summary = param_info[-1]
            f.write(f"{'TOTAL':<6} {'':<60} {'':<25} {summary['numel']:>15,} {summary['memory_mb']:>12.4f}\n")
            f.write("=" * 120 + "\n")
            
            # Additional statistics
            f.write("\n\nPARAMETER STATISTICS BY LAYER TYPE:\n")
            f.write("-" * 60 + "\n")
            
            layer_stats = {}
            for info in param_info[:-1]:
                # Extract layer type from name (e.g., 'model.decoder.layers.0.self_attn.q_proj.weight' -> 'self_attn.q_proj')
                name_parts = info['name'].split('.')
                if 'weight' in name_parts[-1] or 'bias' in name_parts[-1]:
                    layer_type = '.'.join(name_parts[-2:]) if len(name_parts) >= 2 else name_parts[-1]
                else:
                    layer_type = name_parts[-1]
                
                if layer_type not in layer_stats:
                    layer_stats[layer_type] = {'count': 0, 'numel': 0, 'memory_mb': 0}
                layer_stats[layer_type]['count'] += 1
                layer_stats[layer_type]['numel'] += info['numel']
                layer_stats[layer_type]['memory_mb'] += info['memory_mb']
            
            f.write(f"{'Layer Type':<40} {'Count':>10} {'Elements':>20} {'Memory (MB)':>15}\n")
            f.write("-" * 85 + "\n")
            for layer_type, stats in sorted(layer_stats.items(), key=lambda x: -x[1]['numel']):
                f.write(f"{layer_type:<40} {stats['count']:>10} {stats['numel']:>20,} {stats['memory_mb']:>15.4f}\n")
        
        logger.info(f"Parameter shapes saved to: {output_path}")
        return param_info
    
    def print_parameter_shapes_summary(self):
        """Print a summary of parameter shapes to the console."""
        param_info = self.get_parameter_shapes()
        
        logger.info("\n" + "=" * 80)
        logger.info("PARAMETER SHAPES SUMMARY (for ZO perturbation)")
        logger.info("=" * 80)
        
        # Group by shape
        shape_groups = {}
        for info in param_info[:-1]:
            shape = info['shape']
            if shape not in shape_groups:
                shape_groups[shape] = {'count': 0, 'names': [], 'memory_mb': 0}
            shape_groups[shape]['count'] += 1
            shape_groups[shape]['names'].append(info['name'])
            shape_groups[shape]['memory_mb'] += info['memory_mb']
        
        logger.info(f"\nUnique shapes: {len(shape_groups)}")
        logger.info(f"Total parameters: {param_info[-1]['numel']:,}")
        logger.info(f"Total memory: {param_info[-1]['memory_mb']:.2f} MB")
        
        # Top shapes by count
        logger.info("\nTop 10 shapes by frequency:")
        for shape, stats in sorted(shape_groups.items(), key=lambda x: -x[1]['count'])[:10]:
            logger.info(f"  {str(shape):<30} count={stats['count']:<5} memory={stats['memory_mb']:.2f} MB")
    
    def get_batch(self):
        try:
            batch = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.dataloader)
            batch = next(self.data_iter)
        result = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                result[k] = v.to(self.device)
            else:
                result[k] = v  # Keep lists (like option_len) as is
        return result
    
    def zo_perturb_parameters(self, random_seed, scaling_factor=1):
        """Perturb parameters with random vector z"""
        torch.manual_seed(random_seed)
        for name, param in self.named_parameters_to_optim:
            z = torch.normal(mean=0, std=1, size=param.data.size(), 
                           device=param.data.device, dtype=param.data.dtype)
            param.data = param.data + scaling_factor * z * self.args.zo_eps
    
    def zo_forward(self, inputs):
        """Forward pass without gradients"""
        self.model.eval()
        with torch.inference_mode():
            loss = forward_wrap_with_option_len(self.model, **inputs, return_dict=True).loss
        return loss.detach()
    
    def zo_step(self, inputs):
        """One MeZO step: estimate gradient via finite differences"""
        # Sample random seed for z
        zo_random_seed = np.random.randint(1000000000)
        
        # f(theta + eps * z)
        with record_function("zo_perturb_+eps"):
            self.zo_perturb_parameters(zo_random_seed, scaling_factor=1)
        
        with record_function("zo_forward_1"):
            loss1 = self.zo_forward(inputs)
        
        # f(theta - eps * z)
        with record_function("zo_perturb_-2eps"):
            self.zo_perturb_parameters(zo_random_seed, scaling_factor=-2)
        
        with record_function("zo_forward_2"):
            loss2 = self.zo_forward(inputs)
        
        # Compute gradient estimate
        projected_grad = ((loss1 - loss2) / (2 * self.args.zo_eps)).item()
        
        # Reset to original parameters
        with record_function("zo_perturb_+eps_reset"):
            self.zo_perturb_parameters(zo_random_seed, scaling_factor=1)
        
        return loss1, projected_grad, zo_random_seed
    
    def zo_update(self, projected_grad, zo_random_seed):
        """Update parameters with estimated gradient"""
        torch.manual_seed(zo_random_seed)
        with record_function("zo_update"):
            for name, param in self.named_parameters_to_optim:
                z = torch.normal(mean=0, std=1, size=param.data.size(),
                               device=param.data.device, dtype=param.data.dtype)
                param.data = param.data - self.args.learning_rate * projected_grad * z


class FlatBufferMeZOTrainer:
    """
    MeZO trainer using flat buffer technique for all parameters.
    
    This implementation creates a single contiguous flat buffer containing all
    trainable parameters, then uses view() to create views back to original shapes.
    
    Key features:
    1. Single contiguous memory allocation for all parameters
    2. Single RNG call generates perturbation for all parameters at once
    3. Operations on flat buffer to minimize kernel launches
    4. Views maintain original tensor shapes for model compatibility
    
    This serves as a PyTorch reference implementation to verify algorithmic
    equivalence before integrating CUDA/Triton fused kernels.
    """
    
    def __init__(self, model, tokenizer, args: ProfilerArguments, dataloader):
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.dataloader = dataloader
        self.data_iter = iter(dataloader)
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        
        # Collect trainable parameters with their metadata
        self.param_metadata = []  # List of (name, param, offset, numel, shape)
        self.total_numel = 0
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.param_metadata.append({
                    'name': name,
                    'param': param,
                    'offset': self.total_numel,
                    'numel': param.numel(),
                    'shape': param.shape,
                })
                self.total_numel += param.numel()
        
        logger.info(f"[FlatBuffer] Total trainable parameters: {len(self.param_metadata)}")
        logger.info(f"[FlatBuffer] Total elements: {self.total_numel:,}")
        logger.info(f"[FlatBuffer] Memory for flat buffer: {self.total_numel * (2 if self.dtype == torch.float16 else 4) / 1024**2:.2f} MB")
        
        # Create flat buffer for perturbation vector z
        # This is allocated once and reused for each step
        self.z_flat = torch.zeros(self.total_numel, device=self.device, dtype=self.dtype)
        
        # Timing statistics
        self.timing = {
            'rng': [],
            'perturb': [],
            'update': [],
            'forward': [],
        }
        
        # Compatibility mode flag - when True, generate random numbers same way as baseline
        # This is for equivalence testing only; set False for actual optimization (single randn call)
        self.compat_mode = not args.flat_buffer_optimized
        if self.compat_mode:
            logger.info("[FlatBuffer] Compatibility mode: ON (same RNG as baseline)")
        else:
            logger.info("[FlatBuffer] Optimized mode: ON (single batched RNG call)")
    
    # Reuse base class methods for getting batches
    def get_batch(self):
        try:
            batch = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.dataloader)
            batch = next(self.data_iter)
        result = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                result[k] = v.to(self.device)
            else:
                result[k] = v
        return result
    
    def _generate_z_flat(self, seed: int):
        """
        Generate flat perturbation vector z.
        
        When compat_mode=True: Generate per-parameter (same RNG behavior as baseline)
        When compat_mode=False: Single batched RNG call (optimized)
        """
        torch.manual_seed(seed)
        
        if self.compat_mode:
            # Generate random numbers per-parameter to match baseline behavior
            # This ensures exact same random values as the baseline MeZOTrainer
            for i, meta in enumerate(self.param_metadata):
                z = torch.normal(mean=0, std=1, size=meta['shape'],
                               device=self.device, dtype=self.dtype)
                self.z_flat[meta['offset']:meta['offset'] + meta['numel']].copy_(z.view(-1))
        else:
            # Optimized: Single batched RNG call for ALL parameters
            self.z_flat = torch.randn(self.total_numel, device=self.device, dtype=self.dtype)
    
    def _get_z_view(self, param_idx: int) -> torch.Tensor:
        """
        Get a view of z_flat corresponding to parameter at param_idx.
        
        The view has the same shape as the original parameter, allowing
        direct addition/subtraction with parameter tensors.
        """
        meta = self.param_metadata[param_idx]
        # Create view with original parameter shape
        return self.z_flat[meta['offset']:meta['offset'] + meta['numel']].view(meta['shape'])
    
    def zo_perturb_parameters_flat(self, random_seed: int, scaling_factor: float = 1.0):
        """
        Perturb all parameters using flat buffer technique.
        
        For positive perturbation (scaling_factor=1): θ = θ + ε*z
        For negative perturbation (scaling_factor=-2): θ = θ - 2ε*z
        For reset (scaling_factor=1 after -2): θ = θ + ε*z
        
        Implementation notes:
        - On first call with a new seed, generate z_flat
        - Subsequent calls with same seed reuse z_flat (no regeneration)
        - Uses view() to maintain original tensor shapes
        """
        t0 = time.perf_counter()
        
        # Generate or reuse z_flat based on seed
        # The seed is stored to detect when a new z is needed
        if not hasattr(self, '_current_seed') or self._current_seed != random_seed:
            self._generate_z_flat(random_seed)
            self._current_seed = random_seed
            self.timing['rng'].append((time.perf_counter() - t0) * 1000)
            t0 = time.perf_counter()
        
        # Apply perturbation to each parameter using views
        with record_function("flat_buffer_perturb"):
            for i, meta in enumerate(self.param_metadata):
                z_view = self._get_z_view(i)
                meta['param'].data.add_(z_view, alpha=scaling_factor * self.args.zo_eps)
        
        self.timing['perturb'].append((time.perf_counter() - t0) * 1000)
    
    def zo_forward(self, inputs):
        """Forward pass without gradients (same as baseline)"""
        t0 = time.perf_counter()
        self.model.eval()
        with torch.inference_mode():
            loss = forward_wrap_with_option_len(self.model, **inputs, return_dict=True).loss
        self.timing['forward'].append((time.perf_counter() - t0) * 1000)
        return loss.detach()
    
    def zo_step(self, inputs):
        """
        One MeZO step using flat buffer technique.
        
        This is algorithmically equivalent to the baseline MeZOTrainer.zo_step(),
        but uses:
        1. Single RNG call for all parameters
        2. Views instead of separate allocations
        3. Potential for future single-kernel operations
        """
        # Sample random seed for z
        zo_random_seed = np.random.randint(1000000000)
        
        # f(θ + ε*z)
        with record_function("flat_zo_perturb_+eps"):
            self.zo_perturb_parameters_flat(zo_random_seed, scaling_factor=1)
        
        with record_function("flat_zo_forward_1"):
            loss1 = self.zo_forward(inputs)
        
        # f(θ - ε*z) - achieved by subtracting 2ε*z from current θ+ε*z
        with record_function("flat_zo_perturb_-2eps"):
            self.zo_perturb_parameters_flat(zo_random_seed, scaling_factor=-2)
        
        with record_function("flat_zo_forward_2"):
            loss2 = self.zo_forward(inputs)
        
        # Compute gradient estimate: g = (f(θ+ε*z) - f(θ-ε*z)) / (2ε)
        projected_grad = ((loss1 - loss2) / (2 * self.args.zo_eps)).item()
        
        # Reset to original θ
        with record_function("flat_zo_perturb_+eps_reset"):
            self.zo_perturb_parameters_flat(zo_random_seed, scaling_factor=1)
        
        return loss1, projected_grad, zo_random_seed
    
    def zo_update(self, projected_grad: float, zo_random_seed: int):
        """
        Update parameters using flat buffer technique.
        
        θ = θ - lr * g * z
        
        Uses the same z that was generated during zo_step.
        """
        t0 = time.perf_counter()
        
        # Ensure z_flat matches the seed (should already be set from zo_step)
        if not hasattr(self, '_current_seed') or self._current_seed != zo_random_seed:
            self._generate_z_flat(zo_random_seed)
            self._current_seed = zo_random_seed
        
        with record_function("flat_zo_update"):
            for i, meta in enumerate(self.param_metadata):
                z_view = self._get_z_view(i)
                # θ = θ - lr * g * z
                meta['param'].data.add_(z_view, alpha=-self.args.learning_rate * projected_grad)
        
        self.timing['update'].append((time.perf_counter() - t0) * 1000)
    
    def print_timing_stats(self):
        """Print timing statistics for flat buffer operations"""
        logger.info("\n" + "=" * 60)
        logger.info("FLAT BUFFER TIMING STATISTICS")
        logger.info("=" * 60)
        for key, times in self.timing.items():
            if times:
                avg = np.mean(times)
                std = np.std(times) if len(times) > 1 else 0
                logger.info(f"  {key:15s}: {avg:8.3f} ± {std:6.3f} ms (n={len(times)})")


def run_equivalence_test(args: ProfilerArguments, model, tokenizer, task, dataloader):
    """
    Run equivalence test comparing baseline MeZO vs flat buffer MeZO.
    
    This test verifies that the flat buffer implementation produces
    identical loss values as the baseline, step by step.
    
    Returns:
        dict with loss histories and comparison results
    """
    import copy
    
    logger.info("\n" + "=" * 80)
    logger.info("FLAT BUFFER EQUIVALENCE TEST")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Steps: {args.equivalence_steps}")
    logger.info("=" * 80)
    
    # Create two separate model copies with identical initial weights
    logger.info("Creating two separate model instances for comparison...")
    
    # Store original state
    original_state = copy.deepcopy(model.state_dict())
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    
    # Create baseline model (using the passed model)
    model.load_state_dict(original_state)
    baseline_trainer = MeZOTrainer(model, tokenizer, args, dataloader)
    
    # Create a second model instance for flat buffer trainer
    logger.info("Loading second model instance for flat buffer trainer...")
    config = AutoConfig.from_pretrained(args.model_name)
    torch_dtype = torch.float16 if args.load_float16 else torch.float32
    
    model_flat = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        config=config,
        torch_dtype=torch_dtype,
        device_map='auto'
    )
    model_flat.eval()
    
    # Apply LoRA if needed
    if args.use_lora:
        LoRA(model_flat, r=args.lora_r, alpha=args.lora_alpha, float16=args.load_float16)
    
    # Ensure same initial weights
    model_flat.load_state_dict(original_state)
    
    # Create second dataloader with same data
    from dataset import FewShotDataset
    train_data = prepare_data(task, tokenizer, args)
    dataloader_flat = DataLoader(
        SimpleDataset(train_data),
        batch_size=args.batch_size,
        shuffle=False,  # Don't shuffle to ensure same data order
        collate_fn=lambda x: collate_fn(x, tokenizer.pad_token_id or 0)
    )
    
    # Also reset baseline dataloader to not shuffle
    dataloader_baseline = DataLoader(
        SimpleDataset(train_data),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda x: collate_fn(x, tokenizer.pad_token_id or 0)
    )
    
    # Recreate baseline trainer with non-shuffled dataloader
    baseline_trainer = MeZOTrainer(model, tokenizer, args, dataloader_baseline)
    flat_trainer = FlatBufferMeZOTrainer(model_flat, tokenizer, args, dataloader_flat)
    
    # Results storage
    results = {
        'baseline_losses': [],
        'flat_buffer_losses': [],
        'loss_diffs': [],
        'projected_grads_baseline': [],
        'projected_grads_flat': [],
        'grad_diffs': [],
        'seeds': [],
    }
    
    # Reset data iterators
    baseline_trainer.data_iter = iter(dataloader_baseline)
    flat_trainer.data_iter = iter(dataloader_flat)
    
    logger.info(f"\nRunning {args.equivalence_steps} steps for both implementations...")
    logger.info("-" * 80)
    
    for step in range(args.equivalence_steps):
        # Get batch (both trainers have same data since shuffle=False)
        inputs_baseline = baseline_trainer.get_batch()
        inputs_flat = flat_trainer.get_batch()
        
        # Use same random seed for ZO perturbation
        # The seed is generated inside zo_step, so we need to control np.random
        np.random.seed(42 + step)
        loss_baseline, grad_baseline, seed_baseline = baseline_trainer.zo_step(inputs_baseline)
        baseline_trainer.zo_update(grad_baseline, seed_baseline)
        
        # Reset np.random to same state for flat buffer
        np.random.seed(42 + step)
        loss_flat, grad_flat, seed_flat = flat_trainer.zo_step(inputs_flat)
        flat_trainer.zo_update(grad_flat, seed_flat)
        
        # Verify seeds match
        assert seed_baseline == seed_flat, f"Seeds don't match: {seed_baseline} vs {seed_flat}"
        
        # Record results
        loss_b = loss_baseline.item() if hasattr(loss_baseline, 'item') else loss_baseline
        loss_f = loss_flat.item() if hasattr(loss_flat, 'item') else loss_flat
        
        results['baseline_losses'].append(loss_b)
        results['flat_buffer_losses'].append(loss_f)
        results['loss_diffs'].append(abs(loss_b - loss_f))
        results['projected_grads_baseline'].append(grad_baseline)
        results['projected_grads_flat'].append(grad_flat)
        results['grad_diffs'].append(abs(grad_baseline - grad_flat))
        results['seeds'].append(seed_baseline)
        
        if step % 10 == 0 or step == args.equivalence_steps - 1:
            logger.info(f"Step {step:4d}: Baseline loss={loss_b:.6f}, FlatBuf loss={loss_f:.6f}, "
                       f"Diff={abs(loss_b - loss_f):.2e}, seed={seed_baseline}")
    
    # Analysis
    logger.info("\n" + "=" * 80)
    logger.info("EQUIVALENCE TEST RESULTS")
    logger.info("=" * 80)
    
    max_loss_diff = max(results['loss_diffs'])
    avg_loss_diff = np.mean(results['loss_diffs'])
    max_grad_diff = max(results['grad_diffs'])
    avg_grad_diff = np.mean(results['grad_diffs'])
    
    logger.info(f"Loss differences:")
    logger.info(f"  Max:     {max_loss_diff:.2e}")
    logger.info(f"  Average: {avg_loss_diff:.2e}")
    
    logger.info(f"\nProjected gradient differences:")
    logger.info(f"  Max:     {max_grad_diff:.2e}")
    logger.info(f"  Average: {avg_grad_diff:.2e}")
    
    # Determine if equivalent (within numerical precision)
    # For fp16, expect ~1e-2 precision due to limited mantissa bits
    # Loss differences accumulate over steps due to weight drift
    loss_tolerance = 0.05 if args.load_float16 else 1e-4
    # For gradient differences, scale tolerance by typical gradient magnitude
    avg_grad_magnitude = (np.mean(np.abs(results['projected_grads_baseline'])) + 
                          np.mean(np.abs(results['projected_grads_flat']))) / 2
    grad_tolerance = max(loss_tolerance * 100, avg_grad_magnitude * 0.05)  # 5% relative tolerance
    
    loss_equiv = max_loss_diff < loss_tolerance
    grad_equiv = max_grad_diff < grad_tolerance or avg_grad_diff < (avg_grad_magnitude * 0.01)  # 1% avg relative
    is_equivalent = loss_equiv and grad_equiv
    
    logger.info(f"\nEquivalence check:")
    logger.info(f"  Loss tolerance: {loss_tolerance:.2e} (max diff: {max_loss_diff:.2e}) -> {'PASS' if loss_equiv else 'FAIL'}")
    logger.info(f"  Grad tolerance: {grad_tolerance:.2e} (max diff: {max_grad_diff:.2e}) -> {'PASS' if grad_equiv else 'FAIL'}")
    
    if is_equivalent:
        logger.info("  ✓ PASS: Flat buffer implementation is numerically equivalent to baseline")
    else:
        logger.info("  ✗ FAIL: Significant numerical differences detected")
        logger.info("  This may indicate:")
        logger.info("    - Different RNG behavior")
        logger.info("    - Memory ordering issues")
        logger.info("    - Numerical precision differences beyond fp16 tolerance")
    
    # Print flat buffer timing stats
    flat_trainer.print_timing_stats()
    
    # Plot loss curves if requested
    if args.plot_loss_curve:
        output_subdir = args.get_output_subdir()
        os.makedirs(output_subdir, exist_ok=True)
        
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Loss curves
            ax1 = axes[0, 0]
            steps = range(len(results['baseline_losses']))
            ax1.plot(steps, results['baseline_losses'], 'b-', label='Baseline MeZO', linewidth=1.5)
            ax1.plot(steps, results['flat_buffer_losses'], 'r--', label='Flat Buffer MeZO', linewidth=1.5)
            ax1.set_xlabel('Step')
            ax1.set_ylabel('Loss')
            ax1.set_title('Loss Curves: Baseline vs Flat Buffer')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Loss difference
            ax2 = axes[0, 1]
            ax2.semilogy(steps, results['loss_diffs'], 'g-', linewidth=1.5)
            ax2.axhline(y=loss_tolerance, color='r', linestyle='--', label=f'Loss Tolerance ({loss_tolerance})')
            ax2.set_xlabel('Step')
            ax2.set_ylabel('|Loss Diff| (log scale)')
            ax2.set_title('Loss Difference per Step')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Projected gradients
            ax3 = axes[1, 0]
            ax3.plot(steps, results['projected_grads_baseline'], 'b-', label='Baseline', linewidth=1.5)
            ax3.plot(steps, results['projected_grads_flat'], 'r--', label='Flat Buffer', linewidth=1.5)
            ax3.set_xlabel('Step')
            ax3.set_ylabel('Projected Gradient')
            ax3.set_title('Projected Gradient Estimates')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Gradient difference
            ax4 = axes[1, 1]
            ax4.semilogy(steps, results['grad_diffs'], 'g-', linewidth=1.5)
            ax4.axhline(y=grad_tolerance, color='r', linestyle='--', label=f'Grad Tolerance ({grad_tolerance:.1f})')
            ax4.set_xlabel('Step')
            ax4.set_ylabel('|Grad Diff| (log scale)')
            ax4.set_title('Gradient Difference per Step')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            plot_path = os.path.join(output_subdir, f"{args.get_model_short_name()}_equivalence_test.png")
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"\nEquivalence plot saved to: {plot_path}")
            
        except Exception as e:
            logger.warning(f"Failed to create plot: {e}")
    
    # Save results to CSV
    output_subdir = args.get_output_subdir()
    csv_path = os.path.join(output_subdir, f"{args.get_model_short_name()}_equivalence_results.csv")
    with open(csv_path, 'w') as f:
        f.write("step,baseline_loss,flat_buffer_loss,loss_diff,baseline_grad,flat_buffer_grad,grad_diff\n")
        for i in range(len(results['baseline_losses'])):
            f.write(f"{i},{results['baseline_losses'][i]:.8f},{results['flat_buffer_losses'][i]:.8f},"
                   f"{results['loss_diffs'][i]:.8e},{results['projected_grads_baseline'][i]:.8f},"
                   f"{results['projected_grads_flat'][i]:.8f},{results['grad_diffs'][i]:.8e}\n")
    logger.info(f"Results saved to: {csv_path}")
    
    return results, is_equivalent


def prepare_data(task, tokenizer, args):
    """Prepare training data"""
    train_samples = task.samples['train'][:100]  # Use subset for profiling
    
    data = []
    for sample in train_samples:
        encoded_candidates, option_lens = encode_prompt(
            task, task.get_template(), [], sample, tokenizer,
            max_length=args.max_length, generation=task.generation, generation_with_gold=True,
            max_new_tokens=50
        )
        
        if task.generation:
            correct_candidate_id = 0
        elif isinstance(sample.correct_candidate, list):
            correct_candidate_id = sample.candidates.index(sample.correct_candidate[0])
        else:
            correct_candidate_id = sample.candidates.index(sample.correct_candidate)
        
        data.append({
            "input_ids": torch.tensor(encoded_candidates[correct_candidate_id]),
            "labels": torch.tensor(encoded_candidates[correct_candidate_id]),
            "option_len": option_lens[correct_candidate_id]
        })
    
    return data


def collate_fn(batch, pad_token_id):
    """Custom collate function"""
    max_len = max(item['input_ids'].size(0) for item in batch)
    
    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    option_lens = []
    
    for i, item in enumerate(batch):
        seq_len = item['input_ids'].size(0)
        # Left padding
        input_ids[i, max_len - seq_len:] = item['input_ids']
        labels[i, max_len - seq_len:] = item['labels']
        option_lens.append(item['option_len'])
    
    return {
        'input_ids': input_ids,
        'labels': labels,
        'option_len': option_lens
    }


def run_profiling(args: ProfilerArguments):
    """Main profiling function"""
    set_seed(42)
    
    # Create model-specific output directory
    output_subdir = args.get_output_subdir()
    os.makedirs(output_subdir, exist_ok=True)
    
    model_short_name = args.get_model_short_name()
    logger.info(f"="*80)
    logger.info(f"PROFILING MODEL: {args.model_name}")
    if args.use_lora:
        logger.info(f"LoRA: enabled (r={args.lora_r}, alpha={args.lora_alpha})")
    else:
        logger.info(f"LoRA: disabled (full model ZO)")
    logger.info(f"Output directory: {output_subdir}")
    logger.info(f"="*80)
    
    # Load model
    logger.info(f"Loading model: {args.model_name}")
    torch_dtype = torch.float16 if args.load_float16 else torch.float32
    
    config = AutoConfig.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        config=config,
        torch_dtype=torch_dtype,
        device_map='auto'
    )
    model.eval()
    
    # Apply LoRA if requested
    if args.use_lora:
        logger.info(f"Applying LoRA with r={args.lora_r}, alpha={args.lora_alpha}...")
        lora_wrapper = LoRA(model, r=args.lora_r, alpha=args.lora_alpha, float16=args.load_float16)
        num_lora_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"LoRA applied. Trainable parameters: {num_lora_params:,} (vs full model)")
    
    # Apply torch.compile if requested
    if args.torch_compile:
        # Increase dynamo cache size to avoid recompilation due to varying input shapes
        torch._dynamo.config.cache_size_limit = 512
        
        logger.info(f"Applying torch.compile with mode='{args.compile_mode}'...")
        model = torch.compile(model, mode=args.compile_mode)
        logger.info("Model compiled successfully")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    if "opt" in args.model_name:
        tokenizer.bos_token_id = 0
    
    # Load task and prepare data
    logger.info(f"Loading task: {args.task_name}")
    task = get_task(args.task_name)
    
    logger.info("Preparing training data...")
    train_data = prepare_data(task, tokenizer, args)
    
    dataloader = DataLoader(
        SimpleDataset(train_data),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda x: collate_fn(x, tokenizer.pad_token_id or 0)
    )
    
    # Create trainer
    if args.flat_buffer:
        logger.info("Using FLAT BUFFER trainer for ZO operations")
        trainer = FlatBufferMeZOTrainer(model, tokenizer, args, dataloader)
        # FlatBufferMeZOTrainer doesn't have save_parameter_shapes, so skip that
        logger.info(f"[FlatBuffer] Trainer initialized")
    else:
        trainer = MeZOTrainer(model, tokenizer, args, dataloader)
        # Save parameter shapes to file
        param_shapes_path = os.path.join(output_subdir, f"{model_short_name}_parameter_shapes.txt")
        trainer.save_parameter_shapes(param_shapes_path)
        trainer.print_parameter_shapes_summary()
    
    # =========================================================================
    # EQUIVALENCE TEST MODE - Compare baseline vs flat buffer
    # =========================================================================
    if args.equivalence_test:
        # Reload model for equivalence test (need fresh copies)
        logger.info("Running equivalence test mode...")
        model_for_test = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            config=config,
            torch_dtype=torch_dtype,
            device_map='auto'
        )
        model_for_test.eval()
        
        if args.use_lora:
            LoRA(model_for_test, r=args.lora_r, alpha=args.lora_alpha, float16=args.load_float16)
        
        # Recreate dataloader for equivalence test
        equiv_dataloader = DataLoader(
            SimpleDataset(train_data),
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda x: collate_fn(x, tokenizer.pad_token_id or 0)
        )
        
        results, is_equivalent = run_equivalence_test(args, model_for_test, tokenizer, task, equiv_dataloader)
        logger.info(f"\nEquivalence test completed. Result: {'PASS' if is_equivalent else 'FAIL'}")
        return output_subdir
    
    # =========================================================================
    # TORCH.COMPILE WARMUP - Complete compilation BEFORE profiling
    # =========================================================================
    # torch.compile uses lazy compilation - it compiles on first execution.
    # We need to run warmup steps to trigger compilation before profiling,
    # otherwise profiling will capture compilation overhead instead of
    # steady-state optimized execution.
    if args.torch_compile and args.compile_warmup_steps > 0:
        logger.info(f"Running {args.compile_warmup_steps} compile warmup steps (outside profiler)...")
        logger.info("This completes torch.compile graph compilation before profiling starts.")
        
        for warmup_step in range(args.compile_warmup_steps):
            # Get batch
            inputs = trainer.get_batch()
            
            # Run ZO step (triggers forward pass compilation)
            loss, projected_grad, zo_random_seed = trainer.zo_step(inputs)
            
            # Run ZO update
            trainer.zo_update(projected_grad, zo_random_seed)
            
            logger.info(f"  Compile warmup step {warmup_step + 1}/{args.compile_warmup_steps}, loss = {loss.item():.4f}")
        
        # Ensure all CUDA operations (including compilation) are complete
        torch.cuda.synchronize()
        
        # Reset memory stats after warmup
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        logger.info("Compile warmup complete. All graphs compiled. Starting profiling...")
    
    # Define profiling schedule
    # wait=1: skip first step (warmup)
    # warmup=1: warmup profiling
    # active=num_steps: actively profile these steps
    # repeat=1: only one cycle
    profiler_schedule = schedule(
        wait=1,
        warmup=1,
        active=args.num_steps,
        repeat=1
    )
    
    # Activities to profile
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    
    logger.info(f"Starting profiling for {args.num_steps} steps...")
    logger.info(f"Output directory: {output_subdir}")
    
    # Start memory timeline recording if requested
    if args.memory_timeline:
        logger.info("Starting CUDA memory history recording...")
        # PyTorch 2.3.x API: enabled, context, stacks, max_entries, device
        torch.cuda.memory._record_memory_history(
            enabled="all",
            context="all",
            stacks="python",
            max_entries=100000,
        )
    
    # Track peak memory per phase
    memory_stats = {
        'peak_allocated': [],
        'peak_reserved': [],
        'phase_memory': {}
    }
    
    # PyTorch Profiler
    with profile(
        activities=activities,
        schedule=profiler_schedule,
        on_trace_ready=tensorboard_trace_handler(output_subdir),
        record_shapes=True,
        profile_memory=args.profile_memory,
        with_stack=args.with_stack,
        with_flops=True,
        with_modules=True
    ) as prof:
        
        total_steps = 1 + 1 + args.num_steps + 1  # wait + warmup + active + buffer
        
        for step in range(total_steps):
            with record_function(f"training_step_{step}"):
                # Get batch
                with record_function("data_loading"):
                    inputs = trainer.get_batch()
                
                # ZO step (gradient estimation)
                with record_function("zo_gradient_estimation"):
                    loss, projected_grad, zo_random_seed = trainer.zo_step(inputs)
                
                # ZO update
                with record_function("zo_parameter_update"):
                    trainer.zo_update(projected_grad, zo_random_seed)
                
                if step >= 2:  # Skip wait and warmup
                    logger.info(f"Step {step-1}: loss = {loss.item():.4f}")
                
                # Track memory stats
                if torch.cuda.is_available():
                    memory_stats['peak_allocated'].append(torch.cuda.max_memory_allocated() / 1024**2)
                    memory_stats['peak_reserved'].append(torch.cuda.max_memory_reserved() / 1024**2)
            
            prof.step()
    
    # Export memory snapshot if requested
    if args.export_memory_snapshot and torch.cuda.is_available():
        snapshot_path = os.path.join(output_subdir, f"{model_short_name}_memory_snapshot.pickle")
        try:
            torch.cuda.memory._dump_snapshot(snapshot_path)
            logger.info(f"Memory snapshot saved to: {snapshot_path}")
            logger.info("Visualize at: https://pytorch.org/memory_viz")
        except Exception as e:
            logger.warning(f"Failed to dump memory snapshot: {e}")
    
    # Stop memory history recording
    if args.memory_timeline:
        torch.cuda.memory._record_memory_history(enabled=None)
        logger.info("Memory history recording stopped")
    
    # Export Chrome trace explicitly for visualization
    chrome_trace_path = os.path.join(output_subdir, args.get_trace_filename())
    try:
        prof.export_chrome_trace(chrome_trace_path)
        logger.info(f"Chrome trace exported to: {chrome_trace_path}")
    except RuntimeError:
        # tensorboard_trace_handler may have already saved, find those files
        import glob
        trace_files = glob.glob(os.path.join(output_subdir, "*.pt.trace.json"))
        if trace_files:
            chrome_trace_path = trace_files[-1]
            logger.info(f"Chrome trace available at: {chrome_trace_path}")
    
    # Also export stacks if available (useful for flame graphs)
    stacks_path = os.path.join(output_subdir, f"{model_short_name}_stacks.txt")
    try:
        prof.export_stacks(stacks_path, "self_cuda_time_total")
        logger.info(f"CUDA stacks exported to: {stacks_path} (can be used with flamegraph.pl)")
    except Exception:
        pass  # Stacks may not be available without with_stack=True
    
    # Print summary statistics
    logger.info("\n" + "="*80)
    logger.info("PROFILING SUMMARY")
    logger.info("="*80)
    
    # CPU time summary
    logger.info("\n--- CPU Time (sorted by total time) ---")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
    
    # CUDA time summary
    logger.info("\n--- CUDA Time (sorted by total time) ---")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    
    # Memory summary (if enabled)
    if args.profile_memory:
        logger.info("\n--- Memory Usage (sorted by self CUDA memory) ---")
        print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=20))
    
    # Save text summary
    summary_path = os.path.join(output_subdir, f"{model_short_name}_profiler_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"DiZO PROFILING SUMMARY - {args.model_name}\n")
        f.write("="*80 + "\n\n")
        f.write(f"Model: {args.model_name}\n")
        f.write(f"Task: {args.task_name}\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Steps: {args.num_steps}\n")
        f.write("\n")
        
        f.write("--- CPU Time (sorted by total time) ---\n")
        f.write(prof.key_averages().table(sort_by="cpu_time_total", row_limit=50))
        f.write("\n\n")
        
        f.write("--- CUDA Time (sorted by total time) ---\n")
        f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=50))
        f.write("\n\n")
        
        if args.profile_memory:
            f.write("--- Memory Usage ---\n")
            f.write(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=50))
    
    logger.info(f"Summary saved to: {summary_path}")
    
    # Print memory statistics
    if torch.cuda.is_available() and memory_stats['peak_allocated']:
        logger.info("\n" + "="*80)
        logger.info("MEMORY STATISTICS")
        logger.info("="*80)
        logger.info(f"Peak Allocated Memory: {max(memory_stats['peak_allocated']):.2f} MB")
        logger.info(f"Peak Reserved Memory: {max(memory_stats['peak_reserved']):.2f} MB")
        logger.info(f"Avg Allocated per Step: {np.mean(memory_stats['peak_allocated']):.2f} MB")
        
        # Save memory timeline plot
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(12, 4))
            steps = range(len(memory_stats['peak_allocated']))
            ax.plot(steps, memory_stats['peak_allocated'], 'b-', label='Allocated', linewidth=2)
            ax.plot(steps, memory_stats['peak_reserved'], 'r--', label='Reserved', linewidth=2)
            ax.set_xlabel('Step')
            ax.set_ylabel('Memory (MB)')
            ax.set_title('CUDA Memory Usage Over Training Steps')
            ax.legend()
            ax.grid(True, alpha=0.3)
            memory_plot_path = os.path.join(output_subdir, f"{model_short_name}_memory_timeline.png")
            plt.savefig(memory_plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"Memory timeline plot saved to: {memory_plot_path}")
        except Exception as e:
            logger.warning(f"Failed to create memory plot: {e}")
    
    logger.info("\n" + "="*80)
    logger.info("VISUALIZATION OPTIONS:")
    logger.info("="*80)
    logger.info(f"1. Chrome Trace Viewer: Open chrome://tracing and load .pt.trace.json files in {output_subdir}")
    logger.info(f"2. TensorBoard: tensorboard --logdir={output_subdir}")
    logger.info(f"3. Perfetto (online): https://ui.perfetto.dev/ - drag and drop the .pt.trace.json file")
    logger.info("="*80)
    
    # Print flat buffer timing stats if applicable
    if args.flat_buffer and hasattr(trainer, 'print_timing_stats'):
        trainer.print_timing_stats()
    
    return output_subdir


def main():
    parser = argparse.ArgumentParser(description="Profile DiZO training")
    parser.add_argument("--model_name", type=str, default="facebook/opt-350m",
                       help="Model name (default: facebook/opt-350m for quick profiling)")
    parser.add_argument("--task_name", type=str, default="SST2",
                       help="Task name (default: SST2)")
    parser.add_argument("--num_steps", type=int, default=5,
                       help="Number of training steps to profile (default: 5)")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size (default: 4)")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Max sequence length (default: 512)")
    parser.add_argument("--zo_eps", type=float, default=1e-3,
                       help="ZO epsilon (default: 1e-3)")
    parser.add_argument("--learning_rate", type=float, default=1e-7,
                       help="Learning rate (default: 1e-7)")
    parser.add_argument("--load_float16", action="store_true", default=True,
                       help="Load model in float16")
    parser.add_argument("--output_dir", type=str, default="./profiler_logs",
                       help="Output directory for profiling results")
    parser.add_argument("--profile_memory", action="store_true", default=True,
                       help="Profile memory usage")
    parser.add_argument("--with_stack", action="store_true", default=False,
                       help="Include Python stack traces (slower but more detailed)")
    parser.add_argument("--torch_compile", action="store_true", default=False,
                       help="Use torch.compile() on the model for potential speedup")
    parser.add_argument("--compile_mode", type=str, default="default",
                       choices=["default", "reduce-overhead", "max-autotune"],
                       help="torch.compile mode (default: default)")
    parser.add_argument("--compile_warmup_steps", type=int, default=3,
                       help="Number of warmup steps to complete torch.compile before profiling (default: 3)")
    parser.add_argument("--memory_timeline", action="store_true", default=False,
                       help="Record detailed CUDA memory allocation timeline")
    parser.add_argument("--export_memory_snapshot", action="store_true", default=False,
                       help="Export memory snapshot for visualization at pytorch.org/memory_viz")
    # LoRA arguments
    parser.add_argument("--use_lora", action="store_true", default=False,
                       help="Use LoRA for parameter-efficient ZO training")
    parser.add_argument("--lora_r", type=int, default=8,
                       help="LoRA rank (default: 8)")
    parser.add_argument("--lora_alpha", type=int, default=16,
                       help="LoRA alpha scaling factor (default: 16)")
    # Flat buffer equivalence experiment arguments
    parser.add_argument("--flat_buffer", action="store_true", default=False,
                       help="Use flat buffer method for ZO perturbation/update (tests PyTorch equivalent)")
    parser.add_argument("--flat_buffer_optimized", action="store_true", default=False,
                       help="Use optimized single-RNG flat buffer (faster but different from baseline)")
    parser.add_argument("--equivalence_test", action="store_true", default=False,
                       help="Run equivalence test comparing baseline vs flat buffer implementations")
    parser.add_argument("--equivalence_steps", type=int, default=50,
                       help="Number of steps for equivalence test (default: 50)")
    parser.add_argument("--plot_loss_curve", action="store_true", default=True,
                       help="Plot loss curves during equivalence test")
    
    args_parsed = parser.parse_args()
    
    # Convert to ProfilerArguments
    args = ProfilerArguments(
        model_name=args_parsed.model_name,
        task_name=args_parsed.task_name,
        num_steps=args_parsed.num_steps,
        batch_size=args_parsed.batch_size,
        max_length=args_parsed.max_length,
        zo_eps=args_parsed.zo_eps,
        learning_rate=args_parsed.learning_rate,
        load_float16=args_parsed.load_float16,
        output_dir=args_parsed.output_dir,
        profile_memory=args_parsed.profile_memory,
        with_stack=args_parsed.with_stack,
        torch_compile=args_parsed.torch_compile,
        compile_mode=args_parsed.compile_mode,
        compile_warmup_steps=args_parsed.compile_warmup_steps,
        memory_timeline=args_parsed.memory_timeline,
        export_memory_snapshot=args_parsed.export_memory_snapshot,
        use_lora=args_parsed.use_lora,
        lora_r=args_parsed.lora_r,
        lora_alpha=args_parsed.lora_alpha,
        flat_buffer=args_parsed.flat_buffer,
        flat_buffer_optimized=args_parsed.flat_buffer_optimized,
        equivalence_test=args_parsed.equivalence_test,
        equivalence_steps=args_parsed.equivalence_steps,
        plot_loss_curve=args_parsed.plot_loss_curve
    )
    
    run_profiling(args)


if __name__ == "__main__":
    main()
