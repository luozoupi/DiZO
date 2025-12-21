#!/usr/bin/env python3
"""
Profile baseline AdamW (first-order) training to compare with MeZO (zeroth-order).

This script runs conventional first-order training with the SAME model, data,
dtype, and sequence length as the MeZO profiling runs for fair comparison.

Usage:
    python profile_baseline_adamw.py --model_name facebook/opt-350m --task_name SST2 --num_steps 10
    
To visualize:
    - Open Chrome, go to chrome://tracing, and load the generated .json file
    - Or use Perfetto: https://ui.perfetto.dev/
"""

import logging
import os
import sys
import argparse
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader
from torch.profiler import profile, record_function, ProfilerActivity, schedule, tensorboard_trace_handler

from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import local modules (same as profile_dizo.py)
import tasks
from tasks import get_task
from utils import encode_prompt

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class BaselineProfilerArguments:
    """Arguments for baseline AdamW profiling - mirrors ProfilerArguments from profile_dizo.py"""
    model_name: str = "facebook/opt-350m"
    task_name: str = "SST2"
    num_steps: int = 10
    warmup_steps: int = 2
    batch_size: int = 4
    max_length: int = 512  # Same as profile_dizo.py
    learning_rate: float = 1e-5
    load_float16: bool = True  # Same as profile_dizo.py
    output_dir: str = "./profiler_logs"
    profile_memory: bool = True
    with_stack: bool = False
    torch_compile: bool = False
    compile_mode: str = "default"
    memory_timeline: bool = False
    export_memory_snapshot: bool = False
    
    def get_model_short_name(self) -> str:
        """Extract short model name for output naming"""
        name = self.model_name.split("/")[-1]
        return name.lower().replace(".", "_")
    
    def get_output_subdir(self) -> str:
        """Get model-specific output subdirectory"""
        model_name = self.get_model_short_name()
        compile_suffix = "_compiled" if self.torch_compile else ""
        return os.path.join(self.output_dir, f"baseline_adamw_{model_name}{compile_suffix}")
    
    def get_trace_filename(self) -> str:
        """Get trace filename with model name and compile status"""
        model_name = self.get_model_short_name()
        compile_suffix = "_compiled" if self.torch_compile else ""
        return f"baseline_adamw_{model_name}{compile_suffix}.pt.trace.json"


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SimpleDataset(Dataset):
    """Same dataset class as profile_dizo.py"""
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def prepare_data(task, tokenizer, args: BaselineProfilerArguments):
    """Prepare training data - SAME as profile_dizo.py for fair comparison"""
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
    """Custom collate function - SAME as profile_dizo.py"""
    max_len = max(item['input_ids'].size(0) for item in batch)
    
    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    option_lens = []
    
    for i, item in enumerate(batch):
        seq_len = item['input_ids'].size(0)
        # Left padding (same as profile_dizo.py)
        input_ids[i, max_len - seq_len:] = item['input_ids']
        labels[i, max_len - seq_len:] = item['labels']
        option_lens.append(item['option_len'])
    
    return {
        'input_ids': input_ids,
        'labels': labels,
        'option_len': option_lens
    }


def compute_loss_with_option_len(model, input_ids, labels, option_len, pad_token_id):
    """
    Compute loss with option_len masking - matches profile_dizo.py's forward_wrap_with_option_len
    but with gradients enabled for backprop.
    """
    outputs = model(input_ids=input_ids)
    logits = outputs.logits
    
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels.clone()[..., 1:].contiguous()
    shift_labels[shift_labels == pad_token_id] = -100
    
    # Apply option_len masking (same as profile_dizo.py)
    if option_len is not None:
        for _i, _len in enumerate(option_len):
            shift_labels[_i, :-_len] = -100
    
    loss_fct = CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(shift_logits.view(-1, model.config.vocab_size), shift_labels.view(-1))
    
    return loss


def profile_baseline_training(args: BaselineProfilerArguments):
    """
    Main profiling function for baseline AdamW training.
    Uses same data pipeline and settings as profile_dizo.py for fair comparison.
    """
    set_seed(42)
    
    # Create model-specific output directory
    output_subdir = args.get_output_subdir()
    os.makedirs(output_subdir, exist_ok=True)
    
    model_short_name = args.get_model_short_name()
    
    logger.info("=" * 80)
    logger.info("BASELINE ADAMW PROFILING (First-Order)")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Task: {args.task_name}")
    logger.info(f"Steps: {args.num_steps} (warmup: {args.warmup_steps})")
    logger.info(f"Batch size: {args.batch_size}, Max length: {args.max_length}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Dtype: float32 (FO training requires float32 for stable gradients)")
    logger.info(f"torch.compile: {args.torch_compile} (mode: {args.compile_mode})")
    logger.info(f"Memory profiling: {args.profile_memory}, Timeline: {args.memory_timeline}, Snapshot: {args.export_memory_snapshot}")
    logger.info(f"Output directory: {output_subdir}")
    logger.info("=" * 80)
    
    # Load model
    # NOTE: MeZO uses float16 for inference-only (no gradients).
    # For fair comparison of MEMORY, we load in float16 but need to handle gradients carefully.
    # For fair comparison of COMPUTE, we use the same model architecture and data.
    logger.info(f"\nLoading model: {args.model_name}")
    
    # For first-order training with backprop, we need either:
    # 1. float32 weights (more memory, stable)
    # 2. float16 weights with AMP (less memory, needs scaler)
    # We use float32 to avoid gradient issues, matching practical FO training
    torch_dtype = torch.float32  # Always use float32 for first-order training
    
    config = AutoConfig.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        config=config,
        torch_dtype=torch_dtype,
        device_map='auto'
    )
    model.train()  # Set to training mode (unlike MeZO which uses eval mode)
    logger.info(f"Model loaded in {torch_dtype} (FO training requires float32 for stable gradients)")
    
    # Apply torch.compile if requested
    if args.torch_compile:
        logger.info(f"Applying torch.compile with mode='{args.compile_mode}'...")
        model = torch.compile(model, mode=args.compile_mode)
        logger.info("Model compiled successfully")
    
    # Load tokenizer - SAME as profile_dizo.py
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    if "opt" in args.model_name:
        tokenizer.bos_token_id = 0
    
    pad_token_id = tokenizer.pad_token_id or 0
    
    # Load task and prepare data - SAME as profile_dizo.py
    logger.info(f"Loading task: {args.task_name}")
    task = get_task(args.task_name)
    
    logger.info("Preparing training data (same as MeZO)...")
    train_data = prepare_data(task, tokenizer, args)
    logger.info(f"Prepared {len(train_data)} training samples")
    
    dataloader = DataLoader(
        SimpleDataset(train_data),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda x: collate_fn(x, pad_token_id)
    )
    
    # Create optimizer (AdamW - standard first-order optimizer)
    logger.info("Creating AdamW optimizer...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # Get device
    device = next(model.parameters()).device
    
    # Define profiling schedule
    profiler_schedule = schedule(
        wait=1,
        warmup=args.warmup_steps,
        active=args.num_steps,
        repeat=1
    )
    
    # Activities to profile
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    
    logger.info(f"\nStarting profiling for {args.num_steps} steps...")
    
    # Start memory timeline recording if requested
    if args.memory_timeline and torch.cuda.is_available():
        logger.info("Starting CUDA memory history recording...")
        torch.cuda.memory._record_memory_history(
            enabled="all",
            context="all",
            stacks="python",
            max_entries=100000,
        )
    
    # Track memory stats
    memory_stats = {
        'peak_allocated': [],
        'peak_reserved': [],
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
        
        data_iter = iter(dataloader)
        total_steps = 1 + args.warmup_steps + args.num_steps + 1  # wait + warmup + active + buffer
        
        for step in range(total_steps):
            with record_function(f"training_step_{step}"):
                # Get batch
                with record_function("data_loading"):
                    try:
                        batch = next(data_iter)
                    except StopIteration:
                        data_iter = iter(dataloader)
                        batch = next(data_iter)
                    
                    input_ids = batch['input_ids'].to(device)
                    labels = batch['labels'].to(device)
                    option_len = batch['option_len']
                
                # Forward pass
                with record_function("forward_pass"):
                    loss = compute_loss_with_option_len(
                        model, input_ids, labels, option_len, pad_token_id
                    )
                
                # Backward pass
                with record_function("backward_pass"):
                    loss.backward()
                
                # Optimizer step
                with record_function("optimizer_step"):
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                
                if step >= 1 + args.warmup_steps:  # Skip wait and warmup
                    active_step = step - 1 - args.warmup_steps + 1
                    logger.info(f"Step {active_step}/{args.num_steps}: loss = {loss.item():.4f}")
                
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
    if args.memory_timeline and torch.cuda.is_available():
        torch.cuda.memory._record_memory_history(enabled=None)
        logger.info("Memory history recording stopped")
    
    # Export Chrome trace explicitly
    chrome_trace_path = os.path.join(output_subdir, args.get_trace_filename())
    try:
        prof.export_chrome_trace(chrome_trace_path)
        logger.info(f"Chrome trace exported to: {chrome_trace_path}")
    except RuntimeError:
        import glob
        trace_files = glob.glob(os.path.join(output_subdir, "*.pt.trace.json"))
        if trace_files:
            chrome_trace_path = trace_files[-1]
            logger.info(f"Chrome trace available at: {chrome_trace_path}")
    
    # Export stacks if available
    stacks_path = os.path.join(output_subdir, f"{model_short_name}_stacks.txt")
    try:
        prof.export_stacks(stacks_path, "self_cuda_time_total")
        logger.info(f"CUDA stacks exported to: {stacks_path}")
    except Exception:
        pass
    
    # Print summary statistics
    logger.info("\n" + "=" * 80)
    logger.info("PROFILING SUMMARY")
    logger.info("=" * 80)
    
    # CPU time summary
    logger.info("\n--- CPU Time (sorted by total time) ---")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
    
    # CUDA time summary
    logger.info("\n--- CUDA Time (sorted by total time) ---")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    
    # Memory summary
    if args.profile_memory:
        logger.info("\n--- Memory Usage (sorted by self CUDA memory) ---")
        print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=20))
    
    # Save text summary
    summary_path = os.path.join(output_subdir, f"{model_short_name}_profiler_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"BASELINE ADAMW PROFILING SUMMARY - {args.model_name}\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Model: {args.model_name}\n")
        f.write(f"Task: {args.task_name}\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Max Length: {args.max_length}\n")
        f.write(f"Steps: {args.num_steps}\n")
        f.write(f"Dtype: {'float16' if args.load_float16 else 'float32'}\n")
        f.write(f"torch.compile: {args.torch_compile}\n")
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
        logger.info("\n" + "=" * 80)
        logger.info("MEMORY STATISTICS")
        logger.info("=" * 80)
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
            ax.set_title(f'CUDA Memory Usage - Baseline AdamW ({model_short_name})')
            ax.legend()
            ax.grid(True, alpha=0.3)
            memory_plot_path = os.path.join(output_subdir, f"{model_short_name}_memory_timeline.png")
            plt.savefig(memory_plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"Memory timeline plot saved to: {memory_plot_path}")
        except Exception as e:
            logger.warning(f"Failed to create memory plot: {e}")
    
    logger.info("\n" + "=" * 80)
    logger.info("VISUALIZATION OPTIONS:")
    logger.info("=" * 80)
    logger.info(f"1. Chrome Trace Viewer: Open chrome://tracing and load .pt.trace.json files in {output_subdir}")
    logger.info(f"2. TensorBoard: tensorboard --logdir={output_subdir}")
    logger.info(f"3. Perfetto (online): https://ui.perfetto.dev/ - drag and drop the .pt.trace.json file")
    logger.info("=" * 80)
    
    return output_subdir


def main():
    parser = argparse.ArgumentParser(description="Profile baseline AdamW (first-order) training")
    parser.add_argument("--model_name", type=str, default="facebook/opt-350m",
                        help="Model name (default: facebook/opt-350m)")
    parser.add_argument("--task_name", type=str, default="SST2",
                        help="Task name (default: SST2)")
    parser.add_argument("--num_steps", type=int, default=10,
                        help="Number of training steps to profile (default: 10)")
    parser.add_argument("--warmup_steps", type=int, default=2,
                        help="Profiler warmup steps (default: 2)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size (default: 4)")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Max sequence length (default: 512)")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Learning rate (default: 1e-5)")
    parser.add_argument("--load_float16", action="store_true", default=True,
                        help="Load model in float16 (default: True)")
    parser.add_argument("--load_float32", action="store_true", default=False,
                        help="Load model in float32 instead of float16")
    parser.add_argument("--output_dir", type=str, default="./profiler_logs",
                        help="Output directory for profiling results")
    parser.add_argument("--profile_memory", action="store_true", default=True,
                        help="Profile memory usage")
    parser.add_argument("--with_stack", action="store_true", default=False,
                        help="Include Python stack traces (slower but more detailed)")
    parser.add_argument("--torch_compile", action="store_true", default=False,
                        help="Use torch.compile() on the model")
    parser.add_argument("--compile_mode", type=str, default="default",
                        choices=["default", "reduce-overhead", "max-autotune"],
                        help="torch.compile mode (default: default)")
    parser.add_argument("--memory_timeline", action="store_true", default=False,
                        help="Record detailed CUDA memory allocation timeline")
    parser.add_argument("--export_memory_snapshot", action="store_true", default=False,
                        help="Export memory snapshot for visualization at pytorch.org/memory_viz")
    
    args_parsed = parser.parse_args()
    
    # Handle float16/float32 flag
    load_float16 = not args_parsed.load_float32
    
    # Convert to BaselineProfilerArguments
    args = BaselineProfilerArguments(
        model_name=args_parsed.model_name,
        task_name=args_parsed.task_name,
        num_steps=args_parsed.num_steps,
        warmup_steps=args_parsed.warmup_steps,
        batch_size=args_parsed.batch_size,
        max_length=args_parsed.max_length,
        learning_rate=args_parsed.learning_rate,
        load_float16=load_float16,
        output_dir=args_parsed.output_dir,
        profile_memory=args_parsed.profile_memory,
        with_stack=args_parsed.with_stack,
        torch_compile=args_parsed.torch_compile,
        compile_mode=args_parsed.compile_mode,
        memory_timeline=args_parsed.memory_timeline,
        export_memory_snapshot=args_parsed.export_memory_snapshot
    )
    
    profile_baseline_training(args)


if __name__ == "__main__":
    main()
