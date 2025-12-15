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
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
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
    trainer = MeZOTrainer(model, tokenizer, args, dataloader)
    
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
    logger.info(f"Output directory: {args.output_dir}")
    
    # PyTorch Profiler
    with profile(
        activities=activities,
        schedule=profiler_schedule,
        on_trace_ready=tensorboard_trace_handler(args.output_dir),
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
            
            prof.step()
    
    # Export Chrome trace explicitly for visualization
    chrome_trace_path = os.path.join(args.output_dir, "dizo_chrome_trace.json")
    try:
        prof.export_chrome_trace(chrome_trace_path)
        logger.info(f"Chrome trace exported to: {chrome_trace_path}")
    except RuntimeError:
        # tensorboard_trace_handler may have already saved, find those files
        import glob
        trace_files = glob.glob(os.path.join(args.output_dir, "*.pt.trace.json"))
        if trace_files:
            chrome_trace_path = trace_files[-1]
            logger.info(f"Chrome trace available at: {chrome_trace_path}")
    
    # Also export stacks if available (useful for flame graphs)
    stacks_path = os.path.join(args.output_dir, "dizo_stacks.txt")
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
    summary_path = os.path.join(args.output_dir, "profiler_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("DiZO PROFILING SUMMARY\n")
        f.write("="*80 + "\n\n")
        
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
    
    logger.info("\n" + "="*80)
    logger.info("VISUALIZATION OPTIONS:")
    logger.info("="*80)
    logger.info(f"1. Chrome Trace Viewer: Open chrome://tracing and load .pt.trace.json files in {args.output_dir}")
    logger.info(f"2. TensorBoard: tensorboard --logdir={args.output_dir}")
    logger.info(f"3. Perfetto (online): https://ui.perfetto.dev/ - drag and drop the .pt.trace.json file")
    logger.info("="*80)


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
        with_stack=args_parsed.with_stack
    )
    
    run_profiling(args)


if __name__ == "__main__":
    main()
