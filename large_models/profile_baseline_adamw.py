#!/usr/bin/env python3
"""
Profile baseline AdamW training to compare with MeZO.

This script runs conventional first-order training with the same model,
data, and number of steps as the MeZO profiling runs, then exports a
Chrome trace for analysis with analyze_timeline.py
"""

import os
import torch
from torch.profiler import profile, ProfilerActivity, schedule
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
import argparse


def create_dummy_dataloader(model_name, batch_size=4, seq_length=128):
    """Create a minimal synthetic dataloader to avoid extra deps (tasks/run)."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    vocab_size = tokenizer.vocab_size
    dataset_samples = []
    for _ in range(batch_size * 5):  # small pool, recycled across steps
        input_ids = torch.randint(0, vocab_size, (seq_length,))
        dataset_samples.append({
            'input_ids': input_ids,
            'labels': input_ids.clone()
        })

    def collate_fn(batch):
        input_ids = torch.stack([item['input_ids'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        return {'input_ids': input_ids, 'labels': labels}

    return DataLoader(dataset_samples, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)


def profile_baseline_training(
    model_name="facebook/opt-350m",
    num_steps=20,
    warmup_steps=5,
    batch_size=4,
    seq_length=128,
    lr=1e-5,
    output_trace="profiler_logs/baseline_adamw.pt.trace.json",
    device="cuda"
):
    """
    Run baseline AdamW training with profiling enabled.
    
    Args:
        model_name: HuggingFace model identifier
        task_name: Task name (SST2, RTE, etc.)
        num_steps: Number of training steps to profile
        warmup_steps: Profiler warmup steps
        batch_size: Batch size
        seq_length: Sequence length
        lr: Learning rate
        output_trace: Output path for Chrome trace JSON
        device: Device to use
    """
    
    print(f"=" * 80)
    print(f"BASELINE ADAMW PROFILING")
    print(f"=" * 80)
    print(f"Model: {model_name}")
    print(f"Steps: {num_steps} (warmup: {warmup_steps})")
    print(f"Batch size: {batch_size}, Seq length: {seq_length}")
    print(f"Learning rate: {lr}")
    print(f"Output: {output_trace}")
    print(f"=" * 80)
    
    # Load model
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map=None,
    )
    model = model.to(device)
    model.train()
    
    # Create dataloader
    print("Creating dataloader...")
    dataloader = create_dummy_dataloader(model_name, batch_size, seq_length)
    
    # Create optimizer (AdamW like in regular training)
    print("Creating AdamW optimizer...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    # Setup profiler
    print(f"\nStarting profiling for {num_steps} steps...")
    
    # Profile schedule: skip warmup, then record
    prof_schedule = schedule(
        wait=0,
        warmup=warmup_steps,
        active=num_steps - warmup_steps,
        repeat=1
    )
    
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=prof_schedule,
        record_shapes=False,
        with_stack=False,
        profile_memory=False,
        on_trace_ready=lambda p: p.export_chrome_trace(output_trace),
    ) as prof:
        
        data_iter = iter(dataloader)
        
        for step in range(num_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)
            
            # Move to device
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            
            # Optimizer step
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
            # Profiler step
            prof.step()
            
            if (step + 1) % 5 == 0:
                print(f"  Step {step + 1}/{num_steps}, Loss: {loss.item():.4f}")
    
    print(f"\n✓ Profiling complete!")
    print(f"✓ Trace saved to: {output_trace}")
    print(f"\nAnalyze with:")
    print(f"  python analyze_timeline.py {output_trace}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Profile baseline AdamW training")
    parser.add_argument("--model_name", type=str, default="facebook/opt-350m",
                        help="HuggingFace model name")
    parser.add_argument("--num_steps", type=int, default=20,
                        help="Number of profiling steps")
    parser.add_argument("--warmup_steps", type=int, default=5,
                        help="Profiler warmup steps")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size")
    parser.add_argument("--seq_length", type=int, default=128,
                        help="Sequence length")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--output_trace", type=str, default="profiler_logs/baseline_adamw.pt.trace.json",
                        help="Output trace file path")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_trace), exist_ok=True)
    
    profile_baseline_training(
        model_name=args.model_name,
        num_steps=args.num_steps,
        warmup_steps=args.warmup_steps,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        lr=args.lr,
        output_trace=args.output_trace,
        device=args.device
    )


if __name__ == "__main__":
    main()
