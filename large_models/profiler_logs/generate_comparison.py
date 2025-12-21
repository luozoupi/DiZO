#!/usr/bin/env python3
import os
import re

models = ['opt-350m', 'opt-2_7b', 'opt-6_7b', 'opt-13b']
model_params = {'opt-350m': '331M', 'opt-2_7b': '2.7B', 'opt-6_7b': '6.7B', 'opt-13b': '13B'}

results = []

for model in models:
    summary_file = f"profile_{model}/{model}_profiler_summary.txt"
    if os.path.exists(summary_file):
        with open(summary_file, 'r') as f:
            content = f.read()
        
        # Extract Self CUDA time total
        cuda_match = re.search(r'Self CUDA time total: ([\d.]+)([a-z]+)', content)
        cuda_time = f"{cuda_match.group(1)}{cuda_match.group(2)}" if cuda_match else "N/A"
        
        # Extract Self CPU time total
        cpu_match = re.search(r'Self CPU time total: ([\d.]+)([a-z]+)', content)
        cpu_time = f"{cpu_match.group(1)}{cpu_match.group(2)}" if cpu_match else "N/A"
        
        # Parse the time values for scaling analysis
        if cuda_match:
            val = float(cuda_match.group(1))
            unit = cuda_match.group(2)
            if unit == 'ms':
                cuda_ms = val
            elif unit == 's':
                cuda_ms = val * 1000
            else:
                cuda_ms = 0
        else:
            cuda_ms = 0
            
        results.append({
            'model': model,
            'params': model_params[model],
            'cuda_time': cuda_time,
            'cpu_time': cpu_time,
            'cuda_ms': cuda_ms
        })

# Calculate scaling factors
if results and results[0]['cuda_ms'] > 0:
    base_time = results[0]['cuda_ms']
    for r in results:
        r['scale'] = r['cuda_ms'] / base_time if r['cuda_ms'] > 0 else 0

# Print comparison table
print("=" * 100)
print("DiZO MULTI-MODEL PROFILING COMPARISON")
print("=" * 100)
print()
print(f"{'Model':<20} {'Parameters':<12} {'CUDA Time':<15} {'CPU Time':<15} {'Scale vs 350M':<15}")
print("-" * 80)
for r in results:
    scale_str = f"{r.get('scale', 0):.2f}x"
    print(f"{r['model']:<20} {r['params']:<12} {r['cuda_time']:<15} {r['cpu_time']:<15} {scale_str:<15}")

print()
print("=" * 100)
print("KEY OBSERVATIONS")
print("=" * 100)
print()

# Calculate approximate time per parameter
print("Time per Billion Parameters (normalized):")
print("-" * 50)
param_map = {'350M': 0.331, '2.7B': 2.7, '6.7B': 6.7, '13B': 13}
for r in results:
    params_b = param_map.get(r['params'], 1)
    time_per_b = r['cuda_ms'] / params_b
    print(f"  {r['model']}: {time_per_b:.1f} ms/B params")

print()
print("Key ZO Operations Scaling:")
print("-" * 50)

# Extract specific operation times for each model
for model in models:
    summary_file = f"profile_{model}/{model}_profiler_summary.txt"
    if os.path.exists(summary_file):
        with open(summary_file, 'r') as f:
            content = f.read()
        print(f"\n{model}:")
        # Find zo operations
        zo_ops = re.findall(r'(zo_\w+)\s+[\d.]+%\s+[\d.]+[a-z]+\s+[\d.]+%\s+[\d.]+[a-z]+\s+[\d.]+[a-z]+\s+[\d.]+[a-z]+\s+[\d.]+%\s+([\d.]+)([a-z]+)', content)
        seen = set()
        for op, time_val, unit in zo_ops:
            if op not in seen:
                print(f"  {op}: {time_val}{unit}")
                seen.add(op)

print()
print("=" * 100)
