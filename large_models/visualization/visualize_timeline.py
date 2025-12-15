#!/usr/bin/env python3
"""
Visualize CPU/GPU timeline from Chrome trace files.
Creates timeline plots showing the interleaving pattern.
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict


def load_trace(filepath):
    """Load Chrome trace JSON file"""
    print(f"Loading trace file: {filepath}")
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def parse_events(trace_data):
    """Parse and categorize events"""
    events = trace_data.get('traceEvents', trace_data)
    
    cpu_events = []
    gpu_events = []
    mezo_events = []
    
    for event in events:
        if not isinstance(event, dict):
            continue
            
        name = event.get('name', '')
        cat = event.get('cat', '')
        ph = event.get('ph', '')
        ts = event.get('ts', 0)
        dur = event.get('dur', 0)
        
        if ph in ['M', 's', 't', 'f'] or dur <= 0:
            continue
            
        event_info = {
            'name': name,
            'cat': cat,
            'ts': ts,
            'dur': dur,
            'end': ts + dur
        }
        
        # Categorize
        if 'zo_' in name:
            mezo_events.append(event_info)
        elif 'cuda' in cat.lower() or 'kernel' in cat.lower():
            gpu_events.append(event_info)
        elif cat == 'cpu_op' or 'aten::' in name:
            cpu_events.append(event_info)
    
    return cpu_events, gpu_events, mezo_events


def create_timeline_visualization(trace_file, output_prefix='timeline'):
    """Create timeline visualization plots"""
    
    trace_data = load_trace(trace_file)
    cpu_events, gpu_events, mezo_events = parse_events(trace_data)
    
    if not gpu_events:
        print("No GPU events found!")
        return
    
    # Get time range
    all_events = cpu_events + gpu_events
    min_ts = min(e['ts'] for e in all_events)
    max_ts = max(e['end'] for e in all_events)
    
    # Normalize timestamps to start from 0, convert to ms
    for events in [cpu_events, gpu_events, mezo_events]:
        for e in events:
            e['ts_ms'] = (e['ts'] - min_ts) / 1000
            e['dur_ms'] = e['dur'] / 1000
            e['end_ms'] = (e['end'] - min_ts) / 1000
    
    total_time_ms = (max_ts - min_ts) / 1000
    
    # =========================================================================
    # Plot 1: CPU/GPU Timeline (zoomed to first 50ms for visibility)
    # =========================================================================
    fig, axes = plt.subplots(3, 1, figsize=(16, 10))
    
    # Subplot 1: First 50ms detailed view
    ax1 = axes[0]
    window_ms = min(50, total_time_ms)
    
    # Draw GPU events
    for e in gpu_events:
        if e['ts_ms'] < window_ms:
            ax1.barh(y=0.5, width=e['dur_ms'], left=e['ts_ms'], 
                    height=0.3, color='green', alpha=0.7, edgecolor='darkgreen', linewidth=0.5)
    
    # Draw CPU events (only aten:: ops for clarity)
    cpu_in_window = [e for e in cpu_events if e['ts_ms'] < window_ms and 'aten::' in e['name']]
    for e in cpu_in_window[:500]:  # Limit for performance
        ax1.barh(y=1.0, width=e['dur_ms'], left=e['ts_ms'],
                height=0.3, color='blue', alpha=0.5, edgecolor='darkblue', linewidth=0.5)
    
    ax1.set_xlim(0, window_ms)
    ax1.set_ylim(0, 1.5)
    ax1.set_yticks([0.5, 1.0])
    ax1.set_yticklabels(['GPU', 'CPU'])
    ax1.set_xlabel('Time (ms)')
    ax1.set_title(f'CPU/GPU Timeline (First {window_ms:.0f}ms) - Notice GPU Gaps!')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Add legend
    gpu_patch = mpatches.Patch(color='green', alpha=0.7, label='GPU Kernels')
    cpu_patch = mpatches.Patch(color='blue', alpha=0.5, label='CPU Ops (aten::)')
    ax1.legend(handles=[gpu_patch, cpu_patch], loc='upper right')
    
    # Subplot 2: GPU idle pattern analysis
    ax2 = axes[1]
    
    # Calculate GPU busy/idle over time bins
    bin_size_ms = 1.0  # 1ms bins
    n_bins = int(total_time_ms / bin_size_ms) + 1
    gpu_busy = np.zeros(n_bins)
    
    for e in gpu_events:
        start_bin = int(e['ts_ms'] / bin_size_ms)
        end_bin = int(e['end_ms'] / bin_size_ms)
        for b in range(start_bin, min(end_bin + 1, n_bins)):
            # Calculate overlap with this bin
            bin_start = b * bin_size_ms
            bin_end = (b + 1) * bin_size_ms
            overlap_start = max(e['ts_ms'], bin_start)
            overlap_end = min(e['end_ms'], bin_end)
            if overlap_end > overlap_start:
                gpu_busy[b] += (overlap_end - overlap_start) / bin_size_ms * 100
    
    # Clip to 100%
    gpu_busy = np.clip(gpu_busy, 0, 100)
    
    bins = np.arange(n_bins) * bin_size_ms
    ax2.fill_between(bins, gpu_busy, alpha=0.7, color='green', label='GPU Utilization')
    ax2.axhline(y=np.mean(gpu_busy), color='red', linestyle='--', 
                label=f'Average: {np.mean(gpu_busy):.1f}%')
    ax2.set_xlim(0, total_time_ms)
    ax2.set_ylim(0, 100)
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('GPU Utilization %')
    ax2.set_title('GPU Utilization Over Time (1ms bins)')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # Subplot 3: MeZO phase timeline
    ax3 = axes[2]
    
    if mezo_events:
        # Color map for MeZO phases
        colors = {
            'zo_perturb_+eps': '#FF6B6B',
            'zo_perturb_-2eps': '#4ECDC4',
            'zo_perturb_+eps_reset': '#45B7D1',
            'zo_forward_1': '#96CEB4',
            'zo_forward_2': '#FFEAA7',
            'zo_update': '#DDA0DD',
            'zo_gradient_estimation': '#98D8C8',
            'zo_parameter_update': '#F7DC6F',
        }
        
        y_pos = 0
        for e in mezo_events:
            color = colors.get(e['name'], 'gray')
            ax3.barh(y=y_pos, width=e['dur_ms'], left=e['ts_ms'],
                    height=0.8, color=color, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        ax3.set_xlim(0, total_time_ms)
        ax3.set_ylim(-0.5, 1)
        ax3.set_yticks([0])
        ax3.set_yticklabels(['MeZO'])
        ax3.set_xlabel('Time (ms)')
        ax3.set_title('MeZO Operation Phases')
        
        # Create legend
        legend_patches = [mpatches.Patch(color=c, label=n, alpha=0.8) 
                         for n, c in colors.items()]
        ax3.legend(handles=legend_patches, loc='upper right', ncol=4, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_cpu_gpu.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_prefix}_cpu_gpu.png")
    plt.close()
    
    # =========================================================================
    # Plot 2: Gap Analysis
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Calculate gaps between GPU kernels
    sorted_gpu = sorted(gpu_events, key=lambda x: x['ts'])
    gaps = []
    for i in range(1, len(sorted_gpu)):
        gap = sorted_gpu[i]['ts_ms'] - sorted_gpu[i-1]['end_ms']
        if gap > 0.001:  # > 1μs
            gaps.append({
                'gap_ms': gap,
                'after': sorted_gpu[i-1]['name'],
                'before': sorted_gpu[i]['name']
            })
    
    gap_values = [g['gap_ms'] * 1000 for g in gaps]  # Convert to μs
    
    # Gap histogram
    ax = axes[0, 0]
    if gap_values:
        ax.hist(gap_values, bins=50, color='coral', edgecolor='black', alpha=0.7)
        ax.axvline(x=np.median(gap_values), color='red', linestyle='--', 
                   label=f'Median: {np.median(gap_values):.1f}μs')
        ax.set_xlabel('Gap Duration (μs)')
        ax.set_ylabel('Count')
        ax.set_title(f'Distribution of GPU Idle Gaps (n={len(gaps)})')
        ax.legend()
        ax.set_xlim(0, min(500, np.percentile(gap_values, 99)))
    
    # Gap cumulative
    ax = axes[0, 1]
    if gap_values:
        sorted_gaps = np.sort(gap_values)
        cumsum = np.cumsum(sorted_gaps)
        ax.plot(range(len(sorted_gaps)), cumsum / 1000, color='blue', linewidth=2)
        ax.set_xlabel('Gap Index (sorted by size)')
        ax.set_ylabel('Cumulative Gap Time (ms)')
        ax.set_title('Cumulative GPU Idle Time')
        ax.grid(True, alpha=0.3)
        
        # Mark 80% point
        idx_80 = np.searchsorted(cumsum, 0.8 * cumsum[-1])
        ax.axvline(x=idx_80, color='red', linestyle='--', alpha=0.7)
        ax.text(idx_80, cumsum[-1]/2000, f'  80% from top {len(gaps)-idx_80} gaps', fontsize=10)
    
    # Kernel count by type
    ax = axes[1, 0]
    kernel_counts = defaultdict(int)
    for e in gpu_events:
        # Simplify kernel name
        name = e['name']
        if 'elementwise' in name:
            kernel_counts['elementwise'] += 1
        elif 'gemm' in name.lower() or 'matmul' in name.lower():
            kernel_counts['GEMM/MatMul'] += 1
        elif 'softmax' in name.lower():
            kernel_counts['softmax'] += 1
        elif 'layernorm' in name.lower() or 'layer_norm' in name.lower():
            kernel_counts['LayerNorm'] += 1
        elif 'embedding' in name.lower():
            kernel_counts['embedding'] += 1
        else:
            kernel_counts['other'] += 1
    
    names = list(kernel_counts.keys())
    counts = list(kernel_counts.values())
    colors_bar = plt.cm.Set3(np.linspace(0, 1, len(names)))
    ax.bar(names, counts, color=colors_bar, edgecolor='black')
    ax.set_ylabel('Count')
    ax.set_title(f'GPU Kernel Types (Total: {len(gpu_events)})')
    ax.tick_params(axis='x', rotation=30)
    
    for i, (n, c) in enumerate(zip(names, counts)):
        ax.text(i, c + max(counts)*0.02, str(c), ha='center', fontsize=9)
    
    # Time breakdown
    ax = axes[1, 1]
    kernel_times = defaultdict(float)
    for e in gpu_events:
        name = e['name']
        if 'elementwise' in name:
            kernel_times['elementwise'] += e['dur_ms']
        elif 'gemm' in name.lower() or 'matmul' in name.lower():
            kernel_times['GEMM/MatMul'] += e['dur_ms']
        elif 'softmax' in name.lower():
            kernel_times['softmax'] += e['dur_ms']
        elif 'layernorm' in name.lower() or 'layer_norm' in name.lower():
            kernel_times['LayerNorm'] += e['dur_ms']
        elif 'embedding' in name.lower():
            kernel_times['embedding'] += e['dur_ms']
        else:
            kernel_times['other'] += e['dur_ms']
    
    names = list(kernel_times.keys())
    times = list(kernel_times.values())
    ax.bar(names, times, color=colors_bar, edgecolor='black')
    ax.set_ylabel('Time (ms)')
    ax.set_title('GPU Time by Kernel Type')
    ax.tick_params(axis='x', rotation=30)
    
    for i, (n, t) in enumerate(zip(names, times)):
        ax.text(i, t + max(times)*0.02, f'{t:.1f}', ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_gap_analysis.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_prefix}_gap_analysis.png")
    plt.close()
    
    # =========================================================================
    # Plot 3: Optimization Potential Analysis
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Current vs Optimized comparison
    ax = axes[0, 0]
    
    # Calculate current metrics
    total_gpu_time = sum(e['dur_ms'] for e in gpu_events)
    total_gap_time = sum(g['gap_ms'] for g in gaps)
    total_time_ms = max(e['end_ms'] for e in all_events)
    
    current_metrics = {
        'Kernel Launches': len(gpu_events),
        'Total Gap Time (ms)': total_gap_time,
        'GPU Util %': total_gpu_time / total_time_ms * 100
    }
    
    # Estimated optimized (fused kernel approach)
    optimized_metrics = {
        'Kernel Launches': len(gpu_events) // 50,  # ~50x reduction with fusion
        'Total Gap Time (ms)': total_gap_time * 0.1,  # 90% gap reduction
        'GPU Util %': min(95, total_gpu_time / total_time_ms * 100 * 2.5)  # ~2.5x improvement
    }
    
    x = np.arange(3)
    width = 0.35
    
    ax.bar(x - width/2, list(current_metrics.values()), width, label='Current', color='red', alpha=0.7)
    ax.bar(x + width/2, list(optimized_metrics.values()), width, label='Optimized (Est.)', color='green', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(list(current_metrics.keys()), rotation=15)
    ax.set_title('Current vs Optimized (Estimated)')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Where time is spent
    ax = axes[0, 1]
    cpu_overhead = max(0, total_time_ms - total_gpu_time - total_gap_time)
    time_breakdown = {
        'GPU Kernels': max(0.1, total_gpu_time),
        'GPU Idle (Gaps)': max(0.1, total_gap_time),
        'CPU Overhead': max(0.1, cpu_overhead)
    }
    
    colors_pie = ['#4CAF50', '#FFC107', '#F44336']
    wedges, texts, autotexts = ax.pie(time_breakdown.values(), 
                                       labels=time_breakdown.keys(),
                                       colors=colors_pie,
                                       autopct='%1.1f%%',
                                       startangle=90)
    ax.set_title(f'Time Breakdown (Total: {total_time_ms:.1f}ms)')
    
    # Optimization opportunity by category
    ax = axes[1, 0]
    
    # Count operations by type that could be fused
    fuseable_ops = {
        'Per-param mul': len([e for e in cpu_events if e['name'] == 'aten::mul']),
        'Per-param add': len([e for e in cpu_events if e['name'] == 'aten::add']),
        'Per-param RNG': len([e for e in cpu_events if 'normal' in e['name']]),
        'Small kernels': len([e for e in gpu_events if 'elementwise' in e['name']]),
    }
    
    ax.bar(fuseable_ops.keys(), fuseable_ops.values(), color='orange', edgecolor='black')
    ax.set_ylabel('Count')
    ax.set_title('Operations Fuseable into Single Kernel')
    ax.tick_params(axis='x', rotation=15)
    
    for i, (k, v) in enumerate(fuseable_ops.items()):
        ax.text(i, v + max(fuseable_ops.values())*0.02, str(v), ha='center', fontsize=9)
    
    # Projected speedup
    ax = axes[1, 1]
    
    speedup_scenarios = {
        'Current': 1.0,
        'Fused Perturbation\n(CUDA kernel)': 1.5,
        'Batched RNG': 1.25,
        'Fused + Batched': 1.8,
        'Full Optimization\n(All fused)': 2.5
    }
    
    colors_bar = ['red', 'orange', 'yellow', 'lightgreen', 'green']
    ax.bar(speedup_scenarios.keys(), speedup_scenarios.values(), color=colors_bar, edgecolor='black')
    ax.set_ylabel('Relative Speedup')
    ax.set_title('Projected Speedup from Optimizations')
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.tick_params(axis='x', rotation=15)
    
    for i, (k, v) in enumerate(speedup_scenarios.items()):
        ax.text(i, v + 0.05, f'{v:.1f}x', ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_optimization.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_prefix}_optimization.png")
    plt.close()
    
    print(f"\nAll timeline visualizations saved!")
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print("OPTIMIZATION ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"\n📊 Current State:")
    print(f"   Total kernel launches: {len(gpu_events):,}")
    print(f"   Total GPU time: {total_gpu_time:.2f} ms")
    print(f"   Total gap time: {total_gap_time:.2f} ms")
    print(f"   GPU utilization: {total_gpu_time/total_time_ms*100:.1f}%")
    
    print(f"\n🎯 Fusion Targets:")
    for k, v in fuseable_ops.items():
        print(f"   {k}: {v:,} operations")
    
    print(f"\n💡 Expected Improvements with Fused Kernels:")
    print(f"   Kernel launches: {len(gpu_events):,} → ~{len(gpu_events)//50:,} (50x reduction)")
    print(f"   Gap time: {total_gap_time:.1f}ms → ~{total_gap_time*0.1:.1f}ms (90% reduction)")
    print(f"   GPU util: {total_gpu_time/total_time_ms*100:.1f}% → ~{min(90, total_gpu_time/total_time_ms*100*2.5):.0f}%")
    print(f"   Overall speedup: ~2-3x expected")


def main():
    parser = argparse.ArgumentParser(description="Visualize CPU/GPU timeline from Chrome trace")
    parser.add_argument("trace_file", help="Path to Chrome trace JSON file")
    parser.add_argument("--output", "-o", default="./profiler_logs/timeline", 
                        help="Output prefix for PNG files")
    args = parser.parse_args()
    
    create_timeline_visualization(args.trace_file, args.output)


if __name__ == "__main__":
    main()
