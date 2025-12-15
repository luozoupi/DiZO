#!/usr/bin/env python3
"""
Analyze Chrome trace JSON files to extract CPU/GPU timeline information
and identify latency-bound patterns.

This script parses the PyTorch profiler Chrome trace output to:
1. Show CPU-GPU overlap/gaps
2. Identify kernel launch delays
3. Calculate GPU utilization
4. Find serialization bottlenecks
"""

import json
import argparse
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plot_gantt(cpu_events, gpu_events, mezo_events, out_path="timeline_gantt.png", window_ms=None):
    # Normalize to start at zero and convert to ms
    all_events = cpu_events + gpu_events + mezo_events
    if not all_events:
        print("No events to plot.")
        return
    min_ts = min(e['ts'] for e in all_events)
    for e in all_events:
        e['start_ms'] = (e['ts'] - min_ts) / 1000
        e['dur_ms'] = e['dur'] / 1000
        e['end_ms'] = e['start_ms'] + e['dur_ms']

    # Optional window
    if window_ms is not None:
        all_events = [e for e in all_events if e['start_ms'] <= window_ms]

    fig, ax = plt.subplots(figsize=(14, 4))
    lanes = {'GPU': (gpu_events, 'green'), 'CPU': (cpu_events, 'blue'), 'MeZO': (mezo_events, 'purple')}
    y = 0
    yticks = []
    ylabels = []
    for label, (events, color) in lanes.items():
        for e in events:
            if window_ms is not None and e['start_ms'] > window_ms:
                continue
            ax.barh(y, e['dur_ms'], left=e['start_ms'], height=0.6, color=color, alpha=0.7, edgecolor='black', linewidth=0.4)
        yticks.append(y)
        ylabels.append(label)
        y += 1

    ax.set_xlabel("Time (ms)")
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    if window_ms:
        ax.set_xlim(0, window_ms)
    ax.set_title("CPU/GPU/MeZO Gantt (Chrome trace)")
    ax.grid(True, axis='x', alpha=0.3)
    ax.legend(handles=[
        mpatches.Patch(color='green', label='GPU kernels'),
        mpatches.Patch(color='blue', label='CPU ops'),
        mpatches.Patch(color='purple', label='MeZO phases'),
    ], loc='upper right')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved {out_path}")

def load_trace(filepath):
    """Load Chrome trace JSON file"""
    print(f"Loading trace file: {filepath}")
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def analyze_trace(trace_data):
    """Analyze the trace data for CPU/GPU patterns"""
    
    # Chrome trace format has 'traceEvents' list
    events = trace_data.get('traceEvents', trace_data)
    
    cpu_events = []
    gpu_events = []
    kernel_launches = []
    cuda_runtime = []
    
    # Categorize events
    for event in events:
        if not isinstance(event, dict):
            continue
            
        name = event.get('name', '')
        cat = event.get('cat', '')
        ph = event.get('ph', '')  # Phase: B=begin, E=end, X=complete
        ts = event.get('ts', 0)   # Timestamp in microseconds
        dur = event.get('dur', 0) # Duration in microseconds
        tid = event.get('tid', '')
        pid = event.get('pid', '')
        
        # Skip metadata events
        if ph in ['M', 's', 't', 'f']:
            continue
            
        event_info = {
            'name': name,
            'cat': cat,
            'ts': ts,
            'dur': dur,
            'tid': tid,
            'pid': pid,
            'ph': ph
        }
        
        # Categorize by type
        if 'cuda' in cat.lower() or 'kernel' in cat.lower():
            gpu_events.append(event_info)
        elif 'cudaLaunchKernel' in name or 'cudaLaunch' in name:
            kernel_launches.append(event_info)
            cuda_runtime.append(event_info)
        elif 'cuda' in name.lower():
            cuda_runtime.append(event_info)
        elif cat == 'cpu_op' or 'aten::' in name or 'zo_' in name:
            cpu_events.append(event_info)
    
    return cpu_events, gpu_events, kernel_launches, cuda_runtime


def compute_gpu_utilization(gpu_events, total_time_us):
    """Compute GPU utilization from kernel events"""
    if not gpu_events or total_time_us == 0:
        return 0.0
    
    # Sort by timestamp
    sorted_events = sorted(gpu_events, key=lambda x: x['ts'])
    
    # Merge overlapping intervals
    busy_intervals = []
    for event in sorted_events:
        if event['dur'] > 0:
            start = event['ts']
            end = start + event['dur']
            
            if busy_intervals and start <= busy_intervals[-1][1]:
                # Merge with previous
                busy_intervals[-1] = (busy_intervals[-1][0], max(busy_intervals[-1][1], end))
            else:
                busy_intervals.append((start, end))
    
    total_busy = sum(end - start for start, end in busy_intervals)
    return total_busy / total_time_us * 100


def find_cpu_gpu_gaps(cpu_events, gpu_events):
    """Find gaps where GPU is idle waiting for CPU"""
    
    if not gpu_events:
        return []
    
    # Sort GPU events by timestamp
    sorted_gpu = sorted([e for e in gpu_events if e['dur'] > 0], key=lambda x: x['ts'])
    
    gaps = []
    for i in range(1, len(sorted_gpu)):
        prev_end = sorted_gpu[i-1]['ts'] + sorted_gpu[i-1]['dur']
        curr_start = sorted_gpu[i]['ts']
        gap = curr_start - prev_end
        
        if gap > 10:  # Only gaps > 10 microseconds
            gaps.append({
                'gap_us': gap,
                'after_kernel': sorted_gpu[i-1]['name'],
                'before_kernel': sorted_gpu[i]['name'],
                'timestamp': prev_end
            })
    
    return gaps


def analyze_kernel_launches(kernel_launches):
    """Analyze kernel launch patterns"""
    
    if not kernel_launches:
        return {}
    
    durations = [e['dur'] for e in kernel_launches if e['dur'] > 0]
    
    if not durations:
        return {}
    
    return {
        'count': len(kernel_launches),
        'total_time_ms': sum(durations) / 1000,
        'avg_time_us': np.mean(durations),
        'min_time_us': np.min(durations),
        'max_time_us': np.max(durations),
        'std_time_us': np.std(durations)
    }


def analyze_operation_patterns(cpu_events):
    """Analyze CPU operation patterns to find repeated sequences"""
    
    # Group by operation name
    op_stats = defaultdict(lambda: {'count': 0, 'total_dur': 0, 'durations': []})
    
    for event in cpu_events:
        name = event['name']
        dur = event['dur']
        if dur > 0:
            op_stats[name]['count'] += 1
            op_stats[name]['total_dur'] += dur
            op_stats[name]['durations'].append(dur)
    
    # Calculate statistics
    results = []
    for name, stats in op_stats.items():
        if stats['count'] > 0:
            results.append({
                'name': name,
                'count': stats['count'],
                'total_ms': stats['total_dur'] / 1000,
                'avg_us': np.mean(stats['durations']),
                'std_us': np.std(stats['durations']) if len(stats['durations']) > 1 else 0
            })
    
    # Sort by total time
    results.sort(key=lambda x: x['total_ms'], reverse=True)
    return results


def find_serialization_points(cpu_events, gpu_events):
    """Find points where CPU waits for GPU (synchronization)"""
    
    sync_events = []
    for event in cpu_events:
        name = event['name'].lower()
        if any(s in name for s in ['synchronize', 'sync', 'wait', 'cudadevice']):
            sync_events.append(event)
    
    return sync_events


def print_timeline_analysis(trace_file):
    """Main analysis function"""
    
    trace_data = load_trace(trace_file)
    # cpu_events, gpu_events, kernel_launches, cuda_runtime = analyze_trace(trace_data)
    cpu_events, gpu_events, kernel_launches, cuda_runtime = analyze_trace(trace_data)
    mezo_events = [e for e in cpu_events if 'zo_' in e['name']]  # reuse parsed CPU events tagged by zo_
    plot_gantt(cpu_events, gpu_events, mezo_events, out_path="timeline_gantt.png", window_ms=200)
        
    print("\n" + "="*80)
    print("TIMELINE ANALYSIS REPORT")
    print("="*80)
    
    # Basic stats
    print(f"\n📊 Event Counts:")
    print(f"   CPU operations: {len(cpu_events)}")
    print(f"   GPU kernels: {len(gpu_events)}")
    print(f"   Kernel launches: {len(kernel_launches)}")
    print(f"   CUDA runtime calls: {len(cuda_runtime)}")
    
    # Time range
    all_events = cpu_events + gpu_events
    if all_events:
        timestamps = [e['ts'] for e in all_events if e['ts'] > 0]
        if timestamps:
            min_ts = min(timestamps)
            max_ts = max(timestamps)
            durations = [e['dur'] for e in all_events if e['dur'] > 0]
            max_end = max(e['ts'] + e['dur'] for e in all_events if e['dur'] > 0) if durations else max_ts
            total_time_us = max_end - min_ts
            
            print(f"\n⏱️  Time Range:")
            print(f"   Total trace duration: {total_time_us/1000:.2f} ms")
    else:
        total_time_us = 1  # Avoid division by zero
    
    # GPU Utilization
    gpu_util = compute_gpu_utilization(gpu_events, total_time_us)
    print(f"\n🖥️  GPU Utilization: {gpu_util:.1f}%")
    if gpu_util < 50:
        print(f"   ⚠️  LOW GPU UTILIZATION - CPU-bound workload detected!")
    
    # Kernel launch analysis
    launch_stats = analyze_kernel_launches(kernel_launches)
    if launch_stats:
        print(f"\n🚀 Kernel Launch Analysis:")
        print(f"   Total launches: {launch_stats['count']}")
        print(f"   Total launch time: {launch_stats['total_time_ms']:.2f} ms")
        print(f"   Avg launch time: {launch_stats['avg_time_us']:.2f} μs")
        print(f"   Launch time std: {launch_stats['std_time_us']:.2f} μs")
        
        # Estimate overhead
        launches_per_ms = launch_stats['count'] / (total_time_us / 1000)
        print(f"   Launch rate: {launches_per_ms:.1f} launches/ms")
        if launches_per_ms > 30:
            print(f"   ⚠️  HIGH LAUNCH RATE - Consider kernel fusion!")
    
    # CPU-GPU gaps
    gaps = find_cpu_gpu_gaps(cpu_events, gpu_events)
    if gaps:
        gap_times = [g['gap_us'] for g in gaps]
        total_gap_time = sum(gap_times)
        print(f"\n⏸️  GPU Idle Gaps:")
        print(f"   Number of gaps (>10μs): {len(gaps)}")
        print(f"   Total gap time: {total_gap_time/1000:.2f} ms")
        print(f"   Avg gap: {np.mean(gap_times):.1f} μs")
        print(f"   Max gap: {np.max(gap_times):.1f} μs")
        print(f"   Gap time %: {total_gap_time/total_time_us*100:.1f}%")
        
        # Show largest gaps
        gaps.sort(key=lambda x: x['gap_us'], reverse=True)
        print(f"\n   Top 5 largest gaps:")
        for i, gap in enumerate(gaps[:5]):
            print(f"   {i+1}. {gap['gap_us']:.0f}μs after '{gap['after_kernel'][:40]}'")
    
    # Operation patterns
    op_patterns = analyze_operation_patterns(cpu_events)
    if op_patterns:
        print(f"\n📈 Top CPU Operations by Time:")
        print(f"   {'Operation':<45} {'Count':>8} {'Total(ms)':>10} {'Avg(μs)':>10}")
        print(f"   {'-'*45} {'-'*8} {'-'*10} {'-'*10}")
        for op in op_patterns[:15]:
            name = op['name'][:45]
            print(f"   {name:<45} {op['count']:>8} {op['total_ms']:>10.2f} {op['avg_us']:>10.1f}")
    
    # Synchronization points
    sync_events = find_serialization_points(cpu_events, gpu_events)
    if sync_events:
        print(f"\n🔄 Synchronization Events: {len(sync_events)}")
        sync_time = sum(e['dur'] for e in sync_events if e['dur'] > 0)
        print(f"   Total sync time: {sync_time/1000:.2f} ms")
    
    # MeZO-specific analysis
    print(f"\n🎯 MeZO Operation Analysis:")
    mezo_ops = ['zo_perturb', 'zo_forward', 'zo_update', 'zo_gradient']
    for op_prefix in mezo_ops:
        matching = [e for e in cpu_events if op_prefix in e['name']]
        if matching:
            total_time = sum(e['dur'] for e in matching if e['dur'] > 0)
            count = len([e for e in matching if e['dur'] > 0])
            if count > 0:
                print(f"   {op_prefix}: {count} calls, {total_time/1000:.2f} ms total")
    
    # Summary
    print(f"\n" + "="*80)
    print("BOTTLENECK SUMMARY")
    print("="*80)
    
    issues = []
    if gpu_util < 50:
        issues.append(f"🔴 GPU utilization is only {gpu_util:.1f}% - severe CPU bottleneck")
    if launch_stats and launch_stats['count'] > 10000:
        issues.append(f"🔴 {launch_stats['count']} kernel launches - need kernel fusion")
    if gaps and sum(g['gap_us'] for g in gaps) / total_time_us > 0.2:
        issues.append(f"🟡 GPU idle {sum(g['gap_us'] for g in gaps)/total_time_us*100:.1f}% of time waiting for CPU")
    
    if issues:
        print("\nIdentified Issues:")
        for issue in issues:
            print(f"   {issue}")
    else:
        print("\n   ✅ No major bottlenecks identified")
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description="Analyze PyTorch profiler Chrome trace")
    parser.add_argument("trace_file", help="Path to Chrome trace JSON file")
    args = parser.parse_args()
    
    print_timeline_analysis(args.trace_file)


if __name__ == "__main__":
    main()
