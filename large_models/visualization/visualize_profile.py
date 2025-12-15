#!/usr/bin/env python3
"""
Simple script to visualize profiling results from DiZO training.
This creates basic plots from the profiler summary data.

Usage:
    python visualize_profile.py --summary_file ./profiler_logs/profiler_summary.txt
"""

import argparse
import re
import matplotlib.pyplot as plt
import numpy as np


def parse_profiler_summary(filepath):
    """Parse the profiler summary text file"""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Extract data from CPU time section
    cpu_data = []
    cuda_data = []
    
    # Find the CPU time section
    lines = content.split('\n')
    in_cpu_section = False
    in_cuda_section = False
    
    for line in lines:
        if '--- CPU Time (sorted by total time) ---' in line:
            in_cpu_section = True
            in_cuda_section = False
            continue
        elif '--- CUDA Time (sorted by total time) ---' in line:
            in_cpu_section = False
            in_cuda_section = True
            continue
        elif '--- Memory Usage' in line:
            in_cpu_section = False
            in_cuda_section = False
            continue
        
        # Parse data lines (they start with spaces and have the operation name)
        if (in_cpu_section or in_cuda_section) and line.strip() and not line.startswith('---'):
            # Skip header and separator lines
            if 'Name' in line or '------' in line or 'Self CPU' in line:
                continue
            
            parts = line.split()
            if len(parts) >= 6 and '%' in line:
                try:
                    name = parts[0]
                    # Skip entries that are clearly not operation names
                    if name.startswith('Self') or name.startswith('total'):
                        continue
                    
                    # Find percentages
                    percentages = re.findall(r'(\d+\.\d+)%', line)
                    if len(percentages) >= 2:
                        cpu_pct = float(percentages[0])
                        
                        if in_cpu_section:
                            cpu_data.append((name, cpu_pct))
                        elif in_cuda_section:
                            cuda_data.append((name, cpu_pct))
                except (ValueError, IndexError):
                    continue
    
    return cpu_data, cuda_data


def create_pie_chart(data, title, output_file, top_n=10):
    """Create a pie chart of the top N operations"""
    # Get top N operations
    sorted_data = sorted(data, key=lambda x: x[1], reverse=True)[:top_n]
    
    # Calculate "Other" percentage
    top_sum = sum(d[1] for d in sorted_data)
    other_pct = max(0, 100 - top_sum)
    
    labels = [d[0] for d in sorted_data]
    sizes = [d[1] for d in sorted_data]
    
    if other_pct > 0:
        labels.append('Other')
        sizes.append(other_pct)
    
    # Create pie chart
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
    
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%',
                                       colors=colors, startangle=90,
                                       pctdistance=0.8)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Make labels more readable
    for text in texts:
        text.set_fontsize(9)
    for autotext in autotexts:
        autotext.set_fontsize(8)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def create_bar_chart(data, title, output_file, top_n=15):
    """Create a horizontal bar chart of top operations"""
    sorted_data = sorted(data, key=lambda x: x[1], reverse=True)[:top_n]
    
    labels = [d[0] for d in sorted_data]
    values = [d[1] for d in sorted_data]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    y_pos = np.arange(len(labels))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(labels)))
    
    bars = ax.barh(y_pos, values, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()  # Labels read top-to-bottom
    ax.set_xlabel('Time (%)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{value:.1f}%', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def create_mezo_breakdown_chart(cpu_data, output_file):
    """Create a breakdown chart specific to MeZO operations"""
    
    # Define MeZO-specific operations
    mezo_ops = {
        'zo_gradient_estimation': 0,
        'zo_perturb_+eps': 0,
        'zo_perturb_-2eps': 0,
        'zo_perturb_+eps_reset': 0,
        'zo_forward_1': 0,
        'zo_forward_2': 0,
        'zo_update': 0,
        'zo_parameter_update': 0,
    }
    
    other_time = 0
    
    for name, pct in cpu_data:
        if name in mezo_ops:
            mezo_ops[name] = pct
        else:
            other_time += pct
    
    # Create the stacked bar
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Colors for each component
    colors = {
        'zo_perturb_+eps': '#FF6B6B',
        'zo_perturb_-2eps': '#4ECDC4',
        'zo_perturb_+eps_reset': '#45B7D1',
        'zo_forward_1': '#96CEB4',
        'zo_forward_2': '#FFEAA7',
        'zo_update': '#DDA0DD',
        'zo_parameter_update': '#98D8C8',
        'zo_gradient_estimation': '#F7DC6F',
    }
    
    # Create stacked bars
    left = 0
    for name, pct in mezo_ops.items():
        if pct > 0:
            color = colors.get(name, '#cccccc')
            ax.barh(['MeZO Step'], pct, left=left, color=color, label=name, edgecolor='white')
            if pct > 3:  # Only label if big enough
                ax.text(left + pct/2, 0, f'{name}\n{pct:.1f}%', ha='center', va='center', fontsize=8)
            left += pct
    
    ax.set_xlabel('Time (%)', fontsize=12)
    ax.set_title('MeZO Training Step Breakdown', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 100)
    
    # Add legend
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize DiZO profiling results")
    parser.add_argument("--summary_file", type=str, 
                       default="./profiler_logs/profiler_summary.txt",
                       help="Path to profiler summary file")
    parser.add_argument("--output_dir", type=str, default="./profiler_logs",
                       help="Output directory for plots")
    
    args = parser.parse_args()
    
    print(f"Reading profiler summary from: {args.summary_file}")
    
    try:
        cpu_data, cuda_data = parse_profiler_summary(args.summary_file)
    except FileNotFoundError:
        print(f"Error: Could not find {args.summary_file}")
        print("Please run the profiling first with: python profile_dizo.py")
        return
    
    print(f"Found {len(cpu_data)} CPU operations and {len(cuda_data)} CUDA operations")
    
    if cpu_data:
        # Create CPU time breakdown
        create_bar_chart(
            cpu_data,
            "CPU Time Breakdown (Top 15 Operations)",
            f"{args.output_dir}/cpu_time_breakdown.png"
        )
        
        # Create pie chart
        create_pie_chart(
            cpu_data,
            "CPU Time Distribution",
            f"{args.output_dir}/cpu_time_pie.png"
        )
        
        # Create MeZO-specific breakdown
        create_mezo_breakdown_chart(
            cpu_data,
            f"{args.output_dir}/mezo_breakdown.png"
        )
    
    if cuda_data:
        # Create CUDA time breakdown
        create_bar_chart(
            cuda_data,
            "CUDA Time Breakdown (Top 15 Operations)",
            f"{args.output_dir}/cuda_time_breakdown.png"
        )
    
    print("\nVisualization complete!")
    print(f"Check the output files in: {args.output_dir}")


if __name__ == "__main__":
    main()
