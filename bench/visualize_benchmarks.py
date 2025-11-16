#!/usr/bin/env python3
"""
Benchmark Visualization Tool

Creates visual plots for benchmark comparisons:
- Latency vs Energy scatter plots
- Memory vs Accuracy plots
- Pareto frontier visualization
- Bar charts comparing patterns

Usage:
    python bench/visualize_benchmarks.py \\
        --input bench/results/all_results.json \\
        --output bench/plots/

Requires: matplotlib
"""

import argparse
import json
import os
import sys
from typing import List, Dict, Tuple

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("Error: matplotlib is required for visualization")
    print("Install with: pip install matplotlib")
    sys.exit(1)


# Color scheme for patterns
PATTERN_COLORS = {
    "sliding_window": "#2E86AB",
    "block_local_global": "#A23B72",
    "nm_structured": "#F18F01",
    "lsh": "#C73E1D",
    "landmark": "#6A994E",
}

# Marker styles for precisions
PRECISION_MARKERS = {
    "fp32": "o",
    "bf16": "s",
    "i8": "^",
    "i4": "D",
}

# Size mapping for better visibility
PRECISION_SIZES = {
    "fp32": 100,
    "bf16": 80,
    "i8": 60,
    "i4": 50,
}


def load_results(results_file: str) -> List[Dict]:
    """Load benchmark results from JSON file"""
    if not os.path.exists(results_file):
        print(f"Error: {results_file} not found")
        sys.exit(1)
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    # Handle different JSON formats
    if isinstance(data, list):
        return data
    elif 'results' in data:
        return data['results']
    else:
        return [data]


def plot_latency_vs_energy(results: List[Dict], output_file: str):
    """
    Create scatter plot of cycles vs energy.
    
    Shows trade-off between speed and power consumption.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for result in results:
        pattern = result.get('pattern', 'unknown')
        precision = result.get('precision', 'fp32')
        
        metrics = result.get('metrics', {})
        # Use cycles instead of latency_ms
        cycles = metrics.get('performance', {}).get('cycles', 0)
        energy = metrics.get('energy', {}).get('total_uj', 0)
        
        if cycles > 0 and energy > 0:
            color = PATTERN_COLORS.get(pattern, '#808080')
            marker = PRECISION_MARKERS.get(precision, 'o')
            size = PRECISION_SIZES.get(precision, 50)
            
            ax.scatter(cycles, energy, 
                      c=color, marker=marker, s=size,
                      alpha=0.7, edgecolors='black', linewidths=0.5,
                      label=f"{pattern} ({precision})")
    
    ax.set_xlabel('Cycles (millions)', fontsize=12)
    ax.set_ylabel('Energy (µJ)', fontsize=12)
    ax.set_title('Cycles vs Energy Trade-off', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add ideal region annotation
    if results:
        min_cycles = min(r['metrics']['performance']['cycles'] 
                     for r in results if 'metrics' in r)
        min_eng = min(r['metrics']['energy']['total_uj'] 
                     for r in results if 'metrics' in r and 'energy' in r['metrics'])
        ax.axvline(min_cycles/1e6, color='green', linestyle='--', alpha=0.3, label='Best Cycles')
        ax.axhline(min_eng, color='green', linestyle='--', alpha=0.3, label='Best Energy')
    
    # Convert cycles to millions for better readability
    ax.ticklabel_format(style='plain', axis='x')
    import matplotlib.ticker as ticker
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))
    
    # Legend (deduplicate)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), 
             loc='upper right', fontsize=8, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_file}")


def plot_memory_vs_accuracy(results: List[Dict], output_file: str):
    """
    Create scatter plot of memory vs accuracy.
    
    Shows trade-off between memory footprint and numerical quality.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for result in results:
        pattern = result.get('pattern', 'unknown')
        precision = result.get('precision', 'fp32')
        
        metrics = result.get('metrics', {})
        memory_mb = metrics.get('memory_mb', 0)
        
        # Try different accuracy metric names
        accuracy = None
        if 'accuracy' in metrics:
            # Higher cosine similarity = better (close to 1.0)
            accuracy = metrics['accuracy'].get('cosine_similarity')
            if accuracy is None:
                # Lower relative MAE = better, so invert it
                rel_mae = metrics['accuracy'].get('relative_mae')
                if rel_mae is not None:
                    accuracy = 1.0 - rel_mae  # Convert to "accuracy" metric
        
        if memory_mb > 0 and accuracy is not None:
            color = PATTERN_COLORS.get(pattern, '#808080')
            marker = PRECISION_MARKERS.get(precision, 'o')
            size = PRECISION_SIZES.get(precision, 50)
            
            ax.scatter(memory_mb, accuracy, 
                      c=color, marker=marker, s=size,
                      alpha=0.7, edgecolors='black', linewidths=0.5,
                      label=f"{pattern} ({precision})")
    
    ax.set_xlabel('Memory Footprint (MB)', fontsize=12)
    ax.set_ylabel('Accuracy (Cosine Similarity)', fontsize=12)
    ax.set_title('Memory vs Accuracy Trade-off', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.95, 1.005])  # Focus on high-accuracy region
    
    # Legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), 
             loc='lower right', fontsize=8, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_file}")


def plot_pattern_comparison(results: List[Dict], output_file: str):
    """
    Create bar chart comparing patterns across metrics.
    """
    # Group by pattern
    pattern_data = {}
    for result in results:
        pattern = result.get('pattern', 'unknown')
        precision = result.get('precision', 'fp32')
        
        if pattern not in pattern_data:
            pattern_data[pattern] = {}
        
        metrics = result.get('metrics', {})
        # Get memory_mb from derived or calculate from metrics
        memory_mb = result.get('derived', {}).get('memory_mb', 0)
        if memory_mb == 0:
            bytes_total = metrics.get('memory', {}).get('bytes_read', 0) + metrics.get('memory', {}).get('bytes_written', 0)
            memory_mb = bytes_total / (1024 * 1024)
        
        pattern_data[pattern][precision] = {
            'cycles': metrics.get('performance', {}).get('cycles', 0),
            'memory': memory_mb,
            'energy': metrics.get('energy', {}).get('total_uj', 0),
        }
    
    # Create subplot for each metric
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    patterns = list(pattern_data.keys())
    precisions = ['fp32', 'bf16', 'i8', 'i4']
    x = np.arange(len(patterns))
    width = 0.2
    
    metrics_to_plot = [
        ('cycles', 'Cycles (M)', axes[0]),
        ('memory', 'Memory (MB)', axes[1]),
        ('energy', 'Energy (µJ)', axes[2]),
    ]
    
    for metric_key, metric_label, ax in metrics_to_plot:
        for i, prec in enumerate(precisions):
            values = []
            for pattern in patterns:
                val = pattern_data[pattern].get(prec, {}).get(metric_key, 0)
                # Convert cycles to millions for readability
                if metric_key == 'cycles':
                    val = val / 1e6
                values.append(val)
            
            offset = width * (i - 1.5)
            bars = ax.bar(x + offset, values, width, label=prec, alpha=0.8)
        
        ax.set_xlabel('Pattern', fontsize=10)
        ax.set_ylabel(metric_label, fontsize=10)
        ax.set_title(metric_label, fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(patterns, rotation=15, ha='right', fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_file}")


def plot_pareto_frontier(results: List[Dict], output_file: str, 
                        x_metric: str = 'memory_mb',
                        y_metric: str = 'performance.cycles'):
    """
    Visualize Pareto-optimal frontier for two objectives.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract data
    data_points = []
    for result in results:
        pattern = result.get('pattern', 'unknown')
        precision = result.get('precision', 'fp32')
        
        metrics = result.get('metrics', {})
        
        # Handle nested metrics
        def get_metric(m, key):
            # Special handling for memory_mb - check derived first
            if key == 'memory_mb':
                mb = result.get('derived', {}).get('memory_mb', 0)
                if mb > 0:
                    return mb
                # Calculate from raw metrics if not in derived
                bytes_total = m.get('memory', {}).get('bytes_read', 0) + m.get('memory', {}).get('bytes_written', 0)
                return bytes_total / (1024 * 1024)
            
            if '.' in key:
                parts = key.split('.')
                val = m
                for part in parts:
                    val = val.get(part, {})
                return val if isinstance(val, (int, float)) else 0
            return m.get(key, 0)
        
        x_val = get_metric(metrics, x_metric)
        y_val = get_metric(metrics, y_metric)
        
        if x_val > 0 and y_val > 0:
            data_points.append({
                'x': x_val,
                'y': y_val,
                'pattern': pattern,
                'precision': precision,
            })
    
    # Identify Pareto frontier (minimize both)
    pareto_points = []
    for i, p1 in enumerate(data_points):
        is_dominated = False
        for j, p2 in enumerate(data_points):
            if i == j:
                continue
            # p2 dominates p1 if p2 is better in both objectives
            if p2['x'] < p1['x'] and p2['y'] < p1['y']:
                is_dominated = True
                break
        if not is_dominated:
            pareto_points.append(p1)
    
    # Plot all points
    for point in data_points:
        color = PATTERN_COLORS.get(point['pattern'], '#808080')
        marker = PRECISION_MARKERS.get(point['precision'], 'o')
        size = PRECISION_SIZES.get(point['precision'], 50)
        
        is_pareto = point in pareto_points
        alpha = 0.9 if is_pareto else 0.3
        edgewidth = 2 if is_pareto else 0.5
        
        ax.scatter(point['x'], point['y'],
                  c=color, marker=marker, s=size,
                  alpha=alpha, edgecolors='black', linewidths=edgewidth,
                  label=f"{point['pattern']} ({point['precision']})")
    
    # Draw Pareto frontier line
    if pareto_points:
        pareto_sorted = sorted(pareto_points, key=lambda p: p['x'])
        pareto_x = [p['x'] for p in pareto_sorted]
        pareto_y = [p['y'] for p in pareto_sorted]
        ax.plot(pareto_x, pareto_y, 'k--', alpha=0.5, linewidth=2, label='Pareto Frontier')
    
    x_label = x_metric.replace('_', ' ').replace('.', ' ').title()
    y_label = y_metric.replace('_', ' ').replace('.', ' ').title()
    
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title('Pareto-Optimal Frontier', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Legend (deduplicate)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), 
             loc='upper right', fontsize=7, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_file}")


def plot_efficiency_radar(results: List[Dict], output_file: str):
    """
    Create radar chart showing multi-dimensional efficiency.
    """
    from math import pi
    
    # Select representative configurations
    configs_to_plot = []
    for result in results:
        pattern = result.get('pattern', 'unknown')
        precision = result.get('precision', 'fp32')
        
        # Only plot i4 and fp32 for clarity
        if precision in ['i4', 'fp32']:
            configs_to_plot.append(result)
    
    if not configs_to_plot:
        print("⚠️  Not enough data for radar chart")
        return
    
    # Normalize metrics to 0-1 scale (higher is better)
    categories = ['Speed', 'Memory\nEfficiency', 'Energy\nEfficiency', 'Accuracy']
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    
    angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))]
    angles += angles[:1]
    
    for result in configs_to_plot[:8]:  # Limit to 8 for clarity
        pattern = result.get('pattern', 'unknown')
        precision = result.get('precision', 'fp32')
        metrics = result.get('metrics', {})
        
        # Normalize each metric (inverse for minimization metrics)
        cycles = metrics.get('performance', {}).get('cycles', 10000000)
        memory_mb = result.get('derived', {}).get('memory_mb', 0)
        if memory_mb == 0:
            bytes_total = metrics.get('memory', {}).get('bytes_read', 0) + metrics.get('memory', {}).get('bytes_written', 0)
            memory_mb = bytes_total / (1024 * 1024) if bytes_total > 0 else 10
        energy = metrics.get('energy', {}).get('total_uj', 500)
        # Accuracy metric - use checksum as proxy (normalized)
        accuracy = min(1.0, metrics.get('accuracy', {}).get('checksum', 100) / 500)
        
        # Convert to 0-1 scale (higher = better)
        values = [
            1.0 / (cycles / 10000000),  # Speed (inverse of cycles, normalized)
            1.0 / memory_mb * 10,        # Memory efficiency (inverse)
            1.0 / energy * 500,          # Energy efficiency (inverse)
            accuracy,                    # Accuracy (0-1)
        ]
        values += values[:1]
        
        color = PATTERN_COLORS.get(pattern, '#808080')
        label = f"{pattern[:10]} ({precision})"
        
        ax.plot(angles, values, 'o-', linewidth=2, label=label, color=color)
        ax.fill(angles, values, alpha=0.1, color=color)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title('Multi-Dimensional Efficiency', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=8)
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Visualize benchmark results')
    parser.add_argument('--input', required=True, help='Input JSON results file')
    parser.add_argument('--output-dir', default='bench/plots', help='Output directory for plots')
    parser.add_argument('--all', action='store_true', help='Generate all plots')
    parser.add_argument('--latency-energy', action='store_true', help='Latency vs energy plot')
    parser.add_argument('--memory-accuracy', action='store_true', help='Memory vs accuracy plot')
    parser.add_argument('--pattern-comparison', action='store_true', help='Pattern comparison bars')
    parser.add_argument('--pareto', action='store_true', help='Pareto frontier plot')
    parser.add_argument('--radar', action='store_true', help='Efficiency radar chart')
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from: {args.input}")
    results = load_results(args.input)
    print(f"Loaded {len(results)} benchmark results")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate plots
    if args.all or args.latency_energy:
        plot_latency_vs_energy(results, os.path.join(args.output_dir, 'latency_vs_energy.png'))
    
    if args.all or args.memory_accuracy:
        plot_memory_vs_accuracy(results, os.path.join(args.output_dir, 'memory_vs_accuracy.png'))
    
    if args.all or args.pattern_comparison:
        plot_pattern_comparison(results, os.path.join(args.output_dir, 'pattern_comparison.png'))
    
    if args.all or args.pareto:
        plot_pareto_frontier(results, os.path.join(args.output_dir, 'pareto_frontier.png'))
    
    if args.all or args.radar:
        plot_efficiency_radar(results, os.path.join(args.output_dir, 'efficiency_radar.png'))
    
    if not any([args.all, args.latency_energy, args.memory_accuracy, 
                args.pattern_comparison, args.pareto, args.radar]):
        print("No plots requested. Use --all or specify individual plots.")
        print("See --help for options.")
    
    print("\n✅ Visualization complete!")


if __name__ == "__main__":
    main()

