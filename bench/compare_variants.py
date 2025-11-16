#!/usr/bin/env python3
"""
Variant Comparison Tool

Runs all variants for a given pattern and generates comparison tables.

Usage:
    python bench/compare_variants.py \\
        --pattern sliding_window \\
        --backend rvv \\
        --L 2048 \\
        --output-csv bench/results/variant_comparison.csv \\
        --output-md bench/results/variant_comparison.md
"""

import argparse
import json
import os
import subprocess
import sys
from typing import List, Dict

def run_benchmark(pattern: str, variant: str, backend: str, L: int, D: int) -> Dict:
    """Run a single benchmark and return the results"""
    output_file = f"/tmp/bench_{pattern}_{variant}.json"
    
    cmd = [
        sys.executable, "bench/unified_bench.py",
        "--pattern", pattern,
        "--variant", variant,
        "--backend", backend,
        "--L", str(L),
        "--D", str(D),
        "--iterations", "3",
        "--output", output_file
    ]
    
    print(f"Running: {pattern} / {variant}...", end=" ", flush=True)
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)
        with open(output_file, 'r') as f:
            results = json.load(f)
        print("✅")
        return results
    except Exception as e:
        print(f"❌ ({e})")
        return None


def generate_markdown_table(results: List[Dict], pattern: str) -> str:
    """Generate a markdown comparison table"""
    
    md = f"# Variant Comparison: {pattern}\n\n"
    md += "## Performance Comparison\n\n"
    md += "| Variant | Latency (ms) | Cycles | Memory Read (MB) | Memory Written (MB) | MAC Ops (M) | Efficiency Score |\n"
    md += "|---------|--------------|--------|------------------|---------------------|-------------|------------------|\n"
    
    for r in results:
        if not r:
            continue
        variant = r['variant']
        metrics = r['metrics']
        latency = metrics['performance']['latency_ms']
        cycles = metrics['performance'].get('cycles', 0)
        bytes_read = metrics['memory'].get('bytes_read', 0) / 1024 / 1024
        bytes_written = metrics['memory'].get('bytes_written', 0) / 1024 / 1024
        mac_ops = metrics['compute'].get('mac_ops', 0) / 1_000_000
        
        # Simple efficiency score: MAC ops / (cycles + memory traffic proxy)
        efficiency = mac_ops / (cycles / 1e6 + bytes_read + bytes_written + 0.001)
        
        md += f"| {variant:20s} | {latency:8.2f} | {cycles:10.0f} | {bytes_read:12.2f} | {bytes_written:15.2f} | {mac_ops:9.2f} | {efficiency:12.2f} |\n"
    
    md += "\n## Configuration Details\n\n"
    md += "| Variant | Window Size | Global Tokens | Tile Rows | Precision |\n"
    md += "|---------|-------------|---------------|-----------|--------|\n"
    
    for r in results:
        if not r:
            continue
        variant = r['variant']
        config = r['configuration']
        window = config.get('window_size', '-')
        global_tokens = config.get('global_tokens', '-')
        tile_rows = config.get('tile_rows', '-')
        precision = config.get('precision', 'fp32')
        
        md += f"| {variant:20s} | {window:11} | {global_tokens:13} | {tile_rows:9} | {precision:10s} |\n"
    
    md += "\n## Recommendations\n\n"
    
    # Find best for each objective
    valid_results = [r for r in results if r]
    if valid_results:
        best_latency = min(valid_results, key=lambda r: r['metrics']['performance']['latency_ms'])
        best_memory = min(valid_results, key=lambda r: r['metrics']['memory'].get('bytes_read', float('inf')))
        best_compute = max(valid_results, key=lambda r: r['metrics']['compute'].get('mac_ops', 0))
        
        md += f"- **Lowest Latency**: `{best_latency['variant']}` ({best_latency['metrics']['performance']['latency_ms']:.2f} ms)\n"
        md += f"- **Lowest Memory Traffic**: `{best_memory['variant']}` ({best_memory['metrics']['memory'].get('bytes_read', 0) / 1024 / 1024:.2f} MB read)\n"
        md += f"- **Highest Compute**: `{best_compute['variant']}` ({best_compute['metrics']['compute'].get('mac_ops', 0) / 1_000_000:.2f} M MAC ops)\n"
    
    return md


def generate_csv(results: List[Dict]) -> str:
    """Generate CSV output"""
    lines = []
    lines.append("variant,latency_ms,cycles,bytes_read,bytes_written,mac_ops,window_size,global_tokens,tile_rows,precision")
    
    for r in results:
        if not r:
            continue
        variant = r['variant']
        metrics = r['metrics']
        config = r['configuration']
        
        latency = metrics['performance']['latency_ms']
        cycles = metrics['performance'].get('cycles', 0)
        bytes_read = metrics['memory'].get('bytes_read', 0)
        bytes_written = metrics['memory'].get('bytes_written', 0)
        mac_ops = metrics['compute'].get('mac_ops', 0)
        window = config.get('window_size', '')
        global_tokens = config.get('global_tokens', '')
        tile_rows = config.get('tile_rows', '')
        precision = config.get('precision', 'fp32')
        
        lines.append(f"{variant},{latency:.3f},{cycles:.0f},{bytes_read},{bytes_written},{mac_ops},{window},{global_tokens},{tile_rows},{precision}")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description='Compare all variants for a sparse attention pattern')
    parser.add_argument('--pattern', required=True,
                       choices=['sliding_window', 'block_topk', 'nm_structured', 'lsh'],
                       help='Sparse attention pattern')
    parser.add_argument('--backend', default='rvv',
                       choices=['cpu', 'gpu', 'rvv', 'custom_isa'],
                       help='Execution backend')
    parser.add_argument('--L', type=int, default=128, help='Sequence length')
    parser.add_argument('--D', type=int, default=32, help='Head dimension')
    parser.add_argument('--output-csv', default='', help='Output CSV file')
    parser.add_argument('--output-md', default='', help='Output markdown file')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print(f"Comparing All Variants: {args.pattern}")
    print("=" * 70)
    print()
    
    variants = ['ultra_low_power', 'low_power', 'balanced', 'high_performance', 'ultra_high_perf']
    
    results = []
    for variant in variants:
        result = run_benchmark(args.pattern, variant, args.backend, args.L, args.D)
        results.append(result)
    
    print()
    print("=" * 70)
    print("Comparison Complete!")
    print("=" * 70)
    print()
    
    # Generate outputs
    if args.output_md:
        md_content = generate_markdown_table(results, args.pattern)
        os.makedirs(os.path.dirname(args.output_md) if os.path.dirname(args.output_md) else '.', exist_ok=True)
        with open(args.output_md, 'w') as f:
            f.write(md_content)
        print(f"✅ Markdown table saved to: {args.output_md}")
        print()
        print(md_content)
    
    if args.output_csv:
        csv_content = generate_csv(results)
        os.makedirs(os.path.dirname(args.output_csv) if os.path.dirname(args.output_csv) else '.', exist_ok=True)
        with open(args.output_csv, 'w') as f:
            f.write(csv_content)
        print(f"✅ CSV saved to: {args.output_csv}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

