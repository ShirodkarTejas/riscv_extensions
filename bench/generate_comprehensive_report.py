#!/usr/bin/env python3
"""
Comprehensive Report Generator

Processes Docker benchmark results with energy metrics and generates
detailed comparison reports in Markdown format.

Usage:
    python bench/generate_comprehensive_report.py \\
        --input bench/results/comprehensive_docker_results.json \\
        --output bench/results/COMPREHENSIVE_BENCHMARK_REPORT.md
"""

import argparse
import json
import os
from typing import Dict, List
from datetime import datetime


def load_results(input_file: str) -> Dict:
    """Load benchmark results from JSON"""
    with open(input_file, 'r') as f:
        return json.load(f)


def generate_summary_table(results: List[Dict]) -> str:
    """Generate overall summary table"""
    lines = [
        "## Overall Summary",
        "",
        "| Pattern | Profile | Precision | Cycles | Memory (MB) | Energy (µJ) | Power (mW) | GOPs/W |",
        "|---------|---------|-----------|--------|-------------|-------------|------------|--------|",
    ]
    
    for r in sorted(results, key=lambda x: (x['pattern'], x['profile'])):
        pattern = r['pattern']
        profile = r['profile']
        precision = r['precision']
        cycles = r['metrics']['performance']['cycles']
        memory_mb = r['derived']['memory_mb']
        energy = r['metrics']['energy'].get('total_uj', 0)
        power = r['metrics']['energy'].get('average_power_mw', 0)
        efficiency = r['metrics']['energy'].get('efficiency_gops_per_w', 0)
        
        lines.append(
            f"| {pattern:18s} | {profile:15s} | {precision:9s} | "
            f"{cycles:10.0f} | {memory_mb:11.2f} | {energy:11.2f} | "
            f"{power:10.2f} | {efficiency/1e6:6.1f}M |"
        )
    
    lines.append("")
    return "\n".join(lines)


def generate_best_by_objective(results: List[Dict]) -> str:
    """Find and display best configurations by each objective"""
    lines = [
        "## 🏆 Best Configurations by Objective",
        "",
    ]
    
    # Find bests
    best_energy = min(results, key=lambda x: x['metrics']['energy'].get('total_uj', float('inf')))
    best_memory = min(results, key=lambda x: x['derived']['memory_mb'])
    best_cycles = min(results, key=lambda x: x['metrics']['performance']['cycles'])
    best_efficiency = max(results, key=lambda x: x['metrics']['energy'].get('efficiency_gops_per_w', 0))
    
    lines.extend([
        f"### ⚡ Lowest Energy",
        f"- **{best_energy['pattern']} ({best_energy['precision']})** - {best_energy['profile']}",
        f"- Energy: **{best_energy['metrics']['energy']['total_uj']:.2f} µJ**",
        f"- Memory: {best_energy['derived']['memory_mb']:.2f} MB",
        f"- Cycles: {best_energy['metrics']['performance']['cycles']:.0f}",
        "",
        f"### 💾 Lowest Memory",
        f"- **{best_memory['pattern']} ({best_memory['precision']})** - {best_memory['profile']}",
        f"- Memory: **{best_memory['derived']['memory_mb']:.2f} MB**",
        f"- Energy: {best_memory['metrics']['energy']['total_uj']:.2f} µJ",
        f"- Cycles: {best_memory['metrics']['performance']['cycles']:.0f}",
        "",
        f"### 🚀 Fastest (Lowest Cycles)",
        f"- **{best_cycles['pattern']} ({best_cycles['precision']})** - {best_cycles['profile']}",
        f"- Cycles: **{best_cycles['metrics']['performance']['cycles']:.0f}**",
        f"- Energy: {best_cycles['metrics']['energy']['total_uj']:.2f} µJ",
        f"- Memory: {best_cycles['derived']['memory_mb']:.2f} MB",
        "",
        f"### 📊 Best Efficiency (GOPs/W)",
        f"- **{best_efficiency['pattern']} ({best_efficiency['precision']})** - {best_efficiency['profile']}",
        f"- Efficiency: **{best_efficiency['metrics']['energy']['efficiency_gops_per_w']/1e6:.1f}M GOPs/W**",
        f"- Energy: {best_efficiency['metrics']['energy']['total_uj']:.2f} µJ",
        f"- Memory: {best_efficiency['derived']['memory_mb']:.2f} MB",
        "",
    ])
    
    return "\n".join(lines)


def generate_pattern_breakdown(results: List[Dict]) -> str:
    """Generate per-pattern analysis"""
    lines = [
        "## 📋 Per-Pattern Analysis",
        "",
    ]
    
    # Group by pattern
    by_pattern = {}
    for r in results:
        pattern = r['pattern']
        if pattern not in by_pattern:
            by_pattern[pattern] = []
        by_pattern[pattern].append(r)
    
    for pattern, pattern_results in sorted(by_pattern.items()):
        lines.extend([
            f"### Pattern: `{pattern}`",
            "",
            "| Profile | Precision | Cycles | Memory (MB) | Energy (µJ) | Power (mW) | Efficiency |",
            "|---------|-----------|--------|-------------|-------------|------------|------------|",
        ])
        
        for r in sorted(pattern_results, key=lambda x: x['profile']):
            profile = r['profile']
            precision = r['precision']
            cycles = r['metrics']['performance']['cycles']
            memory_mb = r['derived']['memory_mb']
            energy = r['metrics']['energy'].get('total_uj', 0)
            power = r['metrics']['energy'].get('average_power_mw', 0)
            efficiency = r['metrics']['energy'].get('efficiency_gops_per_w', 0)
            
            lines.append(
                f"| {profile:15s} | {precision:9s} | {cycles:10.0f} | "
                f"{memory_mb:11.2f} | {energy:11.2f} | {power:10.2f} | "
                f"{efficiency/1e6:6.1f}M GOPs/W |"
            )
        
        # Find best for this pattern
        best = min(pattern_results, key=lambda x: x['metrics']['energy'].get('total_uj', float('inf')))
        lines.extend([
            "",
            f"**Best for {pattern}**: `{best['profile']}` ({best['precision']}) - "
            f"{best['metrics']['energy']['total_uj']:.2f} µJ, "
            f"{best['derived']['memory_mb']:.2f} MB",
            "",
        ])
    
    return "\n".join(lines)


def generate_recommendations(results: List[Dict]) -> str:
    """Generate use-case recommendations"""
    lines = [
        "## 🎯 Recommendations by Use Case",
        "",
    ]
    
    # Ultra Low Power (minimize energy)
    ulp_candidates = [r for r in results if r['profile'] == 'ultra_low_power']
    if ulp_candidates:
        best_ulp = min(ulp_candidates, key=lambda x: x['metrics']['energy'].get('total_uj', float('inf')))
        lines.extend([
            "### 🔋 Ultra Low Power (IoT, Battery-Powered)",
            f"**Recommended**: `{best_ulp['pattern']}` with `{best_ulp['precision']}` precision",
            "",
            f"- Energy: **{best_ulp['metrics']['energy']['total_uj']:.2f} µJ**",
            f"- Memory: {best_ulp['derived']['memory_mb']:.2f} MB",
            f"- Power: {best_ulp['metrics']['energy']['average_power_mw']:.2f} mW",
            "",
            f"**Why**: Minimizes energy consumption, ideal for battery-powered IoT devices.",
            "",
        ])
    
    # Low Power (mobile)
    lp_candidates = [r for r in results if r['profile'] == 'low_power']
    if lp_candidates:
        # Balance between energy and cycles
        best_lp = min(lp_candidates, key=lambda x: (x['metrics']['energy'].get('total_uj', float('inf')), 
                                                     x['metrics']['performance']['cycles']))
        lines.extend([
            "### 📱 Low Power (Mobile, Edge Devices)",
            f"**Recommended**: `{best_lp['pattern']}` with `{best_lp['precision']}` precision",
            "",
            f"- Energy: {best_lp['metrics']['energy']['total_uj']:.2f} µJ",
            f"- Memory: {best_lp['derived']['memory_mb']:.2f} MB",
            f"- Cycles: {best_lp['metrics']['performance']['cycles']:.0f}",
            "",
            f"**Why**: Good balance of energy efficiency and performance for mobile devices.",
            "",
        ])
    
    # Balanced (general purpose)
    balanced_candidates = [r for r in results if r['profile'] == 'balanced']
    if balanced_candidates:
        best_balanced = min(balanced_candidates, key=lambda x: x['metrics']['performance']['cycles'])
        lines.extend([
            "### ⚖️ Balanced (General Purpose)",
            f"**Recommended**: `{best_balanced['pattern']}` with `{best_balanced['precision']}` precision",
            "",
            f"- Cycles: **{best_balanced['metrics']['performance']['cycles']:.0f}** (fastest!)",
            f"- Energy: {best_balanced['metrics']['energy']['total_uj']:.2f} µJ",
            f"- Memory: {best_balanced['derived']['memory_mb']:.2f} MB",
            "",
            f"**Why**: Best latency with acceptable energy/memory trade-off.",
            "",
        ])
    
    # High Performance (servers)
    hp_candidates = [r for r in results if r['profile'] == 'high_performance']
    if hp_candidates:
        best_hp = min(hp_candidates, key=lambda x: x['metrics']['performance']['cycles'])
        lines.extend([
            "### ⚡ High Performance (Servers, Data Centers)",
            f"**Recommended**: `{best_hp['pattern']}` with `{best_hp['precision']}` precision",
            "",
            f"- Cycles: {best_hp['metrics']['performance']['cycles']:.0f}",
            f"- Memory: {best_hp['derived']['memory_mb']:.2f} MB",
            f"- Full fp32 precision (no quantization loss)",
            "",
            f"**Why**: Maximum accuracy and throughput, energy is less critical.",
            "",
        ])
    
    return "\n".join(lines)


def generate_energy_analysis(results: List[Dict]) -> str:
    """Generate energy-specific analysis"""
    lines = [
        "## ⚡ Energy Analysis",
        "",
        "### Energy Savings vs FP32",
        "",
    ]
    
    # Group by pattern, compare precisions
    by_pattern = {}
    for r in results:
        pattern = r['pattern']
        if pattern not in by_pattern:
            by_pattern[pattern] = {}
        by_pattern[pattern][r['precision']] = r
    
    lines.append("| Pattern | bf16 vs fp32 | i8 vs fp32 | i4 vs fp32 |")
    lines.append("|---------|--------------|------------|------------|")
    
    for pattern, precisions in sorted(by_pattern.items()):
        if 'fp32' not in precisions:
            continue
        
        fp32_energy = precisions['fp32']['metrics']['energy'].get('total_uj', 1)
        
        bf16_savings = ""
        if 'bf16' in precisions:
            bf16_energy = precisions['bf16']['metrics']['energy'].get('total_uj', 1)
            savings_pct = (1 - bf16_energy / fp32_energy) * 100
            bf16_savings = f"{savings_pct:.1f}% ({bf16_energy/fp32_energy:.2f}x)"
        
        i8_savings = ""
        if 'i8' in precisions:
            i8_energy = precisions['i8']['metrics']['energy'].get('total_uj', 1)
            savings_pct = (1 - i8_energy / fp32_energy) * 100
            i8_savings = f"{savings_pct:.1f}% ({i8_energy/fp32_energy:.2f}x)"
        
        i4_savings = ""
        if 'i4' in precisions:
            i4_energy = precisions['i4']['metrics']['energy'].get('total_uj', 1)
            savings_pct = (1 - i4_energy / fp32_energy) * 100
            i4_savings = f"{savings_pct:.1f}% ({i4_energy/fp32_energy:.2f}x)"
        
        lines.append(f"| {pattern:18s} | {bf16_savings:12s} | {i8_savings:10s} | {i4_savings:10s} |")
    
    lines.append("")
    return "\n".join(lines)


def generate_report(input_file: str, output_file: str):
    """Generate comprehensive markdown report"""
    
    print(f"Loading results from: {input_file}")
    data = load_results(input_file)
    results = data['results']
    metadata = data['metadata']
    
    print(f"Processing {len(results)} benchmark results...")
    
    # Generate report sections
    report_lines = [
        f"# Comprehensive Benchmark Report",
        f"",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Benchmark Date**: {metadata.get('timestamp', 'N/A')}",
        f"**Problem Size**: L={metadata['L']}, D={metadata['D']}",
        f"**Tech Node**: {metadata.get('tech_node', '7nm')}",
        f"**Total Configurations**: {metadata['total_configs']}",
        f"",
        f"---",
        f"",
    ]
    
    # Add sections
    report_lines.append(generate_summary_table(results))
    report_lines.append("")
    report_lines.append(generate_best_by_objective(results))
    report_lines.append("")
    report_lines.append(generate_recommendations(results))
    report_lines.append("")
    report_lines.append(generate_energy_analysis(results))
    report_lines.append("")
    report_lines.append(generate_pattern_breakdown(results))
    report_lines.append("")
    
    # Add footer
    report_lines.extend([
        "---",
        "",
        "## 📊 Visual Analysis",
        "",
        "Generate visual plots to explore trade-offs:",
        "",
        "```bash",
        "python bench/visualize_benchmarks.py \\",
        f"  --input {input_file} \\",
        "  --all \\",
        "  --output-dir bench/plots",
        "```",
        "",
        "**Generated Plots**:",
        "- `bench/plots/latency_vs_energy.png` - Cycles vs Energy scatter plot",
        "- `bench/plots/memory_vs_accuracy.png` - Memory vs Accuracy trade-off",
        "- `bench/plots/pattern_comparison.png` - Side-by-side bar charts",
        "- `bench/plots/pareto_frontier.png` - Pareto-optimal configurations",
        "- `bench/plots/efficiency_radar.png` - Multi-dimensional efficiency",
        "",
        "---",
        "",
        "## How to Use These Results",
        "",
        "1. **For IoT/Battery**: Choose `ultra_low_power` profile with i4 precision",
        "2. **For Mobile**: Choose `low_power` profile with i8 precision",
        "3. **For Real-Time**: Choose `balanced` profile with bf16 precision",
        "4. **For Accuracy**: Choose `high_performance` profile with fp32 precision",
        "",
        "Use the variant selector to find optimal configs for your constraints:",
        "",
        "```bash",
        f"python bench/variant_selector.py --max-memory-mb 1.0 --L {metadata['L']} --D {metadata['D']}",
        "```",
        "",
        "---",
        "",
        f"*Report generated by `bench/generate_comprehensive_report.py`*",
    ])
    
    # Write report
    report_text = "\n".join(report_lines)
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(report_text)
    
    print(f"✅ Report saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Generate comprehensive benchmark report')
    parser.add_argument('--input', required=True, help='Input JSON results file')
    parser.add_argument('--output', default='bench/results/COMPREHENSIVE_BENCHMARK_REPORT.md',
                       help='Output markdown file')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"❌ Error: Input file not found: {args.input}")
        return 1
    
    generate_report(args.input, args.output)
    
    print("\n✅ Report generation complete!")
    print(f"\nView the report:")
    print(f"  cat {args.output}")
    
    return 0


if __name__ == "__main__":
    exit(main())

