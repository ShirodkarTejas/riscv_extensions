#!/usr/bin/env python3
"""
Comprehensive Benchmark Tool

Benchmarks all precision modes (fp32, bf16, i8, i4) across all window sizes
to find the truly optimal configurations for each power profile.

Usage:
    python bench/comprehensive_benchmark.py \\
        --pattern sliding_window \\
        --backend rvv \\
        --L 128 \\
        --output bench/results/comprehensive_benchmark.csv
"""

import argparse
import json
import os
import subprocess
import sys
import time
from typing import List, Dict, Tuple

# All precision modes to test
PRECISIONS = ["fp32", "bf16", "i8", "i4"]

# Configuration profiles to test (pattern-specific)
SLIDING_WINDOW_PROFILES = {
    "tiny": {"window_size": 8, "global_tokens": 2, "tile_rows": 1},
    "small": {"window_size": 16, "global_tokens": 4, "tile_rows": 2},
    "medium": {"window_size": 32, "global_tokens": 8, "tile_rows": 4},
    "large": {"window_size": 64, "global_tokens": 16, "tile_rows": 8},
    "xlarge": {"window_size": 128, "global_tokens": 32, "tile_rows": 16},
}

BLOCK_TOPK_PROFILES = {
    "tiny": {"block_size": 16, "keep_ratio": 0.08, "global_tokens": 4, "tile_rows": 1},
    "small": {"block_size": 32, "keep_ratio": 0.10, "global_tokens": 8, "tile_rows": 2},
    "medium": {"block_size": 64, "keep_ratio": 0.12, "global_tokens": 16, "tile_rows": 4},
    "large": {"block_size": 128, "keep_ratio": 0.16, "global_tokens": 32, "tile_rows": 8},
    "xlarge": {"block_size": 256, "keep_ratio": 0.24, "global_tokens": 64, "tile_rows": 16},
}

NM_STRUCTURED_PROFILES = {
    "2_4": {"nm_n": 2, "nm_m": 4},   # 50% sparsity
    "2_8": {"nm_n": 2, "nm_m": 8},   # 75% sparsity
    "4_8": {"nm_n": 4, "nm_m": 8},   # 50% sparsity
    "1_4": {"nm_n": 1, "nm_m": 4},   # 75% sparsity
    "4_16": {"nm_n": 4, "nm_m": 16}, # 75% sparsity
}

# Default quantization scales
DEFAULT_SCALES = {
    "i8": {"scale_q_x1000": 50, "scale_k_x1000": 50, "scale_v_x1000": 50},
    "i4": {"scale_q_x1000": 125, "scale_k_x1000": 125, "scale_v_x1000": 125},
}


def run_benchmark(pattern: str, backend: str, L: int, D: int, 
                 config: Dict, precision: str,
                 scales: Dict = None) -> Dict:
    """Run a single benchmark configuration"""
    
    profile_key = config.get("block_size", config.get("window_size", 0))
    output_file = f"/tmp/bench_{pattern}_{profile_key}_{precision}.json"
    
    # Build QEMU command directly
    build_dir = "build/rvv-riscv64"
    exe = os.path.join(build_dir, "sattn_rvv_runner")
    
    qemu_cmd = [
        "qemu-riscv64", "-L", "/usr/riscv64-linux-gnu",
        "-cpu", "rv64,v=true,vlen=128,elen=64",
        exe,
        "--spec", pattern,
        "--L", str(L),
        "--D", str(D),
    ]
    
    # Add pattern-specific parameters
    if pattern == "sliding_window":
        qemu_cmd += ["--window", str(config["window_size"])]
        if config.get("global_tokens"):
            qemu_cmd += ["--global_tokens", str(config["global_tokens"])]
    elif pattern == "block_topk" or pattern == "block_local_global":
        qemu_cmd += ["--block_size", str(config["block_size"])]
        qemu_cmd += ["--keep_x1000", str(int(config["keep_ratio"] * 1000))]
        if config.get("global_tokens"):
            qemu_cmd += ["--global_tokens", str(config["global_tokens"])]
    elif pattern == "nm_structured":
        qemu_cmd += ["--nm_n", str(config["nm_n"])]
        qemu_cmd += ["--nm_m", str(config["nm_m"])]
    
    if config.get("tile_rows"):
        qemu_cmd += ["--tile_rows", str(config["tile_rows"])]
    
    if precision != "fp32":
        qemu_cmd += ["--precision", precision]
    
    if precision in ["i8", "i4"] and scales:
        for key, val in scales.items():
            qemu_cmd += [f"--{key}", str(val)]
    
    try:
        # Run 3 iterations and average
        latencies = []
        metrics = None
        
        for _ in range(3):
            start = time.perf_counter()
            result = subprocess.run(qemu_cmd, check=True, capture_output=True, 
                                   text=True, timeout=60)
            end = time.perf_counter()
            
            latencies.append((end - start) * 1000)
            
            # Parse output
            output = result.stdout + result.stderr
            for line in output.split('\n'):
                if 'rvv_bytes_read=' in line:
                    metrics = {}
                    for token in line.split():
                        if '=' in token:
                            k, v = token.split('=', 1)
                            try:
                                metrics[k] = float(v)
                            except:
                                metrics[k] = v
        
        if not metrics:
            return None
        
        result = {
            "pattern": pattern,
            "precision": precision,
            "latency_ms": sum(latencies) / len(latencies),
            "cycles": metrics.get("rvv_cycles", 0),
            "bytes_read": metrics.get("rvv_bytes_read", 0),
            "bytes_written": metrics.get("bytes_written", 0),
            "mac_ops": metrics.get("mac_flops", 0),
            "checksum": metrics.get("checksum", 0),
        }
        # Add pattern-specific config details
        result.update(config)
        return result
    except Exception as e:
        print(f"    ❌ Failed: {e}")
        return None


def calculate_metrics(result: Dict, baseline: Dict) -> Dict:
    """Calculate derived metrics and comparisons"""
    if not result or not baseline:
        return result
    
    result["memory_mb"] = result["bytes_read"] / 1024 / 1024
    result["memory_savings_vs_fp32"] = baseline["bytes_read"] / result["bytes_read"] if result["bytes_read"] > 0 else 0
    result["latency_vs_fp32"] = result["latency_ms"] / baseline["latency_ms"] if baseline["latency_ms"] > 0 else 0
    result["efficiency_score"] = (result["mac_ops"] / 1e6) / (result["cycles"] / 1e6 + result["memory_mb"] + 0.001)
    
    return result


def generate_comprehensive_report(results: List[Dict], output_md: str, pattern: str):
    """Generate markdown report with all results"""
    
    # Group by configuration size (window_size or block_size or nm_m)
    by_config = {}
    if pattern == "sliding_window":
        config_key = "window_size"
        config_label = "Window"
    elif pattern == "nm_structured":
        config_key = "nm_m"
        config_label = "N:M"
    else:
        config_key = "block_size"
        config_label = "Block"
    
    for r in results:
        if not r:
            continue
        cfg_val = r.get(config_key, 0)
        if cfg_val not in by_config:
            by_config[cfg_val] = []
        by_config[cfg_val].append(r)
    md = f"# Comprehensive Benchmark: {pattern} - All Precision × All {config_label} Sizes\n\n"
    md += f"**Total configurations tested**: {len([r for r in results if r])}\n\n"
    
    # Overall comparison table
    md += "## Overall Comparison\n\n"
    md += f"| {config_label} | Precision | Latency (ms) | Memory (MB) | Memory Savings | Cycles (M) | MAC Ops (M) | Efficiency |\n"
    md += "|--------|-----------|--------------|-------------|----------------|------------|-------------|------------|\n"
    
    for cfg_val in sorted(by_config.keys()):
        for r in sorted(by_config[cfg_val], key=lambda x: ["fp32", "bf16", "i8", "i4"].index(x["precision"])):
            md += f"| {cfg_val:3d} | {r['precision']:5s} | {r['latency_ms']:8.2f} | {r['memory_mb']:8.2f} | "
            md += f"{r.get('memory_savings_vs_fp32', 1.0):10.2f}x | {r['cycles']/1e6:7.1f} | "
            md += f"{r['mac_ops']/1e6:8.2f} | {r.get('efficiency_score', 0):8.3f} |\n"
    
    # Best configurations by objective
    md += "\n## Best Configurations by Objective\n\n"
    
    valid_results = [r for r in results if r and r["bytes_read"] > 0]
    
    if valid_results:
        best_latency = min(valid_results, key=lambda r: r["latency_ms"])
        best_memory = min(valid_results, key=lambda r: r["bytes_read"])
        best_efficiency = max(valid_results, key=lambda r: r.get("efficiency_score", 0))
        
        cfg_val_lat = best_latency.get(config_key, 0)
        cfg_val_mem = best_memory.get(config_key, 0)
        cfg_val_eff = best_efficiency.get(config_key, 0)
        
        md += f"### Lowest Latency\n"
        md += f"- **Config**: {config_key}={cfg_val_lat}, precision={best_latency['precision']}\n"
        md += f"- **Latency**: {best_latency['latency_ms']:.2f} ms\n"
        md += f"- **Memory**: {best_latency['memory_mb']:.2f} MB\n\n"
        
        md += f"### Lowest Memory (Best for Power)\n"
        md += f"- **Config**: {config_key}={cfg_val_mem}, precision={best_memory['precision']}\n"
        md += f"- **Memory**: {best_memory['memory_mb']:.2f} MB\n"
        md += f"- **Latency**: {best_memory['latency_ms']:.2f} ms\n"
        md += f"- **Savings**: {best_memory.get('memory_savings_vs_fp32', 1):.1f}x vs fp32\n\n"
        
        md += f"### Best Efficiency (Compute/Memory)\n"
        md += f"- **Config**: {config_key}={cfg_val_eff}, precision={best_efficiency['precision']}\n"
        md += f"- **Efficiency**: {best_efficiency.get('efficiency_score', 0):.3f}\n"
        md += f"- **Memory**: {best_efficiency['memory_mb']:.2f} MB\n"
        md += f"- **Latency**: {best_efficiency['latency_ms']:.2f} ms\n\n"
    
    # Per-config-size analysis
    md += f"## Per-{config_label}-Size Analysis\n\n"
    
    for cfg_val in sorted(by_config.keys()):
        md += f"### {config_label} Size = {cfg_val}\n\n"
        md += "| Precision | Latency (ms) | Memory (MB) | Savings vs fp32 | Cycles (M) |\n"
        md += "|-----------|--------------|-------------|-----------------|------------|\n"
        
        configs = sorted(by_config[cfg_val], key=lambda x: ["fp32", "bf16", "i8", "i4"].index(x["precision"]))
        fp32_config = next((c for c in configs if c["precision"] == "fp32"), None)
        
        for r in configs:
            savings = fp32_config["bytes_read"] / r["bytes_read"] if fp32_config and r["bytes_read"] > 0 else 1.0
            md += f"| {r['precision']:9s} | {r['latency_ms']:12.2f} | {r['memory_mb']:11.2f} | "
            md += f"{savings:15.2f}x | {r['cycles']/1e6:10.1f} |\n"
        
        md += "\n"
    
    # Recommendations
    md += "## Recommendations\n\n"
    md += "Based on the data:\n\n"
    
    if valid_results:
        # Find best for each use case
        i4_results = [r for r in valid_results if r["precision"] == "i4"]
        i8_results = [r for r in valid_results if r["precision"] == "i8"]
        bf16_results = [r for r in valid_results if r["precision"] == "bf16"]
        
        if i4_results:
            best_i4 = min(i4_results, key=lambda r: r["bytes_read"])
            cfg_i4 = best_i4.get(config_key, 0)
            md += f"- **Ultra Low Power (IoT)**: i4, {config_key}={cfg_i4} "
            md += f"(Memory: {best_i4['memory_mb']:.2f} MB, Latency: {best_i4['latency_ms']:.1f} ms)\n"
        
        if i8_results:
            best_i8 = min(i8_results, key=lambda r: r["bytes_read"])
            cfg_i8 = best_i8.get(config_key, 0)
            md += f"- **Low Power (Mobile)**: i8, {config_key}={cfg_i8} "
            md += f"(Memory: {best_i8['memory_mb']:.2f} MB, Latency: {best_i8['latency_ms']:.1f} ms)\n"
        
        if bf16_results:
            # Find bf16 with best latency
            best_bf16 = min(bf16_results, key=lambda r: r["latency_ms"])
            cfg_bf16 = best_bf16.get(config_key, 0)
            md += f"- **Balanced (Mobile/Edge)**: bf16, {config_key}={cfg_bf16} "
            md += f"(Memory: {best_bf16['memory_mb']:.2f} MB, Latency: {best_bf16['latency_ms']:.1f} ms)\n"
        
        # Best fp32 for performance
        fp32_results = [r for r in valid_results if r["precision"] == "fp32"]
        if fp32_results:
            best_fp32_lat = min(fp32_results, key=lambda r: r["latency_ms"])
            cfg_fp32 = best_fp32_lat.get(config_key, 0)
            md += f"- **High Performance**: fp32, {config_key}={cfg_fp32} "
            md += f"(Latency: {best_fp32_lat['latency_ms']:.1f} ms)\n"
    
    return md


def main():
    parser = argparse.ArgumentParser(description='Comprehensive benchmark: all precision × all configurations')
    parser.add_argument('--pattern', default='sliding_window', 
                       choices=['sliding_window', 'block_topk', 'nm_structured'],
                       help='Sparse attention pattern')
    parser.add_argument('--backend', default='rvv',
                       choices=['rvv'],
                       help='Execution backend')
    parser.add_argument('--L', type=int, default=128, help='Sequence length')
    parser.add_argument('--D', type=int, default=32, help='Head dimension')
    parser.add_argument('--output-csv', default='bench/results/comprehensive_benchmark.csv',
                       help='Output CSV file')
    parser.add_argument('--output-md', default='bench/results/comprehensive_benchmark.md',
                       help='Output markdown report')
    
    args = parser.parse_args()
    
    # Select profiles based on pattern
    if args.pattern == "sliding_window":
        PROFILES = SLIDING_WINDOW_PROFILES
        config_label = "Window"
        config_key = "window_size"
    elif args.pattern == "block_topk" or args.pattern == "block_local_global":
        PROFILES = BLOCK_TOPK_PROFILES
        config_label = "Block"
        config_key = "block_size"
    elif args.pattern == "nm_structured":
        PROFILES = NM_STRUCTURED_PROFILES
        config_label = "N:M"
        config_key = "nm_m"
    else:
        print(f"Error: Unknown pattern {args.pattern}")
        return 1
    
    print("=" * 80)
    print(f"Comprehensive Benchmark: {args.pattern} - All Precision × All {config_label} Sizes")
    print("=" * 80)
    print(f"Pattern: {args.pattern}")
    print(f"Problem size: L={args.L}, D={args.D}")
    print(f"Testing {len(PRECISIONS)} precisions × {len(PROFILES)} {config_label.lower()} sizes")
    print(f"Total: {len(PRECISIONS) * len(PROFILES)} configurations")
    print("=" * 80)
    print()
    
    results = []
    total = len(PROFILES) * len(PRECISIONS)
    current = 0
    
    # Baseline results for comparison (fp32)
    baselines = {}
    
    for profile_name, profile_config in PROFILES.items():
        config_val = profile_config.get(config_key, 0)
        
        print(f"\n### {config_label} Size: {config_val} ({profile_name}) ###")
        
        for precision in PRECISIONS:
            current += 1
            print(f"[{current}/{total}] Testing {precision:5s}...", end=" ", flush=True)
            
            scales = DEFAULT_SCALES.get(precision, {})
            
            result = run_benchmark(
                args.pattern, args.backend, args.L, args.D,
                profile_config, precision, scales
            )
            
            if result:
                print(f"✅ Latency: {result['latency_ms']:.1f}ms, Memory: {result['bytes_read']/1024/1024:.2f}MB")
                
                # Store baseline for this config size
                if precision == "fp32":
                    baselines[config_val] = result
                
                # Calculate comparison metrics
                if config_val in baselines:
                    result = calculate_metrics(result, baselines[config_val])
                
                results.append(result)
            else:
                print("❌ Failed")
                results.append(None)
    
    print()
    print("=" * 80)
    print("Benchmark Complete!")
    print("=" * 80)
    print()
    
    # Write CSV
    if args.output_csv:
        os.makedirs(os.path.dirname(args.output_csv) if os.path.dirname(args.output_csv) else '.', exist_ok=True)
        with open(args.output_csv, 'w') as f:
            # Write header with config_key
            f.write(f"profile,{config_key},precision,latency_ms,cycles,bytes_read,bytes_written,mac_ops,memory_mb,memory_savings_vs_fp32,latency_vs_fp32,efficiency_score\n")
            for r in results:
                if not r:
                    continue
                # Find profile name
                cfg_val = r.get(config_key, 0)
                profile = next((name for name, cfg in PROFILES.items() if cfg.get(config_key) == cfg_val), "unknown")
                f.write(f"{profile},{cfg_val},{r['precision']},{r['latency_ms']:.3f},{r['cycles']:.0f},")
                f.write(f"{r['bytes_read']:.0f},{r['bytes_written']:.0f},{r['mac_ops']:.0f},")
                f.write(f"{r.get('memory_mb', 0):.3f},{r.get('memory_savings_vs_fp32', 1):.3f},")
                f.write(f"{r.get('latency_vs_fp32', 1):.3f},{r.get('efficiency_score', 0):.6f}\n")
        print(f"✅ CSV saved to: {args.output_csv}")
    
    # Write markdown report
    if args.output_md:
        os.makedirs(os.path.dirname(args.output_md) if os.path.dirname(args.output_md) else '.', exist_ok=True)
        report = generate_comprehensive_report(results, args.output_md, args.pattern)
        with open(args.output_md, 'w') as f:
            f.write(report)
        print(f"✅ Markdown report saved to: {args.output_md}")
        print()
        print(report)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

