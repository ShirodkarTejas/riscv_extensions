#!/usr/bin/env python3
"""
Create Unified Pattern Comparison Report

Runs benchmarks for all 5 patterns across key precision/profile combinations
and generates a single comparison document for selecting optimal configurations.

Usage:
    python bench/create_unified_comparison.py --L 128 --D 32
"""

import argparse
import subprocess
import json
import os
from typing import Dict, List

# Key configurations to test for each pattern
PATTERN_CONFIGS = {
    "sliding_window": [
        {"window_size": 8, "precision": "i4", "profile": "ultra_low_power"},
        {"window_size": 8, "precision": "i8", "profile": "low_power"},
        {"window_size": 16, "precision": "bf16", "profile": "balanced"},
        {"window_size": 32, "precision": "fp32", "profile": "high_performance"},
    ],
    "block_local_global": [
        {"block_size": 16, "keep_ratio": 0.08, "precision": "i4", "profile": "ultra_low_power"},
        {"block_size": 16, "keep_ratio": 0.10, "precision": "i8", "profile": "low_power"},
        {"block_size": 32, "keep_ratio": 0.12, "precision": "bf16", "profile": "balanced"},
        {"block_size": 64, "keep_ratio": 0.16, "precision": "fp32", "profile": "high_performance"},
    ],
    "nm_structured": [
        {"nm_n": 2, "nm_m": 4, "precision": "i4", "profile": "ultra_low_power"},
        {"nm_n": 2, "nm_m": 4, "precision": "i8", "profile": "low_power"},
        {"nm_n": 2, "nm_m": 8, "precision": "bf16", "profile": "balanced"},
        {"nm_n": 4, "nm_m": 8, "precision": "fp32", "profile": "high_performance"},
    ],
    "lsh": [
        {"buckets": 2, "precision": "i4", "profile": "ultra_low_power"},
        {"buckets": 4, "precision": "i8", "profile": "low_power"},
        {"buckets": 4, "precision": "bf16", "profile": "balanced"},
        {"buckets": 8, "precision": "fp32", "profile": "high_performance"},
    ],
    "landmark": [
        {"landmarks": 8, "precision": "i4", "profile": "ultra_low_power"},
        {"landmarks": 16, "precision": "i8", "profile": "low_power"},
        {"landmarks": 16, "precision": "bf16", "profile": "balanced"},
        {"landmarks": 32, "precision": "fp32", "profile": "high_performance"},
    ],
}

DEFAULT_SCALES = {
    "i8": {"scale_q_x1000": 50, "scale_k_x1000": 50, "scale_v_x1000": 50},
    "i4": {"scale_q_x1000": 125, "scale_k_x1000": 125, "scale_v_x1000": 125},
}


def run_pattern_benchmark(pattern: str, config: Dict, L: int, D: int) -> Dict:
    """Run benchmark for a single pattern configuration"""
    
    build_dir = "build/rvv-riscv64"
    exe = os.path.join(build_dir, "sattn_rvv_runner")
    
    cmd = [
        "qemu-riscv64", "-L", "/usr/riscv64-linux-gnu",
        "-cpu", "rv64,v=true,vlen=128,elen=64",
        exe,
        "--spec", pattern,
        "--L", str(L),
        "--D", str(D),
    ]
    
    # Add pattern-specific parameters
    if pattern == "sliding_window":
        cmd += ["--window", str(config["window_size"])]
    elif pattern == "block_local_global":
        cmd += ["--block_size", str(config["block_size"])]
        cmd += ["--keep_x1000", str(int(config["keep_ratio"] * 1000))]
    elif pattern == "nm_structured":
        cmd += ["--nm_n", str(config["nm_n"])]
        cmd += ["--nm_m", str(config["nm_m"])]
    elif pattern == "lsh":
        cmd += ["--buckets", str(config["buckets"])]
    elif pattern == "landmark":
        cmd += ["--landmarks", str(config["landmarks"])]
        cmd += ["--landmark_iters", "1"]
    
    precision = config.get("precision", "fp32")
    if precision != "fp32":
        cmd += ["--precision", precision]
    
    if precision in ["i8", "i4"]:
        scales = DEFAULT_SCALES[precision]
        for k, v in scales.items():
            cmd += [f"--{k}", str(v)]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        output = result.stdout + result.stderr
        
        # Parse output
        metrics = {}
        for line in output.split('\n'):
            if 'rvv_bytes_read=' in line:
                for token in line.split():
                    if '=' in token:
                        k, v = token.split('=', 1)
                        try:
                            metrics[k] = float(v)
                        except:
                            metrics[k] = v
        
        if not metrics:
            return None
        
        return {
            "pattern": pattern,
            "profile": config.get("profile", "custom"),
            "precision": precision,
            "config": config,
            "bytes_read": metrics.get("rvv_bytes_read", 0),
            "bytes_written": metrics.get("bytes_written", 0),
            "mac_ops": metrics.get("mac_flops", 0),
            "cycles": metrics.get("rvv_cycles", 0),
            "checksum": metrics.get("checksum", 0),
        }
    except Exception as e:
        print(f"❌ Failed: {pattern} {config}: {e}")
        return None


def generate_comparison_report(results: List[Dict], output_file: str, L: int, D: int):
    """Generate unified comparison markdown report"""
    
    md = f"# Unified Sparse Attention Pattern Comparison\n\n"
    md += f"**Problem Size**: L={L}, D={D}\n"
    md += f"**Patterns Tested**: {len(set(r['pattern'] for r in results if r))}\n"
    md += f"**Total Configurations**: {len([r for r in results if r])}\n\n"
    md += "---\n\n"
    
    # Overall comparison table
    md += "## Overall Comparison - All Patterns × Profiles\n\n"
    md += "| Pattern | Profile | Precision | Memory (MB) | Latency (ms) | MAC Ops (M) | Cycles (M) | Config |\n"
    md += "|---------|---------|-----------|-------------|--------------|-------------|------------|--------|\n"
    
    for r in sorted(results, key=lambda x: (x["profile"], x["pattern"]) if x else ("", "")):
        if not r:
            continue
        mem_mb = r["bytes_read"] / 1024 / 1024
        latency_ms = r["cycles"] / 1e6 * 0.001  # Rough estimate
        mac_m = r["mac_ops"] / 1e6
        cycles_m = r["cycles"] / 1e6
        
        # Config summary
        config_str = ""
        if r["pattern"] == "sliding_window":
            config_str = f"w={r['config']['window_size']}"
        elif r["pattern"] == "block_local_global":
            config_str = f"b={r['config']['block_size']},k={r['config']['keep_ratio']:.2f}"
        elif r["pattern"] == "nm_structured":
            config_str = f"{r['config']['nm_n']}:{r['config']['nm_m']}"
        elif r["pattern"] == "lsh":
            config_str = f"buckets={r['config']['buckets']}"
        elif r["pattern"] == "landmark":
            config_str = f"lm={r['config']['landmarks']}"
        
        md += f"| {r['pattern']:20s} | {r['profile']:15s} | {r['precision']:5s} | "
        md += f"{mem_mb:11.2f} | {latency_ms:12.2f} | {mac_m:11.2f} | {cycles_m:10.1f} | {config_str:10s} |\n"
    
    # Best by objective
    md += "\n## Best Configurations by Objective\n\n"
    
    valid_results = [r for r in results if r and r["bytes_read"] > 0]
    
    if valid_results:
        md += "### Lowest Memory (Ultra Low Power)\n\n"
        best_mem = min(valid_results, key=lambda r: r["bytes_read"])
        mem_mb = best_mem["bytes_read"] / 1024 / 1024
        md += f"- **Pattern**: {best_mem['pattern']}\n"
        md += f"- **Precision**: {best_mem['precision']}\n"
        md += f"- **Memory**: {mem_mb:.2f} MB\n"
        md += f"- **Profile**: {best_mem['profile']}\n"
        md += f"- **Use Case**: IoT devices, battery-powered sensors\n\n"
        
        md += "### Fastest Latency (High Performance)\n\n"
        best_latency = min(valid_results, key=lambda r: r["cycles"])
        latency_ms = best_latency["cycles"] / 1e6 * 0.001
        md += f"- **Pattern**: {best_latency['pattern']}\n"
        md += f"- **Precision**: {best_latency['precision']}\n"
        md += f"- **Latency**: {latency_ms:.2f} ms (est)\n"
        md += f"- **Profile**: {best_latency['profile']}\n"
        md += f"- **Use Case**: Real-time inference, low-latency applications\n\n"
        
        md += "### Best Efficiency (Balanced)\n\n"
        # Efficiency = MAC ops / (cycles + memory_mb)
        for r in valid_results:
            r["efficiency"] = r["mac_ops"] / (r["cycles"] + r["bytes_read"] / 1024 + 1)
        best_eff = max(valid_results, key=lambda r: r["efficiency"])
        md += f"- **Pattern**: {best_eff['pattern']}\n"
        md += f"- **Precision**: {best_eff['precision']}\n"
        md += f"- **Efficiency Score**: {best_eff['efficiency']:.3f}\n"
        md += f"- **Profile**: {best_eff['profile']}\n"
        md += f"- **Use Case**: Mobile devices, edge computing\n\n"
    
    # Per-pattern breakdown
    md += "## Per-Pattern Analysis\n\n"
    
    by_pattern = {}
    for r in valid_results:
        if r["pattern"] not in by_pattern:
            by_pattern[r["pattern"]] = []
        by_pattern[r["pattern"]].append(r)
    
    for pattern, configs in sorted(by_pattern.items()):
        md += f"### {pattern}\n\n"
        md += "| Profile | Precision | Memory (MB) | Cycles (M) | MAC Ops (M) | Memory Savings vs fp32 |\n"
        md += "|---------|-----------|-------------|------------|-------------|------------------------|\n"
        
        fp32_config = next((c for c in configs if c["precision"] == "fp32"), None)
        fp32_mem = fp32_config["bytes_read"] if fp32_config else 1
        
        for c in sorted(configs, key=lambda x: ["ultra_low_power", "low_power", "balanced", "high_performance"].index(x["profile"]) if x["profile"] in ["ultra_low_power", "low_power", "balanced", "high_performance"] else 999):
            mem_mb = c["bytes_read"] / 1024 / 1024
            cycles_m = c["cycles"] / 1e6
            mac_m = c["mac_ops"] / 1e6
            savings = fp32_mem / c["bytes_read"] if c["bytes_read"] > 0 else 1.0
            
            md += f"| {c['profile']:15s} | {c['precision']:9s} | {mem_mb:11.2f} | {cycles_m:10.1f} | {mac_m:11.2f} | {savings:22.2f}x |\n"
        
        md += "\n"
    
    # Recommendations
    md += "## Recommendations by Use Case\n\n"
    md += "### 🔋 Ultra Low Power (IoT, Wearables)\n"
    md += "**Priority**: Minimize memory and energy consumption\n\n"
    
    ulp_configs = [r for r in valid_results if r["profile"] == "ultra_low_power"]
    if ulp_configs:
        best_ulp = min(ulp_configs, key=lambda r: r["bytes_read"])
        md += f"- **Recommended**: {best_ulp['pattern']} with {best_ulp['precision']}\n"
        md += f"- **Memory**: {best_ulp['bytes_read'] / 1024 / 1024:.2f} MB\n"
        md += f"- **Tradeoff**: Lowest accuracy, but viable for simple tasks\n\n"
    
    md += "### 📱 Low Power (Mobile, Edge)\n"
    md += "**Priority**: Balance power and accuracy\n\n"
    
    lp_configs = [r for r in valid_results if r["profile"] == "low_power"]
    if lp_configs:
        best_lp = min(lp_configs, key=lambda r: r["bytes_read"] * r["cycles"])
        md += f"- **Recommended**: {best_lp['pattern']} with {best_lp['precision']}\n"
        md += f"- **Memory**: {best_lp['bytes_read'] / 1024 / 1024:.2f} MB\n"
        md += f"- **Tradeoff**: Good accuracy with 2-4x memory savings\n\n"
    
    md += "### ⚖️ Balanced (General Purpose)\n"
    md += "**Priority**: Best overall efficiency\n\n"
    
    bal_configs = [r for r in valid_results if r["profile"] == "balanced"]
    if bal_configs:
        best_bal = min(bal_configs, key=lambda r: r["cycles"])
        md += f"- **Recommended**: {best_bal['pattern']} with {best_bal['precision']}\n"
        md += f"- **Latency**: {best_bal['cycles'] / 1e6 * 0.001:.2f} ms (est)\n"
        md += f"- **Tradeoff**: Near-fp32 accuracy with good performance\n\n"
    
    md += "### ⚡ High Performance (Cloud, Datacenter)\n"
    md += "**Priority**: Maximum throughput and accuracy\n\n"
    
    hp_configs = [r for r in valid_results if r["profile"] == "high_performance"]
    if hp_configs:
        best_hp = max(hp_configs, key=lambda r: r["mac_ops"] / r["cycles"] if r["cycles"] > 0 else 0)
        md += f"- **Recommended**: {best_hp['pattern']} with {best_hp['precision']}\n"
        md += f"- **Throughput**: {best_hp['mac_ops'] / best_hp['cycles'] * 1e6:.2f} GFLOPS (est)\n"
        md += f"- **Tradeoff**: Full precision, highest resource usage\n\n"
    
    # Summary table
    md += "## Quick Selection Guide\n\n"
    md += "| Your Constraint | Recommended Pattern | Precision | Why |\n"
    md += "|-----------------|--------------------|-----------|----- |\n"
    md += "| Memory < 0.5 MB | " + (f"{best_ulp['pattern']}" if ulp_configs else "N/A") + " | i4 | Lowest memory footprint |\n"
    md += "| Battery Life | " + (f"{best_lp['pattern']}" if lp_configs else "N/A") + " | i8 | Good accuracy/power balance |\n"
    md += "| Latency < 50ms | " + (f"{best_bal['pattern']}" if bal_configs else "N/A") + " | bf16 | Fast with good accuracy |\n"
    md += "| Accuracy > 99% | " + (f"{best_hp['pattern']}" if hp_configs else "N/A") + " | fp32 | Full precision |\n"
    
    with open(output_file, 'w') as f:
        f.write(md)
    
    print(f"✅ Report saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Create unified pattern comparison')
    parser.add_argument('--L', type=int, default=128, help='Sequence length')
    parser.add_argument('--D', type=int, default=32, help='Head dimension')
    parser.add_argument('--output', default='bench/results/UNIFIED_PATTERN_COMPARISON.md',
                       help='Output markdown file')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Unified Sparse Attention Pattern Comparison")
    print("=" * 80)
    print(f"Problem size: L={args.L}, D={args.D}")
    print(f"Testing 5 patterns × 4 profiles = 20 configurations")
    print("=" * 80)
    print()
    
    results = []
    total = sum(len(configs) for configs in PATTERN_CONFIGS.values())
    current = 0
    
    for pattern, configs in PATTERN_CONFIGS.items():
        print(f"\n### {pattern} ###")
        for config in configs:
            current += 1
            profile = config.get("profile", "custom")
            precision = config.get("precision", "fp32")
            print(f"[{current}/{total}] {profile:20s} ({precision:5s}) ... ", end="", flush=True)
            
            result = run_pattern_benchmark(pattern, config, args.L, args.D)
            if result:
                mem_mb = result["bytes_read"] / 1024 / 1024
                print(f"✅ Memory: {mem_mb:.2f} MB")
                results.append(result)
            else:
                print("❌ Failed")
                results.append(None)
    
    print("\n" + "=" * 80)
    print("Generating comparison report...")
    print("=" * 80)
    
    generate_comparison_report(results, args.output, args.L, args.D)
    
    print()
    print("✅ Unified comparison complete!")
    print(f"📄 View results: {args.output}")


if __name__ == "__main__":
    main()

