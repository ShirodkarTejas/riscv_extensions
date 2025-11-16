#!/usr/bin/env python3
"""
Comprehensive Docker Benchmark Runner

Runs all sparse attention patterns × all optimization profiles × all precisions
in Docker (with QEMU) and collects comprehensive metrics including energy.

Usage:
    # Run inside Docker container:
    python bench/run_comprehensive_docker_benchmark.py --L 128 --D 32
    
    # Or from host (will exec into running container):
    docker exec -it <container> python bench/run_comprehensive_docker_benchmark.py --L 128 --D 32
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from typing import Dict, List

# Pattern configurations
PATTERNS_AND_PROFILES = {
    "sliding_window": {
        "ultra_low_power": {"window_size": 8, "global_tokens": 2, "precision": "i4", 
                           "scale_q_x1000": 125, "scale_k_x1000": 125, "scale_v_x1000": 125},
        "low_power": {"window_size": 8, "global_tokens": 2, "precision": "i8",
                     "scale_q_x1000": 50, "scale_k_x1000": 50, "scale_v_x1000": 50},
        "balanced": {"window_size": 16, "global_tokens": 4, "precision": "bf16"},
        "high_performance": {"window_size": 32, "global_tokens": 8, "precision": "fp32"},
    },
    "block_local_global": {
        "ultra_low_power": {"block_size": 16, "keep_ratio": 0.08, "global_tokens": 4, "precision": "i4",
                           "scale_q_x1000": 125, "scale_k_x1000": 125, "scale_v_x1000": 125},
        "low_power": {"block_size": 16, "keep_ratio": 0.10, "global_tokens": 4, "precision": "i8",
                     "scale_q_x1000": 50, "scale_k_x1000": 50, "scale_v_x1000": 50},
        "balanced": {"block_size": 32, "keep_ratio": 0.12, "global_tokens": 8, "precision": "bf16"},
        "high_performance": {"block_size": 64, "keep_ratio": 0.16, "global_tokens": 16, "precision": "fp32"},
    },
    "nm_structured": {
        "ultra_low_power": {"nm_n": 2, "nm_m": 4, "precision": "i4",
                           "scale_q_x1000": 125, "scale_k_x1000": 125, "scale_v_x1000": 125},
        "low_power": {"nm_n": 2, "nm_m": 4, "precision": "i8",
                     "scale_q_x1000": 50, "scale_k_x1000": 50, "scale_v_x1000": 50},
        "balanced": {"nm_n": 2, "nm_m": 4, "precision": "bf16"},
        "high_performance": {"nm_n": 2, "nm_m": 4, "precision": "fp32"},
    },
    "lsh": {
        "ultra_low_power": {"buckets": 8, "precision": "i4",
                           "scale_q_x1000": 125, "scale_k_x1000": 125, "scale_v_x1000": 125},
        "low_power": {"buckets": 8, "precision": "i8",
                     "scale_q_x1000": 50, "scale_k_x1000": 50, "scale_v_x1000": 50},
        "balanced": {"buckets": 16, "precision": "bf16"},
        "high_performance": {"buckets": 16, "precision": "fp32"},
    },
    "landmark": {
        "ultra_low_power": {"num_landmarks": 8, "precision": "i4",
                           "scale_q_x1000": 125, "scale_k_x1000": 125, "scale_v_x1000": 125},
        "low_power": {"num_landmarks": 8, "precision": "i8",
                     "scale_q_x1000": 50, "scale_k_x1000": 50, "scale_v_x1000": 50},
        "balanced": {"num_landmarks": 16, "precision": "bf16"},
        "high_performance": {"num_landmarks": 16, "precision": "fp32"},
    },
}


def ensure_build(build_dir: str = "build/rvv-riscv64"):
    """Ensure RVV backend is built"""
    exe = os.path.join(build_dir, "sattn_rvv_runner")
    if os.path.exists(exe):
        print(f"✅ Found built executable: {exe}")
        return exe
    
    print(f"Building RVV backend...")
    os.makedirs(build_dir, exist_ok=True)
    
    # Run cmake
    subprocess.check_call([
        "cmake", "-S", "backends/rvv", "-B", build_dir,
        "-DCMAKE_TOOLCHAIN_FILE=cmake/riscv64-linux-gnu.cmake",
        "-DCMAKE_BUILD_TYPE=Release"
    ])
    
    # Build
    subprocess.check_call(["cmake", "--build", build_dir, "-j"])
    
    if not os.path.exists(exe):
        raise FileNotFoundError(f"Build failed: {exe} not found")
    
    print(f"✅ Built: {exe}")
    return exe


def run_single_benchmark(
    exe: str,
    pattern: str,
    profile: str,
    config: Dict,
    L: int,
    D: int,
    iterations: int = 3
) -> Dict:
    """Run a single benchmark configuration"""
    
    precision = config.get("precision", "fp32")
    
    # Build QEMU command
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
        if "global_tokens" in config:
            qemu_cmd += ["--global_tokens", str(config["global_tokens"])]
    elif pattern == "block_local_global":
        qemu_cmd += ["--block_size", str(config["block_size"])]
        qemu_cmd += ["--keep_x1000", str(int(config["keep_ratio"] * 1000))]
        if "global_tokens" in config:
            qemu_cmd += ["--global_tokens", str(config["global_tokens"])]
    elif pattern == "nm_structured":
        qemu_cmd += ["--nm_n", str(config["nm_n"])]
        qemu_cmd += ["--nm_m", str(config["nm_m"])]
    elif pattern == "lsh":
        qemu_cmd += ["--buckets", str(config.get("buckets", 8))]
    elif pattern == "landmark":
        qemu_cmd += ["--num_landmarks", str(config.get("num_landmarks", 8))]
    
    # Add precision if not fp32
    if precision != "fp32":
        qemu_cmd += ["--precision", precision]
    
    # Add quantization scales if needed
    if precision in ["i8", "i4"]:
        qemu_cmd += ["--scale_q_x1000", str(config.get("scale_q_x1000", 50))]
        qemu_cmd += ["--scale_k_x1000", str(config.get("scale_k_x1000", 50))]
        qemu_cmd += ["--scale_v_x1000", str(config.get("scale_v_x1000", 50))]
    
    print(f"  Running: {pattern} / {profile} ({precision})")
    
    # Run iterations
    results = []
    for i in range(iterations):
        try:
            result = subprocess.run(
                qemu_cmd, 
                capture_output=True, 
                text=True, 
                timeout=120,
                check=True
            )
            
            # Parse output
            output = result.stdout + result.stderr
            metrics = parse_metrics(output)
            results.append(metrics)
            
        except subprocess.TimeoutExpired:
            print(f"    ⚠️  Iteration {i+1} timed out")
        except subprocess.CalledProcessError as e:
            print(f"    ⚠️  Iteration {i+1} failed: {e}")
    
    if not results:
        return None
    
    # Average results
    avg_metrics = {
        "rvv_cycles": sum(r.get("rvv_cycles", 0) for r in results) / len(results),
        "rvv_bytes_read": sum(r.get("rvv_bytes_read", 0) for r in results) / len(results),
        "bytes_written": sum(r.get("bytes_written", 0) for r in results) / len(results),
        "mac_flops": sum(r.get("mac_flops", 0) for r in results) / len(results),
        "checksum": results[-1].get("checksum", 0),
    }
    
    print(f"    ✅ cycles={avg_metrics['rvv_cycles']:.0f}, memory={avg_metrics['rvv_bytes_read']/1024:.1f}KB")
    
    return avg_metrics


def parse_metrics(output: str) -> Dict:
    """Parse metrics from RVV runner output"""
    metrics = {}
    for line in output.split('\n'):
        for token in line.split():
            if '=' in token:
                key, value = token.split('=', 1)
                try:
                    metrics[key] = float(value)
                except ValueError:
                    metrics[key] = value
    return metrics


def compute_energy_metrics(raw_metrics: Dict, precision: str, tech_node: str = "7nm") -> Dict:
    """Compute energy metrics from raw counters"""
    
    # Import energy estimator
    sys.path.insert(0, os.path.dirname(__file__))
    try:
        from energy_estimator import EnergyEstimator
    except ImportError:
        print("⚠️  Energy estimator not available")
        return {}
    
    estimator = EnergyEstimator(tech_node=tech_node, cache_model="optimistic")
    
    energy = estimator.estimate(
        cycles=int(raw_metrics.get("rvv_cycles", 0)),
        mac_ops=int(raw_metrics.get("mac_flops", 0)),
        bytes_read=int(raw_metrics.get("rvv_bytes_read", 0)),
        bytes_written=int(raw_metrics.get("bytes_written", 0)),
        precision=precision
    )
    
    return {
        "total_uj": energy.total_energy_uj,
        "compute_uj": energy.compute_energy_uj,
        "memory_read_uj": energy.memory_read_energy_uj,
        "memory_write_uj": energy.memory_write_energy_uj,
        "average_power_mw": energy.average_power_mw,
        "efficiency_gops_per_w": energy.energy_efficiency_gops_per_w,
    }


def compute_derived_metrics(raw_metrics: Dict, L: int, D: int) -> Dict:
    """Compute derived metrics"""
    bytes_read = raw_metrics.get("rvv_bytes_read", 0)
    bytes_written = raw_metrics.get("bytes_written", 0)
    
    return {
        "memory_mb": (bytes_read + bytes_written) / (1024 * 1024),
        "bytes_per_token": (bytes_read + bytes_written) / L if L > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser(description='Run comprehensive benchmarks in Docker')
    parser.add_argument('--L', type=int, default=128, help='Sequence length')
    parser.add_argument('--D', type=int, default=32, help='Head dimension')
    parser.add_argument('--tech-node', default='7nm', choices=['7nm', '5nm', '3nm'],
                       help='Technology node for energy estimation')
    parser.add_argument('--iterations', type=int, default=3, help='Iterations per config')
    parser.add_argument('--output', default='bench/results/comprehensive_docker_results.json',
                       help='Output JSON file')
    parser.add_argument('--patterns', nargs='+', 
                       choices=list(PATTERNS_AND_PROFILES.keys()) + ['all'],
                       default=['all'],
                       help='Patterns to benchmark (default: all)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("COMPREHENSIVE DOCKER BENCHMARK")
    print("=" * 80)
    print(f"Problem size: L={args.L}, D={args.D}")
    print(f"Tech node: {args.tech_node}")
    print(f"Iterations: {args.iterations}")
    print("=" * 80)
    print()
    
    # Ensure build
    exe = ensure_build()
    
    # Select patterns
    if 'all' in args.patterns:
        patterns_to_run = list(PATTERNS_AND_PROFILES.keys())
    else:
        patterns_to_run = args.patterns
    
    # Count total configurations
    total_configs = sum(len(PATTERNS_AND_PROFILES[p]) for p in patterns_to_run)
    print(f"Running {total_configs} configurations...")
    print()
    
    # Run benchmarks
    all_results = []
    completed = 0
    
    for pattern in patterns_to_run:
        print(f"\n{'='*80}")
        print(f"Pattern: {pattern}")
        print(f"{'='*80}")
        
        profiles = PATTERNS_AND_PROFILES[pattern]
        
        for profile, config in profiles.items():
            completed += 1
            precision = config.get("precision", "fp32")
            
            print(f"\n[{completed}/{total_configs}] {pattern} / {profile} ({precision})")
            
            # Run benchmark
            raw_metrics = run_single_benchmark(
                exe, pattern, profile, config, 
                args.L, args.D, args.iterations
            )
            
            if raw_metrics is None:
                print(f"  ❌ Failed")
                continue
            
            # Compute energy metrics
            energy_metrics = compute_energy_metrics(raw_metrics, precision, args.tech_node)
            
            # Compute derived metrics
            derived_metrics = compute_derived_metrics(raw_metrics, args.L, args.D)
            
            # Build result
            result = {
                "benchmark_id": f"{pattern}_{profile}_{precision}_{args.L}x{args.D}",
                "timestamp": datetime.now().isoformat(),
                "pattern": pattern,
                "profile": profile,
                "precision": precision,
                "configuration": {
                    "L": args.L,
                    "D": args.D,
                    **config
                },
                "metrics": {
                    "performance": {
                        "cycles": raw_metrics.get("rvv_cycles", 0),
                    },
                    "memory": {
                        "bytes_read": int(raw_metrics.get("rvv_bytes_read", 0)),
                        "bytes_written": int(raw_metrics.get("bytes_written", 0)),
                    },
                    "compute": {
                        "mac_ops": int(raw_metrics.get("mac_flops", 0)),
                    },
                    "energy": energy_metrics,
                    "accuracy": {
                        "checksum": raw_metrics.get("checksum", 0),
                    },
                },
                "derived": derived_metrics,
            }
            
            all_results.append(result)
            
            # Print summary
            print(f"  Energy: {energy_metrics.get('total_uj', 0):.2f} µJ")
            print(f"  Memory: {derived_metrics.get('memory_mb', 0):.2f} MB")
            print(f"  Power: {energy_metrics.get('average_power_mw', 0):.2f} mW")
    
    # Save results
    print(f"\n{'='*80}")
    print(f"BENCHMARK COMPLETE!")
    print(f"{'='*80}")
    print(f"Completed: {len(all_results)}/{total_configs} configurations")
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump({
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "L": args.L,
                "D": args.D,
                "tech_node": args.tech_node,
                "total_configs": len(all_results),
            },
            "results": all_results,
        }, f, indent=2)
    
    print(f"\n✅ Results saved to: {args.output}")
    print(f"\nNext steps:")
    print(f"  1. Generate unified comparison: python bench/create_unified_comparison.py")
    print(f"  2. Visualize: python bench/visualize_benchmarks.py --input {args.output} --all")


if __name__ == "__main__":
    main()

