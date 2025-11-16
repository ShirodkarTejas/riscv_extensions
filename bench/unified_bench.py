#!/usr/bin/env python3
"""
Unified Benchmark Runner for Sparse Attention Library

Supports multiple backends (CPU, GPU, RVV, Custom ISA) and variants.
Collects comprehensive metrics and outputs structured JSON.

Usage:
    python bench/unified_bench.py \\
        --pattern sliding_window \\
        --variant balanced \\
        --backend rvv \\
        --L 2048 --D 64 \\
        --output results/benchmark_001.json
"""

import argparse
import json
import time
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional
import subprocess

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import numpy as np
except ImportError:
    print("Error: NumPy is required. Install with: pip install numpy")
    sys.exit(1)


class MetricsCollector:
    """Base class for collecting metrics from different backends"""
    
    def __init__(self, backend: str):
        self.backend = backend
        self.start_time = None
        self.end_time = None
        
    def start(self):
        """Start measurement"""
        self.start_time = time.perf_counter()
        
    def stop(self):
        """Stop measurement"""
        self.end_time = time.perf_counter()
        
    def get_metrics(self) -> Dict:
        """Return all metrics in standard format"""
        latency_ms = (self.end_time - self.start_time) * 1000 if self.start_time and self.end_time else 0
        return {
            "performance": {
                "latency_ms": latency_ms,
            },
            "memory": {},
            "compute": {},
            "energy": {},
            "accuracy": {},
        }


class RVVMetricsCollector(MetricsCollector):
    """Metrics collector for RVV backend"""
    
    def __init__(self):
        super().__init__("rvv")
        self.rvv_metrics = None
        
    def parse_rvv_output(self, output: str) -> Dict:
        """Parse RVV runner output for metrics"""
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
        
    def get_metrics(self) -> Dict:
        base = super().get_metrics()
        if self.rvv_metrics:
            # Map RVV output keys to standard metric keys
            rvv_cycles = float(self.rvv_metrics.get("rvv_cycles", 0))
            rvv_bytes_read = float(self.rvv_metrics.get("rvv_bytes_read", 0))
            bytes_written = float(self.rvv_metrics.get("bytes_written", 0))
            mac_flops = float(self.rvv_metrics.get("mac_flops", 0))
            checksum = float(self.rvv_metrics.get("checksum", 0))
            
            base["performance"].update({
                "cycles": rvv_cycles,
            })
            base["memory"].update({
                "bytes_read": rvv_bytes_read,
                "bytes_written": bytes_written,
            })
            base["compute"].update({
                "mac_ops": mac_flops,
            })
            base["accuracy"].update({
                "checksum": checksum,
            })
        return base


def load_variant_config(pattern: str, variant: str) -> Dict:
    """Load configuration for a specific pattern/variant"""
    # This would load from variant_configs.py in production
    # For now, return default configurations

    # Configurations based on comprehensive benchmark data! 📊
    # See bench/results/comprehensive_benchmark.md for justification
    configs = {
        "sliding_window": {
            "ultra_low_power": {
                # i4, window=8: 0.16 MB memory (4.9x savings!), best for battery-powered IoT
                "window_size": 8, "global_tokens": 2, "tile_rows": 1,
                "precision": "i4",
                "scale_q_x1000": 125, "scale_k_x1000": 125, "scale_v_x1000": 125
            },
            "low_power": {
                # i8, window=8: 0.32 MB memory (2.4x savings), balanced for mobile
                "window_size": 8, "global_tokens": 2, "tile_rows": 1,
                "precision": "i8",
                "scale_q_x1000": 50, "scale_k_x1000": 50, "scale_v_x1000": 50
            },
            "balanced": {
                # bf16, window=16: LOWEST latency (19-22ms) + memory savings (1.2x)
                "window_size": 16, "global_tokens": 4, "tile_rows": 2,
                "precision": "bf16"
            },
            "high_performance": {
                # fp32, window=32: Full precision, good throughput
                "window_size": 32, "global_tokens": 8, "tile_rows": 4,
                "precision": "fp32"
            },
            "ultra_high_perf": {
                # fp32, window=64: Maximum throughput for large contexts
                "window_size": 64, "global_tokens": 16, "tile_rows": 8,
                "precision": "fp32"
            },
        },
        "block_topk": {
            "ultra_low_power": {
                # i4, block=16: Minimum memory for IoT
                "block_size": 16, "keep_ratio": 0.08, "global_tokens": 4,
                "precision": "i4",
                "scale_q_x1000": 125, "scale_k_x1000": 125, "scale_v_x1000": 125
            },
            "low_power": {
                # i8, block=16: Mobile-friendly
                "block_size": 16, "keep_ratio": 0.10, "global_tokens": 4,
                "precision": "i8",
                "scale_q_x1000": 50, "scale_k_x1000": 50, "scale_v_x1000": 50
            },
            "balanced": {
                # bf16, block=32: Good balance
                "block_size": 32, "keep_ratio": 0.12, "global_tokens": 8,
                "precision": "bf16"
            },
            "high_performance": {
                # fp32, block=64: Full precision
                "block_size": 64, "keep_ratio": 0.16, "global_tokens": 16,
                "precision": "fp32"
            },
            "ultra_high_perf": {
                # fp32, block=128: Maximum throughput
                "block_size": 128, "keep_ratio": 0.24, "global_tokens": 32,
                "precision": "fp32"
            },
        },
    }
    
    if pattern not in configs:
        raise ValueError(f"Unknown pattern: {pattern}")
    if variant not in configs[pattern]:
        raise ValueError(f"Unknown variant: {variant} for pattern {pattern}")
        
    return configs[pattern][variant]


def run_rvv_benchmark(pattern: str, config: Dict, B: int, H: int, L: int, D: int, 
                      warmup: int = 2, iterations: int = 5) -> Dict:
    """Run benchmark on RVV backend"""
    
    # Build command for RVV runner
    build_dir = "build/rvv-riscv64"
    exe = os.path.join(build_dir, "sattn_rvv_runner")
    
    # Ensure QEMU build exists
    if not os.path.exists(exe):
        print(f"Building RVV backend...")
        subprocess.run([
            sys.executable, "scripts/build_and_run_rvv_qemu.py",
            "--build-dir", build_dir,
            "--exe", "sattn_rvv_runner",
            "--args", f"--spec {pattern} --L 8 --D 8"
        ], check=False, capture_output=True)
    
    # Map pattern names
    spec_map = {
        "sliding_window": "sliding_window",
        "block_topk": "block_local_global",
        "nm_structured": "nm_structured",
        "lsh": "lsh",
    }
    spec = spec_map.get(pattern, pattern)
    
    # Build command
    qemu_cmd = [
        "qemu-riscv64", "-L", "/usr/riscv64-linux-gnu",
        "-cpu", "rv64,v=true,vlen=128,elen=64",
        exe,
        "--spec", spec,
        "--L", str(L),
        "--D", str(D),
    ]
    
    # Add pattern-specific parameters
    if pattern == "sliding_window":
        qemu_cmd += ["--window", str(config.get("window_size", 32))]
        qemu_cmd += ["--tile_rows", str(config.get("tile_rows", 4))]
    elif pattern == "block_topk":
        qemu_cmd += ["--block_size", str(config.get("block_size", 64))]
        qemu_cmd += ["--keep_x1000", str(int(config.get("keep_ratio", 0.12) * 1000))]
        if "global_tokens" in config:
            qemu_cmd += ["--global_tokens", str(config["global_tokens"])]
        # Only use tile_rows for fp32 (quantized variants don't have tiled implementation)
        if config.get("precision", "fp32") == "fp32" and "tile_rows" in config:
            qemu_cmd += ["--tile_rows", str(config["tile_rows"])]
    
    # Precision
    precision = config.get("precision", "fp32")
    if precision != "fp32":
        qemu_cmd += ["--precision", precision]
    
    # Quantization scales for i8/i4
    if precision in ["i8", "i4"]:
        qemu_cmd += ["--scale_q_x1000", str(config.get("scale_q_x1000", 50))]
        qemu_cmd += ["--scale_k_x1000", str(config.get("scale_k_x1000", 50))]
        qemu_cmd += ["--scale_v_x1000", str(config.get("scale_v_x1000", 50))]
    
    # Debug: print command
    print(f"Command: {' '.join(qemu_cmd)}")
    
    # Run iterations
    collector = RVVMetricsCollector()
    results = []
    
    print(f"Running {warmup} warmup iterations...")
    for _ in range(warmup):
        try:
            subprocess.run(qemu_cmd, check=True, capture_output=True, text=True, timeout=60)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            pass
    
    print(f"Running {iterations} measurement iterations...")
    for i in range(iterations):
        collector.start()
        try:
            result = subprocess.run(qemu_cmd, check=True, capture_output=True, text=True, timeout=60)
            collector.stop()
            # Parse both stdout and stderr (QEMU may output to stderr)
            combined_output = result.stdout + "\n" + result.stderr
            collector.rvv_metrics = collector.parse_rvv_output(combined_output)
            results.append(collector.get_metrics())
            checksum = collector.rvv_metrics.get('checksum', 'N/A')
            bytes_read = collector.rvv_metrics.get('rvv_bytes_read', 0)
            print(f"  Iteration {i+1}/{iterations}: checksum={checksum}, bytes_read={bytes_read}")
        except subprocess.CalledProcessError as e:
            print(f"  Iteration {i+1} failed: {e}")
            collector.stop()
        except subprocess.TimeoutExpired:
            print(f"  Iteration {i+1} timed out")
            collector.stop()
    
    if not results:
        raise RuntimeError("All benchmark iterations failed")
    
    # Aggregate results
    aggregated = {
        "performance": {
            "latency_ms": np.mean([r["performance"]["latency_ms"] for r in results]),
            "latency_std": np.std([r["performance"]["latency_ms"] for r in results]),
            "cycles": np.mean([r["performance"].get("cycles", 0) for r in results]),
        },
        "memory": {
            "bytes_read": int(np.mean([r["memory"].get("bytes_read", 0) for r in results])),
            "bytes_written": int(np.mean([r["memory"].get("bytes_written", 0) for r in results])),
        },
        "compute": {
            "mac_ops": int(np.mean([r["compute"].get("mac_ops", 0) for r in results])),
        },
        "energy": {},
        "accuracy": {
            "checksum": results[-1]["accuracy"].get("checksum", 0),
        },
    }
    
    return aggregated


def generate_benchmark_report(benchmark_data: Dict) -> str:
    """Generate a markdown summary of the benchmark"""
    md = f"""# Benchmark Report: {benchmark_data['pattern']} ({benchmark_data['variant']})

**Backend**: {benchmark_data['backend']}  
**Date**: {benchmark_data['timestamp']}  

## Configuration
- **Batch Size (B)**: {benchmark_data['configuration']['B']}
- **Heads (H)**: {benchmark_data['configuration']['H']}
- **Sequence Length (L)**: {benchmark_data['configuration']['L']}
- **Head Dimension (D)**: {benchmark_data['configuration']['D']}
- **Precision**: {benchmark_data['configuration'].get('precision', 'fp32')}

### Pattern Parameters
"""
    
    for k, v in benchmark_data['configuration'].items():
        if k not in ['B', 'H', 'L', 'D', 'precision']:
            md += f"- **{k}**: {v}\n"
    
    metrics = benchmark_data['metrics']
    
    md += f"""
## Performance
- **Latency**: {metrics['performance']['latency_ms']:.3f} ms (± {metrics['performance'].get('latency_std', 0):.3f} ms)
- **Cycles**: {metrics['performance'].get('cycles', 0):.0f}

## Memory
- **Bytes Read**: {metrics['memory'].get('bytes_read', 0):,}
- **Bytes Written**: {metrics['memory'].get('bytes_written', 0):,}

## Compute
- **MAC Operations**: {metrics['compute'].get('mac_ops', 0):,}

## Accuracy
- **Checksum**: {metrics['accuracy'].get('checksum', 'N/A')}
"""
    
    return md


def main():
    parser = argparse.ArgumentParser(description='Unified benchmark runner for sparse attention')
    parser.add_argument('--pattern', required=True, 
                        choices=['sliding_window', 'block_topk', 'nm_structured', 'lsh'],
                        help='Sparse attention pattern')
    parser.add_argument('--variant', required=True,
                        choices=['ultra_low_power', 'low_power', 'balanced', 'high_performance', 'ultra_high_perf'],
                        help='Optimization variant')
    parser.add_argument('--backend', default='rvv',
                        choices=['cpu', 'gpu', 'rvv', 'custom_isa'],
                        help='Execution backend')
    parser.add_argument('--B', type=int, default=1, help='Batch size')
    parser.add_argument('--H', type=int, default=8, help='Number of heads')
    parser.add_argument('--L', type=int, default=2048, help='Sequence length')
    parser.add_argument('--D', type=int, default=64, help='Head dimension')
    parser.add_argument('--warmup', type=int, default=2, help='Warmup iterations')
    parser.add_argument('--iterations', type=int, default=5, help='Measurement iterations')
    parser.add_argument('--output', default='', help='Output JSON file path')
    parser.add_argument('--report', default='', help='Output markdown report path')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Unified Sparse Attention Benchmark")
    print("=" * 70)
    print(f"Pattern: {args.pattern}")
    print(f"Variant: {args.variant}")
    print(f"Backend: {args.backend}")
    print(f"Problem size: B={args.B}, H={args.H}, L={args.L}, D={args.D}")
    print("=" * 70)
    print()
    
    # Load variant configuration
    try:
        config = load_variant_config(args.pattern, args.variant)
        print(f"Loaded variant configuration: {config}")
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    
    # Run benchmark
    try:
        if args.backend == 'rvv':
            metrics = run_rvv_benchmark(
                args.pattern, config, args.B, args.H, args.L, args.D,
                args.warmup, args.iterations
            )
        else:
            print(f"Error: Backend '{args.backend}' not yet implemented")
            print("Currently supported: rvv")
            return 1
    except Exception as e:
        print(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Build result structure
    benchmark_data = {
        "benchmark_id": f"{args.pattern}_{args.variant}_{args.backend}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "timestamp": datetime.now().isoformat(),
        "pattern": args.pattern,
        "variant": args.variant,
        "backend": args.backend,
        "platform": {
            "arch": "riscv64",
            "backend_type": "qemu_rvv",
        },
        "configuration": {
            "B": args.B,
            "H": args.H,
            "L": args.L,
            "D": args.D,
            **config,
        },
        "metrics": metrics,
    }
    
    # Print summary
    print()
    print("=" * 70)
    print("Benchmark Results")
    print("=" * 70)
    print(f"Latency: {metrics['performance']['latency_ms']:.3f} ms")
    print(f"Cycles: {metrics['performance'].get('cycles', 0):.0f}")
    print(f"Memory Read: {metrics['memory'].get('bytes_read', 0):,} bytes")
    print(f"Memory Written: {metrics['memory'].get('bytes_written', 0):,} bytes")
    print(f"MAC Operations: {metrics['compute'].get('mac_ops', 0):,}")
    print("=" * 70)
    
    # Save JSON output
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(benchmark_data, f, indent=2)
        print(f"\nSaved results to: {args.output}")
    
    # Save markdown report
    if args.report:
        report_md = generate_benchmark_report(benchmark_data)
        os.makedirs(os.path.dirname(args.report), exist_ok=True)
        with open(args.report, 'w') as f:
            f.write(report_md)
        print(f"Saved report to: {args.report}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

