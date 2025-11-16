#!/bin/bash
# Example: Running unified benchmarks for sparse attention

# This script demonstrates how to use the unified benchmarking system
# to compare different variants across patterns and backends

set -e

echo "========================================="
echo "Sparse Attention Benchmarking Demo"
echo "========================================="
echo ""

# Create results directory
mkdir -p bench/results/demo
mkdir -p bench/results/reports

# Example 1: Benchmark sliding_window pattern with balanced variant on RVV
echo "Example 1: Sliding Window (Balanced) on RVV"
echo "-------------------------------------------"
python bench/unified_bench.py \
  --pattern sliding_window \
  --variant balanced \
  --backend rvv \
  --L 128 --D 32 \
  --output bench/results/demo/sw_balanced_rvv.json \
  --report bench/results/reports/sw_balanced_rvv.md

echo ""
echo "✅ Results saved to bench/results/demo/sw_balanced_rvv.json"
echo ""

# Example 2: Compare low_power vs high_performance variants
echo "Example 2: Low Power vs High Performance Comparison"
echo "----------------------------------------------------"
python bench/unified_bench.py \
  --pattern sliding_window \
  --variant low_power \
  --backend rvv \
  --L 128 --D 32 \
  --output bench/results/demo/sw_low_power_rvv.json

python bench/unified_bench.py \
  --pattern sliding_window \
  --variant high_performance \
  --backend rvv \
  --L 128 --D 32 \
  --output bench/results/demo/sw_high_perf_rvv.json

echo ""
echo "✅ Comparison results saved!"
echo ""

# Example 3: Benchmark block_topk pattern
echo "Example 3: Block-TopK (Balanced) on RVV"
echo "----------------------------------------"
python bench/unified_bench.py \
  --pattern block_topk \
  --variant balanced \
  --backend rvv \
  --L 128 --D 32 \
  --output bench/results/demo/btopk_balanced_rvv.json \
  --report bench/results/reports/btopk_balanced_rvv.md

echo ""
echo "✅ Block-TopK results saved!"
echo ""

# Summary
echo "========================================="
echo "Demo Complete!"
echo "========================================="
echo ""
echo "Results are in: bench/results/demo/"
echo "Reports are in: bench/results/reports/"
echo ""
echo "View results with:"
echo "  cat bench/results/demo/sw_balanced_rvv.json | jq '.metrics'"
echo ""
echo "View report with:"
echo "  cat bench/results/reports/sw_balanced_rvv.md"
echo ""

