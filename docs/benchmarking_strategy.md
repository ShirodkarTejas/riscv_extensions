# Comprehensive Benchmarking Strategy

## Executive Summary

This document defines the benchmarking framework for evaluating sparse attention operators across multiple backends (CPU, GPU, RVV, Custom ISA) with different optimization variants (low-power, balanced, high-performance).

## ✅ **UPDATE: Unified Comparison Complete!**

**Status**: All 5 patterns now have unified comparison benchmarks across 4 precision modes!

**Quick Start**:
```bash
# Run unified comparison (all 5 patterns × 4 profiles = 20 configs)
python bench/create_unified_comparison.py --L 128 --D 32

# View comprehensive results
cat bench/results/UNIFIED_PATTERN_COMPARISON.md
```

**Key Achievements**:
- ✅ 100% pattern quantization coverage (5/5 patterns: sliding_window, block_local_global, nm_structured, lsh, landmark)
- ✅ 4 precision modes per pattern (fp32, bf16, i8, i4)
- ✅ Unified comparison tool for pattern selection
- ✅ Best-by-objective recommendations (memory, latency, efficiency)
- ✅ Use-case-specific guidance (IoT, Mobile, Edge, Cloud)

**Best Configurations Found** (L=128, D=32):
- 🔋 **Ultra Low Power**: `sliding_window + i4` (0.16 MB, 16.7x savings!)
- 📱 **Low Power**: `sliding_window + i8` (0.32 MB, 8.3x savings)
- ⚖️ **Balanced**: `landmark + bf16` (fastest latency: 5.9M cycles)
- ⚡ **High Performance**: `nm_structured + fp32` (full precision)

**Documentation**: See `bench/results/UNIFIED_PATTERN_COMPARISON.md` for complete analysis.

## Current State Analysis

### Existing Metrics ✅

| Metric Category | Current Implementation | Location |
|-----------------|------------------------|----------|
| **Cycles** | `sattn_rdcycle()` | `backends/rvv/include/sparse_attention_rvv.h` |
| **Memory Traffic** | `bytes_read`, `bytes_written` | RVV counters |
| **Compute** | `mac_flops` (MAC operations) | RVV counters |
| **Accuracy** | Mean absolute error vs dense | `bench/eval/eval_sparse_attention.py` |
| **Latency** | Wall-clock time (ms) | `bench/scripts/perf_sparse_attention.py` |
| **Hardware Counters** | CSRs for custom ISA | `hw/spec/csr_map.md` (0x7D0-0x7D5) |

### Existing Tools ✅

| Tool | Purpose | Output |
|------|---------|--------|
| `rvv_runner` | Execute with metrics | `checksum=... bytes_read=... cycles=...` |
| `rvv_autotune_sweep.py` | Optimize tile_rows | CSV/Markdown table |
| `eval_sparse_attention.py` | Accuracy + latency | CSV with deltas |
| `perf_sparse_attention.py` | Performance sweep | CSV with latency |
| `cycle_model.py` | Analytical model | Estimated cycles |

### What's Missing ⚠️

1. **Unified benchmark runner** across all backends
2. **Power/energy measurements** (not just proxy)
3. **Quality metrics** beyond mean absolute error
4. **Version management** for variants (low-power vs high-performance)
5. **Multi-objective optimization** (Pareto frontiers)
6. **Standardized reporting** (JSON schema)
7. **Comparison dashboard** (visual)

---

## Proposed Metric Framework

### 1. Performance Metrics

#### Primary Metrics

| Metric | Unit | Backend Support | Formula/Source |
|--------|------|-----------------|----------------|
| **Latency** | ms | All | Wall-clock `time.perf_counter()` |
| **Throughput** | tokens/sec | All | `B * H * L / latency` |
| **Cycles** | count | RVV, Custom ISA, Sim | `rdcycle` or CSR 0x7D0 |
| **IPC** | instructions/cycle | Custom ISA | Instructions issued / cycles |

#### Derived Metrics

| Metric | Unit | Formula |
|--------|------|---------|
| **Cycles per token** | cycles/token | `total_cycles / (B * H * L)` |
| **Efficiency** | % | `theoretical_min_cycles / actual_cycles * 100` |
| **Speedup** | ratio | `baseline_latency / optimized_latency` |

### 2. Memory Metrics

| Metric | Unit | Backend Support | Description |
|--------|------|-----------------|-------------|
| **Bytes Read** | bytes | All | Total data loaded (Q, K, V, indices) |
| **Bytes Written** | bytes | All | Output data (O) |
| **Memory Traffic** | GB/s | All | `(bytes_read + bytes_written) / latency` |
| **Arithmetic Intensity** | FLOPs/byte | All | `mac_flops * 2 / (bytes_read + bytes_written)` |
| **Cache Misses** | count | x86, Custom ISA | Hardware perf counters |

### 3. Compute Metrics

| Metric | Unit | Description |
|--------|------|-------------|
| **MAC Operations** | count | Multiply-accumulate operations |
| **FLOPs** | count | `mac_flops * 2` (multiply + add) |
| **Utilization** | % | `actual_flops / peak_flops * 100` |
| **GFLOPS** | GFLOPS | `flops / (latency * 1e6)` |

### 4. Energy/Power Metrics

| Metric | Unit | Backend | Measurement Method |
|--------|------|---------|-------------------|
| **Energy** | Joules | CPU | RAPL counters (`/sys/class/powercap/intel-rapl/`) |
| **Power** | Watts | GPU | NVML (`nvidia-smi --query-gpu=power.draw`) |
| **Energy Proxy** | au (arbitrary units) | RVV | `cycles * tech_factor` (e.g., 7nm: factor=0.5) |
| **Energy per Token** | mJ/token | All | `total_energy / (B * H * L)` |
| **TOPS/W** | TOPS/W | All | `(flops / 1e12) / avg_power` |

### 5. Accuracy/Quality Metrics

| Metric | Unit | Description | Threshold |
|--------|------|-------------|-----------|
| **Mean Absolute Error (MAE)** | float | `mean(abs(O_sparse - O_dense))` | < 0.01 (good) |
| **Max Absolute Error** | float | `max(abs(O_sparse - O_dense))` | < 0.1 (acceptable) |
| **Cosine Similarity** | 0-1 | Attention map similarity | > 0.95 (good) |
| **Perplexity Delta** | float | LM quality degradation | < 5% increase |
| **Attention Coverage** | % | `selected_tokens / total_tokens * 100` | Pattern-dependent |
| **Sparsity** | % | `1 - coverage` | Higher = more sparse |

### 6. Hardware-Specific Metrics

#### Custom ISA (CSR Performance Counters)

| CSR Address | Name | Description |
|-------------|------|-------------|
| 0x7D0 | `SATTN_CYCLE_COUNT` | Total cycles for last operation |
| 0x7D1 | `SATTN_MAC_OPS` | MAC operations performed |
| 0x7D2 | `SATTN_GATHER_CYCLES` | Cycles in gather/DMA |
| 0x7D3 | `SATTN_COMPUTE_CYCLES` | Cycles in compute (MAC/exp) |
| 0x7D4 | `SATTN_MEMORY_STALLS` | Cycles stalled on memory |
| 0x7D5 | `SATTN_IDLE_CYCLES` | Cycles idle waiting |

---

## Version/Variant Management

### Variant Dimensions

For each sparse attention pattern (sliding_window, block_topk, nm_structured, etc.), we define variants along these dimensions:

```
Pattern → Variant → Configuration
```

### 1. Optimization Target Variants

| Variant | Optimization Goal | Trade-offs | Use Case |
|---------|-------------------|------------|----------|
| **ultra_low_power** | Minimize energy | Lower throughput, higher latency | Battery-powered IoT |
| **low_power** | Balance power/perf | Moderate performance | Mobile devices |
| **balanced** | Balanced metrics | General purpose | Default |
| **high_performance** | Maximize throughput | Higher power consumption | Cloud/datacenter |
| **ultra_high_perf** | Maximum speed | Peak power, may reduce accuracy | Latency-critical |

### 2. Configuration Parameters per Variant

Example for `sliding_window` pattern:

```python
SLIDING_WINDOW_VARIANTS = {
    "ultra_low_power": {
        "window_size": 8,
        "global_tokens": 2,
        "tile_rows": 1,
        "precision": "i8",
        "quantization_scales": {"q": 0.05, "k": 0.05, "v": 0.05},
        "backend_preference": ["custom_isa", "rvv"],
    },
    "low_power": {
        "window_size": 16,
        "global_tokens": 4,
        "tile_rows": 2,
        "precision": "bf16",
        "backend_preference": ["rvv", "cpu"],
    },
    "balanced": {
        "window_size": 32,
        "global_tokens": 8,
        "tile_rows": 4,
        "precision": "bf16",
        "backend_preference": ["rvv", "gpu"],
    },
    "high_performance": {
        "window_size": 64,
        "global_tokens": 16,
        "tile_rows": 8,
        "precision": "bf16",
        "backend_preference": ["gpu", "rvv"],
    },
    "ultra_high_perf": {
        "window_size": 128,
        "global_tokens": 32,
        "tile_rows": 16,
        "precision": "fp32",
        "backend_preference": ["gpu"],
    },
}
```

Example for `block_topk` pattern:

```python
BLOCK_TOPK_VARIANTS = {
    "ultra_low_power": {
        "block_size": 16,
        "keep_ratio": 0.08,
        "global_tokens": 4,
        "gqa_group_size": 4,      # Share selection across 4 heads
        "comp_block_size": 8,     # Use compression blocks
        "precision": "i8",
        "tile_rows": 1,
    },
    "low_power": {
        "block_size": 32,
        "keep_ratio": 0.10,
        "global_tokens": 8,
        "gqa_group_size": 2,
        "comp_block_size": 16,
        "precision": "bf16",
        "tile_rows": 2,
    },
    "balanced": {
        "block_size": 64,
        "keep_ratio": 0.12,
        "global_tokens": 16,
        "gqa_group_size": 1,
        "comp_block_size": 0,  # Disabled
        "precision": "bf16",
        "tile_rows": 4,
    },
    "high_performance": {
        "block_size": 128,
        "keep_ratio": 0.16,
        "global_tokens": 32,
        "gqa_group_size": 1,
        "comp_block_size": 0,
        "precision": "bf16",
        "tile_rows": 8,
    },
    "ultra_high_perf": {
        "block_size": 256,
        "keep_ratio": 0.24,
        "global_tokens": 64,
        "gqa_group_size": 1,
        "comp_block_size": 0,
        "precision": "fp32",
        "tile_rows": 16,
    },
}
```

### 3. Variant Selection Strategy

```python
def select_variant(
    pattern: str,
    target_latency_ms: Optional[float] = None,
    target_energy_mj: Optional[float] = None,
    min_accuracy_threshold: float = 0.95,
    backend: str = "auto"
) -> Dict:
    """
    Automatically select the best variant based on constraints.
    
    Returns configuration dict with expected metrics.
    """
    # Implementation in bench/autotune/variant_selector.py
    pass
```

---

## Unified Benchmarking Framework

### Benchmark Suite Structure

```
bench/
├── unified_bench.py           # Main unified benchmark runner
├── variant_configs.py          # Variant definitions
├── metrics_collector.py        # Metric collection abstraction
├── results/
│   ├── benchmark_results.json  # Structured results
│   ├── comparison_matrix.csv   # Cross-variant comparison
│   └── pareto_frontiers.json   # Multi-objective optimization
└── visualization/
    ├── plot_latency_vs_power.py
    ├── plot_accuracy_vs_sparsity.py
    └── generate_report.py      # HTML report generator
```

### Benchmark JSON Schema

```json
{
  "benchmark_id": "sliding_window_balanced_rvv_2024-01-15_143022",
  "timestamp": "2024-01-15T14:30:22Z",
  "pattern": "sliding_window",
  "variant": "balanced",
  "backend": "rvv",
  "platform": {
    "arch": "riscv64",
    "cpu": "SiFive U74 + RVV",
    "vlen": 128,
    "elen": 64,
    "frequency_mhz": 1400
  },
  "configuration": {
    "B": 1, "H": 8, "L": 2048, "D": 64,
    "window_size": 32,
    "global_tokens": 8,
    "tile_rows": 4,
    "precision": "bf16"
  },
  "metrics": {
    "performance": {
      "latency_ms": 15.234,
      "throughput_tokens_per_sec": 134510,
      "cycles": 21327680,
      "cycles_per_token": 1303,
      "ipc": 0.85
    },
    "memory": {
      "bytes_read": 4194304,
      "bytes_written": 1048576,
      "memory_traffic_gbs": 344.6,
      "arithmetic_intensity": 3.2
    },
    "compute": {
      "mac_ops": 16777216,
      "gflops": 2.2
    },
    "energy": {
      "energy_joules": 0.152,
      "power_watts": 10.0,
      "energy_per_token_mj": 0.093,
      "tops_per_watt": 3.35
    },
    "accuracy": {
      "mae": 0.0023,
      "max_error": 0.045,
      "cosine_similarity": 0.985,
      "sparsity_percent": 98.4
    }
  },
  "comparison_vs_baseline": {
    "baseline": "dense_attention_fp32",
    "speedup": 12.4,
    "energy_reduction": 15.2,
    "accuracy_delta": 0.0023
  }
}
```

---

## Standard Benchmarking Protocol

### 1. Pre-Benchmark Setup

```python
# Warm up (prevent cold start effects)
for _ in range(warmup_iterations):
    run_kernel(...)

# Reset performance counters
sattn_rvv_counters_reset()
if backend == "custom_isa":
    sattn_reset_perf_counters()

# Start power monitoring
power_monitor.start()
```

### 2. Measurement

```python
# Run multiple iterations for statistical significance
results = []
for i in range(num_iterations):
    start_time = time.perf_counter()
    start_cycles = sattn_rdcycle()
    
    # Execute kernel
    output = sparse_attention(Q, K, V, ...)
    
    end_cycles = sattn_rdcycle()
    end_time = time.perf_counter()
    
    # Collect metrics
    metrics = collect_all_metrics(start_time, end_time, start_cycles, end_cycles)
    results.append(metrics)

# Aggregate statistics
final_metrics = {
    "mean": np.mean(results, axis=0),
    "std": np.std(results, axis=0),
    "min": np.min(results, axis=0),
    "max": np.max(results, axis=0),
    "median": np.median(results, axis=0),
}
```

### 3. Validation

```python
# Accuracy check
if L <= 256:  # Only for small sequences (dense baseline is expensive)
    O_dense = dense_attention(Q, K, V)
    mae = np.mean(np.abs(O - O_dense))
    assert mae < accuracy_threshold, f"Accuracy degraded: MAE={mae}"

# Checksum verification
checksum = np.sum(O)
assert not np.isnan(checksum), "Output contains NaN"
assert not np.isinf(checksum), "Output contains Inf"
```

---

## Multi-Objective Optimization

### Pareto Frontier Analysis

For each pattern, we compute Pareto-optimal variants across multiple objectives:

```python
objectives = ["latency", "energy", "accuracy"]

# Lower is better for latency and energy
# Higher is better for accuracy (minimize negative)
pareto_frontier = compute_pareto_optimal(
    variants=all_variants,
    objectives=[
        ("latency_ms", "minimize"),
        ("energy_joules", "minimize"),
        ("mae", "minimize"),
    ]
)
```

**Visualization**: 3D scatter plot with Pareto surface

---

## Comparison Matrix

### Cross-Backend Comparison

| Pattern | Variant | Backend | Latency (ms) | Energy (mJ) | Accuracy (MAE) | Memory (MB) |
|---------|---------|---------|--------------|-------------|----------------|-------------|
| sliding_window | balanced | CPU | 125 | 1250 | 0.002 | 64 |
| sliding_window | balanced | GPU | 8 | 320 | 0.002 | 64 |
| sliding_window | balanced | RVV | 18 | 180 | 0.002 | 64 |
| sliding_window | balanced | Custom ISA | 3 | 45 | 0.003 | 64 |
| sliding_window | low_power | Custom ISA | 5 | 25 | 0.008 | 32 |
| block_topk | balanced | GPU | 12 | 480 | 0.005 | 96 |
| block_topk | high_performance | GPU | 6 | 720 | 0.004 | 128 |

### Cross-Variant Comparison (Same Backend)

For RVV backend, `sliding_window` pattern, L=2048:

| Variant | Latency (ms) ↓ | Energy (mJ) ↓ | Accuracy (MAE) ↓ | Speedup vs Dense |
|---------|---------------|---------------|------------------|------------------|
| ultra_low_power | 28 | 140 | 0.015 | 8x |
| low_power | 22 | 176 | 0.008 | 10x |
| **balanced** | 18 | 180 | 0.002 | 12x |
| high_performance | 14 | 224 | 0.002 | 16x |
| ultra_high_perf | 11 | 330 | 0.001 | 20x |

---

## Standardized Reports

### 1. Summary Report (Markdown)

```markdown
# Sparse Attention Benchmark Report

**Pattern**: sliding_window
**Variant**: balanced
**Backend**: RVV (QEMU)
**Date**: 2024-01-15

## Configuration
- Sequence Length: 2048
- Window Size: 32
- Precision: bf16

## Performance Summary
| Metric | Value |
|--------|-------|
| Latency | 18.2 ms |
| Throughput | 112k tokens/sec |
| Speedup vs Dense | 12.4x |

## Resource Usage
| Resource | Value |
|----------|-------|
| Memory Traffic | 5.2 MB |
| Energy per Token | 0.088 mJ |
| GFLOPS | 2.2 |

## Quality
| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| MAE | 0.0023 | < 0.01 | ✅ PASS |
| Cosine Similarity | 0.985 | > 0.95 | ✅ PASS |
```

### 2. JSON Export

Full structured data for programmatic analysis (see schema above).

### 3. HTML Dashboard

Interactive visualization with:
- Latency vs Energy scatter plots
- Accuracy vs Sparsity curves
- Backend comparison bar charts
- Pareto frontier surfaces

---

## Implementation Roadmap

### Phase 1: Core Infrastructure (Week 1-2)
- [ ] Create `unified_bench.py` main runner
- [ ] Implement `metrics_collector.py` abstraction
- [ ] Define variant configurations in `variant_configs.py`
- [ ] Add JSON schema validation

### Phase 2: Metric Collection (Week 2-3)
- [ ] Integrate energy measurement (RAPL, NVML)
- [ ] Add custom ISA CSR performance counter support
- [ ] Implement accuracy metrics (cosine similarity, perplexity)
- [ ] Cross-backend metric normalization

### Phase 3: Variant Management (Week 3-4)
- [ ] Define variants for all patterns
- [ ] Implement automatic variant selection
- [ ] Add constraint-based search

### Phase 4: Analysis & Visualization (Week 4-5)
- [ ] Pareto frontier computation
- [ ] Comparison matrix generation
- [ ] Interactive HTML dashboard
- [ ] Automated report generation

### Phase 5: Integration (Week 5-6)
- [ ] CI/CD integration (run on each commit)
- [ ] Regression detection
- [ ] Performance tracking over time
- [ ] Public results dashboard

---

## Next Steps

See `docs/benchmarking_implementation_plan.md` for detailed implementation tasks.

