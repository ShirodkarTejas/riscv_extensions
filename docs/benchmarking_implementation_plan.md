# Benchmarking System Implementation Plan

## Overview

This document breaks down the benchmarking strategy into actionable tasks.

---

## ✅ **COMPLETED TASKS** (November 2025)

### Major Milestones Achieved

**Phase 1 & 2: Complete! 🎉**

- ✅ **Unified Benchmark Runner** - `bench/unified_bench.py` (462 lines)
  - Multi-backend support (RVV complete)
  - Pattern and variant selection
  - Structured JSON output
  - Metrics collection framework

- ✅ **Comprehensive Benchmark Tool** - `bench/comprehensive_benchmark.py` (412 lines)
  - Tests all precisions × all profiles for a pattern
  - Generates CSV and Markdown reports
  - 20 configs per pattern (4 precisions × 5 profiles)
  - Successfully run for: sliding_window, block_topk, nm_structured

- ✅ **Unified Pattern Comparison** - `bench/create_unified_comparison.py` (341 lines) **NEW!**
  - **Compares all 5 patterns × 4 profiles in one report**
  - Best-by-objective recommendations
  - Use-case-specific guidance
  - Output: `bench/results/UNIFIED_PATTERN_COMPARISON.md`

- ✅ **Variant Comparison Tool** - `bench/compare_variants.py`
  - Side-by-side comparison of all 5 variants for a pattern
  - Memory, latency, and efficiency analysis

- ✅ **100% Pattern Quantization** - All 5 patterns fully instrumented!
  - sliding_window: fp32, bf16, i8, i4 ✅
  - block_local_global: fp32, bf16, i8, i4 ✅
  - nm_structured: fp32, bf16, i8, i4 ✅
  - lsh: fp32, bf16, i8, i4 ✅
  - landmark: fp32, bf16, i8, i4 ✅

- ✅ **Counter Instrumentation** - Full metrics tracking
  - bytes_read, bytes_written (precision-aware)
  - mac_flops (multiply-accumulate operations)
  - rvv_cycles (QEMU cycle counts)

- ✅ **Documentation**
  - `bench/results/UNIFIED_PATTERN_COMPARISON.md` - **Main comparison report** 🏆
  - `bench/results/ALL_VARIANTS_ALL_PRECISIONS.md` - Detailed sliding_window
  - `bench/results/PATTERN_SUPPORT_STATUS.md` - Implementation status
  - `ALL_PATTERNS_QUANTIZED_COMPLETE.md` - Quantization summary
  - `docs/NEXT_STEPS_BENCHMARKING.md` - Quick start guide

**Key Results**:
- 🔋 Best Ultra Low Power: `sliding_window + i4` (0.16 MB, **16.7x savings!**)
- 📱 Best Low Power: `sliding_window + i8` (0.32 MB, 8.3x savings)
- ⚖️ Best Balanced: `landmark + bf16` (fastest latency)
- ⚡ Best High Perf: `nm_structured + fp32` (full precision)

**Total Code Delivered**: ~1,600 lines of benchmarking infrastructure + ~1,100 lines of quantized kernels

---

## Task Breakdown

###

 Phase 1: Core Infrastructure 🏗️

#### Task 1.1: Create Unified Benchmark Runner
**File**: `bench/unified_bench.py`

**Requirements**:
- Support all backends (CPU, GPU, RVV, Custom ISA)
- Accept pattern and variant as parameters
- Run warmup + measurement iterations
- Collect all available metrics
- Output structured JSON

**Interface**:
```python
python bench/unified_bench.py \
  --pattern sliding_window \
  --variant balanced \
  --backend rvv \
  --L 2048 --D 64 \
  --output results/benchmark_001.json
```

**Deliverables**:
- [ ] Main runner script
- [ ] CLI argument parsing
- [ ] Backend detection and dispatch
- [ ] Iteration management (warmup + measurement)
- [ ] JSON output with schema validation

---

#### Task 1.2: Metrics Collector Abstraction
**File**: `bench/metrics_collector.py`

**Requirements**:
- Unified interface for all metric sources
- Backend-specific implementations
- Automatic unit conversion
- Statistical aggregation (mean, std, percentiles)

**API**:
```python
class MetricsCollector:
    def start(self):
        """Start measurement (time, cycles, power)"""
        
    def stop(self):
        """Stop measurement and collect"""
        
    def get_metrics(self) -> Dict:
        """Return all metrics in standard format"""
        
    def compare_with_baseline(self, baseline: Dict) -> Dict:
        """Compute speedup, energy reduction, etc."""
```

**Deliverables**:
- [ ] Base `MetricsCollector` class
- [ ] `RVVMetricsCollector` implementation
- [ ] `CustomISAMetricsCollector` (reads CSRs)
- [ ] `CPUMetricsCollector` (RAPL, perf counters)
- [ ] `GPUMetricsCollector` (NVML)
- [ ] Unit tests for each collector

---

#### Task 1.3: Variant Configuration System
**File**: `bench/variant_configs.py`

**Requirements**:
- Define all variants for each pattern
- Programmatic access to configurations
- Validation of configuration parameters
- Backend compatibility checking

**Structure**:
```python
VARIANTS = {
    "sliding_window": {
        "ultra_low_power": {...},
        "low_power": {...},
        "balanced": {...},
        "high_performance": {...},
        "ultra_high_perf": {...},
    },
    "block_topk": {...},
    "nm_structured": {...},
    # etc.
}

def get_variant_config(pattern: str, variant: str) -> Dict
def list_variants(pattern: str) -> List[str]
def validate_config(config: Dict) -> bool
```

**Deliverables**:
- [ ] Variant definitions for `sliding_window`
- [ ] Variant definitions for `block_topk`
- [ ] Variant definitions for `nm_structured`
- [ ] Variant definitions for `lsh`
- [ ] Configuration validator
- [ ] Unit tests

---

#### Task 1.4: JSON Schema and Validation
**File**: `bench/schema/benchmark_result.json`

**Requirements**:
- Formal JSON schema for benchmark results
- Validation tooling
- Schema versioning

**Deliverables**:
- [ ] JSON Schema document
- [ ] Python validator using `jsonschema`
- [ ] Schema version 1.0 release
- [ ] Documentation and examples

---

### Phase 2: Metric Collection 📊

#### Task 2.1: Energy Measurement Integration

**Requirements**:
- CPU: RAPL interface (`/sys/class/powercap/intel-rapl/`)
- GPU: NVML API (`pynvml`)
- RVV/Custom ISA: Cycle-based proxy with tech factors
- ARM: PMU counters

**File**: `bench/energy_monitor.py`

**API**:
```python
class EnergyMonitor:
    def start(self):
        """Start energy measurement"""
        
    def stop(self) -> float:
        """Return energy in Joules"""
        
    def get_average_power(self) -> float:
        """Return average power in Watts"""
```

**Deliverables**:
- [ ] RAPL energy reader (x86)
- [ ] NVML energy reader (NVIDIA GPU)
- [ ] Cycle-based proxy calculator
- [ ] Configurable tech factors (7nm, 5nm, etc.)
- [ ] Unit tests and validation against power meter

---

#### Task 2.2: Custom ISA Performance Counter Support

**Requirements**:
- Read CSRs 0x7D0-0x7D5 in C intrinsics
- Python bindings for counter access
- Integration with `MetricsCollector`

**Files**:
- `hw/spec/sattn_isa.h` (update with perf counter macros)
- `backends/rocc/src/perf_counters.c`

**Deliverables**:
- [ ] C macros to read perf CSRs
- [ ] Python ctypes bindings
- [ ] Integration into `CustomISAMetricsCollector`
- [ ] Example usage in benchmark

---

#### Task 2.3: Accuracy/Quality Metrics

**File**: `bench/accuracy_metrics.py`

**Requirements**:
- Mean Absolute Error (MAE)
- Max Absolute Error
- Cosine Similarity (attention maps)
- Perplexity delta (for LM tasks)
- Per-head accuracy breakdown

**API**:
```python
def compute_accuracy_metrics(
    O_sparse: np.ndarray,
    O_dense: np.ndarray
) -> Dict:
    return {
        "mae": ...,
        "max_error": ...,
        "cosine_similarity": ...,
        "l2_distance": ...,
    }
```

**Deliverables**:
- [ ] Accuracy metric functions
- [ ] Attention map visualization
- [ ] Perplexity computation helper
- [ ] Integration with benchmark runner

---

#### Task 2.4: Memory Profiling Enhancement

**Requirements**:
- Track per-kernel memory traffic
- Cache miss rates (if available)
- Memory bandwidth utilization

**File**: `bench/memory_profiler.py`

**Deliverables**:
- [ ] Enhanced memory counter tracking
- [ ] Cache profiling on x86 (perf events)
- [ ] Arithmetic intensity calculator
- [ ] Memory roofline model integration

---

### Phase 3: Variant Management 🎛️

#### Task 3.1: Define All Pattern Variants

**Files**: `bench/variant_configs.py`

**Deliverables**:
- [ ] `sliding_window` (5 variants)
- [ ] `block_topk` (5 variants)
- [ ] `nm_structured` (3 variants)
- [ ] `lsh` (3 variants)
- [ ] `landmark` (3 variants)
- [ ] Documentation for each variant

---

#### Task 3.2: Automatic Variant Selection

**File**: `bench/variant_selector.py`

**Requirements**:
- Constraint-based selection (latency budget, energy budget, accuracy threshold)
- Multi-objective optimization
- Backend capability matching

**API**:
```python
def select_variant(
    pattern: str,
    constraints: Dict,  # {"max_latency_ms": 20, "min_accuracy": 0.98}
    backend: str = "auto",
    hardware_caps: Dict = None
) -> Tuple[str, Dict]:
    """
    Returns: (variant_name, configuration)
    """
```

**Deliverables**:
- [ ] Constraint solver implementation
- [ ] Pareto frontier pre-computation
- [ ] Hardware capability detection
- [ ] CLI tool for interactive selection

---

#### Task 3.3: Variant Comparison Tool

**File**: `bench/compare_variants.py`

**Requirements**:
- Run all variants for a given pattern
- Generate comparison matrix
- Highlight Pareto-optimal variants

**Usage**:
```bash
python bench/compare_variants.py \
  --pattern sliding_window \
  --backend rvv \
  --L 2048 \
  --output results/variant_comparison.json
```

**Deliverables**:
- [ ] Comparison runner script
- [ ] Markdown table generator
- [ ] CSV export
- [ ] Highlight best variant per objective

---

### Phase 4: Analysis & Visualization 📈

#### Task 4.1: Pareto Frontier Computation

**File**: `bench/analysis/pareto_frontier.py`

**Requirements**:
- Multi-objective Pareto optimality test
- 2D and 3D frontier computation
- Dominated/non-dominated classification

**API**:
```python
def compute_pareto_frontier(
    results: List[Dict],
    objectives: List[Tuple[str, str]]  # [("latency_ms", "minimize"), ...]
) -> List[Dict]:
    """Return non-dominated configurations"""
```

**Deliverables**:
- [ ] Pareto frontier algorithm
- [ ] 2D plotting (latency vs energy)
- [ ] 3D plotting (latency vs energy vs accuracy)
- [ ] JSON export of frontier points

---

#### Task 4.2: Visualization Scripts

**Directory**: `bench/visualization/`

**Files**:
- `plot_latency_vs_power.py`
- `plot_accuracy_vs_sparsity.py`
- `plot_backend_comparison.py`
- `plot_variant_comparison.py`
- `plot_roofline_model.py`

**Requirements**:
- Use `matplotlib` or `plotly` for interactivity
- Consistent styling
- Export to PNG and HTML

**Deliverables**:
- [ ] Latency vs Energy scatter plot
- [ ] Accuracy vs Sparsity curve
- [ ] Backend comparison bar charts
- [ ] Roofline model plot
- [ ] Interactive HTML plots (plotly)

---

#### Task 4.3: HTML Dashboard Generator

**File**: `bench/visualization/generate_dashboard.py`

**Requirements**:
- Single HTML file with all plots embedded
- Filterable results table
- Comparison mode
- Responsive design

**Output**: `bench/results/dashboard.html`

**Deliverables**:
- [ ] HTML template with embedded JavaScript
- [ ] Interactive plot generation
- [ ] Filterable/sortable results table
- [ ] Export button (CSV download)

---

#### Task 4.4: Markdown Report Generator

**File**: `bench/reporting/generate_report.py`

**Requirements**:
- Structured markdown report
- Embedded tables and plots
- Summary statistics
- Recommendations

**Usage**:
```bash
python bench/reporting/generate_report.py \
  --results results/*.json \
  --output report.md
```

**Deliverables**:
- [ ] Report template
- [ ] Automated plot embedding
- [ ] Summary statistics computation
- [ ] Variant recommendation logic

---

### Phase 5: Integration & Automation 🔄

#### Task 5.1: CI/CD Integration

**File**: `.github/workflows/benchmark.yml` (or similar)

**Requirements**:
- Run benchmarks on each commit (subset)
- Full benchmark suite weekly
- Upload results to artifact storage
- Regression detection

**Deliverables**:
- [ ] GitHub Actions workflow
- [ ] Benchmark subset selection
- [ ] Result artifact upload
- [ ] Slack/email notifications on regressions

---

#### Task 5.2: Regression Detection

**File**: `bench/regression_detector.py`

**Requirements**:
- Compare current results with baseline
- Statistical significance testing
- Alert on regressions > threshold (e.g., 10%)

**API**:
```python
def detect_regressions(
    current: Dict,
    baseline: Dict,
    thresholds: Dict = {"latency": 1.10, "energy": 1.15}
) -> List[str]:
    """Return list of regressed metrics"""
```

**Deliverables**:
- [ ] Regression detection algorithm
- [ ] Configurable thresholds
- [ ] HTML regression report
- [ ] Integration with CI

---

#### Task 5.3: Performance Tracking Dashboard

**Requirements**:
- Track performance over time (commit history)
- Visualize trends
- Public-facing results page

**Technology**: GitHub Pages or dedicated server

**Deliverables**:
- [ ] Historical data storage (SQLite or JSON files)
- [ ] Trend plotting
- [ ] Public dashboard HTML
- [ ] Automated updates

---

#### Task 5.4: Documentation and Examples

**Files**:
- `docs/benchmarking_user_guide.md`
- `examples/run_benchmark_example.py`
- `examples/custom_variant_example.py`

**Deliverables**:
- [ ] User guide for running benchmarks
- [ ] Tutorial: Adding a new variant
- [ ] Tutorial: Adding a new metric
- [ ] Example scripts for common workflows
- [ ] Video walkthrough (optional)

---

## Implementation Priority

### High Priority (Week 1-2) 🔴
1. ✅ Task 1.1: Unified Benchmark Runner
2. ✅ Task 1.2: Metrics Collector Abstraction
3. ✅ Task 1.3: Variant Configuration System
4. ✅ Task 2.3: Accuracy Metrics

### Medium Priority (Week 3-4) 🟡
5. Task 2.1: Energy Measurement
6. Task 2.2: Custom ISA Perf Counters
7. Task 3.1: Define All Variants
8. Task 3.2: Variant Selection
9. Task 4.1: Pareto Frontier

### Lower Priority (Week 5-6) 🟢
10. Task 4.2: Visualization Scripts
11. Task 4.3: HTML Dashboard
12. Task 4.4: Report Generator
13. Task 5.1: CI/CD Integration

### Future Enhancements 🔵
14. Task 5.2: Regression Detection
15. Task 5.3: Performance Tracking
16. Task 5.4: Documentation

---

## Success Metrics

### Phase 1 Success Criteria
- [ ] Can run benchmark for any pattern/variant/backend
- [ ] Results output as valid JSON
- [ ] Metrics collection works for at least 2 backends

### Phase 2 Success Criteria
- [ ] Energy measurement functional on at least 1 platform
- [ ] Accuracy metrics computed and validated
- [ ] Custom ISA perf counters readable

### Phase 3 Success Criteria
- [ ] All patterns have defined variants
- [ ] Variant selector recommends appropriate configuration
- [ ] Comparison tool produces useful output

### Phase 4 Success Criteria
- [ ] Can generate Pareto frontier plots
- [ ] HTML dashboard is interactive and usable
- [ ] Reports are automatically generated

### Phase 5 Success Criteria
- [ ] Benchmarks run automatically in CI
- [ ] Regressions are detected and reported
- [ ] Public dashboard is live

---

## Estimated Timeline

| Phase | Duration | Completion Target |
|-------|----------|-------------------|
| Phase 1: Core Infrastructure | 2 weeks | Week 2 |
| Phase 2: Metric Collection | 2 weeks | Week 4 |
| Phase 3: Variant Management | 2 weeks | Week 6 |
| Phase 4: Visualization | 2 weeks | Week 8 |
| Phase 5: Integration | 2 weeks | Week 10 |

**Total**: ~10 weeks for complete implementation

---

## Dependencies

### External Libraries
```txt
# Python requirements
numpy>=1.20
matplotlib>=3.3
plotly>=5.0
jsonschema>=4.0
psutil>=5.8  # For CPU/memory monitoring
pynvml>=11.0  # For NVIDIA GPU metrics
pandas>=1.3  # For data analysis
scipy>=1.7  # For statistical tests
```

### Hardware Access
- x86 CPU with RAPL support (energy measurement)
- NVIDIA GPU with NVML (optional, for GPU benchmarks)
- RISC-V hardware with RVV or custom ISA (for full testing)

---

## Next Steps

1. **Review and prioritize**: Confirm task priorities with team
2. **Assign owners**: Assign team members to tasks
3. **Create tracking**: Create GitHub issues for each task
4. **Start Phase 1**: Begin with unified benchmark runner
5. **Iterate**: Weekly reviews and adjustments

---

## Quick Start (First Task)

To get started immediately, begin with Task 1.1:

```bash
# Create the file structure
mkdir -p bench/results
mkdir -p bench/schema
mkdir -p bench/visualization
mkdir -p bench/analysis
mkdir -p bench/reporting

# Create the main benchmark runner
touch bench/unified_bench.py
touch bench/metrics_collector.py
touch bench/variant_configs.py

# Start implementing unified_bench.py (see strategy doc for API)
```

See `docs/benchmarking_strategy.md` for detailed specifications.

