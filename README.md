# RISC-V Sparse Attention Operator Library (MLIR + RoCC)

Multi-spec deep learning operator library centered on Sparse Attention, with backends for CPU (reference), GPU (Triton), RVV baseline, custom RISC-V (RoCC path), and IMC cost model.

## Quickstart

Python API (CPU reference):
```python
from runtime.api.python import sparse_attention
import numpy as np
Q = np.random.randn(1, 2, 128, 64).astype(np.float32)
K = np.random.randn(1, 2, 128, 64).astype(np.float32)
V = np.random.randn(1, 2, 128, 64).astype(np.float32)
O = sparse_attention(Q, K, V,
    pattern="block_topk",
    params={"block_size": 64, "keep_ratio": 0.12, "global_tokens": 16},
    precision="bf16",
    training=False)
```

GPU (CUDA + Triton): pass CUDA tensors. Sliding-window uses Triton kernel; block-topk uses Triton gather kernel with Torch prepass.

Perf scripts:
```bash
# CPU/GPU microbenchmarks
python bench/scripts/perf_sparse_attention.py --pattern block_topk --device cpu --L 2048
python bench/scripts/perf_sparse_attention.py --pattern sliding_global --device cuda --L 8192
```

Autotune:
```bash
python bench/autotune/autotune_sparse_attention.py --pattern sliding_global --device cuda --lengths 2048 8192 32768
```

Evaluation:
```bash
python bench/eval/eval_sparse_attention.py --pattern block_topk --device cpu --lengths 2048 8192 32768
```

## Layout
- `ops/sparse_attention/` CPU+GPU kernels
- `runtime/api/` Python and C++ API
- `compiler/mlir/` SATTN dialect, passes docs, example
- `backends/rvv/` RVV baseline + cycles harness
- `hw/spec/` custom primitives, intrinsics, cycle model
- `hw/rtl/` RoCC skeleton
- `hw/runtime/` MMIO driver stub
- `imc/neurosim/` sparse mapping and proxy energy/latency
- `bench/` configs, scripts, autotune, eval
- `docs/` API, shapes/contract, scope decisions

## Install
```bash
pip install -r requirements.txt
# Optional
pip install triton torch
```

## Tests
```bash
pytest -q
```

## Profiles
See `docs/profiles.md` for suggested knobs for highperf / lowpower / imc.

## Custom RISC-V Instructions

This library includes custom RISC-V instruction definitions for sparse attention primitives:

- **Opcode**: `custom-1` (0x2B)
- **Format**: R-type with funct3 field selecting primitive
- **Parameter Passing**: CSR-based (0x7C0-0x7DF) or MMIO fallback
- **Primitives**: 7 operations (blk_reduce, topk_idx, gath2d, scat2d, spdot_bsr, softmax_fused, spmm_bsr)

**Documentation**:
- `hw/spec/instruction_encoding.md` - ISA specification and encoding
- `hw/spec/csr_map.md` - CSR address map and register layout  
- `hw/spec/assembly_guide.md` - Assembly and C programming examples
- `hw/spec/sattn_isa.h` - C header with instruction macros and CSR definitions

**Hardware**:
- `hw/rtl/sattn_inst_decode.sv` - Instruction decoder module
- `hw/rtl/rocc_sattn.sv` - Accelerator with CSR interface (backward-compatible with MMIO)

**Software**:
- Compile with `-DSATTN_USE_CSR_INSTRUCTIONS` to enable instruction-based mode
- Default (no flag): uses MMIO for simulation compatibility
- Unified API automatically selects appropriate backend

## Backend Selection

This library provides multiple backends - choose based on your target platform:

| Backend | When to Use | QEMU Support | Status |
|---------|-------------|--------------|--------|
| **RVV** | Portable RISC-V, testing | ✅ `-cpu rv64,v=true` | ✅ Ready |
| **Custom ISA** | Hardware accelerator | ⚠️ Requires plugin | ✅ Encodings ready |
| **CPU** | Reference, any platform | ✅ Native | ✅ Ready |
| **GPU** | High throughput | N/A (CUDA/Triton) | ✅ Ready |

**Quick Start with QEMU**:
```bash
# RVV backend works out of the box!
python scripts/build_and_run_rvv_qemu.py
```

**Documentation**:
- 📘 `docs/QEMU_GUIDE.md` - Quick start guide for running on QEMU
- 📐 `docs/backend_architecture.md` - Complete architecture and backend comparison
- ❓ `docs/FAQ.md` - Common questions answered
- 📊 `docs/architecture_diagram.md` - Visual architecture diagrams

## Benchmarking & Variants

This library provides **100% quantization coverage** across all 5 sparse attention patterns with 4 precision modes each:

### 🎯 **Unified Pattern Comparison** (NEW!)

Compare **all 5 patterns × 4 profiles** in one place:

```bash
# Generate unified comparison report
python bench/create_unified_comparison.py --L 128 --D 32

# View results
cat bench/results/UNIFIED_PATTERN_COMPARISON.md
```

**Key Findings** (L=128, D=32, Real QEMU Results):

| Use Case | Best Pattern | Precision | Energy | Memory | Cycles |
|----------|--------------|-----------|--------|--------|--------|
| 🔋 **Ultra Low Power** | `sliding_window` | **i4** | **32.17 µJ** | **0.42 MB** | 15.3M |
| 📱 **Low Power** | `sliding_window` | **i8** | 38.79 µJ | 0.58 MB | 15.3M |
| ⚖️ **Balanced** | `sliding_window` | **bf16** | 97.79 µJ | 1.69 MB | **9.0M** (fastest!) |
| ⚡ **High Performance** | `landmark` | **fp32** | 131.71 µJ | 2.50 MB | 9.8M |

**Energy Savings vs FP32**: i4 = **84%**, i8 = **81%**, bf16 = **51%**

### 📊 Per-Pattern Variants

Each pattern supports 4 optimization profiles:

```
Pattern (e.g., sliding_window)
  ├── ultra_low_power    (i4)   → 16.7x less memory! 🔋
  ├── low_power          (i8)   → 8.3x less memory! 📱
  ├── balanced          (bf16)  → FASTEST latency! ⚡
  └── high_performance  (fp32)  → Full precision 🎯
```

**All 5 Patterns Fully Quantized** (Real Results):
- ✅ `sliding_window` - **Best energy** (32 µJ i4, 84% savings vs fp32)
- ✅ `block_local_global` - Best efficiency (12.0M GOPs/W at i4)
- ✅ `nm_structured` - Good balance (136 µJ i4, 58% savings)
- ✅ `lsh` - Moderate savings (218 µJ i4, 40% savings)
- ✅ `landmark` - Fastest cycles (9.0-10M), minimal quantization benefit

**Quick Start - Run Real Benchmarks**:
```bash
# 1. Run comprehensive benchmarks in Docker (all 20 configs)
./scripts/run_benchmark_in_docker.sh --L 128 --D 32

# 2. Generate detailed report with energy metrics
python bench/generate_comprehensive_report.py \
  --input bench/results/comprehensive_docker_results.json

# 3. Find optimal config for your constraints
python bench/variant_selector.py \
  --max-memory-mb 1.0 \
  --max-energy-uj 50 \
  --L 128 --D 32
```

**Documentation**:
- 🏆 **`COMPLETE_BENCHMARK_WORKFLOW.md`** - **Complete benchmarking guide** (START HERE!)
- 📊 **`bench/results/COMPREHENSIVE_BENCHMARK_REPORT.md`** - Real QEMU results with energy metrics
- 📈 `bench/results/UNIFIED_PATTERN_COMPARISON.md` - All patterns compared
- 🔋 `docs/ENERGY_ACCURACY_IMPLEMENTATION_COMPLETE.md` - Energy & accuracy module details
- 📘 `docs/benchmarking_strategy.md` - Complete metrics and variant definitions
- 📋 `docs/benchmarking_implementation_plan.md` - Development roadmap

## Status
- CPU ref, GPU sliding-window + block-topk, **RVV baseline ready and tested on QEMU** ✅
- MLIR dialect/pass stubs, RoCC skeleton, IMC proxy ready
- **Custom RISC-V instructions**: Specification, hardware decoder, CSR interface, and C intrinsics complete (encodings validated) ✅
- **Benchmarking system**: Complete with energy estimation, 20 configs tested, real results available ✅
  - 5 patterns × 4 precisions (fp32/bf16/i8/i4) fully quantized ✅
  - Energy metrics (7nm/5nm/3nm tech nodes) ✅
  - Comprehensive Docker-based benchmark infrastructure ✅
- Next: MLIR passes, RoCC spdot_bsr prototype, end-to-end sim
