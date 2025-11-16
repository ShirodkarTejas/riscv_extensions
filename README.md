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

**Key Findings** (L=128, D=32):

| Use Case | Best Pattern | Precision | Memory | Savings |
|----------|--------------|-----------|--------|---------|
| 🔋 **Ultra Low Power** | `sliding_window` | **i4** | **0.16 MB** | **16.7x!** |
| 📱 **Low Power** | `sliding_window` | **i8** | 0.32 MB | 8.3x |
| ⚖️ **Balanced** | `landmark` | **bf16** | 1.01 MB | Fastest! |
| ⚡ **High Performance** | `nm_structured` | **fp32** | 5.25 MB | Full precision |

### 📊 Per-Pattern Variants

Each pattern supports 4 optimization profiles:

```
Pattern (e.g., sliding_window)
  ├── ultra_low_power    (i4)   → 16.7x less memory! 🔋
  ├── low_power          (i8)   → 8.3x less memory! 📱
  ├── balanced          (bf16)  → FASTEST latency! ⚡
  └── high_performance  (fp32)  → Full precision 🎯
```

**All 5 Patterns Fully Quantized**:
- ✅ `sliding_window` - Best for power-constrained (up to 16.7x savings!)
- ✅ `block_local_global` - Hybrid local+global (up to 8.5x savings)
- ✅ `nm_structured` - N:M structured sparsity (up to 6.0x savings)
- ✅ `lsh` - Hash-based attention (up to 8.0x savings)
- ✅ `landmark` - Centroid-based, fastest latency

**Quick Start**:
```bash
# Compare ALL patterns and profiles in one report
python bench/create_unified_comparison.py --L 128 --D 32

# Benchmark a specific pattern across all precisions
python bench/comprehensive_benchmark.py \
  --pattern sliding_window \
  --L 128 --D 32

# Compare variants for a single pattern
python bench/compare_variants.py \
  --pattern lsh \
  --L 128 --D 32
```

**Documentation**:
- 🏆 **`bench/results/UNIFIED_PATTERN_COMPARISON.md`** - **ALL patterns compared!** (Recommended starting point)
- 📊 `bench/results/ALL_VARIANTS_ALL_PRECISIONS.md` - Detailed sliding_window analysis
- 📈 `bench/results/PATTERN_SUPPORT_STATUS.md` - Implementation status for all patterns
- 📘 `docs/NEXT_STEPS_BENCHMARKING.md` - Quick start guide
- 📈 `docs/benchmarking_strategy.md` - Complete metrics and variant definitions
- 📋 `docs/benchmarking_implementation_plan.md` - Development roadmap

## Status
- CPU ref, GPU sliding-window + block-topk, **RVV baseline ready and tested on QEMU** ✅
- MLIR dialect/pass stubs, RoCC skeleton, IMC proxy ready
- **Custom RISC-V instructions**: Specification, hardware decoder, CSR interface, and C intrinsics complete (encodings validated) ✅
- **Benchmarking system**: Unified runner with 5 variants per pattern, working on RVV backend ✅
- Next: MLIR passes, RoCC spdot_bsr prototype, end-to-end sim
