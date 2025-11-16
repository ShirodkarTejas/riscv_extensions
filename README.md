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

## Status
- CPU ref, GPU sliding-window + block-topk, **RVV baseline ready and tested on QEMU** ✅
- MLIR dialect/pass stubs, RoCC skeleton, IMC proxy ready
- **Custom RISC-V instructions**: Specification, hardware decoder, CSR interface, and C intrinsics complete (encodings validated) ✅
- Next: MLIR passes, RoCC spdot_bsr prototype, end-to-end sim
