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

## Status
- CPU ref, GPU sliding-window + block-topk, RVV baseline ready
- MLIR dialect/pass stubs, RoCC skeleton, IMC proxy ready
- Next: MLIR passes, RoCC spdot_bsr prototype, end-to-end sim
