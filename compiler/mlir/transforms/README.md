# SATTN MLIR Transforms

This directory documents the planned transformation passes for lowering `sattn.sparse_attention` to concrete backends (RVV or RoCC).

Passes (planned):
- sattn-materialize-indices
  - If pattern is dynamic (block_topk or sliding_global without precomputed indices), generate per-row block indices (BSR) as SSA values.
- sattn-tile
  - Choose tile sizes `{M, D, block_size, k_blocks}`; introduce explicit tiled loops/ops and bufferization-friendly layouts.
- sattn-fuse-softmax
  - Fuse scale + max-reduce + exp + sum-reduce + normalize into a tile-local op; enable mapping to `softmax_fused` primitive.
- sattn-lower-to-rvv
  - Convert tiled ops into vector ops with gather/scatter and segmented reductions suitable for RVV intrinsics.
- sattn-lower-to-rocc
  - Convert tiled ops into calls/ops that correspond to custom primitives (`spdot_bsr`, `spmm_bsr`, `softmax_fused`, etc.).

Example pipeline:
```
python compiler/mlir/tools/sattn_opt.py \
  --passes materialize-indices,fuse-softmax,lower-to-rvv \
  --in compiler/mlir/examples/sattn_example.mlir --out /tmp/out.mlir
```

Autotuning:
- A script will grid-search `{block_size, keep_ratio, tile}` and emit the best configs per sequence length and target.
