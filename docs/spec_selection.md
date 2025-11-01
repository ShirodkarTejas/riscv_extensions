## Spec selection: attributes and commands

### How selection works
- If `global_tokens > 0` is set on `sattn.sparse_attention`, the selector sets `spec = "block_local_global"`.
- If `nm_n` and `nm_m` are set, the selector sets `spec = "nm_structured"` (unless `SATTN_DISABLE_NM`).
- If `topk_k` is set, the selector sets `spec = "topk_per_query"` (unless `SATTN_DISABLE_TOPK`).
- If `lsh_buckets` is set, the selector sets `spec = "lsh"` (unless `SATTN_DISABLE_LSH`).
- Else, the selector compares a simple cost model:
  - sliding_window uses `window_size`, `tile_S`, `tile_D` (and adds any `global_tokens` if present)
  - bsr uses `keep_ratio`, `block_size`, and `tile_M`, `tile_S`, `tile_D`
- If neither cost can be computed, defaults to `spec = "bsr"`.

### Attributes the selector reads
- `global_tokens: i64` → forces `block_local_global` if > 0
- `nm_n: i64`, `nm_m: i64` → forces N:M structured selection
- `topk_k: i64` → forces top-k per query selection
- `lsh_buckets: i64` → forces LSH hashed buckets selection
- `window_size: i64` → influences sliding_window cost
- `keep_ratio: f64`, `block_size: i64` → influence bsr cost
- `tile_S: i64`, `tile_M: i64`, `tile_D: i64` → size hints for the model

Example (unlowered):
```
module {
  "sattn.sparse_attention"() { global_tokens = 8 : i64, block_size = 64 : i64, tile_M = 16 : i64, tile_D = 32 : i64, tile_S = 128 : i64 } : () -> ()
}
```

### Pipelines and expected propagation
- RoCC pipeline: `-pass-pipeline=builtin.module(sattn-lower-rocc)`
- RVV pipeline: `-pass-pipeline=builtin.module(sattn-lower-rvv)`
- The selected `spec` is propagated to `sattn.rocc_call`/`sattn.rvv_call`.

### Commands
- Run pipelines directly:
```
build/mlir/tools/sattn-opt/sattn-opt input.mlir --allow-unregistered-dialect -pass-pipeline=builtin.module(sattn-lower-rocc)
build/mlir/tools/sattn-opt/sattn-opt input.mlir --allow-unregistered-dialect -pass-pipeline=builtin.module(sattn-lower-rvv)
```

- Run RVV kernels directly from MLIR (end-to-end):
```
/opt/venv/bin/python compiler/mlir/tools/sattn_run_rvv_from_mlir.py --mlir input.mlir
# Example MLIR (sliding_window):
# module {
#   "sattn.sparse_attention"() { window_size = 8 : i64, tile_D = 32 : i64, tile_S = 128 : i64 } : () -> ()
# }
# Output: spec=sliding_window checksum=...
```

### Precision and quantization (attribute-driven)

You can select compute precision via attributes on `sattn.sparse_attention` (preferred) or `sattn.rvv_call`. Supported values and optional per-tensor scales:

- `precision`: one of `"fp32"`, `"bf16"`, `"i8"`, `"i4"`
- `scale_q`: float (optional; used for `i8`/`i4`), symmetric per-tensor
- `scale_k`: float (optional; used for `i8`/`i4`), symmetric per-tensor
- `scale_v`: float (optional; used for `i8`/`i4`), symmetric per-tensor

Behavior:
- If `precision` is omitted, defaults to `fp32`.
- If `precision` is `i8`/`i4` and scales are omitted, reasonable defaults are used in the RVV kernels.
- Attributes are preserved through lowering (`sattn.sparse_attention` → `sattn.rvv_call`), and the MLIR→RVV bridge forwards them to the RVV runner.

Examples:

Sliding window with bfloat16:
```
module {
  "sattn.sparse_attention"() { precision = "bf16", window_size = 8 : i64, tile_D = 32 : i64, tile_S = 128 : i64 } : () -> ()
}
```

Sliding window with int8 and explicit per-tensor scales:
```
module {
  "sattn.sparse_attention"() { precision = "i8", scale_q = 0.05 : f32, scale_k = 0.05 : f32, scale_v = 0.05 : f32, window_size = 8 : i64, tile_D = 32 : i64, tile_S = 128 : i64 } : () -> ()
}
```

### Calibrating scales

To derive symmetric per-tensor scales (`i8`/`i4`) from synthetic data consistent with the RVV runner (sine/cosine initialization), use the calibration tool:

```
/opt/venv/bin/python compiler/mlir/tools/sattn_calibrate_scales.py --mlir input.mlir --precision i8
# Output: calibrate: precision=i8 scale_q=... scale_k=... scale_v=... scale_q_x1000=... ...
```

You can then add `scale_q/scale_k/scale_v` to your MLIR or pass `--scale_*_x1000` to the runner.

Quantization defaults and guidance

- Recommended symmetric per‑tensor scales for int8: `scale_q = scale_k = scale_v = 0.05` (good starting point for normalized activations). For int4, start with `0.10–0.125`.
- Use calibration to derive dataset‑specific scales from representative samples.
- Relative tolerance guide for checksum comparisons: bf16 ≈ 1e‑2, int8 ≈ 5e‑2, int4 ≈ 1e‑1.

### Grouped-query sharing and compression blocks

Attributes understood by both RVV and RoCC paths:

- `gqa_group_size: i64` — number of query heads that share a single key/value cache (grouped-query attention). Selection is shared across heads in the group.
- `comp_block_size: i64` — optional compression block size used to form intermediate attention scores; these are mapped/pooled into selection blocks. Set 0 to disable (default).

Runner flags:

```
--gqa_group_size N        # e.g., 2
--comp_block_size C       # e.g., 8
```

These attributes propagate through lowering. The RVV path uses them to share block selection across heads (GQA) and to optionally use compression-block scoring before selection. The RoCC sim reflects them in a simple latency model and prints them in `spec_info`.

Selector influence (heuristic):
- If `gqa_group_size > 1`, the selector penalizes `sliding_window` (more per-head work) so `block_local_global` is preferred when otherwise close.
- If `comp_block_size < block_size`, the selector discounts `block_local_global` cost (cheaper importance scoring), nudging selection toward block-based specs.

Hardware hints (environment variables):
- `SATTN_HW_L1_BYTES`: override the L1 size used in the block cache-fit heuristic (default 32768). Larger values favor block-based selection.
- `SATTN_PREFER_BSR`: nudge the selector to prefer block-based specs (lowers BSR cost, raises sliding-window cost slightly).
- `SATTN_PREFER_SW`: nudge the selector to prefer sliding-window.
- Existing probes: `SATTN_DISABLE_SW`, `SATTN_DISABLE_BSR`, `SATTN_DISABLE_NM`, `SATTN_DISABLE_TOPK`, `SATTN_DISABLE_LSH`.

### Tooling helpers

Unified artifacts emitter (used by both RVV and simulation flows):

```
/opt/venv/bin/python compiler/mlir/tools/sattn_emit_artifacts.py --mlir input.mlir --out-stem indices
# emits indices.txt and indices.desc and patches optional attrs (nm/topk/lsh/keep_ratio, gqa_group_size, comp_block_size)
```

The RVV runner bridge and the RoCC simulation wrapper both call this helper, keeping a single source of truth for emitted indices and descriptor fields.

Selector preference flags (hardware hints)

Both the RVV MLIR bridge and the RoCC compile+sim wrapper accept flags to hint the selector without changing code:

```
# RVV bridge
python3 compiler/mlir/tools/sattn_run_rvv_from_mlir.py --mlir input.mlir \
  [--prefer-bsr] [--prefer-sw] [--l1-bytes 65536] [--use-hw-probe]

# RoCC compile+sim
python3 compiler/mlir/tools/sattn_compile_and_sim.py --mlir input.mlir \
  [--prefer-bsr] [--prefer-sw] [--l1-bytes 65536] [--use-hw-probe]
```

These map to selector environment variables (`SATTN_PREFER_BSR`, `SATTN_PREFER_SW`, `SATTN_HW_L1_BYTES`). With `--use-hw-probe`, the tool runs the RoCC sim briefly to read capability bits and sets `SATTN_DISABLE_BSR`/`SATTN_DISABLE_SW` accordingly before running selection.

One-command profile wrapper

```
python3 scripts/sattn_profile.py --mlir input.mlir --backend rvv \
  [--prefer-bsr|--prefer-sw] [--l1-bytes 65536] [--use-hw-probe] [--autotune]

python3 scripts/sattn_profile.py --mlir input.mlir --backend sim \
  [--prefer-bsr|--prefer-sw] [--l1-bytes 65536] [--use-hw-probe]
```

This wrapper forwards flags to the appropriate tools and prints the resulting counters/outputs.

Python API wrappers

```python
from sattn import run_rvv_from_mlir, compile_and_sim
# RVV
out = run_rvv_from_mlir('input.mlir', prefer_bsr=True, autotune=True)
print(out)
# RoCC sim
compile_and_sim('input.mlir', use_hw_probe=True)
```

Install (editable):
```
cd python && python3 -m pip install -e .
```

### Dilated/ring sliding-window (new)

Attributes:
- `dilation: i64` (>=1) — stride between neighboring tokens in the window; defaults to 1.
- `wrap: i64` (0/1) — if set, indices wrap around the sequence (ring mode).

Runner flags:
```
--dilation N
--wrap 1
```

These affect the sliding-window kernels in the RVV path. Ring mode samples `2*window_size+1` positions per row with wrap-around; dilation spaces those positions by the given stride.

### Landmark attention (new)

Attributes:
- `num_landmarks: i64` — number of landmark tokens to attend over (evenly spaced, simple variant).
- `landmark_iters: i64` — optional refinement iterations (k-means-lite) to improve landmark centroids.

Runner flag:
```
--landmarks N
--landmark_iters I
```

Current implementation uses evenly spaced representatives and attends over them (compressed attention). This provides a baseline for landmark-style sparsity.

- End-to-end compile+sim wrapper:
```
/opt/venv/bin/python compiler/mlir/tools/sattn_compile_and_sim.py --mlir input.mlir
```
This emits `indices.txt` and `indices.desc` (now includes `global_tokens` if present) and runs `hw/sim/obj_dir/Vrocc_sattn`.

### Current limitations
- Explicit user override of `spec` is not yet supported (selector will choose automatically).
- Per-spec lowering differences are not yet implemented; only attribute propagation and selection are in place.
 
### Overrides and probe flags
- Attribute override: set `spec = "..."` on `sattn.sparse_attention` (or `force_spec = "..."`) to bypass selection.
- Env override: set `SATTN_FORCE_SPEC=bsr|sliding_window|block_local_global`.
- Probe flags: `SATTN_DISABLE_SW=1`, `SATTN_DISABLE_BSR=1`, `SATTN_DISABLE_NM=1`, `SATTN_DISABLE_TOPK=1`, `SATTN_DISABLE_LSH=1` to bias selection away from an unsupported pattern.

### Per-spec hooks in lowering
- When `spec = "block_local_global"`, lowerings add `blg_enabled = true` on `sattn.rocc_call`/`sattn.rvv_call` so backends can specialize.
- When `spec = "nm_structured"`, lowerings add `nm_enabled = true`.
- When `spec = "topk_per_query"`, lowerings add `topk_enabled = true`.
- When `spec = "lsh"`, lowerings add `lsh_enabled = true`.

