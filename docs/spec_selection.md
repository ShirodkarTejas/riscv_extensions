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

