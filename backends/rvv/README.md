# RVV Baseline — Sparse Attention (Multi‑Spec)

This project contains a portable baseline implementation targeting RISC-V Vector (RVV). It includes scalar baselines with RVV vectorized inner loops where applicable, plus a cycles/counters harness and CPU reference comparators.

## Build (example)

Using riscv64-unknown-elf-gcc or clang with RVV (adjust march/abi):

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_C_COMPILER=riscv64-unknown-elf-gcc \
  -DCMAKE_C_FLAGS="-O3 -march=rv64gcv -mabi=lp64d"
cmake --build . -j
```

Run the benches on Spike/QEMU or your RVV board:

```bash
# Spike/QEMU load paths differ; on a board just run the ELF
./sattn_rvv_bench                  # sliding_window baseline
./sattn_rvv_bench_blocktopk        # block_topk (BLG) baseline
./sattn_rvv_bench_cli --B 1 --H 2 --L 1024 --D 64 --window 16

# CPU vs RVV compare tools (sanity):
./sattn_rvv_compare_sw             # sliding_window compare
./sattn_rvv_compare_blocktopk      # block_topk compare
./sattn_rvv_compare_nm             # N:M wrapper compare
./sattn_rvv_compare_lsh            # LSH bucketed compare

# Spec-driven runner (dispatches to kernel by --spec)
./sattn_rvv_runner --spec sliding_window --L 128 --D 32 --window 8
./sattn_rvv_runner --spec sliding_window --L 128 --D 32 --window 8 --tile_rows 4   # tiled variant
./sattn_rvv_runner --spec block_local_global --L 128 --D 32 --block_size 64 --keep_x1000 120 --global_tokens 8
./sattn_rvv_runner --spec block_local_global_tiled --L 128 --D 32 --block_size 64 --keep_x1000 120 --global_tokens 8   # tiled BLG
./sattn_rvv_runner --spec nm_structured --L 128 --D 32 --nm_n 2 --nm_m 4
./sattn_rvv_runner --spec lsh --L 128 --D 32 --lsh_buckets 8

# Runner prints counters as well:
# spec=sliding_window checksum=... rvv_bytes_read=... bytes_written=... mac_flops=...

# Simple autotune for tile_rows (minimizes rvv_bytes_read):
./sattn_rvv_runner --spec sliding_window --L 512 --D 64 --window 16 --autotune
./sattn_rvv_runner --spec block_local_global --L 512 --D 64 --block_size 64 --keep_x1000 120 --autotune
```

### Precision and scales

Select precision and optional per-tensor scales via runner flags:

```
# bfloat16
./sattn_rvv_runner --spec sliding_window --L 128 --D 32 --window 8 --precision bf16

# int8 with symmetric scales (x1000)
./sattn_rvv_runner --spec sliding_window --L 128 --D 32 --window 8 --precision i8 \
  --scale_q_x1000 50 --scale_k_x1000 50 --scale_v_x1000 50

# int4 with symmetric scales (x1000)
./sattn_rvv_runner --spec block_local_global --L 128 --D 32 --block_size 16 --keep_x1000 120 \
  --precision i4 --scale_q_x1000 125 --scale_k_x1000 125 --scale_v_x1000 125

### Grouped-query sharing and compression blocks

```
# Shared selection across heads in a group (GQA)
./sattn_rvv_runner --spec block_local_global --L 128 --D 32 --block_size 16 --gqa_group_size 2

# Use compression blocks (compute on pooled keys), then map to selection blocks
./sattn_rvv_runner --spec block_local_global --L 128 --D 32 --block_size 16 --comp_block_size 8

Quick sweep script (bytes and flops vs GQA/comp):

```
python3 scripts/rvv_util_sweep.py --L 128 --D 32 --block_size 16 --keep_x1000 120
# Prints a small markdown table with bytes_read/bytes_written/mac_flops across settings
```

Tuning guidance:

- `gqa_group_size`: Sharing selection across heads (2–4) typically reduces repeated K/V traffic and compute. Start with 2, validate quality, then try 4. Larger groups may reduce flexibility for head‑specialization.
- `comp_block_size`: Using compression blocks below `block_size` reduces importance‑score work. A good starting point is `comp_block_size = block_size/2` (or 8–16). If selection quality drops, increase toward `block_size`.
- Use the sweep script to compare bytes and flops across a small grid for your L/D/block_size; pick the lowest bytes_read that preserves acceptable checksum/quality in your end‑to‑end checks.
```

Calibrate suggested scales (synthetic data matching runner initialization):

```
/opt/venv/bin/python compiler/mlir/tools/sattn_calibrate_scales.py --mlir examples/sw_simple.mlir --precision i8
# calibrate: precision=i8 scale_q=... scale_k=... scale_v=... scale_q_x1000=...
```
```

Output:

```
cycles=1234567 B=1 H=2 L=256 D=64 window=16
checksum=...  # sanity
spec=... checksum=... rvv_bytes_read=... bytes_written=... mac_flops=... [rvv_cycles=N]
```

## Files
- `include/sparse_attention_rvv.h`: C API (shapes, params, rdcycle)
- `src/sparse_attention_rvv.c`: SW/TopK (BLG), N:M wrapper, LSH stub; vectorized dot/axpy helpers; segmented reductions; softmax-row helper
- `bench/measure_cycles.c`: minimal sliding_window harness printing cycles and checksum
- `CMakeLists.txt`: static lib + bench
- `bench/compare_*_ref.c`: CPU reference vs RVV comparators for SW/TopK/N:M/LSH
- `bench/rvv_runner.c`: spec-driven runner dispatching to kernels

## Notes
- Vectorization: sliding_window uses RVV dot/axpy; other paths may mix scalar + vector helpers
- Helpers: segmented reductions and softmax-row available for reuse
- Counters: proxy bandwidth/compute counters exposed via C API (`sattn_rvv_counters_get`)

## Status and metrics (current)

Supported attention types and precisions, plus emitted metrics (reproducible via the runner):

| Spec                         | Kernel path                    | Tiled | fp32 | bf16 | i8  | i4  | Metrics emitted                 |
|------------------------------|--------------------------------|-------|------|------|-----|-----|---------------------------------|
| sliding_window               | `sattn_rvv_sliding_global`     | Yes   | Yes  | Yes  | Yes | Yes | bytes_read/written, mac_flops   |
| block_local_global (block_topk) | `sattn_rvv_block_topk`      | Yes   | Yes  | Yes  | Yes | Yes | bytes_read/written, mac_flops   |
| nm_structured (wraps block_topk) | `sattn_rvv_nm_structured` | —     | Yes  | Yes  | Yes | Yes | bytes_read/written, mac_flops   |
| lsh (bucketed)               | `sattn_rvv_lsh`                | Yes   | Yes  | —    | —   | —   | bytes_read/written, mac_flops   |

Reproduce sample metrics:

```
# sliding_window fp32
./sattn_rvv_runner --spec sliding_window --L 128 --D 32 --window 8

# sliding_window int8 with scales
./sattn_rvv_runner --spec sliding_window --L 128 --D 32 --window 8 --precision i8 \
  --scale_q_x1000 50 --scale_k_x1000 50 --scale_v_x1000 50

# block_local_global int4 with scales
./sattn_rvv_runner --spec block_local_global --L 128 --D 32 --block_size 16 --keep_x1000 120 \
  --precision i4 --scale_q_x1000 125 --scale_k_x1000 125 --scale_v_x1000 125

# autotune tiled variant (minimizes bytes_read)
./sattn_rvv_runner --spec sliding_window --L 512 --D 64 --window 16 --autotune
```

Each run prints a single summary line including the kernel spec, checksum, and proxy counters, e.g.:

```
spec=sliding_window checksum=272.034012 rvv_bytes_read=398336 bytes_written=138240 mac_flops=32512
```

Calibration (optional, for i8/i4 scales):

```
/opt/venv/bin/python compiler/mlir/tools/sattn_calibrate_scales.py --mlir examples/sw_simple.mlir --precision i8
# calibrate: precision=i8 scale_q=... scale_k=... scale_v=... scale_q_x1000=...
```

### Snapshot metrics

Current proxy metrics collected in this environment (runner on host, no RVV hardware):

<!-- metrics:start -->
| Spec | L | D | RVV bytes_read | bytes_written | mac_flops |
|---|---|---|---|---|---|
| sliding_window | 128 | 32 | 824320 | 285696 | 67328 |
| sliding_window i8 | 128 | 32 | 0 | 0 | 0 |
| block_local_global | 128 | 32 | 3211264 | 327680 | 606208 |
| block_local_global i4 | 128 | 32 | 0 | 0 | 0 |

<!-- metrics:end -->

“On-device” metrics (rdcycle) are TBD and will be added after running on RVV hardware. RoCC simulation counters (cycles/bytes) are available in the compiler docs and sim output.

---

## Integration

Use the stack at the attention-operator boundary. Provide Q/K/V tensors shaped `[B,H,L,D]`, choose a spec/precision, and receive O with the same shape. Pick the path that matches your environment.

### Option A — C API (C/C++)

Link the RVV static library and call a kernel:

```c
#include "sparse_attention_rvv.h"

void run_blg(const float* Q,const float* K,const float* V,float* O,
             int64_t B,int64_t H,int64_t L,int64_t D) {
  sattn_shape_t s = { .B=B, .H=H, .L=L, .D=D };
  sattn_blocktopk_params_t p = { .block_size=64, .keep_ratio=0.12f,
                                 .global_tokens=8, .gqa_group_size=1, .comp_block_size=0 };
  // fp32 path
  sattn_rvv_block_topk(Q, K, V, O, s, p);
  // or quantized variants (bf16/i8/i4) if scales provided
}
```

Notes:
- Inputs/outputs are contiguous row‑major `[B,H,L,D]` floats. Quantized paths accept fp32 inputs plus per‑tensor scales.
- For block specs you can pass `--indices` via the runner (see Option C) or compute selection inside your app.

### Option B — Python bindings (planned)

- Minimal ctypes wheel exposing the C API (drop-in for quick prototyping)
- Pybind11 module with typed wrappers and spec enums
- Packaging: `pip install sattn-rvv` with prebuilt wheels where feasible

### Option C — PyTorch custom op (planned)

- Custom op: `sattn_sparse_attention(q, k, v, *, spec=..., block_size=..., keep_ratio=..., precision=..., scales=...) -> o`
- Tensors must be contiguous `[B,H,L,D]` on CPU; deployment handles RVV execution
- Optional integration with indices emission for block specs

### Option D — MLIR route (research/E2E)

- Emit `sattn.sparse_attention` in MLIR with shapes/knobs
- Run selector/lowering; for block specs emit `indices.txt` + `.desc`
- Execute:
  - RVV: `compiler/mlir/tools/sattn_run_rvv_from_mlir.py --mlir <file> [--autotune] [--precision ... --scale_*]`
  - RoCC: `compiler/mlir/tools/sattn_compile_and_sim.py --mlir <file>`

### Choosing precision and scales

- fp32 default; bf16/i8/i4 supported
- For i8/i4, either supply scales or run:

```bash
/opt/venv/bin/python compiler/mlir/tools/sattn_calibrate_scales.py --mlir examples/sw_simple.mlir --precision i8
```

### Integration examples (planned)

- End‑to‑end examples in C/C++, Python, and PyTorch
- Reference MLIR snippets and bridge scripts

---

# Explain it like I'm Five

### What this project is
- A multi-spec sparse attention stack with:
  - MLIR front-end (`sattn` dialect, passes, selector, lowerings)
  - Backends: RVV (RISC‑V Vector) kernels and a spec-driven runner; RoCC RTL sim path
  - CPU/GPU baselines and tests for verification
  - Unified tooling to emit artifacts (indices/desc), run, and collect metrics

### Capabilities (specs and knobs)
- Specs: sliding_window (incl. dilated/ring), block_local_global (block_topk), nm_structured, topk_per_query, lsh, landmark.
- Knobs influencing selection/impl: window_size, keep_ratio, block_size, global_tokens, gqa_group_size, comp_block_size, dilation, wrap.
- Precisions: fp32, bf16, int8, int4 (with per-tensor scales).
- Env hints: force/disable/prefer spec, simple HW capacity hints (e.g., L1 size) to bias selection.

### High-level data flow
```
Author input (MLIR + shapes/knobs)
  → SelectSpec pass picks spec using heuristics (keep, window span, cache-fit, GQA, comp, env hints)
  → Lowering inserts backend-specific call ops and per-spec toggles (e.g., blg_enabled, nm_enabled)
  → Unified artifacts emission (for block specs): indices.txt (+ desc) from MLIR
  → Backend execution:
      - RVV: spec-driven runner dispatches correct kernel (+ optional tiled, quantized variants)
      - RoCC: Verilator sim consumes indices/desc via driver
  ← Outputs: checksum/metrics (bytes read/written, MAC flops), counters, logs
```

### End-to-end MLIR workflow
- Write a `sattn.sparse_attention` op (tile sizes, spec hints/knobs).
- Run passes:
  - SelectSpec chooses `spec`.
  - Lowering to RVV/RoCC inserts attributes and per-spec enables.
- For block-based specs, tooling emits `indices.txt` (plus `*.desc`).
- Launch backend:
  - RVV path: bridge script parses lowered IR, forwards flags to `sattn_rvv_runner`, optionally `--indices`, `--precision`, scales, `--tile_rows`, `--autotune`.
  - RoCC path: compile-and-sim script runs passes, emits artifacts, runs Verilator with indices + desc.

### RVV backend workflow
- `sattn_rvv_runner`:
  - Parses CLI flags derived from MLIR attrs.
  - Dispatches by `--spec` to kernels:
    - sliding_window (fp32/bf16/i8/i4; tiled variant)
    - block_local_global (and nm/topk wrappers; quantized variants; tiled)
    - lsh, landmark
  - Optional `--indices` to consume precomputed selections; `--autotune` sweeps tile_rows.
  - Emits proxy counters: bytes_read/written, mac_flops, plus checksums.

### RoCC sim workflow
- Lowered MLIR → unified artifacts (indices + descriptor) → Verilator harness.
- Driver programs MMIO, feeds indices/desc, reads counters; tests assert invariants.

### Verification and benchmarking
- CPU/GPU references (Triton/CUDA and CPU) compare against RVV/RTL outputs.
- Unit tests for selector choices, per-spec hooks, indices paths, RVV tiled/quantized paths.
- Metrics snapshot tooling updates RVV README table; autotune finds tile_rows minimizing bandwidth.

### Quantization and tuning
- Precisions controlled from MLIR/CLI; scales via CLI or calibration helper.
- Tiled variants and `--autotune` improve bandwidth/compute balance.

### Summary
- The system maps a high-level sparse attention op to the most suitable sparsity pattern using a principled, hardware-aware selector; it then lowers to concrete backends, emits required block-selection artifacts when needed, and runs optimized kernels/sim paths while reporting reproducible metrics. Verification is anchored by CPU/GPU references and unit tests, making the stack research-friendly for exploring sparsity policies, quantization, and vectorization trade-offs.

- Key outcomes: multi-spec coverage; end-to-end from MLIR to RVV/RTL; reproducible counters; knobs for modeling grouped queries and compression blocks; and quantization-ready paths with calibration.

- Pending emphasis: production-grade MLIR tiling/bufferization, refined selector HW probes, and deeper RoCC functional pipelines/cpu-ref checks.

- In short: a modular, testable pipeline that selects, lowers, and executes sparse attention across multiple specs and backends, with unified artifact generation and measurable performance signals.