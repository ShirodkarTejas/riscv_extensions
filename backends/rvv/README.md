# RVV Baseline — Sparse Attention (Multi‑Spec)

This directory contains a portable baseline implementation targeting RISC-V Vector (RVV). It includes scalar baselines with RVV vectorized inner loops where applicable, plus a cycles/counters harness and CPU reference comparators.

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
