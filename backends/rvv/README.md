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
