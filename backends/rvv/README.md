# RVV Baseline — Sparse Attention (Sliding-Global)

This directory contains a portable baseline implementation targeting RISC-V Vector (RVV). The initial drop uses a scalar core loop with hooks for RVV toolchains and a cycles harness.

## Build (example)

Using riscv64-unknown-elf-gcc or clang with RVV (adjust march/abi):

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_C_COMPILER=riscv64-unknown-elf-gcc \
  -DCMAKE_C_FLAGS="-O3 -march=rv64gcv -mabi=lp64d"
cmake --build . -j
```

Run the bench on Spike or your RVV board:

```bash
# Spike/QEMU load paths differ; on a board just run the ELF
./sattn_rvv_bench
./sattn_rvv_bench_blocktopk
# CLI bench for sliding_window (choose sizes)
./sattn_rvv_bench_cli --B 1 --H 2 --L 1024 --D 64 --window 16
```

Output:

```
cycles=1234567 B=1 H=2 L=256 D=64 window=16
checksum=...  # sanity
```

## Files
- `include/sparse_attention_rvv.h`: C API (shapes, params, rdcycle)
- `src/sparse_attention_rvv.c`: sliding-global baseline, numerically stable softmax
- `bench/measure_cycles.c`: minimal harness printing cycles and checksum
- `CMakeLists.txt`: static lib + bench

## Next steps
- Replace inner loops with RVV intrinsics (vectorized dot/reduction) — PARTIALLY DONE for sliding_global
- Add segmented reductions and block gather/scatter utilities
- Extend to block_topk (preselect indices) and fused softmax path
