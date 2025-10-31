# Roadmap and Remaining Work

This document summarizes what exists in the repo and what remains to reach a "multi-spec" production-ready sparse attention library.

## Done (MVP)
- CPU reference kernels; tests vs dense; randomized tests
- GPU: Triton sliding-window; block-topk via Triton gather + Torch prepass; CUDA tests
- RVV: baseline scalar implementation + cycles harness
- MLIR: `sattn` dialect + example; pass emulator tool (`sattn_opt.py`) and docs
- RoCC: MMIO RTL skeleton; Verilator harness; runtime driver; spdot_bsr issue path
- IMC: proxy mapping tool producing sparse vs dense estimates
- Bench: microbench, autotune, eval
- Docs: quickstart, profiles, shapes/contract, scope decisions

## Remaining Work (prioritized)
1) MLIR production integration
   - Implement actual MLIR passes (materialize-indices, tile, fuse-softmax, lower-to-{rvv,rocc})
   - CMake build; register passes with `mlir-opt`; add tests
2) RVV performance path
   - Replace scalar loops with RVV intrinsics (vector dot, segmented reductions, tile softmax)
   - Validate on Spike/QEMU-RVV or dev board; add counters for bandwidth/energy proxy
3) RoCC compute datapath
   - Implement spdot_bsr pipeline with gather/DMA and MAC array; tile-local softmax_fused; spmm_bsr
   - End-to-end compare to CPU reference; utilization and DMA efficiency profiling
4) Training + quantization
   - Backward pass for key paths; gradient tests; bf16/fp16 numerics
   - Int8 PTQ (KL/percentile calibration); optional int4 for V path
5) GPU selection path
   - Move block_topk selection from Torch prepass to Triton (topk_idx) or fused approach
6) Packaging/UX
   - Real CLI for profiles/specs; Python packaging (pyproject); CI for CPU (CUDA optional)
7) Additional patterns (optional)
   - BigBird-style variations; softmax-free experiments

## Milestone tracking
- Stage 3–5 focus: RVV vectorization and one RoCC primitive in RTL with sim match to CPU
- Stage 6: MLIR real passes + autotuner integration
- Stage 8: Expanded evaluation matrix and plots
