# Roadmap and Remaining Work

This document summarizes what exists in the repo and what remains to reach a "multi-spec" production-ready sparse attention library.

## Done (MVP)
- CPU reference kernels; tests vs dense; randomized tests
- GPU: Triton sliding-window; block-topk via Triton gather + Torch prepass; CUDA tests
- RVV: baseline scalar implementation + cycles harness; initial RVV gather/scatter helpers and block reductions
- MLIR: `sattn` dialect + example; pass emulator tool (`sattn_opt.py`) and docs
- RoCC: MMIO RTL skeleton; Verilator harness; runtime driver; spdot_bsr issue path
- IMC: proxy mapping tool producing sparse vs dense estimates
- Bench: microbench, autotune, eval
- Docs: quickstart, profiles, shapes/contract, scope decisions
 - Docker: dev image with Verilator + LLVM/MLIR 18 (Ubuntu 24.04) and Python venv
 - Build: minimal `sattn-opt` built against system MLIR; optional pass test (smoke) wired
 - E2E: `sattn_compile_and_sim.py` with auto-fallback, emits indices/desc and runs Verilator sim

## Recent progress (since last update)
- RVV: vectorized helpers for gather/scatter and block reductions; cycles bench updated.
- RoCC: index RAM + Q/K scratchpads + spdot/softmax/spmm stubs with checksums; counters wired to toggle events; harness reads non-zero MMIO counters.
- MLIR: CMake wiring against system LLVM/MLIR; `sattn-opt` minimal tool built and smoke-tested; compile+sim script added with fallback.
- Tooling: descriptor emission fixed; indices emitter corrected; Docker image updated to build and run all flows.
 - Multi-spec: added SelectSpec pass with simple cost model (window span threshold) and pipelines registered in `sattn-opt`; tests added for RoCC/RVV flows.
 - Selector extended: now considers keep_ratio, block_size cache-fit, and window span; added tests to validate selection flips.
 - New spec: added `block_local_global` selection when `global_tokens` is present; verified in both RoCC and RVV pipelines.
 - Selector overrides: `force_spec` attr and env vars (`SATTN_FORCE_SPEC`, `SATTN_DISABLE_SW`, `SATTN_DISABLE_BSR`); per-spec hooks add `blg_enabled` in lowered ops.

## Remaining Work (prioritized)
1) MLIR production integration
   - Expand pipelines with real tiling/vectorization/bufferization and backend-specific ops; keep FileCheck tests green
   - Hook indices/BSR masks from IR into emitters; unify text/JSON emission
2) Multi-spec support and selection
   - Extend `sattn.spec` choices (block_local_global done; N:M, topk_per_query, LSH pending); carry through to backends (BLG hooks in place)
   - Register per-spec lowering for RVV and RoCC; optional CPU reference for verification (BLG hook added; CPU ref pending)
   - Upgrade selector to a lightweight cost model (extended); add hardware probe; allow override (env/attr overrides added; probe flags minimal)
3) RVV performance path
   - Complete segmented reductions and gather/scatter coverage across specs; vector dot and tile softmax
   - Validate on Spike/QEMU-RVV or dev board; add bandwidth/compute counters to benches
4) RoCC compute datapath
   - Flesh out functional pipelines beyond stubs (gather/DMA timing, MAC utilization, softmax tile)
   - Compare to CPU reference per-tile; utilization and DMA efficiency profiling
5) Additional sparse attention types
   - Sliding-window/dilated; block-local + global tokens; N:M structured; top-k per query; LSH/hashed buckets; (optional) ring/landmark
6) Packaging/UX
   - CLI for profiles/specs and selection; Python packaging; CI for CPU (CUDA optional) and sim

## Milestone tracking
- Stage 3–5 focus: RVV vectorization and one RoCC primitive in RTL with sim match to CPU
- Stage 6: MLIR real passes + autotuner integration
- Stage 8: Expanded evaluation matrix and plots

## Next actions (short)
- MLIR: restore real `sattn-opt` passes and add FileCheck tests; wire pass CI target
- Multi-spec: add `sattn.spec` attribute and support BSR + sliding_window end-to-end; add a simple selector
- RVV: finish segmented reductions + gather/scatter coverage; add e2e tests for the RVV path
- RoCC: keep counters and checksums; incrementally refine gather/DMA/MAC timing
- E2E: add a test target invoking `sattn_compile_and_sim.py` with sample MLIR and asserting PASS
