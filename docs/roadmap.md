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
 - Multi-spec: extended `sattn.spec` choices (BLG, N:M, topk_per_query, LSH) and carried through to backends (per-spec lowerings for RVV and RoCC registered; RVV kernels/stubs implemented)
 - MLIR → RVV indices/BSR path hooked; unified artifacts emission used across tools (RoCC sim consumes desc; RVV runner consumes indices)
 - Additional sparse attention types implemented: sliding-window (incl. dilated/ring), landmark, block-local + global, N:M structured, top-k per query, LSH

## Recent progress (since last update)
 - RVV: vectorized helpers for gather/scatter and block reductions; cycles bench updated.
 - RoCC: index RAM + Q/K scratchpads + spdot/softmax/spmm stubs with checksums; counters wired to toggle events; harness reads non-zero MMIO counters.
 - MLIR: CMake wiring against system LLVM/MLIR; `sattn-opt` minimal tool built and smoke-tested; compile+sim script added with fallback.
 - Tooling: descriptor emission fixed; indices emitter corrected; Docker image updated to build and run all flows.
 - Multi-spec: added SelectSpec pass with simple cost model (window span threshold) and pipelines registered in `sattn-opt`; tests added for RoCC/RVV flows.
 - Selector extended: now considers keep_ratio, block_size cache-fit, and window span; added tests to validate selection flips.
 - New spec: added `block_local_global` selection when `global_tokens` is present; verified in both RoCC and RVV pipelines.
 - Selector overrides: `force_spec` attr and env vars (`SATTN_FORCE_SPEC`, `SATTN_DISABLE_SW`, `SATTN_DISABLE_BSR`); per-spec hooks add `blg_enabled` in lowered ops.
 - Specs expanded: selector recognizes `nm_structured` (nm_n/nm_m) and `topk_per_query` (topk_k); lowerings add `nm_enabled`/`topk_enabled`.
 - LSH added: selector recognizes `lsh_buckets` → `spec = lsh`; lowerings add `lsh_enabled`.
 - RVV: added kernels and helpers for SW/TopK (BLG), N:M (wrapper), LSH (bucketed); segmented reductions and softmax-row helpers; CPU references and compare tests all passing.
 - RVV runner and MLIR bridge: `sattn_rvv_runner` dispatches kernels by spec; `sattn_run_rvv_from_mlir.py` lowers, parses `sattn.rvv_call`, and runs the runner; tests for SW/BLG/N:M/LSH pass.
 - RoCC counters refined: per-cycle gather/mac increments and DMA byte counting; pytest asserts non-zero counters.
 - RVV vectorization (expanded): tiled variants for `sliding_window`, `block_local_global` and `lsh`; runner prints counters and supports `--tile_rows`.
 - Quantization (RVV): added bf16, int8, and int4 paths for `sliding_window` and `block_local_global`; runner supports `--precision` and `--scale_{q,k,v}_x1000`; MLIR bridge accepts `precision`/`scale_*` attributes and forwards to runner; tests added.
 - Selector tests and heuristics: added checks (FileCheck/substring) for spec flips; selector now also considers `gqa_group_size` and `comp_block_size`.
 - RVV bridge autotune: `sattn_run_rvv_from_mlir.py` now forwards `--autotune` to the runner; test added.
 - RVV metrics snapshot: script `scripts/update_rvv_metrics_table.py` auto-updates a table in `backends/rvv/README.md` with current proxy counters.
 - RoCC sim GQA/comp: MMIO for `gqa_group_size`/`comp_block_size` plus tests asserting iteration changes; testbench prints these in `spec_info`.
 - MLIR → RVV indices path: block-based specs now emit indices and the RVV runner can consume them via `--indices` (with global token expansion). Added test to exercise the path.
 - Docs updated: `spec_selection.md` documents precision, GQA/compression knobs and selector influence; `backends/rvv/README.md` includes metrics table and new flags.
 - Unified artifacts helper: added `compiler/mlir/tools/sattn_emit_artifacts.py` and migrated sim/RVV scripts to use it for indices + desc emission (single source of truth).
 - Per-spec roundtrip checks: added MLIR-driven tests that run both RVV and RoCC sim for sliding_window, block_local_global, nm_structured, LSH, and landmark.
 - RoCC functional check vs CPU reference: compile+sim test asserts checksum PASS for representative specs.
 - Grouped-query & compression blocks:
   - RVV: `gqa_group_size` shares selection across heads; `comp_block_size` enables compression-block scoring; bridged from MLIR and exercised in tests.
   - RoCC sim: new MMIOs for `gqa_group_size` and `comp_block_size`; simple latency model reflects their effect; tests assert cycle changes.
   - Selector: heuristics penalize sliding-window when GQA > 1 and discount block selection when compression blocks are smaller than selection blocks.

## Remaining Work (prioritized)
1) MLIR production integration
   - Expand pipelines with real tiling/vectorization/bufferization and backend-specific ops; keep FileCheck tests green
2) Multi-spec support and selection
   - Add hardware capability probe integration and refine per-model heuristics
3) RVV performance path
   - Broaden vectorization across kernels; tile-level softmax/fused paths; benchmark vs scalar — initial tiled variants done
   - Add simple autotune hooks in runner (tile_rows sweep); validate on Spike/QEMU-RVV or dev board; maintain bandwidth/compute counters in benches
   - Quantization docs: optional heuristics beyond calibration
   - Evaluate impact of GQA/compression settings on bandwidth/compute counters — keep validating on device
4) RoCC compute datapath
   - Flesh out functional pipelines beyond stubs (gather/DMA timing, MAC utilization, softmax tile)
   - Compare to CPU reference per-tile; utilization and DMA efficiency profiling
5) Packaging/UX
   - CLI for profiles/specs and selection; Python packaging; CI for CPU (CUDA optional) and sim
   - Stabilize C API for all specs and precisions (documented headers + versioning)
   - Python bindings: minimal ctypes wheel and a pybind11-backed module with typed wrappers
   - PyTorch extension: custom op exposing sparse attention specs; contiguous `[B,H,L,D]` tensors
   - MLIR examples: reference IR snippets + bridge scripts for E2E integration from Python
   - Integration docs: end-to-end examples (C/C++, Python, PyTorch, MLIR) and packaging notes

## Milestone tracking
- Stage 3–5 focus: RVV vectorization and one RoCC primitive in RTL with sim match to CPU
- Stage 6: MLIR real passes + autotuner integration
- Stage 8: Expanded evaluation matrix and plots

## Next actions (short)
 - Multi-spec: add a lightweight hardware capability probe and fold into selector; keep tests green (selector env hints added; HW probe for RoCC surfaced via MMIO)
 - E2E: maintain unified artifacts flow; add minor cleanup and examples in docs
 - RVV: on-device run (or precise sim) to collect real cycles; extend metrics table with “on-device” column
 - RoCC: refine counters and stub fidelity; keep invariants tests
