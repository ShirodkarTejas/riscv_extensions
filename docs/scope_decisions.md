# Stage 0 — Scope & Decisions

This document fixes the initial scope and knobs for the sparse attention operator library. It also defines the mini benchmark suite to validate latency, an energy proxy, and accuracy deltas on simple tasks.

## Sparse patterns (initial)
- block_topk (NSA-style block sparsity)
  - block_size (tokens): 64 (allowed: 32, 64, 128)
  - keep_ratio (per row): 0.12 (range: 0.06–0.25)
  - global_tokens: 16 (range: 0–64)
  - selection: top-k over block scores (mean/max of QKᵀ within block)
- sliding_global (Longformer-like)
  - window_size (tokens per side): 512 (allowed: 256, 512, 1024)
  - global_tokens: 16 (range: 0–64)
  - optional dilation: 1 (allowed: 1, 2)

Notes
- K/V indices are represented in BSR (Blocked Sparse Row). CSR/COO adapters will be provided.
- Masks may be precomputed (inference) or materialized dynamically (selection kernels) depending on backend.

## Numeric modes
- Primary: bf16 (accumulate in fp32), alternative: fp16 (accumulate in fp32)
- Planned: int8 (PTQ; symmetric per-tensor or per-channel for V path), optional int4 (V path only)
- Dense fallback is available for correctness baselines.

## Reference transformer configs
- LLM small head
  - d_model: 1024, n_head: 16, head_dim: 64
- ViT block
  - d_model: 768, n_head: 12, head_dim: 64

## Sequence lengths
- { 2048, 8192, 32768 }

## Operator knobs (per head unless noted)
- block_size (tokens): 64
- keep_ratio: 0.12
- global_tokens: 16
- window_size (sliding_global): 512
- training: false (initially inference-only; backward added later if time permits)
- precision: bf16 (fp16 optional)

## Deliverables for Stage 0
- One-pager spec of knobs: this document
- Mini benchmark list: see docs/benchmarks/mini_benchmarks.md


