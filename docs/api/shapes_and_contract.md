# SparseAttention API: Shapes, Layouts, Masks, Errors

## Tensors and layout
- Q, K, V: [B, H, L, D] with heads-contiguous layout
  - B: batch, H: heads, L: sequence length, D: head dimension
  - Strides must be consistent with contiguous heads (H inner before L)
- Output O: [B, H, L, D]

## Sparse patterns and params
- block_topk
  - params: block_size (tokens), keep_ratio (0–1), global_tokens (int)
- sliding_global
  - params: window_size (tokens per side), global_tokens (int)

## Indices and masks
- Primary format: BSR (blocked sparse row) for K/V selection
- Adapters: CSR/COO accepted, internally converted to BSR
- Dynamic selection: optional; may require pre-pass kernels

## Numerics and stability
- Log-sum-exp softmax: per-row max subtraction, accumulation in fp32
- Mixed-precision: compute in bf16/fp16, accumulate in fp32
- Softmax-free mode may be added later for exploration

## Error handling
- Shape mismatch: any of B/H/L/D mismatches across Q/K/V
- Unsupported precision: anything other than {bf16, fp16} in initial drop
- Invalid params: non-positive block_size/window_size; keep_ratio∉(0,1]
- Stride/layout: non-contiguous heads or inconsistent strides

## Determinism
- Numerically deterministic within tolerance per precision
- Randomized selection (if used) must be seeded and reproducible
