# Profiles

Suggested knobs per profile.

## --spec highperf
- Target: GPU or custom accelerator
- Pattern: block_topk or sliding_global (whichever gives lower latency)
- Precision: bf16/fp16
- Knobs: block_size ∈ {64,128}, keep_ratio ~ 0.12, global_tokens 16; window_size ~ 512

## --spec lowpower
- Target: RVV + int8 (future), higher sparsity
- Pattern: block_topk
- Precision: bf16 initially, int8 later (PTQ)
- Knobs: block_size 64, keep_ratio 0.08, global_tokens 8

## --spec imc
- Target: IMC cost model guidance
- Pattern: block_topk or sliding_global
- Precision: bf16 (consider LUT/log-domain softmax)
- Knobs: choose keep_ratio to fit array rows/cols; use `imc/neurosim/mapping.py` to size S
