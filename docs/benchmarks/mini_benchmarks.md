# Mini Benchmarks (Stage 0)

This suite validates latency, an energy proxy, and simple accuracy deltas for early iterations.

## Targets and modes
- Models: LLM small head, ViT block
- Sequence lengths: 2048, 8192, 32768
- Modes: Dense, block_topk, sliding_global (global tokens on/off)
- Precisions: bf16 (primary), fp16 (alt)

## Metrics
- Latency
  - ms/token (decoder) or ms/batch (encoder block)
  - End-to-end per attention op
- Energy proxy
  - CPU: RAPL/Estimator; GPU: NVML power-integral; RVV/RoCC: cycles × tech-factor (documented)
- Memory traffic (bytes/token) via counters or model-based estimates
- Accuracy deltas (sanity tasks)
  - LLM: masked-LM or next-token perplexity on a small corpus subset
  - ViT: attention map cosine similarity vs dense and 1-shot classification delta on a small split

## Configuration knobs swept
- block_size ∈ {32, 64, 128}
- keep_ratio ∈ {0.08, 0.12, 0.16, 0.24}
- window_size ∈ {256, 512, 1024}
- global_tokens ∈ {0, 8, 16, 32}

## Output
- CSV per run with columns: model, L, mode, precision, knobs..., latency_ms, energy_proxy, bytes_token, acc_delta
- Plots: latency vs keep_ratio; energy proxy vs L; acc_delta vs sparsity
