# Evaluation Scripts

Run quick evaluations of latency and accuracy deltas.

Examples:

CPU, block_topk at L∈{2k,8k,32k}:
```bash
python bench/eval/eval_sparse_attention.py --pattern block_topk --device cpu --lengths 2048 8192 32768 \
  --block_size 64 --keep_ratio 0.12 --global_tokens 16
```

GPU (CUDA), sliding_global:
```bash
python bench/eval/eval_sparse_attention.py --pattern sliding_global --device cuda --lengths 2048 8192 32768 \
  --window_size 512 --global_tokens 16 --block_size 64
```

Outputs CSV at `bench/results/eval_summary.csv` with latency and (for small L) mean absolute delta vs dense.
