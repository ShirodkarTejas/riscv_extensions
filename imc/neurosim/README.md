# NeuroSim Mapping (Sparse Attention)

This folder provides a simple translator to produce sparse IMC layer configs and proxy energy/latency estimates.

- `mapping.py`: computes active tokens S and scales dense energy/latency by sparsity; emits CSV rows for quick comparison.

Examples:
```bash
python imc/neurosim/mapping.py --pattern block_topk --L 8192 --D 64 --block_size 64 --keep_ratio 0.12 --global_tokens 16
python imc/neurosim/mapping.py --pattern sliding_global --L 8192 --D 64 --window_size 512 --global_tokens 16 --block_size 64
```

Outputs `imc/neurosim/results_sparse_vs_dense.csv`.

Notes:
- This is a proxy model. For full fidelity, integrate with NeuroSim by generating per-layer config files and feeding measured bandwidths/ADC settings. The sparse benefit is primarily captured via reduced active tokens `S` and reduced MACs; we add a gather overhead term to reflect index/gather cost.
- Consider log-domain/lookup softmax approximations to reduce ADC precision needs.
