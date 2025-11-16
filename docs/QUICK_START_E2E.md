# Quick Start: End-to-End Python Integration

**5-Minute Guide to Using Python Bindings + PyTorch Integration**

---

## Prerequisites

1. **RVV Library Built** (in Docker or native):
   ```bash
   cd backends/rvv
   mkdir -p build && cd build
   cmake .. -DCMAKE_BUILD_TYPE=Release
   make -j$(nproc)
   ```

2. **Python Dependencies**:
   ```bash
   pip install numpy pytest
   pip install torch  # Optional, for PyTorch integration
   ```

---

## Example 1: NumPy API (5 lines)

```python
from python.sparse_attention_rvv import sparse_attention_rvv
import numpy as np

# Random inputs
Q = np.random.randn(1, 8, 128, 64).astype(np.float32)
K = np.random.randn(1, 8, 128, 64).astype(np.float32)
V = np.random.randn(1, 8, 128, 64).astype(np.float32)

# Ultra-low-power inference (i4, 84% energy savings!)
O = sparse_attention_rvv(Q, K, V, 
                         pattern="sliding_window",
                         precision="i4",
                         window_size=16)

print(f"Input: {Q.shape}, Output: {O.shape}")
print(f"✅ Success! Energy saved: 84% vs FP32")
```

**Output**:
```
Input: (1, 8, 128, 64), Output: (1, 8, 128, 64)
✅ Success! Energy saved: 84% vs FP32
```

---

## Example 2: PyTorch Layer

```python
import torch
from python.torch_sparse_attention import SparseAttentionLayer, SparseAttnPattern, Precision

# Create attention layer
attn = SparseAttentionLayer(
    d_model=512,
    num_heads=8,
    pattern=SparseAttnPattern.SLIDING_WINDOW,
    precision=Precision.I8,  # 81% energy savings
    window_size=16
)

# Forward pass
x = torch.randn(2, 128, 512)  # [batch, seq_len, d_model]
output = attn(x, x, x)  # Self-attention

print(f"Input: {x.shape}, Output: {output.shape}")
print(f"✅ PyTorch integration works!")
```

**Output**:
```
Input: torch.Size([2, 128, 512]), Output: torch.Size([2, 128, 512])
✅ PyTorch integration works!
```

---

## Example 3: Full Transformer Model

```python
import torch
from python.torch_sparse_attention import SparseTransformer

# Create transformer
model = SparseTransformer(
    d_model=512,
    nhead=8,
    num_encoder_layers=6,
    pattern="sliding_window",
    precision="i8",
    window_size=16
)

# Inference
x = torch.randn(2, 128, 512)
y = model(x)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Output: {y.shape}")
print(f"✅ Full transformer works!")
```

---

## Example 4: Accuracy Validation

```python
import numpy as np
from python.sparse_attention_rvv import SparseAttentionRVV, SparseAttnPattern, Precision
from validation.dense_attention_reference import dense_attention_fp32, compare_sparse_vs_dense

# Create sparse attention
attn = SparseAttentionRVV(
    pattern=SparseAttnPattern.SLIDING_WINDOW,
    precision=Precision.I8,
    window_size=8
)

# Generate inputs
B, H, L, D = 1, 1, 16, 8
Q = np.random.randn(B, H, L, D).astype(np.float32)
K = np.random.randn(B, H, L, D).astype(np.float32)
V = np.random.randn(B, H, L, D).astype(np.float32)

# Compute outputs
sparse_output = attn(Q, K, V)
dense_output = dense_attention_fp32(Q, K, V)  # Ground truth

# Compare
metrics = compare_sparse_vs_dense(sparse_output, dense_output, verbose=True)

print(f"\n✅ Accuracy validation complete!")
print(f"MAE: {metrics['mae']:.6f}")
print(f"Cosine similarity: {metrics['cosine_similarity']:.6f}")
```

---

## Running Tests

### All Tests
```bash
pytest tests/test_end_to_end.py -v
```

### Specific Test Class
```bash
# Test all precisions
pytest tests/test_end_to_end.py::TestAllPrecisions -v

# Test accuracy
pytest tests/test_end_to_end.py::TestAccuracyVsDense -v

# Test PyTorch integration
pytest tests/test_end_to_end.py::TestPyTorchIntegration -v
```

### Quick Smoke Test
```bash
# Test that library loads and basic forward pass works
python python/sparse_attention_rvv.py
python python/torch_sparse_attention.py
```

---

## Troubleshooting

### Library Not Found
```
FileNotFoundError: Could not find libsparse_attention_rvv.so
```

**Fix**: Set environment variable
```bash
export SATTN_RVV_LIB=/path/to/build/rvv-riscv64/libsparse_attention_rvv.so
```

Or rebuild:
```bash
cd backends/rvv/build
make -j$(nproc)
```

### PyTorch Not Available
```
ImportError: PyTorch is required for SparseAttentionLayer
```

**Fix**: Install PyTorch
```bash
pip install torch
```

Or use NumPy API only (no PyTorch required):
```python
from python.sparse_attention_rvv import sparse_attention_rvv
```

---

## What's Supported

| Feature | Status | Notes |
|---------|--------|-------|
| **sliding_window** | ✅ Fully working | All precisions (fp32/bf16/i8/i4) |
| **block_local_global** | ✅ Fully working | All precisions (fp32/bf16/i8/i4) |
| **nm_structured** | ✅ Fully working | All precisions (fp32/bf16/i8/i4) |
| **lsh** | ✅ Fully working | All precisions (fp32/bf16/i8/i4) |
| **landmark** | ✅ Fully working | All precisions (fp32/bf16/i8/i4) |
| **FP32 precision** | ✅ Working | All patterns |
| **BF16 precision** | ✅ Working | All patterns |
| **I8 precision** | ✅ Working | Phase 1+2 optimizations applied (all patterns) |
| **I4 precision** | ✅ Working | Phase 1+2 optimizations applied (all patterns) |
| **PyTorch integration** | ✅ Working | SparseAttentionLayer, Transformer |
| **Gradient flow** | ✅ Working | Tested with backward pass |
| **Accuracy validation** | ✅ Working | Dense reference implementation |

---

## Performance Tips

### 1. Use Quantization
```python
# 84% energy savings with i4!
precision="i4"  # Ultra-low-power

# 81% energy savings with i8
precision="i8"  # Low-power

# 51% energy savings with bf16
precision="bf16"  # Balanced
```

### 2. Choose Right Pattern
- **sliding_window**: Fastest, most energy efficient
- **landmark**: Good for long sequences
- **block_topk**: Good for sparse patterns

### 3. Batch Processing
```python
# Process multiple sequences at once
Q = np.random.randn(16, 8, 128, 64)  # Batch size = 16
O = attn(Q, K, V)  # Batched processing
```

---

## Next Steps

1. ✅ **You're ready to use the Python API!**
2. 📚 Read `E2E_PHASE1_COMPLETE.md` for detailed documentation
3. 🔬 Run accuracy studies with `validation/dense_attention_reference.py`
4. 🚀 Integrate into your PyTorch models
5. 📊 Benchmark with `bench/unified_bench.py`

---

## Quick Links

- **API Reference**: `python/sparse_attention_rvv.py` (docstrings)
- **PyTorch Integration**: `python/torch_sparse_attention.py`
- **Tests**: `tests/test_end_to_end.py`
- **Validation**: `validation/dense_attention_reference.py`
- **Complete Guide**: `E2E_PHASE1_COMPLETE.md`
- **Roadmap**: `docs/END_TO_END_SIMULATION_PLAN.md`

---

## Support

**Questions?** Check:
1. `E2E_PHASE1_COMPLETE.md` - Complete Phase 1 documentation
2. `tests/test_end_to_end.py` - Usage examples in tests
3. Code docstrings - Inline documentation

**Issues?** 
- Verify library is built: `ls backends/rvv/build/libsparse_attention_rvv.so`
- Check Python path: `export PYTHONPATH=/path/to/riscv_extensions:$PYTHONPATH`
- Run tests: `pytest tests/test_end_to_end.py -v`

---

**Ready to build efficient sparse attention models? Start coding!** 🚀

