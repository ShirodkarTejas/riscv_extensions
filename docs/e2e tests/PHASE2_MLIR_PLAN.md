# Phase 2: MLIR Compiler Pipeline - Implementation Plan

**Status**: 🚧 In Progress  
**Goal**: Enable automatic code generation from high-level models to optimized RISC-V code

---

## 🎯 Objectives

### Primary Goals
1. **Enhance MLIR Dialect** - Support all 5 patterns × 4 precisions
2. **Implement Lowering Passes** - Transform high-level ops → RVV primitives
3. **Integration Testing** - Validate MLIR → C code generation
4. **PyTorch → MLIR Export** - Connect ML frameworks to compiler

### Success Criteria
- ✅ All 5 patterns representable in MLIR
- ✅ All 4 precisions supported (fp32, bf16, i8, i4)
- ✅ Lowering passes generate correct RVV code
- ✅ Integration tests pass
- ✅ PyTorch models exportable to MLIR

---

## 📋 Phase 2 Tasks

### Task 1: Enhance MLIR Dialect (Week 1)

**Current State**:
- Basic `sattn.sparse_attention` op defined
- Supports `block_topk` and `sliding_global` patterns
- Limited precision support

**Enhancements Needed**:

#### 1.1 Add Missing Patterns
- ✅ `sliding_window` (already supported as `sliding_global`)
- ✅ `block_topk` (already supported as `block_topk`)  
- 🔲 `nm_structured` - N:M structured sparsity
- 🔲 `lsh` - Locality-sensitive hashing
- 🔲 `landmark` - Landmark compression

#### 1.2 Add Precision Support
- ✅ `fp32` (implicit default)
- ✅ `bf16` (already in dialect)
- 🔲 `i8` - Int8 quantization
- 🔲 `i4` - Int4 quantization

#### 1.3 Add Quantization Attributes
- 🔲 `scale_q`, `scale_k`, `scale_v` for i8/i4
- 🔲 `zero_point` for asymmetric quantization

#### 1.4 Pattern-Specific Attributes
- 🔲 `nm_n`, `nm_m` for N:M structured
- 🔲 `buckets` for LSH
- 🔲 `num_landmarks` for landmark

**Deliverables**:
- Enhanced `SattnOps.td` with all patterns
- Updated dialect tests
- Example MLIR for each pattern

---

### Task 2: Implement Core Lowering Passes (Weeks 2-3)

#### 2.1 Pattern Selection Pass
**Input**: High-level `sattn.sparse_attention` op  
**Output**: Specific pattern op with optimal parameters  
**Implementation**: `SelectSpec.cpp` (already exists)

**Enhancements**:
- 🔲 Cost model for all 5 patterns
- 🔲 Hardware-aware selection (RVV vs RoCC)
- 🔲 Auto-tuning integration

#### 2.2 Quantization Pass
**Input**: FP32 ops  
**Output**: Quantized ops (bf16/i8/i4)  
**Implementation**: NEW - `Quantize.cpp`

**Features**:
- 🔲 Calibration-based scale selection
- 🔲 Symmetric/asymmetric quantization
- 🔲 Per-tensor vs per-channel scales

#### 2.3 Lowering to RVV Pass
**Input**: High-level sparse attention op  
**Output**: RVV intrinsic calls or func calls  
**Implementation**: `LowerToRVV.cpp` (already exists)

**Enhancements**:
- 🔲 All 5 patterns supported
- 🔲 All 4 precisions supported
- 🔲 Optimal tile sizes
- 🔲 Generate calls to our C functions

#### 2.4 Bufferization Pass
**Input**: Tensor-based ops  
**Output**: Memref-based ops  
**Implementation**: `Bufferize.cpp` (already exists)

**Enhancements**:
- 🔲 Efficient buffer allocation
- 🔲 In-place updates where possible
- 🔲 Alignment for RVV

---

### Task 3: Code Generation (Week 3)

#### 3.1 C Code Emission
**Goal**: Generate C code that calls our RVV functions

**Example Pipeline**:
```
MLIR Op → Lowered MLIR → Func Call → C Code
```

**Generated Code Template**:
```c
// Generated from MLIR
void sparse_attention_generated(
    const float* Q, const float* K, const float* V, float* O,
    int B, int H, int L, int D
) {
    sattn_shape_t shape = {B, H, L, D};
    sattn_params_t params = {.window_size = 16};
    
    sattn_rvv_sliding_global_i8(
        Q, K, V, O, shape, params,
        0.008f, 0.008f, 0.008f  // Quantization scales
    );
}
```

#### 3.2 MLIR → C Lowering
**Implementation**: Extend `LowerToRVV.cpp`

**Approach**:
- 🔲 Emit `func.call` to external C functions
- 🔲 Generate struct initialization code
- 🔲 Handle tensor → memref conversion
- 🔲 Emit quantization code if needed

---

### Task 4: PyTorch Integration (Week 4)

#### 4.1 PyTorch → MLIR Export
**Goal**: Export PyTorch models to MLIR

**Approach**:
```python
# PyTorch model
model = SparseTransformer(...)

# Export to MLIR
mlir_module = torch_mlir.compile(
    model, 
    example_input,
    output_type="linalg-on-tensors"
)

# Lower to sattn dialect
sattn_module = lower_to_sattn_dialect(mlir_module)

# Generate C code
c_code = generate_c_from_mlir(sattn_module)
```

**Implementation**:
- 🔲 Pattern matching for sparse attention
- 🔲 Attribute extraction (window_size, etc.)
- 🔲 Shape inference
- 🔲 Type conversion

#### 4.2 Custom Op Registration
**Goal**: Register `SparseAttentionLayer` as custom MLIR op

**Implementation**:
```python
@torch.jit.script
def sparse_attention_pattern(Q, K, V):
    # Annotate for MLIR lowering
    return torch.ops.sattn.sparse_attention(Q, K, V, "sliding_window", 16)
```

---

### Task 5: Integration Testing (Week 4)

#### 5.1 End-to-End Tests
**Test Flow**: PyTorch → MLIR → C → QEMU → Validation

**Test Cases**:
1. **Pattern Coverage**: Test all 5 patterns
2. **Precision Coverage**: Test all 4 precisions
3. **Size Coverage**: Test L ∈ {32, 64, 128, 256}
4. **Accuracy**: Compare with Python reference

#### 5.2 Performance Testing
**Metrics**:
- Compilation time (PyTorch → C)
- Code size (generated C)
- Runtime performance (vs handwritten C)
- Memory usage

---

## 🏗️ Implementation Strategy

### Week 1: Dialect Enhancement
- **Day 1-2**: Update `SattnOps.td` with all patterns
- **Day 3**: Add precision and quantization attributes
- **Day 4-5**: Write dialect tests, update examples

### Week 2: Core Lowering
- **Day 1-2**: Implement quantization pass
- **Day 3-4**: Enhance `LowerToRVV.cpp` for all patterns
- **Day 5**: Integration with existing passes

### Week 3: Code Generation
- **Day 1-2**: Implement C code emission
- **Day 3-4**: Generate calls to RVV functions
- **Day 5**: Test generated code vs handwritten

### Week 4: PyTorch Integration
- **Day 1-2**: PyTorch → MLIR export
- **Day 3-4**: Custom op registration
- **Day 5**: End-to-end integration tests

---

## 📊 Deliverables

### Code
1. **Enhanced Dialect** - `compiler/mlir/dialect/SattnOps.td`
2. **Quantization Pass** - `compiler/mlir/transforms/Quantize.cpp`
3. **Enhanced Lowering** - `compiler/mlir/transforms/LowerToRVV.cpp`
4. **PyTorch Bridge** - `python/torch_to_mlir.py`
5. **Integration Tests** - `compiler/mlir/tests/test_phase2_*.py`

### Documentation
1. **MLIR Dialect Guide** - `compiler/mlir/DIALECT_GUIDE.md`
2. **Lowering Pass Guide** - `compiler/mlir/LOWERING_GUIDE.md`
3. **PyTorch Integration** - `compiler/mlir/PYTORCH_INTEGRATION.md`
4. **Phase 2 Complete** - `compiler/mlir/PHASE2_COMPLETE.md`

### Examples
1. **Per-Pattern MLIR** - `compiler/mlir/examples/<pattern>.mlir`
2. **Generated C Code** - `compiler/mlir/examples/<pattern>_generated.c`
3. **PyTorch Models** - `compiler/mlir/examples/torch_*.py`

---

## 🎯 Success Metrics

### Functional
- ✅ All 5 patterns × 4 precisions representable in MLIR
- ✅ Lowering passes generate valid C code
- ✅ Generated code produces correct outputs
- ✅ PyTorch models export successfully

### Performance
- 🎯 Compilation time < 1s for typical models
- 🎯 Generated code within 10% of handwritten performance
- 🎯 Code size comparable to handwritten

### Quality
- 🎯 100% test coverage for dialect ops
- 🎯 All integration tests passing
- 🎯 Documentation complete

---

## 🔄 Integration with Phase 1

Phase 2 builds on Phase 1 by:
1. **Using Python Bindings** - Generated C code calls Phase 1 functions
2. **Leveraging Validation** - Use same test framework
3. **Same Performance** - No overhead from code generation
4. **Compatibility** - Works with existing benchmarks

**Architecture**:
```
PyTorch Model
    ↓ (torch_mlir)
MLIR (Linalg/Tensor)
    ↓ (pattern matching)
MLIR (sattn dialect)
    ↓ (lowering passes)
MLIR (func calls to RVV)
    ↓ (C code generation)
C Code calling Phase 1
    ↓ (compile + link)
RISC-V Binary
    ↓ (QEMU or hardware)
Execution + Validation
```

---

## 🚀 Quick Start (After Implementation)

### Compile a PyTorch Model
```bash
python compiler/mlir/tools/compile_from_pytorch.py \
    --model SparseTransformer \
    --pattern sliding_window \
    --precision i8 \
    --output generated_model.c
```

### Use Generated Code
```c
#include "generated_model.h"
#include "backends/rvv/include/sparse_attention_rvv.h"

int main() {
    float Q[1*8*128*64], K[...], V[...], O[...];
    sparse_attention_generated(Q, K, V, O, 1, 8, 128, 64);
    return 0;
}
```

### Validate Output
```bash
./scripts/validate_generated_code.py \
    --generated generated_model.c \
    --reference python/torch_sparse_attention.py
```

---

## 📚 References

- **MLIR Documentation**: https://mlir.llvm.org/
- **Torch-MLIR**: https://github.com/llvm/torch-mlir
- **Our Phase 1**: `docs/e2e tests/E2E_PHASE1_COMPLETE.md`
- **Existing MLIR**: `compiler/mlir/transforms/README.md`

---

## 🎓 Key Decisions

### 1. Why MLIR?
- **Extensibility**: Easy to add custom dialects
- **Optimization**: Rich transformation infrastructure
- **Integration**: Works with PyTorch, TensorFlow, JAX
- **Targeting**: Can lower to multiple backends

### 2. Why Not Direct PyTorch → C?
- **Flexibility**: MLIR enables pattern-specific optimizations
- **Reusability**: Same dialect for RVV, RoCC, other targets
- **Maintainability**: Easier to add new patterns/optimizations
- **Debugging**: MLIR IR is human-readable

### 3. Code Generation vs JIT?
- **Ahead-of-Time (chosen)**: Generates C code for deployment
- **Pros**: No runtime overhead, easy debugging, works everywhere
- **Cons**: Requires recompilation for changes
- **Alternative**: Could add JIT later if needed

---

**Status**: Ready to implement! Starting with dialect enhancement...

