# i8/i4 Kernel Optimization Plan

**Goal**: Reduce cycle count for quantized kernels to match or beat bf16 performance while maintaining energy savings.

**Current State** (L=64, D=16):
- fp32: 4.1M cycles, 43 µJ
- bf16: **2.7M cycles** (fastest!), 23 µJ
- i8: 4.3M cycles (59% slower than bf16), 9.4 µJ ✅ (2.4x less energy)
- i4: 3.9M cycles (44% slower than bf16), 7.8 µJ ✅ (2.9x less energy)

**Target**: i8/i4 at ~2.7M cycles = **Same latency as bf16 with 2-3x energy savings!**

---

## 🔍 Root Cause Analysis

### Why Are i8/i4 Slower?

Let me check the actual implementations to identify bottlenecks:

**Potential Issues**:
1. **Quantization overhead** - Converting fp32 ↔ i8/i4 in inner loops
2. **Poor vectorization** - Not using RVV efficiently for INT operations
3. **Scalar operations** - Packing/unpacking may be scalar instead of vectorized
4. **Memory alignment** - Sub-byte data (i4) causing unaligned accesses
5. **Branch overhead** - Extra conditionals in quantized paths

---

## 📊 Investigation Steps

### Step 1: Profile Hot Spots
```bash
# Run with profiling to find bottlenecks
docker exec sattn_rvv_dev qemu-riscv64 \
  -cpu rv64,v=true,vlen=128,elen=64 \
  -d in_asm \
  build/rvv-riscv64/sattn_rvv_runner \
  --spec sliding_window --L 64 --D 16 \
  --precision i4 > profile_i4.log

# Compare with bf16
docker exec sattn_rvv_dev qemu-riscv64 \
  -cpu rv64,v=true,vlen=128,elen=64 \
  -d in_asm \
  build/rvv-riscv64/sattn_rvv_runner \
  --spec sliding_window --L 64 --D 16 \
  --precision bf16 > profile_bf16.log

# Analyze instruction counts
grep -c "vmul" profile_bf16.log
grep -c "vmul" profile_i4.log
```

### Step 2: Examine Current Implementation
Files to review:
- `backends/rvv/src/sparse_attention_rvv.c`
  - `sattn_rvv_sliding_global_i8()`
  - `sattn_rvv_sliding_global_i4()`
  - `sattn_rvv_sliding_global_bf16()`

Look for:
- [ ] Are quantization operations inside hot loops?
- [ ] Are we using vector instructions for quant/dequant?
- [ ] Are loads/stores properly aligned?
- [ ] Are there unnecessary branches?

---

## 🚀 Optimization Strategies

### Strategy 1: Vectorize Quantization (Quick Win)

**Current** (likely scalar):
```c
// Inside loop - BAD!
for (int d = 0; d < D; d++) {
    int8_t q_val = quantize_scalar(Q[d], scale);  // Scalar!
    // ... use q_val
}
```

**Optimized** (vectorized):
```c
// Before loop - GOOD!
int8_t Q_i8[D] __attribute__((aligned(16)));
quantize_vector_rvv(Q, Q_i8, D, scale);  // Vector operation!

// Inside loop - use pre-quantized data
for (int d = 0; d < D; d++) {
    int8_t q_val = Q_i8[d];  // Already quantized!
}
```

**Expected Impact**: 30-40% reduction in cycles

---

### Strategy 2: Use RVV Integer SIMD Instructions

**Key RVV instructions for i8/i4**:
- `vmul.vv` - Vector multiply (works on i8)
- `vmacc.vv` - Vector multiply-accumulate
- `vwadd` - Widening add (i8 → i16)
- `vnclipu` - Narrowing clip (i16 → i8)

**Example optimized dot product**:
```c
// i8 dot product using RVV
int32_t dot_product_i8_rvv(const int8_t* a, const int8_t* b, size_t n) {
    int32_t sum = 0;
    size_t vl;
    
    for (size_t i = 0; i < n; ) {
        vl = vsetvl_e8m1(n - i);  // Set vector length for i8
        
        vint8m1_t va = vle8_v_i8m1(&a[i], vl);  // Load i8 vector
        vint8m1_t vb = vle8_v_i8m1(&b[i], vl);
        
        vint16m2_t vp = vwmul_vv_i16m2(va, vb, vl);  // Widening multiply
        vint32m4_t vacc = vwredsum_vs_i16m2_i32m1(vp, vzero, vl);  // Reduce
        
        sum += vmv_x_s_i32m1_i32(vacc);
        i += vl;
    }
    return sum;
}
```

**Expected Impact**: 40-50% reduction in cycles

---

### Strategy 3: Optimize i4 Packing/Unpacking

**Current** (likely bit manipulation in loop):
```c
// Slow - per-element bit ops
for (int i = 0; i < n; i++) {
    uint8_t packed = packed_data[i / 2];
    int8_t val = (i % 2) ? (packed >> 4) : (packed & 0x0F);  // Branch!
}
```

**Optimized** (vectorized unpacking):
```c
// Fast - vector unpacking
void unpack_i4_to_i8_rvv(const uint8_t* packed, int8_t* unpacked, size_t n) {
    for (size_t i = 0; i < n/2; i += vl) {
        vl = vsetvl_e8m1(n/2 - i);
        
        vuint8m1_t vpacked = vle8_v_u8m1(&packed[i], vl);
        
        // Extract low nibbles
        vuint8m1_t vlow = vand_vx_u8m1(vpacked, 0x0F, vl);
        // Extract high nibbles
        vuint8m1_t vhigh = vsrl_vx_u8m1(vpacked, 4, vl);
        
        // Store interleaved
        vsse8_v_u8m1(&unpacked[i*2], 2, vlow, vl);
        vsse8_v_u8m1(&unpacked[i*2+1], 2, vhigh, vl);
    }
}
```

**Expected Impact**: 20-30% reduction in cycles for i4

---

### Strategy 4: Fuse Operations

**Combine quantize + compute**:
```c
// Instead of: quantize → load → compute → dequantize
// Do: load → quantize+compute (fused) → store

// Fused quantized attention score
float compute_attention_score_i8_fused(
    const float* Q, const float* K, int D,
    float scale_q, float scale_k) {
    
    float score = 0.0f;
    
    for (int d = 0; d < D; ) {
        vl = vsetvl_e32m1(D - d);
        
        vfloat32m1_t vq = vle32_v_f32m1(&Q[d], vl);
        vfloat32m1_t vk = vle32_v_f32m1(&K[d], vl);
        
        // Quantize on-the-fly (fused)
        vfloat32m1_t vq_scaled = vfmul_vf_f32m1(vq, scale_q, vl);
        vfloat32m1_t vk_scaled = vfmul_vf_f32m1(vk, scale_k, vl);
        
        // Multiply and accumulate (stay in FP32, avoid i8 round-trip)
        vfloat32m1_t vprod = vfmul_vv_f32m1(vq_scaled, vk_scaled, vl);
        score += vfredosum_vs_f32m1_f32m1(vprod, vzero, vl);
        
        d += vl;
    }
    return score;
}
```

**Expected Impact**: 15-25% reduction in cycles

---

## 📋 Implementation Plan

### Phase 1: Quick Wins (2-4 hours)

**Task 1.1**: Move quantization outside hot loops
- [ ] Identify all loops doing quantization per-iteration
- [ ] Pre-quantize Q, K, V before main attention loop
- [ ] Measure cycle improvement

**Task 1.2**: Add vector alignment hints
- [ ] Add `__attribute__((aligned(16)))` to arrays
- [ ] Use `vle8_v_i8m1` instead of scalar loads
- [ ] Verify with `-S` assembly output

**Expected**: 20-30% cycle reduction

---

### Phase 2: Vectorize Integer Operations (1 day)

**Task 2.1**: Implement vectorized i8 dot product
- [ ] Create `dot_product_i8_rvv()` helper
- [ ] Use `vwmul` for widening multiply
- [ ] Replace scalar loops in attention computation

**Task 2.2**: Implement vectorized i4 pack/unpack
- [ ] Create `unpack_i4_to_i8_rvv()` helper
- [ ] Use vector shuffle/shift operations
- [ ] Benchmark vs scalar version

**Expected**: 30-40% additional cycle reduction

---

### Phase 3: Fuse Operations (1 day)

**Task 3.1**: Fused quantized attention
- [ ] Combine quantize + score computation
- [ ] Minimize data movement
- [ ] Keep intermediate results in vector registers

**Task 3.2**: Optimize memory access patterns
- [ ] Tile for cache efficiency
- [ ] Prefetch next blocks
- [ ] Minimize write-after-read hazards

**Expected**: 10-20% additional cycle reduction

---

### Phase 4: Validation (4 hours)

**Task 4.1**: Benchmark all patterns
- [ ] Run comprehensive benchmarks
- [ ] Compare cycle counts: before vs after
- [ ] Verify accuracy unchanged (checksum)

**Task 4.2**: Update documentation
- [ ] Regenerate plots
- [ ] Update benchmark reports
- [ ] Document optimization techniques

---

## 🎯 Success Criteria

### Target Metrics (L=64, D=16):

| Precision | Current Cycles | Target Cycles | Current Energy | Target Energy |
|-----------|----------------|---------------|----------------|---------------|
| bf16      | 2.7M          | 2.7M          | 23 µJ          | 23 µJ         |
| **i8**    | **4.3M** ❌   | **≤2.7M** ✅  | 9.4 µJ ✅      | 9.4 µJ ✅     |
| **i4**    | **3.9M** ❌   | **≤2.7M** ✅  | 7.8 µJ ✅      | 7.8 µJ ✅     |

**Ultimate Goal**: 
- ✅ i8/i4 match bf16 latency (~2.7M cycles)
- ✅ i8/i4 keep 2-3x energy advantage
- ✅ Result: **Same speed, 3x less energy!** 🎉

---

## 📈 Expected Results

**After optimization**:

```
Pattern: sliding_window (L=64, D=16)
├── fp32:  4.1M cycles, 43 µJ
├── bf16:  2.7M cycles, 23 µJ
├── i8:    2.7M cycles, 9.4 µJ  ← 2.4x less energy than bf16!
└── i4:    2.7M cycles, 7.8 µJ  ← 2.9x less energy than bf16!
```

**Impact**: Ultra-low-power devices can now have **real-time latency** with **massive energy savings**!

---

## 🛠️ Tools & Commands

### Build with optimization flags:
```bash
docker exec sattn_rvv_dev cmake -B build/rvv-riscv64 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_FLAGS="-O3 -march=rv64gcv -ffast-math"
```

### Profile instruction counts:
```bash
# Get instruction trace
docker exec sattn_rvv_dev qemu-riscv64 -d in_asm,cpu \
  build/rvv-riscv64/sattn_rvv_runner --spec sliding_window \
  --L 64 --D 16 --precision i8 2>&1 | grep -E "vmul|vle|vse" | wc -l
```

### Benchmark before/after:
```bash
# Before optimization
./scripts/run_benchmark_in_docker.sh --L 128 --D 32

# After optimization
./scripts/run_benchmark_in_docker.sh --L 128 --D 32

# Compare
python bench/generate_comprehensive_report.py \
  --input bench/results/comprehensive_docker_results.json
```

---

## 📝 Next Steps

1. **Start with Task 1.1** (move quantization outside loops) - quickest win
2. **Measure improvement** after each change
3. **Document assembly differences** for learning
4. **Share results** - this could be a great optimization case study!

**Estimated Time**: 2-3 days for full optimization
**Expected Speedup**: 1.4-1.6x (bringing i8/i4 to parity with bf16)
**Energy Benefit**: Maintain 2-3x energy advantage = **same latency, 3x battery life!**

---

**Ready to start?** Let's begin with profiling the current i8 implementation to find the hottest spots! 🔥

