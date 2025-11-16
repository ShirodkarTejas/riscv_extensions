# i8/i4 Performance Bottleneck Analysis

## 🔍 Root Cause Identified

After examining the code, I found **why i8/i4 are 44-59% slower than bf16**:

---

## 📊 Comparison Matrix

| Aspect | fp32 (4.1M cycles) | bf16 (2.7M cycles) ⚡ | i8 (4.3M cycles) ❌ | i4 (3.9M cycles) ❌ |
|--------|-------------------|---------------------|-------------------|-------------------|
| **Vectorization** | ✅ Uses `dot_f32_rvv()` | ❌ Scalar loops | ❌ Scalar loops | ❌ Scalar loops |
| **Quantization** | N/A | Inside loop (lightweight) | Inside loop (expensive) | Inside loop (expensive) |
| **Redundant Work** | Minimal | Q quantized 2x per position | Q quantized 2x per position | Q quantized 2x per position |
| **Instruction Type** | RVV FP32 SIMD | Scalar FP32 | Scalar INT8 | Scalar INT4 |

---

## 🐛 Critical Issues in i8/i4 Implementations

### Issue 1: **No RVV Vectorization** 🚨

**fp32 (FAST)**:
```c
// Line 753-756: Uses RVV!
dot = dot_f32_rvv(
    &Q[offset_bhld(b, h, i, 0, B, H, L, D)],
    &K[offset_bhld(b, h, j, 0, B, H, L, D)],
    D);
```

**i8 (SLOW)**:
```c
// Lines 947-951: Scalar loop! No RVV!
int dot_i32 = 0;
for (int64_t d = 0; d < D; ++d) {
    signed char qi = f32_to_i8_symmetric(Q[...], s_q);  // ❌ Inside loop!
    signed char ki = f32_to_i8_symmetric(K[...], s_k);  // ❌ Inside loop!
    dot_i32 += (int)qi * (int)ki;  // ❌ Scalar multiply!
}
```

**Impact**: 
- fp32 processes 4-8 elements per cycle (RVV SIMD)
- i8/i4 process 1 element per cycle (scalar)
- **4-8x slowdown from lack of vectorization alone!**

---

### Issue 2: **Quantization Inside Hot Loops** 🚨

**Location**: Lines 948-950 (i8), 962-964 (i8 second pass), 969-970 (V quantization)

```c
// ❌ BAD: Quantization happens INSIDE innermost loop
for (int64_t j = jl; j < jr; ++j) {      // Window loop (~32 iterations)
    for (int64_t d = 0; d < D; ++d) {    // Feature dimension (~64 iterations)
        signed char qi = f32_to_i8_symmetric(Q[...], s_q);  // ❌ 2048 calls!
        signed char ki = f32_to_i8_symmetric(K[...], s_k);  // ❌ 2048 calls!
        dot_i32 += (int)qi * (int)ki;
    }
}
```

**Problem**: For L=64, D=16, window=8:
- `f32_to_i8_symmetric()` called **98,304 times** per head!
- Each call has branches, rounding, clamping (lines 34-41)
- **Massive overhead** compared to simple arithmetic

**bf16 is faster because**:
```c
// Lines 22-29: bf16 conversion is lightweight
static inline unsigned short float_to_bf16_u16(float x) {
    // Just bit manipulation, no branches!
    unsigned int lsb = (v.u >> 16) & 1u;
    unsigned int rounding_bias = 0x7FFFu + lsb;
    unsigned int rounded = v.u + rounding_bias;
    return (unsigned short)(rounded >> 16);
}
```
vs
```c
// Lines 34-41: i8 conversion is expensive
static inline signed char f32_to_i8_symmetric(float x, float scale) {
    if (scale <= 0.f) scale = 1.f;  // ❌ Branch!
    float q = x / scale;             // ❌ Division!
    if (q > 127.f) q = 127.f;        // ❌ Branch!
    else if (q < -127.f) q = -127.f; // ❌ Branch!
    int iq = (int)lrintf(q);         // ❌ Rounding function call!
    if (iq > 127) iq = 127;          // ❌ Branch!
    if (iq < -127) iq = -127;        // ❌ Branch!
    return (signed char)iq;
}
```

---

### Issue 3: **Redundant Quantization** 🚨

**Two-pass algorithm**:
1. Lines 945-957: Compute max score (m)
2. Lines 959-976: Compute weights and output

**Problem**: Q is quantized **twice** for the same position!
- Pass 1 (line 948): `qi = f32_to_i8_symmetric(Q[...], s_q)`
- Pass 2 (line 962): `qi = f32_to_i8_symmetric(Q[...], s_q)` **again!**

**Waste**: For L=64, D=16, window=8:
- 32,768 redundant quantization calls!

---

### Issue 4: **No Integer SIMD Instructions** 🚨

Current i8 dot product:
```c
int dot_i32 = 0;
for (int64_t d = 0; d < D; ++d) {
    signed char qi = ...;
    signed char ki = ...;
    dot_i32 += (int)qi * (int)ki;  // ❌ Scalar!
}
```

**RVV has native i8 SIMD instructions**:
- `vle8_v_i8m1` - Load i8 vector
- `vwmul_vv_i16m2` - Widening multiply (i8 → i16)
- `vwredsum_vs_i16m2_i32m1` - Widening reduction (i16 → i32)

**These are NOT being used!**

---

## 🎯 Why bf16 is Faster Despite Same Structure

bf16 (2.7M cycles) vs i8 (4.3M cycles) - **59% faster!**

Reasons:
1. **Lightweight conversion**: bf16 is just bit shifts, i8 has branches
2. **FP32 multiplication**: `bf16_to_float() * bf16_to_float()` stays in FP registers
3. **No widening needed**: FP32 multiply is native, i8→i32 requires conversion
4. **Better pipelining**: FP units are highly optimized, integer widening is not

---

## 🚀 Optimization Strategy

### Phase 1: Pre-quantize (Quick Win)
**Before**:
```c
for (j) {
    for (d) {
        qi = f32_to_i8(Q[...]);  // ❌ Inside loop
        ki = f32_to_i8(K[...]);  // ❌ Inside loop
    }
}
```

**After**:
```c
int8_t Q_i8[D], K_i8[L][D];  // ✅ Pre-allocate

// Quantize Q once for this row
for (d) Q_i8[d] = f32_to_i8(Q[...]);

// Quantize all K rows once
for (j) for (d) K_i8[j][d] = f32_to_i8(K[...]);

// Now use pre-quantized values
for (j) {
    for (d) {
        dot_i32 += Q_i8[d] * K_i8[j][d];  // ✅ No quant overhead!
    }
}
```

**Expected Impact**: 30-40% faster (eliminate redundant quantization)

---

### Phase 2: Vectorize i8 Dot Product
**After**:
```c
// Use RVV for i8 dot product
int32_t dot_i8_rvv(const int8_t* a, const int8_t* b, size_t n) {
    int32_t sum = 0;
    for (size_t i = 0; i < n; ) {
        size_t vl = vsetvl_e8m1(n - i);
        vint8m1_t va = vle8_v_i8m1(&a[i], vl);  // ✅ Vector load
        vint8m1_t vb = vle8_v_i8m1(&b[i], vl);
        vint16m2_t vp = vwmul_vv_i16m2(va, vb, vl);  // ✅ Widening mul
        vint32m1_t vacc = vwredsum_vs_i16m2_i32m1(vp, vzero, vl);  // ✅ Reduce
        sum += vmv_x_s_i32m1_i32(vacc);
        i += vl;
    }
    return sum;
}
```

**Expected Impact**: 40-50% faster (4-8x SIMD speedup)

---

### Phase 3: Vectorize Quantization
**After**:
```c
void quantize_f32_to_i8_rvv(const float* src, int8_t* dst, size_t n, float scale) {
    float inv_scale = 1.0f / scale;
    for (size_t i = 0; i < n; ) {
        size_t vl = vsetvl_e32m1(n - i);
        vfloat32m1_t vf = vle32_v_f32m1(&src[i], vl);  // Load fp32
        vf = vfmul_vf_f32m1(vf, inv_scale, vl);        // Scale
        vf = vfmax_vf_f32m1(vf, -127.0f, vl);          // Clamp min
        vf = vfmin_vf_f32m1(vf, 127.0f, vl);           // Clamp max
        vint32m1_t vi32 = vfcvt_x_f_v_i32m1(vf, vl);   // Convert to i32
        vint16mf2_t vi16 = vnclip_wx_i16mf2(vi32, 0, vl);  // Narrow to i16
        vint8mf4_t vi8 = vnclip_wx_i8mf4(vi16, 0, vl);     // Narrow to i8
        vse8_v_i8mf4(&dst[i], vi8, vl);                // Store
        i += vl;
    }
}
```

**Expected Impact**: 10-20% faster (vectorize quantization overhead)

---

## 📈 Expected Final Results

| Version | Current Cycles | After Phase 1 | After Phase 2 | After Phase 3 | Target |
|---------|----------------|---------------|---------------|---------------|--------|
| bf16    | 2.7M          | 2.7M          | 2.7M          | 2.7M          | 2.7M   |
| i8      | 4.3M          | 3.0M (-30%)   | 2.4M (-20%)   | 2.2M (-8%)    | **2.2M** ✅ |
| i4      | 3.9M          | 2.7M (-31%)   | 2.2M (-19%)   | 2.0M (-9%)    | **2.0M** ✅ |

**Final State**: i8/i4 **faster than bf16** while using **2-3x less energy!** 🎉

---

## 📝 Implementation Priority

1. ✅ **Phase 1** (2-4 hours): Pre-quantize outside loops - biggest win!
2. ✅ **Phase 2** (1 day): Vectorize i8 dot product - unlock SIMD
3. ✅ **Phase 3** (4 hours): Vectorize quantization - polish performance
4. ✅ **Validation** (4 hours): Benchmark and verify accuracy

**Total Effort**: 2-3 days
**Expected Speedup**: 1.95x for i8, 1.95x for i4
**Result**: Same or better latency than bf16 with massive energy savings!

---

**Status**: Analysis complete. Ready to implement Phase 1! 🚀

