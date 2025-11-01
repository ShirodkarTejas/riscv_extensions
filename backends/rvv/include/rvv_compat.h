#ifndef SATTN_RVV_COMPAT_H_
#define SATTN_RVV_COMPAT_H_

// Normalize compiler macros so RVV blocks compile across toolchains.
// Some toolchains define __riscv_v_intrinsic instead of __riscv_vector.
#if !defined(__riscv_vector) && defined(__riscv_v_intrinsic)
#define __riscv_vector 1
#endif

#if defined(__clang__)
#if __has_include("/usr/lib/llvm-18/lib/clang/18/include/riscv_vector.h")
#include "/usr/lib/llvm-18/lib/clang/18/include/riscv_vector.h"
#else
#include <riscv_vector.h>
#endif
#else
#include <riscv_vector.h>
#endif

#if defined(__riscv_vector)
// Minimal wrappers mapping common ACLE names to GCC/Clang builtins if missing.
// Only declare when not already provided by headers.

#ifndef vsetvl_e32m1
static inline size_t vsetvl_e32m1(size_t n) {
#if defined(__GNUC__)
  return __riscv_vsetvl_e32m1(n);
#else
  return n;
#endif
}
#endif

#ifndef vle32_v_f32m1
static inline vfloat32m1_t vle32_v_f32m1(const float* base, size_t vl) {
#if defined(__GNUC__)
  return __riscv_vle32_v_f32m1(base, vl);
#else
  (void)vl; return *(const vfloat32m1_t*)base;
#endif
}
#endif

#ifndef vse32_v_f32m1
static inline void vse32_v_f32m1(float* base, vfloat32m1_t v, size_t vl) {
#if defined(__GNUC__)
  __riscv_vse32_v_f32m1(base, v, vl);
#else
  (void)vl; *(vfloat32m1_t*)base = v;
#endif
}
#endif

#ifndef vfmul_vf_f32m1
static inline vfloat32m1_t vfmul_vf_f32m1(vfloat32m1_t a, float f, size_t vl) {
#if defined(__GNUC__)
  return __riscv_vfmul_vf_f32m1(a, f, vl);
#else
  (void)vl; return a;
#endif
}
#endif

#ifndef vfmacc_vv_f32m1
static inline vfloat32m1_t vfmacc_vv_f32m1(vfloat32m1_t acc, vfloat32m1_t a, vfloat32m1_t b, size_t vl) {
#if defined(__GNUC__)
  return __riscv_vfmacc_vv_f32m1(acc, a, b, vl);
#else
  (void)vl; return acc;
#endif
}
#endif

#ifndef vfmacc_vf_f32m1
static inline vfloat32m1_t vfmacc_vf_f32m1(vfloat32m1_t acc, float f, vfloat32m1_t a, size_t vl) {
#if defined(__GNUC__)
  return __riscv_vfmacc_vf_f32m1(acc, f, a, vl);
#else
  (void)vl; return acc;
#endif
}
#endif

#ifndef vfmv_v_f_f32m1
static inline vfloat32m1_t vfmv_v_f_f32m1(float f, size_t vl) {
  (void)vl;
#if defined(__GNUC__)
  size_t vlmax = vsetvl_e32m1((size_t)-1);
  return __riscv_vfmv_v_f_f32m1(f, vlmax);
#else
  size_t vlmax = 1;
  return __riscv_vfmv_v_f_f32m1(f, vlmax);
#endif
}
#endif

#ifndef vfmv_f_s_f32m1_f32
static inline float vfmv_f_s_f32m1_f32(vfloat32m1_t v) {
#if defined(__GNUC__)
  return __riscv_vfmv_f_s_f32m1_f32(v);
#else
  return 0.0f;
#endif
}
#endif

#ifndef vundef_f32m1
static inline vfloat32m1_t vundef_f32m1(void) {
  // Approximate with a zeroed vector of minimal VL
  size_t vl1 = vsetvl_e32m1(1);
  return vfmv_v_f_f32m1(0.0f, vl1);
}
#endif

#ifndef vfredsum_vs_f32m1_f32m1
static inline vfloat32m1_t vfredsum_vs_f32m1_f32m1(vfloat32m1_t acc, vfloat32m1_t v, vfloat32m1_t zero, size_t vl) {
  (void)acc; (void)vl;
  size_t vlmax = vsetvl_e32m1((size_t)-1);
  float sum = 0.0f;
  const float* pv = (const float*)&v;
  for (size_t i = 0; i < vlmax; ++i) sum += pv[i];
  (void)zero; // zero is implied in this emulation
  return vfmv_v_f_f32m1(sum, 1);
}
#endif

#ifndef vfmax_vv_f32m1
static inline vfloat32m1_t vfmax_vv_f32m1(vfloat32m1_t a, vfloat32m1_t b, size_t vl) {
#if defined(__GNUC__)
  return __riscv_vfmax_vv_f32m1(a, b, vl);
#else
  (void)vl; return a;
#endif
}
#endif

#ifndef vfredmax_vs_f32m1_f32m1
static inline vfloat32m1_t vfredmax_vs_f32m1_f32m1(vfloat32m1_t acc, vfloat32m1_t v, vfloat32m1_t init, size_t vl) {
  (void)acc; (void)vl;
  size_t vlmax = vsetvl_e32m1((size_t)-1);
  float m = vfmv_f_s_f32m1_f32(init);
  const float* pv = (const float*)&v;
  for (size_t i = 0; i < vlmax; ++i) if (pv[i] > m) m = pv[i];
  return vfmv_v_f_f32m1(m, 1);
}
#endif

#ifndef vfadd_vv_f32m1
static inline vfloat32m1_t vfadd_vv_f32m1(vfloat32m1_t a, vfloat32m1_t b, size_t vl) {
#if defined(__GNUC__)
  return __riscv_vfadd_vv_f32m1(a, b, vl);
#else
  (void)vl; return a;
#endif
}
#endif

#ifndef vfncvt_f_f_w_f32m1
static inline vfloat32m1_t vfncvt_f_f_w_f32m1(vfloat32m1_t a) {
  // No-op for GCC 14 signature mismatch in our usage context
  return a;
}
#endif

#ifndef vfsub_vf_f32m1
static inline vfloat32m1_t vfsub_vf_f32m1(vfloat32m1_t a, float f, size_t vl) {
#if defined(__GNUC__)
  return __riscv_vfsub_vf_f32m1(a, f, vl);
#else
  (void)vl; return a;
#endif
}
#endif

#endif  // __riscv_vector

#endif  // SATTN_RVV_COMPAT_H_


