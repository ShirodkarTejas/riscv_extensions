#include "sparse_attention_rvv.h"

#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#ifdef __riscv_vector
#include <riscv_vector.h>
#endif

// Forward decls used by early helper implementations
static inline int64_t offset_bhld(int64_t b, int64_t h, int64_t l, int64_t d,
                                  int64_t B, int64_t H, int64_t L, int64_t D);

// ------------------------
// Quantization helpers
// ------------------------
static inline unsigned short float_to_bf16_u16(float x) {
  union { float f; unsigned int u; } v = { .f = x };
  // round to nearest even on lower 16 bits
  unsigned int lsb = (v.u >> 16) & 1u;
  unsigned int rounding_bias = 0x7FFFu + lsb;
  unsigned int rounded = v.u + rounding_bias;
  return (unsigned short)(rounded >> 16);
}
static inline float bf16_u16_to_float(unsigned short h) {
  union { unsigned int u; float f; } v = { .u = ((unsigned int)h) << 16 };
  return v.f;
}
static inline signed char f32_to_i8_symmetric(float x, float scale) {
  if (scale <= 0.f) scale = 1.f;
  float q = x / scale;
  if (q > 127.f) q = 127.f; else if (q < -127.f) q = -127.f;
  int iq = (int)lrintf(q);
  if (iq > 127) iq = 127; if (iq < -127) iq = -127;
  return (signed char)iq;
}
static inline float dequant_i8(int v, float scale) { return scale * (float)v; }
static inline signed char f32_to_i4_symmetric(float x, float scale) {
  if (scale <= 0.f) scale = 1.f;
  float q = x / scale;
  if (q > 7.f) q = 7.f; else if (q < -8.f) q = -8.f;
  int iq = (int)lrintf(q);
  if (iq > 7) iq = 7; if (iq < -8) iq = -8;
  return (signed char)iq; // we will use only low 4 bits signed range [-8,7]
}
static inline float dequant_i4(int v, float scale) { return scale * (float)v; }
#ifdef __riscv_vector
static inline float dot_f32_rvv(const float* a, const float* b, int64_t n);
static inline void axpy_f32_rvv(float alpha, const float* x, float* y, int64_t n);
#endif
#ifdef __riscv_vector
void sattn_rvv_lsh(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    sattn_shape_t shape,
    sattn_lsh_params_t params) {
  const int64_t B = shape.B, H = shape.H, L = shape.L, D = shape.D;
  const int buckets = params.buckets > 0 ? params.buckets : 1;
  const float scale = 1.0f / sqrtf((float)D);
  for (int64_t b = 0; b < B; ++b) for (int64_t h = 0; h < H; ++h) {
    for (int64_t i = 0; i < L; ++i) {
      for (int64_t d = 0; d < D; ++d) O[offset_bhld(b,h,i,d,B,H,L,D)] = 0.f;
      float denom = 0.f;
      int my_bucket = (int)(i % buckets);
      for (int64_t j = 0; j < L; ++j) {
        if ((int)(j % buckets) != my_bucket) continue;
        float dot = dot_f32_rvv(&Q[offset_bhld(b,h,i,0,B,H,L,D)], &K[offset_bhld(b,h,j,0,B,H,L,D)], D);
        float w = expf(dot * scale);
        denom += w;
        axpy_f32_rvv(w, &V[offset_bhld(b,h,j,0,B,H,L,D)], &O[offset_bhld(b,h,i,0,B,H,L,D)], D);
        _rvv_ctrs.br += (uint64_t)D * sizeof(float) * 2; _rvv_ctrs.bw += (uint64_t)D * sizeof(float); _rvv_ctrs.mac += (uint64_t)D;
      }
      float inv = 1.f / (denom + 1e-12f);
      size_t idx = 0; for (; idx < (size_t)D;) { size_t vl = vsetvl_e32m1((size_t)(D - idx)); vfloat32m1_t vy = vle32_v_f32m1(&O[offset_bhld(b,h,i,0,B,H,L,D)] + idx, vl); vy = vfmul_vf_f32m1(vy, inv, vl); vse32_v_f32m1(&O[offset_bhld(b,h,i,0,B,H,L,D)] + idx, vy, vl); idx += vl; }
    }
  }
}
#else
void sattn_rvv_lsh(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    sattn_shape_t shape,
    sattn_lsh_params_t params) {
  const int64_t B = shape.B, H = shape.H, L = shape.L, D = shape.D;
  const int buckets = params.buckets > 0 ? params.buckets : 1;
  const float scale = 1.0f / sqrtf((float)D);
  for (int64_t b = 0; b < B; ++b) for (int64_t h = 0; h < H; ++h) {
    for (int64_t i = 0; i < L; ++i) {
      for (int64_t d = 0; d < D; ++d) O[offset_bhld(b,h,i,d,B,H,L,D)] = 0.f;
      float denom = 0.f; int my_bucket = (int)(i % buckets);
      for (int64_t j = 0; j < L; ++j) {
        if ((int)(j % buckets) != my_bucket) continue;
        float dot = 0.f; for (int64_t d = 0; d < D; ++d) dot += Q[offset_bhld(b,h,i,d,B,H,L,D)] * K[offset_bhld(b,h,j,d,B,H,L,D)];
        float w = expf(dot * scale); denom += w;
        for (int64_t d = 0; d < D; ++d) O[offset_bhld(b,h,i,d,B,H,L,D)] += w * V[offset_bhld(b,h,j,d,B,H,L,D)];
      }
      float inv = 1.f / (denom + 1e-12f); for (int64_t d = 0; d < D; ++d) O[offset_bhld(b,h,i,d,B,H,L,D)] *= inv;
    }
  }
}
#endif

void sattn_rvv_lsh_tiled(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    sattn_shape_t shape,
    sattn_lsh_params_t params,
    int tile_rows) {
  if (tile_rows <= 0) tile_rows = 4;
  const int64_t B = shape.B, H = shape.H, L = shape.L, D = shape.D;
  const int buckets = params.buckets > 0 ? params.buckets : 1;
  const float scale = 1.0f / sqrtf((float)D);
  for (int64_t b = 0; b < B; ++b) for (int64_t h = 0; h < H; ++h) {
    for (int64_t ib = 0; ib < L; ib += tile_rows) {
      int64_t iend = ib + tile_rows; if (iend > L) iend = L;
      for (int64_t i = ib; i < iend; ++i) {
        for (int64_t d = 0; d < D; ++d) O[offset_bhld(b,h,i,d,B,H,L,D)] = 0.f;
        float denom = 0.f; int my_bucket = (int)(i % buckets);
        for (int64_t j = 0; j < L; ++j) {
          if ((int)(j % buckets) != my_bucket) continue;
#ifdef __riscv_vector
          float dot = dot_f32_rvv(&Q[offset_bhld(b,h,i,0,B,H,L,D)], &K[offset_bhld(b,h,j,0,B,H,L,D)], D);
          float w = expf(dot * scale); denom += w;
          axpy_f32_rvv(w, &V[offset_bhld(b,h,j,0,B,H,L,D)], &O[offset_bhld(b,h,i,0,B,H,L,D)], D);
          _rvv_ctrs.br += (uint64_t)D * sizeof(float) * 2; _rvv_ctrs.bw += (uint64_t)D * sizeof(float); _rvv_ctrs.mac += (uint64_t)D;
#else
          float dot = 0.f; for (int64_t d = 0; d < D; ++d) dot += Q[offset_bhld(b,h,i,d,B,H,L,D)] * K[offset_bhld(b,h,j,d,B,H,L,D)];
          float w = expf(dot * scale); denom += w;
          for (int64_t d = 0; d < D; ++d) O[offset_bhld(b,h,i,d,B,H,L,D)] += w * V[offset_bhld(b,h,j,d,B,H,L,D)];
#endif
        }
        float inv = 1.f / (denom + 1e-12f);
#ifdef __riscv_vector
        size_t idx = 0; for (; idx < (size_t)D;) { size_t vl = vsetvl_e32m1((size_t)(D - idx)); vfloat32m1_t vy = vle32_v_f32m1(&O[offset_bhld(b,h,i,0,B,H,L,D)] + idx, vl); vy = vfmul_vf_f32m1(vy, inv, vl); vse32_v_f32m1(&O[offset_bhld(b,h,i,0,B,H,L,D)] + idx, vy, vl); idx += vl; }
#else
        for (int64_t d = 0; d < D; ++d) O[offset_bhld(b,h,i,d,B,H,L,D)] *= inv;
#endif
      }
    }
  }
}
#ifdef __riscv
static inline uint64_t rdcycle() {
  uint64_t c = 0;
  asm volatile("rdcycle %0" : "=r"(c));
  return c;
}
#else
static inline uint64_t rdcycle() { return 0ull; }
#endif

uint64_t sattn_rdcycle() { return rdcycle(); }

// RVV proxy counters
static struct { uint64_t br, bw, mac; } _rvv_ctrs = {0,0,0};
void sattn_rvv_counters_reset() { _rvv_ctrs.br = _rvv_ctrs.bw = _rvv_ctrs.mac = 0; }
void sattn_rvv_counters_get(sattn_rvv_counters_t* out) {
  if (!out) return; out->bytes_read = _rvv_ctrs.br; out->bytes_written = _rvv_ctrs.bw; out->mac_flops = _rvv_ctrs.mac;
}

void sattn_rvv_landmark(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    sattn_shape_t shape,
    sattn_landmark_params_t params) {
  const int64_t B = shape.B, H = shape.H, L = shape.L, D = shape.D;
  int nl = params.num_landmarks > 0 ? params.num_landmarks : (int)(L > 0 ? (L < 32 ? L : 32) : 1);
  if (nl < 1) nl = 1;
  const float scale = 1.0f / sqrtf((float)D);
  int iters = params.iters > 0 ? params.iters : 0;
  // Landmarks per (b,h)
  for (int64_t b = 0; b < B; ++b) for (int64_t h = 0; h < H; ++h) {
    // Initialize centroids as evenly spaced K rows
    float* C = (float*)malloc((size_t)nl * (size_t)D * sizeof(float));
    int* lm = (int*)malloc(sizeof(int) * (size_t)nl);
    for (int i = 0; i < nl; ++i) { lm[i] = (int)((int64_t)i * L / nl); if (lm[i] >= (int)L) lm[i] = (int)L - 1; if (lm[i] < 0) lm[i] = 0; }
    for (int i = 0; i < nl; ++i) {
      for (int64_t d = 0; d < D; ++d) C[(size_t)i*D + d] = K[offset_bhld(b,h,lm[i],d,B,H,L,D)];
    }
    // Optional k-means-lite refinement: assign to nearest centroid (L2), update centroids by mean
    for (int it = 0; it < iters; ++it) {
      int* counts = (int*)calloc((size_t)nl, sizeof(int));
      float* newC = (float*)calloc((size_t)nl * (size_t)D, sizeof(float));
      if (!counts || !newC) { if (counts) free(counts); if (newC) free(newC); break; }
      for (int64_t i = 0; i < L; ++i) {
        // find closest centroid
        int best = 0; float bestd = 1e30f;
        for (int c = 0; c < nl; ++c) {
          float dist = 0.f; for (int64_t d = 0; d < D; ++d) {
            float kd = K[offset_bhld(b,h,i,d,B,H,L,D)] - C[(size_t)c*D + d]; dist += kd*kd; }
          if (dist < bestd) { bestd = dist; best = c; }
        }
        counts[best]++;
        for (int64_t d = 0; d < D; ++d) newC[(size_t)best*D + d] += K[offset_bhld(b,h,i,d,B,H,L,D)];
        _rvv_ctrs.br += (uint64_t)D * sizeof(float); // K read
      }
      for (int c = 0; c < nl; ++c) {
        int cnt = counts[c] > 0 ? counts[c] : 1;
        for (int64_t d = 0; d < D; ++d) C[(size_t)c*D + d] = newC[(size_t)c*D + d] / (float)cnt;
      }
      free(counts); free(newC);
    }
    // Attention over centroids; use V at assigned landmark indices for simplicity
    for (int64_t i = 0; i < L; ++i) {
      for (int64_t d = 0; d < D; ++d) O[offset_bhld(b,h,i,d,B,H,L,D)] = 0.f;
      float m = -INFINITY;
      for (int li = 0; li < nl; ++li) {
        float dot = 0.f; for (int64_t d = 0; d < D; ++d)
          dot += Q[offset_bhld(b,h,i,d,B,H,L,D)] * C[(size_t)li*D + d];
        dot *= scale; if (dot > m) m = dot; _rvv_ctrs.br += (uint64_t)D * sizeof(float) * 2; _rvv_ctrs.mac += (uint64_t)D;
      }
      float denom = 0.f;
      for (int li = 0; li < nl; ++li) {
        float dot = 0.f; for (int64_t d = 0; d < D; ++d)
          dot += Q[offset_bhld(b,h,i,d,B,H,L,D)] * C[(size_t)li*D + d];
        float w = expf(dot * scale - m); denom += w;
        // For V, use nearest original token to the centroid as representative (lm index)
        int j = lm[li];
        for (int64_t d = 0; d < D; ++d)
          O[offset_bhld(b,h,i,d,B,H,L,D)] += w * V[offset_bhld(b,h,j,d,B,H,L,D)];
        _rvv_ctrs.br += (uint64_t)D * sizeof(float) * 2; _rvv_ctrs.bw += (uint64_t)D * sizeof(float);
      }
      float inv = 1.f / (denom + 1e-12f);
      for (int64_t d = 0; d < D; ++d) O[offset_bhld(b,h,i,d,B,H,L,D)] *= inv;
    }
    free(lm); free(C);
  }
}

static inline int64_t offset_bhld(int64_t b, int64_t h, int64_t l, int64_t d,
                                  int64_t B, int64_t H, int64_t L, int64_t D) {
  (void)B;
  return (((b * H + h) * L + l) * D + d);
}

static inline float stable_softmax_den(const float* scores, int len) {
  float m = -INFINITY;
  for (int i = 0; i < len; ++i) {
    if (scores[i] > m) m = scores[i];
  }
  float s = 0.f;
  for (int i = 0; i < len; ++i) s += expf(scores[i] - m);
  return s > 0.f ? s : 1.f;
}

#ifdef __riscv_vector
static inline float dot_f32_rvv(const float* a, const float* b, int64_t n) {
  float acc = 0.f;
  size_t i = 0;
  vfloat32m1_t vacc = vfmv_v_f_f32m1(0.0f, 1);
  for (; i < (size_t)n;) {
    size_t vl = vsetvl_e32m1((size_t)(n - i));
    vfloat32m1_t va = vle32_v_f32m1(a + i, vl);
    vfloat32m1_t vb = vle32_v_f32m1(b + i, vl);
    vfloat32m1_t vmacc = vfmacc_vv_f32m1(vacc, va, vb, vl);
    vacc = vmacc;
    i += vl;
  }
  acc = vfmv_f_s_f32m1_f32(vfredsum_vs_f32m1_f32m1(vundef_f32m1(), vacc, vfmv_v_f_f32m1(0.0f, 1), 1));
  return acc;
}

static inline void axpy_f32_rvv(float alpha, const float* x, float* y, int64_t n) {
  size_t i = 0;
  for (; i < (size_t)n;) {
    size_t vl = vsetvl_e32m1((size_t)(n - i));
    vfloat32m1_t vx = vle32_v_f32m1(x + i, vl);
    vfloat32m1_t vy = vle32_v_f32m1(y + i, vl);
    vy = vfmacc_vf_f32m1(vy, alpha, vx, vl);
    vse32_v_f32m1(y + i, vy, vl);
    i += vl;
  }
}

// Gather/scatter helpers (row-major [L,D])
static inline void gather_rows_indexed_f32(const float* src, const int* idx,
                                           int64_t n_idx, int64_t D, float* dst) {
  for (int64_t t = 0; t < n_idx; ++t) {
    int row = idx[t];
    const float* s = src + ((int64_t)row * D);
    float* d = dst + (t * D);
    size_t i = 0;
    for (; i < (size_t)D;) {
      size_t vl = vsetvl_e32m1((size_t)(D - i));
      vfloat32m1_t v = vle32_v_f32m1(s + i, vl);
      vse32_v_f32m1(d + i, v, vl);
      i += vl;
    }
  }
}

static inline void gather_rows_contiguous_f32(const float* src, float* dst,
                                              int64_t start_row, int64_t end_row,
                                              int64_t D) {
  for (int64_t r = start_row; r < end_row; ++r) {
    const float* s = src + (r * D);
    float* d = dst + ((r - start_row) * D);
    size_t i = 0;
    for (; i < (size_t)D;) {
      size_t vl = vsetvl_e32m1((size_t)(D - i));
      vfloat32m1_t v = vle32_v_f32m1(s + i, vl);
      vse32_v_f32m1(d + i, v, vl);
      i += vl;
    }
  }
}

// Reduce sum of dot products of a fixed vector q against a block of rows
static inline float reduce_block_sumdot_f32_rvv(const float* q, const float* k_block,
                                                int64_t block_len, int64_t D) {
  float sum = 0.f;
  for (int64_t r = 0; r < block_len; ++r) {
    sum += dot_f32_rvv(q, k_block + r * D, D);
  }
  return sum;
}
#endif

void sattn_rvv_sliding_global(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    sattn_shape_t shape,
    sattn_params_t params) {
  const int64_t B = shape.B, H = shape.H, L = shape.L, D = shape.D;
  const int window = params.window_size;
  const int step = (params.dilation > 0) ? params.dilation : 1;
  const int wrap = (params.wrap != 0);
  const float scale = 1.0f / sqrtf((float)D);

  for (int64_t b = 0; b < B; ++b) {
    for (int64_t h = 0; h < H; ++h) {
        for (int64_t i = 0; i < L; ++i) {
        const int64_t j_left = i - (int64_t)window * step > 0 ? i - (int64_t)window * step : 0;
        const int64_t j_right = i + (int64_t)window * step + 1 < L ? i + (int64_t)window * step + 1 : L;
        const int span = wrap ? (2*window + 1) : (int)((j_right - j_left + (step - 1)) / step);
        // compute scores over [j_left, j_right)
        // store in stack buffer if small; fallback alloc if large
        // conservative simple path
        for (int64_t d = 0; d < D; ++d) {
          O[offset_bhld(b, h, i, d, B, H, L, D)] = 0.f;
        }
        if (span <= 0) continue;
        // For simplicity, compute softmax in two passes without extra alloc
        float m = -INFINITY;
        if (wrap) {
          for (int k = -window; k <= window; ++k) {
            int64_t j = (int64_t)i + (int64_t)k * step;
            j %= L; if (j < 0) j += L;
            float dot = 0.f;
#ifdef __riscv_vector
            dot = dot_f32_rvv(
                &Q[offset_bhld(b, h, i, 0, B, H, L, D)],
                &K[offset_bhld(b, h, j, 0, B, H, L, D)],
                D);
            _rvv_ctrs.br += (uint64_t)D * sizeof(float) * 2; // Q and K reads
            _rvv_ctrs.mac += (uint64_t)D;                    // D FMAs
#else
            for (int64_t d = 0; d < D; ++d) {
              float qd = Q[offset_bhld(b, h, i, d, B, H, L, D)];
              float kd = K[offset_bhld(b, h, j, d, B, H, L, D)];
              dot += qd * kd;
            }
            _rvv_ctrs.br += (uint64_t)D * sizeof(float) * 2; _rvv_ctrs.mac += (uint64_t)D;
#endif
            dot *= scale;
            if (dot > m) m = dot;
          }
        } else {
        for (int64_t j = j_left; j < j_right; j += step) {
          float dot = 0.f;
#ifdef __riscv_vector
          dot = dot_f32_rvv(
              &Q[offset_bhld(b, h, i, 0, B, H, L, D)],
              &K[offset_bhld(b, h, j, 0, B, H, L, D)],
              D);
          _rvv_ctrs.br += (uint64_t)D * sizeof(float) * 2; // Q and K reads
          _rvv_ctrs.mac += (uint64_t)D;                    // D FMAs
#else
          for (int64_t d = 0; d < D; ++d) {
            float qd = Q[offset_bhld(b, h, i, d, B, H, L, D)];
            float kd = K[offset_bhld(b, h, j, d, B, H, L, D)];
            dot += qd * kd;
          }
          _rvv_ctrs.br += (uint64_t)D * sizeof(float) * 2; _rvv_ctrs.mac += (uint64_t)D;
#endif
          dot *= scale;
          if (dot > m) m = dot;
        }
        }
        float denom = 0.f;
        if (wrap) {
          for (int k = -window; k <= window; ++k) {
            int64_t j = (int64_t)i + (int64_t)k * step;
            j %= L; if (j < 0) j += L;
            float dot = 0.f;
#ifdef __riscv_vector
            dot = dot_f32_rvv(
                &Q[offset_bhld(b, h, i, 0, B, H, L, D)],
                &K[offset_bhld(b, h, j, 0, B, H, L, D)],
                D);
#else
            for (int64_t d = 0; d < D; ++d) {
              float qd = Q[offset_bhld(b, h, i, d, B, H, L, D)];
              float kd = K[offset_bhld(b, h, j, d, B, H, L, D)];
              dot += qd * kd;
            }
#endif
            float w = expf(dot * scale - m);
            denom += w;
#ifdef __riscv_vector
            axpy_f32_rvv(w, &V[offset_bhld(b, h, j, 0, B, H, L, D)],
                         &O[offset_bhld(b, h, i, 0, B, H, L, D)], D);
            _rvv_ctrs.br += (uint64_t)D * sizeof(float); _rvv_ctrs.bw += (uint64_t)D * sizeof(float);
#else
            for (int64_t d = 0; d < D; ++d) {
              float vd = V[offset_bhld(b, h, j, d, B, H, L, D)];
              O[offset_bhld(b, h, i, d, B, H, L, D)] += w * vd;
            }
            _rvv_ctrs.br += (uint64_t)D * sizeof(float); _rvv_ctrs.bw += (uint64_t)D * sizeof(float);
#endif
          }
        } else {
        for (int64_t j = j_left; j < j_right; j += step) {
          float dot = 0.f;
#ifdef __riscv_vector
          dot = dot_f32_rvv(
              &Q[offset_bhld(b, h, i, 0, B, H, L, D)],
              &K[offset_bhld(b, h, j, 0, B, H, L, D)],
              D);
#else
          for (int64_t d = 0; d < D; ++d) {
            float qd = Q[offset_bhld(b, h, i, d, B, H, L, D)];
            float kd = K[offset_bhld(b, h, j, d, B, H, L, D)];
            dot += qd * kd;
          }
#endif
          float w = expf(dot * scale - m);
          denom += w;
#ifdef __riscv_vector
          axpy_f32_rvv(w, &V[offset_bhld(b, h, j, 0, B, H, L, D)],
                       &O[offset_bhld(b, h, i, 0, B, H, L, D)], D);
          _rvv_ctrs.br += (uint64_t)D * sizeof(float); _rvv_ctrs.bw += (uint64_t)D * sizeof(float);
#else
          for (int64_t d = 0; d < D; ++d) {
            float vd = V[offset_bhld(b, h, j, d, B, H, L, D)];
            O[offset_bhld(b, h, i, d, B, H, L, D)] += w * vd;
          }
          _rvv_ctrs.br += (uint64_t)D * sizeof(float); _rvv_ctrs.bw += (uint64_t)D * sizeof(float);
#endif
        }
        }
        float inv = 1.f / (denom + 1e-12f);
#ifdef __riscv_vector
        // scale O row by inv
        size_t idx = 0; for (; idx < (size_t)D;) { size_t vl = vsetvl_e32m1((size_t)(D - idx)); vfloat32m1_t vy = vle32_v_f32m1(&O[offset_bhld(b, h, i, 0, B, H, L, D)] + idx, vl); vy = vfmul_vf_f32m1(vy, inv, vl); vse32_v_f32m1(&O[offset_bhld(b, h, i, 0, B, H, L, D)] + idx, vy, vl); idx += vl; }
        _rvv_ctrs.br += (uint64_t)D * sizeof(float); _rvv_ctrs.bw += (uint64_t)D * sizeof(float);
#else
        for (int64_t d = 0; d < D; ++d) {
          O[offset_bhld(b, h, i, d, B, H, L, D)] *= inv;
        }
        _rvv_ctrs.br += (uint64_t)D * sizeof(float); _rvv_ctrs.bw += (uint64_t)D * sizeof(float);
#endif
      }
    }
  }
}

void sattn_rvv_sliding_global_bf16(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    sattn_shape_t shape,
    sattn_params_t params) {
  const int64_t B = shape.B, H = shape.H, L = shape.L, D = shape.D;
  const int window = params.window_size;
  const float scale = 1.0f / sqrtf((float)D);
  for (int64_t b = 0; b < B; ++b) for (int64_t h = 0; h < H; ++h) {
    for (int64_t i = 0; i < L; ++i) {
      for (int64_t d = 0; d < D; ++d) O[offset_bhld(b,h,i,d,B,H,L,D)] = 0.f;
      int64_t jl = i - window > 0 ? i - window : 0;
      int64_t jr = i + window + 1 < L ? i + window + 1 : L;
      if (jr <= jl) continue;
      float m = -INFINITY;
      for (int64_t j = jl; j < jr; ++j) {
        float dot = 0.f;
        for (int64_t d = 0; d < D; ++d) {
          unsigned short qh = float_to_bf16_u16(Q[offset_bhld(b,h,i,d,B,H,L,D)]);
          unsigned short kh = float_to_bf16_u16(K[offset_bhld(b,h,j,d,B,H,L,D)]);
          dot += bf16_u16_to_float(qh) * bf16_u16_to_float(kh);
        }
        dot *= scale; if (dot > m) m = dot;
      }
      float denom = 0.f;
      for (int64_t j = jl; j < jr; ++j) {
        float dot = 0.f;
        for (int64_t d = 0; d < D; ++d) {
          unsigned short qh = float_to_bf16_u16(Q[offset_bhld(b,h,i,d,B,H,L,D)]);
          unsigned short kh = float_to_bf16_u16(K[offset_bhld(b,h,j,d,B,H,L,D)]);
          dot += bf16_u16_to_float(qh) * bf16_u16_to_float(kh);
        }
        float w = expf(dot * scale - m); denom += w;
        for (int64_t d = 0; d < D; ++d) {
          unsigned short vh = float_to_bf16_u16(V[offset_bhld(b,h,j,d,B,H,L,D)]);
          O[offset_bhld(b,h,i,d,B,H,L,D)] += w * bf16_u16_to_float(vh);
        }
      }
      float inv = 1.f / (denom + 1e-12f);
      for (int64_t d = 0; d < D; ++d) O[offset_bhld(b,h,i,d,B,H,L,D)] *= inv;
    }
  }
}

void sattn_rvv_sliding_global_i8(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    sattn_shape_t shape,
    sattn_params_t params,
    float scale_q,
    float scale_k,
    float scale_v) {
  const int64_t B = shape.B, H = shape.H, L = shape.L, D = shape.D;
  const int window = params.window_size;
  const float attn_scale = 1.0f / sqrtf((float)D);
  const float s_q = scale_q > 0.f ? scale_q : 0.05f;
  const float s_k = scale_k > 0.f ? scale_k : 0.05f;
  const float s_v = scale_v > 0.f ? scale_v : 0.05f;
  for (int64_t b = 0; b < B; ++b) for (int64_t h = 0; h < H; ++h) {
    for (int64_t i = 0; i < L; ++i) {
      for (int64_t d = 0; d < D; ++d) O[offset_bhld(b,h,i,d,B,H,L,D)] = 0.f;
      int64_t jl = i - window > 0 ? i - window : 0;
      int64_t jr = i + window + 1 < L ? i + window + 1 : L;
      if (jr <= jl) continue;
      float m = -INFINITY;
      for (int64_t j = jl; j < jr; ++j) {
        int dot_i32 = 0;
        for (int64_t d = 0; d < D; ++d) {
          signed char qi = f32_to_i8_symmetric(Q[offset_bhld(b,h,i,d,B,H,L,D)], s_q);
          signed char ki = f32_to_i8_symmetric(K[offset_bhld(b,h,j,d,B,H,L,D)], s_k);
          dot_i32 += (int)qi * (int)ki;
        }
        float dot = (s_q * s_k) * (float)dot_i32;
        dot *= attn_scale; if (dot > m) m = dot;
      }
      float denom = 0.f;
      for (int64_t j = jl; j < jr; ++j) {
        int dot_i32 = 0;
        for (int64_t d = 0; d < D; ++d) {
          signed char qi = f32_to_i8_symmetric(Q[offset_bhld(b,h,i,d,B,H,L,D)], s_q);
          signed char ki = f32_to_i8_symmetric(K[offset_bhld(b,h,j,d,B,H,L,D)], s_k);
          dot_i32 += (int)qi * (int)ki;
        }
        float dot = (s_q * s_k) * (float)dot_i32;
        float w = expf(dot * attn_scale - m); denom += w;
        for (int64_t d = 0; d < D; ++d) {
          signed char vi = f32_to_i8_symmetric(V[offset_bhld(b,h,j,d,B,H,L,D)], s_v);
          float v = dequant_i8((int)vi, s_v);
          O[offset_bhld(b,h,i,d,B,H,L,D)] += w * v;
        }
      }
      float inv = 1.f / (denom + 1e-12f);
      for (int64_t d = 0; d < D; ++d) O[offset_bhld(b,h,i,d,B,H,L,D)] *= inv;
    }
  }
}

void sattn_rvv_sliding_global_i4(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    sattn_shape_t shape,
    sattn_params_t params,
    float scale_q,
    float scale_k,
    float scale_v) {
  const int64_t B = shape.B, H = shape.H, L = shape.L, D = shape.D;
  const int window = params.window_size;
  const float attn_scale = 1.0f / sqrtf((float)D);
  const float s_q = scale_q > 0.f ? scale_q : 0.1f;
  const float s_k = scale_k > 0.f ? scale_k : 0.1f;
  const float s_v = scale_v > 0.f ? scale_v : 0.1f;
  for (int64_t b = 0; b < B; ++b) for (int64_t h = 0; h < H; ++h) {
    for (int64_t i = 0; i < L; ++i) {
      for (int64_t d = 0; d < D; ++d) O[offset_bhld(b,h,i,d,B,H,L,D)] = 0.f;
      int64_t jl = i - window > 0 ? i - window : 0;
      int64_t jr = i + window + 1 < L ? i + window + 1 : L;
      if (jr <= jl) continue;
      float m = -INFINITY;
      for (int64_t j = jl; j < jr; ++j) {
        int dot_i32 = 0;
        for (int64_t d = 0; d < D; ++d) {
          int qi = (int)f32_to_i4_symmetric(Q[offset_bhld(b,h,i,d,B,H,L,D)], s_q);
          int ki = (int)f32_to_i4_symmetric(K[offset_bhld(b,h,j,d,B,H,L,D)], s_k);
          dot_i32 += qi * ki;
        }
        float dot = (s_q * s_k) * (float)dot_i32;
        dot *= attn_scale; if (dot > m) m = dot;
      }
      float denom = 0.f;
      for (int64_t j = jl; j < jr; ++j) {
        int dot_i32 = 0;
        for (int64_t d = 0; d < D; ++d) {
          int qi = (int)f32_to_i4_symmetric(Q[offset_bhld(b,h,i,d,B,H,L,D)], s_q);
          int ki = (int)f32_to_i4_symmetric(K[offset_bhld(b,h,j,d,B,H,L,D)], s_k);
          dot_i32 += qi * ki;
        }
        float dot = (s_q * s_k) * (float)dot_i32;
        float w = expf(dot * attn_scale - m); denom += w;
        for (int64_t d = 0; d < D; ++d) {
          int vi = (int)f32_to_i4_symmetric(V[offset_bhld(b,h,j,d,B,H,L,D)], s_v);
          float v = dequant_i4(vi, s_v);
          O[offset_bhld(b,h,i,d,B,H,L,D)] += w * v;
        }
      }
      float inv = 1.f / (denom + 1e-12f);
      for (int64_t d = 0; d < D; ++d) O[offset_bhld(b,h,i,d,B,H,L,D)] *= inv;
    }
  }
}

void sattn_rvv_sliding_global_tiled(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    sattn_shape_t shape,
    sattn_params_t params,
    int tile_rows) {
  const int64_t B = shape.B, H = shape.H, L = shape.L, D = shape.D;
  const int window = params.window_size;
  const float scale = 1.0f / sqrtf((float)D);
  if (tile_rows <= 0) tile_rows = 4;
  for (int64_t b = 0; b < B; ++b) {
    for (int64_t h = 0; h < H; ++h) {
      for (int64_t ib = 0; ib < L; ib += tile_rows) {
        int64_t iend = ib + tile_rows; if (iend > L) iend = L;
        // zero O tile
        for (int64_t i = ib; i < iend; ++i) for (int64_t d = 0; d < D; ++d) O[offset_bhld(b,h,i,d,B,H,L,D)] = 0.f;
        // For each row in tile, compute local window
        for (int64_t i = ib; i < iend; ++i) {
          int64_t jl = i - window > 0 ? i - window : 0;
          int64_t jr = i + window + 1 < L ? i + window + 1 : L;
          if (jr <= jl) continue;
          float m = -INFINITY;
          for (int64_t j = jl; j < jr; ++j) {
            float dot = 0.f;
#ifdef __riscv_vector
            dot = dot_f32_rvv(&Q[offset_bhld(b,h,i,0,B,H,L,D)], &K[offset_bhld(b,h,j,0,B,H,L,D)], D);
            _rvv_ctrs.br += (uint64_t)D * sizeof(float) * 2; _rvv_ctrs.mac += (uint64_t)D;
#else
            for (int64_t d = 0; d < D; ++d) dot += Q[offset_bhld(b,h,i,d,B,H,L,D)] * K[offset_bhld(b,h,j,d,B,H,L,D)];
            _rvv_ctrs.br += (uint64_t)D * sizeof(float) * 2; _rvv_ctrs.mac += (uint64_t)D;
#endif
            dot *= scale; if (dot > m) m = dot;
          }
          float denom = 0.f;
          for (int64_t j = jl; j < jr; ++j) {
            float dot = 0.f;
#ifdef __riscv_vector
            dot = dot_f32_rvv(&Q[offset_bhld(b,h,i,0,B,H,L,D)], &K[offset_bhld(b,h,j,0,B,H,L,D)], D);
#else
            for (int64_t d = 0; d < D; ++d) dot += Q[offset_bhld(b,h,i,d,B,H,L,D)] * K[offset_bhld(b,h,j,d,B,H,L,D)];
#endif
            float w = expf(dot * scale - m); denom += w;
#ifdef __riscv_vector
            axpy_f32_rvv(w, &V[offset_bhld(b,h,j,0,B,H,L,D)], &O[offset_bhld(b,h,i,0,B,H,L,D)], D);
            _rvv_ctrs.br += (uint64_t)D * sizeof(float); _rvv_ctrs.bw += (uint64_t)D * sizeof(float);
#else
            for (int64_t d = 0; d < D; ++d) O[offset_bhld(b,h,i,d,B,H,L,D)] += w * V[offset_bhld(b,h,j,d,B,H,L,D)];
            _rvv_ctrs.br += (uint64_t)D * sizeof(float); _rvv_ctrs.bw += (uint64_t)D * sizeof(float);
#endif
          }
          float inv = 1.f / (denom + 1e-12f);
#ifdef __riscv_vector
          size_t idx = 0; for (; idx < (size_t)D;) { size_t vl = vsetvl_e32m1((size_t)(D - idx)); vfloat32m1_t vy = vle32_v_f32m1(&O[offset_bhld(b,h,i,0,B,H,L,D)] + idx, vl); vy = vfmul_vf_f32m1(vy, inv, vl); vse32_v_f32m1(&O[offset_bhld(b,h,i,0,B,H,L,D)] + idx, vy, vl); idx += vl; }
          _rvv_ctrs.br += (uint64_t)D * sizeof(float); _rvv_ctrs.bw += (uint64_t)D * sizeof(float);
#else
          for (int64_t d = 0; d < D; ++d) O[offset_bhld(b,h,i,d,B,H,L,D)] *= inv;
          _rvv_ctrs.br += (uint64_t)D * sizeof(float); _rvv_ctrs.bw += (uint64_t)D * sizeof(float);
#endif
        }
      }
    }
  }
}

void sattn_rvv_block_topk(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    sattn_shape_t shape,
    sattn_blocktopk_params_t params) {
  const int64_t B = shape.B, H = shape.H, L = shape.L, D = shape.D;
  const int block = params.block_size > 0 ? params.block_size : 64;
  const int64_t num_blocks = (L + block - 1) / block;
  int k_blocks = (int)((float)num_blocks * (params.keep_ratio > 0.f ? params.keep_ratio : 0.12f) + 0.999f);
  if (k_blocks < 1) k_blocks = 1;
  const float scale = 1.0f / sqrtf((float)D);

  // workspace
  int* block_idx = (int*)malloc((size_t)num_blocks * sizeof(int));
  float* block_scores = (float*)malloc((size_t)num_blocks * sizeof(float));
  if (!block_idx || !block_scores) { if (block_idx) free(block_idx); if (block_scores) free(block_scores); return; }

  int gsz = params.gqa_group_size > 0 ? params.gqa_group_size : 1;
  int cbs = params.comp_block_size > 0 ? params.comp_block_size : 0;
  if (gsz == 1 && cbs == 0) {
    for (int64_t b = 0; b < B; ++b) {
      for (int64_t h = 0; h < H; ++h) {
        for (int64_t i = 0; i < L; ++i) {
          for (int64_t d = 0; d < D; ++d) O[offset_bhld(b, h, i, d, B, H, L, D)] = 0.f;
          for (int64_t nb = 0; nb < num_blocks; ++nb) {
            int64_t s = nb * block; int64_t e = s + block; if (e > L) e = L;
            float dot = 0.f; int64_t cnt = (int64_t)(e - s);
            for (int64_t j = s; j < e; ++j) for (int64_t d = 0; d < D; ++d) dot += Q[offset_bhld(b, h, i, d, B, H, L, D)] * K[offset_bhld(b, h, j, d, B, H, L, D)];
            _rvv_ctrs.br += (uint64_t)cnt * (uint64_t)D * sizeof(float) + (uint64_t)D * sizeof(float);
            _rvv_ctrs.mac += (uint64_t)cnt * (uint64_t)D;
            block_scores[nb] = cnt > 0 ? (dot / (float)cnt) : -1e30f;
            block_idx[nb] = (int)nb;
          }
          for (int64_t x = 0; x < num_blocks - 1; ++x) for (int64_t y = x + 1; y < num_blocks; ++y)
            if (block_scores[y] > block_scores[x]) { float ts = block_scores[x]; block_scores[x] = block_scores[y]; block_scores[y] = ts; int ti = block_idx[x]; block_idx[x] = block_idx[y]; block_idx[y] = ti; }
          float denom = 0.f; const int gtok = params.global_tokens > 0 ? (params.global_tokens < (int)L ? params.global_tokens : (int)L) : 0;
          int sel_cap = (int)(k_blocks * block + gtok); int sel_cnt = 0; int* sel_idx = (int*)malloc((size_t)sel_cap * sizeof(int));
          for (int j = 0; j < gtok && sel_cnt < sel_cap; ++j) sel_idx[sel_cnt++] = j;
          for (int kb = 0; kb < k_blocks && kb < num_blocks; ++kb) { int nb = block_idx[kb]; int64_t s = (int64_t)nb * block; int64_t e = s + block; if (e > L) e = L; for (int64_t j = s; j < e && sel_cnt < sel_cap; ++j) sel_idx[sel_cnt++] = (int)j; }
          for (int t = 0; t < sel_cnt; ++t) { int j = sel_idx[t]; float dot = 0.f; for (int64_t d = 0; d < D; ++d) dot += Q[offset_bhld(b,h,i,d,B,H,L,D)] * K[offset_bhld(b,h,j,d,B,H,L,D)]; float w = expf(dot * scale); denom += w; for (int64_t d = 0; d < D; ++d) O[offset_bhld(b,h,i,d,B,H,L,D)] += w * V[offset_bhld(b,h,j,d,B,H,L,D)]; _rvv_ctrs.br += (uint64_t)D * sizeof(float) * 3; _rvv_ctrs.bw += (uint64_t)D * sizeof(float); _rvv_ctrs.mac += (uint64_t)D; }
          free(sel_idx);
          float inv = 1.f / (denom + 1e-12f); for (int64_t d = 0; d < D; ++d) O[offset_bhld(b,h,i,d,B,H,L,D)] *= inv;
        }
      }
    }
  } else {
    for (int64_t b = 0; b < B; ++b) {
      for (int64_t hg = 0; hg < H; hg += gsz) {
        for (int64_t i = 0; i < L; ++i) {
          for (int64_t d = 0; d < D; ++d) for (int64_t hh = 0; hh < gsz && (hg+hh) < H; ++hh) O[offset_bhld(b,hg+hh,i,d,B,H,L,D)] = 0.f;
          for (int64_t nb = 0; nb < num_blocks; ++nb) { block_scores[nb] = 0.f; block_idx[nb] = (int)nb; }
          for (int64_t hh = 0; hh < gsz && (hg+hh) < H; ++hh) {
            int64_t h = hg + hh;
            if (cbs > 0) {
              int64_t ncomp = (L + cbs - 1) / cbs; const float sc = 1.0f / sqrtf((float)D);
              float* pc = (float*)malloc((size_t)ncomp * sizeof(float)); if (!pc) continue;
              for (int64_t cb = 0; cb < ncomp; ++cb) { int64_t s = cb * cbs, e = s + cbs; if (e > L) e = L; int64_t cnt = e - s; float dot = 0.f; for (int64_t d = 0; d < D; ++d) {
                  float acc = 0.f; for (int64_t j = s; j < e; ++j) acc += K[offset_bhld(b,h,j,d,B,H,L,D)]; dot += Q[offset_bhld(b,h,i,d,B,H,L,D)] * (acc / (float)cnt);
                } pc[cb] = dot * sc; }
              for (int64_t nb = 0; nb < num_blocks; ++nb) { int64_t s = nb * block; int64_t e = s + block; if (e > L) e = L; int64_t c0 = s / cbs; int64_t c1 = (e-1) / cbs; float acc = 0.f; for (int64_t cb = c0; cb <= c1 && cb < ncomp; ++cb) acc += pc[cb]; block_scores[nb] += acc; }
              free(pc);
            } else {
              for (int64_t nb = 0; nb < num_blocks; ++nb) { int64_t s = nb * block; int64_t e = s + block; if (e > L) e = L; int64_t cnt = (int64_t)(e - s); float dot = 0.f; for (int64_t j = s; j < e; ++j) for (int64_t d = 0; d < D; ++d) dot += Q[offset_bhld(b,h,i,d,B,H,L,D)] * K[offset_bhld(b,h,j,d,B,H,L,D)]; block_scores[nb] += (cnt>0 ? (dot/(float)cnt) : -1e30f); _rvv_ctrs.br += (uint64_t)cnt * (uint64_t)D * sizeof(float) + (uint64_t)D * sizeof(float); _rvv_ctrs.mac += (uint64_t)cnt * (uint64_t)D; }
            }
          }
          for (int64_t x = 0; x < num_blocks - 1; ++x) for (int64_t y = x + 1; y < num_blocks; ++y)
            if (block_scores[y] > block_scores[x]) { float ts = block_scores[x]; block_scores[x] = block_scores[y]; block_scores[y] = ts; int ti = block_idx[x]; block_idx[x] = block_idx[y]; block_idx[y] = ti; }
          float denom = 0.f; const int gtok = params.global_tokens > 0 ? (params.global_tokens < (int)L ? params.global_tokens : (int)L) : 0;
          int sel_cap = (int)(k_blocks * block + gtok); int sel_cnt = 0; int* sel_idx = (int*)malloc((size_t)sel_cap * sizeof(int));
          for (int j = 0; j < gtok && sel_cnt < sel_cap; ++j) sel_idx[sel_cnt++] = j;
          for (int kb = 0; kb < k_blocks && kb < num_blocks; ++kb) { int nb = block_idx[kb]; int64_t s = (int64_t)nb * block; int64_t e = s + block; if (e > L) e = L; for (int64_t j = s; j < e && sel_cnt < sel_cap; ++j) sel_idx[sel_cnt++] = (int)j; }
          for (int64_t hh = 0; hh < gsz && (hg+hh) < H; ++hh) { int64_t h = hg + hh; for (int t = 0; t < sel_cnt; ++t) { int j = sel_idx[t]; float dot = 0.f; for (int64_t d = 0; d < D; ++d) dot += Q[offset_bhld(b,h,i,d,B,H,L,D)] * K[offset_bhld(b,h,j,d,B,H,L,D)]; float w = expf(dot * scale); denom += w; for (int64_t d = 0; d < D; ++d) O[offset_bhld(b,h,i,d,B,H,L,D)] += w * V[offset_bhld(b,h,j,d,B,H,L,D)]; }}
          free(sel_idx);
          float inv = 1.f / (denom + 1e-12f); for (int64_t hh = 0; hh < gsz && (hg+hh) < H; ++hh) { int64_t h = hg + hh; for (int64_t d = 0; d < D; ++d) O[offset_bhld(b,h,i,d,B,H,L,D)] *= inv; }
        }
      }
    }
  }

  free(block_idx);
  free(block_scores);
}

void sattn_rvv_block_topk_apply_selection(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    sattn_shape_t shape,
    int block_size,
    const int* sel_idx,
    int sel_cnt) {
  const int64_t B = shape.B, H = shape.H, L = shape.L, D = shape.D;
  (void)block_size; // selection already expanded to token indices
  const float scale = 1.0f / sqrtf((float)D);
  for (int64_t b = 0; b < B; ++b) for (int64_t h = 0; h < H; ++h) {
    for (int64_t i = 0; i < L; ++i) {
      for (int64_t d = 0; d < D; ++d) O[offset_bhld(b,h,i,d,B,H,L,D)] = 0.f;
      float denom = 0.f;
      for (int t = 0; t < sel_cnt; ++t) {
        int j = sel_idx[t]; if (j < 0 || j >= (int)L) continue;
        float dot = 0.f; for (int64_t d = 0; d < D; ++d)
          dot += Q[offset_bhld(b,h,i,d,B,H,L,D)] * K[offset_bhld(b,h,j,d,B,H,L,D)];
        float w = expf(dot * scale); denom += w;
        for (int64_t d = 0; d < D; ++d)
          O[offset_bhld(b,h,i,d,B,H,L,D)] += w * V[offset_bhld(b,h,j,d,B,H,L,D)];
        _rvv_ctrs.br += (uint64_t)D * sizeof(float) * 3; _rvv_ctrs.bw += (uint64_t)D * sizeof(float); _rvv_ctrs.mac += (uint64_t)D;
      }
      float inv = 1.f / (denom + 1e-12f);
      for (int64_t d = 0; d < D; ++d) O[offset_bhld(b,h,i,d,B,H,L,D)] *= inv;
    }
  }
}

void sattn_rvv_block_topk_bf16(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    sattn_shape_t shape,
    sattn_blocktopk_params_t params) {
  const int64_t B = shape.B, H = shape.H, L = shape.L, D = shape.D;
  const int block = params.block_size > 0 ? params.block_size : 64;
  const int64_t num_blocks = (L + block - 1) / block;
  int k_blocks = (int)((float)num_blocks * (params.keep_ratio > 0.f ? params.keep_ratio : 0.12f) + 0.999f);
  if (k_blocks < 1) k_blocks = 1;
  const float scale = 1.0f / sqrtf((float)D);
  int* block_idx = (int*)malloc((size_t)num_blocks * sizeof(int));
  float* block_scores = (float*)malloc((size_t)num_blocks * sizeof(float));
  if (!block_idx || !block_scores) { if (block_idx) free(block_idx); if (block_scores) free(block_scores); return; }
  for (int64_t b = 0; b < B; ++b) for (int64_t h = 0; h < H; ++h) {
    for (int64_t i = 0; i < L; ++i) {
      for (int64_t d = 0; d < D; ++d) O[offset_bhld(b,h,i,d,B,H,L,D)] = 0.f;
      for (int64_t nb = 0; nb < num_blocks; ++nb) {
        int64_t s = nb * block; int64_t e = s + block; if (e > L) e = L;
        float dot = 0.f; int64_t cnt = (int64_t)(e - s);
        for (int64_t j = s; j < e; ++j) for (int64_t d = 0; d < D; ++d) {
          float qf = bf16_u16_to_float(float_to_bf16_u16(Q[offset_bhld(b,h,i,d,B,H,L,D)]));
          float kf = bf16_u16_to_float(float_to_bf16_u16(K[offset_bhld(b,h,j,d,B,H,L,D)]));
          dot += qf * kf;
        }
        block_scores[nb] = cnt > 0 ? (dot / (float)cnt) : -1e30f; block_idx[nb] = (int)nb;
      }
      for (int64_t x = 0; x < num_blocks - 1; ++x) for (int64_t y = x + 1; y < num_blocks; ++y)
        if (block_scores[y] > block_scores[x]) { float ts = block_scores[x]; block_scores[x] = block_scores[y]; block_scores[y] = ts; int ti = block_idx[x]; block_idx[x] = block_idx[y]; block_idx[y] = ti; }
      float denom = 0.f;
      const int gtok = params.global_tokens > 0 ? (params.global_tokens < (int)L ? params.global_tokens : (int)L) : 0;
      int sel_cap = (int)(k_blocks * block + gtok); int sel_cnt = 0; int* sel_idx = (int*)malloc((size_t)sel_cap * sizeof(int));
      for (int j = 0; j < gtok && sel_cnt < sel_cap; ++j) sel_idx[sel_cnt++] = j;
      for (int kb = 0; kb < k_blocks && kb < num_blocks; ++kb) { int nb = block_idx[kb]; int64_t s = (int64_t)nb * block; int64_t e = s + block; if (e > L) e = L; for (int64_t j = s; j < e && sel_cnt < sel_cap; ++j) sel_idx[sel_cnt++] = (int)j; }
      for (int t = 0; t < sel_cnt; ++t) {
        int j = sel_idx[t]; float dot = 0.f;
        for (int64_t d = 0; d < D; ++d) {
          float qf = bf16_u16_to_float(float_to_bf16_u16(Q[offset_bhld(b,h,i,d,B,H,L,D)]));
          float kf = bf16_u16_to_float(float_to_bf16_u16(K[offset_bhld(b,h,j,d,B,H,L,D)]));
          dot += qf * kf;
        }
        float w = expf(dot * scale); denom += w;
        for (int64_t d = 0; d < D; ++d) {
          float vf = bf16_u16_to_float(float_to_bf16_u16(V[offset_bhld(b,h,j,d,B,H,L,D)]));
          O[offset_bhld(b,h,i,d,B,H,L,D)] += w * vf;
        }
      }
      free(sel_idx);
      float inv = 1.f / (denom + 1e-12f); for (int64_t d = 0; d < D; ++d) O[offset_bhld(b,h,i,d,B,H,L,D)] *= inv;
    }
  }
  free(block_idx); free(block_scores);
}

void sattn_rvv_block_topk_i8(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    sattn_shape_t shape,
    sattn_blocktopk_params_t params,
    float scale_q,
    float scale_k,
    float scale_v) {
  const int64_t B = shape.B, H = shape.H, L = shape.L, D = shape.D;
  const int block = params.block_size > 0 ? params.block_size : 64;
  const int64_t num_blocks = (L + block - 1) / block;
  int k_blocks = (int)((float)num_blocks * (params.keep_ratio > 0.f ? params.keep_ratio : 0.12f) + 0.999f);
  if (k_blocks < 1) k_blocks = 1;
  const float sc = 1.0f / sqrtf((float)D);
  const float s_q = scale_q > 0.f ? scale_q : 0.05f;
  const float s_k = scale_k > 0.f ? scale_k : 0.05f;
  const float s_v = scale_v > 0.f ? scale_v : 0.05f;
  int* block_idx = (int*)malloc((size_t)num_blocks * sizeof(int)); float* block_scores = (float*)malloc((size_t)num_blocks * sizeof(float));
  if (!block_idx || !block_scores) { if (block_idx) free(block_idx); if (block_scores) free(block_scores); return; }
  for (int64_t b = 0; b < B; ++b) for (int64_t h = 0; h < H; ++h) {
    for (int64_t i = 0; i < L; ++i) {
      for (int64_t d = 0; d < D; ++d) O[offset_bhld(b,h,i,d,B,H,L,D)] = 0.f;
      for (int64_t nb = 0; nb < num_blocks; ++nb) {
        int64_t s = nb * block; int64_t e = s + block; if (e > L) e = L; int64_t cnt = (int64_t)(e - s); int dot_i32 = 0;
        for (int64_t j = s; j < e; ++j) for (int64_t d = 0; d < D; ++d) {
          int qi = (int)f32_to_i8_symmetric(Q[offset_bhld(b,h,i,d,B,H,L,D)], s_q);
          int ki = (int)f32_to_i8_symmetric(K[offset_bhld(b,h,j,d,B,H,L,D)], s_k);
          dot_i32 += qi * ki;
        }
        float dot = (s_q * s_k) * (float)dot_i32; block_scores[nb] = cnt > 0 ? (dot / (float)cnt) : -1e30f; block_idx[nb] = (int)nb;
      }
      for (int64_t x = 0; x < num_blocks - 1; ++x) for (int64_t y = x + 1; y < num_blocks; ++y)
        if (block_scores[y] > block_scores[x]) { float ts = block_scores[x]; block_scores[x] = block_scores[y]; block_scores[y] = ts; int ti = block_idx[x]; block_idx[x] = block_idx[y]; block_idx[y] = ti; }
      float denom = 0.f; const int gtok = params.global_tokens > 0 ? (params.global_tokens < (int)L ? params.global_tokens : (int)L) : 0;
      int sel_cap = (int)(k_blocks * block + gtok); int sel_cnt = 0; int* sel_idx = (int*)malloc((size_t)sel_cap * sizeof(int));
      for (int j = 0; j < gtok && sel_cnt < sel_cap; ++j) sel_idx[sel_cnt++] = j;
      for (int kb = 0; kb < k_blocks && kb < num_blocks; ++kb) { int nb = block_idx[kb]; int64_t s = (int64_t)nb * block; int64_t e = s + block; if (e > L) e = L; for (int64_t j = s; j < e && sel_cnt < sel_cap; ++j) sel_idx[sel_cnt++] = (int)j; }
      for (int t = 0; t < sel_cnt; ++t) {
        int j = sel_idx[t]; int dot_i32 = 0;
        for (int64_t d = 0; d < D; ++d) {
          int qi = (int)f32_to_i8_symmetric(Q[offset_bhld(b,h,i,d,B,H,L,D)], s_q);
          int ki = (int)f32_to_i8_symmetric(K[offset_bhld(b,h,j,d,B,H,L,D)], s_k);
          dot_i32 += qi * ki;
        }
        float dot = (s_q * s_k) * (float)dot_i32; float w = expf(dot * sc); denom += w;
        for (int64_t d = 0; d < D; ++d) { int vi = (int)f32_to_i8_symmetric(V[offset_bhld(b,h,j,d,B,H,L,D)], s_v); float v = dequant_i8(vi, s_v); O[offset_bhld(b,h,i,d,B,H,L,D)] += w * v; }
      }
      free(sel_idx);
      float inv = 1.f / (denom + 1e-12f); for (int64_t d = 0; d < D; ++d) O[offset_bhld(b,h,i,d,B,H,L,D)] *= inv;
    }
  }
  free(block_idx); free(block_scores);
}

void sattn_rvv_block_topk_i4(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    sattn_shape_t shape,
    sattn_blocktopk_params_t params,
    float scale_q,
    float scale_k,
    float scale_v) {
  const int64_t B = shape.B, H = shape.H, L = shape.L, D = shape.D;
  const int block = params.block_size > 0 ? params.block_size : 64;
  const int64_t num_blocks = (L + block - 1) / block;
  int k_blocks = (int)((float)num_blocks * (params.keep_ratio > 0.f ? params.keep_ratio : 0.12f) + 0.999f);
  if (k_blocks < 1) k_blocks = 1;
  const float sc = 1.0f / sqrtf((float)D);
  const float s_q = scale_q > 0.f ? scale_q : 0.1f;
  const float s_k = scale_k > 0.f ? scale_k : 0.1f;
  const float s_v = scale_v > 0.f ? scale_v : 0.1f;
  int* block_idx = (int*)malloc((size_t)num_blocks * sizeof(int)); float* block_scores = (float*)malloc((size_t)num_blocks * sizeof(float));
  if (!block_idx || !block_scores) { if (block_idx) free(block_idx); if (block_scores) free(block_scores); return; }
  for (int64_t b = 0; b < B; ++b) for (int64_t h = 0; h < H; ++h) {
    for (int64_t i = 0; i < L; ++i) {
      for (int64_t d = 0; d < D; ++d) O[offset_bhld(b,h,i,d,B,H,L,D)] = 0.f;
      for (int64_t nb = 0; nb < num_blocks; ++nb) {
        int64_t s = nb * block; int64_t e = s + block; if (e > L) e = L; int64_t cnt = (int64_t)(e - s); int dot_i32 = 0;
        for (int64_t j = s; j < e; ++j) for (int64_t d = 0; d < D; ++d) {
          int qi = (int)f32_to_i4_symmetric(Q[offset_bhld(b,h,i,d,B,H,L,D)], s_q);
          int ki = (int)f32_to_i4_symmetric(K[offset_bhld(b,h,j,d,B,H,L,D)], s_k);
          dot_i32 += qi * ki;
        }
        float dot = (s_q * s_k) * (float)dot_i32; block_scores[nb] = cnt > 0 ? (dot / (float)cnt) : -1e30f; block_idx[nb] = (int)nb;
      }
      for (int64_t x = 0; x < num_blocks - 1; ++x) for (int64_t y = x + 1; y < num_blocks; ++y)
        if (block_scores[y] > block_scores[x]) { float ts = block_scores[x]; block_scores[x] = block_scores[y]; block_scores[y] = ts; int ti = block_idx[x]; block_idx[x] = block_idx[y]; block_idx[y] = ti; }
      float denom = 0.f; const int gtok = params.global_tokens > 0 ? (params.global_tokens < (int)L ? params.global_tokens : (int)L) : 0;
      int sel_cap = (int)(k_blocks * block + gtok); int sel_cnt = 0; int* sel_idx = (int*)malloc((size_t)sel_cap * sizeof(int));
      for (int j = 0; j < gtok && sel_cnt < sel_cap; ++j) sel_idx[sel_cnt++] = j;
      for (int kb = 0; kb < k_blocks && kb < num_blocks; ++kb) { int nb = block_idx[kb]; int64_t s = (int64_t)nb * block; int64_t e = s + block; if (e > L) e = L; for (int64_t j = s; j < e && sel_cnt < sel_cap; ++j) sel_idx[sel_cnt++] = (int)j; }
      for (int t = 0; t < sel_cnt; ++t) {
        int j = sel_idx[t]; int dot_i32 = 0;
        for (int64_t d = 0; d < D; ++d) {
          int qi = (int)f32_to_i4_symmetric(Q[offset_bhld(b,h,i,d,B,H,L,D)], s_q);
          int ki = (int)f32_to_i4_symmetric(K[offset_bhld(b,h,j,d,B,H,L,D)], s_k);
          dot_i32 += qi * ki;
        }
        float dot = (s_q * s_k) * (float)dot_i32; float w = expf(dot * sc); denom += w;
        for (int64_t d = 0; d < D; ++d) { int vi = (int)f32_to_i4_symmetric(V[offset_bhld(b,h,j,d,B,H,L,D)], s_v); float v = dequant_i4(vi, s_v); O[offset_bhld(b,h,i,d,B,H,L,D)] += w * v; }
      }
      free(sel_idx);
      float inv = 1.f / (denom + 1e-12f); for (int64_t d = 0; d < D; ++d) O[offset_bhld(b,h,i,d,B,H,L,D)] *= inv;
    }
  }
  free(block_idx); free(block_scores);
}

void sattn_rvv_block_topk_tiled(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    sattn_shape_t shape,
    sattn_blocktopk_params_t params,
    int tile_rows) {
  if (tile_rows <= 0) tile_rows = 4;
  const int64_t B = shape.B, H = shape.H, L = shape.L, D = shape.D;
  const int block = params.block_size > 0 ? params.block_size : 64;
  const int64_t num_blocks = (L + block - 1) / block;
  int k_blocks = (int)((float)num_blocks * (params.keep_ratio > 0.f ? params.keep_ratio : 0.12f) + 0.999f);
  if (k_blocks < 1) k_blocks = 1;
  const float scale = 1.0f / sqrtf((float)D);

  int* block_idx = (int*)malloc((size_t)num_blocks * sizeof(int));
  float* block_scores = (float*)malloc((size_t)num_blocks * sizeof(float));
  if (!block_idx || !block_scores) { if (block_idx) free(block_idx); if (block_scores) free(block_scores); return; }

  for (int64_t b = 0; b < B; ++b) for (int64_t h = 0; h < H; ++h) {
    for (int64_t ib = 0; ib < L; ib += tile_rows) {
      int64_t iend = ib + tile_rows; if (iend > L) iend = L;
      for (int64_t i = ib; i < iend; ++i) for (int64_t d = 0; d < D; ++d) O[offset_bhld(b,h,i,d,B,H,L,D)] = 0.f;
      for (int64_t i = ib; i < iend; ++i) {
        for (int64_t nb = 0; nb < num_blocks; ++nb) {
          int64_t s = nb * block; int64_t e = s + block; if (e > L) e = L;
          float dot = 0.f; int64_t cnt = (int64_t)(e - s);
#ifdef __riscv_vector
          dot = reduce_block_sumdot_f32_rvv(&Q[offset_bhld(b,h,i,0,B,H,L,D)], &K[offset_bhld(b,h,s,0,B,H,L,D)], cnt, D);
          _rvv_ctrs.br += (uint64_t)cnt * (uint64_t)D * sizeof(float) + (uint64_t)D * sizeof(float);
          _rvv_ctrs.mac += (uint64_t)cnt * (uint64_t)D;
#else
          for (int64_t j = s; j < e; ++j) for (int64_t d = 0; d < D; ++d) dot += Q[offset_bhld(b,h,i,d,B,H,L,D)] * K[offset_bhld(b,h,j,d,B,H,L,D)];
          _rvv_ctrs.br += (uint64_t)cnt * (uint64_t)D * sizeof(float) + (uint64_t)D * sizeof(float);
          _rvv_ctrs.mac += (uint64_t)cnt * (uint64_t)D;
#endif
          block_scores[nb] = cnt > 0 ? (dot / (float)cnt) : -1e30f; block_idx[nb] = (int)nb;
        }
        for (int64_t x = 0; x < num_blocks - 1; ++x) for (int64_t y = x + 1; y < num_blocks; ++y)
          if (block_scores[y] > block_scores[x]) { float ts = block_scores[x]; block_scores[x] = block_scores[y]; block_scores[y] = ts; int ti = block_idx[x]; block_idx[x] = block_idx[y]; block_idx[y] = ti; }
        float denom = 0.f; int gtok = params.global_tokens > 0 ? (params.global_tokens < (int)L ? params.global_tokens : (int)L) : 0;
        int sel_cap = (int)(k_blocks * block + gtok); int sel_cnt = 0;
        int* sel_idx = (int*)malloc((size_t)sel_cap * sizeof(int));
        for (int j = 0; j < gtok && sel_cnt < sel_cap; ++j) sel_idx[sel_cnt++] = j;
        for (int kb = 0; kb < k_blocks && kb < num_blocks; ++kb) { int nb = block_idx[kb]; int64_t s = (int64_t)nb * block; int64_t e = s + block; if (e > L) e = L; for (int64_t j = s; j < e && sel_cnt < sel_cap; ++j) sel_idx[sel_cnt++] = (int)j; }
#ifdef __riscv_vector
        float* K_sel = (float*)malloc((size_t)sel_cnt * (size_t)D * sizeof(float));
        float* V_sel = (float*)malloc((size_t)sel_cnt * (size_t)D * sizeof(float));
        if (K_sel && V_sel) {
          gather_rows_indexed_f32(&K[offset_bhld(b,h,0,0,B,H,L,D)], sel_idx, sel_cnt, D, K_sel);
          gather_rows_indexed_f32(&V[offset_bhld(b,h,0,0,B,H,L,D)], sel_idx, sel_cnt, D, V_sel);
          _rvv_ctrs.br += (uint64_t)sel_cnt * (uint64_t)D * sizeof(float) * 2;
          for (int t = 0; t < sel_cnt; ++t) {
            float dot = dot_f32_rvv(&Q[offset_bhld(b,h,i,0,B,H,L,D)], &K_sel[(int64_t)t * D], D);
            float w = expf(dot * scale); denom += w;
            axpy_f32_rvv(w, &V_sel[(int64_t)t * D], &O[offset_bhld(b,h,i,0,B,H,L,D)], D);
            _rvv_ctrs.br += (uint64_t)D * sizeof(float); _rvv_ctrs.bw += (uint64_t)D * sizeof(float); _rvv_ctrs.mac += (uint64_t)D;
          }
        } else {
          for (int t = 0; t < sel_cnt; ++t) { int j = sel_idx[t]; float dot = dot_f32_rvv(&Q[offset_bhld(b,h,i,0,B,H,L,D)], &K[offset_bhld(b,h,j,0,B,H,L,D)], D); float w = expf(dot * scale); denom += w; axpy_f32_rvv(w, &V[offset_bhld(b,h,j,0,B,H,L,D)], &O[offset_bhld(b,h,i,0,B,H,L,D)], D); }
        }
        if (K_sel) free(K_sel); if (V_sel) free(V_sel);
#else
        for (int t = 0; t < sel_cnt; ++t) {
          int j = sel_idx[t];
          float dot = 0.f;
          for (int64_t d = 0; d < D; ++d) dot += Q[offset_bhld(b,h,i,d,B,H,L,D)] * K[offset_bhld(b,h,j,d,B,H,L,D)];
          _rvv_ctrs.br += (uint64_t)D * sizeof(float) * 2;
          _rvv_ctrs.mac += (uint64_t)D;
          float w = expf(dot * scale); denom += w;
          for (int64_t d = 0; d < D; ++d) O[offset_bhld(b,h,i,d,B,H,L,D)] += w * V[offset_bhld(b,h,j,d,B,H,L,D)];
          _rvv_ctrs.br += (uint64_t)D * sizeof(float);
          _rvv_ctrs.bw += (uint64_t)D * sizeof(float);
        }
#endif
        free(sel_idx);
        float inv = 1.f / (denom + 1e-12f);
#ifdef __riscv_vector
        size_t idx = 0; for (; idx < (size_t)D;) { size_t vl = vsetvl_e32m1((size_t)(D - idx)); vfloat32m1_t vy = vle32_v_f32m1(&O[offset_bhld(b,h,i,0,B,H,L,D)] + idx, vl); vy = vfmul_vf_f32m1(vy, inv, vl); vse32_v_f32m1(&O[offset_bhld(b,h,i,0,B,H,L,D)] + idx, vy, vl); idx += vl; }
#else
        for (int64_t d = 0; d < D; ++d) O[offset_bhld(b,h,i,d,B,H,L,D)] *= inv;
#endif
      }
    }
  }
  free(block_idx); free(block_scores);
}


#undef MIN
#undef MAX
#define MIN(a,b) ((a)<(b)?(a):(b))
#define MAX(a,b) ((a)>(b)?(a):(b))

void sattn_rvv_nm_structured(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    sattn_shape_t shape,
    sattn_nm_params_t params) {
  int M = params.m > 0 ? params.m : 64;
  float keep = (params.n > 0 && params.m > 0) ? ((float)params.n / (float)params.m) : 0.25f;
  sattn_blocktopk_params_t p;
  p.block_size = M;
  p.keep_ratio = keep;
  p.global_tokens = 0;
  sattn_rvv_block_topk(Q,K,V,O,shape,p);
}

#ifdef __riscv_vector
void sattn_rvv_segmented_sum_f32(const float* src, float* dst,
                                 int64_t segments, int64_t seg_len) {
  for (int64_t s = 0; s < segments; ++s) {
    const float* p = src + s * seg_len;
    size_t i = 0; vfloat32m1_t vacc = vfmv_v_f_f32m1(0.0f, 1);
    for (; i < (size_t)seg_len;) {
      size_t vl = vsetvl_e32m1((size_t)(seg_len - i));
      vfloat32m1_t vx = vle32_v_f32m1(p + i, vl);
      vacc = vfadd_vv_f32m1(vacc, vx, vl);
      i += vl;
    }
    float sum = vfmv_f_s_f32m1_f32(vfredsum_vs_f32m1_f32m1(vundef_f32m1(), vacc, vfmv_v_f_f32m1(0.0f, 1), 1));
    dst[s] = sum;
    _rvv_ctrs.br += (uint64_t)seg_len * sizeof(float);
    _rvv_ctrs.bw += sizeof(float);
  }
}

void sattn_rvv_softmax_row_f32(float* row, int64_t D) {
  // Compute max
  float m = -INFINITY;
  {
    size_t i = 0; vfloat32m1_t vmaxv = vfmv_v_f_f32m1(-INFINITY, 1);
    for (; i < (size_t)D;) { size_t vl = vsetvl_e32m1((size_t)(D - i)); vfloat32m1_t vx = vle32_v_f32m1(row + i, vl); vmaxv = vfmax_vv_f32m1(vmaxv, vx, vl); i += vl; }
    m = vfmv_f_s_f32m1_f32(vfredmax_vs_f32m1_f32m1(vundef_f32m1(), vmaxv, vfmv_v_f_f32m1(-INFINITY, 1), 1));
  }
  // Compute exp(x-m) and sum
  float denom = 0.f; {
    size_t i = 0; vfloat32m1_t vsum = vfmv_v_f_f32m1(0.0f, 1);
    for (; i < (size_t)D;) { size_t vl = vsetvl_e32m1((size_t)(D - i)); vfloat32m1_t vx = vle32_v_f32m1(row + i, vl); vfloat32m1_t vy = vfncvt_f_f_w_f32m1(vfsub_vf_f32m1(vx, m, vl));
      // no vector exp; do scalar loop fallback per chunk
      float tmp[256]; size_t j = 0; for (; j < vl && j < 256; ++j) { tmp[j] = expf(((float*)&vy)[j]); }
      // store back
      for (j = 0; j < vl && j < 256; ++j) { row[i + j] = tmp[j]; }
      vfloat32m1_t vtmp = vle32_v_f32m1(row + i, vl); vsum = vfadd_vv_f32m1(vsum, vtmp, vl); i += vl; }
    denom = vfmv_f_s_f32m1_f32(vfredsum_vs_f32m1_f32m1(vundef_f32m1(), vsum, vfmv_v_f_f32m1(0.0f, 1), 1));
  }
  float inv = 1.f / (denom + 1e-12f);
  // Normalize
  { size_t i = 0; for (; i < (size_t)D;) { size_t vl = vsetvl_e32m1((size_t)(D - i)); vfloat32m1_t vx = vle32_v_f32m1(row + i, vl); vx = vfmul_vf_f32m1(vx, inv, vl); vse32_v_f32m1(row + i, vx, vl); i += vl; } }
  _rvv_ctrs.br += (uint64_t)D * sizeof(float); _rvv_ctrs.bw += (uint64_t)D * sizeof(float);
}
#else
void sattn_rvv_segmented_sum_f32(const float* src, float* dst,
                                 int64_t segments, int64_t seg_len) {
  for (int64_t s = 0; s < segments; ++s) {
    float acc = 0.f; for (int64_t i = 0; i < seg_len; ++i) acc += src[s*seg_len + i]; dst[s] = acc;
    _rvv_ctrs.br += (uint64_t)seg_len * sizeof(float); _rvv_ctrs.bw += sizeof(float);
  }
}
void sattn_rvv_softmax_row_f32(float* row, int64_t D) {
  float m = -INFINITY; for (int64_t i = 0; i < D; ++i) if (row[i] > m) m = row[i];
  float denom = 0.f; for (int64_t i = 0; i < D; ++i) { row[i] = expf(row[i] - m); denom += row[i]; }
  float inv = 1.f / (denom + 1e-12f); for (int64_t i = 0; i < D; ++i) row[i] *= inv;
  _rvv_ctrs.br += (uint64_t)D * sizeof(float); _rvv_ctrs.bw += (uint64_t)D * sizeof(float);
}
#endif

