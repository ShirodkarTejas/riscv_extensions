#include "backends/rvv/include/sparse_attention_rvv.h"

#include <math.h>
#include <stddef.h>
#ifdef __riscv_vector
#include <riscv_vector.h>
#endif

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
  const float scale = 1.0f / sqrtf((float)D);

  for (int64_t b = 0; b < B; ++b) {
    for (int64_t h = 0; h < H; ++h) {
      for (int64_t i = 0; i < L; ++i) {
        const int64_t j_left = i - window > 0 ? i - window : 0;
        const int64_t j_right = i + window + 1 < L ? i + window + 1 : L;
        const int span = (int)(j_right - j_left);
        // compute scores over [j_left, j_right)
        // store in stack buffer if small; fallback alloc if large
        // conservative simple path
        for (int64_t d = 0; d < D; ++d) {
          O[offset_bhld(b, h, i, d, B, H, L, D)] = 0.f;
        }
        if (span <= 0) continue;
        // For simplicity, compute softmax in two passes without extra alloc
        float m = -INFINITY;
        for (int64_t j = j_left; j < j_right; ++j) {
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
          dot *= scale;
          if (dot > m) m = dot;
        }
        float denom = 0.f;
        for (int64_t j = j_left; j < j_right; ++j) {
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
#else
          for (int64_t d = 0; d < D; ++d) {
            float vd = V[offset_bhld(b, h, j, d, B, H, L, D)];
            O[offset_bhld(b, h, i, d, B, H, L, D)] += w * vd;
          }
#endif
        }
        float inv = 1.f / (denom + 1e-12f);
#ifdef __riscv_vector
        // scale O row by inv
        axpy_f32_rvv(0.0f, &O[offset_bhld(b, h, i, 0, B, H, L, D)], &O[offset_bhld(b, h, i, 0, B, H, L, D)], 0); // no-op to satisfy structure
        size_t k = 0; (void)k; // avoid unused
        // reuse axpy with alpha by multiplying y in-place
        size_t idx = 0; for (; idx < (size_t)D;) { size_t vl = vsetvl_e32m1((size_t)(D - idx)); vfloat32m1_t vy = vle32_v_f32m1(&O[offset_bhld(b, h, i, 0, B, H, L, D)] + idx, vl); vy = vfmul_vf_f32m1(vy, inv, vl); vse32_v_f32m1(&O[offset_bhld(b, h, i, 0, B, H, L, D)] + idx, vy, vl); idx += vl; }
#else
        for (int64_t d = 0; d < D; ++d) {
          O[offset_bhld(b, h, i, d, B, H, L, D)] *= inv;
        }
#endif
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

  for (int64_t b = 0; b < B; ++b) {
    for (int64_t h = 0; h < H; ++h) {
      // Precompute K block means: mean over tokens in block for each block
      // Here we compute on-the-fly per row for simplicity; an optimization would cache
      for (int64_t i = 0; i < L; ++i) {
        // zero output row
        for (int64_t d = 0; d < D; ++d) O[offset_bhld(b, h, i, d, B, H, L, D)] = 0.f;
        // score blocks
        for (int64_t nb = 0; nb < num_blocks; ++nb) {
          int64_t s = nb * block;
          int64_t e = s + block; if (e > L) e = L;
          // mean over block
          float dot = 0.f;
          int64_t cnt = 0;
          for (int64_t j = s; j < e; ++j) {
#ifdef __riscv_vector
            dot += dot_f32_rvv(
                &Q[offset_bhld(b, h, i, 0, B, H, L, D)],
                &K[offset_bhld(b, h, j, 0, B, H, L, D)],
                D);
#else
            for (int64_t d = 0; d < D; ++d) dot += Q[offset_bhld(b, h, i, d, B, H, L, D)] * K[offset_bhld(b, h, j, d, B, H, L, D)];
#endif
            cnt++;
          }
          block_scores[nb] = cnt > 0 ? (dot / (float)cnt) : -1e30f;
          block_idx[nb] = (int)nb;
        }
        // select top k_blocks by partial selection (naive O(N log N))
        // simple sort by score descending
        for (int64_t x = 0; x < num_blocks - 1; ++x) {
          for (int64_t y = x + 1; y < num_blocks; ++y) {
            if (block_scores[y] > block_scores[x]) { float ts = block_scores[x]; block_scores[x] = block_scores[y]; block_scores[y] = ts; int ti = block_idx[x]; block_idx[x] = block_idx[y]; block_idx[y] = ti; }
          }
        }
        // accumulate attention over selected tokens
        float denom = 0.f;
        // union with global tokens: tokens [0, global_tokens)
        const int gtok = params.global_tokens > 0 ? (params.global_tokens < (int)L ? params.global_tokens : (int)L) : 0;

        // handle selected blocks
        for (int kb = 0; kb < k_blocks && kb < num_blocks; ++kb) {
          int nb = block_idx[kb];
          int64_t s = (int64_t)nb * block;
          int64_t e = s + block; if (e > L) e = L;
          for (int64_t j = s; j < e; ++j) {
            float dot = 0.f;
#ifdef __riscv_vector
            dot = dot_f32_rvv(
                &Q[offset_bhld(b, h, i, 0, B, H, L, D)],
                &K[offset_bhld(b, h, j, 0, B, H, L, D)],
                D);
#else
            for (int64_t d = 0; d < D; ++d) dot += Q[offset_bhld(b, h, i, d, B, H, L, D)] * K[offset_bhld(b, h, j, d, B, H, L, D)];
#endif
            float w = expf(dot * scale);
            denom += w;
#ifdef __riscv_vector
            axpy_f32_rvv(w, &V[offset_bhld(b, h, j, 0, B, H, L, D)], &O[offset_bhld(b, h, i, 0, B, H, L, D)], D);
#else
            for (int64_t d = 0; d < D; ++d) O[offset_bhld(b, h, i, d, B, H, L, D)] += w * V[offset_bhld(b, h, j, d, B, H, L, D)];
#endif
          }
        }
        // handle global tokens
        for (int j = 0; j < gtok; ++j) {
          float dot = 0.f;
#ifdef __riscv_vector
          dot = dot_f32_rvv(
              &Q[offset_bhld(b, h, i, 0, B, H, L, D)],
              &K[offset_bhld(b, h, j, 0, B, H, L, D)],
              D);
#else
          for (int64_t d = 0; d < D; ++d) dot += Q[offset_bhld(b, h, i, d, B, H, L, D)] * K[offset_bhld(b, h, j, d, B, H, L, D)];
#endif
          float w = expf(dot * scale);
          denom += w;
#ifdef __riscv_vector
          axpy_f32_rvv(w, &V[offset_bhld(b, h, j, 0, B, H, L, D)], &O[offset_bhld(b, h, i, 0, B, H, L, D)], D);
#else
          for (int64_t d = 0; d < D; ++d) O[offset_bhld(b, h, i, d, B, H, L, D)] += w * V[offset_bhld(b, h, j, d, B, H, L, D)];
#endif
        }

        float inv = 1.f / (denom + 1e-12f);
#ifdef __riscv_vector
        size_t idx = 0; for (; idx < (size_t)D;) { size_t vl = vsetvl_e32m1((size_t)(D - idx)); vfloat32m1_t vy = vle32_v_f32m1(&O[offset_bhld(b, h, i, 0, B, H, L, D)] + idx, vl); vy = vfmul_vf_f32m1(vy, inv, vl); vse32_v_f32m1(&O[offset_bhld(b, h, i, 0, B, H, L, D)] + idx, vy, vl); idx += vl; }
#else
        for (int64_t d = 0; d < D; ++d) O[offset_bhld(b, h, i, d, B, H, L, D)] *= inv;
#endif
      }
    }
  }

  free(block_idx);
  free(block_scores);
}


