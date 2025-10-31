#include "backends/rvv/include/sparse_attention_rvv.h"

#include <math.h>
#include <stddef.h>

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
        // dynamic scratch for scores
        // For simplicity, compute softmax in two passes without extra alloc
        float m = -INFINITY;
        for (int64_t j = j_left; j < j_right; ++j) {
          float dot = 0.f;
          for (int64_t d = 0; d < D; ++d) {
            float qd = Q[offset_bhld(b, h, i, d, B, H, L, D)];
            float kd = K[offset_bhld(b, h, j, d, B, H, L, D)];
            dot += qd * kd;
          }
          dot *= scale;
          if (dot > m) m = dot;
        }
        float denom = 0.f;
        for (int64_t j = j_left; j < j_right; ++j) {
          float dot = 0.f;
          for (int64_t d = 0; d < D; ++d) {
            float qd = Q[offset_bhld(b, h, i, d, B, H, L, D)];
            float kd = K[offset_bhld(b, h, j, d, B, H, L, D)];
            dot += qd * kd;
          }
          dot = expf(dot * scale - m);
          denom += dot;
          for (int64_t d = 0; d < D; ++d) {
            float vd = V[offset_bhld(b, h, j, d, B, H, L, D)];
            O[offset_bhld(b, h, i, d, B, H, L, D)] += dot * vd;
          }
        }
        float inv = 1.f / (denom + 1e-12f);
        for (int64_t d = 0; d < D; ++d) {
          O[offset_bhld(b, h, i, d, B, H, L, D)] *= inv;
        }
      }
    }
  }
}


