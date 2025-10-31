#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "sparse_attention_rvv.h"

static void ref_sw(
    const float* Q, const float* K, const float* V, float* O,
    sattn_shape_t s, int window) {
  const int64_t B = s.B, H = s.H, L = s.L, D = s.D;
  const float scale = 1.0f / sqrtf((float)D);
  for (int64_t b = 0; b < B; ++b) for (int64_t h = 0; h < H; ++h) {
    for (int64_t i = 0; i < L; ++i) {
      for (int64_t d = 0; d < D; ++d) O[(((b*H+h)*L+i)*D + d)] = 0.f;
      int64_t jl = i - window > 0 ? i - window : 0;
      int64_t jr = i + window + 1 < L ? i + window + 1 : L;
      float m = -INFINITY;
      for (int64_t j = jl; j < jr; ++j) {
        float dot = 0.f; for (int64_t d = 0; d < D; ++d)
          dot += Q[(((b*H+h)*L+i)*D + d)] * K[(((b*H+h)*L+j)*D + d)];
        dot *= scale; if (dot > m) m = dot;
      }
      float denom = 0.f;
      for (int64_t j = jl; j < jr; ++j) {
        float dot = 0.f; for (int64_t d = 0; d < D; ++d)
          dot += Q[(((b*H+h)*L+i)*D + d)] * K[(((b*H+h)*L+j)*D + d)];
        float w = expf(dot * scale - m); denom += w;
        for (int64_t d = 0; d < D; ++d)
          O[(((b*H+h)*L+i)*D + d)] += w * V[(((b*H+h)*L+j)*D + d)];
      }
      float inv = 1.f / (denom + 1e-12f);
      for (int64_t d = 0; d < D; ++d) O[(((b*H+h)*L+i)*D + d)] *= inv;
    }
  }
}

int main() {
  sattn_shape_t s = { .B = 1, .H = 1, .L = 96, .D = 32 };
  sattn_params_t p = { .window_size = 8, .block_size = 64 };
  size_t elems = (size_t)s.B * s.H * s.L * s.D;
  float *Q=(float*)malloc(elems*sizeof(float)), *K=(float*)malloc(elems*sizeof(float)), *V=(float*)malloc(elems*sizeof(float));
  float *Oref=(float*)malloc(elems*sizeof(float)), *Orvv=(float*)malloc(elems*sizeof(float));
  if(!Q||!K||!V||!Oref||!Orvv) return 2;
  for (size_t i = 0; i < elems; ++i) { Q[i] = sinf((float)i*0.01f); K[i] = cosf((float)i*0.02f); V[i] = sinf((float)i*0.03f); }
  ref_sw(Q,K,V,Oref,s,p.window_size);
  sattn_rvv_sliding_global(Q,K,V,Orvv,s,p);
  double max_abs = 0.0; for (size_t i=0;i<elems;++i){ double d=fabs((double)Oref[i]-(double)Orvv[i]); if(d>max_abs) max_abs=d; }
  if (max_abs < 1e-4) { printf("MATCH max_abs=%.6g\n", max_abs); return 0; }
  else { printf("MISMATCH max_abs=%.6g\n", max_abs); return 1; }
}


