#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "sparse_attention_rvv.h"

static void ref_block_topk(
    const float* Q, const float* K, const float* V, float* O,
    sattn_shape_t s, int block_size, float keep_ratio, int global_tokens) {
  const int64_t B = s.B, H = s.H, L = s.L, D = s.D;
  const int block = block_size > 0 ? block_size : 64;
  const int64_t num_blocks = (L + block - 1) / block;
  int k_blocks = (int)((float)num_blocks * (keep_ratio > 0.f ? keep_ratio : 0.12f) + 0.999f);
  if (k_blocks < 1) k_blocks = 1;
  float scale = 1.0f / sqrtf((float)D);
  int* block_idx = (int*)malloc((size_t)num_blocks * sizeof(int));
  float* block_scores = (float*)malloc((size_t)num_blocks * sizeof(float));
  int* sel_idx = (int*)malloc((size_t)(k_blocks * block + global_tokens) * sizeof(int));
  for (int64_t b = 0; b < B; ++b) for (int64_t h = 0; h < H; ++h) {
    for (int64_t i = 0; i < L; ++i) {
      for (int64_t d = 0; d < D; ++d) O[(((b*H+h)*L+i)*D + d)] = 0.f;
      for (int64_t nb = 0; nb < num_blocks; ++nb) {
        int64_t s0 = nb * block; int64_t e0 = s0 + block; if (e0 > L) e0 = L;
        float dot = 0.f; int64_t cnt = e0 - s0;
        for (int64_t j = s0; j < e0; ++j)
          for (int64_t d = 0; d < D; ++d)
            dot += Q[(((b*H+h)*L+i)*D + d)] * K[(((b*H+h)*L+j)*D + d)];
        block_scores[nb] = cnt > 0 ? (dot / (float)cnt) : -1e30f; block_idx[nb] = (int)nb;
      }
      for (int64_t x = 0; x < num_blocks - 1; ++x)
        for (int64_t y = x + 1; y < num_blocks; ++y)
          if (block_scores[y] > block_scores[x]) { float ts = block_scores[x]; block_scores[x] = block_scores[y]; block_scores[y] = ts; int ti = block_idx[x]; block_idx[x] = block_idx[y]; block_idx[y] = ti; }
      int sel_cnt = 0; int sel_cap = k_blocks * block + (global_tokens > 0 ? global_tokens : 0);
      for (int j = 0; j < global_tokens && sel_cnt < sel_cap; ++j) sel_idx[sel_cnt++] = j;
      for (int kb = 0; kb < k_blocks && kb < num_blocks; ++kb) {
        int nb = block_idx[kb]; int64_t s0 = (int64_t)nb * block; int64_t e0 = s0 + block; if (e0 > L) e0 = L;
        for (int64_t j = s0; j < e0 && sel_cnt < sel_cap; ++j) sel_idx[sel_cnt++] = (int)j;
      }
      float denom = 0.f;
      for (int t = 0; t < sel_cnt; ++t) {
        int j = sel_idx[t]; float dot = 0.f;
        for (int64_t d = 0; d < D; ++d) dot += Q[(((b*H+h)*L+i)*D + d)] * K[(((b*H+h)*L+j)*D + d)];
        float w = expf(dot * scale); denom += w;
        for (int64_t d = 0; d < D; ++d) O[(((b*H+h)*L+i)*D + d)] += w * V[(((b*H+h)*L+j)*D + d)];
      }
      float inv = 1.f / (denom + 1e-12f);
      for (int64_t d = 0; d < D; ++d) O[(((b*H+h)*L+i)*D + d)] *= inv;
    }
  }
  free(block_idx); free(block_scores); free(sel_idx);
}

int main() {
  sattn_shape_t s = { .B = 1, .H = 1, .L = 96, .D = 32 };
  sattn_blocktopk_params_t p = { .block_size = 16, .keep_ratio = 0.25f, .global_tokens = 8 };
  size_t elems = (size_t)s.B * s.H * s.L * s.D;
  float *Q=(float*)malloc(elems*sizeof(float)), *K=(float*)malloc(elems*sizeof(float)), *V=(float*)malloc(elems*sizeof(float));
  float *Oref=(float*)malloc(elems*sizeof(float)), *Orvv=(float*)malloc(elems*sizeof(float));
  if(!Q||!K||!V||!Oref||!Orvv) return 2;
  for (size_t i = 0; i < elems; ++i) { Q[i] = sinf((float)i*0.01f); K[i] = cosf((float)i*0.02f); V[i] = sinf((float)i*0.03f); }
  ref_block_topk(Q,K,V,Oref,s,p.block_size,p.keep_ratio,p.global_tokens);
  sattn_rvv_block_topk(Q,K,V,Orvv,s,p);
  double max_abs = 0.0; for (size_t i=0;i<elems;++i){ double d=fabs((double)Oref[i]-(double)Orvv[i]); if(d>max_abs) max_abs=d; }
  if (max_abs < 1e-4) { printf("MATCH max_abs=%.6g\n", max_abs); return 0; }
  else { printf("MISMATCH max_abs=%.6g\n", max_abs); return 1; }
}


