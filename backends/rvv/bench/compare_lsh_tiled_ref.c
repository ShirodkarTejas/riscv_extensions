#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "sparse_attention_rvv.h"

int main() {
  sattn_shape_t s = { .B = 1, .H = 1, .L = 128, .D = 32 };
  sattn_lsh_params_t p = { .buckets = 8 };
  size_t elems = (size_t)s.B * s.H * s.L * s.D;
  float *Q=(float*)malloc(elems*sizeof(float)), *K=(float*)malloc(elems*sizeof(float)), *V=(float*)malloc(elems*sizeof(float));
  float *O0=(float*)malloc(elems*sizeof(float)), *O1=(float*)malloc(elems*sizeof(float));
  if(!Q||!K||!V||!O0||!O1) return 2;
  for (size_t i = 0; i < elems; ++i) { Q[i] = sinf((float)i*0.01f); K[i] = cosf((float)i*0.02f); V[i] = sinf((float)i*0.03f); }
  sattn_rvv_lsh(Q,K,V,O0,s,p);
  sattn_rvv_lsh_tiled(Q,K,V,O1,s,p,4);
  double max_abs = 0.0; for (size_t i=0;i<elems;++i){ double d=fabs((double)O0[i]-(double)O1[i]); if(d>max_abs) max_abs=d; }
  if (max_abs < 1e-4) { printf("MATCH max_abs=%.6g\n", max_abs); return 0; }
  else { printf("MISMATCH max_abs=%.6g\n", max_abs); return 1; }
}


