#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "sparse_attention_rvv.h"

static int argi(int argc, char** argv, const char* k, long* out) {
  for (int i=1;i<argc-1;++i) if (strcmp(argv[i], k)==0) { *out = strtol(argv[i+1], NULL, 10); return 1; } return 0;
}
static int args(int argc, char** argv, const char* k, const char** out) {
  for (int i=1;i<argc-1;++i) if (strcmp(argv[i], k)==0) { *out = argv[i+1]; return 1; } return 0;
}

int main(int argc, char** argv) {
  const char* spec = "sliding_window";
  const char* precision = "fp32"; // fp32 | bf16 | i8 | i4
  float scale_q = 0.05f, scale_k = 0.05f, scale_v = 0.05f;
  long B=1,H=1,L=128,D=32, window=8, block_size=64, global_tokens=0, nm_n=0, nm_m=0, lsh_buckets=0, tile_rows=0;
  long gqa_group_size=1, comp_block_size=0;
  int autotune = 0;
  int calibrate = 0;
  const char* indices_path = NULL;
  long keep_ratio_x1000 = 120; // 0.12
  (void)args(argc, argv, "--spec", &spec);
  (void)args(argc, argv, "--precision", &precision);
  (void)args(argc, argv, "--indices", &indices_path);
  argi(argc, argv, "--B", &B); argi(argc, argv, "--H", &H); argi(argc, argv, "--L", &L); argi(argc, argv, "--D", &D);
  argi(argc, argv, "--window", &window); argi(argc, argv, "--block_size", &block_size);
  argi(argc, argv, "--global_tokens", &global_tokens); argi(argc, argv, "--nm_n", &nm_n); argi(argc, argv, "--nm_m", &nm_m);
  argi(argc, argv, "--lsh_buckets", &lsh_buckets); argi(argc, argv, "--keep_x1000", &keep_ratio_x1000);
  argi(argc, argv, "--tile_rows", &tile_rows);
  argi(argc, argv, "--gqa_group_size", &gqa_group_size);
  argi(argc, argv, "--comp_block_size", &comp_block_size);
  {
    long tmp;
    if (argi(argc, argv, "--scale_q_x1000", &tmp)) scale_q = (float)tmp / 1000.0f;
    if (argi(argc, argv, "--scale_k_x1000", &tmp)) scale_k = (float)tmp / 1000.0f;
    if (argi(argc, argv, "--scale_v_x1000", &tmp)) scale_v = (float)tmp / 1000.0f;
  }
  for (int i=1;i<argc;++i) {
    if (strcmp(argv[i], "--autotune")==0) autotune = 1;
    if (strcmp(argv[i], "--calibrate")==0) calibrate = 1;
  }
  sattn_shape_t s = { .B = B, .H = H, .L = L, .D = D };
  size_t elems = (size_t)B * H * L * D;
  float *Q=(float*)malloc(elems*sizeof(float)), *K=(float*)malloc(elems*sizeof(float)), *V=(float*)malloc(elems*sizeof(float));
  float *O=(float*)malloc(elems*sizeof(float));
  if(!Q||!K||!V||!O) return 2;
  for (size_t i = 0; i < elems; ++i) { Q[i] = sinf((float)i*0.01f); K[i] = cosf((float)i*0.02f); V[i] = sinf((float)i*0.03f); }

  if (calibrate) {
    // Compute per-tensor max-abs and emit recommended symmetric per-tensor scales
    float maxq = 0.f, maxk = 0.f, maxv = 0.f;
    for (size_t i = 0; i < elems; ++i) { float a=fabsf(Q[i]); if (a>maxq) maxq=a; }
    for (size_t i = 0; i < elems; ++i) { float a=fabsf(K[i]); if (a>maxk) maxk=a; }
    for (size_t i = 0; i < elems; ++i) { float a=fabsf(V[i]); if (a>maxv) maxv=a; }
    int use4 = (strcmp(precision, "i4")==0);
    float denom = use4 ? 7.0f : 127.0f;
    if (maxq <= 0) maxq = 1.f; if (maxk <= 0) maxk = 1.f; if (maxv <= 0) maxv = 1.f;
    float sq = maxq/denom, sk = maxk/denom, sv = maxv/denom;
    printf("calibrate: precision=%s scale_q=%.6f scale_k=%.6f scale_v=%.6f scale_q_x1000=%d scale_k_x1000=%d scale_v_x1000=%d\n",
           use4?"i4":"i8", sq, sk, sv, (int)lrintf(sq*1000.f), (int)lrintf(sk*1000.f), (int)lrintf(sv*1000.f));
    free(Q); free(K); free(V); free(O); return 0;
  }
  if (strcmp(spec, "sliding_window")==0 && autotune) {
    int candidates[] = {1,2,4,8}; size_t nc = sizeof(candidates)/sizeof(candidates[0]);
    unsigned long long best_bytes = ~0ull; int best_tr = 1; double best_ck = 0.0;
    for (size_t ci=0; ci<nc; ++ci) {
      int tr = candidates[ci]; if (tr > L) continue;
      sattn_rvv_counters_reset();
      sattn_params_t p = { .window_size = (int)window, .block_size = (int)block_size };
      if (tr > 1) sattn_rvv_sliding_global_tiled(Q,K,V,O,s,p,tr); else sattn_rvv_sliding_global(Q,K,V,O,s,p);
      sattn_rvv_counters_t ctr; sattn_rvv_counters_get(&ctr);
      double acc=0.0; for (size_t i=0;i<elems;++i) acc += O[i];
      if (ctr.bytes_read < best_bytes) { best_bytes = ctr.bytes_read; best_tr = tr; best_ck = acc; }
    }
    printf("autotune: spec=sliding_window tile_rows=%d rvv_bytes_read=%llu checksum=%.6f\n", best_tr, (unsigned long long)best_bytes, best_ck);
    free(Q); free(K); free(V); free(O); return 0;
  } else if (strcmp(spec, "sliding_window")==0) {
    sattn_params_t p = { .window_size = (int)window, .block_size = (int)block_size };
    if (strcmp(precision, "bf16")==0) {
      sattn_rvv_sliding_global_bf16(Q,K,V,O,s,p);
    } else if (strcmp(precision, "i8")==0) {
      sattn_rvv_sliding_global_i8(Q,K,V,O,s,p, scale_q, scale_k, scale_v);
    } else if (strcmp(precision, "i4")==0) {
      sattn_rvv_sliding_global_i4(Q,K,V,O,s,p, scale_q, scale_k, scale_v);
    } else {
      if (tile_rows > 1) sattn_rvv_sliding_global_tiled(Q,K,V,O,s,p,(int)tile_rows);
      else sattn_rvv_sliding_global(Q,K,V,O,s,p);
    }
  } else if ((strcmp(spec, "block_local_global")==0 || strcmp(spec, "bsr")==0) && autotune) {
    int candidates[] = {1,2,4,8}; size_t nc = sizeof(candidates)/sizeof(candidates[0]);
    unsigned long long best_bytes = ~0ull; int best_tr = 1; double best_ck = 0.0;
    for (size_t ci=0; ci<nc; ++ci) {
      int tr = candidates[ci]; if (tr > L) continue;
      sattn_rvv_counters_reset();
      sattn_blocktopk_params_t p = { .block_size=(int)block_size, .keep_ratio=(float)keep_ratio_x1000/1000.0f, .global_tokens=(int)global_tokens, .gqa_group_size=(int)gqa_group_size, .comp_block_size=(int)comp_block_size };
      if (tr > 1) sattn_rvv_block_topk_tiled(Q,K,V,O,s,p,tr); else sattn_rvv_block_topk(Q,K,V,O,s,p);
      sattn_rvv_counters_t ctr; sattn_rvv_counters_get(&ctr);
      double acc=0.0; for (size_t i=0;i<elems;++i) acc += O[i];
      if (ctr.bytes_read < best_bytes) { best_bytes = ctr.bytes_read; best_tr = tr; best_ck = acc; }
    }
    printf("autotune: spec=block_local_global tile_rows=%d rvv_bytes_read=%llu checksum=%.6f\n", best_tr, (unsigned long long)best_bytes, best_ck);
    free(Q); free(K); free(V); free(O); return 0;
  } else if (strcmp(spec, "block_local_global")==0 || strcmp(spec, "bsr")==0) {
    // If indices are provided, override selection
    if (indices_path) {
      int cap = (int)L; int *sel = (int*)malloc(sizeof(int)*cap); int sc=0;
      FILE* f = fopen(indices_path, "r"); if (f) {
        char buf[128]; while (fgets(buf,sizeof(buf),f)) { int nb = atoi(buf); if (nb < 0) continue; long srow = (long)nb * block_size; long erow = srow + block_size; if (erow > L) erow = L; for (long r=srow; r<erow && sc<cap; ++r) sel[sc++] = (int)r; } fclose(f); }
      for (int j=0; j<global_tokens && sc<cap; ++j) sel[sc++] = j;
      sattn_rvv_block_topk_apply_selection(Q,K,V,O,s,(int)block_size, sel, sc);
      free(sel);
    } else {
      sattn_blocktopk_params_t p = { .block_size=(int)block_size, .keep_ratio=(float)keep_ratio_x1000/1000.0f, .global_tokens=(int)global_tokens, .gqa_group_size=(int)gqa_group_size, .comp_block_size=(int)comp_block_size };
      if (strcmp(precision, "bf16")==0) {
        sattn_rvv_block_topk_bf16(Q,K,V,O,s,p);
      } else if (strcmp(precision, "i8")==0) {
        sattn_rvv_block_topk_i8(Q,K,V,O,s,p, scale_q, scale_k, scale_v);
      } else if (strcmp(precision, "i4")==0) {
        sattn_rvv_block_topk_i4(Q,K,V,O,s,p, scale_q, scale_k, scale_v);
      } else {
        sattn_rvv_block_topk(Q,K,V,O,s,p);
      }
    }
  } else if (strcmp(spec, "nm_structured")==0) {
    sattn_nm_params_t p = { .n = (int)nm_n, .m = (int)nm_m };
    sattn_rvv_nm_structured(Q,K,V,O,s,p);
  } else if (strcmp(spec, "topk_per_query")==0) {
    sattn_blocktopk_params_t p = { .block_size=(int)block_size, .keep_ratio=(float)keep_ratio_x1000/1000.0f, .global_tokens=0, .gqa_group_size=(int)gqa_group_size, .comp_block_size=(int)comp_block_size };
    if (strcmp(precision, "bf16")==0) {
      sattn_rvv_block_topk_bf16(Q,K,V,O,s,p);
    } else if (strcmp(precision, "i8")==0) {
      sattn_rvv_block_topk_i8(Q,K,V,O,s,p, scale_q, scale_k, scale_v);
    } else if (strcmp(precision, "i4")==0) {
      sattn_rvv_block_topk_i4(Q,K,V,O,s,p, scale_q, scale_k, scale_v);
    } else {
      sattn_rvv_block_topk(Q,K,V,O,s,p);
    }
  } else if (strcmp(spec, "lsh")==0) {
    sattn_lsh_params_t p = { .buckets = (int)lsh_buckets };
    sattn_rvv_lsh(Q,K,V,O,s,p);
  } else if (strcmp(spec, "sliding_window_tiled")==0) {
    sattn_params_t p = { .window_size = (int)window, .block_size = (int)block_size };
    sattn_rvv_sliding_global_tiled(Q,K,V,O,s,p,4);
  } else if (strcmp(spec, "block_local_global_tiled")==0) {
    sattn_blocktopk_params_t p = { .block_size=(int)block_size, .keep_ratio=(float)keep_ratio_x1000/1000.0f, .global_tokens=(int)global_tokens, .gqa_group_size=(int)gqa_group_size, .comp_block_size=(int)comp_block_size };
    sattn_rvv_block_topk_tiled(Q,K,V,O,s,p,4);
  } else {
    fprintf(stderr, "unknown spec %s\n", spec);
    return 3;
  }
  double acc=0.0; for (size_t i=0;i<elems;++i) acc += O[i];
  sattn_rvv_counters_t ctrs; sattn_rvv_counters_get(&ctrs);
  printf("spec=%s checksum=%.6f rvv_bytes_read=%llu bytes_written=%llu mac_flops=%llu\n",
         spec, acc,
         (unsigned long long)ctrs.bytes_read,
         (unsigned long long)ctrs.bytes_written,
         (unsigned long long)ctrs.mac_flops);
  free(Q); free(K); free(V); free(O);
  return 0;
}


