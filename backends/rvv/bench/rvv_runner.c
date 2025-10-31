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
  long B=1,H=1,L=128,D=32, window=8, block_size=64, global_tokens=0, nm_n=0, nm_m=0, lsh_buckets=0, tile_rows=0;
  long keep_ratio_x1000 = 120; // 0.12
  (void)args(argc, argv, "--spec", &spec);
  argi(argc, argv, "--B", &B); argi(argc, argv, "--H", &H); argi(argc, argv, "--L", &L); argi(argc, argv, "--D", &D);
  argi(argc, argv, "--window", &window); argi(argc, argv, "--block_size", &block_size);
  argi(argc, argv, "--global_tokens", &global_tokens); argi(argc, argv, "--nm_n", &nm_n); argi(argc, argv, "--nm_m", &nm_m);
  argi(argc, argv, "--lsh_buckets", &lsh_buckets); argi(argc, argv, "--keep_x1000", &keep_ratio_x1000);
  argi(argc, argv, "--tile_rows", &tile_rows);
  sattn_shape_t s = { .B = B, .H = H, .L = L, .D = D };
  size_t elems = (size_t)B * H * L * D;
  float *Q=(float*)malloc(elems*sizeof(float)), *K=(float*)malloc(elems*sizeof(float)), *V=(float*)malloc(elems*sizeof(float));
  float *O=(float*)malloc(elems*sizeof(float));
  if(!Q||!K||!V||!O) return 2;
  for (size_t i = 0; i < elems; ++i) { Q[i] = sinf((float)i*0.01f); K[i] = cosf((float)i*0.02f); V[i] = sinf((float)i*0.03f); }
  if (strcmp(spec, "sliding_window")==0) {
    sattn_params_t p = { .window_size = (int)window, .block_size = (int)block_size };
    if (tile_rows > 1) {
      sattn_rvv_sliding_global_tiled(Q,K,V,O,s,p,(int)tile_rows);
    } else {
      sattn_rvv_sliding_global(Q,K,V,O,s,p);
    }
  } else if (strcmp(spec, "block_local_global")==0 || strcmp(spec, "bsr")==0) {
    sattn_blocktopk_params_t p = { .block_size=(int)block_size, .keep_ratio=(float)keep_ratio_x1000/1000.0f, .global_tokens=(int)global_tokens };
    sattn_rvv_block_topk(Q,K,V,O,s,p);
  } else if (strcmp(spec, "nm_structured")==0) {
    sattn_nm_params_t p = { .n = (int)nm_n, .m = (int)nm_m };
    sattn_rvv_nm_structured(Q,K,V,O,s,p);
  } else if (strcmp(spec, "topk_per_query")==0) {
    sattn_blocktopk_params_t p = { .block_size=(int)block_size, .keep_ratio=(float)keep_ratio_x1000/1000.0f, .global_tokens=0 };
    sattn_rvv_block_topk(Q,K,V,O,s,p);
  } else if (strcmp(spec, "lsh")==0) {
    sattn_lsh_params_t p = { .buckets = (int)lsh_buckets };
    sattn_rvv_lsh(Q,K,V,O,s,p);
  } else if (strcmp(spec, "sliding_window_tiled")==0) {
    sattn_params_t p = { .window_size = (int)window, .block_size = (int)block_size };
    sattn_rvv_sliding_global_tiled(Q,K,V,O,s,p,4);
  } else if (strcmp(spec, "block_local_global_tiled")==0) {
    sattn_blocktopk_params_t p = { .block_size=(int)block_size, .keep_ratio=(float)keep_ratio_x1000/1000.0f, .global_tokens=(int)global_tokens };
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


