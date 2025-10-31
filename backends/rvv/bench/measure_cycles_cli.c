#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "backends/rvv/include/sparse_attention_rvv.h"

static int parse_arg(int argc, char** argv, const char* key, long* out) {
  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], key) == 0 && i + 1 < argc) {
      *out = strtol(argv[i+1], NULL, 10);
      return 1;
    }
  }
  return 0;
}

int main(int argc, char** argv) {
  sattn_shape_t s = { .B = 1, .H = 2, .L = 256, .D = 64 };
  sattn_params_t p = { .window_size = 16, .block_size = 64 };
  long v;
  if (parse_arg(argc, argv, "--B", &v)) s.B = v;
  if (parse_arg(argc, argv, "--H", &v)) s.H = v;
  if (parse_arg(argc, argv, "--L", &v)) s.L = v;
  if (parse_arg(argc, argv, "--D", &v)) s.D = v;
  if (parse_arg(argc, argv, "--window", &v)) p.window_size = v;

  size_t elems = (size_t)s.B * s.H * s.L * s.D;
  float* Q = (float*)malloc(elems * sizeof(float));
  float* K = (float*)malloc(elems * sizeof(float));
  float* V = (float*)malloc(elems * sizeof(float));
  float* O = (float*)malloc(elems * sizeof(float));
  if (!Q || !K || !V || !O) return 2;
  for (size_t i = 0; i < elems; ++i) {
    Q[i] = (float)((i * 1103515245u + 12345u) & 0xFFFF) / 65536.f - 0.5f;
    K[i] = (float)((i * 1664525u + 1013904223u) & 0xFFFF) / 65536.f - 0.5f;
    V[i] = (float)((i * 22695477u + 1u) & 0xFFFF) / 65536.f - 0.5f;
  }

  uint64_t c0 = sattn_rdcycle();
  sattn_rvv_sliding_global(Q, K, V, O, s, p);
  uint64_t c1 = sattn_rdcycle();
  printf("cycles=%llu B=%lld H=%lld L=%lld D=%lld window=%d\n",
         (unsigned long long)(c1 - c0), (long long)s.B, (long long)s.H,
         (long long)s.L, (long long)s.D, p.window_size);

  double acc = 0.0;
  for (size_t i = 0; i < elems; ++i) acc += O[i];
  printf("checksum=%.6f\n", acc);

  free(Q); free(K); free(V); free(O);
  return 0;
}


