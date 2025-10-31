#include <stdio.h>
#include <math.h>
#include "sparse_attention_rvv.h"

static int nearly_eq(float a, float b, float eps) { return fabsf(a-b) <= eps * fmaxf(1.0f, fmaxf(fabsf(a), fabsf(b))); }

int main(void) {
  // Test segmented sum
  float src[6] = {1,2,3, 4,5,6};
  float dst[2] = {0};
  sattn_rvv_segmented_sum_f32(src, dst, 2, 3);
  if (!(nearly_eq(dst[0], 6.f, 1e-6f) && nearly_eq(dst[1], 15.f, 1e-6f))) {
    printf("segmented_sum FAIL: dst0=%.3f dst1=%.3f\n", dst[0], dst[1]);
    return 1;
  }
  // Test softmax row on small vector
  float row[4] = {1.f, 2.f, 3.f, 4.f};
  sattn_rvv_softmax_row_f32(row, 4);
  float s = row[0]+row[1]+row[2]+row[3];
  if (!nearly_eq(s, 1.f, 1e-5f)) { printf("softmax sum FAIL: sum=%.6f\n", s); return 2; }
  if (!(row[3] > row[2] && row[2] > row[1] && row[1] > row[0])) { printf("softmax monotonic FAIL\n"); return 3; }
  printf("RVV helpers PASS\n");
  return 0;
}


