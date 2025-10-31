#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "backends/rocc/include/sparse_attention_rocc.h"

int main(void) {
  sattn_rocc_ctx_t ctx;
  if (sattn_rocc_init(0x40000000, &ctx) != 0) {
    fprintf(stderr, "init failed\n");
    return 1;
  }

  sattn_rocc_desc_t d = {0};
  d.m_rows = 64; d.head_dim_d = 64; d.block_size = 64; d.k_blocks = 8; d.s_tokens = 512; d.scale_fp = 1.0f/8.0f;
  if (sattn_rocc_spdot_bsr(&ctx, &d) != 0) {
    fprintf(stderr, "issue spdot_bsr failed\n");
    return 2;
  }
  printf("issued spdot_bsr via MMIO\n");
  return 0;
}


