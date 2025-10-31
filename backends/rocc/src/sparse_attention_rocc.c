#include "backends/rocc/include/sparse_attention_rocc.h"
#include "hw/runtime/rocc_driver.h"

#include <string.h>

int sattn_rocc_init(uintptr_t mmio_base, sattn_rocc_ctx_t* ctx) {
  if (!ctx) return -1;
  ctx->mmio_base = mmio_base;
  ctx->regs = (void*)sattn_rocc_map(mmio_base);
  return 0;
}

int sattn_rocc_spdot_bsr(sattn_rocc_ctx_t* ctx, const sattn_rocc_desc_t* desc) {
  if (!ctx || !desc) return -1;
  sattn_rocc_regs_t* r = (sattn_rocc_regs_t*)ctx->regs;
  r->q_base = desc->q_base;
  r->k_base = desc->k_base;
  r->v_base = desc->v_base;
  r->o_base = desc->o_base;
  r->idx_base = desc->idx_base;
  r->stride_base = desc->stride_base;
  r->m_rows = desc->m_rows;
  r->head_dim_d = desc->head_dim_d;
  r->block_size = desc->block_size;
  r->k_blocks = desc->k_blocks;
  r->s_tokens = desc->s_tokens;
  // raw float bits for scale
  union { float f; uint32_t u; } sb; sb.f = desc->scale_fp; r->scale_fp_bits = sb.u;
  return sattn_rocc_issue(r, 0x14); // CMD_SPDOT_BSR
}


