#ifndef SATTN_BACKENDS_ROCC_INCLUDE_SPARSE_ATTENTION_ROCC_H_
#define SATTN_BACKENDS_ROCC_INCLUDE_SPARSE_ATTENTION_ROCC_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  uintptr_t mmio_base;
  void* regs;  // opaque mapped register block
} sattn_rocc_ctx_t;

typedef struct {
  uint64_t q_base, k_base, v_base, o_base;
  uint64_t idx_base, stride_base;
  uint32_t m_rows, head_dim_d;
  uint32_t block_size, k_blocks, s_tokens;
  float    scale_fp;
} sattn_rocc_desc_t;

int sattn_rocc_init(uintptr_t mmio_base, sattn_rocc_ctx_t* ctx);
int sattn_rocc_spdot_bsr(sattn_rocc_ctx_t* ctx, const sattn_rocc_desc_t* desc);

#ifdef __cplusplus
}
#endif

#endif  // SATTN_BACKENDS_ROCC_INCLUDE_SPARSE_ATTENTION_ROCC_H_


