#ifndef SATTN_HW_SPEC_ROCC_INTRINSICS_H_
#define SATTN_HW_SPEC_ROCC_INTRINSICS_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Tentative RoCC command IDs
enum {
  SATTN_CMD_BLK_REDUCE   = 0x10,
  SATTN_CMD_TOPK_IDX     = 0x11,
  SATTN_CMD_GATH2D       = 0x12,
  SATTN_CMD_SCAT2D       = 0x13,
  SATTN_CMD_SPDOT_BSR    = 0x14,
  SATTN_CMD_SOFTMAX_FUSED= 0x15,
  SATTN_CMD_SPMM_BSR     = 0x16,
};

typedef struct {
  uint64_t q_base, k_base, v_base, o_base;
  uint64_t idx_base, stride_base;
  uint32_t m_rows, head_dim_d;
  uint32_t block_size, k_blocks, s_tokens;
  float scale_fp;
} sattn_cmd_desc_t;

// Intrinsics interface (software shim). On real hardware this would issue a RoCC
// command; here we define stubs to be linked against the simulator/driver.

int sattn_rocc_blk_reduce(const sattn_cmd_desc_t* desc, uint32_t mode);
int sattn_rocc_topk_idx(const sattn_cmd_desc_t* desc, uint32_t k, uint32_t flags);
int sattn_rocc_gath2d(const sattn_cmd_desc_t* desc);
int sattn_rocc_scat2d(const sattn_cmd_desc_t* desc);
int sattn_rocc_spdot_bsr(const sattn_cmd_desc_t* desc);
int sattn_rocc_softmax_fused(const sattn_cmd_desc_t* desc);
int sattn_rocc_spmm_bsr(const sattn_cmd_desc_t* desc);

#ifdef __cplusplus
}
#endif

#endif  // SATTN_HW_SPEC_ROCC_INTRINSICS_H_


