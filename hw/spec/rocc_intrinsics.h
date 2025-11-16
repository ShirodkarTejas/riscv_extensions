#ifndef SATTN_HW_SPEC_ROCC_INTRINSICS_H_
#define SATTN_HW_SPEC_ROCC_INTRINSICS_H_

#include <stdint.h>
#include "sattn_isa.h"  // Include CSR and instruction definitions

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Legacy Command IDs (for MMIO interface compatibility)
// ============================================================================

enum {
  SATTN_CMD_BLK_REDUCE   = 0x10,
  SATTN_CMD_TOPK_IDX     = 0x11,
  SATTN_CMD_GATH2D       = 0x12,
  SATTN_CMD_SCAT2D       = 0x13,
  SATTN_CMD_SPDOT_BSR    = 0x14,
  SATTN_CMD_SOFTMAX_FUSED= 0x15,
  SATTN_CMD_SPMM_BSR     = 0x16,
};

// Legacy descriptor structure (used by both MMIO and CSR-based interfaces)
typedef struct {
  uint64_t q_base, k_base, v_base, o_base;
  uint64_t idx_base, stride_base;
  uint32_t m_rows, head_dim_d;
  uint32_t block_size, k_blocks, s_tokens;
  float scale_fp;
} sattn_cmd_desc_t;

// ============================================================================
// CSR-Based Intrinsics (new instruction interface)
// ============================================================================

// High-level intrinsics: setup CSRs from descriptor + issue instruction
static inline int sattn_csr_blk_reduce(const sattn_cmd_desc_t* desc, uint32_t mode) {
  sattn_csr_params_t params = {
    .q_base = desc->q_base,
    .k_base = desc->k_base,
    .v_base = desc->v_base,
    .o_base = desc->o_base,
    .idx_base = desc->idx_base,
    .stride_base = desc->stride_base,
    .m_rows = desc->m_rows,
    .head_dim_d = desc->head_dim_d,
    .block_size = desc->block_size,
    .k_blocks = desc->k_blocks,
    .s_tokens = desc->s_tokens,
    .scale_fp = desc->scale_fp,
    .gqa_group_size = 1,
    .comp_block_size = 0
  };
  sattn_setup_csrs(&params);
  SATTN_ASM_BLK_REDUCE();
  sattn_wait_done();
  return sattn_has_error() ? -1 : 0;
}

static inline int sattn_csr_topk_idx(const sattn_cmd_desc_t* desc, uint32_t k, uint32_t flags) {
  sattn_csr_params_t params = {
    .q_base = desc->q_base,
    .k_base = desc->k_base,
    .v_base = desc->v_base,
    .o_base = desc->o_base,
    .idx_base = desc->idx_base,
    .stride_base = desc->stride_base,
    .m_rows = desc->m_rows,
    .head_dim_d = desc->head_dim_d,
    .block_size = desc->block_size,
    .k_blocks = k,  // Use k parameter for k_blocks
    .s_tokens = desc->s_tokens,
    .scale_fp = desc->scale_fp,
    .gqa_group_size = 1,
    .comp_block_size = 0
  };
  sattn_setup_csrs(&params);
  SATTN_ASM_TOPK_IDX();
  sattn_wait_done();
  return sattn_has_error() ? -1 : 0;
}

static inline int sattn_csr_gath2d(const sattn_cmd_desc_t* desc) {
  sattn_csr_params_t params = {
    .q_base = desc->q_base,
    .k_base = desc->k_base,
    .v_base = desc->v_base,
    .o_base = desc->o_base,
    .idx_base = desc->idx_base,
    .stride_base = desc->stride_base,
    .m_rows = desc->m_rows,
    .head_dim_d = desc->head_dim_d,
    .block_size = desc->block_size,
    .k_blocks = desc->k_blocks,
    .s_tokens = desc->s_tokens,
    .scale_fp = desc->scale_fp,
    .gqa_group_size = 1,
    .comp_block_size = 0
  };
  sattn_setup_csrs(&params);
  SATTN_ASM_GATH2D();
  sattn_wait_done();
  return sattn_has_error() ? -1 : 0;
}

static inline int sattn_csr_scat2d(const sattn_cmd_desc_t* desc) {
  sattn_csr_params_t params = {
    .q_base = desc->q_base,
    .k_base = desc->k_base,
    .v_base = desc->v_base,
    .o_base = desc->o_base,
    .idx_base = desc->idx_base,
    .stride_base = desc->stride_base,
    .m_rows = desc->m_rows,
    .head_dim_d = desc->head_dim_d,
    .block_size = desc->block_size,
    .k_blocks = desc->k_blocks,
    .s_tokens = desc->s_tokens,
    .scale_fp = desc->scale_fp,
    .gqa_group_size = 1,
    .comp_block_size = 0
  };
  sattn_setup_csrs(&params);
  SATTN_ASM_SCAT2D();
  sattn_wait_done();
  return sattn_has_error() ? -1 : 0;
}

static inline int sattn_csr_spdot_bsr(const sattn_cmd_desc_t* desc) {
  sattn_csr_params_t params = {
    .q_base = desc->q_base,
    .k_base = desc->k_base,
    .v_base = desc->v_base,
    .o_base = desc->o_base,
    .idx_base = desc->idx_base,
    .stride_base = desc->stride_base,
    .m_rows = desc->m_rows,
    .head_dim_d = desc->head_dim_d,
    .block_size = desc->block_size,
    .k_blocks = desc->k_blocks,
    .s_tokens = desc->s_tokens,
    .scale_fp = desc->scale_fp,
    .gqa_group_size = 1,
    .comp_block_size = 0
  };
  sattn_setup_csrs(&params);
  SATTN_ASM_SPDOT_BSR();
  sattn_wait_done();
  return sattn_has_error() ? -1 : 0;
}

static inline int sattn_csr_softmax_fused(const sattn_cmd_desc_t* desc) {
  sattn_csr_params_t params = {
    .q_base = desc->q_base,
    .k_base = desc->k_base,
    .v_base = desc->v_base,
    .o_base = desc->o_base,
    .idx_base = desc->idx_base,
    .stride_base = desc->stride_base,
    .m_rows = desc->m_rows,
    .head_dim_d = desc->head_dim_d,
    .block_size = desc->block_size,
    .k_blocks = desc->k_blocks,
    .s_tokens = desc->s_tokens,
    .scale_fp = desc->scale_fp,
    .gqa_group_size = 1,
    .comp_block_size = 0
  };
  sattn_setup_csrs(&params);
  SATTN_ASM_SOFTMAX_FUSED();
  sattn_wait_done();
  return sattn_has_error() ? -1 : 0;
}

static inline int sattn_csr_spmm_bsr(const sattn_cmd_desc_t* desc) {
  sattn_csr_params_t params = {
    .q_base = desc->q_base,
    .k_base = desc->k_base,
    .v_base = desc->v_base,
    .o_base = desc->o_base,
    .idx_base = desc->idx_base,
    .stride_base = desc->stride_base,
    .m_rows = desc->m_rows,
    .head_dim_d = desc->head_dim_d,
    .block_size = desc->block_size,
    .k_blocks = desc->k_blocks,
    .s_tokens = desc->s_tokens,
    .scale_fp = desc->scale_fp,
    .gqa_group_size = 1,
    .comp_block_size = 0
  };
  sattn_setup_csrs(&params);
  SATTN_ASM_SPMM_BSR();
  sattn_wait_done();
  return sattn_has_error() ? -1 : 0;
}

// ============================================================================
// Legacy MMIO/Simulator Intrinsics (for backward compatibility)
// ============================================================================

// These stubs are linked against the simulator/driver for environments
// that don't support custom instructions (e.g., QEMU without extensions)

int sattn_rocc_blk_reduce(const sattn_cmd_desc_t* desc, uint32_t mode);
int sattn_rocc_topk_idx(const sattn_cmd_desc_t* desc, uint32_t k, uint32_t flags);
int sattn_rocc_gath2d(const sattn_cmd_desc_t* desc);
int sattn_rocc_scat2d(const sattn_cmd_desc_t* desc);
int sattn_rocc_spdot_bsr(const sattn_cmd_desc_t* desc);
int sattn_rocc_softmax_fused(const sattn_cmd_desc_t* desc);
int sattn_rocc_spmm_bsr(const sattn_cmd_desc_t* desc);

// ============================================================================
// Unified API (auto-selects CSR-based or MMIO based on compile-time flag)
// ============================================================================

#ifdef SATTN_USE_CSR_INSTRUCTIONS
  // Use new CSR-based instruction interface
  #define sattn_blk_reduce(desc, mode)    sattn_csr_blk_reduce(desc, mode)
  #define sattn_topk_idx(desc, k, flags)  sattn_csr_topk_idx(desc, k, flags)
  #define sattn_gath2d(desc)              sattn_csr_gath2d(desc)
  #define sattn_scat2d(desc)              sattn_csr_scat2d(desc)
  #define sattn_spdot_bsr(desc)           sattn_csr_spdot_bsr(desc)
  #define sattn_softmax_fused(desc)       sattn_csr_softmax_fused(desc)
  #define sattn_spmm_bsr(desc)            sattn_csr_spmm_bsr(desc)
#else
  // Use legacy MMIO/simulator interface
  #define sattn_blk_reduce(desc, mode)    sattn_rocc_blk_reduce(desc, mode)
  #define sattn_topk_idx(desc, k, flags)  sattn_rocc_topk_idx(desc, k, flags)
  #define sattn_gath2d(desc)              sattn_rocc_gath2d(desc)
  #define sattn_scat2d(desc)              sattn_rocc_scat2d(desc)
  #define sattn_spdot_bsr(desc)           sattn_rocc_spdot_bsr(desc)
  #define sattn_softmax_fused(desc)       sattn_rocc_softmax_fused(desc)
  #define sattn_spmm_bsr(desc)            sattn_rocc_spmm_bsr(desc)
#endif

#ifdef __cplusplus
}
#endif

#endif  // SATTN_HW_SPEC_ROCC_INTRINSICS_H_


