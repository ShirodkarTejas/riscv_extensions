#ifndef SATTN_HW_SPEC_SATTN_ISA_H_
#define SATTN_HW_SPEC_SATTN_ISA_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// RISC-V Custom Instruction Encoding for Sparse Attention
// ============================================================================

// Custom opcode (custom-1 space)
#define SATTN_OPCODE        0x2B  // 0101011

// funct3 encodings for primitives
#define SATTN_FUNCT3_BLK_REDUCE     0x0  // 000
#define SATTN_FUNCT3_TOPK_IDX       0x1  // 001
#define SATTN_FUNCT3_GATH2D         0x2  // 010
#define SATTN_FUNCT3_SCAT2D         0x3  // 011
#define SATTN_FUNCT3_SPDOT_BSR      0x4  // 100
#define SATTN_FUNCT3_SOFTMAX_FUSED  0x5  // 101
#define SATTN_FUNCT3_SPMM_BSR       0x6  // 110
#define SATTN_FUNCT3_RESERVED       0x7  // 111

// funct7 encodings (currently only 0x00 defined, rest reserved)
#define SATTN_FUNCT7_DEFAULT        0x00

// ============================================================================
// CSR Address Definitions
// ============================================================================

// Base pointers (64-bit)
#define CSR_SATTN_Q_BASE            0x7C0
#define CSR_SATTN_K_BASE            0x7C1
#define CSR_SATTN_V_BASE            0x7C2
#define CSR_SATTN_O_BASE            0x7C3
#define CSR_SATTN_IDX_BASE          0x7C4
#define CSR_SATTN_STRIDE_BASE       0x7C5

// Dimension parameters (32-bit)
#define CSR_SATTN_M_ROWS            0x7C6
#define CSR_SATTN_HEAD_DIM_D        0x7C7
#define CSR_SATTN_BLOCK_SIZE        0x7C8
#define CSR_SATTN_K_BLOCKS          0x7C9
#define CSR_SATTN_S_TOKENS          0x7CA
#define CSR_SATTN_SCALE_FP          0x7CB

// Extended parameters (32-bit)
#define CSR_SATTN_GQA_GROUP_SIZE    0x7CC
#define CSR_SATTN_COMP_BLOCK_SIZE   0x7CD

// Status and control (32-bit)
#define CSR_SATTN_STATUS            0x7CE
#define CSR_SATTN_ERROR             0x7CF

// Performance counters (64-bit, optional)
#define CSR_SATTN_CYCLE_COUNT       0x7D0
#define CSR_SATTN_MAC_OPS           0x7D1
#define CSR_SATTN_GATHER_CYCLES     0x7D2
#define CSR_SATTN_COMPUTE_CYCLES    0x7D3
#define CSR_SATTN_DMA_BYTES         0x7D4
#define CSR_SATTN_CACHE_HITS        0x7D5

// Hardware capabilities (32-bit, read-only)
#define CSR_SATTN_HW_VERSION        0x7D8
#define CSR_SATTN_HW_CAPS           0x7D9

// ============================================================================
// Status Register Bit Definitions
// ============================================================================

#define SATTN_STATUS_DONE           (1U << 0)
#define SATTN_STATUS_BUSY           (1U << 1)
#define SATTN_STATUS_READY          (1U << 2)
#define SATTN_STATUS_ERROR          (1U << 3)

// ============================================================================
// Error Codes
// ============================================================================

#define SATTN_ERROR_NONE            0x00
#define SATTN_ERROR_INVALID_DIM     0x01
#define SATTN_ERROR_MISALIGNED      0x02
#define SATTN_ERROR_UNSUPPORTED     0x03
#define SATTN_ERROR_OVERFLOW        0x04
#define SATTN_ERROR_TIMEOUT         0x05
#define SATTN_ERROR_MEMORY          0x06
#define SATTN_ERROR_UNKNOWN         0xFF

// ============================================================================
// Hardware Capability Bits
// ============================================================================

#define SATTN_CAP_BSR_SUPPORT       (1U << 0)
#define SATTN_CAP_SLIDING_WINDOW    (1U << 1)
#define SATTN_CAP_GQA_SUPPORT       (1U << 2)
#define SATTN_CAP_COMPRESSION       (1U << 3)
#define SATTN_CAP_INT8_QUANT        (1U << 4)
#define SATTN_CAP_INT4_QUANT        (1U << 5)
#define SATTN_CAP_BF16_PRECISION    (1U << 6)
#define SATTN_CAP_FP16_PRECISION    (1U << 7)

// ============================================================================
// CSR Access Macros
// ============================================================================

// Generic CSR read/write using RISC-V CSR instructions
#define csr_read(csr)                                           \
  ({                                                            \
    unsigned long __v;                                          \
    __asm__ __volatile__("csrr %0, %1" : "=r"(__v) : "i"(csr)); \
    __v;                                                        \
  })

#define csr_write(csr, val)                                     \
  ({                                                            \
    unsigned long __v = (unsigned long)(val);                   \
    __asm__ __volatile__("csrw %0, %1" : : "i"(csr), "r"(__v)); \
  })

#define csr_set(csr, val)                                       \
  ({                                                            \
    unsigned long __v = (unsigned long)(val);                   \
    __asm__ __volatile__("csrs %0, %1" : : "i"(csr), "r"(__v)); \
  })

#define csr_clear(csr, val)                                     \
  ({                                                            \
    unsigned long __v = (unsigned long)(val);                   \
    __asm__ __volatile__("csrc %0, %1" : : "i"(csr), "r"(__v)); \
  })

// ============================================================================
// Instruction Encoding Macros
// ============================================================================

// Encode R-type instruction
#define SATTN_ENCODE_R(funct7, rs2, rs1, funct3, rd, opcode)    \
  ((((funct7) & 0x7F) << 25) | (((rs2) & 0x1F) << 20) |         \
   (((rs1) & 0x1F) << 15) | (((funct3) & 0x7) << 12) |          \
   (((rd) & 0x1F) << 7) | ((opcode) & 0x7F))

// Pre-encoded instruction words (rd=x0, rs1=x0, rs2=x0, funct7=0x00)
#define SATTN_INSN_BLK_REDUCE       SATTN_ENCODE_R(SATTN_FUNCT7_DEFAULT, 0, 0, SATTN_FUNCT3_BLK_REDUCE, 0, SATTN_OPCODE)
#define SATTN_INSN_TOPK_IDX         SATTN_ENCODE_R(SATTN_FUNCT7_DEFAULT, 0, 0, SATTN_FUNCT3_TOPK_IDX, 0, SATTN_OPCODE)
#define SATTN_INSN_GATH2D           SATTN_ENCODE_R(SATTN_FUNCT7_DEFAULT, 0, 0, SATTN_FUNCT3_GATH2D, 0, SATTN_OPCODE)
#define SATTN_INSN_SCAT2D           SATTN_ENCODE_R(SATTN_FUNCT7_DEFAULT, 0, 0, SATTN_FUNCT3_SCAT2D, 0, SATTN_OPCODE)
#define SATTN_INSN_SPDOT_BSR        SATTN_ENCODE_R(SATTN_FUNCT7_DEFAULT, 0, 0, SATTN_FUNCT3_SPDOT_BSR, 0, SATTN_OPCODE)
#define SATTN_INSN_SOFTMAX_FUSED    SATTN_ENCODE_R(SATTN_FUNCT7_DEFAULT, 0, 0, SATTN_FUNCT3_SOFTMAX_FUSED, 0, SATTN_OPCODE)
#define SATTN_INSN_SPMM_BSR         SATTN_ENCODE_R(SATTN_FUNCT7_DEFAULT, 0, 0, SATTN_FUNCT3_SPMM_BSR, 0, SATTN_OPCODE)

// ============================================================================
// Assembly Instruction Macros (Inline ASM Helpers)
// ============================================================================

// Issue custom instruction via inline assembly
// These use .word directive to emit the instruction encoding directly

#define SATTN_ASM_BLK_REDUCE() \
  __asm__ __volatile__(".word %0" : : "i"(SATTN_INSN_BLK_REDUCE) : "memory")

#define SATTN_ASM_TOPK_IDX() \
  __asm__ __volatile__(".word %0" : : "i"(SATTN_INSN_TOPK_IDX) : "memory")

#define SATTN_ASM_GATH2D() \
  __asm__ __volatile__(".word %0" : : "i"(SATTN_INSN_GATH2D) : "memory")

#define SATTN_ASM_SCAT2D() \
  __asm__ __volatile__(".word %0" : : "i"(SATTN_INSN_SCAT2D) : "memory")

#define SATTN_ASM_SPDOT_BSR() \
  __asm__ __volatile__(".word %0" : : "i"(SATTN_INSN_SPDOT_BSR) : "memory")

#define SATTN_ASM_SOFTMAX_FUSED() \
  __asm__ __volatile__(".word %0" : : "i"(SATTN_INSN_SOFTMAX_FUSED) : "memory")

#define SATTN_ASM_SPMM_BSR() \
  __asm__ __volatile__(".word %0" : : "i"(SATTN_INSN_SPMM_BSR) : "memory")

// ============================================================================
// High-Level CSR Helper Functions
// ============================================================================

// Write 64-bit pointer to CSR (handles RV32/RV64 portability)
static inline void sattn_csr_write_ptr(unsigned int csr, uint64_t ptr) {
#if __riscv_xlen == 64
  csr_write(csr, ptr);
#else
  // RV32: write low word to csr, high word to csr+1
  csr_write(csr, (uint32_t)(ptr & 0xFFFFFFFFUL));
  csr_write(csr + 1, (uint32_t)((ptr >> 32) & 0xFFFFFFFFUL));
#endif
}

// Read 64-bit pointer from CSR
static inline uint64_t sattn_csr_read_ptr(unsigned int csr) {
#if __riscv_xlen == 64
  return csr_read(csr);
#else
  // RV32: read low and high words
  uint32_t low = csr_read(csr);
  uint32_t high = csr_read(csr + 1);
  return ((uint64_t)high << 32) | low;
#endif
}

// Write 32-bit value to CSR
static inline void sattn_csr_write_u32(unsigned int csr, uint32_t val) {
  csr_write(csr, val);
}

// Read 32-bit value from CSR
static inline uint32_t sattn_csr_read_u32(unsigned int csr) {
  return (uint32_t)csr_read(csr);
}

// Write float as bits to CSR
static inline void sattn_csr_write_float(unsigned int csr, float val) {
  union { float f; uint32_t u; } conv;
  conv.f = val;
  csr_write(csr, conv.u);
}

// Read float from CSR
static inline float sattn_csr_read_float(unsigned int csr) {
  union { float f; uint32_t u; } conv;
  conv.u = (uint32_t)csr_read(csr);
  return conv.f;
}

// ============================================================================
// Status Check Helpers
// ============================================================================

// Wait for operation completion (busy-wait)
static inline void sattn_wait_done(void) {
  while (!(sattn_csr_read_u32(CSR_SATTN_STATUS) & SATTN_STATUS_DONE)) {
    // Busy wait
  }
}

// Check if accelerator is ready
static inline int sattn_is_ready(void) {
  return (sattn_csr_read_u32(CSR_SATTN_STATUS) & SATTN_STATUS_READY) != 0;
}

// Check if operation is busy
static inline int sattn_is_busy(void) {
  return (sattn_csr_read_u32(CSR_SATTN_STATUS) & SATTN_STATUS_BUSY) != 0;
}

// Check for errors
static inline int sattn_has_error(void) {
  return (sattn_csr_read_u32(CSR_SATTN_STATUS) & SATTN_STATUS_ERROR) != 0;
}

// Get error code
static inline uint32_t sattn_get_error(void) {
  return sattn_csr_read_u32(CSR_SATTN_ERROR);
}

// Clear done flag
static inline void sattn_clear_done(void) {
  csr_set(CSR_SATTN_STATUS, SATTN_STATUS_DONE);
}

// ============================================================================
// Hardware Capability Query
// ============================================================================

// Get hardware version
static inline uint32_t sattn_get_hw_version(void) {
  return sattn_csr_read_u32(CSR_SATTN_HW_VERSION);
}

// Get hardware capabilities
static inline uint32_t sattn_get_hw_caps(void) {
  return sattn_csr_read_u32(CSR_SATTN_HW_CAPS);
}

// Check specific capability
static inline int sattn_has_capability(uint32_t cap_bit) {
  return (sattn_get_hw_caps() & cap_bit) != 0;
}

// ============================================================================
// Complete Parameter Setup Helper
// ============================================================================

// Setup all standard parameters from a descriptor-like struct
typedef struct {
  uint64_t q_base, k_base, v_base, o_base;
  uint64_t idx_base, stride_base;
  uint32_t m_rows, head_dim_d;
  uint32_t block_size, k_blocks, s_tokens;
  float scale_fp;
  uint32_t gqa_group_size;
  uint32_t comp_block_size;
} sattn_csr_params_t;

static inline void sattn_setup_csrs(const sattn_csr_params_t* params) {
  // Base pointers
  sattn_csr_write_ptr(CSR_SATTN_Q_BASE, params->q_base);
  sattn_csr_write_ptr(CSR_SATTN_K_BASE, params->k_base);
  sattn_csr_write_ptr(CSR_SATTN_V_BASE, params->v_base);
  sattn_csr_write_ptr(CSR_SATTN_O_BASE, params->o_base);
  sattn_csr_write_ptr(CSR_SATTN_IDX_BASE, params->idx_base);
  sattn_csr_write_ptr(CSR_SATTN_STRIDE_BASE, params->stride_base);
  
  // Dimensions
  sattn_csr_write_u32(CSR_SATTN_M_ROWS, params->m_rows);
  sattn_csr_write_u32(CSR_SATTN_HEAD_DIM_D, params->head_dim_d);
  sattn_csr_write_u32(CSR_SATTN_BLOCK_SIZE, params->block_size);
  sattn_csr_write_u32(CSR_SATTN_K_BLOCKS, params->k_blocks);
  sattn_csr_write_u32(CSR_SATTN_S_TOKENS, params->s_tokens);
  sattn_csr_write_float(CSR_SATTN_SCALE_FP, params->scale_fp);
  
  // Extended params
  sattn_csr_write_u32(CSR_SATTN_GQA_GROUP_SIZE, params->gqa_group_size);
  sattn_csr_write_u32(CSR_SATTN_COMP_BLOCK_SIZE, params->comp_block_size);
}

#ifdef __cplusplus
}
#endif

#endif  // SATTN_HW_SPEC_SATTN_ISA_H_

