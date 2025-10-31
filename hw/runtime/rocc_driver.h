#ifndef SATTN_HW_RUNTIME_ROCC_DRIVER_H_
#define SATTN_HW_RUNTIME_ROCC_DRIVER_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  volatile uint64_t q_base;
  volatile uint64_t k_base;
  volatile uint64_t v_base;
  volatile uint64_t o_base;
  volatile uint64_t idx_base;
  volatile uint64_t stride_base;
  volatile uint32_t m_rows;
  volatile uint32_t head_dim_d;
  volatile uint32_t block_size;
  volatile uint32_t k_blocks;
  volatile uint32_t s_tokens;
  volatile uint32_t scale_fp_bits;
  volatile uint64_t cmd;
} sattn_rocc_regs_t;

// MMIO base maps to this register struct
static inline sattn_rocc_regs_t* sattn_rocc_map(uintptr_t mmio_base) {
  return (sattn_rocc_regs_t*)mmio_base;
}

// Issue a command by writing cmd byte, then poll done bit [0]
static inline int sattn_rocc_issue(sattn_rocc_regs_t* r, uint8_t cmd_id) {
  r->cmd = (uint64_t)cmd_id;
  // busy: bit1, done: bit0
  for (;;) {
    uint64_t st = r->cmd;
    if (st & 1ull) return 0;
  }
}

#ifdef __cplusplus
}
#endif

#endif  // SATTN_HW_RUNTIME_ROCC_DRIVER_H_


