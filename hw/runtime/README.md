# Runtime Stubs for RoCC Accelerator

This directory contains a minimal MMIO driver stub to interact with the `rocc_sattn.sv` skeleton.

- `rocc_driver.h`: maps the MMIO register block and provides `sattn_rocc_issue(...)` to write a command and poll for completion.

Usage sketch:
```c
#include "hw/runtime/rocc_driver.h"

int main() {
  uintptr_t base = 0x40000000; // platform-specific MMIO base
  sattn_rocc_regs_t* regs = sattn_rocc_map(base);
  regs->m_rows = 64;
  regs->head_dim_d = 64;
  regs->block_size = 64;
  regs->k_blocks = 8;
  regs->s_tokens = 512;
  regs->scale_fp_bits = 0x3f3504f3; // ~0.707
  // ... set base pointers as needed ...
  sattn_rocc_issue(regs, 0x14); // spdot_bsr
  return 0;
}
```

Hook this up to your SoC harness (e.g., Chipyard) by connecting the MMIO bus to the register map exposed by `rocc_sattn.sv`. 
