# Assembly Programming Guide for Sparse Attention Instructions

This guide provides examples and best practices for programming the sparse attention custom instructions using RISC-V assembly, C intrinsics, and inline assembly.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Instruction Overview](#instruction-overview)
3. [Parameter Setup via CSRs](#parameter-setup-via-csrs)
4. [Assembly Examples](#assembly-examples)
5. [C Intrinsics Examples](#c-intrinsics-examples)
6. [Performance Tips](#performance-tips)
7. [Debugging and Validation](#debugging-and-validation)

## Prerequisites

### Toolchain Setup

To use custom instructions, you need:
- RISC-V GCC or LLVM with inline assembly support
- Target hardware or simulator with sparse attention extension
- Include files: `sattn_isa.h`, `rocc_intrinsics.h`

### Compilation Flags

```bash
# Enable CSR-based instruction mode
gcc -DSATTN_USE_CSR_INSTRUCTIONS -march=rv64gc -o program program.c

# For inline assembly with .word directive (when assembler doesn't know instruction)
gcc -march=rv64gc -o program program.c
```

## Instruction Overview

All sparse attention instructions use the R-type format:

```
Instruction: sattn.<primitive> rd, rs1, rs2
Opcode: 0x2B (custom-1)
```

| Primitive       | funct3 | Description                                  |
|-----------------|--------|----------------------------------------------|
| blk_reduce      | 0x0    | Block reduction (mean/max)                   |
| topk_idx        | 0x1    | Top-k block index selection                  |
| gath2d          | 0x2    | 2D gather operation                          |
| scat2d          | 0x3    | 2D scatter operation                         |
| spdot_bsr       | 0x4    | Sparse block dot product (Q × K^T)           |
| softmax_fused   | 0x5    | Fused softmax operation                      |
| spmm_bsr        | 0x6    | Sparse block matrix multiply (Attention × V) |

## Parameter Setup via CSRs

Before issuing any instruction, parameters must be written to CSRs.

### CSR Address Map

```
Base Pointers (64-bit):
  0x7C0: Q_BASE       - Query tensor base address
  0x7C1: K_BASE       - Key tensor base address
  0x7C2: V_BASE       - Value tensor base address
  0x7C3: O_BASE       - Output tensor base address
  0x7C4: IDX_BASE     - Index table base address
  0x7C5: STRIDE_BASE  - Stride table base address

Dimensions (32-bit):
  0x7C6: M_ROWS       - Number of query rows
  0x7C7: HEAD_DIM_D   - Head dimension
  0x7C8: BLOCK_SIZE   - Block size (tokens per block)
  0x7C9: K_BLOCKS     - Number of blocks
  0x7CA: S_TOKENS     - Total selected tokens
  0x7CB: SCALE_FP     - Attention scale factor (float bits)

Status:
  0x7CE: STATUS       - Status/control register
  0x7CF: ERROR        - Error code
```

## Assembly Examples

### Example 1: Sparse Dot Product (spdot_bsr)

```assembly
.section .text
.globl sparse_attention_spdot

sparse_attention_spdot:
    # Input: a0 = Q pointer, a1 = K pointer, a2 = idx pointer
    #        a3 = m_rows, a4 = head_dim, a5 = s_tokens
    
    # Write base pointers to CSRs
    csrw 0x7C0, a0           # Q_BASE
    csrw 0x7C1, a1           # K_BASE
    csrw 0x7C4, a2           # IDX_BASE
    
    # Write dimensions
    csrw 0x7C6, a3           # M_ROWS
    csrw 0x7C7, a4           # HEAD_DIM_D
    csrw 0x7CA, a5           # S_TOKENS
    
    # Set block size (e.g., 64)
    li   t0, 64
    csrw 0x7C8, t0           # BLOCK_SIZE
    
    # Set scale factor (1/sqrt(64) ≈ 0.125 as float bits)
    li   t0, 0x3E000000      # 0.125f in IEEE 754
    csrw 0x7CB, t0           # SCALE_FP
    
    # Issue spdot_bsr instruction (funct3=100, rd=x0, rs1=x0, rs2=x0)
    .word 0x0000402B         # sattn.spdot_bsr x0, x0, x0
    
    # Poll for completion
poll_loop:
    csrr t0, 0x7CE           # Read STATUS
    andi t0, t0, 0x1         # Check DONE bit
    beqz t0, poll_loop
    
    # Check for errors
    csrr t0, 0x7CF           # Read ERROR
    bnez t0, error_handler
    
    # Success
    li   a0, 0
    ret

error_handler:
    # Return error code in a0
    mv   a0, t0
    ret
```

### Example 2: Fused Softmax

```assembly
.globl softmax_fused_asm

softmax_fused_asm:
    # Assume CSRs already set up for score matrix
    # Input: a0 = scores pointer, a1 = m_rows, a2 = s_tokens
    
    csrw 0x7C0, a0           # Q_BASE (used for scores)
    csrw 0x7C6, a1           # M_ROWS
    csrw 0x7CA, a2           # S_TOKENS
    
    # Issue softmax_fused instruction (funct3=101)
    .word 0x0000502B         # sattn.softmax_fused x0, x0, x0
    
    # Wait for completion
1:
    csrr t0, 0x7CE
    andi t0, t0, 0x1
    beqz t0, 1b
    
    ret
```

### Example 3: Complete Attention Pipeline

```assembly
.globl sparse_attention_forward

sparse_attention_forward:
    # Full sparse attention: Q×K^T → softmax → Attn×V
    # Inputs in a0-a6: pointers and dimensions
    
    # Save return address
    addi sp, sp, -16
    sd   ra, 0(sp)
    
    # Stage 1: Setup common CSRs
    csrw 0x7C0, a0           # Q_BASE
    csrw 0x7C1, a1           # K_BASE
    csrw 0x7C2, a2           # V_BASE
    csrw 0x7C3, a3           # O_BASE
    csrw 0x7C4, a4           # IDX_BASE
    csrw 0x7C6, a5           # M_ROWS
    csrw 0x7C7, a6           # HEAD_DIM_D
    # ... (additional dimension setup)
    
    # Stage 2: Sparse dot product (Q × K^T)
    .word 0x0000402B         # sattn.spdot_bsr
    jal  wait_done
    
    # Stage 3: Fused softmax
    .word 0x0000502B         # sattn.softmax_fused
    jal  wait_done
    
    # Stage 4: Sparse matrix multiply (Attn × V)
    .word 0x0000602B         # sattn.spmm_bsr
    jal  wait_done
    
    # Restore and return
    ld   ra, 0(sp)
    addi sp, sp, 16
    li   a0, 0
    ret

wait_done:
    csrr t0, 0x7CE
    andi t0, t0, 0x1
    beqz t0, wait_done
    ret
```

## C Intrinsics Examples

### Example 1: Using High-Level API

```c
#include "hw/spec/rocc_intrinsics.h"
#include <stdint.h>
#include <math.h>

int sparse_attention_c_api(
    float* Q, float* K, float* V, float* O,
    uint16_t* indices, int m_rows, int head_dim, int s_tokens)
{
    // Create descriptor
    sattn_cmd_desc_t desc = {
        .q_base = (uint64_t)Q,
        .k_base = (uint64_t)K,
        .v_base = (uint64_t)V,
        .o_base = (uint64_t)O,
        .idx_base = (uint64_t)indices,
        .stride_base = 0,
        .m_rows = m_rows,
        .head_dim_d = head_dim,
        .block_size = 64,
        .k_blocks = s_tokens / 64,
        .s_tokens = s_tokens,
        .scale_fp = 1.0f / sqrtf((float)head_dim)
    };
    
    // Issue operations (automatically uses CSR or MMIO based on compile flag)
    int ret;
    
    ret = sattn_spdot_bsr(&desc);
    if (ret != 0) return ret;
    
    ret = sattn_softmax_fused(&desc);
    if (ret != 0) return ret;
    
    ret = sattn_spmm_bsr(&desc);
    return ret;
}
```

### Example 2: Using Low-Level CSR API

```c
#include "hw/spec/sattn_isa.h"

int sparse_attention_low_level(
    float* Q, float* K, float* V, float* O,
    uint16_t* indices, int m_rows, int head_dim)
{
    // Manually write CSRs
    sattn_csr_write_ptr(CSR_SATTN_Q_BASE, (uint64_t)Q);
    sattn_csr_write_ptr(CSR_SATTN_K_BASE, (uint64_t)K);
    sattn_csr_write_ptr(CSR_SATTN_V_BASE, (uint64_t)V);
    sattn_csr_write_ptr(CSR_SATTN_O_BASE, (uint64_t)O);
    sattn_csr_write_ptr(CSR_SATTN_IDX_BASE, (uint64_t)indices);
    
    sattn_csr_write_u32(CSR_SATTN_M_ROWS, m_rows);
    sattn_csr_write_u32(CSR_SATTN_HEAD_DIM_D, head_dim);
    sattn_csr_write_u32(CSR_SATTN_BLOCK_SIZE, 64);
    sattn_csr_write_u32(CSR_SATTN_S_TOKENS, 1024);
    sattn_csr_write_float(CSR_SATTN_SCALE_FP, 0.125f);
    
    // Issue instruction via inline assembly
    SATTN_ASM_SPDOT_BSR();
    
    // Wait for completion
    sattn_wait_done();
    
    // Check errors
    if (sattn_has_error()) {
        return -1;
    }
    
    return 0;
}
```

### Example 3: Inline Assembly in C

```c
#include <stdint.h>

void issue_spdot_inline(void) {
    // Issue instruction using .word directive
    __asm__ __volatile__(
        ".word 0x0000402B"   // sattn.spdot_bsr x0, x0, x0
        : /* no outputs */
        : /* no inputs */
        : "memory"           // Clobber memory to prevent reordering
    );
}

uint32_t read_status_inline(void) {
    uint32_t status;
    __asm__ __volatile__(
        "csrr %0, 0x7CE"     // Read STATUS CSR
        : "=r"(status)
        :
        :
    );
    return status;
}
```

## Performance Tips

### 1. CSR Write Scheduling

CSR writes can be scheduled independently. Overlap them with computation when possible:

```assembly
    # Good: schedule CSR writes together
    csrw 0x7C0, a0
    csrw 0x7C1, a1
    csrw 0x7C2, a2
    csrw 0x7C6, a3
    csrw 0x7C7, a4
    # Then issue instruction
    .word 0x0000402B
```

### 2. Reuse CSR Values

If parameters don't change between operations, don't rewrite CSRs:

```c
// Setup CSRs once
sattn_csr_write_ptr(CSR_SATTN_Q_BASE, Q);
sattn_csr_write_u32(CSR_SATTN_HEAD_DIM_D, 64);
// ... other static params

for (int i = 0; i < num_heads; i++) {
    // Only update changing parameters
    sattn_csr_write_ptr(CSR_SATTN_K_BASE, K + i * stride);
    SATTN_ASM_SPDOT_BSR();
    sattn_wait_done();
}
```

### 3. Minimize Polling Overhead

Instead of tight polling loops, use fences or yield:

```c
// Tight loop (high power consumption)
while (!(sattn_csr_read_u32(CSR_SATTN_STATUS) & 0x1));

// Better: with pause hint (if supported)
while (!(sattn_csr_read_u32(CSR_SATTN_STATUS) & 0x1)) {
    __asm__ __volatile__("pause");  // or custom yield instruction
}
```

### 4. Batch Operations

Group multiple small operations to amortize setup overhead:

```c
// Bad: many small operations with full setup each time
for (int i = 0; i < n; i++) {
    setup_all_csrs();
    issue_instruction();
    wait();
}

// Good: batch into larger tiles
setup_common_csrs();
for (int i = 0; i < n; i += batch_size) {
    update_dynamic_csrs();
    issue_instruction();
    wait();
}
```

## Debugging and Validation

### Check Hardware Capabilities

```c
#include "hw/spec/sattn_isa.h"

void check_hardware(void) {
    uint32_t version = sattn_get_hw_version();
    uint32_t caps = sattn_get_hw_caps();
    
    printf("HW Version: %d.%d.%d\n", 
           (version >> 24) & 0xFF,
           (version >> 16) & 0xFF,
           version & 0xFFFF);
    
    if (caps & SATTN_CAP_BSR_SUPPORT) {
        printf("BSR format supported\n");
    }
    if (caps & SATTN_CAP_BF16_PRECISION) {
        printf("BF16 precision supported\n");
    }
}
```

### Error Handling

```c
int safe_spdot(const sattn_cmd_desc_t* desc) {
    // Check if ready
    if (!sattn_is_ready()) {
        fprintf(stderr, "Accelerator not ready\n");
        return -1;
    }
    
    // Issue operation
    sattn_csr_spdot_bsr(desc);
    
    // Check for errors
    if (sattn_has_error()) {
        uint32_t error = sattn_get_error();
        switch (error) {
            case SATTN_ERROR_INVALID_DIM:
                fprintf(stderr, "Invalid dimensions\n");
                break;
            case SATTN_ERROR_MISALIGNED:
                fprintf(stderr, "Misaligned pointer\n");
                break;
            case SATTN_ERROR_OVERFLOW:
                fprintf(stderr, "Scratchpad overflow\n");
                break;
            default:
                fprintf(stderr, "Unknown error: 0x%x\n", error);
        }
        return -1;
    }
    
    return 0;
}
```

### Assembly Verification

To verify instruction encoding:

```bash
# Compile to assembly
riscv64-unknown-elf-gcc -S -march=rv64gc program.c -o program.s

# Check encoding
riscv64-unknown-elf-objdump -d program.o | grep -A5 "sattn"

# Expected output:
# 0000402b    .word 0x0000402b  # sattn.spdot_bsr
```

## Comparison: MMIO vs CSR-Instruction Interface

| Aspect              | MMIO Interface         | CSR-Instruction Interface |
|---------------------|------------------------|---------------------------|
| Setup overhead      | Higher (memory access) | Lower (register access)   |
| Parameter passing   | Struct write to memory | CSR writes                |
| Instruction issue   | MMIO CMD register      | Custom instruction        |
| Portability         | Simulator-friendly     | Requires HW support       |
| Performance         | Slower (memory latency)| Faster (register access)  |
| Use case            | Simulation, testing    | Hardware accelerator      |

## Summary

- **CSR-based interface**: Fast, low-overhead, requires custom instruction support
- **MMIO interface**: Portable, simulator-friendly, higher latency
- **Unified API**: Use compile-time flag to switch between modes transparently
- **Best practice**: Use high-level C intrinsics unless manual optimization is needed

For more details, see:
- `instruction_encoding.md` - Instruction format specification
- `csr_map.md` - CSR address map and semantics
- `sattn_isa.h` - C header with macros and intrinsics
- `rocc_intrinsics.h` - High-level API

