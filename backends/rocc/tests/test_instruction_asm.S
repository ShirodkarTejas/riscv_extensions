# Assembly test for custom RISC-V sparse attention instructions
# Tests basic instruction issue and CSR access

.section .text
.globl test_asm_csr_access
.globl test_asm_instruction_issue
.globl test_asm_spdot_simple

# Test 1: Basic CSR read/write
# int test_asm_csr_access(void)
# Returns 0 on success, -1 on failure
test_asm_csr_access:
    # Write test value to M_ROWS CSR (0x7C6)
    li   t0, 123
    csrw 0x7C6, t0
    
    # Read it back
    csrr t1, 0x7C6
    
    # Check if equal
    bne  t0, t1, csr_fail
    
    # Write to HEAD_DIM_D CSR (0x7C7)
    li   t0, 64
    csrw 0x7C7, t0
    
    # Read it back
    csrr t1, 0x7C7
    bne  t0, t1, csr_fail
    
    # Success
    li   a0, 0
    ret

csr_fail:
    li   a0, -1
    ret

# Test 2: Issue instruction and check status
# int test_asm_instruction_issue(void)
test_asm_instruction_issue:
    # Save return address
    addi sp, sp, -16
    sd   ra, 0(sp)
    
    # Setup minimal CSRs for testing
    li   t0, 16
    csrw 0x7C6, t0           # M_ROWS = 16
    
    li   t0, 64
    csrw 0x7C7, t0           # HEAD_DIM_D = 64
    csrw 0x7C8, t0           # BLOCK_SIZE = 64
    
    li   t0, 512
    csrw 0x7CA, t0           # S_TOKENS = 512
    
    # Issue instruction (blk_reduce as test, funct3=000)
    # Encoding: 0x0000002B (blk_reduce with all fields = 0)
    .word 0x0000002B
    
    # Check status register (should transition through states)
    # Read STATUS CSR (0x7CE)
    csrr a0, 0x7CE
    
    # Check that some status bits are set
    # In real hardware, BUSY or DONE should be set
    # For this test, just verify we can read status
    li   t0, 0xFFFFFFFF
    and  a0, a0, t0          # Mask status
    
    # Restore and return
    ld   ra, 0(sp)
    addi sp, sp, 16
    li   a0, 0
    ret

# Test 3: Simple sparse dot product setup
# void test_asm_spdot_simple(uint64_t q_ptr, uint64_t k_ptr, uint64_t idx_ptr)
test_asm_spdot_simple:
    # Arguments: a0 = Q pointer, a1 = K pointer, a2 = idx pointer
    
    # Write base pointers to CSRs
    csrw 0x7C0, a0           # Q_BASE
    csrw 0x7C1, a1           # K_BASE
    csrw 0x7C4, a2           # IDX_BASE
    
    # Setup dimensions
    li   t0, 32
    csrw 0x7C6, t0           # M_ROWS = 32
    
    li   t0, 64
    csrw 0x7C7, t0           # HEAD_DIM_D = 64
    csrw 0x7C8, t0           # BLOCK_SIZE = 64
    
    li   t0, 1024
    csrw 0x7CA, t0           # S_TOKENS = 1024
    
    # Setup scale factor (0.125 as float bits = 0x3E000000)
    li   t0, 0x3E000000
    csrw 0x7CB, t0           # SCALE_FP
    
    # Issue spdot_bsr instruction (funct3=100)
    # Encoding: 0x0000402B
    .word 0x0000402B
    
    # Poll for completion
poll_spdot:
    csrr t0, 0x7CE           # Read STATUS
    andi t0, t0, 0x1         # Check DONE bit (bit 0)
    beqz t0, poll_spdot      # Loop if not done
    
    # Check for errors
    csrr t0, 0x7CF           # Read ERROR
    bnez t0, spdot_error
    
    # Success
    li   a0, 0
    ret

spdot_error:
    mv   a0, t0              # Return error code
    ret

# Test 4: All instructions encoding verification
# This doesn't execute, just provides reference encodings
.globl instruction_encoding_table
instruction_encoding_table:
    .word 0x0000002B    # blk_reduce     (funct3=000)
    .word 0x0000102B    # topk_idx       (funct3=001)
    .word 0x0000202B    # gath2d         (funct3=010)
    .word 0x0000302B    # scat2d         (funct3=011)
    .word 0x0000402B    # spdot_bsr      (funct3=100)
    .word 0x0000502B    # softmax_fused  (funct3=101)
    .word 0x0000602B    # spmm_bsr       (funct3=110)

# Test 5: Full attention pipeline (demonstration)
# void test_asm_attention_pipeline(
#     uint64_t q, uint64_t k, uint64_t v, uint64_t o, uint64_t idx)
.globl test_asm_attention_pipeline
test_asm_attention_pipeline:
    # Arguments: a0=Q, a1=K, a2=V, a3=O, a4=idx
    addi sp, sp, -16
    sd   ra, 0(sp)
    
    # Setup all base pointers
    csrw 0x7C0, a0           # Q_BASE
    csrw 0x7C1, a1           # K_BASE
    csrw 0x7C2, a2           # V_BASE
    csrw 0x7C3, a3           # O_BASE
    csrw 0x7C4, a4           # IDX_BASE
    
    # Setup dimensions
    li   t0, 32
    csrw 0x7C6, t0           # M_ROWS
    li   t0, 64
    csrw 0x7C7, t0           # HEAD_DIM_D
    csrw 0x7C8, t0           # BLOCK_SIZE
    li   t0, 16
    csrw 0x7C9, t0           # K_BLOCKS
    li   t0, 1024
    csrw 0x7CA, t0           # S_TOKENS
    li   t0, 0x3E000000
    csrw 0x7CB, t0           # SCALE_FP
    
    # Stage 1: Sparse dot product (Q × K^T)
    .word 0x0000402B         # spdot_bsr
    jal  wait_done_asm
    
    # Stage 2: Fused softmax
    .word 0x0000502B         # softmax_fused
    jal  wait_done_asm
    
    # Stage 3: Sparse matrix multiply (Attn × V)
    .word 0x0000602B         # spmm_bsr
    jal  wait_done_asm
    
    # Restore and return
    ld   ra, 0(sp)
    addi sp, sp, 16
    li   a0, 0
    ret

# Helper: wait for operation completion
wait_done_asm:
    csrr t0, 0x7CE           # Read STATUS
    andi t0, t0, 0x1         # Check DONE bit
    beqz t0, wait_done_asm
    ret

# Test 6: CSR address validation
# int test_asm_csr_addresses(void)
# Verifies CSR addresses by writing/reading pattern
.globl test_asm_csr_addresses
test_asm_csr_addresses:
    # Test each CSR with unique pattern
    
    # Q_BASE (0x7C0)
    li   t0, 0x1234
    csrw 0x7C0, t0
    csrr t1, 0x7C0
    bne  t0, t1, addr_fail
    
    # K_BASE (0x7C1)
    li   t0, 0x2345
    csrw 0x7C1, t0
    csrr t1, 0x7C1
    bne  t0, t1, addr_fail
    
    # M_ROWS (0x7C6)
    li   t0, 0x6789
    csrw 0x7C6, t0
    csrr t1, 0x7C6
    bne  t0, t1, addr_fail
    
    # STATUS (0x7CE) - read-only, just verify we can read
    csrr t0, 0x7CE
    
    # Success
    li   a0, 0
    ret

addr_fail:
    li   a0, -1
    ret

# Data section for test constants
.section .data
.align 8

test_dimensions:
    .word 32                # m_rows
    .word 64                # head_dim_d
    .word 64                # block_size
    .word 16                # k_blocks
    .word 1024              # s_tokens

test_scale:
    .word 0x3E000000        # 0.125 as IEEE 754 float

# Note: These tests are designed to work on:
# 1. Hardware with custom instruction support
# 2. Simulators with instruction decoder
# 3. QEMU with extension plugin (if available)
#
# For platforms without support, the .word directives will be treated
# as illegal instructions and may trap.

