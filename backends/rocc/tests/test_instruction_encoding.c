// Test suite for custom RISC-V instruction encoding and CSR interface
// This tests the instruction-based programming model

#include "hw/spec/sattn_isa.h"
#include "hw/spec/rocc_intrinsics.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

// Test result tracking
static int tests_passed = 0;
static int tests_failed = 0;

#define TEST_ASSERT(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL: %s\n", msg); \
        tests_failed++; \
        return -1; \
    } else { \
        tests_passed++; \
    } \
} while(0)

#define TEST_PASS(name) do { \
    printf("PASS: %s\n", name); \
    return 0; \
} while(0)

// Test 1: CSR Read/Write functionality
int test_csr_read_write(void) {
    printf("Testing CSR read/write...\n");
    
    // Test 32-bit CSR write/read
    uint32_t test_val_32 = 0x12345678;
    sattn_csr_write_u32(CSR_SATTN_M_ROWS, test_val_32);
    uint32_t read_val_32 = sattn_csr_read_u32(CSR_SATTN_M_ROWS);
    TEST_ASSERT(read_val_32 == test_val_32, "32-bit CSR read/write mismatch");
    
    // Test 64-bit CSR write/read (pointer)
    uint64_t test_val_64 = 0x123456789ABCDEF0ULL;
    sattn_csr_write_ptr(CSR_SATTN_Q_BASE, test_val_64);
    uint64_t read_val_64 = sattn_csr_read_ptr(CSR_SATTN_Q_BASE);
    TEST_ASSERT(read_val_64 == test_val_64, "64-bit CSR read/write mismatch");
    
    // Test float CSR write/read
    float test_float = 0.125f;
    sattn_csr_write_float(CSR_SATTN_SCALE_FP, test_float);
    float read_float = sattn_csr_read_float(CSR_SATTN_SCALE_FP);
    TEST_ASSERT(fabsf(read_float - test_float) < 1e-6f, "Float CSR read/write mismatch");
    
    TEST_PASS("test_csr_read_write");
}

// Test 2: Hardware capability detection
int test_hw_capabilities(void) {
    printf("Testing hardware capability detection...\n");
    
    uint32_t version = sattn_get_hw_version();
    uint32_t caps = sattn_get_hw_caps();
    
    printf("  Hardware version: 0x%08x\n", version);
    printf("  Capabilities: 0x%08x\n", caps);
    
    // Check expected capabilities
    if (caps & SATTN_CAP_BSR_SUPPORT) {
        printf("  - BSR format supported\n");
    }
    if (caps & SATTN_CAP_BF16_PRECISION) {
        printf("  - BF16 precision supported\n");
    }
    
    TEST_PASS("test_hw_capabilities");
}

// Test 3: Status register functionality
int test_status_register(void) {
    printf("Testing status register...\n");
    
    // Should be ready initially
    TEST_ASSERT(sattn_is_ready(), "Hardware not ready initially");
    TEST_ASSERT(!sattn_is_busy(), "Hardware busy initially");
    TEST_ASSERT(!sattn_has_error(), "Hardware has error initially");
    
    uint32_t status = sattn_csr_read_u32(CSR_SATTN_STATUS);
    TEST_ASSERT(status & SATTN_STATUS_READY, "READY bit not set");
    TEST_ASSERT(!(status & SATTN_STATUS_BUSY), "BUSY bit set incorrectly");
    TEST_ASSERT(!(status & SATTN_STATUS_ERROR), "ERROR bit set incorrectly");
    
    TEST_PASS("test_status_register");
}

// Test 4: Instruction encoding macros
int test_instruction_encoding(void) {
    printf("Testing instruction encoding macros...\n");
    
    // Verify pre-encoded instruction words
    uint32_t expected_spdot = 0x0000402B;  // funct3=100, rest=0
    TEST_ASSERT(SATTN_INSN_SPDOT_BSR == expected_spdot, 
                "SPDOT_BSR encoding incorrect");
    
    uint32_t expected_softmax = 0x0000502B;  // funct3=101, rest=0
    TEST_ASSERT(SATTN_INSN_SOFTMAX_FUSED == expected_softmax,
                "SOFTMAX_FUSED encoding incorrect");
    
    // Test encoding macro
    uint32_t custom_encoding = SATTN_ENCODE_R(
        SATTN_FUNCT7_DEFAULT,  // funct7 = 0
        0,                      // rs2 = x0
        0,                      // rs1 = x0
        SATTN_FUNCT3_GATH2D,   // funct3 = 010
        0,                      // rd = x0
        SATTN_OPCODE           // opcode = 0x2B
    );
    uint32_t expected_gath2d = 0x0000202B;
    TEST_ASSERT(custom_encoding == expected_gath2d,
                "Custom encoding macro incorrect");
    
    TEST_PASS("test_instruction_encoding");
}

// Test 5: Parameter setup helper
int test_param_setup(void) {
    printf("Testing parameter setup helper...\n");
    
    // Create test parameters
    sattn_csr_params_t params = {
        .q_base = 0x1000,
        .k_base = 0x2000,
        .v_base = 0x3000,
        .o_base = 0x4000,
        .idx_base = 0x5000,
        .stride_base = 0x6000,
        .m_rows = 32,
        .head_dim_d = 64,
        .block_size = 64,
        .k_blocks = 16,
        .s_tokens = 1024,
        .scale_fp = 0.125f,
        .gqa_group_size = 1,
        .comp_block_size = 0
    };
    
    // Setup all CSRs at once
    sattn_setup_csrs(&params);
    
    // Verify each CSR was written correctly
    TEST_ASSERT(sattn_csr_read_ptr(CSR_SATTN_Q_BASE) == params.q_base,
                "Q_BASE not set correctly");
    TEST_ASSERT(sattn_csr_read_ptr(CSR_SATTN_K_BASE) == params.k_base,
                "K_BASE not set correctly");
    TEST_ASSERT(sattn_csr_read_u32(CSR_SATTN_M_ROWS) == params.m_rows,
                "M_ROWS not set correctly");
    TEST_ASSERT(sattn_csr_read_u32(CSR_SATTN_HEAD_DIM_D) == params.head_dim_d,
                "HEAD_DIM_D not set correctly");
    TEST_ASSERT(sattn_csr_read_u32(CSR_SATTN_BLOCK_SIZE) == params.block_size,
                "BLOCK_SIZE not set correctly");
    TEST_ASSERT(fabsf(sattn_csr_read_float(CSR_SATTN_SCALE_FP) - params.scale_fp) < 1e-6f,
                "SCALE_FP not set correctly");
    
    TEST_PASS("test_param_setup");
}

// Test 6: Descriptor-based API (high-level)
int test_descriptor_api(void) {
    printf("Testing descriptor-based API...\n");
    
    // Allocate small test buffers
    const int m = 8, d = 16, s = 32;
    float* Q = (float*)malloc(m * d * sizeof(float));
    float* K = (float*)malloc(s * d * sizeof(float));
    float* V = (float*)malloc(s * d * sizeof(float));
    float* O = (float*)malloc(m * d * sizeof(float));
    uint16_t* idx = (uint16_t*)malloc(s * sizeof(uint16_t));
    
    TEST_ASSERT(Q && K && V && O && idx, "Memory allocation failed");
    
    // Initialize buffers with test data
    for (int i = 0; i < m * d; i++) Q[i] = (float)i * 0.01f;
    for (int i = 0; i < s * d; i++) {
        K[i] = (float)i * 0.02f;
        V[i] = (float)i * 0.03f;
    }
    for (int i = 0; i < s; i++) idx[i] = i;
    
    // Create descriptor
    sattn_cmd_desc_t desc = {
        .q_base = (uint64_t)Q,
        .k_base = (uint64_t)K,
        .v_base = (uint64_t)V,
        .o_base = (uint64_t)O,
        .idx_base = (uint64_t)idx,
        .stride_base = 0,
        .m_rows = m,
        .head_dim_d = d,
        .block_size = 16,
        .k_blocks = s / 16,
        .s_tokens = s,
        .scale_fp = 1.0f / sqrtf((float)d)
    };
    
#ifdef SATTN_USE_CSR_INSTRUCTIONS
    // Test CSR-based intrinsics
    printf("  Using CSR-based instruction interface\n");
    
    // Note: These will issue instructions but may not complete on test platform
    // We're mainly testing that the API compiles and executes without crashing
    sattn_csr_params_t params = {
        .q_base = desc.q_base,
        .k_base = desc.k_base,
        .v_base = desc.v_base,
        .o_base = desc.o_base,
        .idx_base = desc.idx_base,
        .stride_base = desc.stride_base,
        .m_rows = desc.m_rows,
        .head_dim_d = desc.head_dim_d,
        .block_size = desc.block_size,
        .k_blocks = desc.k_blocks,
        .s_tokens = desc.s_tokens,
        .scale_fp = desc.scale_fp,
        .gqa_group_size = 1,
        .comp_block_size = 0
    };
    sattn_setup_csrs(&params);
    printf("  CSRs configured successfully\n");
#else
    printf("  Using MMIO-based interface (CSR instructions not enabled)\n");
#endif
    
    // Cleanup
    free(Q);
    free(K);
    free(V);
    free(O);
    free(idx);
    
    TEST_PASS("test_descriptor_api");
}

// Test 7: CSR address definitions
int test_csr_addresses(void) {
    printf("Testing CSR address definitions...\n");
    
    // Verify CSR addresses are in expected range (0x7C0-0x7DF)
    TEST_ASSERT(CSR_SATTN_Q_BASE >= 0x7C0 && CSR_SATTN_Q_BASE <= 0x7DF,
                "Q_BASE address out of range");
    TEST_ASSERT(CSR_SATTN_STATUS >= 0x7C0 && CSR_SATTN_STATUS <= 0x7DF,
                "STATUS address out of range");
    
    // Verify CSR addresses don't overlap improperly
    TEST_ASSERT(CSR_SATTN_Q_BASE != CSR_SATTN_K_BASE,
                "Q_BASE and K_BASE have same address");
    TEST_ASSERT(CSR_SATTN_STATUS != CSR_SATTN_ERROR,
                "STATUS and ERROR have same address");
    
    // Verify expected addresses (from spec)
    TEST_ASSERT(CSR_SATTN_Q_BASE == 0x7C0, "Q_BASE address incorrect");
    TEST_ASSERT(CSR_SATTN_M_ROWS == 0x7C6, "M_ROWS address incorrect");
    TEST_ASSERT(CSR_SATTN_STATUS == 0x7CE, "STATUS address incorrect");
    
    TEST_PASS("test_csr_addresses");
}

// Test 8: Error handling
int test_error_handling(void) {
    printf("Testing error handling...\n");
    
    // Initially should have no errors
    uint32_t error = sattn_get_error();
    TEST_ASSERT(error == SATTN_ERROR_NONE, "Initial error code non-zero");
    
    // Verify error code constants are defined
    TEST_ASSERT(SATTN_ERROR_NONE == 0x00, "ERROR_NONE incorrect");
    TEST_ASSERT(SATTN_ERROR_INVALID_DIM == 0x01, "ERROR_INVALID_DIM incorrect");
    TEST_ASSERT(SATTN_ERROR_MISALIGNED == 0x02, "ERROR_MISALIGNED incorrect");
    TEST_ASSERT(SATTN_ERROR_UNKNOWN == 0xFF, "ERROR_UNKNOWN incorrect");
    
    TEST_PASS("test_error_handling");
}

// Main test runner
int main(int argc, char** argv) {
    printf("========================================\n");
    printf("Custom RISC-V Instruction Test Suite\n");
    printf("========================================\n\n");
    
#ifdef SATTN_USE_CSR_INSTRUCTIONS
    printf("Mode: CSR-based instruction interface\n");
#else
    printf("Mode: MMIO-based interface (fallback)\n");
#endif
    printf("\n");
    
    // Run tests
    test_csr_read_write();
    test_hw_capabilities();
    test_status_register();
    test_instruction_encoding();
    test_param_setup();
    test_descriptor_api();
    test_csr_addresses();
    test_error_handling();
    
    // Summary
    printf("\n========================================\n");
    printf("Test Results:\n");
    printf("  Passed: %d\n", tests_passed);
    printf("  Failed: %d\n", tests_failed);
    printf("========================================\n");
    
    return tests_failed > 0 ? 1 : 0;
}

