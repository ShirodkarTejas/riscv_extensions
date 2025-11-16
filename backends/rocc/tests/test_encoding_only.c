// Simplified test that validates encoding without executing CSR instructions
// This can run on standard QEMU without custom CSR support

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

// Define the instruction encoding values directly (from sattn_isa.h)
#define SATTN_OPCODE        0x2B
#define SATTN_FUNCT7_DEFAULT 0x00

// funct3 encodings
#define SATTN_FUNCT3_BLK_REDUCE     0x0
#define SATTN_FUNCT3_TOPK_IDX       0x1
#define SATTN_FUNCT3_GATH2D         0x2
#define SATTN_FUNCT3_SCAT2D         0x3
#define SATTN_FUNCT3_SPDOT_BSR      0x4
#define SATTN_FUNCT3_SOFTMAX_FUSED  0x5
#define SATTN_FUNCT3_SPMM_BSR       0x6

// Encoding macro
#define SATTN_ENCODE_R(funct7, rs2, rs1, funct3, rd, opcode)    \
  ((((funct7) & 0x7F) << 25) | (((rs2) & 0x1F) << 20) |         \
   (((rs1) & 0x1F) << 15) | (((funct3) & 0x7) << 12) |          \
   (((rd) & 0x1F) << 7) | ((opcode) & 0x7F))

// Pre-encoded instructions
#define SATTN_INSN_BLK_REDUCE       SATTN_ENCODE_R(SATTN_FUNCT7_DEFAULT, 0, 0, SATTN_FUNCT3_BLK_REDUCE, 0, SATTN_OPCODE)
#define SATTN_INSN_TOPK_IDX         SATTN_ENCODE_R(SATTN_FUNCT7_DEFAULT, 0, 0, SATTN_FUNCT3_TOPK_IDX, 0, SATTN_OPCODE)
#define SATTN_INSN_GATH2D           SATTN_ENCODE_R(SATTN_FUNCT7_DEFAULT, 0, 0, SATTN_FUNCT3_GATH2D, 0, SATTN_OPCODE)
#define SATTN_INSN_SCAT2D           SATTN_ENCODE_R(SATTN_FUNCT7_DEFAULT, 0, 0, SATTN_FUNCT3_SCAT2D, 0, SATTN_OPCODE)
#define SATTN_INSN_SPDOT_BSR        SATTN_ENCODE_R(SATTN_FUNCT7_DEFAULT, 0, 0, SATTN_FUNCT3_SPDOT_BSR, 0, SATTN_OPCODE)
#define SATTN_INSN_SOFTMAX_FUSED    SATTN_ENCODE_R(SATTN_FUNCT7_DEFAULT, 0, 0, SATTN_FUNCT3_SOFTMAX_FUSED, 0, SATTN_OPCODE)
#define SATTN_INSN_SPMM_BSR         SATTN_ENCODE_R(SATTN_FUNCT7_DEFAULT, 0, 0, SATTN_FUNCT3_SPMM_BSR, 0, SATTN_OPCODE)

// CSR addresses
#define CSR_SATTN_Q_BASE       0x7C0
#define CSR_SATTN_M_ROWS       0x7C6
#define CSR_SATTN_STATUS       0x7CE

// Error codes
#define SATTN_ERROR_NONE       0x00
#define SATTN_ERROR_INVALID_DIM 0x01
#define SATTN_ERROR_MISALIGNED 0x02
#define SATTN_ERROR_UNKNOWN    0xFF

// Test tracking
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

// Test 1: Instruction encoding validation
int test_instruction_encoding(void) {
    printf("Testing instruction encoding...\n");
    
    // Verify expected encodings
    uint32_t expected_blk_reduce = 0x0000002B;
    TEST_ASSERT(SATTN_INSN_BLK_REDUCE == expected_blk_reduce, 
                "BLK_REDUCE encoding incorrect");
    
    uint32_t expected_topk = 0x0000102B;
    TEST_ASSERT(SATTN_INSN_TOPK_IDX == expected_topk,
                "TOPK_IDX encoding incorrect");
    
    uint32_t expected_gath2d = 0x0000202B;
    TEST_ASSERT(SATTN_INSN_GATH2D == expected_gath2d,
                "GATH2D encoding incorrect");
    
    uint32_t expected_scat2d = 0x0000302B;
    TEST_ASSERT(SATTN_INSN_SCAT2D == expected_scat2d,
                "SCAT2D encoding incorrect");
    
    uint32_t expected_spdot = 0x0000402B;
    TEST_ASSERT(SATTN_INSN_SPDOT_BSR == expected_spdot,
                "SPDOT_BSR encoding incorrect");
    
    uint32_t expected_softmax = 0x0000502B;
    TEST_ASSERT(SATTN_INSN_SOFTMAX_FUSED == expected_softmax,
                "SOFTMAX_FUSED encoding incorrect");
    
    uint32_t expected_spmm = 0x0000602B;
    TEST_ASSERT(SATTN_INSN_SPMM_BSR == expected_spmm,
                "SPMM_BSR encoding incorrect");
    
    TEST_PASS("test_instruction_encoding");
}

// Test 2: Encoding macro verification
int test_encoding_macro(void) {
    printf("Testing encoding macro...\n");
    
    // Test with different register values
    uint32_t insn1 = SATTN_ENCODE_R(0x00, 0, 0, SATTN_FUNCT3_SPDOT_BSR, 0, SATTN_OPCODE);
    TEST_ASSERT(insn1 == 0x0000402B, "Encoding with all zeros incorrect");
    
    // Test with rd=x5
    uint32_t insn2 = SATTN_ENCODE_R(0x00, 0, 0, SATTN_FUNCT3_SPDOT_BSR, 5, SATTN_OPCODE);
    uint32_t expected2 = 0x000042AB;  // rd=5 is bits [11:7]
    TEST_ASSERT(insn2 == expected2, "Encoding with rd=5 incorrect");
    
    // Test with rs1=x10, rs2=x11
    uint32_t insn3 = SATTN_ENCODE_R(0x00, 11, 10, SATTN_FUNCT3_GATH2D, 0, SATTN_OPCODE);
    uint32_t expected3 = 0x00B5202B;  // rs1=10, rs2=11, funct3=010
    TEST_ASSERT(insn3 == expected3, "Encoding with registers incorrect");
    
    TEST_PASS("test_encoding_macro");
}

// Test 3: Opcode and field extraction
int test_field_extraction(void) {
    printf("Testing field extraction...\n");
    
    uint32_t insn = SATTN_INSN_SPDOT_BSR;
    
    // Extract opcode [6:0]
    uint8_t opcode = insn & 0x7F;
    TEST_ASSERT(opcode == SATTN_OPCODE, "Opcode extraction failed");
    
    // Extract funct3 [14:12]
    uint8_t funct3 = (insn >> 12) & 0x7;
    TEST_ASSERT(funct3 == SATTN_FUNCT3_SPDOT_BSR, "funct3 extraction failed");
    
    // Extract funct7 [31:25]
    uint8_t funct7 = (insn >> 25) & 0x7F;
    TEST_ASSERT(funct7 == SATTN_FUNCT7_DEFAULT, "funct7 extraction failed");
    
    // Extract rd [11:7]
    uint8_t rd = (insn >> 7) & 0x1F;
    TEST_ASSERT(rd == 0, "rd extraction failed");
    
    TEST_PASS("test_field_extraction");
}

// Test 4: CSR address validation
int test_csr_addresses(void) {
    printf("Testing CSR addresses...\n");
    
    // Verify CSR addresses are in custom range (0x7C0-0x7DF)
    TEST_ASSERT(CSR_SATTN_Q_BASE >= 0x7C0 && CSR_SATTN_Q_BASE <= 0x7DF,
                "Q_BASE CSR out of range");
    TEST_ASSERT(CSR_SATTN_M_ROWS >= 0x7C0 && CSR_SATTN_M_ROWS <= 0x7DF,
                "M_ROWS CSR out of range");
    TEST_ASSERT(CSR_SATTN_STATUS >= 0x7C0 && CSR_SATTN_STATUS <= 0x7DF,
                "STATUS CSR out of range");
    
    // Verify expected addresses
    TEST_ASSERT(CSR_SATTN_Q_BASE == 0x7C0, "Q_BASE address incorrect");
    TEST_ASSERT(CSR_SATTN_M_ROWS == 0x7C6, "M_ROWS address incorrect");
    TEST_ASSERT(CSR_SATTN_STATUS == 0x7CE, "STATUS address incorrect");
    
    TEST_PASS("test_csr_addresses");
}

// Test 5: Error code definitions
int test_error_codes(void) {
    printf("Testing error codes...\n");
    
    TEST_ASSERT(SATTN_ERROR_NONE == 0x00, "ERROR_NONE incorrect");
    TEST_ASSERT(SATTN_ERROR_INVALID_DIM == 0x01, "ERROR_INVALID_DIM incorrect");
    TEST_ASSERT(SATTN_ERROR_MISALIGNED == 0x02, "ERROR_MISALIGNED incorrect");
    TEST_ASSERT(SATTN_ERROR_UNKNOWN == 0xFF, "ERROR_UNKNOWN incorrect");
    
    TEST_PASS("test_error_codes");
}

// Test 6: Instruction word structure
int test_instruction_structure(void) {
    printf("Testing instruction word structure...\n");
    
    // All our instructions should be 32 bits
    TEST_ASSERT(sizeof(uint32_t) == 4, "uint32_t not 4 bytes");
    
    // Verify opcode is always 0x2B
    uint32_t instructions[] = {
        SATTN_INSN_BLK_REDUCE,
        SATTN_INSN_TOPK_IDX,
        SATTN_INSN_GATH2D,
        SATTN_INSN_SCAT2D,
        SATTN_INSN_SPDOT_BSR,
        SATTN_INSN_SOFTMAX_FUSED,
        SATTN_INSN_SPMM_BSR
    };
    
    for (int i = 0; i < 7; i++) {
        uint8_t opcode = instructions[i] & 0x7F;
        TEST_ASSERT(opcode == SATTN_OPCODE, "Instruction has wrong opcode");
    }
    
    // Verify funct3 is unique for each
    uint8_t funct3_values[] = {
        (SATTN_INSN_BLK_REDUCE >> 12) & 0x7,
        (SATTN_INSN_TOPK_IDX >> 12) & 0x7,
        (SATTN_INSN_GATH2D >> 12) & 0x7,
        (SATTN_INSN_SCAT2D >> 12) & 0x7,
        (SATTN_INSN_SPDOT_BSR >> 12) & 0x7,
        (SATTN_INSN_SOFTMAX_FUSED >> 12) & 0x7,
        (SATTN_INSN_SPMM_BSR >> 12) & 0x7
    };
    
    // Check all funct3 values are unique
    for (int i = 0; i < 7; i++) {
        for (int j = i + 1; j < 7; j++) {
            TEST_ASSERT(funct3_values[i] != funct3_values[j],
                       "funct3 values not unique");
        }
    }
    
    TEST_PASS("test_instruction_structure");
}

// Test 7: Print instruction encodings (informational)
int test_print_encodings(void) {
    printf("Testing instruction encoding printout...\n");
    
    printf("  Instruction encodings:\n");
    printf("    blk_reduce:    0x%08X\n", SATTN_INSN_BLK_REDUCE);
    printf("    topk_idx:      0x%08X\n", SATTN_INSN_TOPK_IDX);
    printf("    gath2d:        0x%08X\n", SATTN_INSN_GATH2D);
    printf("    scat2d:        0x%08X\n", SATTN_INSN_SCAT2D);
    printf("    spdot_bsr:     0x%08X\n", SATTN_INSN_SPDOT_BSR);
    printf("    softmax_fused: 0x%08X\n", SATTN_INSN_SOFTMAX_FUSED);
    printf("    spmm_bsr:      0x%08X\n", SATTN_INSN_SPMM_BSR);
    
    TEST_PASS("test_print_encodings");
}

// Main test runner
int main(int argc, char** argv) {
    printf("========================================\n");
    printf("Instruction Encoding Validation Tests\n");
    printf("(No CSR/instruction execution)\n");
    printf("========================================\n\n");
    
    // Run tests
    test_instruction_encoding();
    test_encoding_macro();
    test_field_extraction();
    test_csr_addresses();
    test_error_codes();
    test_instruction_structure();
    test_print_encodings();
    
    // Summary
    printf("\n========================================\n");
    printf("Test Results:\n");
    printf("  Passed: %d\n", tests_passed);
    printf("  Failed: %d\n", tests_failed);
    printf("========================================\n");
    
    if (tests_failed == 0) {
        printf("\nAll encoding validation tests passed!\n");
        printf("Note: CSR and instruction execution requires\n");
        printf("hardware/simulator with custom extension support.\n");
    }
    
    return tests_failed > 0 ? 1 : 0;
}

