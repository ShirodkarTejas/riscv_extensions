/**
 * RoCC Driver for Sparse Attention Accelerator (Hardware)
 * 
 * This driver interfaces with the Chisel-implemented hardware accelerator
 * through RoCC custom instructions.
 * 
 * Supports all 5 patterns × 4 precisions from Phase 1 & 2
 */

#ifndef ROCC_DRIVER_HW_H
#define ROCC_DRIVER_HW_H

#include <stdint.h>
#include <string.h>

// =============================================================================
// RoCC Instruction Macros
// =============================================================================

#define ROCC_INSTRUCTION_DSS(x, rd, rs1, rs2, funct) \
    asm volatile ( \
        ".insn r 0x0b, " #funct ", " #x ", %0, %1, %2" \
        : "=r"(rd) : "r"(rs1), "r"(rs2))

#define ROCC_INSTRUCTION_DS(x, rd, rs1, funct) \
    asm volatile ( \
        ".insn r 0x0b, " #funct ", " #x ", %0, %1, x0" \
        : "=r"(rd) : "r"(rs1))

#define ROCC_INSTRUCTION_D(x, rd, funct) \
    asm volatile ( \
        ".insn r 0x0b, " #funct ", " #x ", %0, x0, x0" \
        : "=r"(rd))

// =============================================================================
// Accelerator Control Functions
// =============================================================================

// Pattern IDs
#define PATTERN_SLIDING_WINDOW      0
#define PATTERN_BLOCK_LOCAL_GLOBAL  1
#define PATTERN_NM_STRUCTURED       2
#define PATTERN_LSH                 3
#define PATTERN_LANDMARK            4

// Precision IDs
#define PRECISION_FP32  0
#define PRECISION_BF16  1
#define PRECISION_I8    2
#define PRECISION_I4    3

/**
 * Load sparsity indices to accelerator
 */
static inline void sattn_hw_load_indices(uint64_t addr, uint32_t size) {
    uint64_t dummy;
    ROCC_INSTRUCTION_DSS(0, dummy, addr, size, 0);
}

/**
 * Configure shape (B, H, L, D)
 */
static inline void sattn_hw_config_shape(uint32_t B, uint32_t H, uint32_t L, uint32_t D) {
    uint64_t dummy;
    uint64_t rs1 = ((uint64_t)B << 16) | H;
    uint64_t rs2 = ((uint64_t)L << 16) | D;
    ROCC_INSTRUCTION_DSS(0, dummy, rs1, rs2, 1);
}

/**
 * Configure pattern and precision
 */
static inline void sattn_hw_config_pattern(uint8_t pattern, uint8_t precision, uint32_t param) {
    uint64_t dummy;
    uint64_t rs1 = ((uint64_t)precision << 8) | pattern;
    ROCC_INSTRUCTION_DSS(0, dummy, rs1, param, 2);
}

/**
 * Load Q, K pointers
 */
static inline void sattn_hw_load_qk(uint64_t Q_addr, uint64_t K_addr) {
    uint64_t dummy;
    ROCC_INSTRUCTION_DSS(0, dummy, Q_addr, K_addr, 5);
}

/**
 * Execute sparse attention (provides V, O pointers)
 */
static inline void sattn_hw_execute(uint64_t V_addr, uint64_t O_addr) {
    uint64_t dummy;
    ROCC_INSTRUCTION_DSS(0, dummy, V_addr, O_addr, 3);
}

/**
 * Check accelerator status
 * Returns: 0 = busy, 1 = done
 */
static inline uint32_t sattn_hw_status(void) {
    uint64_t status;
    ROCC_INSTRUCTION_D(0, status, 4);
    return (uint32_t)status;
}

/**
 * Read performance counters
 * Returns: [31:0] = cycles, [63:32] = memory ops
 */
static inline uint64_t sattn_hw_get_counters(void) {
    uint64_t counters;
    ROCC_INSTRUCTION_D(0, counters, 6);
    return counters;
}

// =============================================================================
// High-Level API
// =============================================================================

typedef struct {
    int batch;
    int heads;
    int seq_len;
    int head_dim;
} sattn_hw_shape_t;

typedef struct {
    uint8_t pattern;
    uint8_t precision;
    union {
        struct { uint32_t window_size; } sliding_window;
        struct { uint32_t block_size; float keep_ratio; uint32_t global_tokens; } block_local_global;
        struct { uint32_t n; uint32_t m; } nm_structured;
        struct { uint32_t buckets; } lsh;
        struct { uint32_t num_landmarks; } landmark;
    } params;
} sattn_hw_config_t;

/**
 * Execute sparse attention on hardware accelerator
 */
static inline int sattn_hw_run(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    sattn_hw_shape_t shape,
    sattn_hw_config_t config
) {
    // 1. Configure shape
    sattn_hw_config_shape(shape.batch, shape.heads, shape.seq_len, shape.head_dim);
    
    // 2. Configure pattern
    uint32_t param = 0;
    switch(config.pattern) {
        case PATTERN_SLIDING_WINDOW:
            param = config.params.sliding_window.window_size;
            break;
        case PATTERN_BLOCK_LOCAL_GLOBAL:
            param = config.params.block_local_global.block_size;
            break;
        case PATTERN_NM_STRUCTURED:
            param = (config.params.nm_structured.m << 16) | config.params.nm_structured.n;
            break;
        case PATTERN_LSH:
            param = config.params.lsh.buckets;
            break;
        case PATTERN_LANDMARK:
            param = config.params.landmark.num_landmarks;
            break;
    }
    sattn_hw_config_pattern(config.pattern, config.precision, param);
    
    // 3. Load data pointers
    sattn_hw_load_qk((uint64_t)Q, (uint64_t)K);
    
    // 4. Execute
    sattn_hw_execute((uint64_t)V, (uint64_t)O);
    
    // 5. Poll for completion
    while (sattn_hw_status() == 0) {
        // Wait for accelerator
    }
    
    // 6. Get performance counters
    uint64_t counters = sattn_hw_get_counters();
    uint32_t cycles = counters & 0xFFFFFFFF;
    uint32_t mem_ops = (counters >> 32) & 0xFFFFFFFF;
    
    return 0;  // Success
}

// =============================================================================
// Pattern-Specific Convenience Functions
// =============================================================================

static inline int sattn_hw_sliding_window(
    const float* Q, const float* K, const float* V, float* O,
    int B, int H, int L, int D,
    int window_size, uint8_t precision
) {
    sattn_hw_shape_t shape = {B, H, L, D};
    sattn_hw_config_t config = {
        .pattern = PATTERN_SLIDING_WINDOW,
        .precision = precision,
        .params.sliding_window.window_size = window_size
    };
    return sattn_hw_run(Q, K, V, O, shape, config);
}

static inline int sattn_hw_block_local_global(
    const float* Q, const float* K, const float* V, float* O,
    int B, int H, int L, int D,
    int block_size, float keep_ratio, int global_tokens, uint8_t precision
) {
    sattn_hw_shape_t shape = {B, H, L, D};
    sattn_hw_config_t config = {
        .pattern = PATTERN_BLOCK_LOCAL_GLOBAL,
        .precision = precision,
        .params.block_local_global = {block_size, keep_ratio, global_tokens}
    };
    return sattn_hw_run(Q, K, V, O, shape, config);
}

static inline int sattn_hw_nm_structured(
    const float* Q, const float* K, const float* V, float* O,
    int B, int H, int L, int D,
    int n, int m, uint8_t precision
) {
    sattn_hw_shape_t shape = {B, H, L, D};
    sattn_hw_config_t config = {
        .pattern = PATTERN_NM_STRUCTURED,
        .precision = precision,
        .params.nm_structured = {n, m}
    };
    return sattn_hw_run(Q, K, V, O, shape, config);
}

static inline int sattn_hw_lsh(
    const float* Q, const float* K, const float* V, float* O,
    int B, int H, int L, int D,
    int buckets, uint8_t precision
) {
    sattn_hw_shape_t shape = {B, H, L, D};
    sattn_hw_config_t config = {
        .pattern = PATTERN_LSH,
        .precision = precision,
        .params.lsh.buckets = buckets
    };
    return sattn_hw_run(Q, K, V, O, shape, config);
}

static inline int sattn_hw_landmark(
    const float* Q, const float* K, const float* V, float* O,
    int B, int H, int L, int D,
    int num_landmarks, uint8_t precision
) {
    sattn_hw_shape_t shape = {B, H, L, D};
    sattn_hw_config_t config = {
        .pattern = PATTERN_LANDMARK,
        .precision = precision,
        .params.landmark.num_landmarks = num_landmarks
    };
    return sattn_hw_run(Q, K, V, O, shape, config);
}

#endif // ROCC_DRIVER_HW_H

