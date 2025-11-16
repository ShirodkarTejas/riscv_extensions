// Generated from MLIR - Phase 2
// Pattern: landmark, Precision: bf16
// This code calls the validated RVV backend from Phase 1

#include "backends/rvv/include/sparse_attention_rvv.h"
#include <stdint.h>

/**
 * Generated sparse attention function
 * 
 * @param Q Query tensor [B x H x L x D]
 * @param K Key tensor [B x H x L x D]
 * @param V Value tensor [B x H x L x D]
 * @param O Output tensor [B x H x L x D]
 * @param B Batch size (default: 1)
 * @param H Number of heads (default: 8)
 * @param L Sequence length (default: 128)
 * @param D Head dimension (default: 64)
 */
void sparse_attention_generated(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int B,
    int H,
    int L,
    int D
) {
    // Shape configuration
    sattn_shape_t shape = {
        .batch = B,
        .heads = H,
        .seq_len = L,
        .head_dim = D
    };
    
    // Pattern-specific parameters
    sattn_landmark_params_t params = {
        .num_landmarks = 16
    };
    
    // Call the validated RVV backend
    sattn_rvv_landmark_bf16(Q, K, V, O, shape, params);
}

// Convenience wrapper with default sizes
void sparse_attention_default(
    const float* Q,
    const float* K,
    const float* V,
    float* O
) {
    sparse_attention_generated(Q, K, V, O, 1, 8, 128, 64);
}

/**
 * Usage example:
 * 
 * #include <stdlib.h>
 * 
 * int main() {
 *     // Allocate tensors
 *     float* Q = (float*)malloc(1*8*128*64 * sizeof(float));
 *     float* K = (float*)malloc(1*8*128*64 * sizeof(float));
 *     float* V = (float*)malloc(1*8*128*64 * sizeof(float));
 *     float* O = (float*)malloc(1*8*128*64 * sizeof(float));
 *     
 *     // Initialize Q, K, V with your data...
 *     
 *     // Run sparse attention
 *     sparse_attention_default(Q, K, V, O);
 *     
 *     // Use output O...
 *     
 *     // Cleanup
 *     free(Q); free(K); free(V); free(O);
 *     return 0;
 * }
 */
