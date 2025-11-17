/**
 * Basic Hardware Test for Sparse Attention Accelerator
 * 
 * Tests the hardware accelerator with a simple sliding_window pattern
 */

#include "rocc_driver_hw.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define B 1
#define H 2
#define L 32
#define D 16

// Helper: Initialize with simple pattern
void init_tensor(float* tensor, int size, float scale) {
    for (int i = 0; i < size; i++) {
        tensor[i] = (float)(i % 10) * scale;
    }
}

// Helper: Print first few values
void print_tensor(const char* name, const float* tensor, int size, int limit) {
    printf("%s (first %d): ", name, limit);
    for (int i = 0; i < limit && i < size; i++) {
        printf("%.4f ", tensor[i]);
    }
    printf("\n");
}

// Helper: Compute checksum
float compute_checksum(const float* tensor, int size) {
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        sum += tensor[i];
    }
    return sum;
}

int main() {
    printf("=================================================================\n");
    printf("Hardware Sparse Attention Test\n");
    printf("=================================================================\n");
    printf("Configuration:\n");
    printf("  Batch (B):     %d\n", B);
    printf("  Heads (H):     %d\n", H);
    printf("  Seq Len (L):   %d\n", L);
    printf("  Head Dim (D):  %d\n", D);
    printf("  Total size:    %d elements\n", B*H*L*D);
    printf("=================================================================\n\n");

    // Allocate tensors
    int total_size = B * H * L * D;
    float *Q = (float*)malloc(total_size * sizeof(float));
    float *K = (float*)malloc(total_size * sizeof(float));
    float *V = (float*)malloc(total_size * sizeof(float));
    float *O = (float*)malloc(total_size * sizeof(float));

    if (!Q || !K || !V || !O) {
        printf("ERROR: Memory allocation failed\n");
        return 1;
    }

    // Initialize
    printf("Initializing tensors...\n");
    init_tensor(Q, total_size, 0.01f);
    init_tensor(K, total_size, 0.02f);
    init_tensor(V, total_size, 0.03f);
    init_tensor(O, total_size, 0.0f);

    // Print inputs
    print_tensor("Q", Q, total_size, 10);
    print_tensor("K", K, total_size, 10);
    print_tensor("V", V, total_size, 10);

    printf("\nInput checksums:\n");
    printf("  Q: %.6f\n", compute_checksum(Q, total_size));
    printf("  K: %.6f\n", compute_checksum(K, total_size));
    printf("  V: %.6f\n", compute_checksum(V, total_size));

    // Run on hardware
    printf("\n=================================================================\n");
    printf("Running on hardware accelerator...\n");
    printf("  Pattern: sliding_window\n");
    printf("  Precision: fp32\n");
    printf("  Window size: 8\n");
    printf("=================================================================\n");

    int result = sattn_hw_sliding_window(
        Q, K, V, O,
        B, H, L, D,
        8,              // window_size
        PRECISION_FP32  // precision
    );

    if (result != 0) {
        printf("ERROR: Hardware execution failed with code %d\n", result);
        goto cleanup;
    }

    printf("✅ Hardware execution complete!\n\n");

    // Get performance counters
    uint64_t counters = sattn_hw_get_counters();
    uint32_t cycles = counters & 0xFFFFFFFF;
    uint32_t mem_ops = (counters >> 32) & 0xFFFFFFFF;

    printf("Performance:\n");
    printf("  Cycles: %u\n", cycles);
    printf("  Memory ops: %u\n", mem_ops);

    // Check output
    print_tensor("O", O, total_size, 10);
    printf("\nOutput checksum: %.6f\n", compute_checksum(O, total_size));

    // Validate (simple sanity check)
    int valid = 1;
    for (int i = 0; i < total_size; i++) {
        if (isnan(O[i]) || isinf(O[i])) {
            printf("ERROR: Invalid output at index %d: %.6f\n", i, O[i]);
            valid = 0;
            break;
        }
    }

    if (valid) {
        printf("\n✅ Output validation passed!\n");
    } else {
        printf("\n❌ Output validation failed!\n");
    }

    printf("\n=================================================================\n");
    printf("Test %s\n", valid ? "PASSED" : "FAILED");
    printf("=================================================================\n");

cleanup:
    free(Q);
    free(K);
    free(V);
    free(O);

    return valid ? 0 : 1;
}

