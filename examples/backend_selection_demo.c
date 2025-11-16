// Example: Using the sparse attention library with automatic backend selection
// This demonstrates how the same code works across different backends

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Detect which backend is available at compile time
#if defined(__riscv) && defined(__riscv_vector)
  // RVV backend: Standard RISC-V Vector extension
  #include "backends/rvv/include/sparse_attention_rvv.h"
  #define BACKEND "RVV (RISC-V Vector)"
  #define USE_RVV 1

#elif defined(SATTN_USE_CSR_INSTRUCTIONS)
  // Custom instruction backend: Hardware accelerator with custom ISA
  #include "hw/spec/rocc_intrinsics.h"
  #define BACKEND "Custom ISA (Hardware Accelerator)"
  #define USE_CUSTOM_ISA 1

#else
  // CPU reference backend (fallback)
  #include "ops/sparse_attention/cpu/python_ref.h"
  #define BACKEND "CPU Reference"
  #define USE_CPU_REF 1

#endif

// Unified function that works across all backends
void sparse_attention_demo(void) {
    printf("========================================\n");
    printf("Sparse Attention Backend Demo\n");
    printf("Backend: %s\n", BACKEND);
    printf("========================================\n\n");
    
    // Problem size (small for demo)
    const int B = 1;     // Batch size
    const int H = 2;     // Number of heads
    const int L = 128;   // Sequence length
    const int D = 32;    // Head dimension
    
    // Allocate tensors
    size_t qkv_size = B * H * L * D * sizeof(float);
    float *Q = (float*)malloc(qkv_size);
    float *K = (float*)malloc(qkv_size);
    float *V = (float*)malloc(qkv_size);
    float *O = (float*)malloc(qkv_size);
    
    // Initialize with simple test pattern
    for (int i = 0; i < B * H * L * D; i++) {
        Q[i] = (float)(i % 100) / 100.0f;
        K[i] = (float)((i + 1) % 100) / 100.0f;
        V[i] = (float)((i + 2) % 100) / 100.0f;
    }
    
    printf("Problem size: B=%d, H=%d, L=%d, D=%d\n\n", B, H, L, D);
    
    // Run sparse attention based on backend
#ifdef USE_RVV
    // RVV Backend: Works on QEMU with -cpu rv64,v=true
    printf("Running with RVV backend (standard Vector extension)...\n");
    printf("This works on QEMU with: qemu-riscv64 -cpu rv64,v=true\n\n");
    
    sattn_shape_t shape = { .B = B, .H = H, .L = L, .D = D };
    sattn_sw_params_t params = { 
        .window_size = 16, 
        .global_tokens = 4,
        .tile_rows = 4
    };
    
    // Call RVV implementation
    sattn_rvv_sliding_global(Q, K, V, O, shape, params);
    
    // Get performance metrics
    sattn_rvv_counters_t counters;
    sattn_rvv_counters_get(&counters);
    printf("Performance metrics:\n");
    printf("  Bytes read:    %lld\n", (long long)counters.bytes_read);
    printf("  Bytes written: %lld\n", (long long)counters.bytes_written);
    printf("  MAC ops:       %lld\n", (long long)counters.mac_flops);
    
#elif defined(USE_CUSTOM_ISA)
    // Custom ISA Backend: Requires hardware or modified QEMU
    printf("Running with custom instruction backend...\n");
    printf("This requires hardware with custom extension\n");
    printf("  or QEMU with plugin: qemu-riscv64 -plugin libsattn_custom.so\n\n");
    
    sattn_cmd_desc_t desc = {
        .q_base = (uint64_t)Q,
        .k_base = (uint64_t)K,
        .v_base = (uint64_t)V,
        .o_base = (uint64_t)O,
        .m_rows = L,
        .head_dim_d = D,
        .block_size = 16,
        .s_tokens = 32,
        .scale_fp = 1.0f / sqrtf((float)D)
    };
    
    // These use custom instructions (opcode 0x2B)
    printf("Issuing custom instructions:\n");
    printf("  1. spdot_bsr (Q × K^T)\n");
    sattn_spdot_bsr(&desc);
    
    printf("  2. softmax_fused\n");
    sattn_softmax_fused(&desc);
    
    printf("  3. spmm_bsr (Attn × V)\n");
    sattn_spmm_bsr(&desc);
    
    // Check hardware capabilities
    uint32_t hw_version = sattn_get_hw_version();
    uint32_t hw_caps = sattn_get_hw_caps();
    printf("\nHardware info:\n");
    printf("  Version: 0x%08X\n", hw_version);
    printf("  Capabilities: 0x%08X\n", hw_caps);
    
#else
    // CPU Reference Backend
    printf("Running with CPU reference backend (NumPy-like)...\n");
    printf("This works on any platform (fallback).\n\n");
    
    // Call CPU reference (would need Python bridge or C implementation)
    printf("CPU reference execution...\n");
    
#endif
    
    // Compute simple checksum for verification
    float checksum = 0.0f;
    for (int i = 0; i < B * H * L * D; i++) {
        checksum += O[i];
    }
    printf("\nOutput checksum: %.6f\n", checksum);
    
    // Cleanup
    free(Q);
    free(K);
    free(V);
    free(O);
    
    printf("\n========================================\n");
    printf("Demo complete!\n");
    printf("========================================\n");
}

int main(int argc, char** argv) {
    printf("\n");
    printf("╔══════════════════════════════════════════════════╗\n");
    printf("║  Sparse Attention Multi-Backend Demo            ║\n");
    printf("║                                                  ║\n");
    printf("║  This code compiles for different backends:     ║\n");
    printf("║  - RVV: Standard Vector extension (QEMU)        ║\n");
    printf("║  - Custom ISA: Hardware accelerator             ║\n");
    printf("║  - CPU: Reference implementation                ║\n");
    printf("╚══════════════════════════════════════════════════╝\n");
    printf("\n");
    
    sparse_attention_demo();
    
    return 0;
}

/* Compilation examples:

1. RVV Backend (works with QEMU):
   riscv64-linux-gnu-gcc -march=rv64gcv -mabi=lp64d \
     -I. backend_selection_demo.c \
     -L build/rvv-riscv64 -lsattn_rvv -lm \
     -o demo_rvv
   
   qemu-riscv64 -L /usr/riscv64-linux-gnu \
     -cpu rv64,v=true,vlen=128,elen=64 \
     ./demo_rvv

2. Custom ISA Backend (needs hardware):
   riscv64-linux-gnu-gcc -march=rv64gc \
     -DSATTN_USE_CSR_INSTRUCTIONS \
     -I. backend_selection_demo.c \
     -o demo_custom
   
   # Run on hardware with custom extension
   ./demo_custom
   
   # Or with QEMU plugin:
   qemu-riscv64 -plugin libsattn_custom.so ./demo_custom

3. CPU Reference (standard compilation):
   gcc -I. backend_selection_demo.c -lm -o demo_cpu
   ./demo_cpu

*/

