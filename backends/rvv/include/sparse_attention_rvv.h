#ifndef SATTN_RVV_INCLUDE_SPARSE_ATTENTION_RVV_H_
#define SATTN_RVV_INCLUDE_SPARSE_ATTENTION_RVV_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  int64_t B, H, L, D;
} sattn_shape_t;

typedef struct {
  int window_size;     // sliding window size (tokens per side)
  int block_size;      // token block size (for compatibility)
} sattn_params_t;

// Sliding-window sparse attention baseline on CPU/RVV. All tensors are row-major
// with shape [B, H, L, D] and contiguous strides.
// Q,K,V,O point to float buffers (compute in fp32 baseline).
void sattn_rvv_sliding_global(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    sattn_shape_t shape,
    sattn_params_t params);

// Read cycle counter if available (RISC-V rdcycle), else returns 0.
uint64_t sattn_rdcycle();

#ifdef __cplusplus
}
#endif

#endif  // SATTN_RVV_INCLUDE_SPARSE_ATTENTION_RVV_H_


