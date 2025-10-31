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

// Forward declare block_topk params to allow prototypes before its full definition.
typedef struct sattn_blocktopk_params_t sattn_blocktopk_params_t;

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

// Quantized variants for sliding_window
void sattn_rvv_sliding_global_bf16(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    sattn_shape_t shape,
    sattn_params_t params);

void sattn_rvv_sliding_global_i8(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    sattn_shape_t shape,
    sattn_params_t params,
    float scale_q,
    float scale_k,
    float scale_v);

void sattn_rvv_sliding_global_i4(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    sattn_shape_t shape,
    sattn_params_t params,
    float scale_q,
    float scale_k,
    float scale_v);

// Quantized variants for block_local_global / block_topk
void sattn_rvv_block_topk_bf16(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    sattn_shape_t shape,
    sattn_blocktopk_params_t params);

void sattn_rvv_block_topk_i8(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    sattn_shape_t shape,
    sattn_blocktopk_params_t params,
    float scale_q,
    float scale_k,
    float scale_v);

void sattn_rvv_block_topk_i4(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    sattn_shape_t shape,
    sattn_blocktopk_params_t params,
    float scale_q,
    float scale_k,
    float scale_v);

// Tiled variant of sliding_window to improve locality for multiple query rows
void sattn_rvv_sliding_global_tiled(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    sattn_shape_t shape,
    sattn_params_t params,
    int tile_rows);

// Read cycle counter if available (RISC-V rdcycle), else returns 0.
uint64_t sattn_rdcycle();

// Lightweight bandwidth/compute proxy counters for RVV baselines
typedef struct {
  uint64_t bytes_read;
  uint64_t bytes_written;
  uint64_t mac_flops;  // count of fused multiply-add operations
} sattn_rvv_counters_t;

void sattn_rvv_counters_reset();
void sattn_rvv_counters_get(sattn_rvv_counters_t* out);

// Block-topk sparse attention baseline (selection scalar, math vectorized where possible)
typedef struct sattn_blocktopk_params_t {
  int block_size;   // tokens per block
  float keep_ratio; // fraction of blocks kept per row
  int global_tokens;
} sattn_blocktopk_params_t;

void sattn_rvv_block_topk(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    sattn_shape_t shape,
    sattn_blocktopk_params_t params);

// Tiled variant of block_topk (BLG) to improve locality across multiple rows
void sattn_rvv_block_topk_tiled(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    sattn_shape_t shape,
    sattn_blocktopk_params_t params,
    int tile_rows);

// N:M structured sparsity – simple wrapper mapping to block_topk with block_size=M, keep_ratio=N/M
typedef struct {
  int n; // tokens kept per group
  int m; // group size
} sattn_nm_params_t;

void sattn_rvv_nm_structured(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    sattn_shape_t shape,
    sattn_nm_params_t params);

// LSH/hashed buckets – simplified: token j participates for query i if (j % buckets) == (i % buckets)
typedef struct {
  int buckets; // number of hash buckets
} sattn_lsh_params_t;

void sattn_rvv_lsh(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    sattn_shape_t shape,
    sattn_lsh_params_t params);

// Tiled variant of LSH selection
void sattn_rvv_lsh_tiled(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    sattn_shape_t shape,
    sattn_lsh_params_t params,
    int tile_rows);

// Segmented reductions: sum across contiguous segments of length seg_len.
// src: [segments, seg_len], dst: [segments]
void sattn_rvv_segmented_sum_f32(const float* src, float* dst,
                                 int64_t segments, int64_t seg_len);

// Softmax over a single row vector of length D (in-place)
void sattn_rvv_softmax_row_f32(float* row, int64_t D);

#ifdef __cplusplus
}
#endif

#endif  // SATTN_RVV_INCLUDE_SPARSE_ATTENTION_RVV_H_


