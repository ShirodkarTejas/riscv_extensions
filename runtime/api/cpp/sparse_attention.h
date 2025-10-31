#ifndef RISC_SPARSE_ATTENTION_API_SPARSE_ATTENTION_H_
#define RISC_SPARSE_ATTENTION_API_SPARSE_ATTENTION_H_

#include <cstdint>
#include <vector>

namespace sattn {

enum class Pattern { BlockTopK, SlidingGlobal };
enum class Precision { FP16, BF16, INT8 };

struct TensorView {
  void* data;
  int64_t dtype;  // placeholder for now
  std::vector<int64_t> shape;   // [B, H, L, D]
  std::vector<int64_t> stride;  // strides in elements
};

struct SparseAttentionParams {
  Pattern pattern;
  Precision precision;
  bool training;
  // block_topk
  int blockSize;     // tokens per block (e.g., 64)
  float keepRatio;   // e.g., 0.12
  int globalTokens;  // e.g., 16
  // sliding_global
  int windowSize;    // tokens per side (e.g., 512)
};

// Returns O with shape [B, H, L, D]. Caller owns memory management policy.
TensorView SparseAttention(const TensorView& Q,
                           const TensorView& K,
                           const TensorView& V,
                           const SparseAttentionParams& params);

}  // namespace sattn

#endif  // RISC_SPARSE_ATTENTION_API_SPARSE_ATTENTION_H_


