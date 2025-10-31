// RUN: sattn-opt %s -sattn-materialize-indices -sattn-tile -sattn-fuse-softmax -sattn-lower-to-rocc | FileCheck %s

module {
  func.func @f(%q: tensor<1x2x64x32xf32>, %k: tensor<1x2x64x32xf32>, %v: tensor<1x2x64x32xf32>) -> tensor<1x2x64x32xf32> {
    // CHECK: sattn.rocc_call
    %o = "sattn.sparse_attention"(%q, %k, %v) {
      pattern = "block_topk",
      block_size = 16 : i64,
      keep_ratio = 0.25 : f32,
      global_tokens = 8 : i64,
      precision = "bf16",
      softmax_mode = "logsumexp",
      tile_M = 16 : i64,
      tile_D = 32 : i64,
      tile_S = 64 : i64
    } : (tensor<1x2x64x32xf32>, tensor<1x2x64x32xf32>, tensor<1x2x64x32xf32>) -> tensor<1x2x64x32xf32>
    return %o : tensor<1x2x64x32xf32>
  }
}
