// RUN: sattn-opt %s -sattn-materialize-indices -sattn-tile -sattn-fuse-softmax -sattn-lower-to-rvv | FileCheck %s

module {
  func.func @f(%q: tensor<1x2x128x64xf32>, %k: tensor<1x2x128x64xf32>, %v: tensor<1x2x128x64xf32>) -> tensor<1x2x128x64xf32> {
    // CHECK: sattn.rvv_call
    %o = "sattn.sparse_attention"(%q, %k, %v) {
      pattern = "sliding_global",
      block_size = 64 : i64,
      keep_ratio = 0.12 : f32,
      global_tokens = 16 : i64,
      window_size = 16 : i64,
      precision = "bf16",
      softmax_mode = "logsumexp",
      tile_M = 64 : i64,
      tile_D = 64 : i64,
      tile_S = 128 : i64
    } : (tensor<1x2x128x64xf32>, tensor<1x2x128x64xf32>, tensor<1x2x128x64xf32>) -> tensor<1x2x128x64xf32>
    return %o : tensor<1x2x128x64xf32>
  }
}
