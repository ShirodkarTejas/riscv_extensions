// Example use of sattn.sparse_attention
module {
  func.func @attn_example(%q: tensor<1x2x64x32xf32>,
                          %k: tensor<1x2x64x32xf32>,
                          %v: tensor<1x2x64x32xf32>) -> tensor<1x2x64x32xf32> {
    %o = "sattn.sparse_attention"(%q, %k, %v) {
      pattern = "sliding_global",
      block_size = 64 : i64,
      keep_ratio = 0.12 : f32,
      global_tokens = 16 : i64,
      window_size = 512 : i64,
      precision = "bf16",
      softmax_mode = "logsumexp"
    } : (tensor<1x2x64x32xf32>, tensor<1x2x64x32xf32>, tensor<1x2x64x32xf32>) -> tensor<1x2x64x32xf32>
    return %o : tensor<1x2x64x32xf32>
  }
}
