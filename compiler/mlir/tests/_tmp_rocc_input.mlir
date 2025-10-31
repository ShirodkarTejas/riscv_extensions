
module {
  "sattn.sparse_attention"() { pattern = "block_topk", block_size = 16 : i64, keep_ratio = 0.25 : f32, precision = "bf16", softmax_mode = "logsumexp", tile_M = 16 : i64, tile_D = 32 : i64, tile_S = 64 : i64 } : () -> ()
}
