module {
  "sattn.sparse_attention"() { block_size = 4 : i64, keep_ratio = 0.12 : f32, global_tokens = 2 : i64, tile_D = 16 : i64, tile_S = 64 : i64 } : () -> ()
}
