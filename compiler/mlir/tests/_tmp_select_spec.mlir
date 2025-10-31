module {
  "sattn.sparse_attention"() { window_size = 16 : i64, keep_ratio = 0.02 : f32, block_size = 64 : i64, tile_M = 16 : i64, tile_D = 32 : i64, tile_S = 128 : i64 } : () -> ()
}
