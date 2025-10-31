module {
  "sattn.sparse_attention"() { global_tokens = 4 : i64, block_size = 64 : i64, tile_M = 16 : i64, tile_D = 32 : i64, tile_S = 128 : i64 } : () -> ()
}
