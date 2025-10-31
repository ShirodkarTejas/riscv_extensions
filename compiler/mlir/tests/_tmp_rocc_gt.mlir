module {
  "sattn.sparse_attention"() { global_tokens = 4 : i64, block_size = 4 : i64, tile_D = 16 : i64, tile_S = 16 : i64 } : () -> ()
}
