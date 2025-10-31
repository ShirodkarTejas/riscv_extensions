module {
  "sattn.sparse_attention"() { spec = "block_local_global", block_size = 16 : i64, tile_D = 32 : i64, tile_S = 128 : i64 } : () -> ()
}
