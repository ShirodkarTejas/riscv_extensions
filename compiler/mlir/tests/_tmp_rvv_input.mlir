module {
  "sattn.sparse_attention"() { window_size = 32 : i64, tile_M = 16 : i64, tile_D = 32 : i64, tile_S = 256 : i64 } : () -> ()
}
