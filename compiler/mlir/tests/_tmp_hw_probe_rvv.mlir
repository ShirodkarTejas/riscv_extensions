module {
  "sattn.sparse_attention"() { window_size = 8 : i64, tile_D = 16 : i64, tile_S = 64 : i64 } : () -> ()
}
