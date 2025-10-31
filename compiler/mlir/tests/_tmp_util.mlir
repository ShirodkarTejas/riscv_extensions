module {
  "sattn.sparse_attention"() { tile_M = 8 : i64, tile_D = 16 : i64, tile_S = 32 : i64 } : () -> ()
}
