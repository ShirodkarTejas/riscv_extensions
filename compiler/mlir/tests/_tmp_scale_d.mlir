module {
  "sattn.sparse_attention"() { tile_M = 16 : i64, tile_D = 32 : i64, tile_S = 64 : i64 } : () -> ()
}
