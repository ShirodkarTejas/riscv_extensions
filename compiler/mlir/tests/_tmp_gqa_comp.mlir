module {
  "sattn.sparse_attention"() { block_size = 8 : i64, comp_block_size = 4 : i64, tile_D = 16 : i64, tile_S = 16 : i64 } : () -> ()
}
