module {
  "sattn.sparse_attention"() { comp_block_size = 32 : i64, block_size = 64 : i64, tile_D = 32 : i64, tile_S = 128 : i64 } : () -> ()
}
