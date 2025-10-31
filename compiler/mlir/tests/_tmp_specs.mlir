module {
  "sattn.sparse_attention"() { topk_k = 8 : i64, tile_S = 128 : i64, tile_D = 32 : i64 } : () -> ()
}
