module {
  "sattn.sparse_attention"() { lsh_buckets = 8 : i64, tile_D = 32 : i64, tile_S = 128 : i64 } : () -> ()
}
