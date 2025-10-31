module {
  "sattn.sparse_attention"() { lsh_buckets = 8 : i64, tile_S = 256 : i64, tile_D = 64 : i64 } : () -> ()
}
