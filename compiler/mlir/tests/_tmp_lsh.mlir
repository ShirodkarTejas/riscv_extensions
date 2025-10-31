module {
  "sattn.sparse_attention"() { lsh_buckets = 64 : i64, tile_S = 512 : i64, tile_D = 64 : i64 } : () -> ()
}
