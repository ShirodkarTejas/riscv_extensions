module {
  "sattn.rvv_call"() {bufferized, head_dim_d = 32 : i64, lsh_buckets = 8 : i64, lsh_enabled = true, s_tokens = 128 : i64, spec = "lsh", tile_D = 32 : i64, tile_S = 128 : i64, vectorized} : () -> ()
}

