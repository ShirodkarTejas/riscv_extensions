module {
  "sattn.rvv_call"() {bufferized, head_dim_d = 16 : i64, s_tokens = 64 : i64, spec = "bsr", tile_D = 16 : i64, tile_S = 64 : i64, vectorized, window_size = 8 : i64} : () -> ()
}

