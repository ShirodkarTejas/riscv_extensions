module {
  "sattn.rvv_call"() {bufferized, head_dim_d = 32 : i64, s_tokens = 128 : i64, spec = "sliding_window", tile_D = 32 : i64, tile_S = 128 : i64, vectorized, window_size = 8 : i64, wrap = 1 : i64} : () -> ()
}

