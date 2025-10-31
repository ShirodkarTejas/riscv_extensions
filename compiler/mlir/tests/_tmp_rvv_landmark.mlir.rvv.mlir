module {
  "sattn.rvv_call"() {bufferized, head_dim_d = 32 : i64, num_landmarks = 16 : i64, s_tokens = 128 : i64, spec = "landmark", tile_D = 32 : i64, tile_S = 128 : i64, vectorized} : () -> ()
}

