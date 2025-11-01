module {
  "sattn.rvv_call"() {buffer_layout = "rowmajor", buffer_strategy = "one-shot", bufferized, head_dim_d = 32 : i64, precision = "i8", s_tokens = 128 : i64, scale_k = 5.000000e-02 : f32, scale_q = 5.000000e-02 : f32, scale_v = 5.000000e-02 : f32, spec = "bsr", tile_D = 32 : i64, tile_S = 128 : i64, vectorized, window_size = 8 : i64} : () -> ()
}

