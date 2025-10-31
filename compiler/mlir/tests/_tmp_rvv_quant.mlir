module {
  "sattn.sparse_attention"() { precision = "i8", scale_q = 0.05 : f32, scale_k = 0.05 : f32, scale_v = 0.05 : f32, window_size = 8 : i64, tile_D = 32 : i64, tile_S = 128 : i64 } : () -> ()
}
