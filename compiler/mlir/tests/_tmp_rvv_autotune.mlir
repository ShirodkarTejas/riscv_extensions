module {
  "sattn.sparse_attention"() { spec = "sliding_window", window_size = 8 : i64, tile_D = 32 : i64, tile_S = 128 : i64 } : () -> ()
}
