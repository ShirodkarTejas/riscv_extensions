module {
  "sattn.rvv_call"() {blg_enabled = true, block_size = 16 : i64, bufferized, comp_block_size = 8 : i64, head_dim_d = 32 : i64, s_tokens = 128 : i64, spec = "block_local_global", tile_D = 32 : i64, tile_S = 128 : i64, vectorized} : () -> ()
}

