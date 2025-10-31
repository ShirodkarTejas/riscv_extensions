#!/usr/bin/env python3
import os
import subprocess
import sys


def test_roundtrip_blg_sim():
    # Create a minimal BLG MLIR
    mlir_path = 'compiler/mlir/tests/_tmp_blg_roundtrip.mlir'
    with open(mlir_path, 'w') as f:
        f.write('''module {
  "sattn.sparse_attention"() { global_tokens = 8 : i64, block_size = 64 : i64, tile_M = 16 : i64, tile_D = 32 : i64, tile_S = 128 : i64 } : () -> ()
}
''')
    cmd = [sys.executable, 'compiler/mlir/tools/sattn_compile_and_sim.py', '--mlir', mlir_path]
    print('[test]', ' '.join(cmd))
    out = subprocess.run(cmd, text=True, capture_output=True)
    print(out.stdout)
    assert out.returncode == 0
    # Ensure BLG flow runs end-to-end; reflect spec info line and counters shown
    assert 'rocc_counters(proxy):' in out.stdout
    assert 'spec_info: global_tokens=' in out.stdout


